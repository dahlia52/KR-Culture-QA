import os
import argparse
import json
import tqdm
from typing import List, TypedDict, Optional
import torch
from datasets import load_from_disk
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from make_prompt import *
from retrieve import load_retriever
from load_llm import load_llm

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/sample_qa.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/result.json')

# Hyperparameters
K = 3

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
    g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
    g.add_argument("--model_id", type=str, default="K-intelligence/Midm-2.0-Base-Instruct", help="huggingface model id")
    g.add_argument("--validator_id", type=str, default="skt/A.X-4.0-Light", help="huggingface model id")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    g.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    g.add_argument("--retrieve_adaptively", action="store_true", help="Whether to use retrieval-augmented generation")
    return parser.parse_args()


def generate(args, retriever, pipe1, pipe2, result_data):
    prompts = []
    system_prompt = make_system_prompt()
    system_prompt_for_verifier = make_system_prompt_for_verifier()
    system_prompt_for_feedback = make_system_prompt_with_feedback()
    
    print("Preparing prompts...")
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        context = ""
        if args.retrieve or args.retrieve_adaptively:
            print("Retrieving relevant documents...")
            documents = retriever.invoke(question)
            context = format_docs(documents)
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=item["input"]["topic_keyword"],
            context=context,
            question=question,
            fewshot=True,
            retrieve = args.retrieve or args.retrieve_adaptively
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)

    print("Generating answers in batch...")
    outputs = pipe1(prompts)

    print("Processing generated answers...")
    prompts = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        instruction = output[0]['generated_text'][1]['content'].split("[지침]\n")[1].split("\n\n")[0]
        generated_text = output[0]['generated_text'][-1]['content']
        question = output[0]['generated_text'][1]['content'].split("[질문]\n")[1].split("[답변]")[0]
        verifier_prompt = make_verifier_prompt(instruction=instruction, question=question, answer=generated_text)
        messages = [
            {"role": "system", "content": system_prompt_for_verifier},
            {"role": "user", "content": verifier_prompt}
        ]
        prompts.append(messages)
        
    outputs = pipe2(prompts)
    
    regenerate_idx = []
    prompts = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        verifier_answer = output[0]['generated_text'][-1]['content']
        if verifier_answer[0] == "예":
            answer = output[0]['generated_text'][-2]['content'].split("[답변]\n")[1].split("이 답변은 질문에 올바르게 대답한 것입니까?\n")[0].replace('\n', '').strip()
            result_data[idx]['output'] = {"answer": answer}
        else:
            regenerate_idx.append(idx)
            instruction = output[0]['generated_text'][1]['content'].split("[질문]\n")[0].split("\n\n")[0]
            generated_text = output[0]['generated_text'][-2]['content'].split("[답변]\n")[1].split("이 답변은 질문에 올바르게 대답한 것입니까?\n")[0].replace('\n', '').strip()
            question = output[0]['generated_text'][1]['content'].split("[질문]\n")[1].split("[답변]")[0]
            verifier_prompt = make_prompt_with_feedback(instruction=instruction, question=question, answer=generated_text, feedback=verifier_answer[3:])
            messages = [
                {"role": "system", "content": system_prompt_for_feedback},
                {"role": "user", "content": verifier_prompt}
            ]
            prompts.append(messages)

    if len(regenerate_idx) > 0:
        outputs = pipe1(prompts)
        for idx, output in enumerate(tqdm.tqdm(outputs)):
            answer = output[0]['generated_text'][-1]['content'].split("<answer>")[1].split("</answer>")[0].replace('\n', '').strip()
            result_data[regenerate_idx[idx]]['output'] = {"answer": answer}
    
    print("Number of Regenerated Answers :", len(regenerate_idx))
    print("Regenerated Answers:", regenerate_idx)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))


def main():
    torch.set_float32_matmul_precision('high')
    args = parse_arguments()
    RETRIEVER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = args.model_id
    VALIDATOR_NAME = args.validator_id

    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    
    print("=" * 50)
    print("Starting Korean Culture QA System")
    print("=" * 50)
    retriever = None
    if args.retrieve:
        retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")

    if args.retrieve_adaptively:
        retriever = load_retriever_adaptively(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")
    
    pipe1, tokenizer1 = load_llm(model_id=GENERATOR_NAME, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
    if not pipe1:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")

    pipe2, tokenizer2 = load_llm(model_id=VALIDATOR_NAME, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
    if not pipe2:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")

    file_test = args.input
    with open(file_test, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)
    generate(args, retriever, pipe1, pipe2, result_data)
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main())