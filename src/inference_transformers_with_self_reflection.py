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

# Create QA directory if it doesn't exist
qa_dir = os.path.join(current_dir, 'resource/QA')
os.makedirs(qa_dir, exist_ok=True)

QA_DATASET_PATH = os.path.join(qa_dir, 'sample_qa.json')
QA_OUTPUT_PATH = os.path.join(qa_dir, 'result_train.json')

# Hyperparameters
K = 3

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
    g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
    g.add_argument("--model_id", type=str, default="skt/A.X-4.0-Light", help="huggingface model id")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=10, help="Batch size for inference.")
    g.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    return parser.parse_args()

cnt = 0
def generate(args, retriever, pipe, result_data):
    prompts = []
    system_prompt = make_system_prompt()
    self_reflection_prompt = make_self_reflection()
    print("Preparing prompts...")
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        context = ""
        if args.retrieve:
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
            retrieve=args.retrieve,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)

    print("Generating answers in batch...")
    outputs = pipe(prompts)

    print("Processing generated answers...")
    prompts = []
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        generated_text.append({"role": "user", "content": self_reflection_prompt})
        prompts.append(generated_text)
        
    outputs = pipe(prompts)
    
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        generated_text = output[0]['generated_text']
        answer = generated_text[2]['content']
        reflection_answer = generated_text[4]['content']
        if reflection_answer[0] != "예":
            result_data[idx]['output'] = {"answer": answer.strip()}
        else:
            print("=" * 50)
            print("예 발생!!!!")
            cnt += 1
            print("=" * 50)
            print(reflection_answer)
            print("=" * 50)
            result_data[idx]['output'] = {"answer": ""}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))


def main():
    args = parse_arguments()
    RETRIEVER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = args.model_id

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
    
    pipe = load_llm(model_id=GENERATOR_NAME, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
    if not pipe:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")

    file_test = args.input
    with open(file_test, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)
    generate(args, retriever, pipe, result_data)
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)
    print("아니요 발생 횟수: ", cnt)
    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main())