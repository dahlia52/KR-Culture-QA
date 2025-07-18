import os
import argparse
import json
import re
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
from retrieve import *
from load_llm import load_llm

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/sample_qa.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/result_train.json')

# Hyperparameters
K = 30

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
    g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
    g.add_argument("--model_id", type=str, default="K-intelligence/Midm-2.0-Base-Instruct", help="huggingface model id")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    g.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    g.add_argument("--retrieve_adaptively", action="store_true", help="Whether to use retrieval-augmented generation")
    return parser.parse_args()


def generate(args, retriever, pipe, result_data):
    prompts = []
    other_indices = []
    system_prompt = make_system_prompt()

    print("Preparing prompts...")
    for idx, item in enumerate(tqdm.tqdm(result_data)):
        question = item["input"]["question"]
        question_type = item["input"]["question_type"]
        context = ""

        # if args.retrieve or args.retrieve_adaptively:
        #     documents = retriever.invoke(question)
        #     context = format_docs(documents)


        if question_type == '선다형':
            if "5\\t" in question:
                choice5 = question.split("5\\t")[1].strip()
                question = question.split("5\\t")[0].strip()
            if "4\\t" in question:
                choice4 = question.split("4\\t")[1].strip()
                question = question.split("4\\t")[0].strip()
            if "3\\t" in question:
                choice3 = question.split("3\\t")[1].strip()
                question = question.split("3\\t")[0].strip()
            if "2\\t" in question:
                choice2 = question.split("2\\t")[1].strip()
                question = question.split("2\\t")[0].strip()
            choice1 = question.split("1\\t")[1].strip()
            question = question.split("1\\t")[0].replace('\n','').strip()

            if "5\\t" in question:
                choices = [choice1, choice2, choice3, choice4, choice5]
            else:
                choices = [choice1, choice2, choice3, choice4]

            if not choices:
                raise ValueError("No choices found in the question.")
            print(question)
            print(choices)

            if args.retrieve or args.retrieve_adaptively:
                documents = retriever.invoke(question)
                context = format_docs(documents)
            
            user_prompt_base = make_prompt(
                question_type=question_type,
                category=item["input"]["category"],
                domain=item["input"]["domain"],
                topic_keyword=item["input"]["topic_keyword"],
                context=context,
                question=question,
                fewshot=False,
                retrieve=args.retrieve or args.retrieve_adaptively
            )

            messages_base = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_base}
            ]
            
            prompt_text = pipe.tokenizer.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)

            choice_scores = []
            for choice in choices:
                with torch.no_grad():
                    inputs = pipe.tokenizer(prompt_text).input_ids
                    choice_tokens = pipe.tokenizer(choice, add_special_tokens=False).input_ids
                    full_input = pipe.tokenizer(prompt_text + choice, return_tensors="pt").to(args.device)

                    labels = torch.tensor([[-100] * len(inputs) + choice_tokens]).to(args.device)
                    outputs = pipe.model(**full_input, labels=labels)
                    logits = outputs.logits
                    
                    # Calculate score for the choice
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # The logits are shifted by one position
                    relevant_log_probs = log_probs[0, -len(choice_tokens)-1:-1, :]
                    choice_token_ids = torch.tensor(choice_tokens).to(args.device)
                    
                    score = relevant_log_probs.gather(1, choice_token_ids.unsqueeze(-1)).squeeze(-1).sum().item()
                    choice_scores.append(score)

            best_choice_idx = torch.argmax(torch.tensor(choice_scores)).item()
            answer = str(best_choice_idx + 1)
            result_data[idx]["output"] = {"answer": answer}

        else:
            if args.retrieve or args.retrieve_adaptively:
                documents = retriever.invoke(question)
                context = format_docs(documents)
            # For other question types, use the original batch processing
            user_prompt = make_prompt(
                question_type=question_type,
                category=item["input"]["category"],
                domain=item["input"]["domain"],
                topic_keyword=item["input"]["topic_keyword"],
                context=context,
                question=question,
                fewshot=True,
                retrieve=args.retrieve or args.retrieve_adaptively
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompts.append(messages)
            other_indices.append(idx)

    if prompts:
        print("Generating answers for non-multiple-choice questions in batch...")
        outputs = pipe(prompts)

        print("Processing generated answers...")
        for i, output in enumerate(tqdm.tqdm(outputs)):
            original_idx = other_indices[i]
            generated_text = output[0]['generated_text']
            # The actual answer is the last message in the generated text
            if isinstance(generated_text, list) and generated_text:
                answer = generated_text[-1]['content']
            else: # Fallback for different output formats
                answer = str(generated_text)
            result_data[original_idx]["output"] = {"answer": answer.strip()}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))


def main():
    torch.set_float32_matmul_precision('high')
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

    if args.retrieve_adaptively:
        retriever = load_retriever_adaptively(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")
    
    pipe, tokenizer = load_llm(model_id=GENERATOR_NAME, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
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

    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main())