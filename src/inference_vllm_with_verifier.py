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
from langchain_huggingface import HuggingFaceEmbeddings
from vllm import LLM, SamplingParams
from make_prompt import *
from retrieve import load_retriever

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
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism")
    g.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    g.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    g.add_argument("--retrieve_adaptively", action="store_true", help="Whether to use retrieval-augmented generation adaptively")
    return parser.parse_args()

def load_vllm_model(model_id, device, tensor_parallel_size=1):
    try:
        llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True
        )
        return llm
    except Exception as e:
        print(f"Error loading vLLM model: {e}")
        return None

def generate_prompts(args, retriever, result_data):
    prompts = []
    for item in tqdm.tqdm(result_data, desc="Preparing prompts"):
        question = item["question"]
        context = ""
        
        if args.retrieve or args.retrieve_adaptively:
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
        
        prompt = create_prompt(question, context)
        prompts.append(prompt)
    
    return prompts

def main():
    torch.set_float32_matmul_precision('high')
    args = parse_arguments()
    RETRIEVER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = args.model_id

    print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("=" * 50)
    print("Starting Korean Culture QA System with vLLM")
    print("=" * 50)
    
    retriever = None
    if args.retrieve or args.retrieve_adaptively:
        retriever = load_retriever(
            model=RETRIEVER_NAME, 
            device=args.device, 
            chroma_db_path=CHROMA_DB_PATH, 
            kowiki_dataset_path=KOWIKI_DATASET_PATH, 
            k=K
        )
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")
    
    llm = load_vllm_model(
        model_id=GENERATOR_NAME,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size
    )
    if not llm:
        raise Exception("Failed to initialize vLLM model")
    print("✅ vLLM model loaded successfully.")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    with open(args.input, "r", encoding="utf-8") as f:
        result_data = json.load(f)
    
    prompts = generate_prompts(args, retriever, result_data)
    
    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        result = {
            "question": result_data[i]["question"],
            "answer": result_data[i].get("answer", ""),
            "generated_text": output.outputs[0].text.strip(),
            "prompt": prompts[i]
        }
        results.append(result)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print(f"Results saved to {args.output}")
    print("=" * 50)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"Maximum VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == '__main__':
    exit(main())
