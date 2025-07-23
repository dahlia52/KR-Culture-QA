import os
import gc
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
from retrieve import *
from load import *
from generators import *
import logging
from datetime import datetime
from peft import PeftModel, PeftConfig

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_test+.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_test+_context.json')

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
    g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
    g.add_argument("--model_id", type=str, default="K-intelligence/Midm-2.0-Base-Instruct", help="huggingface model id")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
    g.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    g.add_argument("--retrieve_adaptively", action="store_true", help="Whether to use retrieval-augmented generation")
    g.add_argument("--k", type=int, default=2, help="Number of retrieved documents.")
    return parser.parse_args()


def main():
    gc.collect()                         
    torch.cuda.empty_cache()
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
        retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=args.k)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")

    if args.retrieve_adaptively:
        retriever = load_retriever_adaptively(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=args.k)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")

    is_lora = os.path.isdir(args.model_id) and 'adapter_config.json' in os.listdir(args.model_id)
    lora_weights = args.model_id if is_lora else None
    
    if is_lora:
        print(f"🔍 Detected a fine-tuned LoRA model at: {args.model_id}")
        config = PeftConfig.from_pretrained(args.model_id)
        base_model_name = config.base_model_name_or_path
        print(f"🔧 Loading base model: {base_model_name}")
        GENERATOR_NAME = base_model_name
    
    pipe, tokenizer = load_llm(model_id=GENERATOR_NAME, device=args.device, quantize=args.quantize, batch_size=args.batch_size, is_lora=is_lora, lora_weights=lora_weights)
    if not pipe:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")

    result_data = load_dataset(args.input)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)

    verify_context(args, retriever, pipe, result_data)
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main())