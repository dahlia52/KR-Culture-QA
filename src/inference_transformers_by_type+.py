import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import argparse
import json
import tqdm
from typing import List, TypedDict, Optional
import torch
import datasets
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

import random
import numpy as np




# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_test+.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/final_new.json')

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
    g.add_argument("--self_reflection", action="store_true", help="Whether to use self-reflection")
    g.add_argument("--verify", action="store_true", help="Whether to use verifier")
    g.add_argument("--retrieval_queries", action="store_true", help="Whether to use retrieval queries")
    g.add_argument("--k", type=int, default=2, help="Number of retrieved documents.")
    g.add_argument("--rationale", action="store_true", help="Whether to use rationale")
    g.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    g.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    gc.collect()                         
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_arguments()
    set_seed(args.seed)
    RETRIEVER_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("Log file: " + log_filename)

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    
    print("=" * 50)
    print("Starting Korean Culture QA System")
    print("=" * 50)

    result_data = load_dataset(args.input)

    # Split Data by question type
    mc_data = []
    sa_data = []
    dc_data = []
    for item in result_data:
        if item['input']['question_type'] == '선다형':
            mc_data.append(item)
        elif item['input']['question_type'] == '단답형':
            sa_data.append(item)
        elif item['input']['question_type'] == '서술형':
            dc_data.append(item)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)

    args.retrieve = True
    embeddings, vector_store = load_vector_store(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH)

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    contexts_mc = make_contexts(args, retriever, mc_data)
    contexts_sa = make_contexts(args, retriever, sa_data)

    del embeddings
    del vector_store
    torch.cuda.empty_cache()
    gc.collect()

    GENERATOR = args.model_id
    base_model_name = args.model_id
    is_lora = os.path.isdir(GENERATOR) and 'adapter_config.json' in os.listdir(GENERATOR)
    lora_weights = GENERATOR if is_lora else None
    
    if is_lora:
        print(f"🔍 Detected a fine-tuned LoRA model at: {GENERATOR}")
        config = PeftConfig.from_pretrained(GENERATOR)
        base_model_name = config.base_model_name_or_path
        print(f"🔧 Loading base model: {base_model_name}")
    
    pipe, tokenizer = load_llm(model_id=GENERATOR, base_model_name=base_model_name, device=args.device, quantize=args.quantize, batch_size=args.batch_size, is_lora=is_lora, lora_weights=lora_weights, temperature=args.temperature)
    if not pipe:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")

    # Multiple Choice
    mc_data = generate_with_rationale(args, retriever, pipe, mc_data, contexts_mc)
    if args.verify:
        mc_data = generate_final(args, pipe, mc_data)
    print("Multiple Choice Completed")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Single Answer
    args.retrieve = False
    args.retrieve_adaptively = True
    retriever = CustomRetriever(lambda query: custom_retriever(query, embeddings, vector_store))
    sa_data = generate_for_type(args, retriever, pipe, sa_data, contexts_sa)
    print("Single Answer Completed")
    torch.cuda.empty_cache()
    gc.collect()

    # Descriptive
    args.retrieve_adaptively = False
    retriever = None #vector_store.as_retriever(search_kwargs={"k": 0})
    dc_data = generate_for_type(args, retriever, pipe, dc_data, None)
    print("Descriptive Completed")
    torch.cuda.empty_cache()
    gc.collect()

    result_data = mc_data + sa_data + dc_data
    result_data = sorted(result_data, key=lambda x: int(x['id']))
    save_dataset(result_data, args.output)
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main())