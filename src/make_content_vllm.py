import os
import gc
import argparse
import json
import asyncio
import tqdm
from typing import List, TypedDict, Optional, Dict, Any
import torch
from datasets import load_from_disk
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from vllm import LLM, SamplingParams
#from vllm.model_executor.adapters import lora
from make_prompt import *
from retrieve import *
from load import *

import logging
from datetime import datetime

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/sample_qa.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/sample_qa_context.json')

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
    g.add_argument("--logit", action="store_true", help="Whether to use logit for multiple_choice")
    g.add_argument("--verify", action="store_true", help="Whether to use verifier")
    g.add_argument("--retrieval_queries", action="store_true", help="Whether to use retrieval queries")
    g.add_argument("--k", type=int, default=2, help="Number of retrieved documents.")
    return parser.parse_args()


async def verify_context(args, retriever, llm, result_data):
    logging.info("### Verify context ###")
    system_prompt = make_system_prompt()
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["</s>", "###"]
    )
    
    logging.info("Processing documents...")
    
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        context = ""
        
        # Retrieve relevant documents
        documents = retriever.invoke(f"{topic_keyword}: {question}")
        
        # Prepare prompts for batch processing
        prompts = []
        for document in documents:
            user_prompt = make_prompt_for_context(
                topic_keyword=topic_keyword,
                question=question,
                context=document
            )
            full_prompt = f"""### System:
{system_prompt}

### User:
{user_prompt}

### Assistant:
"""
            prompts.append(full_prompt)
        
        if not prompts:
            continue
            
        # Generate responses in batch
        outputs = await llm.generate_async(prompts, sampling_params)
        
        # Process responses
        cnt = 0
        for idx, output in enumerate(outputs):
            answer = output.outputs[0].text.strip()
            print(answer)
            
            if answer.startswith("예"):
                context = context + documents[idx].page_content + '\n\n'
                
            result_data[idx]["output"] = {"context": context.strip()}
        
        logging.info(f"Processed {len(documents)} documents")
    
    save_dataset(result_data, args.output)


async def main_async():
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
    print("Starting Korean Culture QA System with vLLM")
    print("=" * 50)
    
    retriever = None
    if args.retrieve:
        retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, 
                                 chroma_db_path=CHROMA_DB_PATH, 
                                 kowiki_dataset_path=KOWIKI_DATASET_PATH, 
                                 k=args.k)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")

    if args.retrieve_adaptively:
        retriever = load_retriever_adaptively(model=RETRIEVER_NAME, device=args.device, 
                                            chroma_db_path=CHROMA_DB_PATH, 
                                            kowiki_dataset_path=KOWIKI_DATASET_PATH, 
                                            k=args.k)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Adaptive retriever loaded successfully.")

    # Initialize vLLM model
    print(f"Loading vLLM model: {GENERATOR_NAME}")
    
    # Check if it's a LoRA model
    is_lora = os.path.isdir(args.model_id) and 'adapter_config.json' in os.listdir(args.model_id)
    lora_weights = args.model_id if is_lora else None
    
    if is_lora:
        print(f"🔍 Detected a fine-tuned LoRA model at: {args.model_id}")
        # For vLLM, we'll load the base model and then apply LoRA
        base_model_name = "K-intelligence/Midm-2.0-Base-Instruct"  # Default base model
        print(f"🔧 Will apply LoRA weights to base model: {base_model_name}")
        GENERATOR_NAME = base_model_name
    
    # Configure vLLM with appropriate parameters
    llm = LLM(
        model=GENERATOR_NAME,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        quantization="awq" if args.quantize else None,
        max_model_len=4096,
        trust_remote_code=True
    )
    
    # Apply LoRA weights if specified
    if is_lora:
        print("🔧 Applying LoRA weights...")
        llm.add_lora(lora_weights)
    
    print("✅ vLLM model loaded successfully.")

    result_data = load_dataset(args.input)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)
    
    # Note: Other generation functions would need to be updated similarly
    if args.self_reflection:
        raise NotImplementedError("Self-reflection not yet implemented with vLLM")
    elif args.logit:
        raise NotImplementedError("Multiple choice not yet implemented with vLLM")
    elif args.verify:
        raise NotImplementedError("Verifier not yet implemented with vLLM")
    elif args.retrieval_queries:
        raise NotImplementedError("Retrieval queries not yet implemented with vLLM")
    else:
        await verify_context(args, retriever, llm, result_data)
    
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

    torch.cuda.synchronize()
    print(f"Maximum VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")

def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    exit(main())