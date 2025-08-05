import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import argparse
import torch
from src.make_prompt import *
from src.retrieve import load_vector_store
from src.data_io import load_dataset, save_dataset
from src.load_model import load_llm
from src.generate import *
import logging
from datetime import datetime


# Get the project root directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, help="input filename")
    g.add_argument("--output", type=str, help="output filename")
    g.add_argument("--model_id", type=str, help="huggingface model id")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
    g.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    return parser.parse_args()


def main():
    gc.collect()                         
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_arguments()
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

    embeddings, vector_store = load_vector_store(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH)

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    contexts_mc = make_contexts(retriever, mc_data)

    retriever = CustomRetriever(lambda query: custom_retriever(query, embeddings, vector_store))
    contexts_sa = make_contexts(retriever, sa_data)

    del embeddings
    del vector_store
    del retriever
    torch.cuda.empty_cache()
    gc.collect()

    GENERATOR = args.model_id
    
    pipe = load_llm(model_id=GENERATOR, device=args.device, quantize=args.quantize, batch_size=args.batch_size, temperature=args.temperature)
    if not pipe:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")

    # Multiple Choice
    mc_data = generate_with_rationale(args, pipe, mc_data, contexts_mc)
    mc_data = regenerate(pipe, mc_data)
    print("Multiple Choice Completed")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Single Answer
    sa_data = generate(args, pipe, sa_data, contexts_sa)
    print("Single Answer Completed")
    torch.cuda.empty_cache()
    gc.collect()

    # Descriptive
    dc_data = generate(args, pipe, dc_data, None)
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