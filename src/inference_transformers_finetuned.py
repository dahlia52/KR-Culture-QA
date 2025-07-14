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
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from make_prompt import *
from retrieve import load_retriever

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/sample_qa.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/result_train.json')

# Hyperparameters
K = 3

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
    g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
    g.add_argument("--model_id", type=str, default="skt/A.X-3.1-Light", help="huggingface model id") # skt/A.X-4.0-Light
    g.add_argument("--tokenizer", type=str, default="skt/A.X-3.1-Light", help="huggingface tokenizer")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=10, help="Batch size for inference.")
    g.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    return parser.parse_args()



def load_llm(model_id, device, quantize=False, batch_size=1, is_lora=False, lora_weights=None):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id if not is_lora else lora_weights)
    tokenizer.padding_side = 'left'
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with or without quantization
    if quantize:
        print("Quantizing model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True
        )
    
    # Load LoRA weights if specified
    if is_lora and lora_weights:
        print(f"Loading LoRA weights from {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
        model = model.merge_and_unload()  # Merge LoRA weights into the model

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        top_k=30,
        temperature=0.8,
        batch_size=batch_size,
        return_full_text=False  # Don't return the input prompt in the output
    )

    return pipe


def generate(args, retriever, pipe, result_data):
    prompts = []
    system_prompt = make_system_prompt()
    
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
            retrieve = args.retrieve
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
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        # The output from the pipeline is a list with a dictionary
        # The generated text is in output[0]['generated_text']
        # We need to extract just the assistant's response
        full_response = output[0]['generated_text']
        
        # Split the response to get just the assistant's part
        # Assuming the format is: <assistant>
        parts = full_response.split('<assistant>\n')
        if len(parts) > 1:
            answer = parts[-1].strip()
        else:
            answer = full_response.strip()
        
        result_data[idx]["output"] = {"answer": answer}

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
    
    # Check if we're using a fine-tuned model
    is_lora = os.path.isdir(args.model_id) and 'adapter_config.json' in os.listdir(args.model_id)
    lora_weights = args.model_id if is_lora else None
    
    # If using LoRA, get the base model name from the config
    if is_lora:
        print(f"🔍 Detected a fine-tuned LoRA model at: {args.model_id}")
        config = PeftConfig.from_pretrained(args.model_id)
        base_model_name = config.base_model_name_or_path
        print(f"🔧 Loading base model: {base_model_name}")
        GENERATOR_NAME = base_model_name
    
    retriever = None
    if args.retrieve:
        retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")
    
    # Load the language model with LoRA weights if applicable
    pipe = load_llm(
        model_id=GENERATOR_NAME,
        device=args.device,
        quantize=args.quantize,
        batch_size=args.batch_size,
        is_lora=is_lora,
        lora_weights=lora_weights
    )
    
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