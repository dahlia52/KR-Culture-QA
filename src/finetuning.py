import os
import argparse
import json
import time
from datetime import timedelta
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from make_prompt import make_prompt, format_docs
from retrieve import load_retriever

# Get the project root directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_ID = "skt/A.X-4.0-Light"
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_train+.json')
DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/fine-tuned-model')
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
RETRIEVER_NAME = "BAAI/bge-m3"
K = 5

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning a language model on a custom QA dataset.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the fine-tuned model")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA for fine-tuning")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    parser.add_argument("--evaluation", action="store_true", help="Evaluate validation set")

    return parser.parse_args()

def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer, retriever) -> DatasetDict:
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Assign persona
    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. 
    단, 동일한 문장을 절대 반복하지 마시오."""

    def generate_prompt(example):
        question = example['input']['question']
        documents = retriever.invoke(question)
        context = format_docs(documents)
        
        user_prompt = make_prompt(
            question_type=example["input"]["question_type"],
            category=example["input"]["category"],
            domain=example["input"]["domain"],
            topic_keyword=example["input"]["topic_keyword"],
            context=context,
            question=question
        )

        # Create a single text field for the trainer using the chat template.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example['output']['answer']}
        ]
        
        # The tokenizer will apply the chat template and add EOS token.
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": full_prompt}

    # Create dataset from the list of dictionaries and then format it
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(generate_prompt)

    # SFTTrainer will handle tokenization. We just need to provide the text field.
    dataset = dataset.map(generate_prompt, remove_columns=['input', 'output'])

    return DatasetDict({"train": dataset})


def main():
    total_start_time = time.time()
    
    args = parse_arguments()

    print("=" * 50)
    print("Starting Fine-Tuning")
    print(f"Model ID: {args.model_id}")
    print(f"Training Data: {args.data_path}")
    if args.lora:
        print("LoRA Fine-Tuning: Enabled")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)

    print("\n1. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16
    )
    
    if args.lora:
        print("\nApplying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✅ LoRA configured.")

    print("✅ Tokenizer and model loaded.")

    print("\n2. Loading retriever...")
    retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
    if not retriever:
        raise Exception("Failed to initialize retriever")
    print("✅ Retriever loaded successfully.")

    print("\n3. Loading and preparing dataset...")
    dataset = load_and_prepare_data(args.data_path, tokenizer, retriever)
    print("✅ Dataset prepared.")
    print(f"Training dataset size: {len(dataset['train'])}")
    if args.evaluation:
        print(f"Validation dataset size: {len(dataset['validation'])}")

    print("\n4. Configuring training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="epoch" if args.evaluation else "no",
        save_total_limit=2,
        fp16=True, # Use mixed precision training
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True, # Pack multiple sequences into one for efficiency
    )
    
    print("✅ Training configured.")

    print("\n5. Starting model training...")
    trainer.train()
    
    total_training_time = time.time() - total_start_time
    
    print("\n" + "="*50)
    print("✅ Training complete.")
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    print("\n6. Saving the fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
