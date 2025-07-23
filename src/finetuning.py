import os
import argparse
import json
import time
from datetime import timedelta
from typing import Dict, List

import torch
from langchain.vectorstores import Chroma
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from make_prompt import *
from retrieve import load_retriever
from compute_metrics import compute_metrics

# Get the project root directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/split_by_type/korean_culture_qa_선다형+단답형+서술형.json')
DEFAULT_VALID_DATA_PATH = os.path.join(current_dir, 'resource/QA/split_by_type/korean_culture_qa_선다형+단답형+서술형.json')

DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/fine-tuned-model-선다형-단답형-서술형-NEW')
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
RETRIEVER_NAME = "BAAI/bge-m3"
K = 3

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning a language model on a custom QA dataset.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH, help="Path to the training data JSON file")
    parser.add_argument("--valid_data_path", type=str, default=DEFAULT_VALID_DATA_PATH, help="Path to the validation data JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the fine-tuned model")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")

    parser.add_argument("--evaluation", action="store_true", help="Evaluate validation set")
    parser.add_argument("--retriever", type=str, default=RETRIEVER_NAME, help="Retriever name")
    parser.add_argument("--retrieve", action="store_true", help="Use retrieval-augmented generation")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")

    return parser.parse_args()

def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer, retriever=None, retrieve = False) -> DatasetDict:
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Assign persona
    system_prompt = make_system_prompt()
    def generate_prompt(example):
        question = example['input']['question']
        context = ""
        if retrieve:
            documents = retriever.invoke(question)
            context = format_docs(documents)
        
        user_prompt = make_prompt(
            question_type=example["input"]["question_type"],
            category=example["input"]["category"],
            domain=example["input"]["domain"],
            topic_keyword=example["input"]["topic_keyword"],
            context=context,
            question=question,
            fewshot=False,
            retrieve=retrieve
        )
        if example['input']['question_type'] == '단답형' and '#' in example['output']['answer']:
            answer = example['output']['answer'].split('#')[0]
        else:
            answer = example['output']['answer']

        # Create a single text field for the trainer using the chat template.
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer}
        ]
        # The tokenizer will apply the chat template and add EOS token.
        text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=False,
        )
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}


    # Create dataset from the list of dictionaries and then format it
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(generate_prompt)
    return dataset


def main():
    total_start_time = time.time()
    
    args = parse_arguments()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    print("=" * 50)
    print("Starting Fine-Tuning")
    print(f"Model ID: {args.model_id}")
    print(f"Retrieval: {args.retrieve}")
    print("LoRA Fine-Tuning: Enabled")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)

    retriever = None
    if args.retrieve:
        print("\n1. Loading retriever...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        retriever = load_retriever(model=RETRIEVER_NAME, device=device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")


    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        print("Setting pad token.")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))  # 반드시 tokenizer 추가 후 호출
        model.config.pad_token_id = tokenizer.pad_token_id
    print("✅ Tokenizer and model loaded.")


    print("\n3. Loading and preparing dataset...")
    full_train_dataset = load_and_prepare_data(args.train_data_path, tokenizer, retriever, args.retrieve)
    train_dataset = full_train_dataset.map(lambda example: {'text': example['text']}, remove_columns=list(full_train_dataset.column_names))
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'],
                                  padding=True, truncation = True, max_length=1024), 
                                  batched=True, batch_size=args.batch_size, remove_columns=train_dataset.column_names)

    valid_dataset = None
    if args.evaluation:
        full_valid_dataset = load_and_prepare_data(args.valid_data_path, tokenizer, retriever, args.retrieve)
        valid_dataset = full_valid_dataset.map(lambda example: {'text': example['text']}, remove_columns=list(full_valid_dataset.column_names))
        valid_dataset = valid_dataset.map(lambda examples: tokenizer(examples['text'],
                                  padding=True, truncation = True, max_length=1024), 
                                  batched=True, batch_size=args.batch_size, remove_columns=valid_dataset.column_names)
    else:
        full_valid_dataset = None

    print("✅ Dataset prepared.")
    print(f"Training dataset size: {len(train_dataset)}")
    if args.evaluation:
        print(f"Validation dataset size: {len(valid_dataset)}")


    print("\n4. Configuring training...")
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )
    
    # Define a lambda function for compute_metrics to pass extra arguments
    compute_metrics_fn = None
    if args.evaluation:
        compute_metrics_fn = lambda eval_preds: compute_metrics(eval_preds, full_valid_dataset, tokenizer)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps" if args.evaluation else "no",
        save_total_limit=2,
        fp16=True, # Use mixed precision training
        packing=False,  # Must be False when using DataCollatorForCompletionOnlyLM
        remove_unused_columns=False, # Important for passing all columns to compute_metrics
    )

    # Use DataCollatorForCompletionOnlyLM to train only on the assistant's response.
    # The response template is the start of the assistant's turn in the ChatML format.
    response_template_with_context = "[답변]"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm = False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        peft_config=lora_config,
        compute_metrics=compute_metrics_fn,
        processing_class=tokenizer,
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
