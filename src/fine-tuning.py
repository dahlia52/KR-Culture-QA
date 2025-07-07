import os
import argparse
import json
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Get the project root directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_ID = "skt/A.X-4.0-Light"
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_train+.json')
DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/fine-tuned-model')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tuning a language model on a custom QA dataset.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save the fine-tuned model")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")

    return parser.parse_args()


def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer) -> DatasetDict:
    """Load data from JSON, format it, and tokenize it."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    def generate_prompt(example):
        # Create a single text field for the trainer.
        # The EOS token is important to signal the end of a sequence for the model.
        full_prompt = f"### 질문:\n{example['input']['question']}\n\n### 답변:\n{example['output']['answer']}{tokenizer.eos_token}"
        return {"text": full_prompt}

    # Create dataset from the list of dictionaries and then format it
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(generate_prompt)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=True, max_length=512),
        batched=True,
        remove_columns=['input', 'output', 'text']  # Remove original and intermediate columns
    )

    return DatasetDict({"train": tokenized_dataset})


def main():
    args = parse_arguments()

    print("=" * 50)
    print("Starting Fine-Tuning")
    print(f"Model ID: {args.model_id}")
    print(f"Training Data: {args.data_path}")
    print("=" * 50)

    print("\n1. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map=args.device
    )
    print("✅ Tokenizer and model loaded.")

    print("\n2. Loading and preparing dataset...")
    dataset = load_and_prepare_data(args.data_path, tokenizer)
    print("✅ Dataset prepared.")
    print(f"Training dataset size: {len(dataset['train'])}")

    print("\n3. Configuring training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",  # No evaluation set provided
        save_total_limit=2,
        fp16=True, # Use mixed precision training
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    print("✅ Training configured.")

    # 4. Start Training
    print("\n4. Starting model training...")
    trainer.train()
    print("✅ Training complete.")

    # 5. Save the Final Model
    print("\n5. Saving the fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
