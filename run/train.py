import os
import argparse
import time
from datetime import timedelta

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from src.make_prompt import *
from src.data_io import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning a language model on a custom QA dataset.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--train_data_path", required=True, type=str, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the fine-tuned model")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()



def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer, retriever=None, retrieve = False) -> DatasetDict:
    raw_data = load_dataset(data_path)
    
    # Assign persona
    system_prompt = make_system_prompt()
    def generate_prompt(example):
        question = example['input']['question']
        
        user_prompt = make_prompt(
            question_type=example["input"]["question_type"],
            category=example["input"]["category"],
            domain=example["input"]["domain"],
            topic_keyword=example["input"]["topic_keyword"],
            context="",
            question=question,
            fewshot=False
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
    print("Start Fine-Tuning")
    print(f"Model ID: {args.model_id}")
    print("LoRA Fine-Tuning: Enabled")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)

    retriever = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        print("Setting pad token.")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))  # 반드시 tokenizer 추가 후 호출
        model.config.pad_token_id = tokenizer.pad_token_id
    print("✅ Tokenizer and model loaded.")

    is_lora = os.path.isdir(args.model_id) and 'adapter_config.json' in os.listdir(args.model_id)
    lora_weights = args.model_id if is_lora else None
    
    if is_lora:
        print(f"🔍 Detected a fine-tuned LoRA model at: {args.model_id}")
        config = PeftConfig.from_pretrained(args.model_id)
        base_model_name = config.base_model_name_or_path
        print(f"🔧 Loading base model: {base_model_name}")
        print(f"Loading LoRA weights from {lora_weights}")
        model = PeftModel.from_pretrained(model, lora_weights)
        model = model.merge_and_unload()


    print("\n3. Loading and preparing dataset...")
    full_train_dataset = load_and_prepare_data(args.train_data_path, tokenizer, retriever, False)
    train_dataset = full_train_dataset.map(lambda example: {'text': example['text']}, remove_columns=list(full_train_dataset.column_names))
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'],
                                  padding=True, truncation = True, max_length=1024), 
                                  batched=True, batch_size=args.batch_size, remove_columns=train_dataset.column_names)

    print("✅ Dataset prepared.")
    print(f"Training dataset size: {len(train_dataset)}")


    print("\n4. Configuring training...")
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none"
        )
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=50,
        eval_strategy="no",
        save_total_limit=2,
        bf16=True, # Use mixed precision training
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
        data_collator=data_collator,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    print("✅ Training configured.")

    print("\n5. Starting model training...")
    trainer.train()
    
    print("\n" + "="*50)
    print("Training complete.")
    print(f"Total training time: {timedelta(seconds=int(time.time() - total_start_time))}")
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    print("\n6. Merging LoRA adapter with base model...")
    model = trainer.model
    model = model.merge_and_unload()
    model = model.to(dtype = torch.bfloat16)

    print("\n7. Saving the fine-tuned model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f" Model saved to {args.output_dir}")
    

if __name__ == "__main__":
    main()
