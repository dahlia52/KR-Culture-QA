import os
import argparse
import json
import time
from datetime import timedelta
from typing import Dict, List

import torch
import numpy as np
import evaluate
from langchain.vectorstores import Chroma
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EvalPrediction,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from make_prompt import make_prompt, format_docs
from retrieve import load_retriever

# Get the project root directory
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_ID = "skt/A.X-4.0-Light"
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_train+.json')
DEFAULT_VALID_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_dev+.json')

DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/fine-tuned-model')
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
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    parser.add_argument("--evaluation", action="store_true", help="Evaluate validation set")
    parser.add_argument("--retriever", type=str, default=RETRIEVER_NAME, help="Retriever name")
    parser.add_argument("--retrieve", action="store_true", help="Use retrieval-augmented generation")

    return parser.parse_args()

def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer, retriever=None, retrieve = False) -> DatasetDict:
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Assign persona
    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. 
    단, 동일한 문장을 절대 반복하지 마시오."""

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

    # The first mapping already created the 'text' field and removed the original columns.
    # The second call was redundant.
    # dataset = dataset.map(generate_prompt, remove_columns=['input', 'output'])
    return dataset


def compute_metrics(eval_preds: EvalPrediction, eval_dataset: Dataset, tokenizer: AutoTokenizer):
    # 1. Load metrics
    accuracy_metric = evaluate.load("accuracy")
    exact_match_metric = evaluate.load("exact_match")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    bleurt_metric = evaluate.load("bleurt", module_type="metric")

    # 2. Decode predictions and labels
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 (token used for masking) with pad_token_id
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 3. Group by question_type
    grouped_examples = {
        '선다형': {'preds': [], 'labels': []},
        '단답형': {'preds': [], 'labels': []},
        '서술형': {'preds': [], 'labels': []},
    }

    for i, item in enumerate(eval_dataset):
        q_type = item['input']['question_type']
        if q_type in grouped_examples:
            grouped_examples[q_type]['preds'].append(decoded_preds[i])
            grouped_examples[q_type]['labels'].append(decoded_labels[i])

    # 4. Compute metrics for each group
    results = {}
    # '선다형': accuracy
    if grouped_examples['선다형']['preds']:
        acc_results = accuracy_metric.compute(predictions=grouped_examples['선다형']['preds'], references=grouped_examples['선다형']['labels'])
        results['accuracy'] = acc_results['accuracy']

    # '단답형': exact_match
    if grouped_examples['단답형']['preds']:
        em_results = exact_match_metric.compute(predictions=grouped_examples['단답형']['preds'], references=grouped_examples['단답형']['labels'])
        results['exact_match'] = em_results['exact_match']

    # '서술형': ROUGE, BERTScore, BLEURT average
    if grouped_examples['서술형']['preds']:
        preds = grouped_examples['서술형']['preds']
        labels = grouped_examples['서술형']['labels']
        
        rouge_results = rouge_metric.compute(predictions=preds, references=labels)
        bertscore_results = bertscore_metric.compute(predictions=preds, references=labels, lang='ko')
        bleurt_results = bleurt_metric.compute(predictions=preds, references=labels)
        
        results['rougeL'] = rouge_results['rougeL']
        results['bertscore_f1'] = np.mean(bertscore_results['f1'])
        results['bleurt'] = np.mean(bleurt_results['scores'])
        
        # 서술형 평균 점수
        descriptive_avg = np.mean([results['rougeL'], results['bertscore_f1'], results['bleurt']])
        results['descriptive_avg_score'] = descriptive_avg

    return results


def main():
    total_start_time = time.time()
    
    args = parse_arguments()

    print("=" * 50)
    print("Starting Fine-Tuning")
    print(f"Model ID: {args.model_id}")
    print(f"Retrieval: {args.retrieve}")
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
    print("✅ Tokenizer and model loaded.")

    retriever = None
    if args.retrieve:
        print("\n2. Loading retriever...")
        retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")

    print("\n3. Loading and preparing dataset...")
    full_train_dataset = load_and_prepare_data(args.train_data_path, tokenizer, retriever, args.retrieve)
    train_dataset = full_train_dataset.map(lambda example: {'text': example['text']}, remove_columns=list(full_train_dataset.column_names))

    valid_dataset = None
    if args.evaluation:
        full_valid_dataset = load_and_prepare_data(args.valid_data_path, tokenizer, retriever, args.retrieve)
        valid_dataset = full_valid_dataset.map(lambda example: {'text': example['text']}, remove_columns=list(full_valid_dataset.column_names))
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
        #dataset_text_field="text",
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=50,
        eval_strategy="epoch" if args.evaluation else "no",
        save_total_limit=2,
        fp16=True, # Use mixed precision training
        packing=False,  # Must be False when using DataCollatorForCompletionOnlyLM
        remove_unused_columns=False, # Important for passing all columns to compute_metrics
    )

    # Use DataCollatorForCompletionOnlyLM to train only on the assistant's response.
    # The response template is the start of the assistant's turn in the ChatML format.
    response_template = "답변:"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        peft_config = lora_config,
        compute_metrics=compute_metrics_fn,
        processing_class=tokenizer
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
