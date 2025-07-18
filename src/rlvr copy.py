import argparse
import os
import re
import json
import time
from datetime import timedelta
from typing import List, Dict, Any

import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
import evaluate
from compute_metrics import compute_metrics
from make_prompt import *
from retrieve import load_retriever


bleurt = evaluate.load("bleurt", module_type="metric")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
DEFAULT_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_total+.json')
DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/rlvr-trained-model')

KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
RETRIEVER_NAME = "BAAI/bge-m3"
K = 3

def parse_arguments():
    parser = argparse.ArgumentParser(description="RLVR training for a language model on a custom QA dataset.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--retriever", type=str, default=RETRIEVER_NAME, help="Retriever name")
    parser.add_argument("--retrieve", action="store_true", help="Use retrieval-augmented generation")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
    
    return parser.parse_args()


def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer, retriever=None, retrieve = False) -> DatasetDict:
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
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
        ]
        answer = example['output']['answer']
        # The tokenizer will apply the chat template and add EOS token.
        text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=False,
        )
        return {"prompt": text, "completion": answer, "question_type": example['input']['question_type']}

    # Create dataset from the list of dictionaries and then format it
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(generate_prompt)
    return dataset


def get_formatting_reward(outputs: Dict[str, Any], **kwargs) -> List[float]:
    print("format reward")
    print("=" * 50)
    print(kwargs)
    print("=" * 50)
    rewards = []
    reward = 0
    for output in outputs:
        question_type = output["question_type"]
        prediction = output["completion"]

        if question_type == "선다형":
            if prediction.isdigit():
                reward = 0.5
            else:
                reward = -0.5
        elif question_type == "단답형":
            if len(prediction.strip().split()) <= 5:
                reward = 0.5
            else:
                reward = -0.5
        elif question_type == "서술형":
            if len(prediction)> 530 or len(prediction) < 250 or len(prediction.strip().split()) <= 5:
                reward = -0.5
            elif 313 <= len(prediction) <= 381:
                reward = 0.5
            else:
                reward = 0.1
        else:
            reward = 0
        rewards.append(reward)
    assert len(rewards) == len(outputs)
    return rewards


def get_correctness_reward(outputs: Dict[str, Any], answer: str) -> float:
    rewards = []
    reward = 0
    for output in outputs:
        question_type = output["question_type"]
        prediction = output["completion"]

        if question_type == "선다형":
            reward = 1.0 if prediction == answer else -1.0
        elif question_type == "단답형" and '#' in answer:
            reward = 1.0 if prediction in answer.split('#') else -1.0
        elif question_type == "단답형":
            reward = 1.0 if prediction == answer else -1.0
        elif question_type == "서술형":
            try:
                bleurt_score = bleurt.compute(predictions=[prediction], references=[answer])["scores"][0]
                rouge_score = rouge.compute(predictions=[prediction], references=[answer])["rouge1"]
                bert_score = bertscore.compute(predictions=[prediction], references=[answer], lang="ko")["f1"][0]
                reward = (bleurt_score + rouge_score + bert_score) / 3.0
            except Exception as e:
                print(f"Error calculating descriptive reward: {e}")
                reward = 0.0
        rewards.append(reward)
    assert len(rewards) == len(outputs)
    return rewards

def get_total_reward(prediction: str, answer: str, question_type: str) -> float:
    output_dict = {"answer": prediction, "question_type": question_type}

    formatting_reward = get_formatting_reward(output_dict)
    if formatting_reward < 0:
        return formatting_reward # Penalize and stop if format is wrong

    correctness_reward = get_correctness_reward(output_dict, answer)

    total_reward = 0.5 * formatting_reward + 0.5 * correctness_reward
    return total_reward

def main():
    total_start_time = time.time()

    args = parse_arguments()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Configure 4-bit quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("=" * 50)
    print("Starting RLVR")
    print(f"Model ID: {args.model_id}")
    print(f"Retrieval: {args.retrieve}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)

    print("\n1. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        print("Pad token is not set. Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model with 4-bit quantization for QLoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    retriever = None
    if args.retrieve:
        print("\n2. Loading retriever...")
        device = "cuda:6" if torch.cuda.is_available() else "cpu"
        retriever = load_retriever(model=RETRIEVER_NAME, device=device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")


    print("\n3. Loading and preparing dataset...")
    full_train_dataset = load_and_prepare_data(args.train_data_path, tokenizer, retriever, args.retrieve)

    # def tokenize_function(examples):
    #     return tokenizer(examples['text'], padding=True, truncation=True, max_length=2048)

    # full_train_dataset = full_train_dataset.map(lambda example: {'query': example['text']})

    print("✅ Dataset prepared.")
    print(f"Training dataset size: {len(full_train_dataset)}")

    # QLoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Lora rank
        lora_alpha=32,  # Alpha parameter for scaling
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target modules for QLoRA
    )

    generation_kwargs = {
        "top_k": 30,
        "top_p": 0.9,
        "temperature": 0.8,
        "do_sample": True,
        "max_new_tokens": 2048,
    }

    grpo_config = GRPOConfig(
        num_train_epochs = args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        logging_steps=1,
        gradient_accumulation_steps=8,
        max_grad_norm=0.1,
        log_on_each_node=False,
        generation_kwargs=generation_kwargs,
    )

    # Initialize the model with QLoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        processing_class=tokenizer,
        model=model,  # Use the model with QLoRA adapters
        reward_funcs=get_total_reward,
        train_dataset=full_train_dataset,
        peft_config=lora_config,
    )

    grpo_trainer.train()

    total_training_time = time.time() - total_start_time
    
    print("\n" + "="*50)
    print("✅ Training complete.")
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    print("\n6. Saving the fine-tuned model...")
    grpo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
