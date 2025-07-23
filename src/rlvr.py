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
from retrieve import *
import logging
from datetime import datetime
import logging

#bleurt = evaluate.load("bleurt", module_type="metric")
rouge = evaluate.load("rouge")
#bertscore = evaluate.load("bertscore")
DEFAULT_MODEL_ID = "K-intelligence/Midm-2.0-Base-Instruct"
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/sample_train.json')
DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/rlvr-trained-model')

KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')
RETRIEVER_NAME = "BAAI/bge-m3"
K = 3

log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def parse_arguments():
    parser = argparse.ArgumentParser(description="RLVR training for a language model on a custom QA dataset.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH, help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--retriever", type=str, default=RETRIEVER_NAME, help="Retriever name")
    parser.add_argument("--retrieve", action="store_true", help="Use retrieval-augmented generation")
    parser.add_argument("--retrieve_adaptively", action="store_true", help="Use retrieval-augmented generation adaptively")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
    
    return parser.parse_args()


def load_and_prepare_data(data_path: str, tokenizer: AutoTokenizer, retriever=None, retrieve = False) -> DatasetDict:
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    system_prompt = make_system_prompt()

    def generate_prompt(example):
        question = example['input']['question']
        topic_keyword = example['input']['topic_keyword']
        context = ""
        if retrieve:
            context = retrieve_documents(topic_keyword, question, retriever)
        
        user_prompt = make_prompt_for_grpo(
            question_type=example["input"]["question_type"],
            category=example["input"]["category"],
            domain=example["input"]["domain"],
            topic_keyword=topic_keyword,
            context=context,
            question=question,
            fewshot=True,
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
        return {"prompt": text, "answer": answer, "question_type": example['input']['question_type']}

    # Create dataset from the list of dictionaries and then format it
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(generate_prompt)
    return dataset


def get_formatting_reward(outputs: Dict[str, Any], **kwargs) -> List[float]:
    logging.info("===format reward===")
    rewards = []
    n = len(outputs['prediction'])
    for idx in range(n):
        reward = 0
        question_type = outputs["question_type"][idx]
        if "<answer>" not in outputs["prediction"][idx] or "</answer>" not in outputs["prediction"][idx]:
            reward -= 1
        elif "<reasoning>" not in outputs["prediction"][idx] or "</reasoning>" not in outputs["prediction"][idx]:
            reward -= 0.5
        try:
            prediction = outputs["prediction"][idx].split("<answer>")[1].split("</answer>")[0].strip()
        except:
            prediction = outputs["prediction"][idx]

        if question_type == "선다형":
            if prediction.isdigit():
                reward += 0.5
            else:
                reward -= 0.5
        elif question_type == "단답형":
            if len(prediction.strip().split()) <= 5:
                reward += 0.5
            else:
                reward -= 0.5
        elif question_type == "서술형":
            if len(prediction)> 530 or len(prediction) < 250 or len(prediction.strip().split()) <= 5:
                reward -= 0.5
            elif 313 <= len(prediction) <= 381:
                reward += 0.5
            else:
                reward += 0.1
        rewards.append(reward)
    assert len(rewards) == n
    return rewards


def get_correctness_reward(outputs: Dict[str, Any], **kwargs) -> List[float]:
    logging.info("===correctness reward===")
    rewards = []
    reward = 0
    n = len(outputs['prediction'])
    for idx in range(n):
        question_type = outputs["question_type"][idx]
        try:
            prediction = outputs["prediction"][idx].split("<answer>")[1].split("</answer>")[0].strip()
        except:
            prediction = outputs["prediction"][idx]
        
        answer = outputs["answer"][idx]
        logging.info(f"prediction: {prediction}")
        logging.info(f"answer: {answer}")

        if question_type == "선다형":
            if "번" in answer:
                answer = answer.split("번")[0][-1]
            elif "답변: " in answer:
                answer = answer.split("답변: ")[1][0]
            elif "정답" in answer:
                answer = answer.split("정답")[1][0]
            reward = 1.0 if prediction == answer else -1.0
        elif question_type == "단답형" and '#' in answer:
            if "정답은 " in answer:
                answer = answer.split("정답은 ")[1][0]
            reward = 1.0 if prediction in answer.split('#') else -1.0
        elif question_type == "단답형":
            if "정답은 " in answer:
                answer = answer.split("정답은 ")[1][0]
            reward = 1.0 if prediction == answer else -1.0
        elif question_type == "서술형":
            try:
                reward = rouge.compute(predictions=[prediction], references=[answer])["rougeL"]
                # bleurt_score = bleurt.compute(predictions=[prediction], references=[answer])["scores"][0]
                # rouge_score = rouge.compute(predictions=[prediction], references=[answer])["rouge1"]
                # bert_score = bertscore.compute(predictions=[prediction], references=[answer], lang="ko")["f1"][0]
                # reward = (bleurt_score + rouge_score + bert_score) / 3.0
            except Exception as e:
                print(f"Error calculating descriptive reward: {e}")
                reward = 0.0
        rewards.append(reward)
    assert len(rewards) == n
    return rewards


def get_total_reward(prediction: str, answer: str, question_type: str, **kwargs) -> float:
    output_dict = {"prediction": prediction, "answer": answer, "question_type": question_type}
    formatting_reward = get_formatting_reward(output_dict)
    correctness_reward = get_correctness_reward(output_dict)

    total_reward = []
    assert len(formatting_reward) == len(correctness_reward)
    
    for idx in range(len(formatting_reward)):
        reward = 0.5 * formatting_reward[idx] + 0.5 * correctness_reward[idx]
        logging.info(f"Reward: {reward}")
        total_reward.append(reward)
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

    logging.info("=" * 50)
    logging.info("Starting RLVR")
    logging.info(f"Model ID: {args.model_id}")
    logging.info(f"Retrieval: {args.retrieve}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info("=" * 50)

    logging.info("\n1. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        logging.info("Pad token is not set. Setting pad token to eos token.")
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model with 4-bit quantization for QLoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        #quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    retriever = None
    if args.retrieve:
        logging.info("\n2. Loading retriever...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        retriever = load_retriever(model=RETRIEVER_NAME, device=device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        logging.info("✅ Retriever loaded successfully.")

    if args.retrieve_adaptively:
        retriever = load_retriever_adaptively(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        logging.info("✅ Retriever loaded successfully.")


    logging.info("\n3. Loading and preparing dataset...")
    full_train_dataset = load_and_prepare_data(args.train_data_path, tokenizer, retriever, args.retrieve or args.retrieve_adaptively)

    logging.info("✅ Dataset prepared.")
    logging.info(f"Training dataset size: {len(full_train_dataset)}")

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
        "max_new_tokens": 1024,
    }

    grpo_config = GRPOConfig(
        num_train_epochs = args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        logging_steps=1,
        gradient_accumulation_steps=2,
        max_grad_norm=0.1,
        log_on_each_node=False,
        generation_kwargs=generation_kwargs,
    )

    # Initialize the model with QLoRA adapters
    #model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters()

    grpo_trainer = GRPOTrainer(
        args=grpo_config,
        processing_class=tokenizer,
        model=model,  # Use the model with QLoRA adapters
        reward_funcs=lambda completions, **kwargs: get_total_reward(prediction=completions, answer=kwargs.get('answer'), question_type=kwargs.get('question_type')),
        train_dataset=full_train_dataset,
        peft_config=lora_config,
    )

    grpo_trainer.train()

    total_training_time = time.time() - total_start_time
    
    logging.info("\n" + "="*50)
    logging.info("✅ Training complete.")
    logging.info(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    logging.info(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
    logging.info(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*50)

    logging.info("\n6. Saving the fine-tuned model...")
    grpo_trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"✅ Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
