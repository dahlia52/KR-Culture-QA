import argparse
import os
import re
from typing import List, Dict, Any

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import evaluate
from compute_metrics import compute_metrics


bleurt = evaluate.load("bleurt", module_type="metric")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
DEFAULT_MODEL_ID = "skt/A.X-4.0-Light"
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_total+.json')
DEFAULT_OUTPUT_DIR = os.path.join(current_dir, 'models/rlvr_trained_model')

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
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--retriever", type=str, default=RETRIEVER_NAME, help="Retriever name")
    parser.add_argument("--retrieve", action="store_true", help="Use retrieval-augmented generation")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1')")
    
    return parser.parse_args()


def parse_output(output: str) -> Dict[str, Any]:
    """Parse the model output to extract question type and answer."""
    # This is a placeholder. You need to adapt this to your specific output format.
    # Example format: "[서술형] 정답: 조선 시대의 주요 왕은 세종대왕입니다."
    q_type_match = re.search(r"\[(.*?)\]", output)
    answer_match = re.search(r"정답: (.*)", output)

    q_type = q_type_match.group(1) if q_type_match else "unknown"
    answer = answer_match.group(1).strip() if answer_match else ""

    return {"question_type": q_type, "answer": answer}


def get_formatting_reward(output: Dict[str, Any]) -> float:
    question_type = output["question_type"]
    prediction = output["answer"]

    if question_type == "선다형":
        if prediction.isdigit():
            return 0.5
        else:
            return -0.5
    elif question_type == "단답형":
        if len(prediction.strip().split()) <= 5:
            return 0.5
        else:
            return -0.5
    elif question_type == "서술형":
        if len(prediction)> 530 or len(prediction) < 250 or len(prediction.strip().split()) <= 5:
            return -0.5
        elif 313 <= len(prediction) <= 381:
            return 0.5
        else:
            return 0.1
    else:
        return 0


def get_correctness_reward(output: Dict[str, Any], answer: str) -> float:
    question_type = output["question_type"]
    prediction = output["answer"]

    if question_type == "선다형":
        return 1.0 if prediction == answer else -1.0
    elif question_type == "단답형" and '#' in answer:
        return 1.0 if prediction in answer.split('#') else -1.0
    elif question_type == "단답형":
        return 1.0 if prediction == answer else -1.0
    elif question_type == "서술형":
        try:
            bleurt_score = bleurt.compute(predictions=[prediction], references=[answer])["scores"][0]
            rouge_score = rouge.compute(predictions=[prediction], references=[answer])["rouge1"]
            bert_score = bertscore.compute(predictions=[prediction], references=[answer], lang="ko")["f1"][0]
            return (bleurt_score + rouge_score + bert_score) / 3.0
        except Exception as e:
            print(f"Error calculating descriptive reward: {e}")
            return 0.0
    else:
        return 0.0

def get_total_reward(output: str, answer: str, question_type: str) -> float:
    parsed_output = parse_output(output)
    parsed_output['question_type'] = question_type # Use ground truth type for reward calculation

    formatting_reward = get_formatting_reward(parsed_output)
    if formatting_reward < 0:
        return formatting_reward # Penalize and stop if format is wrong

    correctness_reward = get_correctness_reward(parsed_output, answer)

    # Combine rewards (e.g., with weighting)
    total_reward = 0.5 * formatting_reward + 0.5 * correctness_reward
    return total_reward

def main():
    total_start_time = time.time()

    args = parse_arguments()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

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

    retriever = None
    if args.retrieve:
        print("\n2. Loading retriever...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        retriever = load_retriever(model=RETRIEVER_NAME, device=device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
        if not retriever:
            raise Exception("Failed to initialize retriever")
        print("✅ Retriever loaded successfully.")

    # 1. PPO Config
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.epochs,
        log_with="wandb",
    )

    # 2. Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")


    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        # peft_config=lora_config
    )

    # 4. Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None, # ref_model is created automatically if None
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # 5. Training Loop
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 2048,
    }

    for epoch in range(ppo_config.ppo_epochs):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]

            # Get response from the model
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = tokenizer.batch_decode(response_tensors)

            # Calculate reward
            rewards = []
            for i in range(len(batch["response"])):
                reward_val = get_total_reward(
                    output=batch["response"][i],
                    ground_truth=batch["ground_truth_answer"][i],
                    ground_truth_q_type=batch["question_type"][i]
                )
                rewards.append(torch.tensor(reward_val))
            
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    # Save the model
    ppo_trainer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
