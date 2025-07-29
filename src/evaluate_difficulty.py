import os
import json
import argparse
from tqdm import tqdm
import torch
from typing import List, Dict, Any, Tuple
from inference_transformers import parse_arguments, main as inference_main
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from make_prompt import *
from retrieve import *
from load import *
import logging
from datetime import datetime
import evaluate
import tqdm

# Set up logging
log_filename = f"difficulty_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO)

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_DATASET_PATH = os.path.join(current_dir, 'resource/QA/split_by_type/korean_culture_qa_선다형_to_서술형.json')
QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/split_by_type/korean_culture_qa_선다형_to_서술형_difficulty_sorted.json')


def generate(pipe, result_data: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    prompts = []
    system_prompt = make_system_prompt()
    
    logging.info("Preparing prompts...")
    
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]

        context = ""
        if args.retrieve or args.retrieve_adaptively:
            context = retrieve_documents(topic_keyword, question, retriever)
        
        user_prompt = make_prompt(
            question_type=item["input"]["question_type"],
            category=item["input"]["category"],
            domain=item["input"]["domain"],
            topic_keyword=topic_keyword,
            context=context,
            question=question,
            fewshot=True,
            retrieve = args.retrieve or args.retrieve_adaptively
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # pipeline's tokenizer will apply the chat template
        prompts.append(messages)

    logging.info("Generating answers in batch...")
    outputs = pipe(prompts)

    logging.info("Processing generated answers...")
    for idx, output in enumerate(tqdm.tqdm(outputs)):
        prediction = output[0]['generated_text']
        print(prediction)
        result_data[idx]["output"]["prediction"] = prediction.strip()

    return result_data

def evaluate_difficulty(result_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bleurt = evaluate.load("bleurt", module_type="metric")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    for idx, item in tqdm.tqdm(enumerate(result_data)):
        if item["input"]['question_type'] == "선다형":
            difficulty_score = 0.5 if item['output']['prediction'] == item['output']['answer'] else 0
        elif item["input"]['question_type'] == "단답형" and '#' in item['output']['answer']:
            difficulty_score = 0.6 if item['output']['prediction'] in item['output']['answer'].split("#") else 0.1
        elif item["input"]['question_type'] == "단답형":
            difficulty_score = 0.6 if item['output']['prediction'] == item['output']['answer'] else 0.1
        elif item["input"]['question_type'] == "서술형":
            bleurt_score = bleurt.compute(predictions=[item['output']['prediction']], references=[item['output']['answer']])["scores"][0]
            rouge_score = rouge.compute(predictions=[item['output']['prediction']], references=[item['output']['answer']])["rouge1"]
            bert_score = bertscore.compute(predictions=[item['output']['prediction']], references=[item['output']['answer']], lang="ko")["f1"][0]
            difficulty_score = (bleurt_score + rouge_score + bert_score) / 3.0
        
        result_data[idx]["difficulty_score"] = difficulty_score
    
    return result_data


def sort_by_difficulty(qa_data: List[Dict[str, Any]], ascending: bool = True) -> List[Dict[str, Any]]:
    return sorted(qa_data, key=lambda x: x.get("difficulty_score", 1.0), reverse=not ascending)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate QA difficulty for curriculum learning")
    parser.add_argument("--input", type=str, default=QA_DATASET_PATH,
                       help="Path to input QA JSON file")
    parser.add_argument("--output", type=str, default=QA_OUTPUT_PATH,
                       help="Path to save the sorted output JSON file")
    parser.add_argument("--model_id", type=str, default="K-intelligence/Midm-2.0-Base-Instruct",
                       help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--retrieve", action="store_true", help="Whether to use retrieval-augmented generation")
    parser.add_argument("--retrieve_adaptively", action="store_true", help="Whether to use retrieval-augmented generation")
    
    args = parser.parse_args()
    
    # Load the QA dataset
    logging.info(f"Loading QA dataset from {args.input}")
    qa_data = load_dataset(args.input)
    logging.info(f"Loaded {len(qa_data)} QA pairs")
    
    # Initialize model and tokenizer
    logging.info(f"Loading model {args.model_id}")

    GENERATOR_NAME = args.model_id
    
    pipe, tokenizer = load_llm(model_id=GENERATOR_NAME, base_model_name=GENERATOR_NAME, device=args.device, quantize=False, batch_size=args.batch_size, is_lora=False, lora_weights=None, return_full_text=False)

    logging.info("Generating answers...")
    result_data = generate(pipe, qa_data, args)

    logging.info("Starting difficulty evaluation...")
    qa_data_with_difficulty = evaluate_difficulty(result_data)
    
    # Sort by difficulty
    logging.info("Sorting by difficulty...")
    sorted_qa_data = sort_by_difficulty(qa_data_with_difficulty)
    
    # Save the results
    logging.info(f"Saving results to {args.output}")
    save_dataset(sorted_qa_data, args.output)
    
    logging.info("Done!")
    
    # Print some statistics
    difficulty_scores = [item.get("difficulty_score", 1.0) for item in sorted_qa_data]
    print(f"\nDifficulty Statistics:")
    print(f"- Easiest 10%: {sorted(difficulty_scores)[:int(len(difficulty_scores)*0.1)][-1]:.2f}")
    print(f"- Median: {sorted(difficulty_scores)[len(difficulty_scores)//2]:.2f}")
    print(f"- Hardest 10%: {sorted(difficulty_scores)[-int(len(difficulty_scores)*0.1)]:.2f}")
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
