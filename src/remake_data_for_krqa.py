import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from load import *
from typing import List, Dict, Any, Optional, TypedDict
import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert multiple-choice questions to descriptive format using LLM')
    parser.add_argument('--input', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'resource/KIIP/sample_KIIP.json'),
                        help='Path to input JSON file')
    parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'resource/KIIP/KIIP_선다형_to_서술형.json'),
                         help='Path to save output JSON file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to run the model on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--model_id', type=str, default='K-intelligence/Midm-2.0-Base-Instruct',
                        help='Model ID to use for generation')
    parser.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    return parser.parse_args()



def make_prompt_for_data(question: str, answer: str) -> str:
    template = """
    다음 선다형 문제를 서술형 문제로 변환하여 아래 형식으로 답하시오. 서술형 문제는 선택지 없이 서술하는 형태이며 질문은 간단하게 한 문장으로 구성되어야 합니다. 
    
    [선다형 문제]
    -질문 : {question}
    -답변 : {answer}

    [서술형 문제]
    -질문: <question> ... </question>
    -답변: <answer> ... </answer>
    """
    return template.format(question=question, answer=answer)


def generate_data(args, pipe, result_data):
    prompts = []
    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 아래 지시에 따라 문제 형식을 변환하여 답하시오."""
    for item in tqdm.tqdm(result_data):
        user_prompt = make_prompt_for_data(question=item['input']['question'], answer=item['output']['answer'])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)

    outputs = pipe(prompts)

    for idx, output in enumerate(tqdm.tqdm(outputs)):
        # The output from the pipeline is a list with a dictionary
        generated_text = output[0]['generated_text']
        answer = generated_text[-1]['content']
        try:
            generated_question = answer.split("<question>")[1].split("</question>")[0].strip()
            generated_answer = answer.split("<answer>")[1].split("</answer>")[0].strip()
            result_data[idx]['input']['question_type'] = '서술형'
            result_data[idx]['input']['question'] = generated_question
            result_data[idx]['output']['answer'] = generated_answer
        except:
            try:
                generated_question = answer.split("질문:")[1].split("답변:")[0].strip()
                generated_answer = answer.split("답변:")[1].strip()
                result_data[idx]['input']['question_type'] = '서술형'
                result_data[idx]['input']['question'] = generated_question
                result_data[idx]['output']['answer'] = generated_answer
            except:
                print(f"===Failed to make generated {idx} question===")
                print(answer)
                continue

    save_dataset(result_data, args.output)

 
def main():
    args = parse_arguments()
    # Load the dataset
    print(f"Loading dataset from {args.input}...")
    result_data = load_dataset(args.input)
    print(f"Loaded {len(result_data)} QA pairs.")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    pipe, tokenizer = load_llm(model_id=args.model_id, base_model_name=args.model_id, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
    if not pipe:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")
    
    print("\n" + "=" * 50)
    print("Making data...")
    print("=" * 50)
    generate_data(args, pipe, result_data)
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)


if __name__ == "__main__":
    exit(main())
    
