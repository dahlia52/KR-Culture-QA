import argparse
import torch
from tqdm import tqdm
from src.data_io import load_dataset, save_dataset
from src.load_model import load_llm
import tqdm
from src.make_prompt import make_prompt_for_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert multiple-choice questions to descriptive format using LLM')
    parser.add_argument('--input', type=str, help='Path to input JSON file')
    parser.add_argument('--output', type=str, help='Path to save output JSON file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--model_id', type=str, help='Model ID to use for generation')
    parser.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    return parser.parse_args()


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

    return result_data

 
def main():
    args = parse_arguments()
    # Load the dataset
    print(f"Loading dataset from {args.input}...")
    result_data = load_dataset(args.input)
    print(f"Loaded {len(result_data)} QA pairs.")

    # Extract MC data
    mc_data = [item for item in result_data if item['input']['question_type'] == '선다형']
    else_data = [item for item in result_data if item['input']['question_type'] != '선다형']
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    pipe = load_llm(model_id=args.model_id, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
    if not pipe:
        raise Exception("Failed to initialize language model pipeline")
    print("✅ Language model pipeline loaded successfully.")
    
    print("\n" + "=" * 50)
    print("Making data...")
    print("=" * 50)
    mc_data = generate_data(args, pipe, mc_data)
    print("\n" + "=" * 50)
    print("Data remake Completed for Multiple Choice Data")
    print("=" * 50)

    result_data = mc_data + else_data
    save_dataset(result_data, args.output)

if __name__ == "__main__":
    main()
