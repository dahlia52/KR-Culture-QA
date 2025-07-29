import json
import os

def preprocess_data(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {input_path}.")
        return

    processed_data = []
    for item in data:
        question_type = item['input']['question_type']
        answer = item['output']['answer'].strip()
        rationale = item['output']['rationale_answer'].strip()
        rationale_answer = rationale.split('<answer>')[1].split('</answer>')[0].strip()
        if rationale_answer != answer:
            rationale = rationale.replace(rationale_answer, answer)
        item['output']['answer'] = rationale
        processed_data.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Preprocessing complete. Processed data saved to {output_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_file = os.path.join(project_root, 'resource', 'QA', 'split_by_type/korean_culture_qa_선다형_with_rationale.json')
    output_file = os.path.join(project_root, 'resource', 'QA', 'split_by_type/korean_culture_qa_선다형_with_rationale_preprocessed.json')
    
    preprocess_data(input_file, output_file)
