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
        if item['output']['answer'][0] == '>':
            item['output']['answer'] = item['output']['answer'][1:]
        question_type = item['input']['question_type']
        answer = item['output']['answer']
        if '답변:' in answer:
            answer = answer.split('답변:')[1].strip()

        if question_type == '선다형':
            processed_answer = answer.strip()[0]
            assert processed_answer in ['1', '2', '3', '4', '5'], f"Invalid multiple choice answer: {answer}"
            item['output']['answer'] = processed_answer
                
        elif question_type == '단답형':
            if '또는' in answer:
                item['output']['answer'] = answer.split('또는')[0].strip()
                
        elif question_type == '서술형':
            item['output']['answer'] = answer.replace('\n\n', ' ').replace('\n', ' ').strip()
        processed_data.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Preprocessing complete. Processed data saved to {output_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_file = os.path.join(project_root, 'resource', 'QA', 'result_self_reflection.json')
    output_file = os.path.join(project_root, 'resource', 'QA', 'result_self_reflection_preprocessed.json')
    
    preprocess_data(input_file, output_file)
