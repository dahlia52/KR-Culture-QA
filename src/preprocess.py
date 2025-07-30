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
        item['output']['answer'] = item['output']['answer'].replace("\n", " ").replace("\t", " ").strip()
        question_type = item['input']['question_type']
        answer = item['output']['answer'].strip()
        question = item['input']['question'].strip()
        print("Answer", answer)

        try:
            if question_type == '선다형' and answer.split("assistant")[1][0].isdigit():
                answer = answer.split("assistant")[1][0]
            else:
                if '</think>' in answer:
                    answer = answer.split('</think>')[1].strip()
                if 'ass' in answer:
                    answer = answer.split('ass')[0].strip()

                if '\u0000' in answer:
                    answer = answer.split('\u0000')[0].strip()

                if '(dAtA' in answer:
                    answer = answer.split('(dAtA')[0].strip()

                if '////' in answer:
                    answer = answer.split('////')[0].strip()

                if '답변:' in answer:
                    answer = answer.split('답변:')[1].strip()

                if answer[0] == '>':
                    answer = answer[1:]
        except:
            pass

        if question_type == '선다형':
            if answer[0].isdigit():
                answer = answer[0]
            elif "정답은" in answer:
                answer = answer.split("정답은")[1].strip()
                q_num = int(question.split("\\t")[-2][-1].strip())
                if answer[0].isdigit():
                    answer = answer[0]
                else:
                    cnt = 1
                    for i in range(-q_num, 0):
                        if question.split("\\t")[i][:-1].strip() in answer:
                            answer = str(cnt)
                            break
                        cnt += 1
            elif answer in item['input']['question']:
                answer = item['input']['question'].split(answer)[0].strip()[-3]
            assert answer in ['1', '2', '3', '4', '5'], f"Invalid multiple choice answer: {answer}"
                
        elif question_type == '단답형':
            if '또는' in answer:
                answer = answer.split('또는')[0].strip()
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
            if ':' in answer:
                answer = answer.split(':')[1].strip()
                
        elif question_type == '서술형':
            answer = answer.replace('\n\n', ' ').replace('\n', ' ').strip()
        
        item['output']['answer'] = answer
        processed_data.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Preprocessing complete. Processed data saved to {output_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_file = os.path.join(project_root, 'resource', 'QA', 'final_finetuned1_q.json')
    output_file = os.path.join(project_root, 'resource', 'QA', 'final_finetuned1_q_preprocessed.json')
    
    preprocess_data(input_file, output_file)
