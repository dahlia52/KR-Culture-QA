import json

def update_difficulty_scores(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        question_type = item['input']['question_type']
        difficulty_score = item['difficulty_score']
        
        if question_type == '단답형':
            if difficulty_score == 0 and '\n\n' in item['output']['prediction']:
                if item['output']['answer'] == item['output']['prediction'].split("\n\n")[0]:
                    item['difficulty_score'] = 0.4

    data.sort(key=lambda x: x['difficulty_score'], reverse=True)
    
    # Save the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully updated and sorted {len(data)} items in {file_path}")

if __name__ == "__main__":
    file_path = "resource/QA/korean_culture_qa_V1.0_total_difficulty_sorted.json"
    update_difficulty_scores(file_path)
