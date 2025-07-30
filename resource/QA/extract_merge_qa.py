import json

# 파일 경로 설정
data1_path = "SOTA_64.json"
data2_path = "final_재현_preprocessed.json"
output_path = "merged_selected_QA.json"

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_questions(data, types):
    # 각 문항의 'input' dict의 'question_type'을 기준으로 필터링
    return [item for item in data if item.get('input', {}).get('question_type') in types]

def main():
    # 파일 불러오기
    data1 = load_json(data1_path)
    data2 = load_json(data2_path)

    # SOTA_64.json: '선다형', '서술형'
    filtered1 = filter_questions(data1, ['선다형', '서술형'])
    # final_재현_preprocessed.json: '단답형'
    filtered2 = filter_questions(data2, ['단답형'])

    # 병합
    merged = filtered1 + filtered2

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged file saved: {output_path} (총 {len(merged)} 문항)")

if __name__ == "__main__":
    main()
