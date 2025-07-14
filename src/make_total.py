import json
import os

output_file = 'korean_culture_qa_V1.0_total+.json'
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_train+.json')
DEFAULT_VALID_DATA_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_dev+.json')

QA_OUTPUT_PATH = os.path.join(current_dir, 'resource/QA/korean_culture_qa_V1.0_total+.json')
# 파일 읽기
with open(DEFAULT_TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(DEFAULT_VALID_DATA_PATH, 'r', encoding='utf-8') as f:
    dev_data = json.load(f)

# 데이터 합치기 (두 파일의 구조에 따라 이 부분을 조정해야 할 수 있음)
# 만약 두 파일이 리스트 형태라면:
if isinstance(train_data, list) and isinstance(dev_data, list):
    combined_data = train_data + dev_data
# 만약 딕셔너리 형태라면:
elif isinstance(train_data, dict) and isinstance(dev_data, dict):
    combined_data = {**train_data, **dev_data}
else:
    raise ValueError("파일 형식이 예상과 다릅니다. 리스트나 딕셔너리 형태여야 합니다.")

# 합쳐진 데이터를 새 파일로 저장
with open(QA_OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"파일이 성공적으로 합쳐져 {QA_OUTPUT_PATH}로 저장되었습니다.")