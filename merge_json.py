import json
import os

# Define file paths
input_dir = "resource/QA/split_by_type"
output_dir = "resource/QA"

# Read 단답형 JSON file
with open(os.path.join(input_dir, "korean_culture_qa_선다형_to_서술형.json"), "r", encoding="utf-8") as f:
    short_answer_data = json.load(f)

# Read 서술형 JSON file
with open(os.path.join(input_dir, "korean_culture_qa_서술형.json"), "r", encoding="utf-8") as f:
    essay_data = json.load(f)

# Merge the two lists
merged_data = short_answer_data + essay_data

# Sort by id to maintain order
merged_data.sort(key=lambda x: int(x["id"]))

# Write to output file
output_file = os.path.join(output_dir, "korean_culture_qa_선다형+서술형.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"Merged file saved to: {output_file}")
