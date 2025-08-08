import json
from pathlib import Path
from src.data_io import load_dataset, save_dataset

def merge_json_files(file1_path, file2_path, output_path):
    data1 = load_dataset(file1_path)
    data2 = load_dataset(file2_path)
    
    merged_data = data1 + data2
    
    save_dataset(merged_data, output_path)
    
    print(f"Successfully merged {len(data1)} items from {file1_path} and {len(data2)} items from {file2_path}")
    print(f"Total items in merged file: {len(merged_data)}")
    print(f"Merged file saved to: {output_path}")


def main():
    base_dir = Path("./resource/QA/data")
    train_file = base_dir / "korean_culture_qa_V1.0_train+.json"
    dev_file = base_dir / "korean_culture_qa_V1.0_dev+.json"
    output_file = base_dir / "korean_culture_qa_V1.0_total+.json"

    merge_json_files(train_file, dev_file, output_file)


if __name__ == "__main__":
    exit(main())