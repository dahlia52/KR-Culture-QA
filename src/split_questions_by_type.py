import json
from pathlib import Path

def load_questions(json_path):
    """Load questions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def split_questions_by_type(questions):
    """Split questions into different types."""
    question_types = {
        '선다형': [],
        '단답형': [],
        '서술형': []
    }
    
    for item in questions:
        q_type = item['input']['question_type']
        if q_type in question_types:
            question_types[q_type].append(item)
    
    return question_types

def save_questions(questions, output_path):
    """Save questions to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

def main():
    # Define paths
    base_dir = Path('/data5/wisdomjeong/Korean_Culture_QA_2025')
    input_file = base_dir / 'resource' / 'QA' / 'korean_culture_qa_V1.0_total+.json'
    output_dir = base_dir / 'resource' / 'QA' / 'split_by_type'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process questions
    print(f"Loading questions from {input_file}...")
    questions = load_questions(input_file)
    print(f"Total questions loaded: {len(questions)}")
    
    # Split questions by type
    print("Splitting questions by type...")
    question_types = split_questions_by_type(questions)
    
    # Save each type to separate files
    for q_type, items in question_types.items():
        output_file = output_dir / f'korean_culture_qa_{q_type}.json'
        save_questions(items, output_file)
        print(f"Saved {len(items)} {q_type} questions to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for q_type, items in question_types.items():
        print(f"- {q_type}: {len(items)} questions")

if __name__ == "__main__":
    main()
