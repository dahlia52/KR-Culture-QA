import json

def compare_multiple_choice_answers():
    # Load the first JSON file (final4.json)
    try:
        with open('/data5/wisdomjeong/Korean_Culture_QA_2025/resource/QA/SOTA_64.json', 'r', encoding='utf-8') as f:
            file1_data = json.load(f)
        print(f"Loaded {len(file1_data)} items from SOTA_64.json")
    except Exception as e:
        print(f"Error loading SOTA_64.json: {e}")
        return

    # Load the second JSON file (result_llama3_korean_blossom_8b.json)
    try:
        with open('/data5/wisdomjeong/Korean_Culture_QA_2025/resource/QA/result_aidx.json', 'r', encoding='utf-8') as f:
            file2_data = json.load(f)
        print(f"Loaded {len(file2_data)} items from result_aidx.json")
    except Exception as e:
        print(f"Error loading result_aidx.json: {e}")
        return

    # Create a dictionary for faster lookup of questions from file2
    file2_dict = {}
    for item in file2_data:
        if isinstance(item, dict) and 'input' in item and 'question' in item['input']:
            question = item['input']['question']
            file2_dict[question] = {
                'answer': item.get('output', {}).get('answer', ''),
                'type': item.get('input', {}).get('question_type', '')
            }

    # Compare multiple-choice questions
    different_answers = []
    for item in file1_data:
        if not isinstance(item, dict):
            continue
            
        question = item.get('input', {}).get('question', '')
        question_type = item.get('input', {}).get('question_type', '')
        
        # Only process multiple-choice questions
        if question_type != '선다형':
            continue
            
        answer1 = str(item.get('output', {}).get('answer', '')).strip()
        
        # Find matching question in file2
        if question in file2_dict:
            file2_item = file2_dict[question]
            if file2_item['type'] == '선다형':  # Make sure it's also multiple-choice in file2
                answer2 = str(file2_item['answer']).strip()
                if answer1.lower() != answer2.lower():  # Case-insensitive comparison
                    different_answers.append({
                        'question': question,
                        'final4_answer': answer1,
                        'llama3_answer': answer2
                    })
    
    # Print results
    print(f"\nFound {len(different_answers)} multiple-choice questions with different answers:")
    print("-" * 80)
    
    for i, diff in enumerate(different_answers, 1):
        print(f"{i}. Question: {diff['question']}")
        print(f"   final4.json answer: {diff['final4_answer']}")
        print(f"   aidx.json answer: {diff['llama3_answer']}")
        print("-" * 80)
    
    # Save results to a file
    output_file = 'different_multiple_choice_answers.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Found {len(different_answers)} multiple-choice questions with different answers:\n")
        f.write("=" * 80 + "\n")
        for i, diff in enumerate(different_answers, 1):
            f.write(f"{i}. Question: {diff['question']}\n")
            f.write(f"   final4.json answer: {diff['final4_answer']}\n")
            f.write(f"   aidx.json answer: {diff['llama3_answer']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\nResults have been saved to: {output_file}")

if __name__ == '__main__':
    compare_multiple_choice_answers()
