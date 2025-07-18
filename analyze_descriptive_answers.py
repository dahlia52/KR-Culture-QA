import json
import numpy as np

# Load the JSON file
with open('resource/QA/korean_culture_qa_V1.0_total+.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract answers for descriptive questions
descriptive_answers = []
for item in data:
    if item['input']['question_type'] == '단답형':
        answer = item['output']['answer']
        descriptive_answers.append(len(answer.split(" ")))
        if (len(answer.split(" ")) > 2):
            print(answer)

# Calculate statistics
if descriptive_answers:
    answers_np = np.array(descriptive_answers)
    stats = {
        'count': len(descriptive_answers),
        'mean': np.mean(answers_np),
        'std': np.std(answers_np),
        'min': np.min(answers_np),
        '25%': np.percentile(answers_np, 25),
        '50% (median)': np.median(answers_np),
        '75%': np.percentile(answers_np, 75),
        'max': np.max(answers_np)
    }
    
    print("Descriptive Answers Length Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
#     # Additional information
#     print(f"\nTotal number of descriptive questions: {len(descriptive_answers)}")
#     print("\nAnswer length distribution (characters):")
#     print(f"- Shortest answer: {np.min(answers_np)} characters")
#     print(f"- Longest answer: {np.max(answers_np)} characters")
#     print(f"- Average length: {np.mean(answers_np):.2f} ± {np.std(answers_np):.2f} characters")
#     print("\nQuartiles:")
#     print(f"- Q1 (25th percentile): {np.percentile(answers_np, 25):.2f} characters")
#     print(f"- Q2 (Median): {np.median(answers_np):.2f} characters")
#     print(f"- Q3 (75th percentile): {np.percentile(answers_np, 75):.2f} characters")
    
#     # Get all descriptive questions with their answers and lengths
#     desc_questions = []
#     for item in data:
#         if item['input']['question_type'] == '단답형':
#             answer = item['output']['answer']
#             desc_questions.append({
#                 'question': item['input']['question'],
#                 'answer': answer,
#                 'length': len(answer)
#             })
    
#     # Sort by answer length
#     desc_questions.sort(key=lambda x: x['length'])
    
#     # Show examples
#     print("\nExamples:")
#     print(f"\n1. Shortest answer ({desc_questions[0]['length']} chars):")
#     print(f"   Question: {desc_questions[0]['question']}")
#     print(f"   Answer: {desc_questions[0]['answer'][:100]}...")
    
#     median_idx = len(desc_questions) // 2
#     print(f"\n2. Median length answer ({desc_questions[median_idx]['length']} chars):")
#     print(f"   Question: {desc_questions[median_idx]['question']}")
#     print(f"   Answer: {desc_questions[median_idx]['answer'][:100]}...")
    
#     print(f"\n3. Longest answer ({desc_questions[-1]['length']} chars):")
#     print(f"   Question: {desc_questions[-1]['question']}")
#     print(f"   Answer: {desc_questions[-1]['answer'][:100]}...")
# else:
#     print("No descriptive questions found in the dataset.")
