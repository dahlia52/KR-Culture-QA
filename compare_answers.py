import json

def compare_answers():
    try:
        with open('/data5/wisdomjeong/Korean_Culture_QA_2025/resource/QA/result_train.json', 'r', encoding='utf-8') as f:
            model_answers = json.load(f)
            print(f"Loaded {len(model_answers)} model answers")
    except json.JSONDecodeError as e:
        print(f"Error reading result_train.json: {e}")
        return
    except FileNotFoundError:
        print("result_train.json not found.")
        return

    try:
        with open('/data5/wisdomjeong/Korean_Culture_QA_2025/resource/QA/korean_culture_qa_V1.0_train+.json', 'r', encoding='utf-8') as f:
            correct_answers_data = json.load(f)
            print(f"Loaded {len(correct_answers_data)} correct answers")
    except json.JSONDecodeError as e:
        print(f"Error reading korean_culture_qa_V1.0_train+.json: {e}")
        return
    except FileNotFoundError:
        print("korean_culture_qa_V1.0_train+.json not found.")
        return

    incorrect_multiple_choice = []
    incorrect_short_answer = []
    descriptive_answers = []
    total_questions = 0
    correct_count = 0

    # 디버깅을 위해 처음 5개 항목의 정답 출력
    print("\nSample correct answers (first 5):")
    for i, item in enumerate(correct_answers_data[:5]):
        print(f"{i+1}. {item.get('input', {}).get('question')}")
        print(f"   Answer: {item.get('output', {}).get('answer')}")
        print(f"   Type: {item.get('input', {}).get('question_type')}")

    print("\nSample model answers (first 5):")
    for i, item in enumerate(model_answers[:5]):
        print(f"{i+1}. {item.get('input', {}).get('question')}")
        print(f"   Answer: {item.get('output', {}).get('answer')}")
        print(f"   Type: {item.get('input', {}).get('question_type')}")

    # 모든 모델 답변에 대해 정답과 비교
    for model_ans in model_answers:
        total_questions += 1
        model_answer = model_ans.get('output', {}).get('answer')
        model_question = model_ans.get('input', {}).get('question')
        model_type = model_ans.get('input', {}).get('question_type')
        
        # 해당 질문과 일치하는 정답 찾기
        matched = False
        for correct_ans_item in correct_answers_data:
            correct_question = correct_ans_item.get('input', {}).get('question')
            correct_answer = correct_ans_item.get('output', {}).get('answer')
            
            # 질문이 정확히 일치하는 경우에만 비교
            if correct_question == model_question:
                matched = True
                if str(correct_answer).strip() == str(model_answer).strip():
                    correct_count += 1
                else:
                    # 틀린 답변 저장
                    if model_type in ['선다형', '단답형']:
                        incorrect_entry = {
                            'question': model_question,
                            'model_answer': model_answer,
                            'correct_answer': correct_answer,
                            'type': model_type
                        }
                        if model_type == '선다형':
                            incorrect_multiple_choice.append(incorrect_entry)
                        else:
                            incorrect_short_answer.append(incorrect_entry)
                    elif model_type == '서술형':
                        descriptive_answers.append({
                            'question': model_question,
                            'model_answer': model_answer,
                            'correct_answer': correct_answer
                        })
                break
        
        if not matched:
            print(f"\nWarning: No matching question found for model answer:")
            print(f"Question: {model_question}")
            print(f"Answer: {model_answer}")
    
    # 정확도 출력
    if total_questions > 0:
        accuracy = (correct_count / total_questions) * 100
        print(f"\n정확도: {accuracy:.2f}% ({correct_count}/{total_questions})")

    # 결과를 파일에 저장
    with open('comparison_result.txt', 'w', encoding='utf-8') as f:
        # 정확도 정보 추가
        f.write(f"총 문제 수: {total_questions}\n")
        f.write(f"정답 수: {correct_count}\n")
        f.write(f"정확도: {accuracy:.2f}%\n\n")
        
        f.write("--- 잘못된 선다형 문제 ---\n")
        if incorrect_multiple_choice:
            for i, item in enumerate(incorrect_multiple_choice, 1):
                f.write(f"{i}. Q: {item['question']}\n")
                f.write(f"   Model: {item['model_answer']}\n")
                f.write(f"   Correct: {item['correct_answer']}\n\n")
        else:
            f.write("없음\n\n")

        f.write("--- 잘못된 단답형 문제 ---\n")
        if incorrect_short_answer:
            for i, item in enumerate(incorrect_short_answer, 1):
                f.write(f"{i}. Q: {item['question']}\n")
                f.write(f"   Model: {item['model_answer']}\n")
                f.write(f"   Correct: {item['correct_answer']}\n\n")
        else:
            f.write("없음\n\n")

        f.write("--- 서술형 문제 (검토 필요) ---\n")
        if descriptive_answers:
            for i, item in enumerate(descriptive_answers, 1):
                f.write(f"{i}. Q: {item['question']}\n")
                f.write(f"   Model: {item['model_answer']}\n")
                f.write(f"   Correct: {item['correct_answer']}\n\n")
        else:
            f.write("없음\n\n")

    print("\n비교 완료. 'comparison_result.txt' 파일을 확인하세요.")
    print(f"- 총 문제 수: {total_questions}")
    print(f"- 정답 수: {correct_count}")
    print(f"- 정확도: {accuracy:.2f}%")
    print(f"- 잘못된 선다형: {len(incorrect_multiple_choice)}")
    print(f"- 잘못된 단답형: {len(incorrect_short_answer)}")
    print(f"- 검토 필요한 서술형: {len(descriptive_answers)}")

if __name__ == '__main__':
    compare_answers()
