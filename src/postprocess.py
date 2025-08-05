def postprocess_data(question_type, question, answer):
    try:
        answer = answer.replace("assistant", "")
        answer = answer.replace("ass", "")
        answer = answer.replace("\u0000", "")
    
        if '답변:' in answer:
            answer = answer.split('답변:')[1].strip()
        if '</think>' in answer:
            answer = answer.split('</think>')[1].strip()
        if answer[0] == '>':
            answer = answer[1:]
        
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
            elif answer in question:
                answer = question.split(answer)[0].strip()[-3]
                
        elif question_type == '단답형':
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
            if ':' in answer:
                answer = answer.split(':')[1].strip()
            if '또는' in answer:
                answer = answer.split('또는')[0].strip()
                
        elif question_type == '서술형':
            answer = answer.replace('\n\n', ' ').replace('\n', ' ').strip()
    except:
        pass
    
    return answer
