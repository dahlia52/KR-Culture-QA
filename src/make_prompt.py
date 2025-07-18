type_instructions_with_fewshot = {
        "선다형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
            "[예시]\n"
            "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
            "1) 주사위 놀이\n"
            "2) 검무\n"
            "3) 격구\n"
            "4) 영고\n"
            "5) 무애무\n"
            "답변: 3"
        ),
        "서술형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답변을 완성된 문장으로 서술하시오. 문단을 나누지 말고 한 문단으로 서술하시오. 문제에서 물어본 내용만 명확히 드러나도록 작성하라. 한자를 사용하지 마시오.\n\n"
            "[예시]\n"
            "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
            "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
            "[예시]\n"
            "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
            "답변: 정약용"
        )
    }

type_instructions = {
        "선다형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
        ),
        "서술형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답변을 완성된 문장으로 서술하시오. 문단을 나누지 말고 한 문단으로 서술하시오. 문제에서 물어본 내용만 명확히 드러나도록 작성하라. 한자를 사용하지 마시오.\n\n"
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
        )
    }


def make_system_prompt():
    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다."""
    return system_prompt

def make_system_prompt_for_verifier():
    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 질문에 대한 답변이 올바른지 판단하십시오."""
    return system_prompt


def make_system_prompt_with_feedback():
    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 질문, 초안 답변과 피드백이 주어지면, 피드백을 반영해 답변을 재작성하십시오."""
    return system_prompt


def make_prompt(question_type: str, category: str, domain: str, topic_keyword: str, context: str, question: str, fewshot: bool = False, retrieve: bool = True) -> str:
    if fewshot:
        instruction = type_instructions_with_fewshot.get(question_type, "")
    else:
        instruction = type_instructions.get(question_type, "")
    if retrieve:
        template = """{instruction}

        [기타 정보]
        - 카테고리: {category}
        - 도메인: {domain}
        - 주제 키워드: {topic_keyword}

        [참고문헌]
        - 제공된 문서는 답변에 도움이 될 수도, 답변과 무관할 수도 있다. 질문과 관련이 없다면 답변에 참고하지 말고 무시하십시오.
        {context}

        [질문]
        {question}

        [답변]
        """
    else:
        template = """{instruction}

        [기타 정보]
        - 카테고리: {category}
        - 도메인: {domain}
        - 주제 키워드: {topic_keyword}

        [질문]
        {question}

        [답변]
        """
    return template.format(instruction=instruction, category=category, domain=domain, topic_keyword=topic_keyword, context=context, question=question)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def make_verifier_prompt(instruction: str, question: str, answer: str):
    verifier_prompt = """{instruction}

    [질문]
    {question}

    [답변]
    {answer}

    이 답변은 질문에 올바르게 대답한 것입니까?
    예/아니오로 답하시오."""
    return verifier_prompt.format(instruction=instruction, question=question, answer=answer)


def make_prompt_with_feedback(instruction: str, question: str, answer: str, feedback: str):
    prompt = """{instruction}

    [질문]
    {question}

    [초안 답변]
    {answer}

    [피드백]
    {feedback}

    [최종 답변]은 아래 형식으로 답하시오.
    <reasoning> 
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    return prompt.format(instruction=instruction, question=question, answer=answer, feedback=feedback)


def make_prompt_for_reflection(instruction: str, question: str, answer: str):
    prompt = """{instruction}

    [질문]
    {question}

    [초안 답변]
    {answer}

    [자기 성찰]
    - 질문과 관련된 내용만 답하였는가?
    - 위 답변의 논리적 일관성을 검토하라.
    - 질문에 대한 핵심 정보를 모두 다루었는가?
    - 잘못된 전제나 누락된 정보가 있는가?
    - 보다 나은 답변을 생성할 수 있는가?

    [최종 답변]은 아래 형식으로 답하시오.
    <reasoning> 
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    return prompt.format(instruction=instruction, question=question, answer=answer)