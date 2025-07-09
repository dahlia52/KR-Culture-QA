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
            "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
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
            "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
        )
    }



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
        {context}

        [질문]
        {question}

        답변:
        """
    else:
        template = """{instruction}

        [기타 정보]
        - 카테고리: {category}
        - 도메인: {domain}
        - 주제 키워드: {topic_keyword}

        [질문]
        {question}

        답변:
        """
    return template.format(instruction=instruction, category=category, domain=domain, topic_keyword=topic_keyword, context=context, question=question)
    

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)