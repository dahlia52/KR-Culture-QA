import os
import argparse
import json
import tqdm
from typing import List, TypedDict, Optional
import torch
from datasets import load_from_disk
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langgraph.graph import StateGraph, END
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from data import CustomDataset

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'retrieval_docs/chroma_db')

# Create QA directory if it doesn't exist
qa_dir = os.path.join(current_dir, 'resource/QA')
os.makedirs(qa_dir, exist_ok=True)

QA_DATASET_PATH = os.path.join(qa_dir, 'korean_culture_qa_V1.0_test+.json')
QA_OUTPUT_PATH = os.path.join(qa_dir, 'result.json')

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
g.add_argument("--model_id", type=str, default="skt/A.X-4.0-Light", help="huggingface model id")
g.add_argument("--tokenizer", type=str, default="skt/A.X-4.0-Light", help="huggingface tokenizer")
g.add_argument("--device", type=str, default="cuda", help="device to load the model")
g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
# fmt: on

args = parser.parse_args()

# Hyperparameters
K = 3

# Model and path configurations
RETRIEVER_NAME = "BAAI/bge-m3"
GENERATOR_NAME = args.model_id


class GraphState(TypedDict):
    question: str
    question_type: str
    category: str
    domain: str
    topic_keyword: str
    documents: List[Document]
    retriever: Chroma
    llm: HuggingFacePipeline
    answer: str


def retrieve_documents(state: GraphState) -> GraphState:
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = state["retriever"].invoke(question)
    return {"documents": documents}


def make_prompt(question_type: str, category: str, domain: str, topic_keyword: str, context: str, question: str) -> str:
    # question type별 instruction 정의
    type_instructions = {
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
        ),
        "교정형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
            "[예시]\n"
            "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
            "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."
        ),
        "선택형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
            "[예시]\n"
            "질문: \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
            "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다."
        )
    }

    instruction = type_instructions.get(question_type, "")

    # RAG에 사용될 최종 프롬프트 템플릿
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
    return template.format(instruction=instruction, category=category, domain=domain, topic_keyword=topic_keyword, context=context, question=question)


def generate_answer(state: GraphState) -> GraphState:
    print("---GENERATING ANSWER---")
    llm = state["llm"]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    system_prompt = """You are a helpful AI assistant. 
    당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. 
    단, 동일한 문장을 절대 반복하지 마시오."""

    context = format_docs(state["documents"])

    user_prompt = make_prompt(
        question_type=state["question_type"],
        category=state["category"],
        domain=state["domain"],
        topic_keyword=state["topic_keyword"],
        context=context,
        question=state["question"]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    answer = llm.invoke(messages)
    return {"answer": answer}


def load_retriever(device, k=3) -> Optional[Chroma]:
    if not os.path.exists(KOWIKI_DATASET_PATH):
        print("Dataset not found. Please run the preparation script first.")
        return None
    
    # Load dataset
    dataset = load_from_disk(KOWIKI_DATASET_PATH)

    # Create documents with metadata
    documents = []
    for text, doc_id in zip(dataset["text"], dataset["id"]):
        doc = Document(
            page_content=text,
            metadata={"id": str(doc_id), "source": "kowiki"}
        )
        documents.append(doc)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=RETRIEVER_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Load or create ChromaDB
    if os.path.exists(CHROMA_DB_PATH):
        print("Loading existing Chroma database...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    else:
        print("Creating new Chroma database...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        vector_store.persist()
    
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_llm(device):
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    #model = AutoModelForCausalLM.from_pretrained(GENERATOR_NAME).to(device)
    
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",  # 또는 torch.float16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_NAME,
        quantization_config=quantization_config,
        device_map=device
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.7,
    )

    return HuggingFacePipeline(pipeline=pipe), tokenizer


def main(args):
    print("=" * 50)
    print("Starting Korean Culture QA System")
    print("=" * 50)
    
    retriever = load_retriever(device=args.device, k=K)
    if not retriever:
        raise Exception("Failed to initialize retriever")
    print("✅ Retriever loaded successfully.")
    
    llm, tokenizer = load_llm(device=args.device)
    if not llm:
        raise Exception("Failed to initialize language model")
    print("✅ Language model loaded successfully.")
    
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    print("Compiling workflow...")
    app = workflow.compile()

    file_test = args.input
    #dataset = CustomDataset(file_test, tokenizer)

    with open(file_test, "r") as f:
        result = json.load(f)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)

    for idx in tqdm.tqdm(range(len(result))):
        topic_keyword = result[idx]["input"]["topic_keyword"]
        question_type = result[idx]["input"]["question_type"]
        category = result[idx]["input"]["category"]
        domain = result[idx]["input"]["domain"]
        q = result[idx]["input"]["question"]
        output_text = app.invoke({"retriever": retriever, "question": q, "topic_keyword": topic_keyword, "question_type": question_type, "category": category, "domain": domain, "llm": llm})["answer"]

        # Process output
        output_text = output_text.split("답변:\n")[1].strip()

        result[idx]["output"] = {"answer": output_text}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

    
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

if __name__ == '__main__':
    exit(main(parser.parse_args()))