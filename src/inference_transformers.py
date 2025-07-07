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
from make_prompt import make_prompt

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')

# Create QA directory if it doesn't exist
qa_dir = os.path.join(current_dir, 'resource/QA')
os.makedirs(qa_dir, exist_ok=True)

QA_DATASET_PATH = os.path.join(qa_dir, 'sample_qa.json')
QA_OUTPUT_PATH = os.path.join(qa_dir, 'result.json')

parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
g.add_argument("--model_id", type=str, default="skt/A.X-4.0-Light", help="huggingface model id")
g.add_argument("--tokenizer", type=str, default="skt/A.X-4.0-Light", help="huggingface tokenizer")
g.add_argument("--device", type=str, default="cuda", help="device to load the model")
g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
g.add_argument("--quantize", default=False, action="store_true", help="Whether to apply 4-bit quantization to the model")
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


def generate_answer(state: GraphState) -> GraphState:
    print("---GENERATING ANSWER---")
    llm = state["llm"]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. 
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

        print("Creating new Chroma database...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        vector_store.persist()
    
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_llm(device, quantize=False):
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if quantize:
        print("Quantizing model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_NAME,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_NAME,
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

    return HuggingFacePipeline(pipeline=pipe)


def main(args):
    print(f"현재 사용 가능한 장치: {torch.cuda.get_device_name(0)}")
    print(f"총 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    
    print("=" * 50)
    print("Starting Korean Culture QA System")
    print("=" * 50)
    
    retriever = load_retriever(device=args.device, k=K)
    if not retriever:
        raise Exception("Failed to initialize retriever")
    print("✅ Retriever loaded successfully.")
    
    llm = load_llm(device=args.device, quantize=args.quantize)
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
        print("="*30)
        print(output_text)
        output_text = output_text.split("답변:\n")[1].strip()

        result[idx]["output"] = {"answer": output_text}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

    
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main(parser.parse_args()))