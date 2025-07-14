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
from graphstate import GraphState
from langgraph.graph import StateGraph, END
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from make_prompt import make_prompt, format_docs
from retrieve import retrieve_documents, load_retriever

# Get the project root directory (one level up from src)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_DATASET_PATH = os.path.join(current_dir, 'resource/retrieval_docs/kowiki_dataset')
CHROMA_DB_PATH = os.path.join(current_dir, 'resource/retrieval_docs/chroma_db')

# Create QA directory if it doesn't exist
qa_dir = os.path.join(current_dir, 'resource/QA')
os.makedirs(qa_dir, exist_ok=True)

QA_DATASET_PATH = os.path.join(qa_dir, 'sample_qa.json')
QA_OUTPUT_PATH = os.path.join(qa_dir, 'result.json')

# Hyperparameters
K = 3

def parse_arguments():
    parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--input", type=str, default=QA_DATASET_PATH, help="input filename")
    g.add_argument("--output", type=str, default=QA_OUTPUT_PATH, help="output filename")
    g.add_argument("--model_id", type=str, default="skt/A.X-3.1-Light", help="huggingface model id") #  skt/A.X-4.0-Light
    g.add_argument("--tokenizer", type=str, default="skt/A.X-3.1-Light", help="huggingface tokenizer")
    g.add_argument("--device", type=str, default="cuda", help="device to load the model")
    g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
    g.add_argument("--quantize", action="store_true", help="Whether to apply 4-bit quantization to the model")
    g.add_argument("--batch_size", type=int, default=10, help="Batch size for inference.")
    return parser.parse_args()


def generate_answer(state: GraphState) -> GraphState:
    print("---GENERATING ANSWER---")
    llm = state["llm"]

    system_prompt = """당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. 
    단, 동일한 문장을 절대 반복하지 마시오."""

    context = format_docs(state["documents"])

    user_prompt = make_prompt(
        question_type=state["question_type"],
        category=state["category"],
        domain=state["domain"],
        topic_keyword=state["topic_keyword"],
        context=context,
        question=state["question"],
        fewshot=True,
        retrieve=True
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    answer = llm.invoke(messages)
    return {"answer": answer}



def load_llm(model_id, device, quantize=False, batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
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
            model_id,
            quantization_config=quantization_config,
            device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        top_k=30,
        temperature=0.7,
        batch_size=batch_size
    )

    return HuggingFacePipeline(pipeline=pipe)


def main():
    args = parse_arguments()
    RETRIEVER_NAME = "BAAI/bge-m3"
    GENERATOR_NAME = args.model_id

    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    
    print("=" * 50)
    print("Starting Korean Culture QA System")
    print("=" * 50)

    retriever = load_retriever(model=RETRIEVER_NAME, device=args.device, chroma_db_path=CHROMA_DB_PATH, kowiki_dataset_path=KOWIKI_DATASET_PATH, k=K)
    if not retriever:
        raise Exception("Failed to initialize retriever")
    print("✅ Retriever loaded successfully.")
    
    llm = load_llm(model_id=GENERATOR_NAME, device=args.device, quantize=args.quantize, batch_size=args.batch_size)
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
    with open(file_test, "r") as f:
        result_data = json.load(f)

    print("\n" + "=" * 50)
    print("Starting QA Session")
    print("=" * 50)

    inputs_for_batch = []
    for item in result_data:
        inputs_for_batch.append({
            "retriever": retriever,
            "llm": llm,
            "question": item["input"]["question"],
            "topic_keyword": item["input"]["topic_keyword"],
            "question_type": item["input"]["question_type"],
            "category": item["input"]["category"],
            "domain": item["input"]["domain"],
        })

    all_outputs = []
    for i in tqdm.tqdm(range(0, len(inputs_for_batch), args.batch_size)):
        batch_inputs = inputs_for_batch[i:i + args.batch_size]
        batch_outputs = app.batch(batch_inputs)
        all_outputs.extend(batch_outputs)

    for idx, output in enumerate(all_outputs):
        output_text = output["answer"]
        print("="*30)
        print(output_text)
        try:
            output_text = output_text.split("답변:\n")[1].strip()
        except Exception:
            output_text = output_text.strip()
        result_data[idx]["output"] = {"answer": output_text}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False, indent=4))

    
    print("\n" + "=" * 50)
    print("QA Session Completed")
    print("=" * 50)

    torch.cuda.synchronize()
    print(f"최대 VRAM 사용량: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")


if __name__ == '__main__':
    exit(main())