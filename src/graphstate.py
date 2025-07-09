from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.llms import VLLM

class GraphState(TypedDict):
    question: str
    question_type: str
    category: str
    domain: str
    topic_keyword: str
    documents: List[Document]
    retriever: Chroma
    llm: VLLM
    answer: str