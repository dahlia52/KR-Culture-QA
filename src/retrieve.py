import os
from collections import defaultdict
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from src.make_prompt import format_docs
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_from_disk
import tqdm



def retrieve_documents(topic_keyword, question, retriever):
    documents = retriever.invoke(topic_keyword + ": " + question)
    logging.info(f"Number of retrieved documents: {len(documents)}")
    context = format_docs(documents)
    return context



def make_contexts(retriever, result_data):
    contexts = []
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]
        context = retrieve_documents(topic_keyword, question, retriever)
        contexts.append(context)
    return contexts



def load_vector_store(model, device, chroma_db_path, kowiki_dataset_path) -> Optional[Chroma]:
    # Initialize embeddings
    print("Loading embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    # embeddings = BF16SentenceTransformerEmbeddings(model, device)

    # Load or create ChromaDB
    if os.path.exists(chroma_db_path):
        print("Loading existing Chroma database...")
        vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings
        )
    else:
        if not os.path.exists(kowiki_dataset_path):
            print("Dataset not found. Please run the preparation script first.")
            return None
    
        # Load dataset
        dataset = load_from_disk(kowiki_dataset_path)

        title_to_texts = defaultdict(list)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100, 
            length_function=len,
            is_separator_regex=False,
        )

        for text, title in zip(dataset["text"], dataset["title"]):
            title_to_texts[title].append(text)

        # Document 생성
        documents = []
        for title, texts in title_to_texts.items():
            combined_text = " ".join(texts)
            split_texts = text_splitter.split_text(combined_text)
            for i, text in enumerate(split_texts):
                doc = Document(
                    page_content=text,
                    metadata={
                        "title": title,
                        "source": "kowiki",
                        "chunk_id": i
                    }
                )
                documents.append(doc)

        logging.info("Creating new Chroma database...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
        vector_store.persist()
    
    return embeddings, vector_store


# Create a custom retriever that uses similarity threshold
def custom_retriever(query: str, embeddings, vector_store: Chroma, k: int = 12, threshold: float = 0.6):
    # Get the base retriever
    base_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # First, get the top k documents
    docs = base_retriever.invoke(query)
    
    # If no documents, return empty list
    if not docs:
        return []
        
    # Get the similarity scores for the retrieved documents
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
    
    # Calculate cosine similarity between query and each document
    from numpy import dot
    from numpy.linalg import norm
    
    similarities = []
    for doc_emb in doc_embeddings:
        # Calculate cosine similarity
        cos_sim = dot(query_embedding, doc_emb) / (norm(query_embedding) * norm(doc_emb))
        similarities.append(cos_sim)
    
    # Filter documents based on similarity threshold
    filtered_docs = [doc for doc, sim in zip(docs, similarities) if sim >= threshold]
    
    # 최소 3개의 문서를 보장하기 위해, 필터링된 문서가 3개 미만이면 가장 유사한 3개의 문서를 반환
    if len(filtered_docs) < 8 and docs:
        return docs[:8]

    return filtered_docs


class CustomRetriever:
    def __init__(self, retriever_func):
        self.retrieve = retriever_func
        
    def invoke(self, query):
        return self.retrieve(query)