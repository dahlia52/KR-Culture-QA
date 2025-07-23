import os
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, TypedDict, Optional
from make_prompt import format_docs, make_prompt_for_retrieval
import logging

# def retrieve_documents(state: GraphState) -> GraphState:
#     print("---RETRIEVING DOCUMENTS---")
#     question = state["question"]
#     documents = state["retriever"].invoke(question)
#     return {"documents": documents}

# def deduplicate_documents(docs):
#     seen = set()
#     unique_docs = []

#     for doc in docs:
#         key = (doc.page_content.strip(), doc.metadata.get("source"), doc.metadata.get("page"))

#         if key not in seen:
#             seen.add(key)
#             unique_docs.append(doc)

#     return unique_docs



def retrieve_documents(topic_keyword, question, retriever):
    documents = retriever.invoke(topic_keyword + ": " + question)
    logging.info(f"Number of retrieved documents: {len(documents)}")
    context = format_docs(documents)
    return context


def retrieve_documents_with_rewritten_query(question, retriever):
    documents = retriever.invoke(question)
    logging.info(f"Number of retrieved documents: {len(documents)}")
    context = format_docs(documents)
    return context



def load_retriever(model, device, chroma_db_path, kowiki_dataset_path, k=3) -> Optional[Chroma]:
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
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

        # Create documents with metadata
        documents = []
        for text, doc_id in zip(dataset["text"], dataset["id"]):
            doc = Document(
                page_content=text,
                metadata={"id": str(doc_id), "source": "kowiki"}
            )
            documents.append(doc)

        logging.info("Creating new Chroma database...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
        vector_store.persist()
    
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_retriever_adaptively(model, device, chroma_db_path, kowiki_dataset_path, k=7, similarity_threshold=0.7) -> Optional[Chroma]:
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
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

        # Create documents with metadata
        documents = []
        for text, doc_id in zip(dataset["text"], dataset["id"]):
            doc = Document(
                page_content=text,
                metadata={"id": str(doc_id), "source": "kowiki"}
            )
            documents.append(doc)

        logging.info("Creating new Chroma database...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
        vector_store.persist()
    
    # Create a custom retriever that uses similarity threshold
    def custom_retriever(query: str, k: int = k, threshold: float = similarity_threshold):
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
        
        # 최소 2개의 문서를 보장하기 위해, 필터링된 문서가 2개 미만이면 가장 유사한 2개의 문서를 반환
        if len(filtered_docs) < 2 and docs:
            return docs[:2]

        return filtered_docs
    
    # Return a callable object that behaves like a retriever
    class CustomRetriever:
        def __init__(self, retriever_func):
            self.retrieve = retriever_func
            
        def invoke(self, query):
            return self.retrieve(query)
    
    return CustomRetriever(lambda query: custom_retriever(query, k, similarity_threshold))
