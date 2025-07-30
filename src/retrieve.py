import os
from collections import defaultdict
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, TypedDict, Optional
from make_prompt import format_docs
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_from_disk
import torch
from sentence_transformers import SentenceTransformer
import tqdm

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
    # if '1\\t' in question:
    #     question = question.split("\\n 1\\t")[0].strip() # 객관식은 선지 제거
    documents = retriever.invoke(topic_keyword + ": " + question)
    logging.info(f"Number of retrieved documents: {len(documents)}")
    context = format_docs(documents)
    return context

def make_contexts(args, retriever, result_data):
    contexts = []
    for item in tqdm.tqdm(result_data):
        question = item["input"]["question"]
        topic_keyword = item["input"]["topic_keyword"]

        context = ""
        if args.retrieve or args.retrieve_adaptively:
            context = retrieve_documents(topic_keyword, question, retriever)
        contexts.append(context)
    return contexts



def load_retriever(model, device, chroma_db_path, kowiki_dataset_path, k=3) -> Optional[Chroma]:
    # Initialize embeddings
    print("Loading embeddings...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device, "trust_remote_code": True},
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

        title_to_texts = defaultdict(list)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
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
    
    return vector_store.as_retriever(search_kwargs={"k": k})


def load_retriever_adaptively(model, device, chroma_db_path, kowiki_dataset_path, k=7, similarity_threshold=0.65) -> Optional[Chroma]:
    # Initialize embeddings
    print("Loading embeddings...")
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
        for text, title, chunk_id in zip(dataset["text"], dataset["title"], dataset["chunk_id"]):
            doc = Document(
                page_content=text,
                metadata={"title": title, "source": "kowiki", "chunk_id": chunk_id}
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
        
        # 최소 3개의 문서를 보장하기 위해, 필터링된 문서가 3개 미만이면 가장 유사한 3개의 문서를 반환
        if len(filtered_docs) < 3 and docs:
            return docs[:3]

        return filtered_docs
    
    # Return a callable object that behaves like a retriever
    class CustomRetriever:
        def __init__(self, retriever_func):
            self.retrieve = retriever_func
            
        def invoke(self, query):
            return self.retrieve(query)
    
    return CustomRetriever(lambda query: custom_retriever(query, k, similarity_threshold))



# class BF16SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name: str, device: str = "cuda"):
#         self.device = torch.device(device)
#         self.model = SentenceTransformer(model_name)
#         self.model = self.model.to(self.device)  # 디바이스 먼저
#         for param in self.model.parameters():
#             param.data = param.data.to(dtype=torch.bfloat16)  # bf16으로 수동 변환

#     def to(self, device: str):
#         self.device = torch.device(device)
#         self.model = self.model.to(self.device)
#         return self

#     def embed_documents(self, texts):
#         embeddings = self.model.encode(texts, convert_to_tensor=True, device=str(self.device))
#         return embeddings.cpu().tolist()

#     def embed_query(self, text):
#         embedding = self.model.encode(text, convert_to_tensor=True, device=str(self.device))
#         return embedding.cpu().tolist()



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
            chunk_size=500, # 800
            chunk_overlap=100, # 150
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
def custom_retriever(query: str, embeddings, vector_store: Chroma, k: int = 7, threshold: float = 0.65):
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
    if len(filtered_docs) < 4 and docs:
        return docs[:4]

    return filtered_docs


class CustomRetriever:
    def __init__(self, retriever_func):
        self.retrieve = retriever_func
        
    def invoke(self, query):
        return self.retrieve(query)