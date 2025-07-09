import os
from graphstate import GraphState
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, TypedDict, Optional

def retrieve_documents(state: GraphState) -> GraphState:
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = state["retriever"].invoke(question)
    return {"documents": documents}


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

        print("Creating new Chroma database...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
        vector_store.persist()
    
    return vector_store.as_retriever(search_kwargs={"k": k})
