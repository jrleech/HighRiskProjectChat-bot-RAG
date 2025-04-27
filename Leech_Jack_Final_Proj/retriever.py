# retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(docs):
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_relevant_docs(db, query, k=5):
    return db.similarity_search(query, k=k)
