import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_vector_store():
    faiss_path = './faiss_index'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = None

    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    
    return db

def retrieve_similar_documents(query, k=2):
    db = get_vector_store()

    results = []
    if db:
        results = db.similarity_search(query, k=k)
        
    return results