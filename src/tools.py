from langchain.tools import tool
from src.vector_store import retrieve_similar_documents
import aiohttp
import os
from langchain_community.vectorstores import FAISS
from pathlib import Path
from src.constants import embeddings

@tool
async def validate_fastapi_doc_url(url: str) -> bool:
    """Validate if a FastAPI documentation URL is reachable asynchronously.

    Args:
        url (str): The FastAPI documentation URL to validate.
    Returns:
        bool: True if the URL is reachable (HTTP 200), False otherwise.
    """

    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.head(url, allow_redirects=True) as response:
                return response.status == 200
        except aiohttp.ClientError:
            return False
        
@tool
def retrieve_fastapi_doc(query: str) -> str:
    """Retrieve relevant FastAPI documentation sections based on a query.

    Args:
        query (str): The user's query related to FastAPI documentation.
    Returns:
        str: An information about relevant FastAPI documentation sections.
    """

    faiss_path = Path(__file__).resolve().parents[1] / 'faiss_index'
    db = None

    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    retrieved_docs = []
    if db:
        retrieved_docs = db.similarity_search(query, k=2)

    return "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
