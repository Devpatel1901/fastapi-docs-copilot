from langchain.agents.middleware import dynamic_prompt, ModelRequest
from vecter_store import retrieve_similar_documents

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = retrieve_similar_documents(last_query, k=2)

    context = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    system_message = (
        "You are an AI assistant for answering questions about FastAPI documentation. "
        "Use the following context from the documentation to answer the user's question. "
        "If the context does not contain the answer, respond with 'I don't know, try to search with different keywords.'"
        "As well as cite the source of the information from the context."
        "You can generate web page url by appending the source path to 'https://fastapi.tiangolo.com/' "
        "at start and only picking last two values of path from metadata source attribute."
        "For example,"
        "1. source: './docs_data/docs/tutorial/body-multiple-params#body-multiple-parameters'"
        "   url: 'https://fastapi.tiangolo.com/tutorial/body-multiple-params/#body-multiple-parameters'." \
        "2. source: './docs_data/docs/python-types/#declaring-types'" \
        "   url: 'https://fastapi.tiangolo.com/python-types/#declaring-types'.\n\n"
        f"Context:\n{context}\n\n"
    )
    return system_message
