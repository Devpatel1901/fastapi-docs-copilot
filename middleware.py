from langchain.agents.middleware import dynamic_prompt, ModelRequest
from vecter_store import retrieve_similar_documents
from langchain.agents.middleware import before_model
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain.messages import RemoveMessage
from typing import Any
from langgraph.graph.message import REMOVE_ALL_MESSAGES

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
        "For example, if source is './docs_data/docs/tutorial/body-multiple-params#body-multiple-parameters' then the url will be:"
        "'https://fastapi.tiangolo.com/tutorial/body-multiple-params#body-multiple-parameters'.\n\n"
        f"Context:\n{context}\n\n"
    )
    return system_message

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }