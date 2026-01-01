from dotenv import load_dotenv
from vecter_store import retrieve_similar_documents
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

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
        "If the context does not contain the answer, respond with 'I don't know.'"
        "As well as cite the source of the information from the context."
        "You can generate web page url by appending the source path to 'https://fastapi.tiangolo.com/' "
        "at start and only picking last two values of path from metadata source attribute."
        "For example, if source is 'docs_data/docs/tutorial/body-fields.md' then the url will be:"
        "'https://fastapi.tiangolo.com/tutorial/body-fields'.\n\n"
        f"Context:\n{context}\n\n"
    )
    return system_message

agent = create_agent(
    llm,
    tools=[],
    middleware=[prompt_with_context],
)

for step in agent.stream(
    {"messages": [HumanMessage("Give me all the information about header parameters in FastAPI that you have.")]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

