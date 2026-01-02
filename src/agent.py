from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelRetryMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from src.middleware import prompt_with_context

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

def format_error(error: Exception) -> str:
    return f"Model call failed: {error}. Please try again later."

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        prompt_with_context,
        SummarizationMiddleware(
            model=llm,
            trigger=("fraction", 0.7), # gemini-3-pro-preview has 1M token context window
            keep=("fraction", 0.3)
        ),
        ModelRetryMiddleware(
            max_retries=2,
            on_failure=format_error
        )
    ],
)