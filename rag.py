from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from middleware import prompt_with_context
from langchain_core.messages.ai import AIMessageChunk

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

def initiate_qa_bot():
    input_query = ""
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
            )
        ],
    )

    while input_query.lower() != "exit":
        input_query = input("Enter your question about FastAPI docs (or type 'exit' to quit): ")
        full_message = ""
        if input_query.lower() == "exit":
            break

        for token, _ in agent.stream(
            {"messages": [HumanMessage(input_query)]},
            config={"configurable": {"thread_id": "1"}},
            stream_mode="messages"
        ):
            if isinstance(token, AIMessageChunk):
                if token.chunk_position != "last":  
                    full_message += token.content[0]["text"]

        print("\nAI Response:")
        print(full_message)

if __name__ == "__main__":
    initiate_qa_bot()
