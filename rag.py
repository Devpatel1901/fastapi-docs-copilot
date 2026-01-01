from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from middleware import prompt_with_context, trim_messages

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

def initiate_qa_bot():
    input_query = ""
    agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=InMemorySaver(),
        middleware=[trim_messages, prompt_with_context],
    )

    while input_query.lower() != "exit":
        input_query = input("Enter your question about FastAPI docs (or type 'exit' to quit): ")
        if input_query.lower() == "exit":
            break

        for step in agent.stream(
            {"messages": [HumanMessage(input_query)]},
            config={"configurable": {"thread_id": "1"}},
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()

if __name__ == "__main__":
    initiate_qa_bot()
