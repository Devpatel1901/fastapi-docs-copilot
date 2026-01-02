from langchain.messages import HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from src.agent import agent

def initiate_qa_bot():
    input_query = ""

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
