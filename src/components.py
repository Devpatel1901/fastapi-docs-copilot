from typing import Literal
from langgraph.graph import MessagesState
from src.constants import GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT, GradeDocuments, llm
from src.tools import retrieve_fastapi_doc, validate_fastapi_doc_url
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def get_latest_user_message(state: MessagesState) -> str:
    """Extract the latest user message from the state."""
    for msg in state["messages"][::-1]:
        if isinstance(msg, HumanMessage):
            return msg
    return ""

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the 'retrieve_fastapi_doc' tool, 
    or simply respond to the user.
    """
    response = (
        llm
        .bind_tools([retrieve_fastapi_doc]).invoke(state["messages"])  
    )
    return {"messages": [response]}

def grade_dcouments(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = get_latest_user_message(state).content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )

    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    
def rewrite_question(state: MessagesState):
    """Rewrite the original user's question"""
    question = get_latest_user_message(state).content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

def generate_answer(state: MessagesState): 
    """Generate an answer"""
    question = get_latest_user_message(state).content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(context=context)

    response = llm.bind_tools([validate_fastapi_doc_url]).invoke([
        {"role": "user", "content": question},
        {"role": "system", "content": prompt}
    ]) 
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_fastapi_doc]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_dcouments
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
