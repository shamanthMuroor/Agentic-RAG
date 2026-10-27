from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from src.llm_router import get_llm
from src.tools import TOOLS

SYSTEM_PROMPT = (
    "You are an assistant for industrial AI questions. "
    "Use search_knowledge_base for concepts about industrial AI, predictive "
    "maintenance, quality control, MLOps, RAG and LLMs. "
    "Use live_web_search for current or recent information. "
    "Answer simple questions such as greetings or arithmetic directly, "
    "without using a tool."
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls_count: int

def call_model(state: AgentState) -> dict:
    llm = get_llm().bind_tools(TOOLS)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}

if __name__ == "__main__":
    questions = [
        "What is predictive maintenance?",
        "What changed in the latest Python release?",
        "What is 2 + 2?",
        "Explain statistical process control.",
        "Any news about new NVIDIA GPUs this month?",
    ]
    for q in questions:
        out = call_model({"messages": [HumanMessage(content=q)], "tool_calls_count": 0})
        msg = out["messages"][0]
        if msg.tool_calls:
            print(f"TOOL   {msg.tool_calls[0]['name']} {msg.tool_calls[0]['args']}  <- {q}")
        else:
            print(f"DIRECT {msg.content[:70]!r}  <- {q}")

