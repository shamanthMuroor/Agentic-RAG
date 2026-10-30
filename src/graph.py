import sys
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from src.llm_router import get_llm
from src.tools import TOOLS
from datetime import date

MAX_TOOL_ITERATIONS = 3

SYSTEM_PROMPT = (
    "You are an assistant for industrial AI questions. "
    "Use search_knowledge_base for concepts about industrial AI, predictive "
    "maintenance, quality control, MLOps, RAG and LLMs. "
    "Use live_web_search for current or recent information. "
    "Answer simple questions such as greetings or arithmetic directly, "
    "without using a tool. "
    "When you use tool results, base your answer only on them. If they do not "
    "contain the answer, say so plainly instead of answering from memory. "
    "Name the source (chunk id or URL) for key claims."
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls_count: int

RETRY_HINT = (
    "Your previous tool call was invalid and was rejected. Each tool takes one "
    "required argument: search_knowledge_base(query: string) and "
    "live_web_search(query: string). Call a tool again with a valid 'query' "
    "string, or answer directly if no tool is needed."
)

def _tool_failure(e: Exception) -> bool:
    return "tool_use_failed" in str(e)

def system_prompt() -> str:
    return f"{SYSTEM_PROMPT} Today's date is {date.today():%B %d, %Y}."

def call_model(state: AgentState) -> dict:
    llm = get_llm()
    if state["tool_calls_count"] < MAX_TOOL_ITERATIONS:
        llm = llm.bind_tools(TOOLS)
    messages = [SystemMessage(content=system_prompt()), *state["messages"]]
    attempts = [messages, [*messages, HumanMessage(content=RETRY_HINT)]]
    for i, msgs in enumerate(attempts):
        try:
            return {"messages": [llm.invoke(msgs)]}
        except Exception as e:
            if not _tool_failure(e):
                raise
            print(f"[warn] provider rejected the tool call (attempt {i + 1})")
    # using this, bcz cloud gpt-oss120b is crashing as it tries to call tool
    try:
        final = get_llm().invoke([
            *messages,
            HumanMessage(content="Do not call any tools. Answer now using only "
                                 "the information already in this conversation."),
        ])
        return {"messages": [final]}
    except Exception as e:
        if not _tool_failure(e):
            raise
        return {"messages": [AIMessage(
            content="Sorry, the model could not produce a valid answer for this "
                    "request. Please try rephrasing the question.")]}

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def execute_tools(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        try:
            content = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        except Exception as e:
            content = f"Tool error: {type(e).__name__}: {e}"
        results.append(
            ToolMessage(content=content, name=call["name"], tool_call_id=call["id"])
        )
    return {"messages": results, "tool_calls_count": state["tool_calls_count"] + 1}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls and state["tool_calls_count"] < MAX_TOOL_ITERATIONS:
        return "tools"
    return "end"

def build_graph():
    grph = StateGraph(AgentState)
    grph.add_node("agent", call_model)
    grph.add_node("tools", execute_tools)
    grph.add_edge(START, "agent")
    grph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    grph.add_edge("tools", "agent")
    return grph.compile()

app_graph = build_graph()

def run(question: str) -> None:
    result = app_graph.invoke(
        {"messages": [HumanMessage(content=question)], "tool_calls_count": 0}
    )
    print("TOOLS USED:", [m.name for m in result["messages"] if m.type == "tool"])
    print("FINAL:", result["messages"][-1].content)

def routing_check() -> None:
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        routing_check()

