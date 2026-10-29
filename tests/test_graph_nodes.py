from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.graph import MAX_TOOL_ITERATIONS, execute_tools, should_continue

def ask_tool(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])

def test_continues_when_tool_requested():
    state = {"messages": [HumanMessage(content="hi"), ask_tool("search_knowledge_base", {"query": "x"})],
             "tool_calls_count": 0}
    assert should_continue(state) == "tools"

def test_stops_at_iteration_cap():
    state = {"messages": [ask_tool("search_knowledge_base", {"query": "x"})],
             "tool_calls_count": MAX_TOOL_ITERATIONS}
    assert should_continue(state) == "end"

def test_stops_without_tool_calls():
    state = {"messages": [AIMessage(content="done")], "tool_calls_count": 0}
    assert should_continue(state) == "end"

def test_unknown_tool_does_not_crash():
    out = execute_tools({"messages": [ask_tool("nonexistent_tool", {})], "tool_calls_count": 0})
    msg = out["messages"][0]
    assert out["tool_calls_count"] == 1
    assert isinstance(msg, ToolMessage) and "error" in msg.content.lower()

def test_knowledge_base_tool_runs():
    out = execute_tools({"messages": [ask_tool("search_knowledge_base", {"query": "predictive maintenance"})],
                         "tool_calls_count": 0})
    assert "predictive_maintenance" in out["messages"][0].content
