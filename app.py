import streamlit as st
from langchain_core.messages import HumanMessage
from src.graph import app_graph

st.set_page_config(page_title="AI Assistant", page_icon="🏭")
st.title("AI Assistant")
st.caption("Agentic RAG: local knowledge base + live web search")

def tool_trace(messages) -> list[dict]:
    args_by_id = {}
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            args_by_id[call["id"]] = call["args"]
    return [
        {"tool": m.name, "args": args_by_id.get(m.tool_call_id, {}), "result": m.content}
        for m in messages
        if m.type == "tool"
    ]

def show_trace(trace: list[dict]) -> None:
    for t in trace:
        with st.expander(f"🔧 {t['tool']}  {t['args']}"):
            st.text(t["result"][:3000])

if "history" not in st.session_state:
    st.session_state.history = []

for role, text, trace in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        show_trace(trace)

if question := st.chat_input("Ask about nerd stuff..."):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app_graph.invoke(
                {"messages": [HumanMessage(content=question)], "tool_calls_count": 0}
            )
        answer = result["messages"][-1].content
        trace = tool_trace(result["messages"])
        st.write(answer)
        show_trace(trace)
    st.session_state.history.append(("user", question, []))
    st.session_state.history.append(("assistant", answer, trace))

