import streamlit as st
from langchain_core.messages import HumanMessage
from src.graph import app_graph

st.set_page_config(page_title="AI Assistant", page_icon="🏭")
st.title("AI Assistant")
st.caption("Agentic RAG: local knowledge base + live web search")

if "history" not in st.session_state:
    st.session_state.history = []

for role, text, tools in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if tools:
            st.caption("Tools used: " + ", ".join(tools))

if question := st.chat_input("Ask about nerd stuff..."):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app_graph.invoke(
                {"messages": [HumanMessage(content=question)], "tool_calls_count": 0}
            )
        answer = result["messages"][-1].content
        tools = [m.name for m in result["messages"] if m.type == "tool"]
        st.write(answer)
        if tools:
            st.caption("Tools used: " + ", ".join(tools))
    st.session_state.history.append(("user", question, []))
    st.session_state.history.append(("assistant", answer, tools))



