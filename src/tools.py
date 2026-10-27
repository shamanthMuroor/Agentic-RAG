from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.retrieval import retrieve
from ddgs import DDGS

MIN_SIMILARITY = 0.60 # based on initial testing, might need to recalibrate

class KBQuery(BaseModel):
    query: str = Field(
        description="Focused search query about industrial AI, predictive "
                    "maintenance, quality control, MLOps, RAG or LLMs."
    )

@tool("search_knowledge_base", args_schema=KBQuery)
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base of documents on industrial AI, predictive
    maintenance, quality control, MLOps, RAG and LLMs. Use for conceptual or
    definitional questions covered by those topics."""
    hits = [(d, s) for d, s in retrieve(query, k=3) if s >= MIN_SIMILARITY]
    if not hits:
        return "No relevant documents found in the knowledge base."
    return "\n\n---\n\n".join(
        f"[{d.metadata['chunk_id']} | similarity={s:.2f}]\n{d.page_content}"
        for d, s in hits
    )

class WebQuery(BaseModel):
    query: str = Field(description="Web search query for current or recent information.")

@tool("live_web_search", args_schema=WebQuery)
def live_web_search(query: str) -> str:
    """Search the live web. Use for recent events, current versions, news, or
    anything likely newer than the local knowledge base."""
    try:
        results = DDGS().text(query, max_results=3)
    except Exception as e:
        return f"Web search failed: {type(e).__name__}: {e}"
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[{i}] {r.get('title')}\n{r.get('href')}\n{(r.get('body') or '')[:400]}"
        for i, r in enumerate(results, 1)
    )

TOOLS = [search_knowledge_base, live_web_search]

if __name__ == "__main__":
    print("Name:", search_knowledge_base.name)
    print("Description:", search_knowledge_base.description)
    print("Args:", search_knowledge_base.args)
    for q in ("What is predictive maintenance?", "Who won the football world cup?"):
        out = search_knowledge_base.invoke({"query": q})
        print(f"\nQ: {q}\n{out[:300]}")
    print("\nWeb:", live_web_search.invoke({"query": "latest Python release"})[:600])
