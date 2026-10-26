from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.retrieval import retrieve

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

if __name__ == "__main__":
    print("Name:", search_knowledge_base.name)
    print("Description:", search_knowledge_base.description)
    print("Args:", search_knowledge_base.args)
    for q in ("What is predictive maintenance?", "Who won the football world cup?"):
        out = search_knowledge_base.invoke({"query": q})
        print(f"\nQ: {q}\n{out[:300]}")
