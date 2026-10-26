from src.store import get_vectorstore

# Embeds question with the same embedder
# Asks Chroma for the 3 chunks whose vectors are closest
# Prints each with its similarity score
def retrieve(query: str, k: int = 3) -> list[tuple]:
    """Return [(Document, similarity)], best first. similarity = 1 - cosine distance."""
    hits = get_vectorstore().similarity_search_with_score(query, k=k)
    return [(doc, 1 - dist) for doc, dist in hits]

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What is predictive maintenance?"
    for doc, sim in retrieve(query):
        flat = doc.page_content.replace("\n", " ")
        print(f"[{sim:.3f}] {doc.metadata['chunk_id']}")
        print("   ", flat[:200], "...\n")


# python -m src.retrieval
# python -m src.retrieval How do control charts detect process variation?
# python -m src.retrieval Who won the football world cup?