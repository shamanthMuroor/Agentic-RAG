from pathlib import Path
from src.chunking import split_document
from src.store import get_vectorstore

DOCS_DIR = Path("data/docs")
DOC_IDS = sorted(p.stem for p in DOCS_DIR.glob("*.md"))

def ingest(doc_ids: list[str]) -> None:
    store = get_vectorstore()
    for doc_id in doc_ids:
        text = (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
        chunks = split_document(doc_id, text)
        ids = [c.metadata["chunk_id"] for c in chunks]
        store.delete(where={"doc_id": doc_id})
        store.add_documents(chunks, ids=ids)
        print(f"{doc_id}: {len(chunks)} chunks added")
    print("Total chunks in index:", len(store.get()["ids"]))

def ensure_index() -> bool:
    """Build the index if it is missing or empty. Returns True if it had to build."""
    if not DOC_IDS:
        raise RuntimeError("data/docs is empty, so there is nothing to index")
    if get_vectorstore().get(limit=1)["ids"]:
        return False
    ingest(DOC_IDS)
    return True

if __name__ == "__main__":
    ingest(DOC_IDS)