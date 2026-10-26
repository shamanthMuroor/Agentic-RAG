from pathlib import Path
from src.chunking import split_document
from src.store import get_vectorstore

DOCS_DIR = Path("data/docs")
DOC_IDS = ["predictive_maintenance"]

def ingest(doc_ids: list[str]) -> None:
    store = get_vectorstore()
    for doc_id in doc_ids:
        text = (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
        chunks = split_document(doc_id, text)
        ids = [c.metadata["chunk_id"] for c in chunks]
        store.add_documents(chunks, ids=ids)
        print(f"{doc_id}: {len(chunks)} chunks added")
    print("Total chunks in index:", len(store.get()["ids"]))

if __name__ == "__main__":
    ingest(DOC_IDS)