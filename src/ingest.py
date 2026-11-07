from pathlib import Path
from pypdf import PdfReader
from src.chunking import split_document
from src.store import get_vectorstore

DOCS_DIR = Path("data/docs")
DOC_PATHS = sorted((*DOCS_DIR.glob("*.txt"), *DOCS_DIR.glob("*.pdf")))
DOC_IDS = [p.stem for p in DOC_PATHS]

def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")

def ingest(doc_ids: list[str]) -> None:
    store = get_vectorstore()
    existing_ids = store.get()["ids"]
    if existing_ids:
        store.delete(ids=existing_ids)
    for doc_id in doc_ids:
        path = next((p for p in DOC_PATHS if p.stem == doc_id), None)
        if path is None:
            raise FileNotFoundError(f"No profile document found for {doc_id}")
        text = read_document(path)
        chunks = split_document(doc_id, text)
        ids = [c.metadata["chunk_id"] for c in chunks]
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