import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.config import settings

DOCS_DIR = Path("data/docs")
CHUNK_TOKENS, OVERLAP = 480, 64

@lru_cache(maxsize=1)
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(settings.embed_model)
    tok.model_max_length = 10**6   # we chunk ourselves; silences the 512-length warning
    return tok

def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))

def split_document(doc_id: str, text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=OVERLAP,
        length_function=count_tokens,
    )
    chunks = splitter.split_documents(
        [Document(page_content=text, metadata={"doc_id": doc_id})]
    )
    for i, c in enumerate(chunks):
        c.metadata["chunk_idx"] = i
        c.metadata["chunk_id"] = f"{doc_id}::{i}"
    return chunks

def token_table() -> None:
    total = 0
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        n = count_tokens(text)
        total += n
        print(f"{path.stem:<36}{len(text):>9,} chars{n:>9,} tokens")
    print(f"\nTotal: {total:,} tokens")

def preview(doc_id: str) -> None:
    text = (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8")
    chunks = split_document(doc_id, text)
    sizes = [count_tokens(c.page_content) for c in chunks]
    print(f"{doc_id}: {len(chunks)} chunks | tokens min={min(sizes)} "
          f"max={max(sizes)} avg={sum(sizes) / len(sizes):.0f}")
    for c, n in list(zip(chunks, sizes))[:2]:
        flat = c.page_content.replace("\n", " ")
        print(f"\n--- {c.metadata['chunk_id']} ({n} tokens) ---")
        print("START:", flat[:150])
        print("END:  ", flat[-150:])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preview(sys.argv[1])
    else:
        token_table()