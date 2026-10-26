from functools import lru_cache
from src.config import settings   # must be imported first: hides CUDA before torch loads
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

COLLECTION = "knowledge_base"

@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embedder(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

if __name__ == "__main__":
    import math
    import time
    from pathlib import Path
    import torch
    from src.chunking import split_document

    text = Path("data/docs/predictive_maintenance.md").read_text(encoding="utf-8")
    chunk = split_document("predictive_maintenance", text)[0]
    embedder = get_embedder()

    for label in ("1st call", "2nd call"):
        t = time.time()
        vec = embedder.embed_documents([chunk.page_content])[0]
        print(f"{label}: {1000 * (time.time() - t):.0f} ms")

    print("CUDA visible:", torch.cuda.is_available())
    print("Dimension:", len(vec))
    print("Vector length:", round(math.sqrt(sum(v * v for v in vec)), 4))
    print("First 5 values:", [round(v, 4) for v in vec[:5]])