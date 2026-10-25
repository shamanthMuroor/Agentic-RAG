from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer
from src.config import settings

DOCS_DIR = Path("data/docs")

@lru_cache(maxsize=1)
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(settings.embed_model)
    tok.model_max_length = 10**6   # we chunk ourselves; silences the 512-length warning
    return tok

def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))

def main() -> None:
    total = 0
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        n = count_tokens(text)
        total += n
        print(f"{path.stem:<36}{len(text):>9,} chars{n:>9,} tokens")
    print(f"\nTotal: {total:,} tokens")

if __name__ == "__main__":
    main()