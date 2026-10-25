import sys
sys.path.insert(0, ".")
from src.llm_router import get_llm

llm = get_llm(max_tokens=16)
print(llm.invoke("Reply with the single word: ready").content)