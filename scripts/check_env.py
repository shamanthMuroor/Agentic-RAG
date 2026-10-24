import sys, time
sys.path.insert(0, ".")
from src.config import settings
import torch

print("Python:", sys.version.split()[0])
print("CUDA visible to torch:", torch.cuda.is_available(), "(expect False)")

from sentence_transformers import SentenceTransformer
t = time.time()
m = SentenceTransformer(settings.embed_model, device="cpu")
v = m.encode("vram budget test")
print(f"Embed OK dim={len(v)} (expect 384) first-call={1000*(time.time()-t):.0f}ms")

from openai import OpenAI
c = OpenAI(base_url=settings.local_base_url, api_key="lm-studio")
t = time.time()
r = c.chat.completions.create(
    model=settings.local_model,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    max_tokens=8, temperature=0,
)
print("LLM:", r.choices[0].message.content.strip(), f"({1000*(time.time()-t):.0f}ms)")