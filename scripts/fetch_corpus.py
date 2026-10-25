import re, sys, time
from pathlib import Path
import requests

OUT = Path("data/docs")
OUT.mkdir(parents=True, exist_ok=True)

TITLES = [
    "Predictive maintenance", "Condition monitoring", "Anomaly detection",
    "Statistical process control", "Overall equipment effectiveness",
    "Industry 4.0", "Digital twin", "Smart manufacturing",
    "Automated optical inspection", "Computer vision", "MLOps",
    "Retrieval-augmented generation", "Large language model",
    "Vector database", "Edge computing", "Machine learning",
]
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "local-agentic-rag/0.1 (portfolio project; contact: birele1270@moblha.com)"}

def slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")

def main() -> None:
    for title in TITLES:
        r = requests.get(API, headers=HEADERS, timeout=30, params={
            "action": "query", "prop": "extracts", "explaintext": 1,
            "titles": title, "redirects": 1, "format": "json"})
        r.raise_for_status()
        page = next(iter(r.json()["query"]["pages"].values()))
        text = page.get("extract", "")
        if not text:
            print(f"SKIP  {title} (no extract)"); continue
        real = page["title"]
        url = "https://en.wikipedia.org/wiki/" + real.replace(" ", "_")
        body = f"# {real}\n\nSource: {url} (CC BY-SA 4.0)\n\n{text}\n"
        (OUT / f"{slug(title)}.md").write_text(body, encoding="utf-8")
        print(f"OK    {slug(title)}.md  ({len(text):,} chars)")
        time.sleep(0.5)


if __name__ == "__main__":
    main()