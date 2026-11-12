"""Reference predictions from the FastAPI demo, compared by parity.html with the in-browser model.

Usage (from the repository root, with the demo running, e.g. `cd demo && uvicorn app:app --port 7860`):
  python3 site/tests/parity_reference.py http://localhost:7860 [OUT.json] [--csv data/raw/setfit/raw_data/test_csv --n 200]
Default OUT: site/tests/parity_reference.json (all demo examples + edge cases, single and batch requests).
Standard library only.
"""
import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = json.loads((HERE.parent / "assets" / "examples.json").read_text())
EDGE = [
    "@elliot check this out https://t.co/abc123 so good", "  lots of spaces    around   ", "I don't hate it.",
    "Ça a l'air génial 😂😂 #blessed", "<s> weird </s> tokens <mask>", "meh.", "!!!!!!", "ok",
    "worst. day. ever. " * 20,  # > 128 tokens: truncated
    "Never been so happy to see the rain stop, the garden finally gets a break",
]


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("api")
    ap.add_argument("out", nargs="?", default=str(HERE / "parity_reference.json"))
    ap.add_argument("--csv")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    texts = [e["text"] for e in EXAMPLES] + EDGE
    if a.csv:
        with open(a.csv, newline="", encoding="utf-8") as f:
            rows = [r["text"] for r in csv.DictReader(f) if r.get("text", "").strip()]
        texts += random.Random(0).sample(rows, a.n)
    single = []
    for t in texts:
        r = post(a.api, "/api/predict", {"text": t})
        single.append({"text": t, "label": r["label"], "probabilities": r["probabilities"]})
    batches = []
    for i in range(0, len(texts), 50):  # batch mode pads to the longest text: checked separately
        chunk = texts[i:i + 50]
        r = post(a.api, "/api/predict-batch", {"texts": chunk})
        batches.append({"texts": chunk, "results": [{"text": x["text"], "label": x["label"], "probabilities": x["probabilities"]}
                                                    for x in r["results"]]})
    Path(a.out).write_text(json.dumps({"api": "FastAPI demo (demo/app.py)", "single": single, "batches": batches},
                                      ensure_ascii=False, indent=1))
    print(f"{len(single)} textes, {len(batches)} lot(s) -> {a.out}")


if __name__ == "__main__":
    main()
