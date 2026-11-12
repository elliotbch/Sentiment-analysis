"""Reference token ids from the Python side (demo/app.py + Hugging Face `tokenizers`).

Writes a JSON file read by tokenizer_parity.mjs, which checks the browser tokenizer (site/assets/tokenizer.js)
against it, step by step: str.strip, normalize_twitter, ids after truncation to max_len.

Usage (from the repository root):
  uv run --with-requirements demo/requirements.txt python site/tests/tokenizer_reference.py OUT.json [CSV ...]
CSV files are optional (e.g. data/raw/setfit/raw_data/test_csv, column "text"); without them only the
built-in examples and edge cases are used.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from app import normalize_twitter  # noqa: E402  (the function served by the FastAPI demo)
from tokenizers import Tokenizer  # noqa: E402

MODEL_DIR = ROOT / "models" / "sentiment_onnx"
meta = json.loads((MODEL_DIR / "sentiment_meta.json").read_text())
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
tok.enable_truncation(max_length=int(meta["max_len"]))

EDGE = [
    "  @elliot check https://t.co/xyz  ", "@ alone", "@", "http", "httpx://foo", "email me@x.com now",
    "I'm sure they'll say we've done it, you'd think so", "Ça a l'air génial, déjà vu… naïve café",
    "日本語のツイート、テスト。", "😂😂 so funny 🤣👍🏽 #blessed", "multiple    spaces\tand\ttabs\nnew line",
    "nbsp here and thin space", "nel\u0085char and bom﻿char", "zero​width joiner‍",
    "text with <s> and </s> special <pad> tokens <unk>", "a   <mask> b", "x<mask>y", "<s><s></s>",
    "numbers 12345 and 3.14 and ١٢٣", "Mixed123abc and abc123", "'s 't 're 've 'm 'll 'd 'S 'LL",
    "trailing spaces   ", "   ", "\x1c\x1dctrl\x1e", "emoji at end 🙂", "a" * 3000, "hello " * 300,
    "Oh great, my laptop decided to update right before the presentation. Love that for me.",
]


def row(text: str) -> dict:
    stripped = text.strip()
    norm = normalize_twitter(stripped)
    return {"text": text, "stripped": stripped, "len": len(stripped), "normalized": norm, "ids": tok.encode(norm).ids}


texts = list(EDGE)
examples = json.loads((ROOT / "demo" / "static" / "examples" / "examples.json").read_text())
texts += [e["text"] for e in examples]
for path in sys.argv[2:]:
    with open(path, newline="", encoding="utf-8") as f:
        texts += [r["text"] for r in csv.DictReader(f) if r.get("text")]
Path(sys.argv[1]).write_text(json.dumps({"max_len": meta["max_len"], "rows": [row(t) for t in texts]}, ensure_ascii=False))
print(f"{len(texts)} textes de référence écrits dans {sys.argv[1]}")
