"""Export ONNX (FP32 puis INT8 dynamique) du meilleur modèle et mesure de la perte.

Entrée : un dossier ``save_pretrained`` produit par ``train.py --save-dir``.
Sortie (``models/sentiment_onnx/``, ignoré par git) :
  model_fp32.onnx, model_int8.onnx, tokenizer.json, sentiment_meta.json

Les deux fichiers ONNX sont évalués sur le MÊME test que train.py (onnxruntime, CPU).
Règle fixée avant la mesure : l'INT8 est retenu pour la démo si sa perte reste
≤ 0,5 point d'accuracy et ≤ 0,005 de F1 macro par rapport au FP32 ONNX.
Résultats fusionnés dans ``training/metrics.json`` sous ``export``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABELS, load_public  # noqa: E402
from train import _merge, normalize_twitter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_DROP_ACC, MAX_DROP_F1 = 0.005, 0.005


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def evaluate(path, tok, texts, y, batch=64):
    import onnxruntime as ort
    from sklearn.metrics import accuracy_score, f1_score
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    preds = []
    for s in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[s:s + batch])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        preds.append(sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0].argmax(1))
    pred = np.concatenate(preds)
    # Latence d'un tweet seul (comme dans la démo), médiane sur 200 tweets.
    lat = []
    for t in texts[:200]:
        e = tok.encode(t)
        t0 = time.perf_counter()
        sess.run(["logits"], {"input_ids": np.array([e.ids], dtype=np.int64),
                              "attention_mask": np.array([e.attention_mask], dtype=np.int64)})
        lat.append((time.perf_counter() - t0) * 1000)
    return pred, {"accuracy": round(float(accuracy_score(y, pred)), 4),
                  "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
                  "latency_ms_single_tweet_median": round(float(np.median(lat)), 1),
                  "size_mb": round(path.stat().st_size / 2**20, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+",
                    default=["twitter_roberta_ft", "twitter_roberta_ft_lr2e-5_truncated"],
                    help="runs candidats (clés de metrics.json, poids dans models/<run>) ; "
                         "on exporte celui qui a le meilleur F1 macro de VALIDATION")
    ap.add_argument("--out", type=Path, default=ROOT / "models/sentiment_onnx")
    ap.add_argument("--metrics", type=Path, default=ROOT / "training/metrics.json")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tokenizers import Tokenizer
    from onnxruntime.quantization import QuantType, quantize_dynamic

    runs = json.loads(args.metrics.read_text())["runs"]
    val_f1 = {r: runs[r]["val"]["f1_macro"] for r in args.candidates if r in runs}
    args.run_name = max(val_f1, key=val_f1.get)
    checkpoint = ROOT / "models" / args.run_name
    print("validation macro-F1:", val_f1, "-> export", args.run_name)
    run = runs[args.run_name]
    hp = run["hyperparameters"]
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
    assert [model.config.id2label[i] for i in range(3)] == LABELS, model.config.id2label
    hf_tok = AutoTokenizer.from_pretrained(checkpoint)

    _, _, test, _ = load_public(ROOT / "data", seed=run["seed"])
    prep = normalize_twitter if hp["normalize_twitter"] else str
    texts = [prep(t) for t in test["text"]]
    y = test["label"].to_numpy()

    # Référence PyTorch (CPU) calculée AVANT l'export : le traçage modifie l'état du modèle.
    with torch.no_grad():
        enc = hf_tok(texts[:16], padding=True, truncation=True, max_length=hp["max_len"], return_tensors="pt")
        ref = model(**enc).logits.numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = args.out / "model_fp32.onnx", args.out / "model_int8.onnx"
    dummy = hf_tok(["a short tweet", "another one, a bit longer than the first"],
                   padding=True, return_tensors="pt")
    torch.onnx.export(
        Wrapper(model), (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=args.opset, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    # Tokenizer rapide (tokenizers seul, pas besoin de transformers dans la démo).
    hf_tok.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    tok = Tokenizer.from_file(str(args.out / "tokenizer.json"))
    tok.enable_truncation(max_length=hp["max_len"])
    tok.enable_padding(pad_id=hf_tok.pad_token_id, pad_token=hf_tok.pad_token)

    import onnxruntime as ort
    out = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(
        ["logits"], {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    max_abs = float(np.abs(out - ref).max())
    print("max |onnx - torch| logits:", max_abs)

    p32, m32 = evaluate(fp32, tok, texts, y)
    p8, m8 = evaluate(int8, tok, texts, y)
    m8["agreement_with_fp32"] = round(float((p8 == p32).mean()), 4)
    drop_acc, drop_f1 = m32["accuracy"] - m8["accuracy"], m32["f1_macro"] - m8["f1_macro"]
    keep_int8 = drop_acc <= MAX_DROP_ACC and drop_f1 <= MAX_DROP_F1
    default = int8.name if keep_int8 else fp32.name
    print("fp32", m32, "\nint8", m8, "\n->", default)

    meta = {"labels": LABELS, "max_len": hp["max_len"], "normalize_twitter": hp["normalize_twitter"],
            "pad_id": hf_tok.pad_token_id, "default_model": default,
            "base_model": run["model"], "run_name": args.run_name}
    (args.out / "sentiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not keep_int8:
        int8.unlink()  # on ne garde pas un fichier qu'on ne sert pas

    _merge(args.metrics, ["export"], {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "from_run": args.run_name,
        "selection": {"criterion": "validation macro-F1", "candidates_val_f1_macro": val_f1}, "opset": args.opset, "runtime": f"onnxruntime {ort.__version__} CPU",
        "max_abs_logit_diff_onnx_vs_torch": round(max_abs, 6),
        "test_n": int(len(y)), "fp32": m32, "int8_dynamic": m8,
        "rule": f"INT8 kept if accuracy drop <= {MAX_DROP_ACC} and macro-F1 drop <= {MAX_DROP_F1}",
        "drop_int8_vs_fp32": {"accuracy": round(drop_acc, 4), "f1_macro": round(drop_f1, 4)},
        "served_model": default,
        "latency_note": "median single-tweet latency, batch 1, measured on the training Mac (M2 Pro) CPU",
    })


if __name__ == "__main__":
    main()
