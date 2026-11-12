"""Fine-tuning reproductible d'un classifieur de sentiment de tweets (3 classes).

Deux recettes :

* ``--head notebook`` : recette du notebook du challenge (BertModel, dropout 0,3 et
  couche linéaire sur la sortie poolée, AdamW sans correction de biais, lr 2e-5,
  sans warm-up, 4 époques, batch 16, max_len 120, texte brut) ;
* ``--head auto`` : ``AutoModelForSequenceClassification`` (ex.
  ``cardiffnlp/twitter-roberta-base-sentiment-latest``), AdamW avec warm-up.

Le meilleur epoch est choisi sur la VALIDATION (F1 macro) ; le test n'est évalué
qu'une fois, à la fin, avec ce checkpoint. Les métriques sont fusionnées dans
``training/metrics.json`` sous ``runs[<run-name>]``.

``--epochs 0`` évalue le modèle tel quel (zéro-shot, sans entraînement).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LABELS, load_challenge, load_public  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_twitter(text: str) -> str:
    """Prétraitement recommandé par la fiche des modèles cardiffnlp."""
    out = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)


class AdamWNoBiasCorrection(torch.optim.Optimizer):
    """Équivalent de ``transformers.AdamW(lr, correct_bias=False)`` (retiré de
    transformers), utilisé par le notebook : eps 1e-6, pas de weight decay."""

    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-group["lr"])
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])


class NotebookClassifier(nn.Module):
    """Tête du notebook : BertModel -> pooled output -> Dropout(0.3) -> Linear."""

    def __init__(self, name: str, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.out(self.drop(pooled))


class AutoClassifier(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if set(id2label.values()) == set(LABELS):
            # On réordonne si la tête pré-entraînée n'a pas l'ordre neg/neu/pos.
            self.perm = [next(k for k, v in id2label.items() if v == lab) for lab in LABELS]
            self.pretrained_head = True
        else:
            self.perm = [0, 1, 2]
            self.pretrained_head = False
        self.model.config.id2label = {i: lab for i, lab in enumerate(LABELS)}
        self.model.config.label2id = {lab: i for i, lab in enumerate(LABELS)}

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits[:, self.perm]


# --------------------------------------------------------------------------- data
def encode(tokenizer, texts, max_len):
    enc = tokenizer(list(texts), truncation=True, max_length=max_len)
    return enc["input_ids"]


PAD_MULTIPLE = 32  # longueur paddée arrondie : peu de formes distinctes, mémoire MPS stable


def mem_info(device) -> str:
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # octets sur macOS
    if device == "mps":
        return (f"mps alloc {torch.mps.current_allocated_memory() / 2**30:.2f} GB, "
                f"driver {torch.mps.driver_allocated_memory() / 2**30:.2f} GB, max RSS {rss:.2f} GB")
    return f"max RSS {rss:.2f} GB"


def batches(ids, labels, batch_size, pad_id, shuffle, rng=None):
    order = np.arange(len(ids))
    if shuffle:
        rng.shuffle(order)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(ids[i]) for i in idx)
        L = -(-L // PAD_MULTIPLE) * PAD_MULTIPLE
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, :len(ids[i])] = torch.tensor(ids[i])
            m[j, :len(ids[i])] = 1
        yield x, m, torch.tensor(labels[idx], dtype=torch.long)


@torch.no_grad()
def predict(model, ids, labels, bs, pad_id, device):
    model.eval()
    logits = []
    for x, m, _ in batches(ids, labels, bs, pad_id, shuffle=False):
        logits.append(model(x.to(device), m.to(device)).float().cpu())
    return torch.cat(logits)


def scores(y, logits):
    pred = logits.argmax(1).numpy()
    loss = nn.functional.cross_entropy(logits, torch.tensor(y)).item()
    return {"loss": round(loss, 4), "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro")), 4)}, pred


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--head", choices=["notebook", "auto"], default="auto")
    ap.add_argument("--data", choices=["public", "challenge"], default="public")
    ap.add_argument("--challenge-csv", type=Path, default=ROOT / "data/challenge/train.csv")
    ap.add_argument("--normalize-twitter", action="store_true",
                    help="@mentions -> @user, URLs -> http (fiche cardiffnlp)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--optimizer", choices=["adamw", "notebook"], default="adamw",
                    help="notebook = AdamW sans correction de biais, eps 1e-6, sans weight decay")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0,
                    help=">0 : mesure de vitesse, s'arrête après N pas, n'écrit que speed_tests")
    ap.add_argument("--time-limit-min", type=float, default=44.0,
                    help="arrêt de sécurité (budget 45 min par run)")
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="sauvegarde du meilleur modèle (format Hugging Face) pour l'export")
    ap.add_argument("--metrics", type=Path, default=ROOT / "training/metrics.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = (("mps" if torch.backends.mps.is_available() else
               "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    log(f"device={device} torch={torch.__version__}")

    if args.data == "public":
        train, val, test, data_info = load_public(ROOT / "data", seed=args.seed)
    else:
        train, val, test, data_info = load_challenge(args.challenge_csv, seed=args.seed)
    log(f"split sizes {data_info['sizes']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prep = normalize_twitter if args.normalize_twitter else (lambda s: str(s))
    enc = {k: encode(tok, d["text"].map(prep), args.max_len) for k, d in
           (("train", train), ("val", val), ("test", test))}
    lab = {k: d["label"].to_numpy() for k, d in (("train", train), ("val", val), ("test", test))}
    pad_id = tok.pad_token_id

    model = (NotebookClassifier(args.model) if args.head == "notebook" else AutoClassifier(args.model))
    model.to(device)

    steps_per_epoch = math.ceil(len(enc["train"]) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    history, best_state, best_f1, best_epoch, stopped = [], None, -1.0, 0, None
    train_time = 0.0

    if args.epochs > 0:
        if args.optimizer == "notebook":
            opt = AdamWNoBiasCorrection(model.parameters(), lr=args.lr)
        else:
            no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
            groups = [
                {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
                 "weight_decay": args.weight_decay},
                {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
                 "weight_decay": 0.0},
            ]
            opt = torch.optim.AdamW(groups, lr=args.lr, eps=1e-8)
        from transformers import get_linear_schedule_with_warmup
        warm = int(args.warmup_ratio * total_steps) if args.optimizer == "adamw" else 0
        sched = get_linear_schedule_with_warmup(opt, warm, total_steps)
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.default_rng(args.seed)
        step = 0
        t_train = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, correct, seen = [], 0, 0
            te = time.time()
            for x, m, y in batches(enc["train"], lab["train"], args.batch_size, pad_id, True, rng):
                x, m, y = x.to(device), m.to(device), y.to(device)
                out = model(x, m)
                loss = loss_fn(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
                correct += (out.argmax(1) == y).sum().item(); seen += len(y)
                step += 1
                if step % 50 == 0:
                    rate = step / (time.time() - t_train)
                    log(f"epoch {epoch} step {step}/{total_steps} loss {np.mean(losses[-50:]):.4f} "
                        f"{rate:.2f} it/s, ETA train {(total_steps - step) / rate / 60:.1f} min | {mem_info(device)}")
                if args.max_steps and step >= args.max_steps:
                    stopped = "max_steps"; break
                if (time.time() - t0) / 60 > args.time_limit_min:
                    stopped = "time_limit"; break
            if args.max_steps:
                break
            v, _ = scores(lab["val"], predict(model, enc["val"], lab["val"], args.eval_batch_size, pad_id, device))
            rec = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                   "train_accuracy": round(correct / max(seen, 1), 4),
                   **{f"val_{k}": val_ for k, val_ in v.items()},
                   "epoch_minutes": round((time.time() - te) / 60, 2)}
            history.append(rec)
            if device == "mps":
                torch.mps.empty_cache()
            log(f"epoch {epoch} done: {rec}")
            if v["f1_macro"] > best_f1:
                best_f1, best_epoch = v["f1_macro"], epoch
                best_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
            if stopped:
                break
        train_time = time.time() - t_train

    if args.max_steps:
        sec_per_step = train_time / max(step, 1)
        res = {"model": args.model, "batch_size": args.batch_size, "max_len": args.max_len,
               "steps": step, "sec_per_step": round(sec_per_step, 3),
               "estimated_min_per_epoch": round(sec_per_step * steps_per_epoch / 60, 1),
               "device": device, "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log(f"speed test: {res}")
        _merge(args.metrics, ["speed_tests", args.run_name], res)
        return

    if best_state is not None:
        model.load_state_dict(best_state)
    v, _ = scores(lab["val"], predict(model, enc["val"], lab["val"], args.eval_batch_size, pad_id, device))
    t_eval = time.time()
    logits = predict(model, enc["test"], lab["test"], args.eval_batch_size, pad_id, device)
    t_eval = time.time() - t_eval
    t, pred = scores(lab["test"], logits)
    rep = classification_report(lab["test"], pred, target_names=LABELS, output_dict=True, digits=4)
    cm = confusion_matrix(lab["test"], pred, labels=[0, 1, 2]).tolist()
    log(f"VAL {v} | TEST {t}")
    log("\n" + classification_report(lab["test"], pred, target_names=LABELS, digits=4))

    result = {
        "model": args.model, "head": args.head, "data": args.data,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "device": device,
        "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "torch": torch.__version__,
        "hyperparameters": {k: getattr(args, k) for k in (
            "max_len", "batch_size", "lr", "epochs", "warmup_ratio", "weight_decay", "optimizer",
            "normalize_twitter")},
        "selection": "best epoch on validation macro-F1",
        "best_epoch": best_epoch, "stopped_early": stopped, "history": history,
        "train_minutes": round(train_time / 60, 1),
        "total_minutes": round((time.time() - t0) / 60, 1),
        "val": v,
        "test": {**t, "n": int(len(lab["test"])),
                 "per_class": {c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                               for c in LABELS},
                 "confusion_matrix": {"labels": LABELS, "rows_true_cols_pred": cm},
                 "inference_seconds_total": round(t_eval, 2)},
        "data_info": data_info,
    }
    _merge(args.metrics, ["runs", args.run_name], result)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        if args.head == "auto":
            model.model.save_pretrained(args.save_dir)
        else:
            from safetensors.torch import save_file
            save_file({k: v_.contiguous() for k, v_ in model.state_dict().items()},
                      args.save_dir / "notebook_classifier.safetensors")
        tok.save_pretrained(args.save_dir)
        (args.save_dir / "train_args.json").write_text(json.dumps(
            {k: str(v_) for k, v_ in vars(args).items()}, indent=2))
        log(f"saved to {args.save_dir}")


def _merge(path: Path, keys, value):
    data = json.loads(path.read_text()) if path.exists() else {}
    d = data
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
