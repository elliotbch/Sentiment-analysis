"""Chargement des données et découpage reproductible.

Deux sources possibles :

* ``public`` (par défaut) : corpus public Tweet Sentiment Extraction (Kaggle 2020,
  données Figure Eight « Data for Everyone »), copie du Hub Hugging Face
  ``SetFit/tweet_sentiment_extraction`` (fichiers ``raw_data/train_csv`` et
  ``raw_data/test_csv``, colonnes ``textID, text, selected_text, sentiment``).
  - test : le jeu de test officiel étiqueté (3 534 tweets), tenu à l'écart ;
  - train/val : le jeu d'entraînement officiel, privé des tweets dont le texte
    apparaît aussi dans le test, découpé 90 / 10 (stratifié, graine fixée).
* ``challenge`` : fichiers officiels du challenge
  (``eurecom-aml-2021-challenge-3/train.csv``), découpés exactement comme dans le
  notebook (95 / 2,5 / 2,5 avec ``train_test_split(random_state=42)``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

HF_DATASET = "SetFit/tweet_sentiment_extraction"
# Révision du dépôt Hub utilisée pour les résultats publiés (training/metrics.json).
HF_REVISION = "0d5359cfbb2470332f2b82e62269cf755e0ac5c3"
RAW_FILES = ("raw_data/train_csv", "raw_data/test_csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_public(data_dir: Path) -> Path:
    """Télécharge les deux CSV publics dans ``data_dir/raw/setfit`` (idempotent)."""
    from huggingface_hub import hf_hub_download

    target = data_dir / "raw" / "setfit"
    for name in RAW_FILES:
        if not (target / name).exists():
            hf_hub_download(HF_DATASET, name, repo_type="dataset",
                            revision=HF_REVISION, local_dir=target)
    return target / "raw_data"


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def load_public(data_dir: Path, seed: int = 42, val_size: float = 0.10):
    raw = download_public(data_dir)
    train_full = pd.read_csv(raw / "train_csv").dropna(subset=["text"])
    test = pd.read_csv(raw / "test_csv").dropna(subset=["text"])

    # Aucun texte du test ne doit être vu à l'entraînement ni utilisé en validation.
    test_texts = set(_norm(test["text"]))
    overlap = _norm(train_full["text"]).isin(test_texts)
    pool = train_full[~overlap].reset_index(drop=True)

    train, val = train_test_split(pool, test_size=val_size, random_state=seed,
                                  stratify=pool["sentiment"])
    info = {
        "source": f"huggingface.co/datasets/{HF_DATASET}@{HF_REVISION}",
        "origin": "Kaggle Tweet Sentiment Extraction (2020), Figure Eight 'Data for Everyone'",
        "raw_rows": {"train_csv": int(len(train_full)), "test_csv": int(len(test))},
        "removed_train_rows_overlapping_test": int(overlap.sum()),
        "split": "test = official labelled test set; train/val = stratified 90/10 split "
                 f"of the official train set (seed {seed})",
        "sha256": {n: sha256(raw / Path(n).name) for n in RAW_FILES},
    }
    return _finish(train, val, test, info)


def load_challenge(csv_path: Path, seed: int = 42):
    """Split identique au notebook (95 % / 2,5 % val / 2,5 % test)."""
    # Pas de dropna : le notebook garde toutes les lignes (str(NaN)), ce qui
    # conditionne l'indexation du split.
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)
    train, rest = train_test_split(df, test_size=0.05, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    info = {"source": str(csv_path), "split": "notebook split 95/2.5/2.5 (random_state=42)",
            "sha256": {Path(csv_path).name: sha256(Path(csv_path))}}
    return _finish(train, val, test, info)


def _finish(train, val, test, info):
    out = []
    for df in (train, val, test):
        df = df[["text", "sentiment"]].copy()
        df["label"] = df["sentiment"].map(LABEL2ID)
        out.append(df.reset_index(drop=True))
    info["sizes"] = {k: int(len(d)) for k, d in zip(("train", "val", "test"), out)}
    info["class_counts"] = {k: d["sentiment"].value_counts().reindex(LABELS).astype(int).to_dict()
                            for k, d in zip(("train", "val", "test"), out)}
    return out[0], out[1], out[2], info
