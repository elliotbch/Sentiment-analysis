"""Tweet sentiment demo: FastAPI + ONNX Runtime (CPU).

Model files (a directory with ``sentiment_meta.json``, ``tokenizer.json`` and the
ONNX file named in the meta) are looked up in this order:

1. ``MODEL_PATH``: an ``.onnx`` file (meta + tokenizer next to it) or the directory;
2. ``../models/sentiment_onnx`` when the app runs from the repository;
3. ``HF_MODEL_REPO`` (e.g. ``elliotb/sentiment-twitter-roberta``) on the Hugging Face Hub.

Without weights the app still starts, ``/health`` says so and the prediction
endpoints answer 503. No prediction is ever simulated.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
MAX_CHARS = 1000
MAX_BATCH = 50

log = logging.getLogger("sentiment-demo")
logging.basicConfig(level=logging.INFO)


def normalize_twitter(text: str) -> str:
    """Same preprocessing as training (cardiffnlp model card): @mentions and URLs masked."""
    out = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)


class Model:
    def __init__(self) -> None:
        self.session = None
        self.tokenizer = None
        self.meta: dict = {}
        self.source = None
        self.error = None

    @property
    def ready(self) -> bool:
        return self.session is not None

    def _resolve_dir(self) -> tuple[Path, str | None] | None:
        env = os.environ.get("MODEL_PATH")
        if env:
            p = Path(env)
            if p.is_file():
                return p.parent, p.name
            if p.is_dir():
                return p, None
            raise FileNotFoundError(f"MODEL_PATH={env} n'existe pas")
        local = HERE.parent / "models" / "sentiment_onnx"
        if (local / "sentiment_meta.json").exists():
            return local, None
        repo = os.environ.get("HF_MODEL_REPO")
        if repo:
            from huggingface_hub import hf_hub_download
            meta_path = Path(hf_hub_download(repo, "sentiment_meta.json"))
            meta = json.loads(meta_path.read_text())
            hf_hub_download(repo, "tokenizer.json")
            hf_hub_download(repo, meta["default_model"])
            return meta_path.parent, None
        return None

    def load(self) -> None:
        try:
            found = self._resolve_dir()
            if found is None:
                self.error = ("Aucun poids de modèle trouvé. Définissez MODEL_PATH (fichier ONNX local) "
                              "ou HF_MODEL_REPO (dépôt du Hub Hugging Face).")
                log.warning(self.error)
                return
            folder, onnx_name = found
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self.meta = json.loads((folder / "sentiment_meta.json").read_text())
            onnx_file = folder / (onnx_name or self.meta["default_model"])
            tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
            tok.enable_truncation(max_length=int(self.meta["max_len"]))
            tok.enable_padding(pad_id=int(self.meta["pad_id"]))
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
            self.session = ort.InferenceSession(str(onnx_file), so, providers=["CPUExecutionProvider"])
            self.tokenizer = tok
            self.source = onnx_file.name
            self.error = None
            log.info("model loaded from %s", onnx_file)
        except Exception as exc:  # the app must start anyway
            self.session = None
            self.error = f"Le modèle n'a pas pu être chargé : {exc}"
            log.exception("model load failed")

    def predict(self, texts: list[str]) -> tuple[np.ndarray, float]:
        if self.meta.get("normalize_twitter"):
            texts = [normalize_twitter(t) for t in texts]
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        t0 = time.perf_counter()
        logits = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        ms = (time.perf_counter() - t0) * 1000
        z = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return probs, ms


MODEL = Model()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    MODEL.load()
    yield


app = FastAPI(title="Démo d'analyse de sentiment de tweets", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(RequestValidationError)
async def _invalid(_request, exc: RequestValidationError) -> JSONResponse:
    fields = sorted({".".join(str(x) for x in e.get("loc", [])[1:]) or "corps" for e in exc.errors()})
    return JSONResponse(status_code=422, content={
        "detail": "Requête invalide : JSON attendu "
                  '({"text": "..."} ou {"texts": ["...", "..."]}), champ(s) en cause : ' + ", ".join(fields) + "."})


class TextIn(BaseModel):
    text: str = Field(..., description="Un tweet ou un texte court en anglais")


class BatchIn(BaseModel):
    texts: list[str] = Field(..., description=f"Jusqu'à {MAX_BATCH} textes")


def _clean(text: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(422, "Le texte est vide.")
    if len(text) > MAX_CHARS:
        raise HTTPException(422, f"Texte trop long ({len(text)} caractères, {MAX_CHARS} au maximum).")
    return text


def _require_model() -> None:
    if not MODEL.ready:
        raise HTTPException(503, MODEL.error or "Modèle non chargé.")


def _result(text: str, p: np.ndarray) -> dict:
    labels = MODEL.meta["labels"]
    return {"text": text, "label": labels[int(p.argmax())],
            "probabilities": {lab: round(float(v), 4) for lab, v in zip(labels, p)}}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL.ready, "model_file": MODEL.source,
            "base_model": MODEL.meta.get("base_model"), "detail": MODEL.error}


@app.get("/api/examples")
def examples() -> list[dict]:
    return json.loads((STATIC / "examples" / "examples.json").read_text())


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    """Metrics of the trained models, read from training/metrics.json (single source)."""
    for p in (os.environ.get("METRICS_PATH"), HERE.parent / "training" / "metrics.json",
              HERE / "metrics.json"):
        if p and Path(p).exists():
            data = json.loads(Path(p).read_text())
            return JSONResponse({"source": "training/metrics.json", **data})
    raise HTTPException(404, "metrics.json introuvable.")


@app.post("/api/predict")
def predict(body: TextIn) -> dict:
    text = _clean(body.text)
    _require_model()
    probs, ms = MODEL.predict([text])
    return {**_result(text, probs[0]), "inference_ms": round(ms, 1), "model_file": MODEL.source}


@app.post("/api/predict-batch")
def predict_batch(body: BatchIn) -> dict:
    texts = [t.strip() for t in body.texts if t and t.strip()]
    if not texts:
        raise HTTPException(422, "Aucune ligne non vide à analyser.")
    if len(texts) > MAX_BATCH:
        raise HTTPException(422, f"Trop de lignes ({len(texts)}) : la limite est de {MAX_BATCH}.")
    texts = [_clean(t) for t in texts]
    _require_model()
    probs, ms = MODEL.predict(texts)
    return {"results": [_result(t, p) for t, p in zip(texts, probs)], "count": len(texts),
            "inference_ms": round(ms, 1), "model_file": MODEL.source}
