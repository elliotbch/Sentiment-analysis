"""Tests of the demo API.

Prediction tests need the exported model (``models/sentiment_onnx`` in the repo or
``MODEL_PATH``); they are skipped otherwise. The "no weights" behaviour is always tested.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import app as demo  # noqa: E402

EXAMPLES = json.loads((DEMO / "static/examples/examples.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(demo.app) as c:
        yield c


needs_model = pytest.mark.skipif(
    not (demo.HERE.parent / "models/sentiment_onnx/sentiment_meta.json").exists()
    and not __import__("os").environ.get("MODEL_PATH"),
    reason="exported ONNX model not available")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "model_loaded" in r.json()


def test_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Analyse de sentiment de tweets" in r.text
    assert 'lang="fr"' in r.text
    assert "textes en anglais" in r.text
    assert "github.com/elliotbch/Sentiment-analysis" in r.text


def test_examples(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    assert 3 <= len(r.json()) <= 6
    assert all(e["text"] and e["title"] for e in r.json())


def test_metrics(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert r.json()["source"] == "training/metrics.json"


def test_metrics_copy_in_sync():
    """demo/metrics.json (shipped with the Space) must equal training/metrics.json."""
    src, copy = demo.HERE.parent / "training/metrics.json", demo.HERE / "metrics.json"
    if src.exists() and copy.exists():
        assert json.loads(src.read_text()) == json.loads(copy.read_text())


@needs_model
@pytest.mark.parametrize("ex", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_predict_examples(client, ex):
    r = client.post("/api/predict", json={"text": ex["text"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] in ("negative", "neutral", "positive")
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["inference_ms"] >= 0


@needs_model
def test_obvious_labels(client):
    by_id = {e["id"]: e["text"] for e in EXAMPLES}
    assert client.post("/api/predict", json={"text": by_id["positive"]}).json()["label"] == "positive"
    assert client.post("/api/predict", json={"text": by_id["negative"]}).json()["label"] == "negative"


@needs_model
def test_batch(client):
    texts = [e["text"] for e in EXAMPLES] + ["", "   "]
    r = client.post("/api/predict-batch", json={"texts": texts})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == len(EXAMPLES)
    single = client.post("/api/predict", json={"text": EXAMPLES[0]["text"]}).json()
    assert r.json()["results"][0]["label"] == single["label"]


def test_batch_limit(client):
    r = client.post("/api/predict-batch", json={"texts": ["ok"] * 51})
    assert r.status_code == 422
    assert "50" in r.json()["detail"] and "limite" in r.json()["detail"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "x" * 1001}, {}])
def test_bad_input(client, payload):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_batch_empty(client):
    assert client.post("/api/predict-batch", json={"texts": ["", " "]}).status_code == 422


def test_no_weights_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.onnx"))
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    m = demo.Model()
    m.load()
    assert not m.ready and "MODEL_PATH" in m.error
    monkeypatch.setattr(demo, "MODEL", m)
    with TestClient(demo.app) as c:
        # lifespan reloads the (still missing) model
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/api/predict", json={"text": "hello"})
        assert r.status_code == 503
        assert "MODEL_PATH" in r.json()["detail"]
