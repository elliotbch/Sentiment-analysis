// Sentiment model in the browser: same steps as demo/app.py (Model.predict), run with onnxruntime-web (WASM).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs";
import { RobertaTokenizer, normalizeTwitter, pyStrip, pyLen } from "./tokenizer.js";
import { fetchCached } from "./loader.js";

export const ORT_VERSION = "1.31.1";
export const HF_REPO = "https://huggingface.co/elliotb/sentiment-twitter-roberta/resolve/main/";
export const FILES = { model: { name: "model_int8.onnx", bytes: 125435663 }, tokenizer: { name: "tokenizer.json", bytes: 3558798 } };
export const MAX_CHARS = 1000;
export const MAX_BATCH = 50;
const CACHE = "sentiment-twitter-roberta-v1";

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// GitHub Pages does not send COOP/COEP headers, so SharedArrayBuffer (threads) is usually unavailable.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export class SentimentModel {
  constructor(base = "./assets/") { this.base = base; this.session = null; }

  // onProgress(loadedBytes, totalBytes, fromCache)
  async load(onProgress = () => {}) {
    this.meta = await (await fetch(this.base + "sentiment_meta.json")).json();
    const total = FILES.model.bytes + FILES.tokenizer.bytes;
    let tokLoaded = 0, modelLoaded = 0, cached = true;
    const report = () => onProgress(tokLoaded + modelLoaded, total, cached);
    const tokBuf = await fetchCached(HF_REPO + FILES.tokenizer.name, { cacheName: CACHE, expectedBytes: FILES.tokenizer.bytes,
      onProgress: (l, _t, c) => { tokLoaded = l; cached = cached && c; report(); } });
    const modelBuf = await fetchCached(HF_REPO + this.meta.default_model, { cacheName: CACHE, expectedBytes: FILES.model.bytes,
      onProgress: (l, _t, c) => { modelLoaded = l; cached = cached && c; report(); } });
    this.tokenizer = new RobertaTokenizer(JSON.parse(new TextDecoder().decode(tokBuf)),
      { maxLen: Number(this.meta.max_len), padId: Number(this.meta.pad_id) });
    this.session = await ort.InferenceSession.create(new Uint8Array(modelBuf), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    this.modelFile = this.meta.default_model;
    this.fromCache = cached;
  }

  // Raw logits -> softmax, as numpy does in demo/app.py (float32 logits, max subtracted).
  // ONNX Runtime Web cannot run two inferences at once on the same session: queue them.
  predict(texts) {
    const run = (this._queue || Promise.resolve()).then(() => this._predict(texts));
    this._queue = run.catch(() => {});
    return run;
  }

  async _predict(texts) {
    const norm = this.meta.normalize_twitter ? texts.map(normalizeTwitter) : texts;
    const enc = this.tokenizer.encodeBatch(norm);
    const feeds = { input_ids: new ort.Tensor("int64", enc.ids, enc.dims), attention_mask: new ort.Tensor("int64", enc.mask, enc.dims) };
    const t0 = performance.now();
    const out = await this.session.run(feeds, ["logits"]);
    const ms = performance.now() - t0;
    const logits = out.logits.data, k = this.meta.labels.length;
    const probs = texts.map((_, r) => {
      const z = Array.from(logits.slice(r * k, r * k + k));
      const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    });
    return { probs, logits: texts.map((_, r) => Array.from(logits.slice(r * k, r * k + k))), ms };
  }

  result(text, p) {
    const labels = this.meta.labels;
    let best = 0;
    p.forEach((v, i) => { if (v > p[best]) best = i; });
    const probabilities = {};
    labels.forEach((l, i) => { probabilities[l] = Math.round(p[i] * 1e4) / 1e4; });
    return { text, label: labels[best], probabilities };
  }

  // Same validation and messages as /api/predict.
  async predictOne(text) {
    text = clean(text);
    const { probs, logits, ms } = await this.predict([text]);
    return { ...this.result(text, probs[0]), raw: probs[0], logits: logits[0], inference_ms: ms, model_file: this.modelFile };
  }

  // Same validation and messages as /api/predict-batch.
  async predictBatch(lines) {
    let texts = lines.map((t) => (t ? pyStrip(t) : "")).filter(Boolean);
    if (!texts.length) throw new Error("Aucune ligne non vide à analyser.");
    if (texts.length > MAX_BATCH) throw new Error(`Trop de lignes (${texts.length}) : la limite est de ${MAX_BATCH}.`);
    texts = texts.map(clean);
    const { probs, logits, ms } = await this.predict(texts);
    return { results: texts.map((t, i) => ({ ...this.result(t, probs[i]), raw: probs[i], logits: logits[i] })), count: texts.length,
      inference_ms: ms, model_file: this.modelFile };
  }
}

function clean(text) {
  text = pyStrip(text);
  if (!text) throw new Error("Le texte est vide.");
  const n = pyLen(text);
  if (n > MAX_CHARS) throw new Error(`Texte trop long (${n} caractères, ${MAX_CHARS} au maximum).`);
  return text;
}
