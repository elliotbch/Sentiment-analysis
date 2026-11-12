// Byte-level BPE tokenizer (RoBERTa) reproducing Hugging Face `tokenizers` for this tokenizer.json:
// added tokens split first, ByteLevel pre-tokenizer (use_regex, no prefix space), BPE merges,
// RobertaProcessing (<s> … </s>), truncation to max_len (right) and padding to the longest (right).
// Also the text helpers of demo/app.py (normalize_twitter, str.strip, len) so that the browser
// sees exactly the same strings as the FastAPI demo. Pure ES module: runs in the browser and in Node.

// Python's str.isspace() set, used by str.strip().
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const STRIP_RE = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "g");
export const pyStrip = (s) => s.replace(STRIP_RE, "");
export const pyLen = (s) => { let n = 0; for (const _ of s) n++; return n; };

// demo/app.py normalize_twitter(): split on single spaces, mask @mentions and URLs.
export function normalizeTwitter(text) {
  return text.split(" ").map((t) => {
    if (t.startsWith("@") && pyLen(t) > 1) t = "@user";
    if (t.startsWith("http")) t = "http";
    return t;
  }).join(" ");
}

// Oniguruma's Unicode \s (used by tokenizers' ByteLevel regex); differs from JS \s on U+0085 / U+FEFF.
const WS = "\\t\\n\\v\\f\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const SPLIT_RE = new RegExp(
  `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^${WS}\\p{L}\\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`, "gu");
// Rust char::is_whitespace (White_Space property), used by the lstrip flag of added tokens.
const RUST_WS = new RegExp(`[${WS}\\x1c-\\x1f]`, "u");

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const map = new Array(256);
  bs.forEach((b, i) => { map[b] = String.fromCodePoint(cs[i]); });
  return map;
}

export class RobertaTokenizer {
  constructor(tokJson, { maxLen = 128, padId = 1 } = {}) {
    const m = tokJson.model;
    if (m.type !== "BPE") throw new Error("tokenizer.json : modèle BPE attendu");
    const pt = tokJson.pre_tokenizer || {};
    if (pt.type !== "ByteLevel" || pt.add_prefix_space || pt.use_regex === false) throw new Error("tokenizer.json : pré-tokeniseur ByteLevel inattendu");
    if (tokJson.normalizer) throw new Error("tokenizer.json : normaliseur inattendu");
    this.vocab = new Map(Object.entries(m.vocab));
    this.ranks = new Map();
    m.merges.forEach((mg, i) => {
      const [a, b] = Array.isArray(mg) ? mg : mg.split(" ");
      this.ranks.set(a + " " + b, i);
    });
    this.unkId = m.unk_token != null ? this.vocab.get(m.unk_token) : undefined;
    this.byteMap = bytesToUnicode();
    this.encoder = new TextEncoder();
    this.cache = new Map();
    const pp = tokJson.post_processor;
    this.clsId = pp.cls[1]; this.sepId = pp.sep[1];
    this.maxLen = maxLen; this.padId = padId;
    this.added = (tokJson.added_tokens || []).map((t) => ({ content: t.content, id: t.id, lstrip: t.lstrip, rstrip: t.rstrip }))
      .sort((a, b) => b.content.length - a.content.length);
  }

  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      const a = parts[at], b = parts[at + 1], merged = [];
      for (let i = 0; i < parts.length;) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i += 2; }
        else { merged.push(parts[i]); i += 1; }
      }
      parts = merged;
    }
    const ids = [];
    for (const p of parts) {
      const id = this.vocab.get(p);
      if (id !== undefined) ids.push(id);
      else if (this.unkId !== undefined) ids.push(this.unkId);
    }
    if (this.cache.size < 50000) this.cache.set(word, ids);
    return ids;
  }

  encodePlain(text, out) {
    for (const m of text.matchAll(SPLIT_RE)) {
      let w = "";
      for (const byte of this.encoder.encode(m[0])) w += this.byteMap[byte];
      for (const id of this.bpe(w)) out.push(id);
    }
  }

  // Ids of the text without special tokens (added tokens such as "<s>" are matched in the raw text first).
  tokenize(text) {
    const out = [];
    let start = 0, i = 0;
    while (i < text.length) {
      const tok = this.added.find((t) => text.startsWith(t.content, i));
      if (!tok) { i++; continue; }
      let s = i, e = i + tok.content.length;
      if (tok.lstrip) while (s > start && RUST_WS.test(text[s - 1])) s--;
      if (tok.rstrip) while (e < text.length && RUST_WS.test(text[e])) e++;
      if (s > start) this.encodePlain(text.slice(start, s), out);
      out.push(tok.id);
      start = i = e;
    }
    if (start < text.length) this.encodePlain(text.slice(start), out);
    return out;
  }

  encode(text) {
    const ids = this.tokenize(text).slice(0, this.maxLen - 2);
    return [this.clsId, ...ids, this.sepId];
  }

  // Same as tokenizers' encode_batch with truncation(max_len) + padding(BatchLongest, right).
  encodeBatch(texts) {
    const enc = texts.map((t) => this.encode(t));
    const L = Math.max(...enc.map((e) => e.length));
    const ids = new BigInt64Array(enc.length * L), mask = new BigInt64Array(enc.length * L);
    enc.forEach((e, r) => {
      for (let c = 0; c < L; c++) {
        ids[r * L + c] = BigInt(c < e.length ? e[c] : this.padId);
        mask[r * L + c] = c < e.length ? 1n : 0n;
      }
    });
    return { ids, mask, dims: [enc.length, L], lists: enc };
  }
}
