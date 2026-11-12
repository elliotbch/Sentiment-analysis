// Checks site/assets/tokenizer.js against the Python reference written by tokenizer_reference.py.
// Usage: node site/tests/tokenizer_parity.mjs REF.json [path/to/tokenizer.json]
import { readFileSync } from "node:fs";
import { RobertaTokenizer, normalizeTwitter, pyStrip, pyLen } from "../assets/tokenizer.js";

const ref = JSON.parse(readFileSync(process.argv[2], "utf8"));
const tokPath = process.argv[3] || new URL("../../models/sentiment_onnx/tokenizer.json", import.meta.url);
const tok = new RobertaTokenizer(JSON.parse(readFileSync(tokPath, "utf8")), { maxLen: ref.max_len });

const fails = { strip: 0, len: 0, normalize: 0, ids: 0 };
const examples = [];
for (const r of ref.rows) {
  const s = pyStrip(r.text), n = normalizeTwitter(r.stripped), ids = tok.encode(r.normalized);
  const bad = [];
  if (s !== r.stripped) { fails.strip++; bad.push("strip"); }
  if (pyLen(r.stripped) !== r.len) { fails.len++; bad.push("len"); }
  if (n !== r.normalized) { fails.normalize++; bad.push("normalize"); }
  if (ids.length !== r.ids.length || ids.some((v, i) => v !== r.ids[i])) { fails.ids++; bad.push("ids"); }
  if (bad.length && examples.length < 10) examples.push({ text: r.text.slice(0, 80), bad, js: ids.slice(0, 20), py: r.ids.slice(0, 20) });
}
const total = ref.rows.length;
console.log(`${total} textes comparés`);
for (const [k, v] of Object.entries(fails)) console.log(`  ${k.padEnd(10)} identiques : ${total - v}/${total}`);
if (examples.length) { console.log(JSON.stringify(examples, null, 1)); process.exit(1); }
console.log("OK : tokenisation identique sur tous les textes.");
