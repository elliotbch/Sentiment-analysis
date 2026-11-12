// Page logic: same behaviour as demo/static/index.html, with the model run locally instead of /api/*.
import { wasmSupported, mb } from "./loader.js";

const $ = (id) => document.getElementById(id);
const LABELS = ["negative", "neutral", "positive"];
const FR = { negative: "négatif", neutral: "neutre", positive: "positif" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x * 100).toFixed(1).replace(".", ",") + " %";
const dec = (x, n) => x.toFixed(n).replace(".", ",");
// Inference runs on the main thread: let the browser paint the "in progress" message first.
const paint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
let examples = [];
let model = null;
let ready = null; // Promise resolved when the model is usable

// ---- model loading
function setProgress(loaded, total, fromCache) {
  const p = total ? Math.min(100, (100 * loaded) / total) : 0;
  $("load-fill").style.width = p.toFixed(1) + "%";
  $("load-bar").setAttribute("aria-valuenow", p.toFixed(0));
  $("load-text").textContent = fromCache && loaded === total
    ? `Modèle lu depuis le cache du navigateur (${mb(total)} Mo)…`
    : `Téléchargement du modèle : ${mb(loaded)} / ${mb(total)} Mo (une seule fois, gardé ensuite en cache)`;
}

function startLoading() {
  if (ready) return ready;
  $("load-btn").hidden = true;
  $("load-bar").hidden = false;
  const t0 = performance.now();
  ready = import("./model.js").then(async (m) => {
    model = new m.SentimentModel();
    await model.load(setProgress);
    $("load-text").textContent = "Initialisation du modèle…";
    await model.predict(["warm up"]); // first run compiles the WASM kernels
    const s = ((performance.now() - t0) / 1000).toFixed(1).replace(".", ",");
    window.__loadSeconds = (performance.now() - t0) / 1000;
    $("loading").hidden = true;
    $("status").textContent = `Modèle prêt en ${s} s${model.fromCache ? " (depuis le cache du navigateur)" : ""}.`;
    return model;
  }).catch((e) => {
    ready = null;
    $("load-text").innerHTML = `<span class="error">Le modèle n’a pas pu être chargé : ${esc(e.message || e)}</span>`;
    $("load-btn").hidden = false; $("load-btn").textContent = "Réessayer";
    throw e;
  });
  return ready;
}

async function getModel() {
  if (!ready) startLoading();
  return ready;
}

// ---- tabs
const tabs = [$("tab-single"), $("tab-batch")];
function select(tab) {
  tabs.forEach((t) => {
    const on = t === tab;
    t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
  });
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => select(t));
  t.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = tabs[(i + 1) % 2]; select(n); n.focus(); }
  });
});

// ---- single
function renderSingle(r) {
  const bars = LABELS.map((l) => `
    <div class="bar"><span>${FR[l]}</span>
      <div class="track" role="img" aria-label="${FR[l]} ${pct(r.probabilities[l])}"><div class="fill f-${l}" style="width:${(r.probabilities[l] * 100).toFixed(1)}%"></div></div>
      <span class="num">${pct(r.probabilities[l])}</span></div>`).join("");
  $("single-result").innerHTML = `
    <div class="verdict"><span class="badge b-${r.label}">${FR[r.label]}</span>
      <span class="hint">confiance ${pct(r.probabilities[r.label])} · inférence ${dec(r.inference_ms, 1)} ms (${esc(r.model_file)})</span></div>
    <div class="bars">${bars}</div>`;
}
let singleSeq = 0;
async function analyse() {
  const text = $("text").value;
  const btn = $("analyse");
  const seq = ++singleSeq; // only the latest request is displayed
  btn.disabled = true;
  $("single-result").innerHTML = `<p class="hint">${model ? "Analyse en cours…" : "Analyse dès que le modèle est chargé…"}</p>`;
  try { const m = await getModel(); await paint(); const r = await m.predictOne(text); if (seq === singleSeq) renderSingle(r); }
  catch (e) { if (seq === singleSeq) $("single-result").innerHTML = `<p class="error">${esc(e.message || e)}</p>`; }
  finally { if (seq === singleSeq) btn.disabled = false; }
}
$("analyse").addEventListener("click", analyse);
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse(); });

// ---- batch
const lines = () => $("batch").value.split("\n").map((s) => s.trim()).filter(Boolean);
const updateCount = () => { const n = lines().length; $("line-count").textContent = `${n} ligne${n > 1 ? "s" : ""}${n > 50 ? " (limite : 50)" : ""}`; };
$("batch").addEventListener("input", updateCount);
$("fill-batch").addEventListener("click", () => { $("batch").value = examples.map((e) => e.text).join("\n"); updateCount(); });
$("batch-file").addEventListener("change", async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if (!f) return;
  if (f.size > 1024 * 1024) { $("batch-result").innerHTML = '<p class="error">Fichier trop volumineux (1 Mo maximum).</p>'; return; }
  $("batch").value = (await f.text()).replace(/\r\n?/g, "\n");
  updateCount();
});
$("analyse-batch").addEventListener("click", async () => {
  const btn = $("analyse-batch");
  btn.disabled = true;
  const n = lines().length;
  const wait = `Analyse en cours : le lot est calculé en une fois comme dans l’API, comptez environ ${Math.max(1, Math.round(n * 0.3))} s dans le navigateur (la page reste figée pendant le calcul)…`;
  $("batch-result").innerHTML = `<p class="hint">${model ? wait : "Analyse dès que le modèle est chargé…"}</p>`;
  try {
    const m = await getModel();
    $("batch-result").innerHTML = `<p class="hint">${wait}</p>`;
    await paint();
    const r = await m.predictBatch($("batch").value.split("\n"));
    const rows = r.results.map((x) => `<tr><td>${esc(x.text)}</td><td><span class="badge small b-${x.label}">${FR[x.label]}</span> ${pct(x.probabilities[x.label])}
      <span class="probs">${LABELS.map((l) => `${FR[l]} ${pct(x.probabilities[l])}`).join(" · ")}</span></td></tr>`).join("");
    $("batch-result").innerHTML = `<p class="hint">${r.count} texte${r.count > 1 ? "s" : ""} · inférence ${dec(r.inference_ms, 1)} ms pour tout le lot</p>
      <div class="table-wrap"><table><thead><tr><th scope="col">Texte</th><th scope="col">Sentiment</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  } catch (e) { $("batch-result").innerHTML = `<p class="error">${esc(e.message || e)}</p>`; }
  finally { btn.disabled = false; }
});

// ---- init
fetch("assets/examples.json").then((r) => r.json()).then((ex) => {
  examples = ex;
  $("chips").innerHTML = ex.map((e, i) => `<button class="chip" data-i="${i}" title="${esc(e.text)}">${esc(e.title)}</button>`).join("");
  $("chips").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { $("text").value = examples[+b.dataset.i].text; analyse(); }));
}).catch(() => { $("chips").textContent = "Exemples indisponibles."; });

fetch("assets/metrics.json").then((r) => r.json()).then((m) => {
  const runs = m.runs || {};
  const served = m.export && m.export.from_run;
  const display = [
    { run: served, name: "Twitter RoBERTa fine-tuné (PyTorch)" },
    { run: "bert_base_notebook", name: "BERT base cased, recette du cours, réentraîné" },
    { run: "twitter_roberta_zero_shot", name: "Twitter RoBERTa tel quel (sans fine-tuning)" },
  ];
  const row = (name, t, strong) => `<tr><td>${strong ? "<strong>" + esc(name) + "</strong>" : esc(name)}</td><td class="num">${pct(t.accuracy)}</td><td class="num">${dec(t.f1_macro, 3)}</td></tr>`;
  const rows = [];
  if (m.export && m.export.served_model) {
    const int8 = m.export.served_model.includes("int8");
    rows.push(row(`Twitter RoBERTa fine-tuné, ONNX ${int8 ? "INT8" : "FP32"} (utilisé par cette démo)`, int8 ? m.export.int8_dynamic : m.export.fp32, true));
  }
  display.filter((d) => d.run && runs[d.run]).forEach((d) => rows.push(row(d.name, runs[d.run].test, false)));
  document.querySelector("#metrics-table tbody").innerHTML = rows.join("") || '<tr><td colspan="3">Aucun modèle entraîné pour l’instant.</td></tr>';
  const n = served && runs[served] ? runs[served].test.n : null;
  $("metrics-source").textContent = `Données publiques Tweet Sentiment Extraction, jeu de test officiel étiqueté${n ? ` (${n.toLocaleString("fr-FR")} tweets)` : ""}, jamais utilisé pour l’entraînement ni le choix du modèle ; toutes les lignes sont mesurées sur ce même test.`;
}).catch(() => { document.querySelector("#metrics-table tbody").innerHTML = '<tr><td colspan="3">Métriques indisponibles.</td></tr>'; });

if (!wasmSupported()) {
  $("load-bar").hidden = true;
  $("load-text").innerHTML = '<span class="error">Votre navigateur ne prend pas en charge WebAssembly (avec SIMD), nécessaire pour exécuter le modèle. Essayez une version récente de Chrome, Edge, Firefox ou Safari.</span>';
  ["analyse", "analyse-batch"].forEach((id) => { $(id).disabled = true; });
} else if (navigator.connection && navigator.connection.saveData) {
  // Data saver on: wait for an explicit click before downloading 123 MB.
  $("load-bar").hidden = true;
  $("load-text").textContent = "Le modèle pèse 123 Mo : il est téléchargé une seule fois, puis gardé en cache.";
  $("load-btn").hidden = false;
} else {
  startLoading();
}
$("load-btn").addEventListener("click", () => startLoading());
