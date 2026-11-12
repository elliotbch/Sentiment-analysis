// Download helpers: progress in MB, Cache API storage for later visits, WebAssembly check.

// WebAssembly with SIMD is required by onnxruntime-web (from 1.19 only SIMD builds are shipped).
export function wasmSupported() {
  try {
    if (typeof WebAssembly !== "object") return false;
    const simd = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
    return WebAssembly.validate(simd);
  } catch (_) { return false; }
}

async function openCache(name) {
  try { return "caches" in self ? await caches.open(name) : null; } catch (_) { return null; }
}

// Returns the file as an ArrayBuffer; onProgress(loadedBytes, totalBytes, fromCache).
export async function fetchCached(url, { cacheName, expectedBytes = 0, onProgress = () => {} } = {}) {
  const cache = await openCache(cacheName);
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        const buf = await hit.arrayBuffer();
        if (!expectedBytes || buf.byteLength === expectedBytes) { onProgress(buf.byteLength, buf.byteLength, true); return buf; }
        await cache.delete(url);
      }
    } catch (_) { /* cache unreadable: download again */ }
  }
  const res = await fetch(url, { mode: "cors" });
  if (!res.ok) throw new Error(`Téléchargement impossible (${res.status}) : ${url}`);
  const total = Number(res.headers.get("Content-Length")) || expectedBytes;
  let buf;
  if (res.body && res.body.getReader) {
    // Written straight into one buffer of the announced size (no second copy of a 125 MB model).
    const reader = res.body.getReader();
    let out = new Uint8Array(total || 1 << 20), loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > out.length) {
        const bigger = new Uint8Array(Math.max(out.length * 2, loaded + value.byteLength));
        bigger.set(out.subarray(0, loaded)); out = bigger;
      }
      out.set(value, loaded); loaded += value.byteLength;
      onProgress(loaded, Math.max(total, loaded), false);
    }
    buf = loaded === out.length ? out.buffer : out.slice(0, loaded).buffer;
  } else {
    buf = await res.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength, false);
  }
  if (expectedBytes && buf.byteLength !== expectedBytes) throw new Error(`Fichier incomplet (${buf.byteLength} octets reçus sur ${expectedBytes}) : rechargez la page.`);
  if (cache) {
    try { await cache.put(url, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } })); } catch (_) { /* quota: no cache */ }
  }
  return buf;
}

export const mb = (bytes) => (bytes / 1048576).toFixed(1).replace(".", ",");
