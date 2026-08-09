// Pure helpers for the ◉ Canvas 🔌 external LoRA plugin nodes: any
// models/loras file pinned on the board and stacked on top of a run.
// Shape everywhere: {filename, strength} — the studio engine's extra_loras
// shape (NOT the Klein preset {file, strength}).
export const MAX_EXTERNAL_LORAS = 16;

export function clampStrength(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 1;
  return Math.max(0, Math.min(2, Math.round(n * 100) / 100));
}

export function normalizeExternalLoras(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  const seen = new Set();
  for (const e of raw) {
    const filename = typeof e?.filename === 'string' ? e.filename.trim() : '';
    if (!filename || seen.has(filename)) continue;
    seen.add(filename);
    const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
    out.push({ filename, strength: clampStrength(e?.strength ?? 1),
      x: num(e?.x), y: num(e?.y) });
    if (out.length >= MAX_EXTERNAL_LORAS) break;
  }
  return out;
}

export function externalLoraPayload(nodes, checked) {
  return (nodes || [])
    .filter((n) => checked && checked.has(n.filename))
    .map((n) => ({ filename: n.filename, strength: n.strength }));
}
