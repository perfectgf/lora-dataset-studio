/* ⚙ "Made with" rows for a GENERATED dataset image, from its generation_meta
   stamp (backend: face_dataset_service._generation_meta_json).

   The dataset's rows carry a JSON dict written at enqueue time by whichever
   lane generated them — variations (Klein / Krea / the API engines),
   ✨ improve, 📷 camera. This module turns that dict into the same label/value
   rows the unified viewer shows for gallery images, because the gap it closes
   is precisely "the dataset knows less about its own generated images than
   the Gallery does". Pure module, no JSX — the cases live in the test.

   Honesty rules, in order:
     · a key the stamp does not carry produces NO row (never a guessed one);
     · known keys get their product wording; UNKNOWN scalar keys still render
       under their raw name — a lane that learns to stamp more must never
       have its extra facts silently swallowed by this formatter;
     · LoRA rows accept both `file` and `filename` (two backend resolvers,
       two historical spellings — the formatter is where they converge). */

const LABELS = [
  ['engine', 'Engine'],
  ['base_model', 'Base model'],
  ['steps', 'Sampling steps'],
  ['seed', 'Seed'],
  ['cfg', 'CFG'],
  ['sampler', 'Sampler'],
  ['reference_strength', 'Reference strength'],
  ['output_megapixels', 'Output size, MP'],
  ['aspect', 'Format'],
];

const ENGINE_NAMES = {
  klein: 'FLUX.2 Klein',
  krea: 'Krea 2 Edit',
  camera: 'Camera angles (Qwen-Image-Edit)',
  seedvr2: 'SeedVR2',
  nanobanana: 'Nano Banana',
  chatgpt: 'ChatGPT',
  openrouter: 'OpenRouter',
};

/** 'z image\\Lola_2000.safetensors' → 'Lola_2000' — the viewer's own display
 *  rule for model files: the folder is plumbing, the extension is noise. */
const modelLabel = (value) => String(value)
  .split(/[\\/]/).pop()
  .replace(/\.(safetensors|sft|gguf)$/i, '');

const loraLabel = (l) => {
  const name = modelLabel(l?.file ?? l?.filename ?? '');
  if (!name) return null;
  const s = l?.strength;
  return (s === undefined || s === null) ? name : `${name} @ ${s}`;
};

export function generationMetaRows(meta) {
  if (!meta || typeof meta !== 'object' || Array.isArray(meta)) return [];
  const rows = [];
  const known = new Set(['loras']);
  for (const [key, label] of LABELS) {
    known.add(key);
    const v = meta[key];
    if (v === undefined || v === null || v === '') continue;
    let value = v;
    if (key === 'engine') value = ENGINE_NAMES[v] || String(v);
    if (key === 'base_model') value = modelLabel(v);
    rows.push({ key, label, value: String(value) });
  }
  const loras = Array.isArray(meta.loras) ? meta.loras.map(loraLabel).filter(Boolean) : [];
  if (loras.length) rows.push({ key: 'loras', label: 'Always-on LoRAs', value: loras.join(', ') });
  for (const [k, v] of Object.entries(meta)) {
    if (known.has(k)) continue;
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      rows.push({ key: k, label: k, value: String(v) });
    }
  }
  return rows;
}
