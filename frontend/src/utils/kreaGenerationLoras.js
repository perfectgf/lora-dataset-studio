/** Always-on generation-LoRA PRESETS for the Krea 2 Identity Edit dataset lane.
 *
 * Deliberately a sibling of generationLoras.js rather than a generalization of
 * it: same shapes, different clamp, and the backend keeps the two lanes as
 * separate copies too (krea_edit_helper vs klein_edit_helper, mirroring
 * inject_krea_loras vs inject_sdxl_loras). Mechanism after @waltm's idea.
 *
 * The user defines named combinations in Settings; per run the workspace PICKS
 * one and the request carries only its NAME — the backend resolves files,
 * strengths and order from config (fail-closed).
 */

/** Server clamp — krea_edit_helper.LORA_STRENGTH_MAX. Not a UX range. */
export const KREA_LORA_STRENGTH_MAX = 20;

/** Slider ceiling for an ordinary LoRA. The utility ones (filter-bypass) have no
 *  effect below ~10, so their slider opens to the full server clamp instead. */
export const KREA_SLIDER_MAX = 6;

/** krea_edit_helper.DEFAULT_ROW_STRENGTH. */
export const KREA_LORA_STRENGTH_DEFAULT = 1.0;

/** Mirror krea_edit_helper.MAX_GENERATION_LORAS / MAX_GENERATION_LORA_PRESETS. */
export const MAX_GENERATION_LORAS = 8;
export const MAX_GENERATION_LORA_PRESETS = 12;

/** Filenames whose slider opens to the full 0..20. Every separator spelling is
 *  matched on purpose: the Studio's original /filterbypass/i handed
 *  `filter_bypass.safetensors` a 0..6 slider, i.e. a LoRA that looks broken
 *  because it does nothing in the only range the UI could reach. */
export const BYPASS_NAME = /filter[-_]?bypass/i;

/** The slider range for one row, from its filename. */
export function kreaStrengthRange(filename) {
  return BYPASS_NAME.test(String(filename ?? ''))
    ? { min: 0, max: KREA_LORA_STRENGTH_MAX }
    : { min: 0, max: KREA_SLIDER_MAX };
}

/** Clamp into the [0, 20] range the backend enforces (NaN/negative -> 0). */
export function clampKreaLoraStrength(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.min(KREA_LORA_STRENGTH_MAX, n);
}

/** Sanitize a config-shaped preset list: drop blank/duplicate names and
 *  blank/malformed rows, normalize strengths (junk -> the default), cap rows and
 *  presets. Order is preserved everywhere — row order IS the chain order. */
export function sanitizeKreaGenerationLoraPresets(list) {
  const out = [];
  const seen = new Set();
  for (const preset of Array.isArray(list) ? list : []) {
    if (!preset || typeof preset !== 'object') continue;
    const name = typeof preset.name === 'string' ? preset.name.trim() : '';
    if (!name || seen.has(name)) continue;
    const rows = [];
    for (const row of Array.isArray(preset.loras) ? preset.loras : []) {
      if (!row || typeof row !== 'object') continue;
      const file = typeof row.file === 'string' ? row.file.trim() : '';
      if (!file) continue;
      const n = Number(row.strength);
      rows.push({
        file,
        strength: Number.isFinite(n)
          ? Math.min(KREA_LORA_STRENGTH_MAX, Math.max(0, n))
          : KREA_LORA_STRENGTH_DEFAULT,
      });
      if (rows.length >= MAX_GENERATION_LORAS) break;
    }
    seen.add(name);
    out.push({ name, loras: rows });
    if (out.length >= MAX_GENERATION_LORA_PRESETS) break;
  }
  return out;
}

/** Body fragment for /generate: the picked preset's NAME under Krea's OWN key —
 *  one run can dispatch to Klein and Krea at once, so the two keys must not
 *  collide. Empty fragment ({}) when Krea isn't in the run, nothing is picked, or
 *  the pick would chain nothing (unknown or empty preset — never send a dead
 *  name). */
export function kreaGenerationLoraPresetPayload({ isKrea = false, presetName = '', presets = [] } = {}) {
  if (!isKrea) return {};
  const name = typeof presetName === 'string' ? presetName.trim() : '';
  if (!name) return {};
  const preset = sanitizeKreaGenerationLoraPresets(presets).find((p) => p.name === name);
  if (!preset || preset.loras.length === 0) return {};
  return { krea_generation_lora_preset: name };
}
