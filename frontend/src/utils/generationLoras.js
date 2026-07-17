/** Optional generation LoRAs (Idea by @waltm — Discord feature request).
 *
 * An ORDERED, user-defined list of extra LoRAs chained onto the LOCAL Klein
 * edit graph after the consistency LoRA (list order = chain order). Each
 * config row is {file, strength, nsfw_only}: the file is pointed by the user
 * in Settings (never hardcoded), the strength is only the per-run slider
 * default, and nsfw_only rows are STRICTLY gated behind the workspace's 🔞
 * NSFW toggle. Per run, every row starts OFF; here we build the request
 * fragment for the rows the user armed.
 */

export const LORA_STRENGTH_MAX = 1.5;

/** Hard cap on the list length — mirrors the backend's
 *  klein_edit_helper.MAX_GENERATION_LORAS (shown in the Settings card). */
export const MAX_GENERATION_LORAS = 8;

/** Clamp a slider/user value into the [0, 1.5] strength range the backend
 *  enforces too (NaN and negatives collapse to 0 = off). */
export function clampLoraStrength(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.min(LORA_STRENGTH_MAX, n);
}

/** Sanitize a config-shaped list (from /api/settings or a Settings edit):
 *  drop blank/malformed rows, normalize fields, cap at MAX_GENERATION_LORAS.
 *  Order is preserved — it IS the chain order. */
export function sanitizeGenerationLoras(list) {
  const out = [];
  for (const row of Array.isArray(list) ? list : []) {
    if (!row || typeof row !== 'object') continue;
    const file = typeof row.file === 'string' ? row.file.trim() : '';
    if (!file) continue;
    const n = Number(row.strength);
    out.push({
      file,
      strength: Number.isFinite(n) ? Math.min(LORA_STRENGTH_MAX, Math.max(0, n)) : 0.6,
      nsfw_only: !!row.nsfw_only,
    });
    if (out.length >= MAX_GENERATION_LORAS) break;
  }
  return out;
}

/** Body fragment for /generate (and /regenerate): the armed optional-LoRA rows
 *  of THIS run, as { generation_loras: [{file, strength}] }. A row rides only
 *  when every gate passes — otherwise it is absent (absent = off server-side,
 *  which re-checks order, files and the 🔞 flag against the config):
 *   - Klein engine only (API engines never see these knobs);
 *   - the row's toggle is on AND its strength is > 0;
 *   - nsfw_only rows additionally require the run's NSFW toggle (fail-closed).
 *  `rows` = [{file, strength, nsfw_only, on}]. Empty result -> {} (no key). */
export function optionalLoraPayload({ isKlein = false, nsfwMode = false, rows = [] } = {}) {
  if (!isKlein) return {};
  const armed = [];
  for (const row of Array.isArray(rows) ? rows : []) {
    if (!row || !row.on) continue;
    const file = typeof row.file === 'string' ? row.file.trim() : '';
    const strength = clampLoraStrength(row.strength);
    if (!file || strength <= 0) continue;
    if (row.nsfw_only && !nsfwMode) continue;
    armed.push({ file, strength });
    if (armed.length >= MAX_GENERATION_LORAS) break;
  }
  return armed.length ? { generation_loras: armed } : {};
}
