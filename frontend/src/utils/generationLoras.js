/** Optional generation LoRAs (Idea by @waltm — Discord feature request).
 *
 * Two opt-in slots chained onto the LOCAL Klein edit graph after the
 * consistency LoRA:
 *   - ultra_real    — skin/texture realism, usable on SFW and NSFW runs alike;
 *   - nsfw_anatomy  — STRICTLY gated behind the workspace's 🔞 NSFW toggle.
 * The LoRA files themselves are pointed by the user in Settings (config
 * klein.ultra_real_lora / klein.nsfw_lora) — never hardcoded; here we only
 * decide which per-generation strengths ride along with a Generate request.
 */

export const LORA_STRENGTH_MAX = 1.5;

/** Clamp a slider/user value into the [0, 1.5] strength range the backend
 *  enforces too (NaN and negatives collapse to 0 = off). */
export function clampLoraStrength(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.min(LORA_STRENGTH_MAX, n);
}

/** Body fragment for /generate (and /regenerate): the optional-LoRA strengths
 *  of THIS run. A slot contributes its key only when every gate passes —
 *  otherwise the key is absent entirely (absent = slot off server-side):
 *   - Klein engine only (API engines never see these knobs);
 *   - the slot's toggle is on AND its strength is > 0;
 *   - nsfw_anatomy additionally requires the run's NSFW toggle (a configured
 *     strength alone must never inject it — fail-closed, mirrored server-side).
 */
export function optionalLoraPayload({
  isKlein = false, nsfwMode = false,
  ultraRealOn = false, ultraRealStrength = 0,
  nsfwLoraOn = false, nsfwLoraStrength = 0,
} = {}) {
  if (!isKlein) return {};
  const out = {};
  const ultra = clampLoraStrength(ultraRealStrength);
  if (ultraRealOn && ultra > 0) out.ultra_real_strength = ultra;
  const anatomy = clampLoraStrength(nsfwLoraStrength);
  if (nsfwMode && nsfwLoraOn && anatomy > 0) out.nsfw_lora_strength = anatomy;
  return out;
}
