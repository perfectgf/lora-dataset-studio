/* The Studio panel's two per-run "quality" blocks, as payload decisions.

   Hi-res fix (second Krea pass) and finishing (sharpen + film grain) have a
   global default in Settings AND a per-run control on the Studio panel. The
   rule between the two is the one every other knob on that panel follows —
   the setting is a STARTING POINT, not a lock — but this pair has a third
   state the others do not: "I have not touched it here", which must defer to
   whatever Settings says rather than send a value of its own. Sending 1.0 for
   an untouched control would silently override a Settings default the user
   relies on; sending nothing when the user explicitly picked "off" would
   silently re-enable it. So the deferral is a distinct wire shape, decided in
   one place.

   PURE JS, no JSX, so `node --test` can exercise it — the same split as
   kreaSamplerChoice.js. */

/** The latent factors offered per run. x2 is the injector's ceiling
 *  (KREA_HIRES_MAX_SCALE); it is x4 the pixels and roughly x4 the second
 *  pass's time and VRAM, which is why the list stops there. */
export const HIRES_SCALE_CHOICES = Object.freeze([1.25, 1.5, 1.75, 2]);

/** What the panel assumes until /api/index_config answers — the shipped
 *  defaults (off). Offering "off" as the default of a server too old to send
 *  the numbers is exactly right: that server never adds the pass. */
export const HIRES_DEFAULTS_FALLBACK = Object.freeze({ scale: 1, denoise: 0.5, steps: 0 });

/** The values the reference workflow uses — shown as the landing spot, never
 *  shipped as defaults. Mirrors config.py's `improve.*` comments. */
export const FINISH_REFERENCE = Object.freeze({ sharpen: 0.55, grain: 0.01 });

/* null for "no number here". The explicit empties are checked BEFORE Number():
   `Number(null)`, `Number('')` and `Number([])` are all 0, which is a perfectly
   finite value — and a missing rewrite that became 0 would be clamped up to
   0.05 and sent as the run's denoise, a silent minimum nobody asked for. */
const finite = (v) => {
  if (v === null || v === undefined || v === '' || Array.isArray(v)) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/** `defaults` as the panel can trust them: numbers, with the fallback for
 *  anything missing or unparseable (a partial payload, a hand-edited file). */
export function normaliseHiresDefaults(defaults) {
  const d = defaults || {};
  const scale = finite(d.scale);
  const denoise = finite(d.denoise);
  return {
    scale: scale === null ? HIRES_DEFAULTS_FALLBACK.scale : scale,
    denoise: denoise === null ? HIRES_DEFAULTS_FALLBACK.denoise : denoise,
    steps: finite(d.steps) ?? HIRES_DEFAULTS_FALLBACK.steps,
  };
}

export const fmtScale = (v) => `${Number(v)}×`;

/** The label of the "leave it to Settings" option — it SAYS what Settings
 *  holds, so deferring is a choice and not a blind one. */
export function hiresDefaultLabel(defaults) {
  const d = normaliseHiresDefaults(defaults);
  return d.scale > 1
    ? `Settings default (${fmtScale(d.scale)}, rewrite ${d.denoise})`
    : 'Settings default (off)';
}

/** Whether the second pass will run for this cell, given the panel's choice
 *  ('' = deferred) and the Settings default. Drives the denoise slider's
 *  visibility: a dial for a pass that will not run is a lie on the screen. */
export function hiresIsOn(scale, defaults) {
  if (scale === '' || scale === null || scale === undefined) {
    return normaliseHiresDefaults(defaults).scale > 1;
  }
  const n = finite(scale);
  return n !== null && n > 1;
}

/** The wire fields for the hi-res block. Three shapes, deliberately:
 *    deferred, Settings off  -> {}                      (nothing to say)
 *    deferred, Settings on   -> { hires_denoise }       (the setting's scale,
 *                                                        this run's rewrite)
 *    explicit off            -> { hires_scale: 1 }       (wins over Settings)
 *    explicit on             -> { hires_scale, hires_denoise }
 *  An unparseable value is treated as deferred — never as 1.0, which would
 *  silently switch a Settings default off. */
export function hiresPayload({ scale, denoise }, defaults) {
  const d = normaliseHiresDefaults(defaults);
  const den = finite(denoise);
  const denoiseField = den === null ? {} : { hires_denoise: Math.max(0.05, Math.min(1, den)) };
  if (scale === '' || scale === null || scale === undefined) {
    return d.scale > 1 ? denoiseField : {};
  }
  const n = finite(scale);
  if (n === null) return d.scale > 1 ? denoiseField : {};
  if (n <= 1) return { hires_scale: 1 };
  return { hires_scale: Math.min(2, n), ...denoiseField };
}

/** The wire fields for the finishing block: a key per stage that is ON. Off
 *  stages are OMITTED rather than sent as 0 — the backend stores NULL for
 *  "off" and a 0 would be the same thing said a second way. */
export function finishPayload({ sharpen, grain }) {
  const out = {};
  const s = finite(sharpen);
  const g = finite(grain);
  if (s !== null && s > 0) out.finish_sharpen = Math.min(3, s);
  if (g !== null && g > 0) out.finish_grain = Math.min(0.2, g);
  return out;
}
