/* One Krea sampler menu, two destinations.

   The Krea graph can be sampled two ways, and they are mutually exclusive:

     • a NAME from ComfyUI's built-in list, written onto the KSampler's
       `sampler_name` (the `sampler` field);
     • a PRESET of the sampler this app ships, which replaces the KSampler with a
       custom-sampling chain (the `sampler_preset` field).

   They are one CHOICE — you sample a render one way — so they are one dropdown.
   They are two FIELDS because they land in different places in the graph, and a
   preset name written into `sampler_name` would be a sampler ComfyUI has never
   heard of: the whole graph is refused at validation, and the user sees a tile
   that failed for a reason nothing on screen explains.

   This module is the only place that converts between the two shapes. PURE JS,
   no JSX, so `node --test` can exercise it — the same split as kreaDials.js.

   The wire value for a preset carries a prefix (`preset:balanced`) rather than
   the bare name. Without it the two namespaces would sit in one <select> with
   nothing telling them apart, and the day a ComfyUI release ships a sampler
   called `balanced` the menu would silently mean two things. */

export const KREA_PRESET_PREFIX = 'preset:';

/* Fallback list, used only until /api/index_config answers. It mirrors the
   server's KREA_SAMPLER_PRESETS, which mirrors the shipped node's PRESETS — a
   backend contract test pins those two together, so the only drift this literal
   can suffer is against a server too old to send the list, where offering the
   presets it does know is exactly right. */
export const KREA_SAMPLER_PRESETS_FALLBACK = Object.freeze([
  'neutral', 'soft', 'balanced', 'detailed', 'max',
]);

/** Is this menu value one of our presets rather than a ComfyUI sampler name? */
export const isPresetChoice = (value) =>
  typeof value === 'string' && value.startsWith(KREA_PRESET_PREFIX);

/** The menu value for a preset name. */
export const presetChoice = (preset) => `${KREA_PRESET_PREFIX}${preset}`;

/** The preset name a menu value carries, or '' when it names a stock sampler. */
export const presetOf = (value) =>
  isPresetChoice(value) ? value.slice(KREA_PRESET_PREFIX.length) : '';

/** One menu value -> the two fields the run payload carries.
 *
 *  Exactly one of them is ever non-empty. That is not a detail: sending both
 *  would ask the server to sample the same render two ways, and it resolves that
 *  by silently preferring one — which is how a user ends up unable to get back
 *  to the sampler they picked. '' (Auto) sends neither, leaving the workflow's
 *  own tuned defaults alone. */
export function splitSamplerChoice(value) {
  if (isPresetChoice(value)) {
    return { sampler: '', sampler_preset: presetOf(value) };
  }
  return { sampler: value || '', sampler_preset: '' };
}

/** The two fields -> one menu value, for restoring a saved or resumed run.
 *
 *  The preset wins when both are set. A run persisted before this existed can
 *  only have `sampler`, so that case never arises from our own data; it arises
 *  from a hand-edited payload, and the preset is the field that actually changed
 *  the graph. */
export function joinSamplerChoice({ sampler, sampler_preset: preset } = {}) {
  if (preset) return presetChoice(preset);
  return sampler || '';
}
