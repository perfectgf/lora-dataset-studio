/* Editing the ✨ Upscale & improve instruction FROM the screen that runs it.
   PURE JS (JSX-free) so `node --test` can exercise every decision here.

   WHAT IS BEING WRITTEN, AND HOW WIDE IT REACHES
   ----------------------------------------------
   `identity_prompts.klein_improve` is the GLOBAL setting — the very value the
   Settings page edits. There is no per-image and no per-run copy. That is a
   deliberate choice, not a shortcut: a second place to store "the improve
   instruction" would mean two answers to "what will improve ask for?", and the
   whole reason this note exists is that the user could not get ONE answer
   (Qeeyana, Reddit — see kleinImproveHint.js). A wide setting is a fixable
   problem; two truths are not.

   Which is exactly why the panel must SAY it is global. A control that sits
   inside a dataset's lightbox looks per-dataset. `IMPROVE_SCOPE_NOTE` is the
   sentence that stops that misread, and it is rendered unconditionally — not on
   hover, not behind a tooltip.

   STORAGE CONTRACT (backend: face_variations.get_identity_prompt,
   face_dataset_service._improve_prompt):
     · `klein_improve` is a FLAT key — subject-agnostic, unlike its neighbours,
       so nothing here goes through identity_prompts.by_subject.
     · '' means "follow the shipped default", which arrives separately in
       `identity_prompt_defaults.klein_improve`. Storing a copy of the default
       would silently pin the user to today's wording forever — the collapse
       back to '' lives in promptOverride.js and is shared with Settings.
     · `klein_improve_enabled === false` means no instruction at all.

   The SERVER stays the authority: PUT /api/settings deep-merges and returns the
   full payload, so the panel re-reads what was actually stored rather than
   trusting what it sent. */

import { LORA_STRENGTH_MAX, sanitizeGenerationLoraPresets } from '../../utils/generationLoras.js';

/** The raw split of the improve setting, read from a /api/settings payload.
 *  `stored` is what config holds ('' = following the default), `shipped` the
 *  built-in text. Both are needed at once by the editor: the box shows the
 *  effective text while the STORE only ever receives the override.
 *
 *  `loraPreset`/`loraPresets` ride along because they are the SAME panel's
 *  other half: which generation-LoRA preset ✨ improve chains
 *  (klein.improve_lora_preset) — global exactly like the instruction, and read
 *  from the same payload so the two can never quote different config states.
 *  A stored name no longer matching a preset is KEPT (shown as-is): the
 *  backend resolves it fail-closed to "none", and hiding it here would leave
 *  the user unable to see, or clear, the stale pick. */
export function improveEditorState(payload) {
  if (!payload || typeof payload !== 'object') {
    // `megapixels: 2` (the shipped default) rather than undefined: the box is
    // disabled while loaded:false, but a controlled input must still hold a
    // value or React flips it to uncontrolled mid-flight.
    return { loaded: false, stored: '', shipped: '', enabled: true,
      loraPreset: '', loraPresets: [], presets: [], consistencyLora: '',
      megapixels: 2 };
  }
  const ip = (payload.config && payload.config.identity_prompts) || {};
  const defaults = payload.identity_prompt_defaults || {};
  const klein = (payload.config && payload.config.klein) || {};
  const mp = Number(klein.improve_megapixels);
  return {
    loaded: true,
    stored: typeof ip.klein_improve === 'string' ? ip.klein_improve : '',
    shipped: typeof defaults.klein_improve === 'string' ? defaults.klein_improve : '',
    enabled: ip.klein_improve_enabled !== false,
    loraPreset: typeof klein.improve_lora_preset === 'string'
      ? klein.improve_lora_preset : '',
    loraPresets: sanitizeGenerationLoraPresets(klein.generation_lora_presets)
      .map((p) => p.name),
    // The preset DEFINITIONS, kept exactly as stored rather than sanitized —
    // the panel writes the array BACK with one strength changed, so a row
    // Settings left half-built (blank file) has to survive a slider drag here
    // instead of being deleted by a save it never asked for.
    presets: Array.isArray(klein.generation_lora_presets)
      ? klein.generation_lora_presets.filter((p) => p && typeof p === 'object')
      : [],
    // The LoRA the engine already loads outside the presets: a preset row
    // naming it is DROPPED by the server, and a slider that moves nothing is
    // exactly the silence the Settings card added its own warning for.
    consistencyLora: typeof klein.consistency_lora === 'string'
      ? klein.consistency_lora : '',
    // The pass's output budget (klein.improve_megapixels) — the third improve
    // knob this panel answers for. The server merges shipped defaults into the
    // payload, so a finite number is always there; junk degrades to the
    // shipped 2, never to an empty box.
    megapixels: Number.isFinite(mp) ? mp : 2,
  };
}

/** The Settings card's own bounds for the output budget, mirrored here so the
 *  two editors of klein.improve_megapixels can never offer different ranges
 *  (backend clamp: face_dataset_service._improve_float ceiling 8). */
export const IMPROVE_MEGAPIXELS_MIN = 0.5;
export const IMPROVE_MEGAPIXELS_MAX = 8;
export const IMPROVE_MEGAPIXELS_STEP = 0.5;

/** A typed output-size value, made storable: clamped to the same bounds the
 *  Settings card enforces, quantised nowhere (2.5 is legal), NaN -> null so a
 *  half-typed box never becomes a settings write. */
function normalizeImproveMegapixels(value) {
  // '' first: Number('') is 0, which would turn an emptied box into a 0.5 write.
  if (value === '' || value === null || value === undefined) return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.min(IMPROVE_MEGAPIXELS_MAX, Math.max(IMPROVE_MEGAPIXELS_MIN, n));
}

/* ── The picked preset's CHAIN, tunable from here ───────────────────────────
   Reported the same way the instruction was: the panel named which preset ✨
   improve chains, and then said nothing about what that preset actually DOES —
   so the one number people came to change (how hard a LoRA pulls) still meant a
   trip to Settings, on a screen where the picture it applies to is right there.

   What this half offers is deliberately NARROW: the strengths of the rows the
   picked preset already has. Adding, removing, reordering or renaming stays in
   Settings ▸ Engines, because those are the acts that change what a preset IS —
   and a preset is shared by every surface that runs Klein, not just this one.
   Turning a dial is the reversible half; rebuilding the chain is not.

   The reach is the same as the rest of the panel: these ARE the app-wide preset
   rows, not a per-run copy. Two answers to "how strong is this LoRA?" is the
   failure this whole file exists to avoid, so the UI says so out loud. */

/** The Settings card's own bounds for a preset row (Klein card: 0 → 1.5, step
 *  0.05, junk reads as 0.6), mirrored so the two editors of ONE value can never
 *  offer different travel. */
export const PRESET_LORA_STRENGTH_MIN = 0;
export const PRESET_LORA_STRENGTH_MAX = LORA_STRENGTH_MAX;
export const PRESET_LORA_STRENGTH_STEP = 0.05;
export const PRESET_LORA_STRENGTH_DEFAULT = 0.6;

/** Into the range the Settings card offers and klein_edit_helper enforces. */
export function clampPresetStrength(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return PRESET_LORA_STRENGTH_DEFAULT;
  return Math.min(PRESET_LORA_STRENGTH_MAX, Math.max(PRESET_LORA_STRENGTH_MIN, n));
}

/** The rows of the preset named `name`, in chain order, ready to draw.
 *
 *  Each entry carries its ORIGINAL index in the stored list. Rows with no file
 *  are not drawn — there is nothing to tune on an empty slot — but they still
 *  occupy their position in config, and a write keyed on the drawn position
 *  would move the strength onto the wrong LoRA. The index is the join. */
export function presetChainRows(presets, name) {
  const wanted = typeof name === 'string' ? name.trim() : '';
  if (!wanted) return [];
  const preset = (Array.isArray(presets) ? presets : [])
    .find((p) => p && typeof p === 'object'
      && (typeof p.name === 'string' ? p.name.trim() : '') === wanted);
  if (!preset) return [];
  const rows = Array.isArray(preset.loras) ? preset.loras : [];
  const out = [];
  rows.forEach((row, index) => {
    const file = row && typeof row.file === 'string' ? row.file.trim() : '';
    if (!file) return;
    const n = Number(row.strength);
    out.push({
      index,
      file,
      // CLAMPED, like the pass: a config holding 3 (hand-edited, or copied from
      // the Krea card whose ceiling is 6) runs at 1.5, and a panel that showed
      // "3.00" next to the picture would be naming a strength nothing uses.
      strength: clampPresetStrength(Number.isFinite(n) ? n : PRESET_LORA_STRENGTH_DEFAULT),
    });
  });
  return withRowLabels(out);
}

/** Trim a stored path to the part that identifies the LoRA — the folder is the
 *  half that repeats. `truncate` cuts from the END, so at 360 px two files of
 *  the same folder became the same ellipsis; the name is what differs.
 *  When two rows of ONE preset share a filename the folder is the only thing
 *  telling them apart, so those keep their whole path. The full value is on
 *  `title` either way. */
function withRowLabels(rows) {
  const base = (f) => f.split(/[\\/]/).pop() || f;
  const seen = new Map();
  for (const r of rows) seen.set(base(r.file), (seen.get(base(r.file)) || 0) + 1);
  return rows.map((r) => ({ ...r, label: seen.get(base(r.file)) > 1 ? r.file : base(r.file) }));
}

/** The same preset list with ONE row's strength replaced, clamped to the range
 *  the Settings card and the backend both enforce.
 *
 *  Everything else is carried through untouched — the other presets, the other
 *  rows, an empty slot, a key some future version added. This panel edits one
 *  number; it must not be the reason anything else in that config changes. A
 *  name matching no preset, or an index naming no row, is returned unchanged
 *  rather than guessed at. */
export function withPresetRowStrength(presets, name, index, value) {
  const list = Array.isArray(presets) ? presets : [];
  const wanted = typeof name === 'string' ? name.trim() : '';
  if (!wanted) return list;
  const n = Number(value);
  // An input event carries a STRING, and a settings file full of strings is a
  // file the server has to guess about. Junk changes nothing at all.
  if (!Number.isFinite(n)) return list;
  const target = list.findIndex((p) => p && typeof p === 'object'
    && (typeof p.name === 'string' ? p.name.trim() : '') === wanted);
  if (target < 0) return list;
  const rows = Array.isArray(list[target].loras) ? list[target].loras : [];
  const row = rows[index];
  if (!row || typeof row !== 'object') return list;
  const strength = clampPresetStrength(n);
  // THE SAME REFERENCE BACK whenever there is nothing to change — an impossible
  // write above, and here a write that stores what is already stored. The
  // caller schedules a save on whatever this returns, so the identity check is
  // what keeps a no-op from becoming a request. Said as a promise the code
  // keeps: the first version of this comment sat above a map that always built
  // a new array.
  if (clampPresetStrength(row.strength) === strength) return list;
  return list.map((preset, i) => (i === target
    ? { ...preset, loras: rows.map((r, j) => (j === index ? { ...r, strength } : r)) }
    : preset));
}

/** The text improve will actually send, given a {stored, shipped} pair.
 *  Mirrors readImproveInstruction's rule byte for byte — a contract test in
 *  kleinImproveEditor.test.js asserts the two agree on real payload shapes, so
 *  the quoted line and the edit box can never disagree about what is in force. */
export function effectiveImprovePrompt({ stored = '', shipped = '' } = {}) {
  return String(stored || '').trim() ? stored : shipped;
}

/** Said in full, always. A global setting edited from a dataset screen reads as
 *  per-dataset unless something states otherwise, and by the time the user finds
 *  out, they have already re-run every other dataset. */
export const IMPROVE_SCOPE_NOTE =
  'This is the app-wide instruction — the same one Settings shows. Changing it here '
  + 'changes every ✨ Upscale & improve from now on — in every dataset, and in '
  + 'every bank.';

/** Shown while the toggle is off, in place of the box's usual footer. */
export const IMPROVE_OFF_NOTE =
  'Improve will only upscale — no instruction is sent.';

/** The PUT /api/settings body for the improve setting. PARTIAL by design: the
 *  endpoint deep-merges, so this touches neither the other identity prompts nor
 *  any other config section. Keys are omitted when absent from the patch — a
 *  toggle-only save must not also rewrite the prompt with a stale value.
 *  `loraPreset` writes klein.improve_lora_preset the same partial way; a patch
 *  carrying only it therefore sends NO identity_prompts section at all. */
export function improveSettingsPatch(patch = {}) {
  const config = {};
  const ip = {};
  if (patch.prompt !== undefined) ip.klein_improve = String(patch.prompt ?? '');
  if (patch.enabled !== undefined) ip.klein_improve_enabled = !!patch.enabled;
  if (Object.keys(ip).length) config.identity_prompts = ip;
  const klein = {};
  if (patch.loraPreset !== undefined) {
    klein.improve_lora_preset = String(patch.loraPreset ?? '');
  }
  // The whole preset list, as edited — a list has no partial form: the server
  // deep-merges sections but REPLACES arrays, which is exactly what a reordered
  // or shortened list needs. Callers send the array they were given with their
  // one change applied (withPresetRowStrength), never a rebuilt one.
  if (Array.isArray(patch.presets)) klein.generation_lora_presets = patch.presets;
  if (patch.megapixels !== undefined) {
    const mp = normalizeImproveMegapixels(patch.megapixels);
    // null means "not a number yet" — a half-typed box is not a write.
    if (mp !== null) klein.improve_megapixels = mp;
  }
  if (Object.keys(klein).length) config.klein = klein;
  return { config };
}

/** Coalescing saver for the instruction box.
 *
 *  Typing emits one change per keystroke and each one must NOT become a settings
 *  write. Pending fields merge into ONE patch and one request, so editing the
 *  text and then flipping the toggle sends a single PUT carrying both.
 *
 *  `flush()` exists for the case that actually loses work: closing the lightbox
 *  mid-sentence. The component calls it on unmount, so the last keystroke is
 *  saved instead of dying with the timer.
 *
 *  Timer functions are injectable so tests never wait in real time. */
export function createImproveSaver(save, {
  delay = 600,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
} = {}) {
  let pending = null;
  let timer = null;
  const fire = () => {
    timer = null;
    if (!pending) return;
    const patch = pending;
    pending = null;
    save(patch);
  };
  return {
    schedule(field, value) {
      pending = { ...(pending || {}), [field]: value };
      if (timer !== null) clearTimeoutFn(timer);
      timer = setTimeoutFn(fire, delay);
    },
    /** Send whatever is pending right now (unmount, toggle, reset). */
    flush() {
      if (timer !== null) { clearTimeoutFn(timer); timer = null; }
      fire();
    },
    cancel() {
      if (timer !== null) { clearTimeoutFn(timer); timer = null; }
      pending = null;
    },
    get pending() { return pending; },
  };
}
