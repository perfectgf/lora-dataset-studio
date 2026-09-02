import { useEffect, useState } from 'react'
import { INPUT_CLASS, Card, SecretField } from './primitives'
import { SettingsGroup, SettingsGroupsToc, useSettingsGroupProps } from './SettingsGroupsView'
import { ENGINES_GROUPS } from './settingsGroups'
import KleinLoraCombobox, { useKleinGenerationLoras } from './KleinLoraCombobox'
import ModelFilePicker, { useModelFiles } from './ModelFilePicker'
import ChatgptSubscriptionConnect from '../common/ChatgptSubscriptionConnect'
import PromptOverrideField from '../common/PromptOverrideField'
import PromptPreview from './PromptPreview'
import ResetToDefault from './ResetToDefault'
import { defaultValueAt } from './settingDefaults.js'
import { kreaStrengthRange, KREA_LORA_STRENGTH_DEFAULT } from '../../utils/kreaGenerationLoras'
import { isFixedLoraDuplicate, fixedLoraDuplicateWarning } from '../../utils/loraDuplicateGuard'
import { kreaBaseNote, KREA_BASE_NOTE_CLASS } from '../../utils/kreaBaseNote'
// The bounds mirror krea_edit_helper's clamps. They live in utils/kreaDials.js
// because the workspace panel offers the SAME four dials — two copies of "512"
// would be two chances to drift away from the server.
import {
  KREA_GROUNDING_MIN, KREA_GROUNDING_MAX, KREA_GROUNDING_STEP,
  KREA_STEPS_MIN, KREA_STEPS_MAX,
  KREA_REF_BOOST_MIN, KREA_REF_BOOST_MAX, KREA_REF_BOOST_STEP,
  KREA_IDENTITY_STRENGTH_MIN, KREA_IDENTITY_STRENGTH_MAX, KREA_IDENTITY_STRENGTH_STEP,
  clampRefBoost, clampIdentityStrength,
  refBoostDescription, identityStrengthDescription, stepsDescription,
} from '../../utils/kreaDials.js'
import {
  identityPromptFields, PROMPT_SUBJECT_TYPES,
  readIdentityPrompt, writeIdentityPrompt, subjectHasOverride,
  GLOBAL_PROMPT_PART_FIELDS, SUBJECT_PROMPT_PART_FIELDS, FRAMING_PROMPT_PART_FIELDS,
} from '../common/promptOverride.js'
import { SUBJECT_TYPE_LABELS } from '../dataset/subjectTypes.js'
import { laneForTarget } from '../setup/seedvr2Tiling.js'

const ENGINE_SECRETS = [
  { key: 'GEMINI_API_KEY', label: 'Gemini API key', testTarget: 'gemini', help: 'Powers the Nano Banana engine.' },
  { key: 'OPENAI_API_KEY', label: 'OpenAI API key', testTarget: 'openai',
    help: 'Powers the ChatGPT engine (gpt-image-2 by default). Optional if you connect a ChatGPT subscription below.' },
  { key: 'OPENROUTER_API_KEY', label: 'OpenRouter API key', testTarget: 'openrouter',
    help: 'Powers the OpenRouter engine: one account and one balance in front of the same '
      + 'upstream image models, including the ones the two engines above call directly. '
      + 'Test only checks that a key is saved — OpenRouter bills per request, so nothing '
      + 'is sent until you generate.' },
]

const ENGINE_OPTIONS = [
  { id: 'nanobanana', label: 'Nano Banana (Gemini)' },
  { id: 'chatgpt', label: 'ChatGPT (OpenAI)' },
  { id: 'openrouter', label: 'OpenRouter' },
  { id: 'klein', label: 'Klein (ComfyUI, local)' },
  { id: 'krea', label: 'Krea 2 Edit (ComfyUI, local)' },
]

/* Optional generation-LoRA PRESETS, originally for the local Klein engine
   (Idea by @waltm — Discord feature request) and now shared with the local
   Krea 2 Edit engine too: named combinations of user-pointed LoRA files (any
   files, any purpose — texture, anatomy, style…). Inside a preset the rows
   chain after the consistency/identity-edit LoRA in LIST ORDER (file +
   strength, reorderable, capped at 8). Per run each engine's own tuning panel
   just PICKS a preset, starting on the engine's own default preset setting
   ("None" until one is chosen) — the choice carries the intent,
   there is no automatic gating. The app never ships or hardcodes a LoRA name. */
const MAX_GENERATION_LORAS = 8        // mirrors the klein_edit_helper AND
const MAX_GENERATION_LORA_PRESETS = 12 // krea_edit_helper caps — both the same

const SMALL_BTN = 'grid h-6 w-6 place-items-center rounded border border-border text-xs ' +
  'text-content-muted hover:bg-surface-raised disabled:opacity-30'
const TEXT_BTN = 'rounded-md border border-border-strong px-2 py-1 text-xs font-medium ' +
  'text-content hover:bg-surface-raised disabled:opacity-50'

/** Fresh name not colliding with the existing presets ("Preset 2", "x (copy)"…). */
function freeName(presets, base) {
  const taken = new Set(presets.map((p) => (p?.name || '').trim()))
  if (!taken.has(base)) return base
  for (let n = 2; ; n += 1) {
    const cand = `${base} ${n}`
    if (!taken.has(cand)) return cand
  }
}

/* One preset: its name, its ordered LoRA rows, and the row controls. Shared by
   the Klein and the Krea cards — the shapes are identical, only the strength
   range, its default and the engine the badge judges for differ. */
function LoraPresetCard({ preset, index, presets, save, loraScan,
                          engineLabel = 'Klein', strengthRange, defaultStrength,
                          placeholder = 'klein/my-lora.safetensors',
                          engineId = 'klein', fixedLora = '' }) {
  const rows = Array.isArray(preset?.loras) ? preset.loras : []
  const patchPreset = (p) => save(presets.map((x, j) => (j === index ? { ...x, ...p } : x)))
  const patchRow = (i, p) => patchPreset({ loras: rows.map((r, j) => (j === i ? { ...r, ...p } : r)) })
  const moveRow = (i, dir) => {
    const j = i + dir
    if (j < 0 || j >= rows.length) return
    const next = [...rows]
    ;[next[i], next[j]] = [next[j], next[i]]
    patchPreset({ loras: next })
  }
  return (
    <div className="rounded-lg border border-border p-3 space-y-2">
      <div className="flex items-center gap-2">
        <input
          type="text" aria-label={`Preset ${index + 1} name`}
          value={preset?.name || ''}
          onChange={(e) => patchPreset({ name: e.target.value })}
          placeholder="Preset name"
          className={`${INPUT_CLASS} mt-0 font-medium`}
        />
        <button type="button" className={TEXT_BTN}
          disabled={presets.length >= MAX_GENERATION_LORA_PRESETS}
          onClick={() => save([...presets,
            { ...preset, name: freeName(presets, `${(preset?.name || 'Preset').trim() || 'Preset'} (copy)`), loras: rows.map((r) => ({ ...r })) }])}
          title="Duplicate this preset">
          Duplicate
        </button>
        <button type="button" className={`${TEXT_BTN} hover:bg-red-500/15 hover:text-red-300`}
          onClick={() => save(presets.filter((_, j) => j !== index))}
          title="Delete this preset">
          Delete
        </button>
      </div>
      {rows.length === 0 && (
        <p className="text-xs text-content-muted">Empty preset — add a LoRA below.</p>
      )}
      {rows.map((row, i) => {
        const strength = Number.isFinite(Number(row?.strength)) ? Number(row.strength) : defaultStrength
        const range = strengthRange(row?.file || '')
        // The row the server will DROP: it names the LoRA this engine already
        // loads outside the presets. Said HERE, where the row is written —
        // until now the only trace was one line in the server log, so a preset
        // whose single row was that file produced a run with no LoRA at all and
        // nothing on screen to explain it. Comparison is normcase+normpath, the
        // server's own, so a '/' or a case difference cannot dodge it.
        const duplicate = isFixedLoraDuplicate(row?.file, fixedLora)
        return (
          <div key={i} className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-content-muted w-4 shrink-0" aria-hidden="true">{i + 1}.</span>
            <KleinLoraCombobox
              ariaLabel={`Preset ${index + 1} LoRA file ${i + 1}`}
              value={row?.file || ''}
              onChange={(next) => patchRow(i, { file: next })}
              engineLabel={engineLabel}
              placeholder={placeholder}
              {...loraScan}
            />
            <label className="flex items-center gap-1.5 text-xs text-content-muted">
              <span className="whitespace-nowrap">{strength.toFixed(2)}</span>
              <input
                type="range" min={range.min} max={range.max} step={0.05} value={strength}
                aria-label={`Preset ${index + 1} LoRA ${i + 1} strength`}
                onChange={(e) => patchRow(i, { strength: Number(e.target.value) })}
                className="w-28 accent-indigo-500"
              />
            </label>
            <button type="button" onClick={() => moveRow(i, -1)} disabled={i === 0}
              aria-label={`Move LoRA ${i + 1} up in preset ${index + 1}`} title="Chain earlier" className={SMALL_BTN}>↑</button>
            <button type="button" onClick={() => moveRow(i, 1)} disabled={i === rows.length - 1}
              aria-label={`Move LoRA ${i + 1} down in preset ${index + 1}`} title="Chain later" className={SMALL_BTN}>↓</button>
            <button type="button" onClick={() => patchPreset({ loras: rows.filter((_, j) => j !== i) })}
              aria-label={`Remove LoRA ${i + 1} from preset ${index + 1}`} title="Remove this LoRA"
              className={`${SMALL_BTN} hover:bg-red-500/15 hover:text-red-300`}>✕</button>
            {/* w-full inside the WRAPPING flex row = its own line under the
                controls, without re-nesting (and re-indenting) the whole row. */}
            {duplicate && (
              <p role="alert" className="w-full pl-6 text-[0.6875rem] text-amber-400">
                {fixedLoraDuplicateWarning(engineId)}
              </p>
            )}
          </div>
        )
      })}
      <div className="flex items-center gap-3">
        <button
          type="button" className={TEXT_BTN}
          onClick={() => patchPreset({ loras: [...rows, { file: '', strength: defaultStrength }] })}
          disabled={rows.length >= MAX_GENERATION_LORAS}
        >
          ＋ Add LoRA
        </button>
        <span className="text-xs text-content-muted">{rows.length}/{MAX_GENERATION_LORAS} in the chain</span>
      </div>
    </div>
  )
}

/* Which preset the run panel OPENS on, per engine.
   The panel used to start on "None" on every single visit, so a preset someone
   had carefully built applied only when they remembered to re-pick it — and a
   run that forgot carried no LoRA anywhere in its PNG metadata, which reads as
   the app ignoring its own settings. This is the missing half of the feature,
   not a new one.
   Deliberately per ENGINE: klein.generation_lora_presets and
   krea.generation_lora_presets are independent lists where the same NAME can
   designate two different chains, so one shared default would be a lie half the
   time. And deliberately a STARTING POINT, which the note under the field says
   out loud: the run panel still offers None and every other preset for that run,
   and choosing there never writes back here. */
function DefaultPresetField({ id, engineLabel, presets, value, onChange }) {
  const names = presets.map((p) => (p?.name || '').trim()).filter(Boolean)
  const current = typeof value === 'string' ? value.trim() : ''
  // Fail-closed, mirroring resolveDefaultPresetName: a name matching nothing
  // (renamed preset, hand-edited config.json) behaves as "None" everywhere, so
  // the field must not silently show "None" as if the setting were empty.
  const stale = !!current && !names.includes(current)
  return (
    <div className="border-t border-border pt-3">
      <label htmlFor={id} className="block text-xs font-medium text-content">
        Preset selected by default
      </label>
      <select
        id={id}
        value={stale ? '' : current}
        onChange={(e) => onChange(e.target.value)}
        className={`${INPUT_CLASS} sm:max-w-xs`}
      >
        <option value="">None</option>
        {names.map((n) => <option key={n} value={n}>{n}</option>)}
      </select>
      <p className="mt-1 text-[0.6875rem] text-content-subtle">
        Which preset the {engineLabel} tuning panel starts on when you open a dataset.
        “None” is the shipped default and keeps today’s behaviour exactly. You can still
        pick another preset — or None — for a single run without changing this setting.
      </p>
      {stale && (
        <p role="alert" className="mt-1 text-[0.6875rem] text-amber-400">
          “{current}” is no longer one of your presets, so runs start on None. Pick a
          preset above to set a new default.
        </p>
      )}
    </div>
  )
}

function KleinLorasCard({ config, setField }) {
  const presets = Array.isArray(config.klein?.generation_lora_presets)
    ? config.klein.generation_lora_presets : []
  const save = (next) => setField('klein', 'generation_lora_presets', next)
  // ONE scan of ComfyUI's loras folder, shared by every row's picker (never one
  // fetch per row). Degrades to free-text on any failure — see the hook.
  const loraScan = useKleinGenerationLoras()
  return (
    <Card
      id="klein-generation-lora-presets"
      title="Klein generation LoRA presets (optional)"
      help={`Named combinations of your own LoRA files, chained after the consistency LoRA on the local Klein engine — inside a preset the order is the chain order (max ${MAX_GENERATION_LORAS} LoRAs each, ${MAX_GENERATION_LORA_PRESETS} presets). Pick each row from the LoRAs found under ComfyUI's models/loras (Klein-compatible ones are listed first; you can still type a path for a file not on disk yet) — any LoRA, any purpose. Per run, pick a preset in the workspace's Klein tuning panel — it opens on the default preset chosen below ("None" until you choose one), and picking something else there applies to that run only. Presets and LoRA autocomplete by @waltm (Discord).`}
    >
      {presets.length === 0 && (
        <p className="text-sm text-content-muted">No presets yet — create your first combination below.</p>
      )}
      {presets.map((preset, i) => (
        <LoraPresetCard key={i} preset={preset} index={i} presets={presets} save={save}
          loraScan={loraScan} engineLabel="Klein"
          strengthRange={() => ({ min: 0, max: 1.5 })}
          defaultStrength={0.6}
          engineId="klein" fixedLora={config.klein?.consistency_lora || ''} />
      ))}
      <div className="flex items-center gap-3">
        <button
          type="button" className={TEXT_BTN}
          onClick={() => save([...presets, { name: freeName(presets, 'My preset'), loras: [] }])}
          disabled={presets.length >= MAX_GENERATION_LORA_PRESETS}
        >
          ＋ New preset
        </button>
        <span className="text-xs text-content-muted">{presets.length}/{MAX_GENERATION_LORA_PRESETS}</span>
      </div>
      <DefaultPresetField
        id="klein-default-lora-preset" engineLabel="Klein" presets={presets}
        value={config.klein?.default_generation_lora_preset || ''}
        onChange={(v) => setField('klein', 'default_generation_lora_preset', v)} />
    </Card>
  )
}

/* The pinnable Klein model slots. `key` is the DOM id the help registry focuses
   (the contract test scans these literals); `cfg` is the klein.* config key;
   `slot` matches caps.comfyui.klein_overrides.
   Ported from socrasteeze's branch (GitHub #20). */
const KLEIN_MODEL_SLOTS = [
  { key: 'klein-model-unet', cfg: 'unet', slot: 'unet', pick: 'klein_unet', label: 'Diffusion model (UNET)',
    hint: 'Full path from anywhere, or relative to a diffusion-model folder — e.g. klein/flux-2-klein-9b-fp8.safetensors under models/unet (a bare filename for a file sitting at a folder root). A filename without "fp8" loads at full precision instead of being quantized.' },
  { key: 'klein-model-text_encoder', cfg: 'text_encoder', slot: 'text_encoder', pick: 'klein_text_encoder', label: 'Text encoder',
    hint: 'Full path, or relative to models/text_encoders — e.g. qwen_3_8b_fp8mixed.safetensors.' },
  { key: 'klein-model-vae', cfg: 'vae', slot: 'vae', pick: 'klein_vae', label: 'VAE',
    hint: 'Full path, or relative to models/vae — e.g. flux2-vae.safetensors.' },
  { key: 'klein-model-consistency_lora', cfg: 'consistency_lora', slot: 'consistency_lora', pick: 'klein_consistency_lora',
    label: 'Consistency LoRA',
    placeholder: 'Empty = no consistency LoRA',
    missText: 'Not found — at the shipped name the LoRA is simply skipped (Setup can download it); a name you chose yourself stops the engine instead',
    hint: 'Full path, or relative to models/loras — the structure-anchoring LoRA chained onto the Klein edit graph. Unlike the three above, this one has a shipped default and clearing it disables the LoRA rather than turning on auto-detection.' },
]

/* One badge per resolve status from caps.comfyui.klein_overrides. Two wordings
   are deliberate rather than cosmetic:
   - 'outside_roots' is NOT "not found" — the file IS there, ComfyUI simply
     cannot reach it, and a different action fixes that;
   - what happens after a miss is not the same for every slot, so the badge does
     not claim it: the three model slots fall back to auto-detection, while the
     consistency LoRA has no detection to fall back to (it is just skipped, and
     reported as a missing asset Setup can download). Saying "auto-detection is
     used" on that row would name a mechanism that does not exist for it. */
function overrideBadge(st, missText) {
  if (!st) return null
  if (st.found) return { cls: 'text-emerald-400', text: 'Found' }
  if (st.status === 'outside_roots') {
    return { cls: 'text-amber-400',
             text: "Could not be linked into ComfyUI's model folders — check permissions, or move the file" }
  }
  return { cls: 'text-amber-400', text: missText || 'Not found — the engine will refuse to run until you fix or clear this' }
}

/* ONE pinnable slot. Its own component because each row needs its OWN scan of a
   DIFFERENT ComfyUI folder, and a hook cannot be called inside a .map. Each scan
   is cached server-side on that folder's mtime, so four rows are four cheap
   requests, not four directory walks. */
function KleinModelSlotRow({ spec, config, setField, override }) {
  const { key, cfg, label, hint, placeholder, missText, pick } = spec
  const scan = useModelFiles(pick)
  const badge = overrideBadge(override, missText)
  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-x-2">
        <label htmlFor={key} className="block text-sm font-medium text-content">{label}</label>
        {badge && (
          <span className={`text-xs sm:text-right ${badge.cls}`}>{badge.text}</span>
        )}
      </div>
      <ModelFilePicker
        id={key}
        ariaLabel={`Klein ${label}`}
        value={config.klein?.[cfg] || ''}
        onChange={(v) => setField('klein', cfg, v)}
        placeholder={placeholder || 'Empty = auto-detect'}
        {...scan}
      />
      <p className="mt-1 text-[0.6875rem] text-content-subtle">{hint}</p>
    </div>
  )
}

function KleinModelFilesCard({ config, setField, caps }) {
  const overrides = caps?.comfyui?.klein_overrides || {}
  return (
    <Card
      id="klein-model-files"
      title="Klein model files (optional)"
      help="Pin the exact files the Klein graph loads instead of relying on auto-detection (the canonical download names, then a narrow token scan). Each field takes a full absolute path OR a ComfyUI-relative loader name. A path under one of ComfyUI's model folders (extra_model_paths.yaml roots included) is converted automatically to what the loader needs; a path from anywhere else is hardlinked into an lds-pinned/ folder so ComfyUI can load it without you moving a multi-GB file. Each field lists the files actually found in that ComfyUI folder — you can still type a name or a full path for a file that is not there yet. Leave a field empty to keep auto-detection for that slot. A pinned file that cannot be resolved STOPS the engine and says which one: it used to fall back to auto-detection, which meant the graph loaded a different file from the one shown here and nobody found out until the images came back wrong. Clearing the field is how you go back to auto-detection. Contributed by socrasteeze (GitHub)."
    >
      {KLEIN_MODEL_SLOTS.map((s) => (
        <KleinModelSlotRow key={s.key} spec={s} config={config} setField={setField}
          override={overrides[s.slot]} />
      ))}
    </Card>
  )
}

/* Klein GENERATION sampling. The shipped workflow hardcodes 5 steps at its
   sampler node and nothing on the generation paths ever passed a value, so the
   engine's own `sampler_steps` parameter was unreachable — "is the number of
   generation steps fixed at 5?" (ashish.sinha, Discord). Default 5 = the exact
   historical render; the ceiling mirrors the backend clamp. Deliberately its own
   card, next to the other Klein knobs and clearly NOT the "Upscale & improve"
   steps, which drive a different pass. */
const KLEIN_GENERATION_STEPS_MAX = 50   // face_dataset_service._IMPROVE_MAX_STEPS
const KLEIN_EDIT_LORA_MAX = 2           // face_dataset_service._IMPROVE_MAX_STRENGTH

function KleinGenerationCard({ config, setField, configDefaults }) {
  // The shipped 5 is read from the server payload, never retyped here: it used
  // to be a literal `?? 5` in this file, i.e. a second copy of a backend default
  // that nothing kept in sync.
  const shipped = defaultValueAt(configDefaults, 'klein', 'generation_steps')
  const steps = config.klein?.generation_steps ?? shipped
  // Enhancement LoRA on the EDIT lanes. The workflow pins node 139 at 0.8 and no
  // lane but "Upscale & improve" overrode it, which only became visible once
  // Setup started downloading the file: every edit gained a style LoRA at 0.8.
  // Default 0 = the render before that download existed.
  const editLoraShipped = defaultValueAt(configDefaults, 'klein', 'edit_base_lora_strength')
  const editLora = config.klein?.edit_base_lora_strength ?? editLoraShipped
  return (
    <Card
      id="klein-generation"
      title="Klein generation quality"
      help="How many sampler steps the local Klein engine spends on each generated variation, and how much of the enhancement LoRA it mixes in. 5 steps is the value the app used before this was exposed, so leaving it alone keeps today's result. More steps render more cleanly but take proportionally longer — 10 steps is roughly twice the wait per image. It will not fix a wrong prompt: anatomy problems (extra limbs, tails) come from the identity prompt, not from the step count. Raised by ashish.sinha (Discord)."
    >
      <div className="sm:max-w-xs">
        <label htmlFor="klein-generation-steps" className="block text-xs font-medium text-content">
          Generation steps
        </label>
        <input
          id="klein-generation-steps"
          type="number"
          min={1}
          max={KLEIN_GENERATION_STEPS_MAX}
          step={1}
          value={steps}
          onChange={(e) => setField('klein', 'generation_steps',
            e.target.value === '' ? shipped : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {shipped} = the shipped value. More steps = slower, usually cleaner; 1–{KLEIN_GENERATION_STEPS_MAX}.
          Applies to variations, regenerations and the small-image rescue — not to
          “Upscale &amp; improve”, which has its own Steps below.
        </p>
        <ResetToDefault label="Generation steps" section="klein" field="generation_steps"
          config={config} configDefaults={configDefaults} setField={setField} />
      </div>
      <div className="mt-4 sm:max-w-xs">
        <label htmlFor="klein-edit-lora" className="block text-xs font-medium text-content">
          Enhancement LoRA on edits
        </label>
        <input
          id="klein-edit-lora"
          type="number"
          min={0}
          max={KLEIN_EDIT_LORA_MAX}
          step={0.05}
          value={editLora}
          onChange={(e) => setField('klein', 'edit_base_lora_strength',
            e.target.value === '' ? editLoraShipped : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          0 = off. The workflow carries a detail LoRA (klein/realistic.safetensors) at
          0.8, and until now nothing turned it down on an edit — it pulled results
          away from the instruction you typed. Raise it to add its detail on purpose.
          Applies to reference edits, variations, regenerations and the small-image
          rescue — “Upscale &amp; improve” keeps its own Enhancement LoRA below.
        </p>
        <ResetToDefault label="Enhancement LoRA on edits"
          section="klein" field="edit_base_lora_strength"
          config={config} configDefaults={configDefaults} setField={setField} />
      </div>
    </Card>
  )
}

/* Krea 2 Identity Edit — the second LOCAL engine. Its headline knob is
   `grounding_px`, THE consistency <-> prompt-adherence dial, so it is first and
   explained in plain words: a number nobody can interpret is not a setting.
   The FOUR calibration dials of this card (grounding, steps, reference pull,
   identity LoRA strength) are the same four the workspace's "Krea 2 Edit
   tuning" panel offers, on purpose: they are judged on the images that panel
   produces and configured here, and since every control writes the SAME global
   key through the same endpoint there is only ever one value to read.
   The two path fields are NOT duplicated there — they are filled once at
   install, not adjusted while looking at a result. They are BLANK-MEANS-AUTO on
   purpose: the resolver finds the files by canonical name then by a narrow
   token across every ComfyUI model root, so an install that looks nothing like
   the developer's works untouched — they exist for the person whose files are
   named something else. */

// Mirror seedvr2_helper's clamps. The SERVER stays the authority (it re-clamps
// every value), these only stop the input offering a number that would be
// silently corrected.
const SEEDVR2_RESOLUTION_MIN = 256
const SEEDVR2_RESOLUTION_MAX = 4096
const SEEDVR2_MAX_RESOLUTION_MAX = 8192
const SEEDVR2_BLOCKS_MAX = 36
// seedvr2_helper.TILE_PX_MIN / TILE_PX_MAX, and the factor the 'auto' crossover
// is derived from (TILE_ABOVE_FACTOR) — shown, never enforced here.
const SEEDVR2_TILE_MIN = 512
const SEEDVR2_TILE_MAX = 2048
const SEEDVR2_TILE_ABOVE_FACTOR = 1.5
// seedvr2_helper.COLOR_CORRECTIONS — the node's own enum, in its own order.
const SEEDVR2_COLOR_MODES = ['lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none']

function KreaCard({ config, setField, configDefaults, caps }) {
  const krea = config.krea || {}
  const reset = { config, configDefaults, setField }
  const dflt = (key) => defaultValueAt(configDefaults, 'krea', key)
  // One scan per slot, fired when the card mounts. Both degrade to an empty list
  // (=> plain free-text field) rather than blocking the panel — an absolute path
  // from outside every ComfyUI root is a legitimate value no scan can enumerate.
  const baseScan = useModelFiles('krea_base_model')
  const identityScan = useModelFiles('krea_identity_lora')
  const grounding = Number(krea.grounding_px ?? dflt('grounding_px'))
  const steps = krea.steps ?? dflt('steps')
  // Clamped for DISPLAY only: a config.json hand-edited past the server's clamp
  // would otherwise park the slider thumb at an end while the label showed a
  // number the graph will never receive. The server re-clamps on its side.
  const refBoost = clampRefBoost(krea.ref_boost, dflt('ref_boost'))
  const identityStrength = clampIdentityStrength(krea.identity_lora_strength,
    dflt('identity_lora_strength'))
  // WHICH Krea base this install loads, named. Resolved SERVER-side
  // (caps.comfyui.krea_base_resolved = the resolve_krea_unet() the generation
  // path calls) — the browser ranks nothing. See utils/kreaBaseNote.js.
  const baseNote = kreaBaseNote(krea.base_model, caps?.comfyui?.krea_base_resolved)
  return (
    <Card
      id="krea-engine"
      title="Krea 2 Edit (local)"
      help="The second local engine. It re-stages your reference photo — new angle, framing, light, background — while keeping the face and the body, from that ONE photo and with no character LoRA, which is what makes it useful before a LoRA exists. It needs the comfyui-krea2edit custom-node pack plus four model files; the engine card in the workspace names whatever is still missing. Krea Fit v1.2 honors the selected shot card's framing and aspect ratio instead of copying the source photo's shape."
    >
      <div className="sm:max-w-md">
        <label htmlFor="krea-grounding" className="block text-xs font-medium text-content">
          Reference grounding ({grounding} px)
        </label>
        <input
          id="krea-grounding"
          type="range"
          min={KREA_GROUNDING_MIN}
          max={KREA_GROUNDING_MAX}
          step={KREA_GROUNDING_STEP}
          value={grounding}
          onChange={(e) => setField('krea', 'grounding_px', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          The resolution your reference is shown to the model&rsquo;s vision encoder at — the
          consistency ↔ prompt dial. At the low end it follows the shot description (more
          variety in pose, outfit and scene, looser likeness). <b>Higher</b> = it resembles
          the reference more, but can copy the pose and outfit you asked it to change.
          512 px is the dataset-restaging balance: it keeps the prompt and selected shot card
          in charge while preserving identity. Raise it deliberately when reference likeness
          matters more. Also adjustable, with this exact value, from the workspace&rsquo;s
          Krea 2 Edit tuning panel.
        </p>
        <ResetToDefault label="Reference grounding" section="krea" field="grounding_px" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="krea-steps" className="block text-xs font-medium text-content">
          Sampler steps
        </label>
        <input
          id="krea-steps"
          type="number"
          min={KREA_STEPS_MIN}
          max={KREA_STEPS_MAX}
          step={1}
          value={steps}
          onChange={(e) => setField('krea', 'steps',
            e.target.value === '' ? dflt('steps') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {stepsDescription(steps)}. {dflt('steps')} is the value the model&rsquo;s own
          reference workflow uses. More is slower and rarely better on this pipeline.
        </p>
        <ResetToDefault label="Sampler steps" section="krea" field="steps" {...reset} />
      </div>

      {/* The two calibration dials that used to have NO input on this page: they
          were reachable only from the workspace panel, so "where do I change
          this?" had a different answer per dial. Same key, same endpoint, same
          value — a slider here and a slider there cannot disagree. */}
      <div className="mt-3 sm:max-w-md">
        <label htmlFor="krea-ref-boost" className="block text-xs font-medium text-content">
          Reference pull ({refBoost})
        </label>
        <input
          id="krea-ref-boost"
          type="range"
          min={KREA_REF_BOOST_MIN}
          max={KREA_REF_BOOST_MAX}
          step={KREA_REF_BOOST_STEP}
          value={refBoost}
          onChange={(e) => setField('krea', 'ref_boost',
            clampRefBoost(e.target.value, dflt('ref_boost')))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {refBoostDescription(refBoost)}. How hard the source latent is pushed back into the
          model at every denoising step — the lever for &ldquo;the subject does not look enough
          like my reference&rdquo;. High values also recopy the composition, pose and outfit the
          shot card asked it to change. Also on the workspace&rsquo;s Krea 2 Edit tuning panel,
          where you judge the result.
        </p>
        <ResetToDefault label="Reference pull" section="krea" field="ref_boost"
          value={refBoost} {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="krea-identity-lora-strength" className="block text-xs font-medium text-content">
          Identity LoRA strength ({identityStrength})
        </label>
        <input
          id="krea-identity-lora-strength"
          type="range"
          min={KREA_IDENTITY_STRENGTH_MIN}
          max={KREA_IDENTITY_STRENGTH_MAX}
          step={KREA_IDENTITY_STRENGTH_STEP}
          value={identityStrength}
          onChange={(e) => setField('krea', 'identity_lora_strength',
            clampIdentityStrength(e.target.value, dflt('identity_lora_strength')))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {identityStrengthDescription(identityStrength)}. The weight of the Krea 2
          identity-edit LoRA itself — the piece that carries the face across. Below 1 loosens
          the likeness, 0 disables the face transfer, above 1 is past what the file was
          trained for and can posterize. Also on the workspace&rsquo;s Krea 2 Edit tuning panel.
        </p>
        <ResetToDefault label="Identity LoRA strength" section="krea"
          field="identity_lora_strength" value={identityStrength} {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="krea-base-model" className="block text-xs font-medium text-content">
          Base model file (optional)
        </label>
        <ModelFilePicker
          id="krea-base-model"
          ariaLabel="Krea base model file"
          value={krea.base_model ?? ''}
          onChange={(v) => setField('krea', 'base_model', v)}
          placeholder="auto — finds a Krea 2 Turbo/Raw build"
          {...baseScan}
        />
        <p className={`mt-1 font-mono text-[0.6875rem] ${KREA_BASE_NOTE_CLASS[baseNote.tone]}`}>
          {baseNote.text}
        </p>
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Leave blank unless you own several Krea builds. Blank = the app picks a Krea 2
          Turbo then Raw model from your ComfyUI. The list is what the app would actually
          elect from: non-Krea-2 checkpoints that merely carry &ldquo;krea&rdquo; in their name are
          skipped there too, because the identity LoRA renders pure noise on them. Pick a
          file that is not on disk and Krea refuses to run rather than loading another one.
        </p>
        {/* The default here is the EMPTY string, and resetting writes exactly
            that: blank means "resolve it yourself", and a reset must give that
            state back rather than freeze whichever file the app happens to
            pick today. */}
        <ResetToDefault label="Base model file" section="krea" field="base_model" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="krea-identity-lora" className="block text-xs font-medium text-content">
          Identity edit LoRA (optional)
        </label>
        <ModelFilePicker
          id="krea-identity-lora"
          ariaLabel="Krea identity edit LoRA"
          value={krea.identity_lora ?? ''}
          onChange={(v) => setField('krea', 'identity_lora', v)}
          placeholder="blank = find krea2_identity_edit automatically"
          {...identityScan}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Blank = the app searches your LoRA folders for a krea2_identity_edit file, so a
          renamed download still works. Name one here and that is the LoRA it loads — if it
          is not on disk, Krea refuses to run instead of substituting another face transfer.
        </p>
        <ResetToDefault label="Identity edit LoRA" section="krea" field="identity_lora" {...reset} />
      </div>
    </Card>
  )
}

/* The Krea hi-res fix — a SECOND latent pass on the generation graph.

   It sits in the Krea group but reads `krea_hires.*`, not `krea.*`, and that is
   not tidiness: the card above tunes the Krea 2 EDIT engine, a hand-built graph
   that never loads krea2_turbo.json. Two graphs, two namespaces; a shared one
   would be a dial that silently applies to only one of them.

   Off by default, and off means no node is added to the graph at all — an
   install that never touches this renders exactly what it rendered before. */
const KREA_HIRES_SCALE_MIN = 1.0
// utils/comfyui.KREA_HIRES_MAX_SCALE — the server re-clamps on its side.
const KREA_HIRES_SCALE_MAX = 2.0
const KREA_HIRES_SCALE_STEP = 0.25
const KREA_HIRES_DENOISE_MIN = 0.05
const KREA_HIRES_DENOISE_MAX = 1.0
const KREA_HIRES_STEPS_MAX = 50

/** What the configured scale costs and buys, in the units people think in. A x2
 *  LATENT is x4 the pixels, which is the number that decides whether it fits. */
function hiresScaleNote(scale, width = 1024, height = 1024) {
  if (!(scale > 1)) return 'Off — the graph is built exactly as it is today, with a single pass.'
  const w = Math.round(width * scale), h = Math.round(height * scale)
  return `A ${width}x${height} first pass is re-sampled at ${w}x${h} `
    + `(${(scale * scale).toFixed(2)}x the pixels, so roughly that much again in time and VRAM).`
}

function KreaHiresCard({ config, setField, configDefaults }) {
  const hires = config.krea_hires || {}
  const reset = { config, configDefaults, setField }
  const dflt = (key) => defaultValueAt(configDefaults, 'krea_hires', key)
  // Clamped for DISPLAY only, like the dials above: a hand-edited config.json
  // past the server's clamp would otherwise park the thumb at one end while the
  // label showed a number the graph will never receive.
  const scale = Math.min(KREA_HIRES_SCALE_MAX,
    Math.max(KREA_HIRES_SCALE_MIN, Number(hires.scale ?? dflt('scale')) || KREA_HIRES_SCALE_MIN))
  const denoise = Math.min(KREA_HIRES_DENOISE_MAX,
    Math.max(KREA_HIRES_DENOISE_MIN, Number(hires.denoise ?? dflt('denoise')) || KREA_HIRES_DENOISE_MIN))
  const steps = hires.steps ?? dflt('steps')
  const on = scale > 1
  return (
    <Card
      id="krea-hires"
      title="Krea hi-res fix (local)"
      help="A second sampling pass at a higher latent resolution, on the Krea generation graph — the Test Studio grid and the LoRA Canvas. Krea composes at the resolution it samples: asked for the final size in one pass it loses framing and starts duplicating subjects, kept small it caps the detail no upscaler can invent afterwards. Sampling small and re-sampling an upscaled latent lets the model DRAW that detail instead. Core ComfyUI nodes only — nothing to install. Off by default."
    >
      <div className="sm:max-w-md">
        <label htmlFor="krea-hires-scale" className="block text-xs font-medium text-content">
          Second pass ({on ? `${scale}x the latent` : 'off'})
        </label>
        <input
          id="krea-hires-scale"
          type="range"
          min={KREA_HIRES_SCALE_MIN}
          max={KREA_HIRES_SCALE_MAX}
          step={KREA_HIRES_SCALE_STEP}
          value={scale}
          onChange={(e) => setField('krea_hires', 'scale', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {hiresScaleNote(scale)} {on ? '' : 'Slide right to turn it on. '}
          1.5x is the value the reference workflow this was ported from uses.
        </p>
        <ResetToDefault label="Second pass" section="krea_hires" field="scale" {...reset} />
      </div>

      {/* The two dials below only mean anything once the pass exists. Shown
          disabled rather than hidden: a control that appears out of nowhere is
          how people miss that it was there to be set. */}
      <div className={`mt-3 sm:max-w-md ${on ? '' : 'opacity-50'}`}>
        <label htmlFor="krea-hires-denoise" className="block text-xs font-medium text-content">
          How much it may rewrite ({denoise})
        </label>
        <input
          id="krea-hires-denoise"
          type="range"
          min={KREA_HIRES_DENOISE_MIN}
          max={KREA_HIRES_DENOISE_MAX}
          step={0.05}
          value={denoise}
          disabled={!on}
          onChange={(e) => setField('krea_hires', 'denoise', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          The dial of the whole feature. Near 1 the second pass ignores the first and
          renders a <b>different</b> picture at the larger size; too low and it costs a
          full extra pass to change nothing. {dflt('denoise')} keeps the composition and
          rewrites the texture.
        </p>
        <ResetToDefault label="How much it may rewrite" section="krea_hires" field="denoise" {...reset} />
      </div>

      <div className={`mt-3 sm:max-w-md ${on ? '' : 'opacity-50'}`}>
        <label htmlFor="krea-hires-steps" className="block text-xs font-medium text-content">
          Second-pass steps (0 = same as the first)
        </label>
        <input
          id="krea-hires-steps"
          type="number"
          min={0}
          max={KREA_HIRES_STEPS_MAX}
          step={1}
          value={steps}
          disabled={!on}
          onChange={(e) => setField('krea_hires', 'steps',
            e.target.value === '' ? dflt('steps') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          0 inherits the first pass&rsquo;s count, which is the sane default: at a denoise
          below 1 the pass only walks part of the schedule anyway.
        </p>
        <ResetToDefault label="Second-pass steps" section="krea_hires" field="steps" {...reset} />
      </div>
    </Card>
  )
}

/* SeedVR2 — the FIDELITY upscaler (issue #32, requested by SurpassHR).

   It is not a generation engine and deliberately does not appear in the enabled-
   engines list above: nothing in the variation catalog can be produced by it. It
   is the OTHER way to run Upscale & improve — the one that resolves detail
   without reinterpreting it — so its settings live next to the engines that feed
   the same pass, not in a section of their own. */
function SeedVr2Card({ config, setField, configDefaults, caps }) {
  const svr = config.seedvr2 || {}
  const improve = config.improve || {}
  const reset = { config, configDefaults, setField }
  const dflt = (key) => defaultValueAt(configDefaults, 'seedvr2', key)
  const comfy = (caps && caps.comfyui) || {}
  const ready = comfy.seedvr2_ready === true
  const [models, setModels] = useState(null)
  // Which builds are ON DISK — asked once per readiness change, never polled:
  // it is a directory listing, and the card is not a monitor.
  useEffect(() => {
    let live = true
    fetch('/api/seedvr2/models')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (live) setModels(d) })
      .catch(() => { if (live) setModels(null) })
    return () => { live = false }
  }, [ready])
  const installed = (models && models.installed) || []
  const catalog = (models && models.catalog) || []
  // Every file in the SEEDVR2 folder, split on whether its NAME looks like a
  // VAE. The unlikely ones are still offered, in their own group and labelled:
  // the pin exists for the install whose VAE is named something the automatic
  // path cannot recognise, and hiding those files would leave that install with
  // a picker it cannot use.
  const vaeChoices = (models && models.vae_choices) || []
  const vaeLikely = vaeChoices.filter((v) => v.likely_vae)
  const vaeOther = vaeChoices.filter((v) => !v.likely_vae)
  const tilePx = Number(svr.tile_px ?? dflt('tile_px')) || SEEDVR2_TILE_MIN
  // What 'auto' will actually use as its crossover: the explicit threshold, or
  // the tile side x1.5 — the same arithmetic the server does.
  const tileAbove = Number(svr.tile_threshold ?? dflt('tile_threshold')) > 0
    ? Number(svr.tile_threshold ?? dflt('tile_threshold'))
    : Math.round(tilePx * SEEDVR2_TILE_ABOVE_FACTOR)
  // ...and what that means for the target actually configured. The crossover is
  // strict AND derived (1.5x the tile), so it lands exactly on round numbers
  // people type — a target sitting on it ran whole with nothing said anywhere.
  const laneLine = laneForTarget(svr.tiling ?? dflt('tiling'),
    Number(svr.resolution ?? dflt('resolution')), tileAbove)
  return (
    <Card
      id="seedvr2-engine"
      title="SeedVR2 upscaling (local)"
      help="The fidelity half of Upscale & improve. Klein re-renders detail from a prompt — sharper, but skin and colour can shift; SeedVR2 resolves detail at a higher resolution and leaves the original look alone. Pick it per batch from the bulk actions in the dataset workspace, or make it the default for the single-image pass below. It needs the ComfyUI-SeedVR2_VideoUpscaler node pack in ComfyUI plus two model files — Setup ▸ ComfyUI downloads the models and says what is missing."
    >
      <p className={ready ? 'text-[0.6875rem] text-emerald-300' : 'text-[0.6875rem] text-amber-300'}>
        {ready
          ? 'Ready — SeedVR2 appears in the workspace bulk actions.'
          : 'Not ready yet. Setup ▸ ComfyUI lists what is missing and can download the weights; the node pack itself is installed from ComfyUI (search “SeedVR2” in ComfyUI-Manager), then restart ComfyUI.'}
      </p>

      <p className="mt-1 text-[0.6875rem] text-content-subtle">
        <a href="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler" target="_blank"
          rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Node pack →</a>
        {' · '}
        <a href="https://huggingface.co/numz/SeedVR2_comfyUI" target="_blank"
          rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Model weights →</a>
        {' · '}
        <a href="https://github.com/ByteDance-Seed/SeedVR" target="_blank"
          rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">SeedVR2 by ByteDance-Seed →</a>
        {' — all Apache-2.0.'}
      </p>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="improve-engine" className="block text-xs font-medium text-content">
          Default engine for Upscale &amp; improve
        </label>
        <select
          id="improve-engine"
          value={improve.engine ?? defaultValueAt(configDefaults, 'improve', 'engine')}
          onChange={(e) => setField('improve', 'engine', e.target.value)}
          className={INPUT_CLASS}
        >
          <option value="klein">Klein — re-renders detail (can shift skin and colour)</option>
          <option value="seedvr2">SeedVR2 — resolves detail, keeps the original look</option>
        </select>
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Used by the button on a single tile and by ↻ Re-improve. Bulk runs always
          state their engine on the button you press, so this never decides a batch
          behind your back.
        </p>
        <ResetToDefault label="Default improve engine" section="improve" field="engine"
          config={config} configDefaults={configDefaults} setField={setField} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-model" className="block text-xs font-medium text-content">
          Model build (optional)
        </label>
        <select
          id="seedvr2-model"
          value={svr.model ?? ''}
          onChange={(e) => setField('seedvr2', 'model', e.target.value)}
          className={INPUT_CLASS}
        >
          <option value="">auto — the 3B FP8 build, or whatever is installed</option>
          {installed.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Only builds already in your ComfyUI&rsquo;s <code>models/SEEDVR2</code> folder are
          listed: the pack&rsquo;s loader downloads an unknown name on first use, and a
          dropdown must not start a multi-gigabyte download. To use another build, put the
          file in that folder — it then appears here.
        </p>
        {catalog.length > 0 && (
          <ul className="mt-1 space-y-0.5 text-[0.6875rem] text-content-subtle">
            {catalog.map((v) => (
              <li key={v.file}>
                {v.installed ? '✓' : '·'} <b>{v.label}</b> — {v.size_gb} GB, ~{v.vram_gb} GB
                {' '}VRAM{v.recommended ? ' (recommended)' : ''}
              </li>
            ))}
          </ul>
        )}
        <ResetToDefault label="Model build" section="seedvr2" field="model" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-vae" className="block text-xs font-medium text-content">
          VAE build (optional)
        </label>
        <select
          id="seedvr2-vae"
          value={svr.vae ?? ''}
          onChange={(e) => setField('seedvr2', 'vae', e.target.value)}
          className={INPUT_CLASS}
        >
          <option value="">auto — ema_vae_fp16, or the first VAE in the folder</option>
          {vaeLikely.map((v) => <option key={v.file} value={v.file}>{v.file}</option>)}
          {vaeOther.length > 0 && (
            <optgroup label="Other files in models/SEEDVR2 (not named like a VAE)">
              {vaeOther.map((v) => <option key={v.file} value={v.file}>{v.file}</option>)}
            </optgroup>
          )}
        </select>
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Leave it on auto unless your VAE file is named something with no
          &ldquo;vae&rdquo; in it — that is the only case the automatic search misses, and
          the reason the second group above is offered at all. Picking a DiT build here
          fails inside the loader node, so choose from that group only if you know the
          file is a VAE.
        </p>
        <ResetToDefault label="VAE build" section="seedvr2" field="vae" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-tiling" className="block text-xs font-medium text-content">
          High-resolution tiling
        </label>
        <select
          id="seedvr2-tiling"
          value={svr.tiling ?? dflt('tiling')}
          onChange={(e) => setField('seedvr2', 'tiling', e.target.value)}
          className={INPUT_CLASS}
        >
          <option value="auto">Tile when it helps (recommended)</option>
          <option value="always">Always tile large frames</option>
          <option value="never">Never tile</option>
        </select>
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Needs the <code>Comfyui_TTP_Toolset</code> node pack; without it this has no
          effect. Tiling is not only about memory: a tile is upscaled at the size the
          model works well at, so a large frame keeps far more fine detail than one
          processed whole — contributed and measured by SurpassHR (GitHub&nbsp;#32).
          On <b>Tile when it helps</b> nothing is tiled at or below {tileAbove} px on the
          short edge: the model is already in its comfortable range there and a grid would
          only add seams. <b>Always</b> tiles any frame bigger than one tile; pick{' '}
          <b>never</b> if you ever see a seam.
        </p>
        {laneLine && (
          <p className="mt-1 text-[0.6875rem] text-sky-300">{laneLine}</p>
        )}
        <ResetToDefault label="High-resolution tiling" section="seedvr2" field="tiling" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-tile-px" className="block text-xs font-medium text-content">
          Tile size (px)
        </label>
        <input
          id="seedvr2-tile-px"
          type="number"
          min={SEEDVR2_TILE_MIN}
          max={SEEDVR2_TILE_MAX}
          step={64}
          value={svr.tile_px ?? dflt('tile_px')}
          onChange={(e) => setField('seedvr2', 'tile_px',
            e.target.value === '' ? dflt('tile_px') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          The memory dial of this engine: a run holds one tile at a time, so
          <b> lower it if upscales run out of VRAM</b> (768 or 512 on an 8 GB card) and
          raise it on a big card for fewer seams and more context per tile.
          {' '}{dflt('tile_px')} px is the contributed default. It also sizes the model&rsquo;s
          own tiled encode/decode, so it helps even <i>without</i> the tiling node
          pack — this is the one setting worth touching before giving up on a large upscale.
        </p>
        <ResetToDefault label="Tile size" section="seedvr2" field="tile_px" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-tile-threshold" className="block text-xs font-medium text-content">
          Start tiling above (px on the short edge, 0 = automatic)
        </label>
        <input
          id="seedvr2-tile-threshold"
          type="number"
          min={0}
          max={SEEDVR2_MAX_RESOLUTION_MAX}
          step={64}
          value={svr.tile_threshold ?? dflt('tile_threshold')}
          onChange={(e) => setField('seedvr2', 'tile_threshold',
            e.target.value === '' ? dflt('tile_threshold') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Where <b>Tile when it helps</b> switches over. <b>0</b> (default) follows the tile
          size — {SEEDVR2_TILE_ABOVE_FACTOR}&times; it, so {tilePx} px tiles start tiling
          above {Math.round(tilePx * SEEDVR2_TILE_ABOVE_FACTOR)} px. Set a number to place
          the crossover yourself: lower it to tile sooner (safer on a small card), raise it
          to keep more targets in one fast pass. It has no effect on <b>always</b> or
          <b> never</b>.
        </p>
        <ResetToDefault label="Start tiling above" section="seedvr2" field="tile_threshold" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-resolution" className="block text-xs font-medium text-content">
          Target resolution (short edge, px)
        </label>
        <input
          id="seedvr2-resolution"
          type="number"
          min={SEEDVR2_RESOLUTION_MIN}
          max={SEEDVR2_RESOLUTION_MAX}
          step={2}
          value={svr.resolution ?? dflt('resolution')}
          onChange={(e) => setField('seedvr2', 'resolution',
            e.target.value === '' ? dflt('resolution') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          The SHORT edge is scaled to this and the aspect ratio is kept, so 1080 on a 3:2
          photo gives 1620&times;1080. LoRA training buckets rarely go above 1024&ndash;1280,
          so higher mostly costs VRAM and time.
        </p>
        <ResetToDefault label="Target resolution" section="seedvr2" field="resolution" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-max-resolution" className="block text-xs font-medium text-content">
          Maximum long edge (px, 0 = no limit)
        </label>
        <input
          id="seedvr2-max-resolution"
          type="number"
          min={0}
          max={SEEDVR2_MAX_RESOLUTION_MAX}
          step={2}
          value={svr.max_resolution ?? dflt('max_resolution')}
          onChange={(e) => setField('seedvr2', 'max_resolution',
            e.target.value === '' ? dflt('max_resolution') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          The safety valve on a wide crop: at a 1080 short edge a 4:1 panorama becomes
          4320 px across, which is where a run runs out of VRAM.
        </p>
        <ResetToDefault label="Maximum long edge" section="seedvr2" field="max_resolution" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-color" className="block text-xs font-medium text-content">
          Colour correction
        </label>
        <select
          id="seedvr2-color"
          value={svr.color_correction ?? dflt('color_correction')}
          onChange={(e) => setField('seedvr2', 'color_correction', e.target.value)}
          className={INPUT_CLASS}
        >
          {SEEDVR2_COLOR_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          How the result is graded back onto the source&rsquo;s colours. <b>lab</b> is the
          model&rsquo;s own default and the most conservative; <b>wavelet</b> holds broad tone
          better on heavily degraded sources; <b>none</b> shows the raw output. Colour
          fidelity is the reason this engine exists, so it is worth trying both ways on one
          image before a big batch.
        </p>
        <ResetToDefault label="Colour correction" section="seedvr2" field="color_correction" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="seedvr2-swap" className="block text-xs font-medium text-content">
          Blocks offloaded to system RAM
        </label>
        <input
          id="seedvr2-swap"
          type="number"
          min={0}
          max={SEEDVR2_BLOCKS_MAX}
          step={1}
          value={svr.blocks_to_swap ?? dflt('blocks_to_swap')}
          onChange={(e) => setField('seedvr2', 'blocks_to_swap',
            e.target.value === '' ? dflt('blocks_to_swap') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          0 = none, and fastest. Raise it to fit a bigger build on a smaller card: it trades
          speed for VRAM headroom and does not change the result.
        </p>
        <ResetToDefault label="Blocks offloaded" section="seedvr2" field="blocks_to_swap" {...reset} />
      </div>

      <p className="mt-3 text-[0.6875rem] text-content-subtle">
        <b>No batch size here, on purpose.</b> SeedVR2&rsquo;s batch size is a <i>video</i> window
        whose frames share attention to stay coherent — feeding it unrelated photos would let
        them bleed into each other. Dataset images are upscaled one per job; the throughput
        comes from the normal generation queue.
      </p>
    </Card>
  )
}

/* The finishing pass — three pixel operations the APP runs on the ✨ improve
   result, after ComfyUI is done with it.

   Its own card rather than fields on the two engine cards above, because it is
   one thing that happens once, to whichever engine ran. The reference workflow
   these come from does them with three third-party node packs; doing them here
   keeps the installer's rule intact (models, never node packs) and makes them
   deterministic — no GPU, no model, no graph. */
const IMPROVE_SHARPEN_MAX = 1.5
const IMPROVE_GRAIN_MAX = 0.05
// The values the reference workflow uses, shown as the "known-good" landing spot
// rather than shipped as defaults — this ships off.
const IMPROVE_REFERENCE = { colour_match: 0.8, sharpen: 0.55, grain: 0.01 }

function ImproveFinishCard({ config, setField, configDefaults }) {
  const improve = config.improve || {}
  const reset = { config, configDefaults, setField }
  const dflt = (key) => defaultValueAt(configDefaults, 'improve', key)
  const num = (key) => Number(improve[key] ?? dflt(key)) || 0
  const colour = num('colour_match')
  const sharpen = num('sharpen')
  const grain = num('grain')
  const grainSat = num('grain_saturation')
  const engine = improve.engine ?? dflt('engine')
  const anyOn = colour > 0 || sharpen > 0 || grain > 0
  return (
    <Card
      id="improve-finish"
      title="Finishing pass (after Upscale &amp; improve)"
      help="Three small operations applied to the finished image by the app, not by ComfyUI: put the source's colours back, sharpen the finest detail, and add a little film grain. They are what separate a render that looks processed from one that looks photographed. Nothing to install — no model, no GPU, no node pack. All three ship off; the image you get today is byte-for-byte the one ComfyUI wrote."
    >
      {!anyOn && (
        <p className="rounded border border-border-subtle bg-surface-subtle px-2 py-1.5 text-[0.6875rem] text-content-subtle">
          Every stage is off — the finished image is left exactly as ComfyUI wrote it.
          The reference workflow this was ported from runs colour match{' '}
          {IMPROVE_REFERENCE.colour_match}, sharpen {IMPROVE_REFERENCE.sharpen}, grain{' '}
          {IMPROVE_REFERENCE.grain}.
        </p>
      )}

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="improve-colour-match" className="block text-xs font-medium text-content">
          Put the source&rsquo;s colours back ({colour === 0 ? 'off' : colour})
        </label>
        <input
          id="improve-colour-match"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={colour}
          onChange={(e) => setField('improve', 'colour_match', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {engine === 'seedvr2' ? (
            <>
              Applies to the passes <b>Klein</b> ran — the engine is decided per pass
              (the single-tile button starts on this default, the bulk buttons name
              theirs), and a SeedVR2 pass skips this automatically: it already grades
              its result back onto the source inside the node (Colour correction,
              above), and doing it twice would be two corrections estimated from the
              same statistics, fighting each other. {IMPROVE_REFERENCE.colour_match}{' '}
              is the reference value; 1 forbids a Klein pass any colour change at all.
            </>
          ) : (
            <>
              A Klein improve keeps the content and loses the grade: skin warms or cools and
              a dataset ends up holding two colour worlds depending on which images went
              through. This measures the colours of the image as it was <b>before</b> the
              pass and puts them back. {IMPROVE_REFERENCE.colour_match} is the reference
              value; 1 forbids the pass any colour change at all.
            </>
          )}
        </p>
        <ResetToDefault label="Put the source's colours back" section="improve"
          field="colour_match" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="improve-sharpen" className="block text-xs font-medium text-content">
          Sharpen ({sharpen === 0 ? 'off' : sharpen})
        </label>
        <input
          id="improve-sharpen"
          type="range"
          min={0}
          max={IMPROVE_SHARPEN_MAX}
          step={0.05}
          value={sharpen}
          onChange={(e) => setField('improve', 'sharpen', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Local contrast at a 1&nbsp;px radius — the finest octave, the one diffusion leaves
          empty — not a global sharpness slider. {IMPROVE_REFERENCE.sharpen} is the reference
          value; past about 1 the halo starts reading as an outline.
        </p>
        <ResetToDefault label="Sharpen" section="improve" field="sharpen" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="improve-grain" className="block text-xs font-medium text-content">
          Film grain ({grain === 0 ? 'off' : grain})
        </label>
        <input
          id="improve-grain"
          type="range"
          min={0}
          max={IMPROVE_GRAIN_MAX}
          step={0.002}
          value={grain}
          onChange={(e) => setField('improve', 'grain', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Diffusion output is smooth in a way photographs are not — the model resolves
          structure confidently and leaves the finest octave nearly empty. A very small
          amount of noise refills it and the eye reads the result as a photograph.
          The scale is deliberately tiny: {IMPROVE_REFERENCE.grain} is about ±2.5 levels of
          an 8-bit image, seen as texture and never as noise.
        </p>
        <ResetToDefault label="Film grain" section="improve" field="grain" {...reset} />
      </div>

      {/* Not greyed on `grain === 0`: this dial also serves the Studio's PER-RUN
          grain (the cell carries its own amount, the colour mix follows this
          setting), so it has to stay editable while the improve pass's grain is
          off. */}
      <div className="mt-3 sm:max-w-md">
        <label htmlFor="improve-grain-sat" className="block text-xs font-medium text-content">
          How coloured that grain is ({grainSat})
        </label>
        <input
          id="improve-grain-sat"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={grainSat}
          onChange={(e) => setField('improve', 'grain_saturation', Number(e.target.value))}
          className="mt-1 w-full accent-violet-500"
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          0 puts identical noise on the three channels, which is what film does and what
          reads as grain. 1 makes each channel independent, which reads as sensor noise.
        </p>
        <ResetToDefault label="How coloured that grain is" section="improve"
          field="grain_saturation" {...reset} />
      </div>
    </Card>
  )
}

/* Krea's own always-on LoRA presets. Same shape as the Klein card — the two lanes
   are deliberate copies — with two differences that matter: the strength ceiling
   opens to 20 for utility LoRAs (the bypass ones do nothing below ~10), and the
   picker judges compatibility against the KREA graph, so a Klein LoRA is badged
   incompatible here instead of compatible. */
function KreaLorasCard({ config, setField }) {
  const presets = Array.isArray(config.krea?.generation_lora_presets)
    ? config.krea.generation_lora_presets : []
  const save = (next) => setField('krea', 'generation_lora_presets', next)
  // ONE scan per card, judged for Krea. Degrades to free text — see the hook.
  const loraScan = useKleinGenerationLoras('krea')
  return (
    <Card
      id="krea-generation-lora-presets"
      title="Krea 2 Edit generation LoRA presets (optional)"
      help={`Named combinations of your own LoRA files, chained after the identity-edit LoRA when Krea 2 Edit generates dataset images — inside a preset the order is the chain order (max ${MAX_GENERATION_LORAS} LoRAs each, ${MAX_GENERATION_LORA_PRESETS} presets). Pick each row from the LoRAs found under ComfyUI's models/loras; Krea-compatible ones are listed first, and a LoRA of another architecture is badged because ComfyUI would load it as a silent no-op here. Strength goes to 6, or to 20 for utility LoRAs whose filename says filter-bypass — those have no effect below ~10. Per run, pick a preset in the workspace's Krea 2 Edit tuning panel — it opens on the default preset chosen below ("None" until you choose one), and picking something else there applies to that run only. Only the model side is patched, so a LoRA's text-encoder weights are ignored. Preset mechanism by @waltm (Discord).`}
    >
      {presets.length === 0 && (
        <p className="text-sm text-content-muted">No presets yet — create your first combination below.</p>
      )}
      {presets.map((preset, i) => (
        <LoraPresetCard key={i} preset={preset} index={i} presets={presets} save={save}
          loraScan={loraScan} engineLabel="Krea 2"
          strengthRange={kreaStrengthRange} defaultStrength={KREA_LORA_STRENGTH_DEFAULT}
          placeholder="krea/my-lora.safetensors"
          engineId="krea" fixedLora={config.krea?.identity_lora || ''} />
      ))}
      <div className="flex items-center gap-3">
        <button
          type="button" className={TEXT_BTN}
          onClick={() => save([...presets, { name: freeName(presets, 'My preset'), loras: [] }])}
          disabled={presets.length >= MAX_GENERATION_LORA_PRESETS}
        >
          ＋ New preset
        </button>
        <span className="text-xs text-content-muted">{presets.length}/{MAX_GENERATION_LORA_PRESETS}</span>
      </div>
      <DefaultPresetField
        id="krea-default-lora-preset" engineLabel="Krea 2 Edit" presets={presets}
        value={config.krea?.default_generation_lora_preset || ''}
        onChange={(v) => setField('krea', 'default_generation_lora_preset', v)} />
    </Card>
  )
}

/* Editable identity / quality prompts (feature request by @bbsorry / 雨田壹).
   The identity "locks" that ride ahead of every generated variation used to be
   hardcoded and invisible; here each is an override shown in ONE editable box
   that already holds the shipped default text, with a Reset — one set PER
   SUBJECT TYPE, picked with the chips at the top of the card.

   The two-box era is over: the field used to be an empty textarea next to a
   read-only copy of the shipped text and a button that pasted it in. One box is
   clearer, but it must not turn "I looked at the default" into a persisted COPY
   of it — that would freeze the prompt for that user and hide every future
   improvement. PromptOverrideField normalises the text back to '' whenever it
   equals the shipped default, so blank-means-default (the backend contract in
   face_variations.get_identity_prompt) still holds.

   The Klein-improve prompt additionally has an on/off toggle: off applies NO
   prompt to the manual "Klein upscale & improve".
   Field metadata (keys mirroring config identity_prompts.*, never renamed) lives
   in common/promptOverride.js, shared with the workspace's Extra-refs modal. */

// Bounds mirror the server-side clamps in face_dataset_service._improve_float /
// _improve_int — the UI should not offer a value the backend will silently pull back.
// The `fallback` numbers this list used to carry (2 / 0 / 0 / 4) are gone: they
// were a hand-kept copy of config.DEFAULTS['klein'], and one of them (0 for the
// consistency strength) had ALREADY drifted from the backend's 1.0. Both the
// displayed value and "Reset to default" now read the server's `config_defaults`.
const IMPROVE_KNOBS = [
  { key: 'improve_megapixels', label: 'Output size (MP)',
    min: 0.5, max: 8, step: 0.5,
    hint: 'The result’s resolution.' },
  { key: 'improve_base_lora_strength', label: 'Enhancement LoRA',
    min: 0, max: 2, step: 0.05,
    hint: '0 = off (the shipped behaviour). Try 0.5–0.8. Needs klein/realistic.safetensors.' },
  // Drives klein.consistency_strength, which enqueue_klein_edit clamps to 1.5 — the
  // UI must not offer a value the engine pulls back. It anchors COMPOSITION, not
  // identity: it was mislabelled "Character LoRA" when these knobs first shipped.
  { key: 'improve_consistency_strength', label: 'Consistency LoRA',
    min: 0, max: 1.5, step: 0.05,
    hint: 'Holds the composition and background. High values resist the edit.' },
  { key: 'improve_steps', label: 'Steps',
    min: 1, max: 50, step: 1, hint: 'More steps = slower, usually cleaner.' },
]

function IdentityPromptsCard({ config, setField, promptDefaults, promptDefaultsBySubject,
                               setIdentityPrompts, configDefaults }) {
  const ip = config.identity_prompts || {}
  const kleinDefault = (key) => defaultValueAt(configDefaults, 'klein', key)
  // Subject type being edited. This screen has NO dataset context, so without an
  // explicit picker it edited "the" identity prompt — which is exactly how an
  // animal-tuned lock ended up on human generations (ashish.sinha, Discord).
  // Human first: it is the default subject and the one the flat legacy keys hold.
  const [subject, setSubject] = useState('human')
  const defaults = (promptDefaultsBySubject || {})[subject] || promptDefaults || {}
  const set = (key, v) => setField('identity_prompts', key, v)
  const setPrompt = (key, v) => setIdentityPrompts((prev) => writeIdentityPrompt(prev, subject, key, v))
  const improveEnabled = ip.klein_improve_enabled !== false
  return (
    <Card
      id="identity-prompts"
      title="Identity, Klein & Krea 2 prompts (advanced)"
      help="The hidden prompts that lock a subject's identity across generated variations, now editable. Pick the subject type first: each type (Human, Animal, Creature, Object, Other, Anime) has its OWN set, and a text you write for one never applies to another. Each box already holds the prompt in use: edit it to override, Reset to go back. Reproducibility note: as long as a box still matches the built-in text, nothing is stored and generation stays byte-identical to before — you also keep receiving improvements to that prompt. Feature request by @bbsorry (雨田壹); per-subject scoping reported by ashish.sinha."
    >
      {/* flex-wrap: five chips fit one row on a laptop and wrap to two or three
          on a phone — never a row that overflows the card. */}
      <div>
        <span className="block text-sm font-medium text-content">Subject type</span>
        <p className="mt-1 mb-2 text-xs text-content-muted">
          Which datasets these three prompts apply to. Each subject type keeps its own texts —
          editing the Animal ones leaves your Human datasets untouched. A dot marks a type you
          have already customised.
        </p>
        <div role="group" aria-label="Subject type to edit" className="flex flex-wrap gap-1.5">
          {PROMPT_SUBJECT_TYPES.map((st) => {
            const on = st === subject
            return (
              <button
                key={st}
                type="button"
                aria-pressed={on}
                onClick={() => setSubject(st)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs ${
                  on ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-200 font-semibold'
                     : 'border-border bg-surface text-content-muted hover:text-content'}`}
              >
                {SUBJECT_TYPE_LABELS[st]}
                {subjectHasOverride(ip, st) && (
                  <span aria-label="customised" title="Customised" className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                )}
              </button>
            )
          })}
        </div>
      </div>

      {identityPromptFields(subject).map((f) => (
        <PromptOverrideField
          key={`${subject}-${f.key}`}
          id={f.id}
          label={f.label}
          desc={f.desc}
          value={readIdentityPrompt(ip, subject, f.key)}
          defaultText={defaults[f.key]}
          onChange={(v) => setPrompt(f.key, v)}
        />
      ))}

      {/* The identity locks were one of SIX sources the prompt is built from.
          The other five shipped hardcoded and invisible; they are edited here,
          split the same way the storage is — per subject above the line, global
          below it. The composed preview closes the card, because the whole point
          of these boxes is to change a part and see the whole move. */}
      <div id="prompt-part-render-tail" className="border-t border-border pt-4">
        <h4 className="text-sm font-medium text-content">
          Klein &amp; Krea — the rest of the prompt ({SUBJECT_TYPE_LABELS[subject]})
        </h4>
        <p className="mt-1 mb-3 text-xs text-content-muted">
          These follow the subject type selected above, like the identity locks: the tail asks
          an Anime dataset for a drawing and every other type for a photograph.
        </p>
        {SUBJECT_PROMPT_PART_FIELDS.map((f) => (
          <PromptOverrideField
            key={`${subject}-${f.key}`}
            id={f.id}
            label={f.label}
            desc={f.desc}
            warn={f.warn}
            rows={f.rows}
            value={readIdentityPrompt(ip, subject, f.key)}
            defaultText={defaults[f.key]}
            onChange={(v) => setPrompt(f.key, v)}
            className="mt-3"
          />
        ))}
      </div>

      <div id="prompt-part-framing" className="border-t border-border pt-4">
        <h4 className="text-sm font-medium text-content">
          Shot detail per framing ({SUBJECT_TYPE_LABELS[subject]})
        </h4>
        <p className="mt-1 mb-1 text-xs text-content-muted">
          Klein and Krea under-fill a short tag prompt and invent the rest, so each shot carries
          a concrete description of what the framing should look like. This is where the lens
          talk (&ldquo;85mm portrait lens look&rdquo;) lives.
        </p>
        {/* Four boxes: two columns on a laptop, stacked on a phone. */}
        <div className="grid gap-3 sm:grid-cols-2">
          {FRAMING_PROMPT_PART_FIELDS.map((f) => (
            <PromptOverrideField
              key={`${subject}-${f.key}`}
              id={f.id}
              label={f.label}
              rows={f.rows}
              value={readIdentityPrompt(ip, subject, f.key)}
              defaultText={defaults[f.key]}
              onChange={(v) => setPrompt(f.key, v)}
              className="mt-2"
            />
          ))}
        </div>
      </div>

      <div id="prompt-part-global" className="border-t border-border pt-4">
        <h4 className="text-sm font-medium text-content">Applied to every subject type</h4>
        <p className="mt-1 mb-1 text-xs text-content-muted">
          These four are <strong>not</strong> per subject type: the two directives are only ever
          injected into human shots, and the skin hold is one sentence about not inventing
          detail. Editing them here changes them everywhere.
        </p>
        {GLOBAL_PROMPT_PART_FIELDS.map((f) => (
          <PromptOverrideField
            key={f.key}
            id={f.id}
            label={f.label}
            desc={f.desc}
            warn={f.warn}
            rows={f.rows}
            value={ip[f.key]}
            defaultText={defaults[f.key]}
            onChange={(v) => set(f.key, v)}
            className="mt-3"
          />
        ))}
      </div>

      <PromptPreview subject={subject} identityPrompts={ip} />

      <div className="border-t border-border pt-4">
        {/* The second sentence is the honest half. The default asks for
            PHOTOGRAPHIC detail, and the app does not vary it by subject type, so
            on a drawn dataset it works against the anime lock every other prompt
            here enforces. The default is deliberately left as-is — people have
            calibrated their results on it — but saying nothing turned that into
            "the tool ruins my anime" (Qeeyana, Reddit). */}
        <p className="mb-2 text-xs text-content-subtle">
          The prompt below is <strong>not</strong> per subject type — it asks for texture and
          detail, which means the same thing for a person, a dog or a car.{' '}
          <span className="text-amber-300">
            It does <strong>not</strong> mean the same thing for a drawing: the built-in text asks for
            photographic detail, so on an Anime dataset it pushes skin and fabric towards realism.
            Rewrite it below, or untick the box above to upscale with no prompt at all.
          </span>
        </p>
        <label htmlFor="identity-prompt-klein-improve-enabled" className="flex items-center gap-2 text-sm font-medium text-content">
          <input
            id="identity-prompt-klein-improve-enabled"
            type="checkbox"
            checked={improveEnabled}
            onChange={(e) => set('klein_improve_enabled', e.target.checked)}
            className="h-4 w-4 rounded border-border-strong"
          />
          Apply an improvement prompt on “Klein upscale &amp; improve”
        </label>
        <PromptOverrideField
          id="identity-prompt-klein-improve"
          label="Klein upscale & improve prompt"
          desc="The fixed instruction the manual “Klein upscale & improve” action sends to add texture and detail. Turn the checkbox above off to upscale with no prompt at all (pure enhancement)."
          rows={3}
          value={ip.klein_improve}
          defaultText={defaults.klein_improve}
          onChange={(v) => set('klein_improve', v)}
          disabled={!improveEnabled}
          className="mt-2"
        />
        {!improveEnabled && (
          <p className="mt-1 text-xs text-content-subtle">Disabled — no prompt is applied.</p>
        )}
        <p className="mt-3 text-xs text-content-subtle">
          Separate from the scraper rescue prompt for small images — see Settings ▸ Scraping ▸ “Klein rescue — small scraped images”.
        </p>
      </div>

      {/* The instruction above was already editable, but the knobs deciding how
          much the pass actually changes were hardcoded — including both LoRA
          strengths at 0, which meant the workflow's own realistic LoRA never
          applied. Defaults here are those historical values. */}
      {/* id spelled out literally: it is the deep-link target of the lightbox's
          "Adjust improve strength →" link, and the contract tests find targets by
          scanning this file for id="…". The BLOCK is the target, not one knob:
          "strength" here is the four values together, and ringing the group is
          the honest answer to what that label promises. */}
      <div id="klein-improve-strength" className="scroll-mt-24 border-t border-border pt-4">
        <h4 className="text-sm font-medium text-content">Upscale &amp; improve — strength</h4>
        <p className="mt-1 mb-2 text-xs text-content-muted">
          Output resolution, and how much the pass is allowed to change the image. All four
          start at the values the action used before they were exposed, so leaving them alone
          keeps today’s result.
        </p>
        <p className="mb-2 text-xs text-content-muted">
          The <strong>enhancement LoRA</strong> needs its weights file
          (<code>klein/realistic.safetensors</code>): without it that node is skipped and the
          strength changes nothing. Setup downloads it with the other Klein assets — if the
          slider seems to do nothing, run <strong>Install everything</strong> there first.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {IMPROVE_KNOBS.map((k) => (
            <div key={k.key}>
              <label htmlFor={`klein-${k.key}`} className="block text-xs font-medium text-content">
                {k.label}
              </label>
              <input
                id={`klein-${k.key}`}
                type="number"
                min={k.min}
                max={k.max}
                step={k.step}
                value={config.klein?.[k.key] ?? kleinDefault(k.key)}
                onChange={(e) => setField('klein', k.key,
                  e.target.value === '' ? kleinDefault(k.key) : Number(e.target.value))}
                className={INPUT_CLASS}
              />
              <p className="mt-1 text-[0.6875rem] text-content-subtle">
                {k.hint} Default {String(kleinDefault(k.key))}.
              </p>
              <ResetToDefault label={k.label} section="klein" field={k.key}
                config={config} configDefaults={configDefaults} setField={setField} />
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

/* One model field per API engine (feature request — the OpenRouter
   engine shipped with a free-text model while Nano Banana and ChatGPT were
   frozen to whatever the release hardcoded, overridable only by an environment
   variable nobody could see from the app).

   Free text on all three, deliberately: providers ship image models far faster
   than this app ships releases, so a dropdown baked into a build would be stale
   the day it landed and would lock people out of a model that works.

   Blank = the historical default, so a field appearing changes nobody's result.
   The resolution order is documented next to each backend engine:
   setting > environment variable > built-in default — a NANOBANANA_MODEL /
   CHATGPT_IMAGE_MODEL exported before these fields existed is still honoured and
   is only overridden when someone actually types a slug here.

   One card rather than three: the three fields answer the same question, and on
   a phone three cards of one input each is a lot of scrolling for very little. */
/* One field. `id` is spelled out literally at each call site on purpose: the
   help-registry contract test scans these files for `id="…"` to prove every
   help topic's focus anchor exists, and a template-built id would be invisible
   to it (and to anyone grepping for the anchor). */
function ModelField({ id, configKey, label, placeholder, config, setField, configDefaults, children }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-content">{label}</label>
      <input
        id={id}
        type="text"
        spellCheck="false"
        autoComplete="off"
        autoCapitalize="off"
        value={config.engines[configKey] ?? ''}
        onChange={(e) => setField('engines', configKey, e.target.value)}
        placeholder={placeholder}
        className={INPUT_CLASS}
      />
      <p className="mt-1 text-xs text-content-muted">{children}</p>
      {/* Two of these three default to BLANK — "let the engine pick", which also
          keeps a pre-existing NANOBANANA_MODEL / CHATGPT_IMAGE_MODEL environment
          variable in charge. Reset writes the shipped default back, so on those
          two it empties the field instead of typing a slug in: it hands the
          implicit state back rather than pinning today's model forever. */}
      <ResetToDefault label={label} section="engines" field={configKey}
        config={config} configDefaults={configDefaults} setField={setField} />
    </div>
  )
}

function ImageModelsCard({ config, setField, configDefaults }) {
  const shared = { config, setField, configDefaults }
  return (
    <Card
      id="engine-image-models"
      title="Image models"
      help="Which model each API engine asks for. Free text on purpose: providers publish new image models far faster than this app publishes releases, and a fixed menu would be out of date the day it shipped. Leave a field blank to keep the model the engine has always used — an empty field changes nothing about your results. All three must accept REFERENCE IMAGES: the dataset generator always sends your reference photos with the prompt, so a text-to-image-only model cannot work here; when a provider refuses one, the failed tile names the model and the provider's own reason instead of blaming your prompt, and the run stops rather than paying for the same refusal once per image."
    >
      <ModelField {...shared} id="engines-nanobanana_model" configKey="nanobanana_model"
        label="Nano Banana (Gemini) model" placeholder="gemini-3-pro-image">
        Blank = <code className="break-all">gemini-3-pro-image</code>, the model this engine has
        always used. Any Gemini <strong>image</strong> model that accepts image input works —
        browse them at{' '}
        <a href="https://ai.google.dev/gemini-api/docs/models" target="_blank" rel="noreferrer"
          className="text-primary underline">ai.google.dev</a>.
      </ModelField>

      <ModelField {...shared} id="engines-chatgpt_image_model" configKey="chatgpt_image_model"
        label="ChatGPT (OpenAI) image model" placeholder="gpt-image-2">
        Blank = <code className="break-all">gpt-image-2</code>, the model this engine has always
        used — and the only current one usable <strong>without OpenAI organization
        verification</strong>. Newer models (<code className="break-all">gpt-image-1.5</code>,{' '}
        <code className="break-all">chatgpt-image-latest</code>) answer 403 until your OpenAI
        organization is verified: if that happens your key is fine, the model is the problem.
        Applies to the API-key lane — the ChatGPT <em>subscription</em> lane renders on whatever
        image model your plan serves and ignores this field.
      </ModelField>

      <ModelField {...shared} id="engines-openrouter_model" configKey="openrouter_model"
        label="OpenRouter model slug" placeholder="google/gemini-3-pro-image">
        Blank = <code className="break-all">google/gemini-3-pro-image</code> — the same weights the
        Nano Banana engine calls, so switching engine changes who bills you, not the picture.
        Browse the list at{' '}
        <a href="https://openrouter.ai/models?output_modalities=image" target="_blank" rel="noreferrer"
          className="text-primary underline">openrouter.ai/models</a>.
      </ModelField>
      <p className="border-t border-border pt-3 text-xs text-content-subtle">
        A model must accept your reference photos. One that only takes text will either be
        refused — the tile then says which model and why — or quietly ignore the references and
        return a picture of someone else, which no app can detect for you. If generated faces stop
        looking like your subject after a model change, change it back.
      </p>
    </Card>
  )
}

const CHATGPT_AUTH_OPTIONS = [
  { id: 'auto', label: 'Auto — subscription when connected, otherwise API key' },
  { id: 'api', label: 'API key only' },
  { id: 'subscription', label: 'Subscription only' },
]

/* ChatGPT subscription (Codex OAuth) — EXPERIMENTAL lane. The device-code login
   itself is ChatgptSubscriptionConnect (components/common): the Setup wizard's
   image step offers the same connection, and two copies of an OAuth polling loop
   would be two chances for them to drift. This card adds what only Settings has:
   the auth-mode preference. */
function ChatgptSubscriptionCard({ caps, config, setField, refreshCaps, toast, configDefaults }) {
  return (
    <Card
      title="ChatGPT subscription (experimental)"
      help="Run the ChatGPT engine on your ChatGPT Plus/Pro image quota instead of a pay-per-use API key. Undocumented lane — it may stop working if OpenAI closes it. Limits vs API mode: up to 5 reference images (instead of 16), your plan's daily image cap applies, SFW only."
    >
      <ChatgptSubscriptionConnect caps={caps} refreshCaps={refreshCaps} toast={toast} />

      <div>
        <label htmlFor="chatgpt-auth-mode" className="block text-sm font-medium text-content">ChatGPT engine auth</label>
        <select
          id="chatgpt-auth-mode"
          value={config.engines.chatgpt_auth || 'auto'}
          onChange={(e) => setField('engines', 'chatgpt_auth', e.target.value)}
          className={INPUT_CLASS}
        >
          {CHATGPT_AUTH_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
        </select>
        <p className="mt-1 text-xs text-content-muted">
          When the subscription quota runs out mid-batch, remaining rows fail with a clear message — the app never silently switches to your paid API key.
        </p>
        <ResetToDefault label="ChatGPT engine auth" section="engines" field="chatgpt_auth"
          config={config} configDefaults={configDefaults} setField={setField} />
      </div>
    </Card>
  )
}

/* The section is organised as a clickable SUMMARY plus collapsible groups —
   eleven flat cards had grown into a wall where API keys sat next to Klein
   pins next to the improve prompt, and finding anything meant scrolling
   (reported from a tablet, mid preset editing). Which cards live in which
   group is data (settingsGroups.ENGINES_GROUPS), the shells are shared
   (SettingsGroupsView.jsx — NOT settingsGroups.jsx with a capital: a name
   differing only by case resolves to the wrong file on a Windows checkout),
   and deep-links keep working untouched: each group is
   a native <details>, which the ?focus= reveal already knows how to open. */
export default function EnginesSection(props) {
  const { config, setField, toggleEngine, caps, refreshCaps, toast, configDefaults } = props
  const [group1, group2, group3, group4, group5, group6] = ENGINES_GROUPS
  const groupProps = useSettingsGroupProps('engines')
  return (
    <div className="space-y-4">
      <SettingsGroupsToc sectionId="engines" groups={ENGINES_GROUPS} />

      <SettingsGroup {...groupProps(group1)}>
      <Card title="API keys" help="Keys are write-only — fields stay blank even when a key is already saved.">
        {ENGINE_SECRETS.map((f) => <SecretField key={f.key} field={f} {...props} />)}
      </Card>

      <ImageModelsCard config={config} setField={setField} configDefaults={configDefaults} />

      <ChatgptSubscriptionCard caps={caps} config={config} setField={setField} refreshCaps={refreshCaps}
        toast={toast} configDefaults={configDefaults} />

      <Card title="Engines" help="Which engines appear in the generate panel, and which one is preselected.">
        <div>
          <label htmlFor="engine-default" className="block text-sm font-medium text-content">Default engine</label>
          <select
            id="engine-default"
            value={config.engines.default}
            onChange={(e) => setField('engines', 'default', e.target.value)}
            className={INPUT_CLASS}
          >
            {ENGINE_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
          <ResetToDefault label="Default engine" section="engines" field="default"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>

        <fieldset id="engines-enabled" className="scroll-mt-24">
          <legend className="mb-1 block text-sm font-medium text-content">Enabled engines</legend>
          <div className="flex flex-col gap-2">
            {ENGINE_OPTIONS.map((o) => (
              <label key={o.id} htmlFor={`engine-enabled-${o.id}`} className="flex items-center gap-2 text-sm text-content">
                <input
                  id={`engine-enabled-${o.id}`}
                  type="checkbox"
                  checked={(config.engines.enabled || []).includes(o.id)}
                  onChange={() => toggleEngine(o.id)}
                  className="h-4 w-4 rounded border-border-strong"
                />
                {o.label}
              </label>
            ))}
          </div>
          {/* The only LIST with a reset. Ticking the boxes back one by one means
              knowing which five shipped enabled — and the catalog grows with
              releases, so that knowledge goes stale. Order is not compared: a
              re-ticked selection is the same selection. */}
          <ResetToDefault label="Enabled engines" section="engines" field="enabled"
            config={config} configDefaults={configDefaults} setField={setField} />
        </fieldset>
      </Card>
      </SettingsGroup>

      <SettingsGroup {...groupProps(group2)}>
      <KleinModelFilesCard config={config} setField={setField} caps={caps} />

      <KleinGenerationCard config={config} setField={setField} configDefaults={configDefaults} />
      </SettingsGroup>

      <SettingsGroup {...groupProps(group3)}>
      <KreaCard config={config} setField={setField} configDefaults={configDefaults} caps={caps} />

      <KreaHiresCard config={config} setField={setField} configDefaults={configDefaults} />
      </SettingsGroup>

      {/* The two preset lists live TOGETHER, not each under its engine: the
          activity is one ("my named LoRA chains"), and it is the block the
          scattered-options report came from. Each card still names its
          engine — the lists stay independent, one name can mean two chains. */}
      <SettingsGroup {...groupProps(group4)}>
      <KleinLorasCard config={config} setField={setField} />

      <KreaLorasCard config={config} setField={setField} />
      </SettingsGroup>

      <SettingsGroup {...groupProps(group5)}>
      <SeedVr2Card config={config} setField={setField} configDefaults={configDefaults}
        caps={caps} />

      {/* After the two engine cards, because it is what happens to their OUTPUT —
          one pass, whichever engine ran. */}
      <ImproveFinishCard config={config} setField={setField} configDefaults={configDefaults} />
      </SettingsGroup>

      <SettingsGroup {...groupProps(group6)}>
      <IdentityPromptsCard config={config} setField={setField} promptDefaults={props.promptDefaults}
        promptDefaultsBySubject={props.promptDefaultsBySubject}
        setIdentityPrompts={props.setIdentityPrompts} configDefaults={configDefaults} />
      </SettingsGroup>
    </div>
  )
}
