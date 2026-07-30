import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { INPUT_CLASS, Card, StatusBadge, SecretField } from './primitives'
import KleinLoraCombobox, { useKleinGenerationLoras } from './KleinLoraCombobox'
import PromptOverrideField from '../common/PromptOverrideField'
import PromptPreview from './PromptPreview'
import ResetToDefault from './ResetToDefault'
import { defaultValueAt } from './settingDefaults.js'
import { kreaStrengthRange, KREA_LORA_STRENGTH_DEFAULT } from '../../utils/kreaGenerationLoras'
import {
  identityPromptFields, PROMPT_SUBJECT_TYPES,
  readIdentityPrompt, writeIdentityPrompt, subjectHasOverride,
  GLOBAL_PROMPT_PART_FIELDS, SUBJECT_PROMPT_PART_FIELDS, FRAMING_PROMPT_PART_FIELDS,
} from '../common/promptOverride.js'
import { SUBJECT_TYPE_LABELS } from '../dataset/subjectTypes.js'

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
   just PICKS a preset ("None" by default) — the choice carries the intent,
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
                          placeholder = 'klein/my-lora.safetensors' }) {
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
      help={`Named combinations of your own LoRA files, chained after the consistency LoRA on the local Klein engine — inside a preset the order is the chain order (max ${MAX_GENERATION_LORAS} LoRAs each, ${MAX_GENERATION_LORA_PRESETS} presets). Pick each row from the LoRAs found under ComfyUI's models/loras (Klein-compatible ones are listed first; you can still type a path for a file not on disk yet) — any LoRA, any purpose. Per run, pick a preset in the workspace's 🖥️ Klein tuning panel ("None" by default). Presets and LoRA autocomplete by @waltm (Discord).`}
    >
      {presets.length === 0 && (
        <p className="text-sm text-content-muted">No presets yet — create your first combination below.</p>
      )}
      {presets.map((preset, i) => (
        <LoraPresetCard key={i} preset={preset} index={i} presets={presets} save={save}
          loraScan={loraScan} engineLabel="Klein"
          strengthRange={() => ({ min: 0, max: 1.5 })}
          defaultStrength={0.6} />
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

function KleinGenerationCard({ config, setField, configDefaults }) {
  // The shipped 5 is read from the server payload, never retyped here: it used
  // to be a literal `?? 5` in this file, i.e. a second copy of a backend default
  // that nothing kept in sync.
  const shipped = defaultValueAt(configDefaults, 'klein', 'generation_steps')
  const steps = config.klein?.generation_steps ?? shipped
  return (
    <Card
      id="klein-generation"
      title="Klein generation quality"
      help="How many sampler steps the local Klein engine spends on each generated variation. 5 is the value the app used before this was exposed, so leaving it alone keeps today's result. More steps render more cleanly but take proportionally longer — 10 steps is roughly twice the wait per image. It will not fix a wrong prompt: anatomy problems (extra limbs, tails) come from the identity prompt, not from the step count. Raised by ashish.sinha (Discord)."
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
    </Card>
  )
}

/* Krea 2 Identity Edit — the second LOCAL engine. Its headline knob is
   `grounding_px`, THE consistency <-> prompt-adherence dial, so it is first and
   explained in plain words: a number nobody can interpret is not a setting.
   The two path fields are BLANK-MEANS-AUTO on purpose: the resolver finds the
   files by canonical name then by a narrow token across every ComfyUI model
   root, so an install that looks nothing like the developer's works untouched —
   they exist for the person whose files are named something else. */
const KREA_GROUNDING_MIN = 512      // mirrors krea_edit_helper.GROUNDING_PX_MIN
const KREA_GROUNDING_MAX = 1536     // mirrors krea_edit_helper.GROUNDING_PX_MAX
const KREA_STEPS_MAX = 50

function KreaCard({ config, setField, configDefaults }) {
  const krea = config.krea || {}
  const reset = { config, configDefaults, setField }
  const dflt = (key) => defaultValueAt(configDefaults, 'krea', key)
  const grounding = Number(krea.grounding_px ?? dflt('grounding_px'))
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
          step={64}
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
          matters more.
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
          min={1}
          max={KREA_STEPS_MAX}
          step={1}
          value={krea.steps ?? dflt('steps')}
          onChange={(e) => setField('krea', 'steps',
            e.target.value === '' ? dflt('steps') : Number(e.target.value))}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          {dflt('steps')} is the value the model&rsquo;s own reference workflow uses. More is
          slower and rarely better on this pipeline.
        </p>
        <ResetToDefault label="Sampler steps" section="krea" field="steps" {...reset} />
      </div>

      <div className="mt-3 sm:max-w-md">
        <label htmlFor="krea-base-model" className="block text-xs font-medium text-content">
          Base model file (optional)
        </label>
        <input
          id="krea-base-model"
          type="text"
          value={krea.base_model ?? ''}
          placeholder="auto — finds a Krea 2 Turbo/Raw build"
          onChange={(e) => setField('krea', 'base_model', e.target.value)}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Leave blank unless you own several Krea builds. Blank = the app picks a Krea 2
          Turbo then Raw model from your ComfyUI. Non-Krea-2 checkpoints that merely carry
          &ldquo;krea&rdquo; in their name are skipped: the identity LoRA renders pure noise on them.
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
        <input
          id="krea-identity-lora"
          type="text"
          value={krea.identity_lora ?? ''}
          placeholder="krea/krea2_identity_edit_v1_2.safetensors"
          onChange={(e) => setField('krea', 'identity_lora', e.target.value)}
          className={INPUT_CLASS}
        />
        <p className="mt-1 text-[0.6875rem] text-content-subtle">
          Path relative to ComfyUI&rsquo;s models/loras. If the file isn&rsquo;t there under this
          name, the app searches your LoRA folders for a krea2_identity_edit file, so a
          renamed download still works.
        </p>
        <ResetToDefault label="Identity edit LoRA" section="krea" field="identity_lora" {...reset} />
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
      help={`Named combinations of your own LoRA files, chained after the identity-edit LoRA when Krea 2 Edit generates dataset images — inside a preset the order is the chain order (max ${MAX_GENERATION_LORAS} LoRAs each, ${MAX_GENERATION_LORA_PRESETS} presets). Pick each row from the LoRAs found under ComfyUI's models/loras; Krea-compatible ones are listed first, and a LoRA of another architecture is badged because ComfyUI would load it as a silent no-op here. Strength goes to 6, or to 20 for utility LoRAs whose filename says filter-bypass — those have no effect below ~10. Per run, pick a preset in the workspace's 🧬 Krea 2 Edit tuning panel ("None" by default). Only the model side is patched, so a LoRA's text-encoder weights are ignored. Preset mechanism by @waltm (Discord).`}
    >
      {presets.length === 0 && (
        <p className="text-sm text-content-muted">No presets yet — create your first combination below.</p>
      )}
      {presets.map((preset, i) => (
        <LoraPresetCard key={i} preset={preset} index={i} presets={presets} save={save}
          loraScan={loraScan} engineLabel="Krea 2"
          strengthRange={kreaStrengthRange} defaultStrength={KREA_LORA_STRENGTH_DEFAULT}
          placeholder="krea/my-lora.safetensors" />
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
      title="Identity & Klein prompts (advanced)"
      help="The hidden prompts that lock a subject's identity across generated variations, now editable. Pick the subject type first: each type (Human, Animal, Creature, Object, Other) has its OWN set, and a text you write for one never applies to another. Each box already holds the prompt in use: edit it to override, Reset to go back. Reproducibility note: as long as a box still matches the built-in text, nothing is stored and generation stays byte-identical to before — you also keep receiving improvements to that prompt. Feature request by @bbsorry (雨田壹); per-subject scoping reported by ashish.sinha."
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

/* ChatGPT subscription (Codex OAuth) — EXPERIMENTAL lane. Device-code login:
   the user opens the verification URL from ANY device and types the one-time
   code; we poll the backend until it reports connected. */
function ChatgptSubscriptionCard({ caps, config, setField, refreshCaps, toast, configDefaults }) {
  const sub = caps.chatgpt_subscription || {}
  const [device, setDevice] = useState(null)     // {verification_url, user_code}
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!device) return undefined
    const id = setInterval(async () => {
      try {
        const r = await apiFetch('/api/settings/chatgpt-oauth/poll', { background: true })
        if (r.status === 'connected') {
          setDevice(null)
          toast.success('ChatGPT subscription connected.')
          await refreshCaps(true)
        } else if (r.status === 'error') {
          setDevice(null)
          setError(r.detail || 'Login failed — try again.')
        }
      } catch { /* transient — keep polling */ }
    }, 3000)
    return () => clearInterval(id)
  }, [device, refreshCaps, toast])

  const start = async () => {
    setBusy(true); setError(null)
    try {
      const r = await postJson('/api/settings/chatgpt-oauth/start', {})
      setDevice(r)
    } catch (e) {
      setError(e.message || 'Could not start the login.')
    } finally {
      setBusy(false)
    }
  }

  const importCodex = async () => {
    setBusy(true); setError(null)
    try {
      await postJson('/api/settings/chatgpt-oauth/import-codex', {})
      setDevice(null)
      toast.success('Codex CLI session imported.')
      await refreshCaps(true)
    } catch (e) {
      setError(e.message || 'Import failed.')
    } finally {
      setBusy(false)
    }
  }

  const disconnect = async () => {
    setBusy(true); setError(null)
    try {
      await postJson('/api/settings/chatgpt-oauth/logout', {})
      toast.success('ChatGPT subscription disconnected.')
      await refreshCaps(true)
    } catch (e) {
      setError(e.message || 'Disconnect failed.')
    } finally {
      setBusy(false)
    }
  }

  const btn = 'rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium ' +
    'text-content hover:bg-surface-raised disabled:opacity-50'

  return (
    <Card
      title="ChatGPT subscription (experimental)"
      help="Run the ChatGPT engine on your ChatGPT Plus/Pro image quota instead of a pay-per-use API key. Undocumented lane — it may stop working if OpenAI closes it. Limits vs API mode: up to 5 reference images (instead of 16), your plan's daily image cap applies, SFW only."
    >
      <div className="flex items-center justify-between">
        <StatusBadge ok={!!sub.connected} okLabel={sub.email ? `Connected — ${sub.email}` : 'Connected'} missingLabel="Not connected" />
        <div className="flex gap-2">
          {!sub.connected && (
            <button type="button" onClick={start} disabled={busy || !!device} className={btn}>
              {device ? 'Waiting for you to enter the code…' : 'Connect with ChatGPT'}
            </button>
          )}
          {!sub.connected && sub.codex_cli_detected && (
            <button type="button" onClick={importCodex} disabled={busy || !!device} className={btn}>
              Import from Codex CLI
            </button>
          )}
          {sub.connected && (
            <button type="button" onClick={disconnect} disabled={busy} className={btn}>
              Disconnect
            </button>
          )}
        </div>
      </div>

      {device && (
        <div role="status" className="rounded-lg border border-primary/40 bg-primary/10 p-3 text-sm text-content">
          <p>1. Open <a href={device.verification_url} target="_blank" rel="noreferrer" className="font-medium underline">{device.verification_url}</a> on any device and sign in.</p>
          <p className="mt-1">2. Enter this one-time code (expires in 15 minutes):</p>
          <p className="mt-1 select-all font-mono text-lg font-semibold tracking-widest">{device.user_code}</p>
        </div>
      )}

      {error && <p className="text-xs text-rose-400"><span aria-hidden="true">✗</span> {error}</p>}

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

export default function EnginesSection(props) {
  const { config, setField, toggleEngine, caps, refreshCaps, toast, configDefaults } = props
  return (
    <div className="space-y-6">
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

      <KleinGenerationCard config={config} setField={setField} configDefaults={configDefaults} />

      <KleinLorasCard config={config} setField={setField} />

      <KreaCard config={config} setField={setField} configDefaults={configDefaults} />

      <KreaLorasCard config={config} setField={setField} />

      <IdentityPromptsCard config={config} setField={setField} promptDefaults={props.promptDefaults}
        promptDefaultsBySubject={props.promptDefaultsBySubject}
        setIdentityPrompts={props.setIdentityPrompts} configDefaults={configDefaults} />
    </div>
  )
}
