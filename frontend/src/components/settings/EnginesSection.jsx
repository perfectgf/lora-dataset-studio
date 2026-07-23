import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { INPUT_CLASS, Card, StatusBadge, SecretField } from './primitives'
import KleinLoraCombobox, { useKleinGenerationLoras } from './KleinLoraCombobox'
import PromptOverrideField from '../common/PromptOverrideField'
import { IDENTITY_PROMPT_FIELDS } from '../common/promptOverride.js'

const ENGINE_SECRETS = [
  { key: 'GEMINI_API_KEY', label: 'Gemini API key', testTarget: 'gemini', help: 'Powers the Nano Banana engine.' },
  { key: 'OPENAI_API_KEY', label: 'OpenAI API key', testTarget: 'openai',
    help: 'Powers the ChatGPT (gpt-image-2) engine. Optional if you connect a ChatGPT subscription below.' },
]

const ENGINE_OPTIONS = [
  { id: 'nanobanana', label: 'Nano Banana (Gemini)' },
  { id: 'chatgpt', label: 'ChatGPT (gpt-image-2)' },
  { id: 'klein', label: 'Klein (ComfyUI, local)' },
]

/* Optional generation-LoRA PRESETS for the local Klein engine (Idea by
   @waltm — Discord feature request): named combinations of user-pointed LoRA
   files (any files, any purpose — texture, anatomy, style…). Inside a preset
   the rows chain after the consistency LoRA in LIST ORDER (file + strength,
   reorderable, capped at 8). Per run the workspace's 🖥️ Klein tuning panel
   just PICKS a preset ("None" by default) — the choice carries the intent,
   there is no automatic gating. The app never ships or hardcodes a LoRA name. */
const MAX_GENERATION_LORAS = 8        // mirrors backend klein_edit_helper caps
const MAX_GENERATION_LORA_PRESETS = 12

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

function KleinLoraPresetCard({ preset, index, presets, save, loraScan }) {
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
        const strength = Number.isFinite(Number(row?.strength)) ? Number(row.strength) : 0.6
        return (
          <div key={i} className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-content-muted w-4 shrink-0" aria-hidden="true">{i + 1}.</span>
            <KleinLoraCombobox
              ariaLabel={`Preset ${index + 1} LoRA file ${i + 1}`}
              value={row?.file || ''}
              onChange={(next) => patchRow(i, { file: next })}
              {...loraScan}
            />
            <label className="flex items-center gap-1.5 text-xs text-content-muted">
              <span className="whitespace-nowrap">{strength.toFixed(2)}</span>
              <input
                type="range" min={0} max={1.5} step={0.05} value={strength}
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
          onClick={() => patchPreset({ loras: [...rows, { file: '', strength: 0.6 }] })}
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
        <KleinLoraPresetCard key={i} preset={preset} index={i} presets={presets} save={save} loraScan={loraScan} />
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
   hardcoded and invisible; here each is a GLOBAL override shown in ONE editable
   box that already holds the shipped default text, with a Reset.

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
const IMPROVE_KNOBS = [
  { key: 'improve_megapixels', label: 'Output size (MP)', fallback: 2,
    min: 0.5, max: 8, step: 0.5,
    hint: 'The result’s resolution. 2 = the shipped value.' },
  { key: 'improve_base_lora_strength', label: 'Enhancement LoRA', fallback: 0,
    min: 0, max: 2, step: 0.05,
    hint: '0 = off (the shipped behaviour). Try 0.5–0.8. Needs klein/realistic.safetensors.' },
  // Drives klein.consistency_strength, which enqueue_klein_edit clamps to 1.5 — the
  // UI must not offer a value the engine pulls back. It anchors COMPOSITION, not
  // identity: it was mislabelled "Character LoRA" when these knobs first shipped.
  { key: 'improve_consistency_strength', label: 'Consistency LoRA', fallback: 0,
    min: 0, max: 1.5, step: 0.05,
    hint: 'Holds the composition and background. High values resist the edit.' },
  { key: 'improve_steps', label: 'Steps', fallback: 4,
    min: 1, max: 50, step: 1, hint: 'More steps = slower, usually cleaner.' },
]

function IdentityPromptsCard({ config, setField, promptDefaults }) {
  const ip = config.identity_prompts || {}
  const defaults = promptDefaults || {}
  const set = (key, v) => setField('identity_prompts', key, v)
  const improveEnabled = ip.klein_improve_enabled !== false
  return (
    <Card
      id="identity-prompts"
      title="Identity & Klein prompts (advanced)"
      help="The hidden prompts that lock a subject's facial identity across generated variations, now editable. Each box already holds the prompt in use: edit it to override, Reset to go back. Each applies globally to every dataset. Reproducibility note: as long as a box still matches the built-in text, nothing is stored and generation stays byte-identical to before — you also keep receiving improvements to that prompt. Feature request by @bbsorry (雨田壹)."
    >
      {IDENTITY_PROMPT_FIELDS.map((f) => (
        <PromptOverrideField
          key={f.key}
          id={f.id}
          label={f.label}
          desc={f.desc}
          value={ip[f.key]}
          defaultText={defaults[f.key]}
          onChange={(v) => set(f.key, v)}
        />
      ))}

      <div className="border-t border-border pt-4">
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
      <div className="border-t border-border pt-4">
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
                value={config.klein?.[k.key] ?? k.fallback}
                onChange={(e) => setField('klein', k.key,
                  e.target.value === '' ? k.fallback : Number(e.target.value))}
                className={INPUT_CLASS}
              />
              <p className="mt-1 text-[0.6875rem] text-content-subtle">{k.hint}</p>
            </div>
          ))}
        </div>
      </div>
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
function ChatgptSubscriptionCard({ caps, config, setField, refreshCaps, toast }) {
  const sub = caps.chatgpt_subscription || {}
  const [device, setDevice] = useState(null)     // {verification_url, user_code}
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!device) return undefined
    const id = setInterval(async () => {
      try {
        const r = await apiFetch('/api/settings/chatgpt-oauth/poll')
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
      </div>
    </Card>
  )
}

export default function EnginesSection(props) {
  const { config, setField, toggleEngine, caps, refreshCaps, toast } = props
  return (
    <div className="space-y-6">
      <Card title="API keys" help="Keys are write-only — fields stay blank even when a key is already saved.">
        {ENGINE_SECRETS.map((f) => <SecretField key={f.key} field={f} {...props} />)}
      </Card>

      <ChatgptSubscriptionCard caps={caps} config={config} setField={setField} refreshCaps={refreshCaps} toast={toast} />

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
        </fieldset>
      </Card>

      <KleinLorasCard config={config} setField={setField} />

      <IdentityPromptsCard config={config} setField={setField} promptDefaults={props.promptDefaults} />
    </div>
  )
}
