import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { INPUT_CLASS, Card, StatusBadge, SecretField } from './primitives'

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

/* Optional generation LoRAs for the local Klein engine (Idea by @waltm —
   Discord feature request): an ORDERED list of user-pointed LoRA files (any
   files, any purpose — texture, anatomy, style…), chained after the
   consistency LoRA in LIST ORDER. Rows are edited here (file + default
   strength + NSFW-only flag, add/remove/reorder, capped at 8) and armed PER
   RUN in the workspace's 🖥️ Klein tuning panel (off by default every visit).
   NSFW-only rows only ever inject on 🔞 variations. The app never ships or
   hardcodes a LoRA name. */
const MAX_GENERATION_LORAS = 8   // mirrors backend klein_edit_helper.MAX_GENERATION_LORAS

function KleinLorasCard({ config, setField }) {
  const rows = Array.isArray(config.klein?.generation_loras) ? config.klein.generation_loras : []
  const save = (next) => setField('klein', 'generation_loras', next)
  const patch = (i, p) => save(rows.map((r, j) => (j === i ? { ...r, ...p } : r)))
  const move = (i, dir) => {
    const j = i + dir
    if (j < 0 || j >= rows.length) return
    const next = [...rows]
    ;[next[i], next[j]] = [next[j], next[i]]
    save(next)
  }
  const smallBtn = 'grid h-6 w-6 place-items-center rounded border border-border text-xs ' +
    'text-content-muted hover:bg-surface-raised disabled:opacity-30'
  return (
    <Card
      title="Klein generation LoRAs (optional)"
      help={`Your own extra LoRAs chained after the consistency LoRA on the local Klein engine, in this order (max ${MAX_GENERATION_LORAS}). Point each row at a file under ComfyUI's models/loras (relative name, e.g. klein/my-lora.safetensors) — any LoRA, any purpose. Every row is OFF by default for each generation; arm it per run in the workspace's 🖥️ Klein tuning panel. NSFW-only rows only apply to 🔞 variations. Idea by @waltm (Discord).`}
    >
      {rows.length === 0 && (
        <p className="text-sm text-content-muted">No generation LoRAs yet — add your first one below.</p>
      )}
      {rows.map((row, i) => {
        const strength = Number.isFinite(Number(row?.strength)) ? Number(row.strength) : 0.6
        return (
          <div key={i} className="rounded-lg border border-border p-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-content-muted w-5 shrink-0" aria-hidden="true">{i + 1}.</span>
              <input
                type="text" aria-label={`LoRA file ${i + 1}`}
                value={row?.file || ''}
                onChange={(e) => patch(i, { file: e.target.value })}
                placeholder="klein/my-lora.safetensors"
                className={`${INPUT_CLASS} mt-0`}
              />
              <button type="button" onClick={() => move(i, -1)} disabled={i === 0}
                aria-label={`Move LoRA ${i + 1} up`} title="Chain earlier" className={smallBtn}>↑</button>
              <button type="button" onClick={() => move(i, 1)} disabled={i === rows.length - 1}
                aria-label={`Move LoRA ${i + 1} down`} title="Chain later" className={smallBtn}>↓</button>
              <button type="button" onClick={() => save(rows.filter((_, j) => j !== i))}
                aria-label={`Remove LoRA ${i + 1}`} title="Remove this LoRA"
                className={`${smallBtn} hover:bg-red-500/15 hover:text-red-300`}>✕</button>
            </div>
            <div className="mt-2 flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-2 text-xs text-content-muted flex-1 min-w-[180px]">
                <span className="whitespace-nowrap">Default strength: {strength.toFixed(2)}</span>
                <input
                  type="range" min={0} max={1.5} step={0.05} value={strength}
                  aria-label={`LoRA ${i + 1} default strength`}
                  onChange={(e) => patch(i, { strength: Number(e.target.value) })}
                  className="flex-1 accent-indigo-500"
                />
              </label>
              <label className="flex items-center gap-2 text-xs text-content">
                <input
                  type="checkbox" checked={!!row?.nsfw_only}
                  onChange={(e) => patch(i, { nsfw_only: e.target.checked })}
                  className="h-4 w-4 rounded border-border-strong"
                />
                🔞 NSFW-only
              </label>
            </div>
          </div>
        )
      })}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => save([...rows, { file: '', strength: 0.6, nsfw_only: false }])}
          disabled={rows.length >= MAX_GENERATION_LORAS}
          className="rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50"
        >
          ＋ Add LoRA
        </button>
        <span className="text-xs text-content-muted">{rows.length}/{MAX_GENERATION_LORAS}</span>
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

        <fieldset>
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
    </div>
  )
}
