import { useEffect, useState, useCallback } from 'react'
import { apiFetch, putJson, postJson, del } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'

const SECRET_FIELDS = [
  { key: 'GEMINI_API_KEY', label: 'Gemini API key', testTarget: 'gemini', help: 'Powers the Nano Banana engine.' },
  { key: 'OPENAI_API_KEY', label: 'OpenAI API key', testTarget: 'openai',
    help: 'Powers the ChatGPT (gpt-image-2) engine. Optional if you connect a ChatGPT subscription below.' },
  { key: 'HF_TOKEN', label: 'Hugging Face token', testTarget: null,
    help: 'Only needed to auto-download license-gated models (the Klein fp8 model). Read token from hf.co/settings/tokens, after accepting the model license.' },
  { key: 'VAST_API_KEY', label: 'vast.ai API key', testTarget: 'vast',
    help: 'Enables cloud GPU training: the app rents a GPU for the run and shuts it down when done (typical run: $1-2). Get a key at cloud.vast.ai → Keys.' },
]

const ENGINE_OPTIONS = [
  { id: 'nanobanana', label: 'Nano Banana (Gemini)' },
  { id: 'chatgpt', label: 'ChatGPT (gpt-image-2)' },
  { id: 'klein', label: 'Klein (ComfyUI, local)' },
]

const CHATGPT_AUTH_OPTIONS = [
  { id: 'auto', label: 'Auto — subscription when connected, otherwise API key' },
  { id: 'api', label: 'API key only' },
  { id: 'subscription', label: 'Subscription only' },
]

const CAPTIONING_OPTIONS = [
  { id: 'auto', label: 'Auto (best available)' },
  { id: 'joycaption', label: 'JoyCaption' },
  { id: 'ollama', label: 'Ollama vision' },
  { id: 'none', label: 'None' },
]

const FAMILY_OPTIONS = ['zimage', 'sdxl', 'krea']

const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none'

// Status is never color-only: an explicit glyph + text label carries the
// meaning, color is a reinforcing cue on top.
function StatusBadge({ ok, okLabel = 'Configured', missingLabel = 'Not set' }) {
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium ${ok ? 'text-emerald-400' : 'text-content-subtle'}`}>
      <span aria-hidden="true">{ok ? '✓' : '✗'}</span>
      {ok ? okLabel : missingLabel}
    </span>
  )
}

function TestResult({ result }) {
  if (!result) return null
  return (
    <p className={`text-xs ${result.ok ? 'text-emerald-400' : 'text-rose-400'}`}>
      <span aria-hidden="true">{result.ok ? '✓' : '✗'}</span> {result.detail}
    </p>
  )
}

function TestButton({ target, onResult, beforeTest }) {
  const [busy, setBusy] = useState(false)
  const run = async () => {
    setBusy(true)
    try {
      // Secret fields pass beforeTest to persist the value still sitting in the
      // write-only input: the probe reads the SAVED key, so testing an unsaved
      // paste would always answer "key missing".
      if (beforeTest) await beforeTest()
      onResult(await postJson(`/api/settings/test/${target}`, {}))
    } catch (e) {
      onResult({ ok: false, detail: e.message || 'Test failed' })
    } finally {
      setBusy(false)
    }
  }
  return (
    <button
      type="button"
      onClick={run}
      disabled={busy}
      className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50"
    >
      {busy ? 'Testing…' : 'Test'}
    </button>
  )
}

/* First-time walkthrough for renting cloud GPUs — collapsed by default so the
   API-keys card stays compact for users who already have a key. */
function VastKeyGuide() {
  const link = 'font-medium text-sky-300 underline hover:text-sky-200'
  return (
    <details className="mb-2 rounded-lg border border-border bg-surface px-3 py-2 open:pb-3">
      <summary className="cursor-pointer select-none text-xs font-medium text-content">
        <span aria-hidden>📖</span> How to get a vast.ai API key (≈2 minutes)
      </summary>
      <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-xs text-content-muted">
        <li>
          Create a free account at{' '}
          <a href="https://cloud.vast.ai/" target="_blank" rel="noreferrer" className={link}>cloud.vast.ai</a>
          {' '}(email or Google sign-in).
        </li>
        <li>
          Add credit: open{' '}
          <a href="https://cloud.vast.ai/billing/" target="_blank" rel="noreferrer" className={link}>Billing</a>
          {' '}in the left sidebar and click <strong>Add Credit</strong> — $5 is plenty to
          start (a typical training run costs ~$1–2, billed by vast.ai, not by this app).
        </li>
        <li>
          Open{' '}
          <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noreferrer" className={link}>Keys</a>
          {' '}(left sidebar, under Account) and copy your API key — create one first if
          the list is empty.
        </li>
        <li>
          Paste the key in the field below and press <strong>Test</strong> — it saves the
          key automatically and should answer “connected as &lt;your account&gt;”.
        </li>
      </ol>
    </details>
  )
}

function Card({ title, help, children }) {
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <h2 className="text-base font-semibold text-content">{title}</h2>
      {help && <p className="mt-1 text-sm text-content-muted">{help}</p>}
      <div className="mt-4 space-y-4">{children}</div>
    </section>
  )
}

function TextField({ id, label, value, onChange, placeholder, help }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-content">{label}</label>
      {help && <p className="mb-1 text-xs text-content-muted">{help}</p>}
      <input
        id={id}
        type="text"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={INPUT_CLASS}
      />
    </div>
  )
}

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

export default function SettingsPage() {
  const toast = useToast()
  const { caps, refresh } = useCapabilities()
  const [config, setConfig] = useState(null)
  const [runtime, setRuntime] = useState({ host: null, port: null })
  const [secretsPresence, setSecretsPresence] = useState({})
  const [secretInputs, setSecretInputs] = useState({})
  const [testResults, setTestResults] = useState({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await apiFetch('/api/settings')
      setConfig(data.config)
      setRuntime(data.runtime || { host: null, port: null })
      setSecretsPresence(data.secrets)
    } catch (e) {
      toast.error(`Failed to load settings: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { load() }, [load])

  const setField = (section, key, value) => {
    setConfig((prev) => ({ ...prev, [section]: { ...prev[section], [key]: value } }))
  }

  const recordTestResult = (target, result) => {
    setTestResults((prev) => ({ ...prev, [target]: result }))
  }

  const toggleEngine = (id) => {
    setConfig((prev) => {
      const enabled = prev.engines.enabled || []
      const next = enabled.includes(id) ? enabled.filter((e) => e !== id) : [...enabled, id]
      return { ...prev, engines: { ...prev.engines, enabled: next } }
    })
  }

  // Clear a saved API key. Explicit action — the write-only field can't wipe a key
  // by going blank — so confirm, delete server-side, then refresh presence + caps
  // so any engine that depended on it flips to unavailable right away.
  const handleDeleteSecret = async (key, label) => {
    if (!window.confirm(`Remove the saved ${label}? Any engine that uses it stops working until you add a new key.`)) return
    try {
      const data = await del(`/api/settings/secret/${key}`)
      setSecretsPresence(data.secrets)
      setSecretInputs((prev) => { const next = { ...prev }; delete next[key]; return next })
      await refresh(true)
      toast.success(`${label} removed.`)
    } catch (e) {
      toast.error(`Remove failed: ${e.message}`)
    }
  }

  // Save a single secret field's pending value (used by the Test button so
  // "paste key -> Test" just works without a separate Save click). No-op when
  // the field is empty; a failed save throws so the test reports it instead of
  // probing a key that never landed.
  const saveSecretIfPending = async (key) => {
    const pending = (secretInputs[key] || '').trim()
    if (!pending) return
    const data = await putJson('/api/settings', { secrets: { [key]: pending } })
    setSecretsPresence(data.secrets)
    setSecretInputs((prev) => { const next = { ...prev }; delete next[key]; return next })
    await refresh(true)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      // Only send secret fields the user actually typed into — the fields
      // stay blank on load, so an untouched field must never overwrite an
      // already-saved key with an empty value. Trim: a pasted key with
      // trailing whitespace/newline would otherwise corrupt the Bearer header.
      const secrets = Object.fromEntries(
        Object.entries(secretInputs)
          .map(([k, v]) => [k, (v || '').trim()])
          .filter(([, v]) => v)
      )
      const data = await putJson('/api/settings', { config, secrets })
      setConfig(data.config)
      setRuntime(data.runtime || { host: null, port: null })
      setSecretsPresence(data.secrets)
      setSecretInputs({})
      // force=true: /api/capabilities caches probes for 30s server-side, so a
      // plain refresh() could leave onboarding/studio_visible stale right
      // after the config that determines them just changed.
      await refresh(true)
      toast.success('Settings saved.')
      return true
    } catch (e) {
      toast.error(`Save failed: ${e.message}`)
      return false
    } finally {
      setSaving(false)
    }
  }

  if (loading || !config) {
    return <p className="text-content-muted">Loading settings…</p>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-xl font-semibold text-content">Settings</h1>
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white transition-transform hover:-translate-y-px disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>

      {!caps.configured && (
        <div role="status" className="rounded-xl border border-primary/40 bg-primary/10 p-4 text-sm text-content">
          <p className="font-medium">Let's get you set up.</p>
          <p className="mt-1 text-content-muted">
            Add at least one image API key to start. Add ComfyUI + ai-toolkit for local generation & training.
          </p>
        </div>
      )}

      <UpdatesCard />
      <TrashCard />

      <ServerCard config={config} setField={setField} runtime={runtime} onSaveThenRestart={handleSave} />

      <Card title="API keys" help="Keys are write-only — fields stay blank even when a key is already saved.">
        {SECRET_FIELDS.map((f) => (
          <div key={f.key} className="flex items-end gap-3">
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <label htmlFor={f.key} className="block text-sm font-medium text-content">{f.label}</label>
                <StatusBadge ok={!!secretsPresence[f.key]} />
              </div>
              <p className="mb-1 text-xs text-content-muted">{f.help}</p>
              {f.key === 'VAST_API_KEY' && <VastKeyGuide />}
              <input
                id={f.key}
                type="password"
                autoComplete="off"
                value={secretInputs[f.key] ?? ''}
                onChange={(e) => setSecretInputs((prev) => ({ ...prev, [f.key]: e.target.value }))}
                placeholder={secretsPresence[f.key] ? 'Already set — enter a new value to replace it' : 'Not set'}
                className={INPUT_CLASS}
              />
              {f.testTarget && <TestResult result={testResults[f.testTarget]} />}
            </div>
            {f.testTarget && (
              <TestButton target={f.testTarget} beforeTest={() => saveSecretIfPending(f.key)}
                onResult={(r) => recordTestResult(f.testTarget, r)} />
            )}
            {secretsPresence[f.key] && (
              <button
                type="button"
                onClick={() => handleDeleteSecret(f.key, f.label)}
                title={`Remove the saved ${f.label}`}
                className="shrink-0 rounded-md border border-rose-500/40 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-500/10"
              >
                Remove
              </button>
            )}
          </div>
        ))}
      </Card>

      <ChatgptSubscriptionCard caps={caps} config={config} setField={setField} refreshCaps={refresh} toast={toast} />

      <Card title="Endpoints">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <TextField
              id="comfyui-api-url"
              label="ComfyUI API URL"
              value={config.comfyui.api_url}
              onChange={(v) => setField('comfyui', 'api_url', v)}
              placeholder="http://127.0.0.1:8188"
            />
            <TestResult result={testResults.comfyui} />
          </div>
          <TestButton target="comfyui" onResult={(r) => recordTestResult('comfyui', r)} />
        </div>

        <div className="flex items-end gap-3">
          <div className="flex-1 space-y-4">
            <TextField
              id="ollama-url"
              label="Ollama URL"
              value={config.ollama.url}
              onChange={(v) => setField('ollama', 'url', v)}
              placeholder="http://127.0.0.1:11434"
            />
            <TextField
              id="ollama-vision-model"
              label="Ollama vision model"
              value={config.ollama.vision_model}
              onChange={(v) => setField('ollama', 'vision_model', v)}
              placeholder="huihui_ai/qwen3-vl-abliterated:8b"
            />
            <TestResult result={testResults.ollama} />
          </div>
          <TestButton target="ollama" onResult={(r) => recordTestResult('ollama', r)} />
        </div>
      </Card>

      <Card title="Paths">
        <TextField
          id="comfyui-base-dir"
          label="ComfyUI install directory"
          value={config.comfyui.base_dir}
          onChange={(v) => setField('comfyui', 'base_dir', v)}
          placeholder="C:\ComfyUI"
          help="Used to derive the output/input/models/loras folders unless overridden."
        />
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <TextField
              id="aitoolkit-dir"
              label="ai-toolkit directory"
              value={config.aitoolkit.dir}
              onChange={(v) => setField('aitoolkit', 'dir', v)}
              placeholder="C:\ai-toolkit"
            />
            <TestResult result={testResults.aitoolkit} />
          </div>
          <TestButton target="aitoolkit" onResult={(r) => recordTestResult('aitoolkit', r)} />
        </div>
        <TextField
          id="dataset-images-root"
          label="Dataset images root"
          value={config.paths.dataset_images_root}
          onChange={(v) => setField('paths', 'dataset_images_root', v)}
          placeholder="Defaults to data/datasets"
        />

        <details className="rounded-lg border border-border p-3">
          <summary className="cursor-pointer text-sm font-medium text-content-muted">
            Advanced: ai-toolkit overrides
          </summary>
          <div className="mt-3 space-y-4">
            <TextField
              id="aitoolkit-datasets-dir"
              label="Datasets directory override"
              value={config.aitoolkit.datasets_dir}
              onChange={(v) => setField('aitoolkit', 'datasets_dir', v)}
              placeholder="Defaults to <ai-toolkit>/datasets"
            />
            <TextField
              id="aitoolkit-output-dir"
              label="Output directory override"
              value={config.aitoolkit.output_dir}
              onChange={(v) => setField('aitoolkit', 'output_dir', v)}
              placeholder="Defaults to <ai-toolkit>/output"
            />
            <TextField
              id="aitoolkit-hf-home"
              label="Hugging Face cache override"
              value={config.aitoolkit.hf_home}
              onChange={(v) => setField('aitoolkit', 'hf_home', v)}
              placeholder="Defaults to <ai-toolkit>/hf-cache/huggingface"
            />
          </div>
        </details>
      </Card>

      <Card title="Choices">
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

        <div>
          <label htmlFor="captioning-backend" className="block text-sm font-medium text-content">Captioning backend</label>
          <select
            id="captioning-backend"
            value={config.captioning.backend}
            onChange={(e) => setField('captioning', 'backend', e.target.value)}
            className={INPUT_CLASS}
          >
            {CAPTIONING_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </div>

        <div>
          <label htmlFor="training-default-family" className="block text-sm font-medium text-content">Default training family</label>
          <select
            id="training-default-family"
            value={config.training.default_family}
            onChange={(e) => setField('training', 'default_family', e.target.value)}
            className={INPUT_CLASS}
          >
            {FAMILY_OPTIONS.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label htmlFor="face-threshold-green" className="block text-sm font-medium text-content">
              Face score — green threshold
            </label>
            <input
              id="face-threshold-green"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.face_scoring.green}
              onChange={(e) => setField('face_scoring', 'green', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS}
            />
          </div>
          <div>
            <label htmlFor="face-threshold-orange" className="block text-sm font-medium text-content">
              Face score — orange threshold
            </label>
            <input
              id="face-threshold-orange"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.face_scoring.orange}
              onChange={(e) => setField('face_scoring', 'orange', parseFloat(e.target.value) || 0)}
              className={INPUT_CLASS}
            />
          </div>
        </div>
      </Card>

      <CloudTrainingCard config={config} setField={setField} />

      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white transition-transform hover:-translate-y-px disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>

      <LogViewer />
    </div>
  )
}

/* Cloud training limits: concurrency cap, offer price ceiling, monthly budget
   and the stall watchdog timeout. Fetches the cloud status ONCE on mount for
   the "Spent this month" info line — no poll, this page is not a dashboard. */
function CloudTrainingCard({ config, setField }) {
  const [spend, setSpend] = useState(null)
  useEffect(() => {
    let alive = true
    // Raw fetch (not apiFetch): this info line is best-effort — a transient
    // 500 must not fire the global error toast over a cosmetic detail.
    fetch('/api/dataset/train/cloud/status', { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d && typeof d.month_spend === 'number') setSpend(d.month_spend) })
      .catch(() => { /* info line is best-effort */ })
    return () => { alive = false }
  }, [])
  return (
    <Card title="Cloud training" help="vast.ai GPU rental guardrails — how many training pods may run at once, the offer price ceiling, your monthly spend limit, and how long a run may go without step progress before it is rescued and killed.">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label htmlFor="cloud-max-concurrent-runs" className="block text-sm font-medium text-content">
            Max simultaneous cloud runs
          </label>
          <input
            id="cloud-max-concurrent-runs"
            type="number"
            min="1"
            max="10"
            step="1"
            value={config.cloud?.max_concurrent_runs ?? 1}
            onChange={(e) => setField('cloud', 'max_concurrent_runs', parseInt(e.target.value) || 1)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-max-price-per-hour" className="block text-sm font-medium text-content">
            Max price per hour ($)
          </label>
          <input
            id="cloud-max-price-per-hour"
            type="number"
            min="0.1"
            max="5"
            step="0.05"
            value={config.cloud?.max_price_per_hour ?? 0.8}
            onChange={(e) => setField('cloud', 'max_price_per_hour', Math.max(0.1, parseFloat(e.target.value) || 0.1))}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-monthly-budget" className="block text-sm font-medium text-content">
            Monthly budget ($, 0 = unlimited)
          </label>
          <input
            id="cloud-monthly-budget"
            type="number"
            min="0"
            step="1"
            value={config.cloud?.monthly_budget_usd ?? 0}
            onChange={(e) => setField('cloud', 'monthly_budget_usd', parseFloat(e.target.value) || 0)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-stall-timeout" className="block text-sm font-medium text-content">
            Stall timeout (minutes)
          </label>
          <input
            id="cloud-stall-timeout"
            type="number"
            min="5"
            max="240"
            step="1"
            value={config.cloud?.stall_timeout_minutes ?? 30}
            onChange={(e) => setField('cloud', 'stall_timeout_minutes', parseInt(e.target.value) || 30)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-min-reliability" className="block text-sm font-medium text-content">
            Min host reliability
          </label>
          <input
            id="cloud-min-reliability"
            type="number"
            min="0.9"
            max="0.999"
            step="0.005"
            value={config.cloud?.min_reliability ?? 0.98}
            onChange={(e) => setField('cloud', 'min_reliability', Math.min(0.999, Math.max(0.9, parseFloat(e.target.value) || 0.98)))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            Lower it (e.g. 0.95) to surface cheaper hosts in the GPU picker — at a higher risk of a pod that never boots (≈ a few wasted cents, auto-cleaned).
          </p>
        </div>
      </div>
      {spend != null && (
        <p className="text-xs text-content-muted">Spent this month: ${spend.toFixed(2)}</p>
      )}
    </Card>
  )
}

/* App-wide trash: everything the app "deletes" (checkpoints, cloud staging,
   deployed LoRA) is MOVED here — this card is the only place bytes actually
   die. Size fetched once on mount (no poll). */
function TrashCard() {
  const [size, setSize] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    let alive = true
    apiFetch('/api/trash')
      .then((d) => { if (alive) setSize(d?.size_bytes ?? null) })
      .catch(() => { /* best-effort */ })
    return () => { alive = false }
  }, [])
  const fmt = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB`
    : b >= 1e6 ? `${Math.round(b / 1e6)} MB`
    : b > 0 ? `${Math.max(1, Math.round(b / 1e3))} KB` : 'empty')
  const empty = async () => {
    if (!window.confirm('Permanently delete everything in the trash?\n\nThis is the ONLY destructive action — deleted checkpoints cannot be recovered afterwards.')) return
    setBusy(true)
    try {
      const d = await postJson('/api/trash/empty', {})
      if (d?.ok) setSize(0)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card title="Trash" help="Everything the app deletes (checkpoints, cloud staging, deployed LoRAs) is moved here first — emptying it is the only action that actually destroys files.">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-content">
          <span aria-hidden>🗑</span> Trash size:{' '}
          <span className="font-semibold tabular-nums">{size == null ? '…' : fmt(size)}</span>
        </span>
        <button type="button" onClick={empty} disabled={busy || !size}
          className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 disabled:opacity-40">
          {busy ? 'Emptying…' : 'Empty trash'}
        </button>
      </div>
    </Card>
  )
}

/* In-app updater: "Check for updates" hits the git-aware check (commits-behind for a
   clone, release tag for a packaged build). "Update & restart" pulls + restarts the
   server; we then poll /api/health until the relaunched process answers and hard-reload
   the SPA so the new frontend/dist loads. */
function UpdatesCard() {
  const [status, setStatus] = useState(null)
  const [checking, setChecking] = useState(false)
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'

  // Passive check on mount (cached server-side, no git fetch): the card shows
  // the current build immediately instead of waiting for a manual check.
  useEffect(() => {
    let alive = true
    apiFetch('/api/update/check')
      .then((d) => { if (alive) setStatus((prev) => prev || d) })
      .catch(() => { /* best-effort — the manual button stays available */ })
    return () => { alive = false }
  }, [])

  const check = async () => {
    setChecking(true)
    try {
      setStatus(await apiFetch('/api/update/check?force=1'))
    } catch (e) {
      setStatus({ ok: false, reason: e.message || 'Check failed' })
    } finally {
      setChecking(false)
    }
  }

  const waitForHealthAndReload = async () => {
    // The server is re-execing: /api/health refuses connections for a few seconds,
    // then answers again on the same port. Poll, then hard-reload to pull new dist.
    for (let i = 0; i < 120; i += 1) {
      await new Promise((r) => setTimeout(r, 1000))
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (res.ok) { window.location.reload(); return }
      } catch { /* still down — keep waiting */ }
    }
    setApplying(false); setPhase('')          // gave up after ~2 min
  }

  const apply = async () => {
    setApplying(true); setPhase('pulling')
    try {
      const res = await postJson('/api/update/apply', {})
      if (res.restarting) {
        setPhase('restarting')
        waitForHealthAndReload()              // not awaited: UI shows "restarting…"
      } else {
        setStatus(res.ok ? { ...res, up_to_date: true } : res)
        setApplying(false); setPhase('')
      }
    } catch (e) {
      setStatus({ ok: false, reason: e.message || 'Update failed' })
      setApplying(false); setPhase('')
    }
  }

  const s = status
  const canPull = s && s.update_available && s.is_git
  return (
    <Card title="Updates" help="Pull the latest version from GitHub and restart — without leaving the app.">
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={check} disabled={checking || applying}
          className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
          {checking ? 'Checking…' : 'Check for updates'}
        </button>
        {s?.current && (
          <span className="text-xs text-content-subtle">
            Current build:{' '}
            <span className="font-medium text-content">v{s.current}{s.current_sha ? ` (${s.current_sha})` : ''}</span>
          </span>
        )}
        {s && (
          <span className="text-xs text-content-subtle">
            Latest build:{' '}
            <span className="font-medium text-content">
              {s.remote_sha
                ? `${s.remote_sha}${typeof s.behind === 'number' && s.behind > 0 ? ` (+${s.behind} commit${s.behind === 1 ? '' : 's'})` : ''}`
                : s.latest ? `v${s.latest}`
                : s.update_available ? 'update available'
                : '— press “Check for updates”'}
            </span>
          </span>
        )}
        {/* Read WHAT the update contains before pulling: the compare view lists
            exactly the incoming commits; otherwise the branch history. Only
            present after a git-aware "Check for updates" (force). */}
        {s && (s.compare_url || s.commits_url) && (
          <a href={s.compare_url || s.commits_url} target="_blank" rel="noreferrer"
            className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
            {s.compare_url ? 'See what’s in this update ↗' : 'Browse recent commits ↗'}
          </a>
        )}
      </div>

      {applying && (
        <p className="text-sm text-content-muted" role="status">
          {phase === 'restarting'
            ? '↻ Updated — the app is restarting. This page reloads automatically when it’s back…'
            : '⬇ Pulling the latest version…'}
        </p>
      )}

      {!applying && s && (
        <div className="text-sm">
          {canPull ? (
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-content">
                <span aria-hidden>⬆</span>{' '}
                {typeof s.behind === 'number'
                  ? `${s.behind} commit${s.behind === 1 ? '' : 's'} behind${s.current_sha && s.remote_sha ? ` (${s.current_sha} → ${s.remote_sha})` : ''}.`
                  : `Update available${s.latest ? ` — v${s.latest}` : ''}.`}
              </span>
              <button type="button" onClick={apply}
                className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white transition-transform hover:-translate-y-px">
                Update &amp; restart
              </button>
            </div>
          ) : s.update_available ? (
            <p className="text-content">
              Update available{s.latest ? ` — v${s.latest}` : ''} —{' '}
              <a href={s.url} target="_blank" rel="noreferrer" className="font-semibold text-emerald-300 underline">
                download the latest release
              </a>{' '}and replace the folder.
            </p>
          ) : s.ok ? (
            <p className="text-emerald-400"><span aria-hidden>✓</span> You’re up to date.</p>
          ) : (
            <p className="text-content-muted"><span aria-hidden>⚠</span> {s.reason || 'Could not check for updates.'}</p>
          )}
        </div>
      )}
    </Card>
  )
}

const LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1']

/* Server bind (host/port/LAN access). host/port live in config.server and are only
   read by run.py at PROCESS START — Flask can't rebind mid-request — so this card
   contrasts the SAVED config against `runtime` (what's actually bound right now,
   stamped by run.py) and offers a one-click save-then-restart, mirroring
   UpdatesCard's "poll /api/health, then hard-reload" pattern. */
function ServerCard({ config, setField, runtime, onSaveThenRestart }) {
  const toast = useToast()
  const [restarting, setRestarting] = useState(false)
  const [copied, setCopied] = useState(false)
  const lan = !LOOPBACK_HOSTS.includes(config.server.host)
  const knownRuntime = runtime.host != null && runtime.port != null
  const dirty = knownRuntime && (runtime.host !== config.server.host || runtime.port !== config.server.port)

  const waitForHealthAndReload = async () => {
    for (let i = 0; i < 120; i += 1) {
      await new Promise((r) => setTimeout(r, 1000))
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (res.ok) { window.location.reload(); return }
      } catch { /* still restarting — keep waiting */ }
    }
    setRestarting(false)   // gave up after ~2 min
  }

  const restart = async () => {
    setRestarting(true)
    // Save first: "Restart to apply" must apply what's on screen, not whatever
    // was last persisted — otherwise a restart right after editing the port
    // would silently come back on the OLD port with no visible change.
    if (!(await onSaveThenRestart())) { setRestarting(false); return }
    try {
      await postJson('/api/settings/restart', {})
      waitForHealthAndReload()
    } catch (e) {
      toast.error(e.message || 'Restart failed')
      setRestarting(false)
    }
  }

  const regenerateToken = () => {
    const bytes = crypto.getRandomValues(new Uint8Array(24))
    const token = btoa(String.fromCharCode(...bytes)).replace(/[+/=]/g, '')
    setField('server', 'access_token', token)
  }

  const copyToken = async () => {
    try {
      await navigator.clipboard.writeText(config.server.access_token || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard unavailable (non-HTTPS remote origin) — token stays selectable */ }
  }

  return (
    <Card title="Server"
      help="Where the app listens. Host/port and LAN access need a restart to take effect — edit below, then use “Restart to apply”.">
      <div>
        <label htmlFor="server-port" className="block text-sm font-medium text-content">Port</label>
        <input id="server-port" type="number" min={1} max={65535}
          value={config.server.port ?? ''}
          onChange={(e) => setField('server', 'port', Math.max(1, Math.min(65535, Number(e.target.value) || 1)))}
          className={`${INPUT_CLASS} max-w-[8rem]`} />
      </div>

      <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
        <div>
          <p className="text-sm font-medium text-content">Available on the local network</p>
          <p className="mt-0.5 text-xs text-content-muted">
            Off (default): only this computer can open the app. On: any device on your
            Wi-Fi/LAN can reach it — e.g. from your phone — guarded by the access token below.
          </p>
        </div>
        <button type="button" role="switch" aria-checked={lan}
          onClick={() => setField('server', 'host', lan ? '127.0.0.1' : '0.0.0.0')}
          aria-label="Available on the local network"
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${lan ? 'bg-emerald-500' : 'bg-surface border border-border-strong'}`}>
          <span aria-hidden
            className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${lan ? 'translate-x-5' : 'translate-x-0.5'}`} />
        </button>
      </div>

      {lan && (
        <div>
          <div className="flex items-center justify-between">
            <label htmlFor="server-token" className="block text-sm font-medium text-content">Access token</label>
            <button type="button" onClick={regenerateToken}
              className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
              Generate new token
            </button>
          </div>
          <p className="mb-1 text-xs text-content-muted">
            Remote devices present this once (via the URL below) — a signed session cookie
            takes over from there. Requests from this computer never need it.
          </p>
          <div className="flex gap-2">
            <input id="server-token" type="text" readOnly
              value={config.server.access_token || '(created automatically the first time LAN access is used — or click “Generate new token”)'}
              className={`${INPUT_CLASS} font-mono text-xs`} />
            {config.server.access_token && (
              <button type="button" onClick={copyToken}
                className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised">
                {copied ? 'Copied ✓' : 'Copy'}
              </button>
            )}
          </div>
          {config.server.access_token && (
            <p className="mt-1 break-all text-xs text-content-subtle">
              From another device: <code className="text-content">http://&lt;this-computer&gt;:{config.server.port}/?token={config.server.access_token}</code>
            </p>
          )}
        </div>
      )}

      {knownRuntime && (
        <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2 text-xs ${
          dirty ? 'border-amber-400/50 bg-amber-400/10' : 'border-border bg-surface-raised'}`}>
          <span className="text-content-muted">
            Running: <span className="font-medium text-content">{runtime.host}:{runtime.port}</span>
            {dirty && (
              <> · Saved: <span className="font-medium text-content">{config.server.host}:{config.server.port}</span></>
            )}
          </span>
          {dirty ? (
            <button type="button" onClick={restart} disabled={restarting}
              className="ml-auto shrink-0 rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
              {restarting ? '↻ Restarting…' : 'Save & restart to apply'}
            </button>
          ) : (
            <span className="ml-auto text-emerald-400"><span aria-hidden>✓</span> Running config matches saved config</span>
          )}
        </div>
      )}
    </Card>
  )
}

/* Server-log viewer: tail data/app.log (fallback data/server.log) so an error
   can be copy-pasted into a bug report without hunting for files. Fetches on
   open, auto-refreshes every 5 s while open. */
function LogViewer() {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState(null)
  const [lines, setLines] = useState([])
  const load = async () => {
    try {
      const d = await apiFetch('/api/logs/tail?n=300')
      setFile(d.file); setLines(d.lines || [])
    } catch { /* viewer is best-effort */ }
  }
  useEffect(() => {
    if (!open) return undefined
    load()
    const id = setInterval(load, 5000)
    return () => clearInterval(id)
  }, [open])
  const copy = () => { try { navigator.clipboard.writeText(lines.join('\n')) } catch { /* ignore */ } }
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex w-full items-center gap-2 text-left">
        <h2 className="text-base font-semibold text-content">🪵 Server log</h2>
        <span className="text-xs text-content-subtle">
          {open ? (file ? `data/${file} — last ${lines.length} lines, refreshes every 5 s` : 'no log file yet')
            : 'something failed? open this and copy the log into your bug report'}
        </span>
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          <div className="flex gap-2">
            <button type="button" onClick={load}
              className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs text-content">
              ↻ Refresh
            </button>
            <button type="button" onClick={copy} disabled={!lines.length}
              className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs text-content disabled:opacity-40">
              📋 Copy all
            </button>
          </div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-app/60 p-2 text-[11px] leading-snug text-content-muted">
            {lines.length ? lines.join('\n') : 'Log is empty.'}
          </pre>
        </div>
      )}
    </section>
  )
}
