import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiFetch, putJson, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import { deriveSetupSteps, deriveCapabilitySummary, SETUP_STEP_IDS } from '../hooks/useSetupSteps'
import GuidedSteps from '../components/setup/GuidedSteps'
import InstallRunner from '../components/setup/InstallRunner'

const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none'

const KEY_FIELDS = [
  { key: 'GEMINI_API_KEY', label: 'Gemini API key', engine: 'nanobanana',
    href: 'https://aistudio.google.com/apikey', help: 'Powers Nano Banana.' },
  { key: 'OPENAI_API_KEY', label: 'OpenAI API key', engine: 'chatgpt',
    href: 'https://platform.openai.com/api-keys', help: 'Powers ChatGPT (gpt-image-2).' },
]

// A wizard "screen" is the welcome/scan, one per setup tool, then done.
const SCREENS = ['welcome', ...SETUP_STEP_IDS, 'done']
const TOTAL_TOOLS = SETUP_STEP_IDS.length

const STATUS_META = {
  ready: { glyph: '✓', label: 'Ready', cls: 'text-emerald-400' },
  partial: { glyph: '◐', label: 'Almost there', cls: 'text-amber-400' },
  available: { glyph: '○', label: 'Not set up', cls: 'text-content-subtle' },
}

export default function SetupPage() {
  const toast = useToast()
  const { caps, refresh } = useCapabilities()
  const [config, setConfig] = useState(null)
  const [secretsPresence, setSecretsPresence] = useState({})
  const [secretInputs, setSecretInputs] = useState({})
  const [busy, setBusy] = useState(false)
  const [loadError, setLoadError] = useState(false)
  const [detected, setDetected] = useState(null)   // autodetect result (path suggestions)
  const [detecting, setDetecting] = useState(false)
  const [scanned, setScanned] = useState(false)     // the on-load scan has completed at least once
  const [screen, setScreen] = useState(0)           // index into SCREENS
  const autodetectedRef = useRef(false)             // run the on-load autodetect only once

  const load = useCallback(async () => {
    try {
      const data = await apiFetch('/api/settings')
      setConfig(data.config); setSecretsPresence(data.secrets); setLoadError(false)
    } catch (e) { setLoadError(true); toast.error(`Failed to load settings: ${e.message}`) }
  }, [toast])
  useEffect(() => { load() }, [load])

  // Auto-detect installed tools. Reachable default ports (Ollama 11434, ComfyUI
  // 8188) are safe to fill + save automatically; disk-scanned paths are only
  // SUGGESTED (a scan can guess wrong) and applied on the user's click.
  const runAutodetect = useCallback(async (baseConfig) => {
    setDetecting(true)
    try {
      const d = await apiFetch('/api/setup/autodetect')
      setDetected(d)
      const next = JSON.parse(JSON.stringify(baseConfig))
      let changed = false
      const fillEmpty = (section, key, val) => {
        if (val && !(next[section] && next[section][key])) {
          next[section] = { ...(next[section] || {}), [key]: val }; changed = true
        }
      }
      fillEmpty('ollama', 'url', d.ollama && d.ollama.url)
      fillEmpty('ollama', 'vision_model', d.ollama && d.ollama.vision_model)
      fillEmpty('comfyui', 'api_url', d.comfyui && d.comfyui.api_url)
      if (changed) {
        const saved = await putJson('/api/settings', { config: next })
        setConfig(saved.config)
      }
      await refresh(true)
      return d
    } catch { return null }
    finally { setDetecting(false); setScanned(true) }
  }, [refresh])

  // The scan runs BY ITSELF the moment settings load — the user watches it on the
  // welcome screen, no button required.
  useEffect(() => {
    if (config && !autodetectedRef.current) { autodetectedRef.current = true; runAutodetect(config) }
  }, [config, runAutodetect])

  // Apply a disk-scanned path suggestion (user-confirmed) into config + save.
  const applyDetectedPath = async (section, key, val) => {
    const next = { ...config, [section]: { ...config[section], [key]: val } }
    try {
      const saved = await putJson('/api/settings', { config: next })
      setConfig(saved.config); await refresh(true); toast.success('Applied.')
    } catch (e) { toast.error(`Save failed: ${e.message}`) }
  }

  const steps = useMemo(() => deriveSetupSteps(caps), [caps])
  const summary = useMemo(() => deriveCapabilitySummary(caps), [caps])
  const readyCount = summary.filter((s) => s.ok).length
  const stepById = useMemo(() => Object.fromEntries(steps.map((s) => [s.id, s])), [steps])

  const setField = (section, key, value) =>
    setConfig((prev) => ({ ...prev, [section]: { ...prev[section], [key]: value } }))

  // Single write path: persist config + typed secrets, then force a re-probe so
  // every step's status recomputes from fresh capabilities.
  const persist = async () => {
    setBusy(true)
    try {
      const secrets = Object.fromEntries(
        Object.entries(secretInputs).map(([k, v]) => [k, (v || '').trim()]).filter(([, v]) => v)
      )
      const data = await putJson('/api/settings', { config, secrets })
      setConfig(data.config); setSecretsPresence(data.secrets); setSecretInputs({})
      await refresh(true)
      toast.success('Saved.')
    } catch (e) { toast.error(`Save failed: ${e.message}`) }
    finally { setBusy(false) }
  }

  // Test the key the user JUST typed. The probe reads the SAVED secret, so save
  // that one key first (no need to fill anything else), then test + re-probe so
  // the step flips to Ready. With no typed value, test whatever is already saved.
  const saveSecretThenTest = async (key, target) => {
    const typed = (secretInputs[key] || '').trim()
    try {
      if (typed) {
        const data = await putJson('/api/settings', { secrets: { [key]: typed } })
        setSecretsPresence(data.secrets); setSecretInputs((p) => ({ ...p, [key]: '' }))
      }
      const r = await postJson(`/api/settings/test/${target}`, {})
      r.ok ? toast.success(r.detail) : toast.warning(r.detail)
      await refresh(true)
    } catch (e) { toast.error(e.message) }
  }

  if (!config) {
    return loadError ? (
      <div className="space-y-3">
        <p className="text-content-muted">Couldn't load setup.</p>
        <button type="button" onClick={load}
          className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised">
          Retry
        </button>
      </div>
    ) : (
      <p className="text-content-muted">Loading setup…</p>
    )
  }

  const guidedField = (label, section, key, placeholder) => (
    <label className="block text-sm">
      <span className="font-medium text-content">{label}</span>
      <input className={INPUT_CLASS} value={config[section][key] ?? ''} placeholder={placeholder}
        onChange={(e) => setField(section, key, e.target.value)} />
    </label>
  )
  const saveRecheckBtn = (
    <button type="button" onClick={persist} disabled={busy}
      className="mt-1 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50">
      {busy ? 'Saving…' : 'Save & re-check'}
    </button>
  )
  // "Found on disk: <path> — Use" chip for a scanned path we didn't auto-apply.
  const detectedPathChip = (section, key) => {
    const val = detected && detected[section] && detected[section][key]
    if (!val || (config[section] && config[section][key]) === val) return null
    return (
      <button type="button" onClick={() => applyDetectedPath(section, key, val)}
        className="mt-1 block text-left text-xs text-primary underline">
        Found on disk: <span className="font-mono">{val}</span> — Use
      </button>
    )
  }

  // --- Per-tool step body (reuses the existing controls, one tool per screen) ---
  const toolBody = (id) => {
    const step = stepById[id]
    if (id === 'image') {
      return (
        <div className="space-y-4">
          {KEY_FIELDS.map((f) => (
            <div key={f.key}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-content">{f.label}</span>
                <span className={`text-xs ${step.engines[f.engine] ? 'text-emerald-400' : 'text-content-subtle'}`}>
                  {step.engines[f.engine] ? '✓ Ready' : '○ Not set'}
                </span>
              </div>
              <p className="text-xs text-content-muted">{f.help}</p>
              <input type="password" autoComplete="off" className={INPUT_CLASS}
                value={secretInputs[f.key] ?? ''}
                placeholder={secretsPresence[f.key] ? 'Already set — enter a new value to replace it' : 'Paste your key'}
                onChange={(e) => setSecretInputs((p) => ({ ...p, [f.key]: e.target.value }))} />
              <div className="mt-1 flex items-center gap-3">
                <a href={f.href} target="_blank" rel="noreferrer" className="text-xs text-primary underline">Get a key</a>
                <button type="button" onClick={() => saveSecretThenTest(f.key, f.engine === 'nanobanana' ? 'gemini' : 'openai')}
                  className="text-xs text-content-muted underline">Save &amp; test</button>
              </div>
            </div>
          ))}
          <p className="text-xs text-content-subtle">Klein (local) needs ComfyUI — the next step.</p>
          {saveRecheckBtn}
        </div>
      )
    }
    if (id === 'comfyui') {
      return (
        <GuidedSteps
          intro="ComfyUI is a local image generator. Install it once, then point the app at it."
          steps={[
            { text: 'Clone ComfyUI and follow its README to install it.', command: 'git clone https://github.com/comfyanonymous/ComfyUI' },
            { text: 'Start it (defaults to port 8188).' },
          ]}
          link={{ href: 'https://github.com/comfyanonymous/ComfyUI', label: 'ComfyUI on GitHub →' }}>
          {guidedField('ComfyUI API URL', 'comfyui', 'api_url', 'http://127.0.0.1:8188')}
          {guidedField('ComfyUI install directory', 'comfyui', 'base_dir', 'C:\\ComfyUI')}
          {detectedPathChip('comfyui', 'base_dir')}
          {step.reachable && !step.hasKlein && (
            <p className="text-xs text-amber-400">Reachable, but no Klein model found. Place it in &lt;ComfyUI&gt;/models/unet/klein/.</p>
          )}
          {saveRecheckBtn}
        </GuidedSteps>
      )
    }
    if (id === 'ollama') {
      return (
        <GuidedSteps
          intro="Ollama runs local models for captioning and auto-framing."
          steps={[{ text: 'Install Ollama and start it (defaults to port 11434).' }]}
          link={{ href: 'https://ollama.com/download', label: 'Download Ollama →' }}>
          {guidedField('Ollama URL', 'ollama', 'url', 'http://127.0.0.1:11434')}
          {guidedField('Vision model', 'ollama', 'vision_model', 'qwen3-vl:8b')}
          {saveRecheckBtn}
          {step.reachable && !step.visionModelReady && (
            <div className="pt-2">
              <p className="mb-1 text-xs text-content-muted">Pull the vision model:</p>
              <InstallRunner action="ollama_model" buttonLabel={`Pull ${step.visionModel || 'model'}`}
                manualCommand={`ollama pull ${step.visionModel || 'qwen3-vl:8b'}`}
                onDone={() => refresh(true)} />
            </div>
          )}
        </GuidedSteps>
      )
    }
    if (id === 'quality') {
      return (
        <div className="space-y-3">
          <p className="text-sm text-content-muted">
            Installs the Python ML extras (insightface, onnxruntime, rembg) into this app's environment.
          </p>
          <InstallRunner action="ml_extras" buttonLabel="Install (pip)"
            manualCommand="pip install -r backend/requirements-ml.txt" onDone={() => refresh(true)} />
        </div>
      )
    }
    // training
    return (
      <GuidedSteps
        intro="ai-toolkit trains the LoRA. Install it once, then point the app at its folder."
        steps={[
          { text: 'Clone ai-toolkit and set up its venv per its README.', command: 'git clone https://github.com/ostris/ai-toolkit' },
        ]}
        link={{ href: 'https://github.com/ostris/ai-toolkit', label: 'ai-toolkit on GitHub →' }}>
        {guidedField('ai-toolkit directory', 'aitoolkit', 'dir', 'C:\\ai-toolkit')}
        {detectedPathChip('aitoolkit', 'dir')}
        {saveRecheckBtn}
      </GuidedSteps>
    )
  }

  const goNext = () => setScreen((i) => Math.min(i + 1, SCREENS.length - 1))
  const goBack = () => setScreen((i) => Math.max(i - 1, 0))
  const kind = SCREENS[screen]

  // Progress dots: one per tool step, filled when that tool is ready.
  const ProgressDots = () => (
    <div className="flex items-center gap-1.5" aria-hidden="true">
      {SETUP_STEP_IDS.map((id, i) => {
        const active = kind === id
        const ready = stepById[id].status === 'ready'
        return (
          <span key={id}
            className={`h-2 rounded-full transition-all ${active ? 'w-6 bg-primary'
              : ready ? 'w-2 bg-emerald-400' : 'w-2 bg-border-strong'}`} />
        )
      })}
    </div>
  )

  const skipLink = (
    <Link to="/datasets" className="text-xs text-content-subtle underline hover:text-content">
      Skip setup — I'll do it later
    </Link>
  )

  // --- Welcome + live machine scan --------------------------------------------
  if (kind === 'welcome') {
    const scanRows = [
      { label: 'Local generation — ComfyUI', ok: stepById.comfyui.reachable,
        hint: detected && detected.comfyui && detected.comfyui.base_dir },
      { label: 'Captioning — Ollama', ok: stepById.ollama.reachable,
        hint: detected && detected.ollama && detected.ollama.url },
      { label: 'LoRA training — ai-toolkit', ok: stepById.training.valid,
        hint: detected && detected.aitoolkit && detected.aitoolkit.dir },
    ]
    return (
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="text-center">
          <div className="text-3xl" aria-hidden="true">🧬</div>
          <h1 className="mt-2 text-2xl font-bold text-content">Welcome to LoRA Dataset Studio</h1>
          <p className="mt-2 text-sm text-content-muted">
            Let's set up your machine. I'll scan what's already installed and help you install the rest —
            you can also start building a dataset from your own photos right now, no setup required.
          </p>
        </div>

        <section className="rounded-xl border border-border bg-surface p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-content">
              {detecting ? 'Scanning your machine…' : 'Machine scan'}
            </h2>
            {detecting
              ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-border-strong border-t-primary" aria-hidden="true" />
              : (
                <button type="button" onClick={() => runAutodetect(config)}
                  className="text-xs text-primary underline">Re-scan</button>
              )}
          </div>
          <ul className="mt-4 space-y-2">
            {scanRows.map((r) => (
              <li key={r.label} className="flex items-center justify-between gap-3 text-sm">
                <span className="flex items-center gap-2">
                  <span aria-hidden="true" className={detecting ? 'text-content-subtle'
                    : r.ok ? 'text-emerald-400' : 'text-content-subtle'}>
                    {detecting ? '…' : r.ok ? '✓' : '✗'}
                  </span>
                  <span className={r.ok ? 'text-content' : 'text-content-muted'}>{r.label}</span>
                </span>
                <span className="truncate text-right font-mono text-xs text-content-subtle">
                  {detecting ? '' : r.ok ? 'found' : (r.hint ? 'installed — needs a click' : 'not found')}
                </span>
              </li>
            ))}
          </ul>
          {scanned && !detecting && (
            <p className="mt-3 text-xs text-content-subtle">
              {readyCount} of {summary.length} capabilities ready. Reachable services were filled in automatically.
            </p>
          )}
        </section>

        <div className="flex items-center justify-between">
          {skipLink}
          <button type="button" onClick={goNext}
            className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white">
            Start setup →
          </button>
        </div>
      </div>
    )
  }

  // --- Done / summary ----------------------------------------------------------
  if (kind === 'done') {
    return (
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="text-center">
          <div className="text-3xl" aria-hidden="true">🎉</div>
          <h1 className="mt-2 text-2xl font-bold text-content">You're all set</h1>
          <p className="mt-1 text-sm text-content-muted">{readyCount} of {summary.length} capabilities ready.</p>
        </div>
        <section className="rounded-xl border border-border bg-surface p-5">
          <h2 className="text-base font-semibold text-content">What's unlocked</h2>
          <ul className="mt-3 grid gap-1.5 sm:grid-cols-2">
            {summary.map((s) => (
              <li key={s.label} className={`flex items-center gap-2 text-sm ${s.ok ? 'text-content' : 'text-content-subtle'}`}>
                <span aria-hidden="true" className={s.ok ? 'text-emerald-400' : 'text-content-subtle'}>{s.ok ? '✓' : '✗'}</span>
                {s.label}
              </li>
            ))}
          </ul>
        </section>
        <div className="flex items-center justify-between">
          <button type="button" onClick={goBack} className="text-xs text-content-subtle underline hover:text-content">
            ← Back
          </button>
          <Link to="/datasets" className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white">
            Build your first dataset →
          </Link>
        </div>
      </div>
    )
  }

  // --- A single tool step ------------------------------------------------------
  const step = stepById[kind]
  const stepNo = SETUP_STEP_IDS.indexOf(kind) + 1
  const meta = STATUS_META[step.status] || STATUS_META.available
  const isLastTool = stepNo === TOTAL_TOOLS
  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <div className="flex items-center justify-between">
        <ProgressDots />
        <span className="text-xs text-content-subtle">Step {stepNo} of {TOTAL_TOOLS}</span>
      </div>

      <section className="rounded-xl border border-border bg-surface p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold text-content">
              {step.title}
              {step.recommended && (
                <span className="ml-2 rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                  Recommended
                </span>
              )}
            </h1>
            <p className="mt-1 text-xs text-content-muted">Unlocks: {step.unlocks.join(' · ')}</p>
          </div>
          <span className={`inline-flex shrink-0 items-center gap-1 text-xs font-medium ${meta.cls}`}>
            <span aria-hidden="true">{meta.glyph}</span>{meta.label}
          </span>
        </div>
        <div className="mt-4">{toolBody(kind)}</div>
      </section>

      <div className="flex items-center justify-between">
        <button type="button" onClick={goBack} className="text-xs text-content-subtle underline hover:text-content">
          ← Back
        </button>
        <div className="flex items-center gap-4">
          {skipLink}
          <button type="button" onClick={goNext}
            className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white">
            {isLastTool ? 'Finish →' : (step.status === 'ready' ? 'Next →' : 'Skip / Next →')}
          </button>
        </div>
      </div>
    </div>
  )
}
