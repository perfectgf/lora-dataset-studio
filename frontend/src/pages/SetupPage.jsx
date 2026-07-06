import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiFetch, putJson, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import { deriveSetupSteps, deriveCapabilitySummary } from '../hooks/useSetupSteps'
import SetupStep from '../components/setup/SetupStep'
import GuidedSteps from '../components/setup/GuidedSteps'
import InstallRunner from '../components/setup/InstallRunner'

const SKIP_KEY = 'setupSkipped'
const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none'

const KEY_FIELDS = [
  { key: 'GEMINI_API_KEY', label: 'Gemini API key', engine: 'nanobanana',
    href: 'https://aistudio.google.com/apikey', help: 'Powers Nano Banana.' },
  { key: 'OPENAI_API_KEY', label: 'OpenAI API key', engine: 'chatgpt',
    href: 'https://platform.openai.com/api-keys', help: 'Powers ChatGPT (gpt-image-2).' },
]

function loadSkipped() {
  try { return new Set(JSON.parse(localStorage.getItem(SKIP_KEY) || '[]')) }
  catch { return new Set() }
}

export default function SetupPage() {
  const toast = useToast()
  const { caps, refresh } = useCapabilities()
  const [config, setConfig] = useState(null)
  const [secretsPresence, setSecretsPresence] = useState({})
  const [secretInputs, setSecretInputs] = useState({})
  const [skipped, setSkipped] = useState(loadSkipped)
  const [busy, setBusy] = useState(false)
  const [loadError, setLoadError] = useState(false)

  const load = useCallback(async () => {
    try {
      const data = await apiFetch('/api/settings')
      setConfig(data.config); setSecretsPresence(data.secrets); setLoadError(false)
    } catch (e) { setLoadError(true); toast.error(`Failed to load settings: ${e.message}`) }
  }, [toast])
  useEffect(() => { load() }, [load])

  const steps = useMemo(() => deriveSetupSteps(caps), [caps])
  const summary = useMemo(() => deriveCapabilitySummary(caps), [caps])
  const readyCount = summary.filter((s) => s.ok).length

  const setField = (section, key, value) =>
    setConfig((prev) => ({ ...prev, [section]: { ...prev[section], [key]: value } }))

  const persistSkip = (next) => {
    setSkipped(next)
    localStorage.setItem(SKIP_KEY, JSON.stringify([...next]))
  }
  const skip = (id) => persistSkip(new Set(skipped).add(id))
  const unskip = (id) => { const n = new Set(skipped); n.delete(id); persistSkip(n) }

  // Single write path: persist config + typed secrets, then force a re-probe so
  // every card's status recomputes from fresh capabilities.
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

  const testTarget = async (target) => {
    try {
      const r = await postJson(`/api/settings/test/${target}`, {})
      r.ok ? toast.success(r.detail) : toast.warning(r.detail)
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

  // Skip collapses any not-yet-ready card (available OR partial); a card that
  // reaches 'ready' always shows ready regardless of a prior skip.
  const effective = (step) => (step.status !== 'ready' && skipped.has(step.id) ? 'skipped' : step.status)
  const stepById = Object.fromEntries(steps.map((s) => [s.id, s]))

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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-content">Setup</h1>
        <p className="mt-1 text-sm text-content-muted">
          You can already build a dataset from your own photos — no setup required.
          Everything below unlocks more. Set up what you need, skip the rest.
        </p>
        <p className="mt-2 text-xs text-content-subtle">{readyCount} of {summary.length} capabilities ready</p>
      </div>

      {/* 1. Image generation */}
      <SetupStep step={stepById.image} index={1} effectiveStatus={effective(stepById.image)}>
        <div className="space-y-4">
          {KEY_FIELDS.map((f) => (
            <div key={f.key}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-content">{f.label}</span>
                <span className={`text-xs ${stepById.image.engines[f.engine] ? 'text-emerald-400' : 'text-content-subtle'}`}>
                  {stepById.image.engines[f.engine] ? '✓ Ready' : '○ Not set'}
                </span>
              </div>
              <p className="text-xs text-content-muted">{f.help}</p>
              <input type="password" autoComplete="off" className={INPUT_CLASS}
                value={secretInputs[f.key] ?? ''}
                placeholder={secretsPresence[f.key] ? 'Already set — enter a new value to replace it' : 'Paste your key'}
                onChange={(e) => setSecretInputs((p) => ({ ...p, [f.key]: e.target.value }))} />
              <div className="mt-1 flex items-center gap-3">
                <a href={f.href} target="_blank" rel="noreferrer" className="text-xs text-primary underline">Get a key</a>
                <button type="button" onClick={() => testTarget(f.engine === 'nanobanana' ? 'gemini' : 'openai')}
                  className="text-xs text-content-muted underline">Test</button>
              </div>
            </div>
          ))}
          <p className="text-xs text-content-subtle">Klein (local) is set up in step 2 (ComfyUI).</p>
          {saveRecheckBtn}
        </div>
      </SetupStep>

      {/* 2. ComfyUI */}
      <SetupStep step={stepById.comfyui} index={2} effectiveStatus={effective(stepById.comfyui)}
        onSkip={() => skip('comfyui')} onUnskip={() => unskip('comfyui')}>
        <GuidedSteps
          intro="ComfyUI is a local image generator. Install it once, then point the app at it."
          steps={[
            { text: 'Clone ComfyUI and follow its README to install it.', command: 'git clone https://github.com/comfyanonymous/ComfyUI' },
            { text: 'Start it (defaults to port 8188).' },
          ]}
          link={{ href: 'https://github.com/comfyanonymous/ComfyUI', label: 'ComfyUI on GitHub →' }}>
          {guidedField('ComfyUI API URL', 'comfyui', 'api_url', 'http://127.0.0.1:8188')}
          {guidedField('ComfyUI install directory', 'comfyui', 'base_dir', 'C:\\ComfyUI')}
          {stepById.comfyui.reachable && !stepById.comfyui.hasKlein && (
            <p className="text-xs text-amber-400">Reachable, but no Klein model found. Place it in &lt;ComfyUI&gt;/models/unet/klein/.</p>
          )}
          {saveRecheckBtn}
        </GuidedSteps>
      </SetupStep>

      {/* 3. Ollama */}
      <SetupStep step={stepById.ollama} index={3} effectiveStatus={effective(stepById.ollama)}
        onSkip={() => skip('ollama')} onUnskip={() => unskip('ollama')}>
        <GuidedSteps
          intro="Ollama runs local models for captioning and auto-framing."
          steps={[{ text: 'Install Ollama and start it (defaults to port 11434).' }]}
          link={{ href: 'https://ollama.com/download', label: 'Download Ollama →' }}>
          {guidedField('Ollama URL', 'ollama', 'url', 'http://127.0.0.1:11434')}
          {guidedField('Vision model', 'ollama', 'vision_model', 'qwen3-vl:8b')}
          {saveRecheckBtn}
          {stepById.ollama.reachable && !stepById.ollama.visionModelReady && (
            <div className="pt-2">
              <p className="mb-1 text-xs text-content-muted">Pull the vision model:</p>
              <InstallRunner action="ollama_model" buttonLabel={`Pull ${stepById.ollama.visionModel || 'model'}`}
                manualCommand={`ollama pull ${stepById.ollama.visionModel || 'qwen3-vl:8b'}`}
                onDone={() => refresh(true)} />
            </div>
          )}
        </GuidedSteps>
      </SetupStep>

      {/* 4. Quality tools */}
      <SetupStep step={stepById.quality} index={4} effectiveStatus={effective(stepById.quality)}
        onSkip={() => skip('quality')} onUnskip={() => unskip('quality')}>
        <p className="text-sm text-content-muted">
          Installs the Python ML extras (insightface, onnxruntime, rembg) into this app's environment.
        </p>
        <InstallRunner action="ml_extras" buttonLabel="Install (pip)"
          manualCommand="pip install -r backend/requirements-ml.txt" onDone={() => refresh(true)} />
      </SetupStep>

      {/* 5. ai-toolkit */}
      <SetupStep step={stepById.training} index={5} effectiveStatus={effective(stepById.training)}
        onSkip={() => skip('training')} onUnskip={() => unskip('training')}>
        <GuidedSteps
          intro="ai-toolkit trains the LoRA. Install it once, then point the app at its folder."
          steps={[
            { text: 'Clone ai-toolkit and set up its venv per its README.', command: 'git clone https://github.com/ostris/ai-toolkit' },
          ]}
          link={{ href: 'https://github.com/ostris/ai-toolkit', label: 'ai-toolkit on GitHub →' }}>
          {guidedField('ai-toolkit directory', 'aitoolkit', 'dir', 'C:\\ai-toolkit')}
          {saveRecheckBtn}
        </GuidedSteps>
      </SetupStep>

      {/* Summary */}
      <section className="rounded-xl border border-border bg-surface p-5">
        <h2 className="text-base font-semibold text-content">Summary</h2>
        <ul className="mt-3 grid gap-1.5 sm:grid-cols-2">
          {summary.map((s) => (
            <li key={s.label} className={`flex items-center gap-2 text-sm ${s.ok ? 'text-content' : 'text-content-subtle'}`}>
              <span aria-hidden="true" className={s.ok ? 'text-emerald-400' : 'text-content-subtle'}>{s.ok ? '✓' : '✗'}</span>
              {s.label}
            </li>
          ))}
        </ul>
        <Link to="/datasets" className="mt-4 inline-block rounded-lg bg-gradient-primary px-4 py-2 text-sm font-semibold text-white">
          Build your first dataset →
        </Link>
      </section>
    </div>
  )
}
