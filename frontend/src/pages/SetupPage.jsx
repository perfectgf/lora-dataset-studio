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

// Default local vision model + rough VRAM notes surfaced in the wizard. The
// ABLITERATED Qwen3-VL is required — vanilla qwen3-vl refuses to caption the NSFW
// concept datasets this app targets. VRAM figures are approximate minimums for the
// fp8/q4 builds (Klein 9B fp8 fits a 24 GB RTX 4090; the 8B vision model ~8 GB).
const DEFAULT_VISION_MODEL = 'huihui_ai/qwen3-vl-abliterated:8b'
const VISION_MODEL_VRAM = '≈ 8 GB VRAM'
const KLEIN_MODEL_VRAM = '≈ 16 GB VRAM (fp8; ~29 GB at bf16)'

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
  const [advancing, setAdvancing] = useState(false) // Next is mid save-&-recheck
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
      // base_dir drives every training-base lister (get_checkpoint_models /
      // get_zimage_models resolve from comfyui.base_dir/models). The detector
      // only reports a folder that HAS main.py + models/, so it's a real ComfyUI
      // install — safe to auto-apply, not just suggest. Without this, a reachable
      // ComfyUI still shows "No SDXL checkpoint found" until the user clicks the chip.
      fillEmpty('comfyui', 'base_dir', d.comfyui && d.comfyui.base_dir)
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
      const fields = (
        <>
          {guidedField('ComfyUI API URL', 'comfyui', 'api_url', 'http://127.0.0.1:8188')}
          {guidedField('ComfyUI install directory', 'comfyui', 'base_dir', 'C:\\ComfyUI')}
          {detectedPathChip('comfyui', 'base_dir')}
          {/* Validate the folder on Save & re-check: it must actually hold main.py +
              models/. A portable-wrapper path is auto-corrected to the nested ComfyUI on
              save (so checkpoints are found); a genuinely wrong path is flagged here. */}
          {config.comfyui.base_dir && (
            step.dirValid ? (
              <p className="text-xs text-emerald-400">
                ✓ ComfyUI found{step.resolvedDir ? <> at <span className="font-mono">{step.resolvedDir}</span></> : ''}.
              </p>
            ) : (
              <p className="text-xs text-amber-400">
                ⚠ No ComfyUI install in this folder — it must contain <span className="font-mono">main.py</span> and
                a <span className="font-mono">models/</span> folder. Check the path, then Save &amp; re-check.
                For the portable build, point at the inner <span className="font-mono">…\ComfyUI_windows_portable\ComfyUI</span>.
              </p>
            )
          )}
          {step.reachable && !step.hasKlein && (
            <div className="space-y-1 text-xs text-content-muted">
              <p>
                Running. The Klein model is <span className="text-content font-medium">optional</span> — add it only if you want
                local generation (you can also use the API engines or your own photos, then export to train elsewhere).
                To enable it, download <span className="font-mono">flux-2-klein-9b-fp8.safetensors</span> ({KLEIN_MODEL_VRAM}) into
                <span className="font-mono"> &lt;ComfyUI&gt;/models/unet/klein/</span>.
              </p>
              <p className="flex flex-wrap gap-x-4 gap-y-1">
                <a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8" target="_blank" rel="noreferrer"
                  className="text-primary underline">Download the Klein model →</a>
                <a href="https://docs.comfy.org/tutorials/flux/flux-2-klein" target="_blank" rel="noreferrer"
                  className="text-primary underline">ComfyUI setup guide →</a>
              </p>
            </div>
          )}
          {saveRecheckBtn}
        </>
      )
      // Already detected/running → skip the from-scratch install guide; show the
      // reachable confirmation and only the remaining gap.
      if (step.reachable) {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
              ✓ ComfyUI is already running at <span className="font-mono">{step.apiUrl || 'the configured URL'}</span>.
              {step.hasKlein ? ' Nothing to do here.' : ' It works — the Klein model (optional, for local generation) isn’t installed.'}
            </div>
            {fields}
          </div>
        )
      }
      return (
        <GuidedSteps
          intro="ComfyUI is a local image generator. Install it once, then point the app at it."
          steps={[
            { text: 'Clone ComfyUI and follow its README to install it.', command: 'git clone https://github.com/comfyanonymous/ComfyUI' },
            { text: 'Start it (defaults to port 8188).' },
          ]}
          link={{ href: 'https://github.com/comfyanonymous/ComfyUI', label: 'ComfyUI on GitHub →' }}>
          {fields}
        </GuidedSteps>
      )
    }
    if (id === 'ollama') {
      // The vision MODEL is the point, not just Ollama being up. When reachable but
      // the model isn't pulled, lead with the pull action (this is the required gate).
      const model = step.visionModel || DEFAULT_VISION_MODEL
      const pullBlock = step.reachable && !step.visionModelReady && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
          <p className="mb-1 text-sm font-medium text-content">
            Ollama is running, but the vision model isn't pulled yet — that's what powers captioning.
          </p>
          <p className="mb-2 text-xs text-content-muted">
            <span className="font-mono">{model}</span> — uncensored, needed for concept captions · {VISION_MODEL_VRAM}
          </p>
          <InstallRunner action="ollama_model" buttonLabel={`Pull ${model}`}
            manualCommand={`ollama pull ${model}`}
            onDone={() => refresh(true)} />
        </div>
      )
      const fields = (
        <>
          {guidedField('Ollama URL', 'ollama', 'url', 'http://127.0.0.1:11434')}
          {guidedField('Vision model', 'ollama', 'vision_model', DEFAULT_VISION_MODEL)}
          <p className="text-xs text-content-subtle">
            Use the ABLITERATED Qwen3-VL ({VISION_MODEL_VRAM}) — the vanilla model refuses NSFW.
            For the best captions the app pairs it with JoyCaption (ai-toolkit) — a Joy+Ollama combo.
          </p>
          {saveRecheckBtn}
        </>
      )
      if (step.reachable) {
        return (
          <div className="space-y-4">
            {step.visionModelReady ? (
              <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
                ✓ Ollama is running at <span className="font-mono">{step.url || 'the configured URL'}</span> and
                the vision model <span className="font-mono">{step.visionModel}</span> is ready. Nothing to do here.
              </div>
            ) : pullBlock}
            {fields}
          </div>
        )
      }
      return (
        <GuidedSteps
          intro="Ollama runs local models for captioning and auto-framing. Installing it is not enough — you also need to pull a vision model."
          steps={[{ text: 'Install Ollama and start it (defaults to port 11434).' }]}
          link={{ href: 'https://ollama.com/download', label: 'Download Ollama →' }}>
          {fields}
        </GuidedSteps>
      )
    }
    if (id === 'quality') {
      return (
        <div className="space-y-3">
          <p className="text-sm text-content-muted">
            Two optional helpers installed into this app's own Python environment (insightface,
            onnxruntime, rembg). Both run on CPU — they never touch the GPU or ComfyUI — and the app
            works fully without them; they just make curation and training cleaner:
          </p>
          <ul className="space-y-2 text-sm text-content-muted">
            <li>
              <span className="font-semibold text-content">🎭 Face-similarity scoring</span> — powers the
              "Analyze faces" pass: it scores how closely each generated image resembles your reference
              photo, so you can keep the ones that truly look like the person and drop the off ones. It
              only ranks — it never deletes anything.
            </li>
            <li>
              <span className="font-semibold text-content">🧍 Person masks</span> — isolates the subject
              from the background for <span className="italic">masked training</span>: the décor is weighted
              down so the LoRA binds the identity to the person, not the room or the lighting. Optional —
              a training without masks is still a valid training.
            </li>
          </ul>
          <InstallRunner action="ml_extras" buttonLabel="Install (pip)"
            manualCommand="python -m pip install -r backend/requirements-ml.txt" onDone={() => refresh(true)} />
        </div>
      )
    }
    // training (ai-toolkit)
    const dir = (config.aitoolkit && config.aitoolkit.dir) || ''
    const detectedDir = detected && detected.aitoolkit && detected.aitoolkit.dir
    const fields = (
      <>
        {guidedField('ai-toolkit directory', 'aitoolkit', 'dir', 'C:\\ai-toolkit')}
        {saveRecheckBtn}
      </>
    )
    if (step.valid) {
      return (
        <div className="space-y-4">
          <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
            ✓ ai-toolkit is set up at <span className="font-mono">{dir}</span>. Nothing to do here.
          </div>
          {fields}
        </div>
      )
    }
    // Found on disk but not applied yet → one prominent click (not a subtle link).
    if (detectedDir && dir !== detectedDir) {
      return (
        <div className="space-y-4">
          <div className="rounded-md border border-primary/40 bg-primary/10 px-3 py-3 text-sm text-content">
            <p className="mb-2">Found an ai-toolkit install at <span className="font-mono">{detectedDir}</span>. Use it?</p>
            <button type="button" onClick={() => applyDetectedPath('aitoolkit', 'dir', detectedDir)}
              className="rounded-lg bg-gradient-primary px-4 py-1.5 text-xs font-semibold text-white">
              Use this ai-toolkit →
            </button>
          </div>
          {fields}
        </div>
      )
    }
    // Pointed at a folder that isn't usable yet (venv missing) → finish it, don't re-clone.
    if (dir) {
      return (
        <div className="space-y-4">
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-content">
            Pointed at <span className="font-mono">{dir}</span>, but it isn't usable yet — set up its Python venv per the README.
          </div>
          {fields}
        </div>
      )
    }
    return (
      <GuidedSteps
        intro="ai-toolkit trains the LoRA. Install it once, then point the app at its folder."
        steps={[
          { text: 'Clone ai-toolkit and set up its venv per its README.', command: 'git clone https://github.com/ostris/ai-toolkit' },
        ]}
        link={{ href: 'https://github.com/ostris/ai-toolkit', label: 'ai-toolkit on GitHub →' }}>
        {fields}
      </GuidedSteps>
    )
  }

  const kind = SCREENS[screen]
  const DONE = SCREENS.length - 1
  const isReady = (id) => stepById[id].status === 'ready'
  const toolIdx = (id) => SETUP_STEP_IDS.indexOf(id)
  const screenOf = (id) => SETUP_STEP_IDS.indexOf(id) + 1   // welcome=0, tools=1..N
  const allReady = SETUP_STEP_IDS.every(isReady)
  const nextUnfinished = (fromIdx) => {
    for (let i = fromIdx + 1; i < SETUP_STEP_IDS.length; i += 1)
      if (!isReady(SETUP_STEP_IDS[i])) return SETUP_STEP_IDS[i]
    return null
  }
  const prevUnfinished = (fromIdx) => {
    for (let i = fromIdx - 1; i >= 0; i -= 1)
      if (!isReady(SETUP_STEP_IDS[i])) return SETUP_STEP_IDS[i]
    return null
  }
  // Captioning is the ONE capability the wizard insists on. Z-Image (the default
  // training type) needs Ollama's vision model for prose captions — JoyCaption only
  // covers SDXL booru tags — so the Ollama gate does NOT lift just because JoyCaption
  // is present. The MODEL, not merely Ollama being up, is what matters. Nothing else
  // is hard-gated (build from your own photos + export to train elsewhere stays open).
  // The global "Skip setup" link is still the deliberate bail-out.
  // Pure gate check on a derived step object, so it can be re-evaluated against FRESH
  // capabilities after a save (not just the render-time snapshot).
  const ollamaGateReason = (s) => {
    if (!s || s.status === 'ready') return null
    if (!s.reachable) return "Ollama isn't detected — Z-Image captioning needs it. Install it and start it (port 11434) to continue."
    if (!s.visionModelReady) return 'Pull the vision model below to continue — Z-Image captioning needs it (JoyCaption only covers SDXL).'
    return 'Finish this step to continue.'
  }
  const blockReason = (id) => (id === 'ollama' ? ollamaGateReason(stepById.ollama) : null)
  // The scan already knows what's installed — so "Start setup" / Next land on the
  // first tool that still needs attention and skip the ones already ready. No
  // re-walking ComfyUI/Ollama when they were just detected as running.
  const startSetup = () => {
    const first = SETUP_STEP_IDS.find((id) => !isReady(id))
    setScreen(first ? screenOf(first) : DONE)
  }
  const goNext = () => {
    if (kind === 'welcome') return startSetup()
    if (kind === 'done') return
    const nxt = nextUnfinished(toolIdx(kind))
    setScreen(nxt ? screenOf(nxt) : DONE)
  }
  const goBack = () => {
    if (kind === 'done') {
      const last = [...SETUP_STEP_IDS].reverse().find((id) => !isReady(id))
      return setScreen(last ? screenOf(last) : 0)
    }
    const prv = prevUnfinished(toolIdx(kind))
    setScreen(prv ? screenOf(prv) : 0)
  }
  // Next on a tool step SAVES + re-checks first (so typed URLs/models take effect and
  // the status refreshes), then advances — unless a required gate is still unmet after
  // the re-check, in which case it stays and says why. Re-evaluates against FRESHLY
  // fetched capabilities because the context update from persist() is async.
  const nextWithSave = async () => {
    setAdvancing(true)
    try {
      await persist()
      let fresh = null
      try { fresh = await apiFetch('/api/capabilities') } catch { /* keep going */ }
      if (fresh && kind === 'ollama') {
        const reason = ollamaGateReason(deriveSetupSteps(fresh).find((x) => x.id === 'ollama'))
        if (reason) { toast.warning(reason); return }
      }
      goNext()
    } finally { setAdvancing(false) }
  }

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
    // Three states per tool: ready (✓ green), partial (⚠ amber — detected but a
    // key piece is missing), missing (✗). Ollama keys on the MODEL, not just being
    // reachable — a running Ollama with no vision model is only "partial".
    // `optional: true` rows (local generation) never look like a problem when not
    // ready — you can build a dataset from your own photos + API engines and export
    // to train elsewhere. They render neutral (grey ○ + "optional"), not amber/✗.
    const triState = (reachable, complete) => reachable ? (complete ? 'ready' : 'partial') : 'missing'
    const scanRows = [
      { label: 'Local generation — ComfyUI', optional: true,
        state: triState(stepById.comfyui.reachable, stepById.comfyui.hasKlein),
        partial: 'running — Klein model optional' },
      { label: 'Captioning — Ollama + vision model',
        state: triState(stepById.ollama.reachable, stepById.ollama.visionModelReady),
        partial: 'running — pull the vision model' },
      { label: 'LoRA training — ai-toolkit',
        state: stepById.training.valid ? 'ready'
          : (detected && detected.aitoolkit && detected.aitoolkit.dir ? 'partial' : 'missing'),
        partial: 'found on disk — one click to use' },
    ]
    const SCAN_META = {
      ready: { glyph: '✓', cls: 'text-emerald-400', word: 'ready' },
      partial: { glyph: '⚠', cls: 'text-amber-400', word: '' },
      missing: { glyph: '✗', cls: 'text-content-subtle', word: 'not found' },
    }
    // Optional + not-ready → don't alarm: neutral glyph/color and an "optional" tone.
    const NEUTRAL = { glyph: '○', cls: 'text-content-subtle' }
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
            {scanRows.map((r) => {
              const soft = r.optional && r.state !== 'ready'   // optional + not ready → neutral, not a warning
              const m = soft ? { ...SCAN_META[r.state], ...NEUTRAL } : SCAN_META[r.state]
              const word = r.state === 'partial' ? r.partial
                : r.state === 'missing' ? (r.optional ? 'optional' : m.word)
                : m.word
              return (
                <li key={r.label} className="flex items-center justify-between gap-3 text-sm">
                  <span className="flex items-center gap-2">
                    <span aria-hidden="true" className={detecting ? 'text-content-subtle' : m.cls}>
                      {detecting ? '…' : m.glyph}
                    </span>
                    <span className={r.state === 'ready' ? 'text-content' : 'text-content-muted'}>{r.label}</span>
                    {r.optional && (
                      <span className="rounded bg-surface-raised px-1.5 py-px text-[10px] font-medium text-content-subtle">optional</span>
                    )}
                  </span>
                  <span className={`truncate text-right font-mono text-xs ${detecting ? 'text-content-subtle' : m.cls}`}>
                    {detecting ? '' : word}
                  </span>
                </li>
              )
            })}
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
            {allReady ? "Everything's ready — review →" : 'Start setup →'}
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
  const reason = blockReason(kind)                 // live hint of what's still missing
  const hasNext = nextUnfinished(toolIdx(kind)) !== null
  // Next always saves + re-checks first; the gate (if any) is enforced AFTER that
  // fresh re-check inside nextWithSave, not by disabling the button on a stale snapshot.
  const nextLabel = advancing ? 'Saving…' : (hasNext ? 'Save & continue →' : 'Save & finish →')
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

      {reason && (
        <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          🔒 {reason}
        </p>
      )}
      <div className="flex items-center justify-between">
        <button type="button" onClick={goBack} className="text-xs text-content-subtle underline hover:text-content">
          ← Back
        </button>
        <div className="flex items-center gap-4">
          {skipLink}
          <button type="button" onClick={nextWithSave} disabled={advancing}
            title={reason || ''}
            className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40">
            {nextLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
