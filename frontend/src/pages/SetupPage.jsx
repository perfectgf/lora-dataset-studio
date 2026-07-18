import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiFetch, putJson, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import { deriveSetupSteps, deriveCapabilitySummary, SETUP_STEP_IDS, kleinMissingLabels,
  comfyuiDirVerdict, COMFYUI_SKIP_LOST, COMFYUI_SKIP_KEPT, installAllPlan } from '../hooks/useSetupSteps'
import GuidedSteps from '../components/setup/GuidedSteps'
import InstallRunner from '../components/setup/InstallRunner'
import InstallEverything from '../components/setup/InstallEverything'
import { HelpBadge } from '../help/HelpMode'

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
const DEFAULT_VISION_MODEL = 'huihui_ai/qwen3-vl-abliterated:8b-instruct'
const VISION_MODEL_VRAM = '≈ 8 GB VRAM'
const KLEIN_MODEL_VRAM = '≈ 16 GB VRAM (fp8; ~29 GB at bf16)'

// A wizard "screen" is the welcome/scan, one per setup tool, then the install step (where
// the app installs what it can for you — AFTER the config, since several installs depend on
// a configured ComfyUI/Ollama), then done.
const SCREENS = ['welcome', ...SETUP_STEP_IDS, 'install', 'done']
const TOTAL_TOOLS = SETUP_STEP_IDS.length

const STATUS_META = {
  ready: { glyph: '✓', label: 'Ready', cls: 'text-emerald-400' },
  partial: { glyph: '◐', label: 'Almost there', cls: 'text-amber-400' },
  available: { glyph: '○', label: 'Not set up', cls: 'text-content-subtle' },
  // Neutral, deliberately not red: the user chose to continue without ComfyUI.
  skipped: { glyph: '⊘', label: 'Skipped', cls: 'text-content-subtle' },
}

// Map each capability in the "What's unlocked" review list (deriveCapabilitySummary,
// useSetupSteps.js) back to the wizard step that installs/configures it, so clicking a
// row jumps straight to that step. Most entries match a step's own `unlocks` wording
// 1:1 (Captioning, Face-similarity scoring, Person masks, LoRA training, Test Studio).
// Two don't, and are set by where the control actually lives: "Klein (local)" is
// downloaded from the comfyui step's body (toolBody('comfyui') has the one-click
// installers), not the image step — the image step only has the API-key fields and a
// note pointing at ComfyUI. "Auto-framing & head-crop" is the ollama step's other two
// unlocks (Auto-classify framing / Auto head-crop), just phrased differently here.
const CAPABILITY_STEP_ID = {
  'Nano Banana (Gemini)': 'image',
  'ChatGPT (gpt-image-2)': 'image',
  'Klein (local)': 'comfyui',
  'Captioning': 'ollama',
  'Auto-framing & head-crop': 'ollama',
  'Face-similarity scoring': 'quality',
  'Person masks': 'quality',
  'Watermark inpainting': 'quality',
  'LoRA training': 'training',
  'Test Studio': 'comfyui',
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
  const [startingOllama, setStartingOllama] = useState(false) // "Start Ollama" in flight
  const [dirCheck, setDirCheck] = useState(null)    // live classify of the typed ComfyUI dir
  const [skipConfirm, setSkipConfirm] = useState(false) // "continue without ComfyUI" panel open
  const autodetectedRef = useRef(false)             // run the on-load autodetect only once
  // Last SERVER-acknowledged config (JSON) — dirty = user edits not yet saved.
  const savedConfigRef = useRef(null)

  const load = useCallback(async () => {
    try {
      const data = await apiFetch('/api/settings')
      setConfig(data.config); setSecretsPresence(data.secrets); setLoadError(false)
      savedConfigRef.current = JSON.stringify(data.config)
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
        savedConfigRef.current = JSON.stringify(saved.config)
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

  // Navigating between wizard screens dismisses a half-open "continue without ComfyUI"
  // panel, so it never re-appears stale when the user comes back to this step.
  useEffect(() => { setSkipConfirm(false) }, [screen])

  // Live, SAVE-FREE classification of the typed ComfyUI directory, so the field gives
  // an actionable verdict (wrong path / empty folder / launcher-parent-with-a-child)
  // while the user is still typing — not only after a "Save & re-check". Debounced;
  // the result carries the exact path it judged so a stale verdict is never shown
  // against a newer string. Blank field → no check (the skip panel owns that case).
  const baseDir = (config && config.comfyui && config.comfyui.base_dir) || ''
  useEffect(() => {
    const path = baseDir.trim()
    if (!path) { setDirCheck(null); return undefined }
    let alive = true
    setDirCheck({ status: 'checking', path })
    const t = setTimeout(async () => {
      try {
        const r = await apiFetch(`/api/setup/comfyui-dir?path=${encodeURIComponent(path)}`)
        if (alive) setDirCheck({ ...r, path })
      } catch { if (alive) setDirCheck(null) }
    }, 350)
    return () => { alive = false; clearTimeout(t) }
  }, [baseDir])

  // Apply a disk-scanned path suggestion (user-confirmed) into config + save.
  const applyDetectedPath = async (section, key, val) => {
    const next = { ...config, [section]: { ...config[section], [key]: val } }
    try {
      const saved = await putJson('/api/settings', { config: next })
      setConfig(saved.config); savedConfigRef.current = JSON.stringify(saved.config)
      await refresh(true); toast.success('Applied.')
    } catch (e) { toast.error(`Save failed: ${e.message}`) }
  }

  const steps = useMemo(() => deriveSetupSteps(caps), [caps])
  const summary = useMemo(() => deriveCapabilitySummary(caps), [caps])
  // Everything "Install everything" can queue right now (mirrors the backend plan).
  const installPlan = useMemo(() => installAllPlan(caps), [caps])
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
      savedConfigRef.current = JSON.stringify(data.config)
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

  // One-click start for an ALREADY-INSTALLED Ollama that just isn't running
  // (caps.ollama.installed true, reachable false). The backend starts `ollama
  // serve` detached and polls readiness (~15s); refresh(true) then flips the step
  // to ready with no app restart. A failure returns 502 -> apiFetch throws (and
  // auto-toasts the generic 5xx notice); the catch adds the specific reason,
  // matching the existing saveSecretThenTest pattern.
  const startOllama = async () => {
    setStartingOllama(true)
    try {
      const r = await postJson('/api/ollama/start', {})
      if (r.reachable) { toast.success('Ollama started.'); await refresh(true) }
      else { toast.error(r.error || 'Ollama did not become ready.') }
    } catch (e) { toast.error(e.message || 'Could not start Ollama.') }
    finally { setStartingOllama(false) }
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
      // Klein needs three weights; the backend tells us which are still absent
      // (step.kleinMissing). Each download button greys to "✓ Installed" on its own,
      // and the header/intro name the exact gap instead of a blanket "model missing".
      const kleinMissing = step.kleinMissing || []
      const missingLabels = kleinMissingLabels(kleinMissing)
      const missingSummary = missingLabels.length ? missingLabels.join(' + ') : ''
      const installBtn = (action, label) => kleinMissing.includes(action)
        ? <InstallRunner action={action} buttonLabel={label} onDone={() => refresh(true)} />
        : <p className="text-xs text-emerald-400">✓ Installed</p>
      // Live verdict on the CURRENTLY-TYPED directory (from /api/setup/comfyui-dir),
      // shown the moment the field changes — a wrong path, an empty folder, or the
      // launcher/parent folder (with a one-click "use the child" adopt). The verdict
      // is only trusted when it matches the exact string in the field.
      const typedDir = (config.comfyui.base_dir || '').trim()
      const liveCheck = dirCheck && dirCheck.path === typedDir ? dirCheck : null
      const dirVerdictNode = typedDir ? (
        (!liveCheck || liveCheck.status === 'checking')
          ? <p className="text-xs text-content-subtle">Checking this folder…</p>
          : (() => {
            const v = comfyuiDirVerdict(liveCheck)
            if (!v.message) return null
            const cls = v.tone === 'ok' ? 'text-emerald-400'
              : v.tone === 'warn' ? 'text-amber-400' : 'text-content-subtle'
            const glyph = v.tone === 'ok' ? '✓' : v.tone === 'warn' ? '⚠' : ''
            return (
              <div className="space-y-1.5">
                <p className={`text-xs ${cls}`}>{glyph} {v.message}</p>
                {v.suggestion && (
                  <button type="button" onClick={() => setField('comfyui', 'base_dir', v.suggestion)}
                    className="rounded-md border border-border-strong px-2.5 py-1 text-xs font-medium text-primary hover:bg-surface-raised">
                    Use this folder instead
                  </button>
                )}
              </div>
            )
          })()
      ) : null
      const fields = (
        <>
          {guidedField('ComfyUI API URL', 'comfyui', 'api_url', 'http://127.0.0.1:8188')}
          {guidedField('ComfyUI install directory', 'comfyui', 'base_dir', 'C:\\ComfyUI')}
          {detectedPathChip('comfyui', 'base_dir')}
          {/* Live, save-free verdict on the typed path (main.py + models/ ⇒ valid;
              a launcher/parent folder proposes its inner ComfyUI to adopt; a wrong,
              empty or missing folder each get their own actionable message). This
              replaces the old "Save & re-check to validate" placeholder — the check
              runs as you type, and the folder it judged is pinned to the field value. */}
          {dirVerdictNode}
          {step.reachable && !step.hasKlein && (
            <div className="space-y-1 text-xs text-content-muted">
              {missingSummary && (
                <p className="text-amber-300">
                  Klein still needs the <span className="text-content font-medium">{missingSummary}</span> — grab
                  {missingLabels.length > 1 ? ' them' : ' it'} below. Local generation is
                  <span className="text-content font-medium"> optional</span>: the API engines and your own photos work without it.
                </p>
              )}
              <p>
                Running. The Klein model is <span className="text-content font-medium">optional</span> — add it only if you want
                local generation (you can also use the API engines or your own photos, then export to train elsewhere).
                To enable it, download <span className="font-mono">flux-2-klein-9b-kv-fp8.safetensors</span> ({KLEIN_MODEL_VRAM}) into
                <span className="font-mono"> &lt;ComfyUI&gt;/models/unet/klein/</span> — the <span className="text-content font-medium">KV build</span>,
                up to 2.5× faster on multi-reference edits at the same quality, and a public download (no token).
              </p>
              <p>
                Also recommended: the <span className="text-content font-medium">consistency LoRA</span>{' '}
                <span className="font-mono">Flux2-Klein-9B-consistency-V2.safetensors</span> (331 MB) — a community LoRA by
                <span className="text-content font-medium"> dx8152</span> (apache-2.0) — into{' '}
                <span className="font-mono">&lt;ComfyUI&gt;/models/loras/klein/</span> — it anchors the composition between
                edits (the "Consistency LoRA" slider drives its strength; ~0.5 is balanced, high values suppress
                pose changes). Face identity itself comes from the reference photo(s).
              </p>
              {step.dirValid ? (
                <div className="space-y-2 rounded-md border border-border bg-white/5 p-2.5">
                  <p className="text-content text-xs font-medium">
                    ⬇ One-click downloads — straight into the validated ComfyUI folders:
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <p className="mb-1 text-[0.6875rem] text-content-muted">
                        Klein 9B KV (fp8) → <span className="font-mono">models/unet/klein/</span>
                        <span className="block text-content-subtle">
                          Direct public download — no token needed. The FLUX Non-Commercial License governs use.
                        </span>
                      </p>
                      {installBtn('klein_model', '⬇ Download Klein model')}
                    </div>
                    <div>
                      <p className="mb-1 text-[0.6875rem] text-content-muted">
                        Consistency LoRA (331 MB) → <span className="font-mono">models/loras/klein/</span>
                      </p>
                      {installBtn('klein_lora', '⬇ Download consistency LoRA')}
                    </div>
                    <div>
                      <p className="mb-1 text-[0.6875rem] text-content-muted">
                        Text encoder (~8.7 GB) → <span className="font-mono">models/text_encoders/</span>
                      </p>
                      {installBtn('klein_text_encoder', '⬇ Download text encoder')}
                    </div>
                    <div>
                      <p className="mb-1 text-[0.6875rem] text-content-muted">
                        VAE (336 MB) → <span className="font-mono">models/vae/</span>
                      </p>
                      {installBtn('klein_vae', '⬇ Download VAE')}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-content-subtle">
                  Validate the ComfyUI install directory above (Save &amp; re-check) to unlock one-click downloads.
                </p>
              )}
              <p className="flex flex-wrap gap-x-4 gap-y-1">
                <a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8" target="_blank" rel="noreferrer"
                  className="text-primary underline">Official Klein 9B KV model page →</a>
                <a href="https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency" target="_blank" rel="noreferrer"
                  className="text-primary underline">Community consistency LoRA (dx8152) →</a>
                <a href="https://docs.comfy.org/tutorials/flux/flux-2-klein" target="_blank" rel="noreferrer"
                  className="text-primary underline">ComfyUI setup guide →</a>
              </p>
            </div>
          )}
          {saveRecheckBtn}
          {/* Discoverable, explicit skip: only when there's genuinely nothing here
              (no dir, not reachable, not already skipped) — a running/configured
              ComfyUI never offers to be skipped. */}
          {!typedDir && !step.reachable && !step.skipped && (
            <button type="button" onClick={() => setSkipConfirm(true)}
              className="text-xs text-content-subtle underline hover:text-content">
              Don't want local generation? Continue without ComfyUI →
            </button>
          )}
        </>
      )
      // The "continue without ComfyUI" confirmation: what turns off vs stays on (from
      // the real capability gates), shown BEFORE the skip is committed.
      const skipPanel = (
        <div className="space-y-4">
          <div className="space-y-3 rounded-md border border-border-strong bg-surface-raised px-4 py-3 text-sm">
            <p className="font-medium text-content">Continue without ComfyUI?</p>
            <p className="text-xs text-content-muted">
              ComfyUI powers local generation and the Test Studio. You can come back anytime —
              entering a directory later turns everything below back on automatically.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-semibold text-amber-300">What you won't have</p>
                <ul className="space-y-1 text-xs text-content-muted">
                  {COMFYUI_SKIP_LOST.map((t) => (
                    <li key={t} className="flex gap-1.5"><span aria-hidden="true" className="text-amber-400">✗</span><span>{t}</span></li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="mb-1 text-xs font-semibold text-emerald-400">What still works</p>
                <ul className="space-y-1 text-xs text-content-muted">
                  {COMFYUI_SKIP_KEPT.map((t) => (
                    <li key={t} className="flex gap-1.5"><span aria-hidden="true" className="text-emerald-400">✓</span><span>{t}</span></li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="flex items-center gap-4 pt-1">
              <button type="button" onClick={skipComfyui} disabled={busy}
                className="rounded-lg bg-gradient-primary px-4 py-1.5 text-xs font-semibold text-white disabled:opacity-50">
                {busy ? 'Saving…' : 'Continue without ComfyUI'}
              </button>
              <button type="button" onClick={() => setSkipConfirm(false)}
                className="text-xs text-content-subtle underline hover:text-content">
                Never mind — I'll set it up
              </button>
            </div>
          </div>
          {fields}
        </div>
      )
      if (skipConfirm) return skipPanel
      // Already skipped by choice: neutral confirmation (not a warning) + the fields,
      // so typing a directory silently un-skips and re-enables local generation.
      if (step.skipped) {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-content-muted">
              ⊘ You chose to continue without ComfyUI. Local generation, the Test Studio and
              custom-base training stay off — enter a directory below anytime to turn them back on.
            </div>
            {fields}
          </div>
        )
      }
      // Already detected/running → skip the from-scratch install guide; show the
      // reachable confirmation and only the remaining gap.
      if (step.reachable) {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
              ✓ ComfyUI is already running at <span className="font-mono">{step.apiUrl || 'the configured URL'}</span>.
              {step.hasKlein
                ? ' Nothing to do here.'
                : ` It works — Klein still needs the ${missingSummary} (optional, for local generation).`}
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
      // Installed but not running → a one-click Start beats sending the user back
      // to the install guide (the binary is detected independently of the server).
      if (step.installed) {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
              <p className="mb-1 text-sm font-medium text-content">
                Ollama is installed{step.binaryPath && (
                  <> at <span className="font-mono">{step.binaryPath}</span></>
                )} but not running.
              </p>
              <p className="mb-2 text-xs text-content-muted">
                Start it (it listens on port 11434) to unlock captioning and auto-framing — no restart needed.
              </p>
              <button type="button" onClick={startOllama} disabled={startingOllama}
                className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50">
                {startingOllama ? 'Starting…' : '▶ Start Ollama'}
              </button>
            </div>
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
      // Each ML helper installs — or REINSTALLS/repairs — on its own now, so a user
      // who's missing just one (e.g. watermark inpainting on an older install) fixes
      // that one without redoing the whole monolithic step. The all-at-once install
      // stays available below for a first-time setup.
      const ML_CAPS = [
        { action: 'face_scoring', cap: 'face_scoring', icon: '🎭', title: 'Face-similarity scoring',
          body: 'Powers the "Analyze faces" pass: scores how closely each generated image resembles your reference photo, so you keep the ones that truly look like the person. It only ranks — it never deletes anything.' },
        { action: 'masks', cap: 'masks', icon: '🧍', title: 'Person masks',
          body: 'Isolates the subject from the background for masked training: the décor is weighted down so the LoRA binds the identity to the person, not the room. A training without masks is still valid.' },
        { action: 'watermark_inpaint', cap: 'watermark_inpaint', icon: '🧽', title: 'Watermark inpainting',
          body: 'Repaints small off-center watermarks (LaMa) during 🧽 Clean instead of only cropping border marks. It can use CUDA or CPU from Settings. Without it, off-center marks are skipped.' },
      ]
      return (
        <div className="space-y-3">
          <p className="text-sm text-content-muted">
            Optional helpers installed into this app's own Python environment. Face scoring and masks run on
            CPU; watermark inpainting can use CUDA or CPU. The app works fully without them; they just make
            curation and training cleaner. Install each on its own below, or all at once at the bottom. Already installed?
            Use <span className="font-medium text-content">↻ Reinstall</span> to repair or update it.
          </p>
          {caps.python && !caps.python.ml_supported && (
            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2.5 text-sm text-content space-y-1">
              <p>
                <span className="font-semibold text-amber-300">⚠ Python {caps.python.version} —</span>{' '}
                these extras need Python {caps.python.ml_range}. insightface / numpy&lt;2 / onnxruntime publish
                no prebuilt packages for {caps.python.version}, so the installs below will try to compile them
                and most likely fail.
              </p>
              <p className="text-content-muted">
                They're optional — you can skip this step, or install them into a separate Python 3.11/3.12
                environment and point <span className="font-mono">face_scoring.python</span> +{' '}
                <span className="font-mono">masks.python</span> at it in Settings.
              </p>
            </div>
          )}
          <div className="space-y-3">
            {ML_CAPS.map((c) => {
              const present = !!caps[c.cap]
              return (
                <div key={c.action} className="rounded-md border border-border bg-surface-raised p-3 space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-content">{c.icon} {c.title}</span>
                    <span className={`shrink-0 text-xs font-medium ${present ? 'text-emerald-400' : 'text-content-subtle'}`}>
                      {present ? '✓ Installed' : '✗ Not installed'}
                    </span>
                  </div>
                  <p className="text-xs text-content-muted">{c.body}</p>
                  {/* Reuse the Setup InstallRunner verbatim — polling, live pip log, and the
                      scoped manual-command fallback come from the backend per action. onDone
                      re-probes caps so ✗ flips to ✓ (or the reinstall confirms) without a restart. */}
                  <InstallRunner action={c.action}
                    buttonLabel={present ? '↻ Reinstall' : 'Install'}
                    onDone={() => refresh(true)} />
                </div>
              )
            })}
          </div>
          <details className="rounded-md border border-border bg-surface-raised px-3 py-2">
            <summary className="cursor-pointer text-xs text-content-subtle hover:text-content">
              Or install everything at once (first-time setup)
            </summary>
            <div className="mt-2">
              <InstallRunner action="ml_extras" buttonLabel="Install all (pip)"
                manualCommand="python -m pip install -r backend/requirements-ml.txt" onDone={() => refresh(true)} />
            </div>
          </details>
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
        <p className="mt-2 text-content-muted text-xs">
          No GPU? You can skip this step: add a <strong>vast.ai API key</strong> in
          Settings instead and train in the cloud (the app rents a GPU per run,
          ~$1-2, and shuts it down automatically).
        </p>
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
  const INSTALL = SCREENS.indexOf('install')   // the install/reinstall step, after config
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
    if (!s.reachable) {
      // Installed-but-stopped gets a Start nudge; genuinely absent gets install.
      if (!s.installed) return "Ollama isn't installed — download it and start it (port 11434) to continue."
      return 'Ollama is installed but not running — click ▶ Start Ollama below to continue.'
    }
    if (!s.visionModelReady) return 'Pull the vision model below to continue — Z-Image captioning needs it (JoyCaption only covers SDXL).'
    return 'Finish this step to continue.'
  }
  const blockReason = (id) => (id === 'ollama' ? ollamaGateReason(stepById.ollama) : null)
  // The scan already knows what's installed — so "Start setup" / Next land on the
  // first tool that still needs attention and skip the ones already ready. No
  // re-walking ComfyUI/Ollama when they were just detected as running.
  const startSetup = () => {
    const first = SETUP_STEP_IDS.find((id) => !isReady(id))
    // All tools already configured -> land on the install step (its reinstall menu), not
    // straight to the summary, so "install everything" / repairs stay one screen away.
    setScreen(first ? screenOf(first) : INSTALL)
  }
  const goNext = () => {
    if (kind === 'welcome') return startSetup()
    if (kind === 'install') return setScreen(DONE)
    if (kind === 'done') return
    const nxt = nextUnfinished(toolIdx(kind))
    // After the last config tool, the install step (never straight to done).
    setScreen(nxt ? screenOf(nxt) : INSTALL)
  }
  // Guard-rail: Back (unlike Save & continue) does NOT save — warn before losing
  // typed-but-unsaved fields (config edits or a typed secret).
  const hasUnsaved = () => (
    (savedConfigRef.current != null && JSON.stringify(config) !== savedConfigRef.current)
    || Object.values(secretInputs).some((v) => (v || '').trim())
  )
  const goBack = () => {
    if (hasUnsaved() && !window.confirm(
      'You have unsaved changes on this step - they will be lost.\n\nGo back without saving?')) return
    if (kind === 'done') return setScreen(INSTALL)   // the install step sits before the summary
    if (kind === 'install') {
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
    // ComfyUI with an empty field and nothing reachable = a conscious skip. Show what
    // it costs (skipPanel) and require the explicit "Continue without ComfyUI" button
    // BEFORE advancing — the bottom Next only opens/keeps the panel, it never skips
    // silently (that path is what records the persistent choice). Already-skipped or
    // reachable falls through normally.
    if (kind === 'comfyui') {
      const cfgDir = ((config.comfyui && config.comfyui.base_dir) || '').trim()
      const s = stepById.comfyui
      if (!cfgDir && !s.reachable && !s.skipped) { setSkipConfirm(true); return }
    }
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

  // Persist the conscious "continue without ComfyUI" choice, then advance. Entering a
  // directory later annuls it on its own (the backend derives comfyui.skipped =
  // setup_skipped AND no base_dir), so this only sticks while the field stays empty.
  const skipComfyui = async () => {
    setBusy(true)
    try {
      const data = await putJson('/api/settings', { config: { comfyui: { setup_skipped: true } } })
      setConfig(data.config); savedConfigRef.current = JSON.stringify(data.config)
      await refresh(true)
      setSkipConfirm(false)
      goNext()
    } catch (e) { toast.error(`Save failed: ${e.message}`) }
    finally { setBusy(false) }
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
    // Defense in depth: also mark the onboarding redirect as "already fired" here,
    // in the same sessionStorage key App.jsx's OnboardingRedirect guards on — so
    // skipping never bounces straight back to #/setup even in an edge case where
    // the guard effect hasn't run yet (e.g. this Link navigates before that effect
    // re-fires with fresh caps).
    <Link to="/datasets" onClick={() => sessionStorage.setItem('lds_setup_redirected', '1')}
      className="text-xs text-content-subtle underline hover:text-content">
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
    // Ollama now has THREE scan outcomes: running (ready, or amber "pull the model"),
    // installed-but-STOPPED (amber "installed — not running" → the ollama step's ▶ Start
    // button fixes it), and genuinely absent (✗). The old triState collapsed the stopped
    // case into "✗ not found", which read as "you don't have Ollama".
    const oll = stepById.ollama
    const ollamaScan = oll.reachable
      ? { state: oll.visionModelReady ? 'ready' : 'partial', partial: 'running — pull the vision model' }
      : oll.installed
        ? { state: 'partial', partial: 'installed — not running' }
        : { state: 'missing', partial: '' }
    // stepId: which wizard step (SETUP_STEP_IDS) installs/configures this capability —
    // each row is a direct link to that step's screen, whether or not it's ready yet.
    const scanRows = [
      { label: 'Local generation — ComfyUI', optional: true, stepId: 'comfyui',
        // A conscious skip reads as "skipped" (neutral), not "not found" — the probe
        // doesn't keep nagging about a choice the user already made. (partial text is
        // only used for the reachable-but-incomplete case.)
        state: stepById.comfyui.skipped ? 'skipped'
          : triState(stepById.comfyui.reachable, stepById.comfyui.hasKlein),
        partial: 'running — Klein model optional' },
      { label: 'Captioning — Ollama + vision model', stepId: 'ollama',
        state: ollamaScan.state, partial: ollamaScan.partial },
      { label: 'LoRA training — ai-toolkit', stepId: 'training',
        state: stepById.training.valid ? 'ready'
          : (detected && detected.aitoolkit && detected.aitoolkit.dir ? 'partial' : 'missing'),
        partial: 'found on disk — one click to use' },
    ]
    const SCAN_META = {
      ready: { glyph: '✓', cls: 'text-emerald-400', word: 'ready' },
      partial: { glyph: '⚠', cls: 'text-amber-400', word: '' },
      missing: { glyph: '✗', cls: 'text-content-subtle', word: 'not found' },
      skipped: { glyph: '⊘', cls: 'text-content-subtle', word: 'skipped' },
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
          <ul className="mt-4 space-y-1">
            {scanRows.map((r) => {
              const soft = r.optional && r.state !== 'ready'   // optional + not ready → neutral, not a warning
              const m = soft ? { ...SCAN_META[r.state], ...NEUTRAL } : SCAN_META[r.state]
              const word = r.state === 'partial' ? r.partial
                : r.state === 'missing' ? (r.optional ? 'optional' : m.word)
                : m.word
              return (
                <li key={r.label}>
                  {/* Whole row navigates to the wizard step that installs this capability —
                      ready ones stay clickable too (revisit/change it), the chevron just
                      stays subtle for those. Disabled mid-scan: the state is still shifting. */}
                  <button type="button" disabled={detecting}
                    onClick={() => setScreen(screenOf(r.stepId))}
                    className="flex w-full items-center justify-between gap-3 rounded-md px-2 py-1.5 -mx-2 text-left text-sm
                      cursor-pointer transition-colors hover:bg-surface-raised focus:outline-none focus-visible:ring-2
                      focus-visible:ring-primary disabled:cursor-default disabled:hover:bg-transparent">
                    <span className="flex items-center gap-2">
                      <span aria-hidden="true" className={detecting ? 'text-content-subtle' : m.cls}>
                        {detecting ? '…' : m.glyph}
                      </span>
                      <span className={r.state === 'ready' ? 'text-content' : 'text-content-muted'}>{r.label}</span>
                      {r.optional && (
                        <span className="rounded bg-surface-raised px-1.5 py-px text-[10px] font-medium text-content-subtle">optional</span>
                      )}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className={`truncate text-right font-mono text-xs ${detecting ? 'text-content-subtle' : m.cls}`}>
                        {detecting ? '' : word}
                      </span>
                      {!detecting && (
                        <span aria-hidden="true"
                          className={`text-xs ${r.state === 'ready' ? 'text-content-subtle/60' : 'text-content-subtle'}`}>
                          ›
                        </span>
                      )}
                    </span>
                  </button>
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
          {/* Welcome leads to configuring the services first; the install step (Install
              everything + the one-by-one menu) comes AFTER, since several installs depend
              on a configured ComfyUI/Ollama. */}
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
          <ul className="mt-3 grid gap-1 sm:grid-cols-2">
            {summary.map((s) => {
              const targetStep = CAPABILITY_STEP_ID[s.label]
              // Every current capability maps to a wizard step (see CAPABILITY_STEP_ID above);
              // this guard is defensive only — an unmapped label just renders inert, as before.
              if (!targetStep) {
                return (
                  <li key={s.label} className={`flex items-center gap-2 px-2 py-1 text-sm ${s.ok ? 'text-content' : 'text-content-subtle'}`}>
                    <span aria-hidden="true" className={s.ok ? 'text-emerald-400' : 'text-content-subtle'}>{s.ok ? '✓' : '✗'}</span>
                    {s.label}
                  </li>
                )
              }
              return (
                <li key={s.label}>
                  <button type="button" onClick={() => setScreen(screenOf(targetStep))}
                    className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1 text-left text-sm
                      cursor-pointer transition-colors hover:bg-surface-raised focus:outline-none focus-visible:ring-2
                      focus-visible:ring-primary ${s.ok ? 'text-content' : 'text-content-subtle'}`}>
                    <span className="flex items-center gap-2">
                      <span aria-hidden="true" className={s.ok ? 'text-emerald-400' : 'text-content-subtle'}>{s.ok ? '✓' : '✗'}</span>
                      {s.label}
                    </span>
                    <span aria-hidden="true" className={`text-xs ${s.ok ? 'text-content-subtle/60' : 'text-content-subtle'}`}>›</span>
                  </button>
                </li>
              )
            })}
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

  // --- Install / reinstall components (after the API/service config) -----------
  if (kind === 'install') {
    return (
      <div className="mx-auto max-w-2xl space-y-5">
        <div className="text-center">
          <div className="text-3xl" aria-hidden="true">⬇</div>
          <h1 className="mt-2 text-2xl font-bold text-content">Install components</h1>
          <p className="mt-2 text-sm text-content-muted">
            Now that your services are configured, install what the app can set up for you —
            all at once, or one at a time. Come back here anytime to reinstall a component and
            repair a broken install.
          </p>
        </div>
        <InstallEverything plan={installPlan} caps={caps} onDone={() => refresh(true)} />
        <div className="flex items-center justify-between">
          <button type="button" onClick={goBack} className="text-xs text-content-subtle underline hover:text-content">
            ← Back
          </button>
          <div className="flex items-center gap-4">
            {skipLink}
            <button type="button" onClick={goNext}
              className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white">
              Finish →
            </button>
          </div>
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
              <HelpBadge topic="page-setup" className="ml-2" />
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
