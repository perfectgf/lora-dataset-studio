import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { apiFetch, getCsrfToken, putJson, postJson } from '../api/fetchClient'
import { useToast } from '../components/common/Toast'
import { useCapabilities } from '../context/CapabilitiesContext'
import { deriveSetupSteps, deriveCapabilitySummary, SETUP_STEP_IDS, kleinMissingLabels,
  comfyuiDirVerdict, comfyuiLauncherState, COMFYUI_SKIP_LOST, COMFYUI_SKIP_KEPT, installAllPlan,
  aitoolkitVerdict, AITOOLKIT_INSTALL_STEPS, chatgptLanes, chatgptLaneSummary }
  from '../hooks/useSetupSteps'
import ChatgptSubscriptionConnect from '../components/common/ChatgptSubscriptionConnect'
import SettingsLink from '../components/common/SettingsLink'
import GuidedSteps from '../components/setup/GuidedSteps'
import { ML_INSTALL_CARDS, cardInstalled } from '../components/setup/mlInstallCards'
import InstallRunner from '../components/setup/InstallRunner'
import InstallEverything from '../components/setup/InstallEverything'
import { HelpBadge } from '../help/HelpMode'
import { kleinAssetBlocks } from '../utils/kleinAssets.js'

const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content ' +
  'placeholder:text-content-subtle focus:border-primary focus:outline-none'

const KEY_FIELDS = [
  { key: 'GEMINI_API_KEY', label: 'Gemini API key', engine: 'nanobanana',
    href: 'https://aistudio.google.com/apikey', help: 'Powers Nano Banana.' },
  // NOT "Powers ChatGPT (gpt-image-2)" any more. That sentence was true and
  // still misleading: it made the paid API key read as the only way in, on the
  // screen where a ChatGPT Plus/Pro subscriber decides what this app can do —
  // while the subscription lane sat two pages away in Settings. The key row now
  // describes ITS lane; the engine's other door is rendered right under it.
  { key: 'OPENAI_API_KEY', label: 'OpenAI API key', engine: 'chatgpt',
    href: 'https://platform.openai.com/api-keys',
    help: 'Pay-per-image, and the lane that accepts up to 16 reference images.' },
  { key: 'OPENROUTER_API_KEY', label: 'OpenRouter API key', engine: 'openrouter',
    href: 'https://openrouter.ai/keys',
    help: 'Powers the OpenRouter engine — one key and one balance for the same '
      + 'upstream models. Pick the model in Settings \u203a Image engines.' },
]

/** Which capability probe the "Save & test" button runs for each key field. */
const KEY_TEST_TARGET = { nanobanana: 'gemini', chatgpt: 'openai', openrouter: 'openrouter' }

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
  initializing: { glyph: '↻', label: 'Initializing…', cls: 'text-amber-400' },
  partial: { glyph: '◐', label: 'Almost there', cls: 'text-amber-400' },
  available: { glyph: '○', label: 'Not set up', cls: 'text-content-subtle' },
  // Neutral, deliberately not red: the user chose to continue without ComfyUI.
  skipped: { glyph: '⊘', label: 'Skipped', cls: 'text-content-subtle' },
}

// Map each capability in the "What's unlocked" review list (deriveCapabilitySummary,
// useSetupSteps.js) back to the wizard step that installs/configures it, so clicking a
// row jumps straight to that step. Most entries match a step's own `unlocks` wording
// 1:1 (Captioning, Face-similarity scoring, Person masks, LoRA training, Test Studio).
// Three don't, and are set by where the control actually lives: "Klein (local)" is
// downloaded from the comfyui step's body (toolBody('comfyui') has the one-click
// installers), not the image step — the image step only has the API-key fields and a
// note pointing at ComfyUI. "Auto-framing & head-crop" is the ollama step's other two
// unlocks (Auto-classify framing / Auto head-crop), just phrased differently here.
// "Krea 2 Edit (local)" is the third, and it had NO entry at all: its row rendered a
// ✗ and led nowhere, which teaches the user something is missing and gives them no
// way to reach it. Its one-click installer is KreaInstallCard, and that card is
// mounted ONLY inside InstallEverything — i.e. on the 'install' screen, not on any
// tool step (the comfyui step has Klein's weights, never Krea's). Hence 'install',
// which is a SCREEN and not a step id — see screenOf().
const CAPABILITY_STEP_ID = {
  'Nano Banana (Gemini)': 'image',
  'ChatGPT (gpt-image-2)': 'image',
  'OpenRouter': 'image',
  'Klein (local)': 'comfyui',
  'Krea 2 Edit (local)': 'install',
  'Captioning': 'ollama',
  'Auto-framing & head-crop': 'ollama',
  'Face-similarity scoring': 'quality',
  'Person masks': 'quality',
  'Watermark inpainting': 'quality',
  'Video bank — reading files': 'quality',
  'Video bank — shot detection': 'quality',
  'Video bank — clip encoding': 'quality',
  // Same step as the four rows above: each has its own one-click card in the
  // quality step's install list (mlInstallCards.js) — bank scoring/SigLIP2/the
  // watermark detector were already there; the scraping-extras card was added
  // alongside this row (it previously only lived on the Concept Sources panel).
  'Bank scoring (aesthetic · NSFW · style)': 'quality',
  'SigLIP2 Bank semantics (optional)': 'quality',
  'Watermark detector (optional)': 'quality',
  'Scraping extras (optional)': 'quality',
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
  // Deep link: /setup?step=<tool id> opens that wizard screen straight away.
  // The Settings ▸ Overview capability rows link here — "✗ Person masks" must
  // land ON the install button, not on the welcome screen with 4 Next clicks in
  // between. Applied to the INITIAL state (not an effect) so the wizard never
  // flashes the welcome screen first; an unknown/absent step falls back to 0.
  const [searchParams] = useSearchParams()
  const [screen, setScreen] = useState(() => {
    const raw = searchParams.get('step')
    const i = SETUP_STEP_IDS.indexOf(raw)
    if (i >= 0) return i + 1                        // welcome=0, tools=1..N
    // The screens that are NOT tool steps are addressable too: the install /
    // repair menu is where the one-click engine installs live (Krea 2 Edit), and
    // a help topic pointing at ?step=install must land there, not on welcome.
    const j = SCREENS.indexOf(raw)
    return j > 0 ? j : 0
  })
  const [advancing, setAdvancing] = useState(false) // Next is mid save-&-recheck
  const [startingOllama, setStartingOllama] = useState(false) // "Start Ollama" in flight
  const [startingComfyui, setStartingComfyui] = useState(false) // secure portable launch in flight
  const [savingOllamaMode, setSavingOllamaMode] = useState(false)
  const [runtimeReadiness, setRuntimeReadiness] = useState(null)
  const [readinessRevision, setReadinessRevision] = useState(0)
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

  // Docker-owned services can need longer than the app container to boot. Poll a
  // dedicated one-second endpoint instead of re-running the expensive capability
  // scan. Recursive timeouts avoid overlapping requests; unmount aborts the active
  // fetch and clears the pending timer.
  useEffect(() => {
    let alive = true
    let timer = null
    let controller = null
    let previous = null
    let capabilityRefreshPending = false

    function schedule(delay) {
      if (alive) timer = setTimeout(check, delay)
    }
    async function check() {
      controller = new AbortController()
      try {
        const next = await apiFetch('/api/setup/runtime-readiness', {
          background: true,
          cache: 'no-store',
          signal: controller.signal,
        })
        if (!alive) return
        const comfyBecameReady = next.comfyui?.mode === 'integrated'
          && next.comfyui.ready && !previous?.comfyui?.ready
        const ollamaBecameReady = ['docker', 'host'].includes(next.ollama?.mode)
          && next.ollama.ready && !previous?.ollama?.ready
        if (comfyBecameReady || ollamaBecameReady) capabilityRefreshPending = true
        if (capabilityRefreshPending) {
          const refreshed = await refresh(true, { background: true })
          capabilityRefreshPending = !refreshed
        }
        if (!alive) return
        previous = next
        setRuntimeReadiness(next)
        // A ready lightweight probe is not enough: the cards consume the full
        // capability snapshot. Keep polling until that refresh is acknowledged,
        // otherwise one silent 500 leaves stale caps and stops forever.
        if (next.comfyui?.poll || next.ollama?.poll || capabilityRefreshPending) schedule(3000)
      } catch {
        if (alive) schedule(5000)
      }
    }

    check()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
      controller?.abort()
    }
  }, [refresh, readinessRevision])

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

  const steps = useMemo(
    () => deriveSetupSteps(caps, runtimeReadiness),
    [caps, runtimeReadiness],
  )
  const summary = useMemo(() => deriveCapabilitySummary(caps), [caps])
  // Everything "Install everything" can queue right now (mirrors the backend plan).
  const installPlan = useMemo(() => installAllPlan(caps), [caps])
  // pending = installed/configured, only waiting for ComfyUI to be launched —
  // counted as ready so "8 of 10" doesn't scold a perfectly set-up machine.
  const readyCount = summary.filter((s) => s.ok || s.pending).length
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

  // One-click launch is intentionally a body-free POST: the backend reads only the
  // saved config and uses a fixed portable command, never a user .bat or UI input.
  const startComfyui = async () => {
    setStartingComfyui(true)
    try {
      const r = await apiFetch('/api/setup/comfyui/start', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCsrfToken() },
      })
      if (r.reachable) {
        toast.success(r.already_running ? 'ComfyUI is already running.' : 'ComfyUI started.')
        await refresh(true)
      } else if (r.starting) {
        toast.info('ComfyUI is already starting. Wait a moment, then re-check.')
      } else {
        toast.error(r.error || "ComfyUI couldn't start. Check the portable install, then try again.")
      }
    } catch (e) {
      toast.error(e.message || "ComfyUI couldn't start. Check the portable install, then try again.")
    } finally {
      setStartingComfyui(false)
    }
  }

  const chooseOllamaDeployment = async (mode) => {
    setSavingOllamaMode(true)
    try {
      const data = await putJson('/api/setup/ollama-deployment', { mode })
      setConfig(data.config)
      savedConfigRef.current = JSON.stringify(data.config)
      setRuntimeReadiness((previous) => ({
        ...(previous || {}),
        ollama: data.readiness,
      }))
      // A first-run page has already stopped polling at "unconfigured". Restart
      // the lightweight loop so a launcher waiting on config.json and its sidecar
      // can become ready without a reload.
      setReadinessRevision((value) => value + 1)
      await refresh(true, { background: true })
      toast.success(mode === 'none' ? 'Ollama skipped.' : 'Ollama deployment saved.')
    } catch (error) {
      toast.error(error.message || 'Could not save the Ollama deployment.')
    } finally {
      setSavingOllamaMode(false)
    }
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
      // Two doors into ONE engine, so the ✓ belongs to the engine and each lane
      // carries its own smaller state. Attaching "✓ Ready" to the API-key row —
      // as this screen did — meant a subscriber with no key at all watched the
      // key field certify itself green, and a keyless user saw "○ Not set" on an
      // engine that was already working.
      const chatgpt = chatgptLanes(caps, secretsPresence)
      return (
        <div className="space-y-4">
          {KEY_FIELDS.map((f) => {
            const isChatgpt = f.engine === 'chatgpt'
            const laneOk = isChatgpt ? chatgpt.keySet : !!step.engines[f.engine]
            const field = (
              <div key={f.key}>
                <div className="flex items-center justify-between gap-2">
                  <span className={isChatgpt
                    ? 'text-sm font-medium text-content-muted'
                    : 'text-sm font-medium text-content'}>{f.label}</span>
                  <span className={`text-xs ${laneOk ? 'text-emerald-400' : 'text-content-subtle'}`}>
                    {laneOk ? '✓ Ready' : '○ Not set'}
                  </span>
                </div>
                <p className="text-xs text-content-muted">{f.help}</p>
                <input type="password" autoComplete="off" className={INPUT_CLASS}
                  value={secretInputs[f.key] ?? ''}
                  placeholder={secretsPresence[f.key] ? 'Already set — enter a new value to replace it' : 'Paste your key'}
                  onChange={(e) => setSecretInputs((p) => ({ ...p, [f.key]: e.target.value }))} />
                <div className="mt-1 flex items-center gap-3">
                  <a href={f.href} target="_blank" rel="noreferrer" className="text-xs text-primary underline">Get a key</a>
                  <button type="button" onClick={() => saveSecretThenTest(f.key, KEY_TEST_TARGET[f.engine])}
                    className="text-xs text-content-muted underline">Save &amp; test</button>
                </div>
              </div>
            )
            if (!isChatgpt) return field
            // FLAT, like its neighbours: the engine gets the same heading line and
            // the same right-hand state as "Gemini API key" or "OpenRouter API
            // key", and its two lanes hang under it as ordinary rows. The nested
            // bordered card this used to be described the same thing correctly
            // and broke the rhythm of the page doing it — the one engine on the
            // screen that looked like a different kind of object.
            return (
              <div key={f.key} className="space-y-3">
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-content">ChatGPT (gpt-image-2)</span>
                    <span className={`text-xs ${chatgpt.ready ? 'text-emerald-400' : 'text-content-subtle'}`}>
                      {chatgpt.ready ? '✓ Ready' : '○ Not set up'}
                    </span>
                  </div>
                  <p className="text-xs text-content-muted">{chatgptLaneSummary(chatgpt)}</p>
                </div>
                {field}
                {/* The SAME component Settings ▸ Image engines mounts — one
                    device-code flow, so the two screens cannot disagree about
                    what "connected" means. Given a label it renders as one sober
                    row (state inline, compact button) instead of a badge adrift
                    opposite a full-width call to action. */}
                <ChatgptSubscriptionConnect caps={caps} refreshCaps={refresh} toast={toast}
                  label="ChatGPT Plus/Pro subscription"
                  description={'Runs on your plan’s image quota instead of a paid key. '
                    + 'Experimental and undocumented by OpenAI — up to 5 reference images, '
                    + 'your plan’s daily cap applies, SFW only.'} />
              </div>
            )
          })}
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
      // THE reason Klein is not usable, worded by the same helper the generation
      // panel uses — so this screen can no longer name a different cause (or a
      // missing file that is sitting on the disk) from the one that refuses the
      // engine two clicks later. Null when Klein is ready.
      const kleinReason = step.kleinReason
      // A file that is ON DISK but unreadable must never render "✓ Installed": that
      // badge is what sent a user with a truncated 9.5 GB UNET to look for a file he
      // already had (zigzag4794, Discord). Three states, three actions — and the
      // word carries the state, not just the colour.
      // EVERY unreadable Klein file, not just the required trio: the consistency
      // LoRA is optional, so it was absent from step.kleinBroken — and a corrupted
      // one therefore rendered "✓ Installed" right under a button offering to
      // download it. Optional is a reason to say it QUIETLY, never a reason to hide
      // it. Severity comes from the asset (kleinAssetBlocks), not from the list.
      const kleinBrokenBy = {}
      ;(step.kleinBrokenAll || step.kleinBroken || []).forEach((i) => { kleinBrokenBy[i.asset] = i })
      const installBtn = (action, label) => {
        const bad = kleinBrokenBy[action]
        if (bad) {
          const gates = kleinAssetBlocks(action)
          return (
            <>
              <p className={`break-words text-xs ${gates ? 'text-rose-300' : 'text-amber-400'}`}>
                ⚠ On disk, unreadable — {bad.filename}.
                {gates ? '' : ' Klein still generates without it.'} Downloading again replaces it.
              </p>
              <InstallRunner action={action} buttonLabel={`↻ Download ${label.replace(/^⬇ Download /, '')} again`}
                onDone={() => refresh(true)} />
            </>
          )
        }
        return kleinMissing.includes(action)
          ? <InstallRunner action={action} buttonLabel={label} onDone={() => refresh(true)} />
          : <p className="text-xs text-emerald-400">✓ Installed</p>
      }
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
                {/* Non-blocking: the folder IS a ComfyUI install, but this process
                    cannot hand files to it (another container, a read-only mount).
                    Wraps freely — the message names a path and must stay readable
                    at 400px. */}
                {v.note && (
                  <p className="break-words text-xs text-amber-400">⚠ {v.note}</p>
                )}
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
      const configPersisted = savedConfigRef.current !== null
        && JSON.stringify(config) === savedConfigRef.current
      // A just-typed folder has a live, path-pinned verdict, while `step.dirValid`
      // describes the last saved configuration. Let a live-valid folder reveal the
      // disabled launcher so its required Save & re-check is obvious; never let that
      // transient verdict enable a start request.
      const liveDirValid = !!liveCheck && liveCheck.status === 'valid'
      const comfyLauncher = comfyuiLauncherState(step, configPersisted, liveDirValid)
      const slowComfyuiNode = step.connectionStatus === 'slow' && (
        <p className="text-xs text-amber-400" aria-live="polite">
          ComfyUI is responding slowly. Wait for it instead of starting another instance.
        </p>
      )
      const startComfyuiNode = comfyLauncher.visible && (
        <div className="space-y-1.5">
          <button type="button" onClick={startComfyui}
            disabled={startingComfyui || !comfyLauncher.enabled}
            className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-50">
            {startingComfyui ? 'Starting…' : '▶ Start ComfyUI'}
          </button>
          {comfyLauncher.reason && (
            <p className="text-xs text-content-muted" aria-live="polite">{comfyLauncher.reason}</p>
          )}
        </div>
      )
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
          {slowComfyuiNode}
          {startComfyuiNode}
          {/* Capability gap on the GRAPH's widget values, not on the files: this
              ComfyUI doesn't offer a value the Klein workflow pins. That is what
              the `beta57` scheduler did — added to ComfyUI's core list by the
              RES4LYF node pack, so it ran on the machine the graph was captured on
              and nowhere else (reported by IndependentProcess0 on Reddit). The
              shipped graph no longer does this; the line stays for the next one and
              for workflow files a user has edited. Deliberately NOT substituted
              with a near-equivalent: a scheduler changes the render, and two users
              with identical settings must not get different images. Rendered ahead
              of the weights block — no download fixes this. */}
          {/* Only when ComfyUI ANSWERED: an unreachable one makes this step's own
              fields the answer, and "⚠ Configure ComfyUI in Settings" printed on
              the ComfyUI configuration screen is noise. */}
          {step.reachable && kleinReason && (
            <p className="break-words text-xs text-rose-300">
              {kleinReason}{' '}
              {(step.unsupportedEnums || []).filter((i) => i && i.url).map((i) => (
                <a key={i.url} href={i.url} target="_blank" rel="noreferrer"
                  className="underline break-all">{i.url}</a>
              ))}
            </p>
          )}
          {/* A ✓ must mean "I checked". With ComfyUI down, the checks that need it
              (the unsupported-value probe, the node probe) fail OPEN — they report
              nothing rather than inventing a gap — so every file being on disk is
              NOT a clean bill of health, and this says so rather than letting an
              empty warning list read as one. */}
          {!step.reachable && !step.managedInitializing
            && step.kleinFilesReady && step.dirValid && (
            <p className="break-words text-xs text-amber-300">
              ⚠ Not checked — every Klein file is on disk and readable, but ComfyUI isn't
              answering, so the app could not verify that Klein actually runs here. Start
              ComfyUI and re-check.
            </p>
          )}
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
              {/* That folder is where the button below DOWNLOADS to, and it read as
                  where the file MUST be: CyberTod (Reddit) duplicated the weights
                  into it and built a symlink to claim the disk space back. The
                  resolver has always scanned both diffusion-model roots, their top
                  level and every extra_model_paths root — so say it, right here. */}
              <p>
                <span className="text-content font-medium">Already have Klein somewhere else?</span> Leave it there —
                no copy and no symlink. The app looks in any <span className="font-mono">klein</span>-named sub-folder of{' '}
                <span className="font-mono">models/unet</span> <em>or</em> <span className="font-mono">models/diffusion_models</span>,
                at the top level of either (when the file name contains <span className="font-mono">klein</span>), and in every
                root declared by your <span className="font-mono">extra_model_paths.yaml</span>. A relocated models folder is
                handled by the <span className="text-content font-medium">ComfyUI models folder</span> override in Settings.
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
          {!typedDir && !step.reachable && !step.skipped
            && !step.managedInitializing && step.managedMode !== 'external-host' && (
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
      if (step.managedInitializing) {
        return (
          <div className="space-y-4">
            <div role="status" aria-live="polite"
              className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-3 text-content">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-border-strong border-t-amber-400"
                  aria-hidden="true" />
                Initializing ComfyUI…
              </div>
              <p className="mt-1 text-xs leading-relaxed text-content-muted">
                The integrated GPU container is starting ComfyUI for you. The first startup can
                take several minutes while its environment is prepared; this page will switch to
                Ready automatically.
              </p>
            </div>
            {fields}
          </div>
        )
      }
      if (step.managedMode === 'external-host' && !step.reachable) {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-3">
              <p className="text-sm font-medium text-content">
                This Docker installation uses your existing host ComfyUI, but its API is not reachable.
              </p>
              <div className="mt-1 space-y-1.5 text-xs leading-relaxed text-content-muted">
                <p>
                  <span className="font-medium text-content">Windows:</span> start ComfyUI with its
                  usual launcher and allow it to listen on an address Docker can reach, not only
                  <span className="font-mono"> 127.0.0.1</span>. Keep port 8188 restricted to Docker
                  or your trusted private network in Windows Firewall. If its folder changed, run
                  <span className="font-mono"> start-docker.bat --configure</span> again on the host.
                </p>
                <p>
                  <span className="font-medium text-content">Linux / manual Compose:</span> mount the
                  existing ComfyUI folder at <span className="font-mono">/external-comfyui</span> and
                  configure a host-reachable API address in your Compose override. Start ComfyUI on
                  the host; this container intentionally never starts it for you.
                </p>
              </div>
              <a href="https://github.com/perfectgf/lora-dataset-studio/blob/main/docs/guide/docker.md"
                target="_blank" rel="noreferrer"
                className="mt-2 inline-block text-xs text-primary underline">
                Docker setup guide →
              </a>
            </div>
            {fields}
          </div>
        )
      }
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
            {/* The banner used to end with "Klein still needs the ${missingSummary}"
                — and missingSummary is EMPTY whenever the gap isn't a missing file
                (a corrupted weight, a widget value this ComfyUI doesn't offer), so
                the sentence rendered a blank where the cause belonged. It now
                either confirms readiness or defers to the one shared reason line
                below, which names the real gap. */}
            <div className={`rounded-md border px-3 py-2 text-sm text-content ${step.hasKlein
              ? 'border-emerald-500/30 bg-emerald-500/10' : 'border-amber-500/40 bg-amber-500/10'}`}>
              <span aria-hidden="true">{step.hasKlein ? '✓' : '⚠'}</span>{' '}
              ComfyUI is running at <span className="font-mono">{step.apiUrl || 'the configured URL'}</span>.
              {step.hasKlein
                ? ' Klein is ready — nothing to do here.'
                : ' Klein is not usable yet (local generation is optional) — see below.'}
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
      const dockerLauncher = runtimeReadiness?.comfyui?.mode === 'integrated'
        ? 'start-docker-gpu.bat' : 'start-docker.bat'
      const deploymentChoices = [
        { mode: 'none', title: 'No Ollama',
          body: 'Skip captioning and auto-framing. Nothing is downloaded.' },
        { mode: 'host', title: 'Existing host Ollama',
          body: 'Use Ollama already running on this computer.' },
        { mode: 'docker', title: 'Docker Ollama',
          body: 'Use an isolated companion container. Models stay opt-in.' },
      ]
      const deploymentCards = step.dockerManaged && (
        <div className="space-y-2">
          <p className="text-sm font-medium text-content">Choose how this Docker install uses Ollama</p>
          <div className="grid gap-2 sm:grid-cols-3">
            {deploymentChoices.map((choice) => {
              const selected = step.deploymentMode === choice.mode
              return (
                <button key={choice.mode} type="button"
                  aria-pressed={selected}
                  disabled={savingOllamaMode}
                  onClick={() => chooseOllamaDeployment(choice.mode)}
                  className={`rounded-md border p-3 text-left transition-colors disabled:opacity-50 ${selected
                    ? 'border-primary bg-primary/10'
                    : 'border-border bg-surface-raised hover:border-border-strong'}`}>
                  <span className="block text-sm font-semibold text-content">
                    {selected ? '✓ ' : ''}{choice.title}
                  </span>
                  <span className="mt-1 block text-xs leading-relaxed text-content-muted">
                    {choice.body}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      )
      const model = step.visionModel || DEFAULT_VISION_MODEL
      const pullBlock = step.reachable && !step.visionModelReady && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
          <p className="mb-1 text-sm font-medium text-content">
            Ollama is running, but the vision model isn't pulled yet — that's what powers captioning.
          </p>
          <p className="mb-2 text-xs text-content-muted">
            <span className="font-mono">{model}</span> — uncensored, needed for concept captions · {VISION_MODEL_VRAM}
          </p>
          <p className="mb-2 text-xs text-content-subtle">
            No model is downloaded automatically. This button streams the pull through LDS
            and shows its progress here.
          </p>
          <InstallRunner action="ollama_model" buttonLabel={`Pull ${model}`}
            onDone={() => refresh(true)} />
        </div>
      )
      const fields = (
        <>
          {!step.dockerManaged
            && guidedField('Ollama URL', 'ollama', 'url', 'http://127.0.0.1:11434')}
          {step.dockerManaged && step.deploymentUrl && (
            <p className="text-xs text-content-muted">
              Managed endpoint: <span className="font-mono">{step.deploymentUrl}</span>
            </p>
          )}
          {guidedField('Vision model', 'ollama', 'vision_model', DEFAULT_VISION_MODEL)}
          <p className="text-xs text-content-subtle">
            Use the ABLITERATED Qwen3-VL ({VISION_MODEL_VRAM}) — the vanilla model refuses NSFW.
            For the best captions the app pairs it with JoyCaption (ai-toolkit) — a Joy+Ollama combo.
          </p>
          {saveRecheckBtn}
        </>
      )
      if (step.unconfigured) {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-border bg-surface-raised px-3 py-3">
              <p className="text-sm font-medium text-content">Ollama is optional.</p>
              <p className="mt-1 text-xs leading-relaxed text-content-muted">
                Choose one deployment below. Selecting Docker starts only its service;
                the vision model remains absent until you explicitly pull it from this page.
              </p>
            </div>
            {deploymentCards}
          </div>
        )
      }
      if (step.disabled) {
        return (
          <div className="space-y-4">
            <div role="status" className="rounded-md border border-border bg-surface-raised px-3 py-3">
              <p className="text-sm font-medium text-content">Ollama is optional and disabled.</p>
              <p className="mt-1 text-xs leading-relaxed text-content-muted">
                Nothing is waiting in the background and no model will be downloaded.
                Choose another card below whenever you want to enable it.
              </p>
            </div>
            {deploymentCards}
          </div>
        )
      }
      if (step.managedInitializing) {
        return (
          <div className="space-y-4">
            {deploymentCards}
            <div role="status" aria-live="polite"
              className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-3 text-content">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-border-strong border-t-amber-400"
                  aria-hidden="true" />
                Starting Ollama…
              </div>
              <p className="mt-1 text-xs leading-relaxed text-content-muted">
                The companion Ollama container is starting. This page checks it quietly and
                updates automatically. No model is pulled during startup.
              </p>
              <p className="mt-1 text-xs leading-relaxed text-content-muted">
                If the launcher window is still open, leave it running. If you closed it,
                relaunch <span className="font-mono">{dockerLauncher}</span>.
              </p>
            </div>
            {fields}
          </div>
        )
      }
      if (step.reachable) {
        return (
          <div className="space-y-4">
            {deploymentCards}
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
      if (step.deploymentState === 'misconfigured') {
        return (
          <div className="space-y-4">
            <div className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-3">
              <p className="text-sm font-medium text-content">The configured Ollama URL is invalid.</p>
              <p className="mt-1 text-xs leading-relaxed text-content-muted">
                Enter an absolute <span className="font-mono">http://</span> or
                <span className="font-mono"> https://</span> origin with no path,
                credentials, query or fragment, then save and re-check.
              </p>
            </div>
            {fields}
          </div>
        )
      }
      if (step.deploymentMode === 'host') {
        return (
          <div className="space-y-4">
            {deploymentCards}
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-3">
              <p className="text-sm font-medium text-content">
                Host Ollama is selected, but it is not reachable from this container.
              </p>
              <div className="mt-1 space-y-1.5 text-xs leading-relaxed text-content-muted">
                <p>
                  <span className="font-medium text-content">Windows:</span> start Ollama on the
                  host, set <span className="font-mono">OLLAMA_HOST=0.0.0.0:11434</span>, then
                  restart Ollama. In Windows Firewall, allow TCP 11434 only from Docker or your
                  trusted private network — never expose it to the public Internet.
                </p>
                <p>
                  <span className="font-medium text-content">Linux:</span> bind Ollama to a host
                  address reachable from the Docker bridge and restrict port 11434 to that bridge
                  with your firewall. The saved LDS endpoint remains
                  <span className="font-mono"> http://host.docker.internal:11434</span>.
                </p>
                <p>After changing the host service, click this selected card again to test it.</p>
              </div>
              <a href="https://github.com/perfectgf/lora-dataset-studio/blob/main/docs/guide/docker.md"
                target="_blank" rel="noreferrer"
                className="mt-2 inline-block text-xs text-primary underline">
                Docker setup guide →
              </a>
            </div>
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
      // The list itself lives in mlInstallCards.js so the bare node --test suite
      // can hold Setup to the promises the capability strips make (JSX never
      // executes there — a list defined here is a list no test can see).
      const ML_CAPS = ML_INSTALL_CARDS
      return (
        <div className="space-y-3">
          <p className="text-sm text-content-muted">
            Optional helpers installed into this app's managed ML environments. Face scoring and masks run on
            CPU; watermark inpainting can use CUDA or CPU. The app works fully without them; they just make
            curation and training cleaner. Install each on its own below. The legacy pip bundle at the bottom covers
            the shared ML requirements only; isolated Bank engines and large model downloads stay explicit. Already installed?
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
              // Every piece the action installs, not just the first — see cardInstalled.
              const present = cardInstalled(c, caps)
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
    // Pointed at a folder that isn't usable yet → finish it, don't re-clone. The
    // copy states what was OBSERVED (no interpreter in there) instead of asserting
    // a venv is missing: portable/conda/uv/system installs have no venv and never
    // will, and telling those users to make one is a dead end. Both routes are
    // offered with the same weight, and when an interpreter is actually sitting in
    // that folder it becomes one click — the same "found on disk → use it" move
    // the directory suggestion above already makes.
    if (dir) {
      const verdict = aitoolkitVerdict(step, dir)
      return (
        <div className="space-y-4">
          <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-3 text-sm text-content space-y-2">
            <p className="font-medium break-words">{verdict.headline}</p>
            <p className="text-xs leading-relaxed text-content-muted">{verdict.body}</p>
            {verdict.candidates.map((p) => (
              <button key={p} type="button"
                onClick={() => applyDetectedPath('aitoolkit', 'python', p)}
                className="block w-full rounded-lg bg-gradient-primary px-3 py-2 text-left text-xs font-semibold text-white sm:w-auto">
                Use this Python: <span className="font-mono break-all">{p}</span>
              </button>
            ))}
            {verdict.action && (
              <SettingsLink section={verdict.settingsSection} focus={verdict.settingsFocus}
                tone="warning" className="block">
                {verdict.action}
              </SettingsLink>
            )}
          </div>
          {fields}
        </div>
      )
    }
    return (
      <GuidedSteps
        intro="ai-toolkit trains the LoRA. Install it once, give it a Python, then point the app at its folder."
        steps={AITOOLKIT_INSTALL_STEPS}
        link={{ href: 'https://github.com/ostris/ai-toolkit', label: 'ai-toolkit on GitHub →' }}>
        {fields}
      </GuidedSteps>
    )
  }

  const kind = SCREENS[screen]
  const DONE = SCREENS.length - 1
  const INSTALL = SCREENS.indexOf('install')   // the install/reinstall step, after config
  const isReady = (id) => stepById[id].status === 'ready' || stepById[id].disabled
  const toolIdx = (id) => SETUP_STEP_IDS.indexOf(id)
  // welcome=0, tools=1..N — and, for the two screens that are not tool steps
  // ('install', 'done'), their index in SCREENS. Without that second lookup a
  // capability row pointing at the install/repair menu (Krea 2 Edit, whose only
  // one-click installer lives there) silently resolved to -1+1 = 0 and dumped the
  // user on the welcome screen. Unknown ids still fall back to welcome, as before.
  const screenOf = (id) => {
    const i = SETUP_STEP_IDS.indexOf(id)
    if (i >= 0) return i + 1
    const j = SCREENS.indexOf(id)
    return j > 0 ? j : 0
  }
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
    if (!s || s.status === 'ready' || s.disabled) return null
    if (s.unconfigured) {
      return 'Choose No Ollama, Existing host Ollama or Docker Ollama on this page before continuing.'
    }
    if (s.managedInitializing) {
      return 'The companion Ollama container is still starting. This page will continue automatically when it is ready.'
    }
    if (!s.reachable) {
      if (s.deploymentMode === 'host') {
        return 'Host Ollama is selected but unreachable from Docker. Start it on the host and make port 11434 reachable from Docker, or choose Docker Ollama on this page.'
      }
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
      if (!cfgDir && !s.reachable && !s.skipped && !s.managedInitializing
          && s.managedMode !== 'external-host') {
        setSkipConfirm(true); return
      }
    }
    setAdvancing(true)
    try {
      await persist()
      let fresh = null
      try { fresh = await apiFetch('/api/capabilities') } catch { /* keep going */ }
      if (fresh && kind === 'ollama') {
        const reason = ollamaGateReason(
          deriveSetupSteps(fresh, runtimeReadiness).find((x) => x.id === 'ollama'),
        )
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
      {SETUP_STEP_IDS.map((id) => {
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
    const ollamaScan = oll.disabled
      ? { state: 'skipped', partial: 'disabled by this deployment' }
      : oll.managedInitializing
        ? { state: 'initializing', partial: 'companion container is starting' }
        : oll.reachable
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
        state: stepById.comfyui.managedInitializing ? 'initializing'
          : stepById.comfyui.skipped ? 'skipped'
            : triState(stepById.comfyui.reachable, stepById.comfyui.hasKlein),
        partial: stepById.comfyui.managedInitializing
          ? 'first startup in progress' : 'running — Klein model optional' },
      { label: 'Captioning — Ollama + vision model', optional: oll.disabled, stepId: 'ollama',
        state: ollamaScan.state, partial: ollamaScan.partial },
      { label: 'LoRA training — ai-toolkit', stepId: 'training',
        state: stepById.training.valid ? 'ready'
          : (detected && detected.aitoolkit && detected.aitoolkit.dir ? 'partial' : 'missing'),
        partial: 'found on disk — one click to use' },
    ]
    const SCAN_META = {
      ready: { glyph: '✓', cls: 'text-emerald-400', word: 'ready' },
      initializing: { glyph: '↻', cls: 'text-amber-400', word: '' },
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
              const soft = r.optional && !['ready', 'initializing'].includes(r.state)
              // optional + not ready → neutral, except a real managed boot state.
              const m = soft ? { ...SCAN_META[r.state], ...NEUTRAL } : SCAN_META[r.state]
              const word = ['partial', 'initializing'].includes(r.state) ? r.partial
                : r.state === 'skipped' && r.partial ? r.partial
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
              const rowOk = s.ok || s.pending
              const noteEl = s.pending && s.note ? (
                <span className="text-xs italic text-content-subtle"> — {s.note}</span>
              ) : null
              if (!targetStep) {
                return (
                  <li key={s.label} className={`flex items-center gap-2 px-2 py-1 text-sm ${rowOk ? 'text-content' : 'text-content-subtle'}`}>
                    <span aria-hidden="true" className={rowOk ? 'text-emerald-400' : 'text-content-subtle'}>{rowOk ? '✓' : '✗'}</span>
                    <span>{s.label}{noteEl}</span>
                  </li>
                )
              }
              return (
                <li key={s.label}>
                  <button type="button" onClick={() => setScreen(screenOf(targetStep))}
                    className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1 text-left text-sm
                      cursor-pointer transition-colors hover:bg-surface-raised focus:outline-none focus-visible:ring-2
                      focus-visible:ring-primary ${rowOk ? 'text-content' : 'text-content-subtle'}`}>
                    <span className="flex items-center gap-2">
                      <span aria-hidden="true" className={rowOk ? 'text-emerald-400' : 'text-content-subtle'}>{rowOk ? '✓' : '✗'}</span>
                      <span>{s.label}{noteEl}</span>
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
