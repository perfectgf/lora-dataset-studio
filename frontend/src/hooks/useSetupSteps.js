// Pure derivation of the guided Setup wizard state from live capabilities.
// No I/O — deterministic, so it is the single source of truth for card status.
// Pure JS on purpose (no JSX): node --test drives it directly, which is what
// keeps the capability destinations below honest.
import { getHelpTopic } from '../help/helpRegistry.js'
import { SETTINGS_SECTIONS } from '../components/settings/registry.js'
// ONE answer to "is this local engine ready", shared with the generation panel and
// the ✦ Edit modal. Setup asking a different question than the page that consumes
// the model is exactly how ✓ and ⚠ ended up on the same install.
import { localEngineReadiness } from '../utils/localEngineReason.js'
import { blockingInvalid, integrityCause } from '../utils/modelIntegrityWords.js'
import { KLEIN_REQUIRED_ASSETS, KLEIN_ASSET_LABELS, kleinMissingLabels, kleinAssetBlocks }
  from '../utils/kleinAssets.js'

// Re-exported from their leaf module (utils/kleinAssets.js) so existing importers
// keep working; they moved down there to let this file import the readiness
// verdict without an import cycle.
export { KLEIN_ASSET_LABELS, kleinMissingLabels }

export const SETUP_STEP_IDS = ['image', 'comfyui', 'ollama', 'quality', 'training']

// Every wizard screen a /setup?step=<id> link may target. The tool steps PLUS
// 'install' — the install/repair menu is a real destination (the one-click
// engine installs live there), it just isn't a tool to configure. SetupPage's
// SCREENS list is the display order; this one is the LINKABLE set, used by the
// What's-new target validator and the help-registry contract test.
export const SETUP_DEEP_LINK_STEPS = [...SETUP_STEP_IDS, 'install']

// Tool reachable + its extra piece present -> ready; reachable only -> partial.
function gateStatus(reachable, complete) {
  if (reachable && complete) return 'ready'
  if (reachable) return 'partial'
  return 'available'
}

function imageStep(caps) {
  const e = caps.engines || {}
  const ready = e.nanobanana || e.chatgpt || e.openrouter || e.klein
  return {
    id: 'image', title: 'Image generation', recommended: true,
    unlocks: ['Nano Banana (Gemini)', 'ChatGPT (gpt-image-2)', 'OpenRouter', 'Klein (local)'],
    status: ready ? 'ready' : 'available',
    engines: { nanobanana: !!e.nanobanana, chatgpt: !!e.chatgpt,
      openrouter: !!e.openrouter, klein: !!e.klein },
  }
}

// The ChatGPT engine has TWO independent doors, and Setup only ever showed one.
// A single "OpenAI API key — ○ Not set" row taught every Plus/Pro subscriber, on
// the first screen they see, that this engine is pay-per-use — while the
// subscription lane (Codex device-code OAuth) had been shipping for months on a
// Settings page they had no reason to open yet. Either door lights the engine up,
// so the step must show both and go green on either.
//
// `keySet` comes from the SECRETS PRESENCE payload, not from engines.chatgpt:
// the backend ORs the two lanes into that one boolean (probe_openai), so once a
// subscription is connected it can no longer answer "is a key also set". Two
// lanes need two signals. `ready` still defers to the backend's OR, with the
// locally-visible lanes as the fallback for a payload that predates it.
export const CHATGPT_API_KEY_SECRET = 'OPENAI_API_KEY'

export function chatgptLanes(caps, secretsPresence) {
  const c = caps || {}
  const sub = c.chatgpt_subscription || {}
  const connected = !!sub.connected
  const keySet = !!(secretsPresence || {})[CHATGPT_API_KEY_SECRET]
  const engine = (c.engines || {}).chatgpt
  return {
    ready: typeof engine === 'boolean' ? (engine || keySet || connected) : (keySet || connected),
    keySet,
    connected,
    // Both doors open is not a conflict to warn about — it is the belt-and-braces
    // setup the `auto` auth mode was written for (subscription first, key as the
    // fallback the user explicitly paid for). Named so the wizard can say which
    // one is actually carrying the engine instead of showing two green ticks with
    // no explanation of which one runs.
    both: keySet && connected,
  }
}

/** The one sentence under the ChatGPT heading in Setup — what is powering the
 *  engine right now, or what is still needed. Pure so node --test locks the
 *  wording, which is the part that was misleading. */
export function chatgptLaneSummary(lanes) {
  const l = lanes || {}
  if (l.both) {
    return 'Both lanes are set up. The engine uses your subscription first and '
      + 'falls back to the API key — change that in Settings ▸ Image engines.'
  }
  if (l.connected) return 'Ready through your ChatGPT subscription. No API key needed.'
  if (l.keySet) return 'Ready through your OpenAI API key, billed per image.'
  return 'Set up either one — an API key, or your ChatGPT Plus/Pro subscription. '
    + 'One is enough.'
}

function kleinMissingRequired(c) {
  if (Array.isArray(c.klein_missing)) {
    return c.klein_missing.filter((a) => KLEIN_REQUIRED_ASSETS.includes(a))
  }
  // Legacy fallback: judge on the UNET scan alone (the old, under-strict signal).
  return (c.models && c.models.klein && c.models.klein.length) ? [] : ['klein_model']
}

// The user's choice to "continue without ComfyUI" (Setup step). The backend
// derives `comfyui.skipped` = the stored flag AND no directory configured, so it
// self-annuls the moment a path is entered. We only treat the step as skipped when
// ComfyUI is ALSO not reachable — a running ComfyUI is worth surfacing (partial/
// ready) even if the user once clicked skip.
function comfyuiStep(caps, runtimeReadiness) {
  const c = caps.comfyui || {}
  const managed = (runtimeReadiness && runtimeReadiness.comfyui) || {}
  const missingRequired = kleinMissingRequired(c)
  // The BACKEND's verdict on each local engine, plus the one sentence explaining
  // it. Setup renders these; it no longer decides them. See localEngineReadiness.
  const klein = localEngineReadiness('klein', caps)
  const krea = localEngineReadiness('krea', caps)
  // Present-but-INVALID required assets (a licence-gate HTML page saved as
  // .safetensors, a truncated download): the file exists so it is NOT in
  // klein_missing, yet it can't load. Without this the step would go green and let
  // a doomed generate crash ComfyUI (the #help "Expecting value: line 1 column 1").
  // Only *blocking* invalids gate readiness; the advisory too_small does not.
  const kleinInvalid = Array.isArray(c.klein_invalid) ? c.klein_invalid : []
  const kleinBroken = blockingInvalid(kleinInvalid, KLEIN_REQUIRED_ASSETS)
  // EVERY blocking-invalid Klein asset, gating or not. `kleinBroken` above is the
  // GATING subset and must stay that way — it is what decides readiness. But the
  // DOWNLOAD buttons read it too, and that is how a corrupted consistency LoRA came
  // out as "✓ Installed": not required, so not in kleinBroken, so never mentioned
  // anywhere on the screen that offers to download it. Two lists, two jobs — one
  // gates the step, one is what the screen has to SHOW.
  const kleinBrokenAll = blockingInvalid(kleinInvalid)
  // Widget values the graph pins that THIS ComfyUI doesn't offer. Every file can be
  // in place and every node class present, and the first generation still dies on a
  // raw ComfyUI 400 — that was the `beta57` scheduler, which the RES4LYF pack adds
  // to ComfyUI's own list (reported by IndependentProcess0 on Reddit). The shipped
  // graph is clean now; this remains the net. Nothing is substituted (a scheduler
  // changes the render), so this step is where the user learns it. Empty on a
  // capable install AND on an unreachable one — the probe fails open.
  const unsupportedEnums = Array.isArray(c.klein_unsupported_enums)
    ? c.klein_unsupported_enums : []
  // The SAME conditions the backend applies — kept ONLY as the fallback for a
  // payload that predates `engines.klein` (and for the unit tests that drive this
  // module with a bare comfyui object). It is not the answer any more.
  const derivedHasKlein = missingRequired.length === 0 && kleinBroken.length === 0
    && unsupportedEnums.length === 0
  // hasKlein IS the engine's readiness, straight from the backend — the same value
  // the Generate page gates its Klein button on. Setup re-deriving it is what let
  // this screen show ✓ while that one refused: a truncated 9.5 GB UNET is present
  // (so nothing is "missing"), and only the integrity verdict tells them apart.
  const hasKlein = typeof (caps.engines || {}).klein === 'boolean'
    ? klein.ready : derivedHasKlein
  // Which assets to still offer a download for (required trio + recommended LoRA),
  // so each button can grey out on its own once its file lands.
  const kleinMissing = Array.isArray(c.klein_missing)
    ? c.klein_missing
    : (derivedHasKlein ? [] : ['klein_model'])
  // Skipped is neutral, not a warning — but only when there's genuinely nothing to
  // show (unreachable). It never overrides a reachable ComfyUI's real status.
  const managedInitializing = managed.mode === 'integrated'
    && managed.state === 'starting' && !c.reachable
  const skipped = !!c.skipped && !c.reachable && !managedInitializing
  const status = managedInitializing
    ? 'initializing'
    : (skipped ? 'skipped' : gateStatus(c.reachable, hasKlein))
  return {
    id: 'comfyui', title: 'ComfyUI — local generation & Test Studio', recommended: false,
    unlocks: ['Klein engine', 'Test Studio'],
    status, reachable: !!c.reachable,
    managedMode: managed.mode || 'external',
    managedInitializing,
    connectionStatus: c.status || (c.reachable ? 'ok' : 'unreachable'),
    hasKlein, kleinMissing, kleinInvalid, apiUrl: c.api_url || '',
    skipped,
    // The ONE sentence for why each local engine is dark, identical to the one the
    // generation panel shows — so the two screens can no longer name different
    // causes for one gap. Null when the engine is ready.
    kleinReason: hasKlein ? null : klein.reason,
    kreaReason: krea.ready ? null : krea.reason,
    // Blocking-invalid REQUIRED weights: on disk, right size, unreadable. Their fix
    // is delete-then-download, not download — a different action from "missing",
    // which is why they get their own field instead of being folded into it.
    kleinBroken,
    // Same list WITHOUT the required-only filter, for the buttons (see above). The
    // consumer decides the tone from the asset, never from membership here.
    kleinBrokenAll,
    // Every required file is on disk and readable. Told apart from hasKlein because
    // "the files are fine but ComfyUI is down" and "a file is broken" are different
    // sentences with different fixes.
    kleinFilesReady: missingRequired.length === 0 && kleinBroken.length === 0,
    // Could the checks that NEED a live ComfyUI actually run? False = "not checked",
    // which the UI must render as ⚠, never as ✓. The unsupported-value probe fails
    // OPEN (an unreachable ComfyUI reports no gap rather than inventing one), so
    // without this flag an empty list would read as a clean bill of health.
    kleinVerified: !!c.reachable,
    // [{node_id, class_type, input, value, pack, url}] — render with
    // comfyEnumUnavailableReason(), which names the node pack when we know it and
    // never invents a ComfyUI version number when we don't.
    unsupportedEnums,
    // Whether comfyui.base_dir actually points at a ComfyUI install (main.py + models/):
    // a wrong/portable-wrapper path scans an empty models/ and finds no checkpoints.
    // baseDir = the path this verdict was PROBED against — the UI must not show the
    // verdict for a freshly typed (unsaved) path, it would judge the wrong string.
    dirConfigured: !!c.dir_configured, dirValid: !!c.dir_valid, resolvedDir: c.resolved_dir || '',
    baseDir: c.base_dir || '',
    portableLauncherSupported: !!c.portable_launcher_supported,
    portableLauncherLocalApi: !!c.portable_launcher_local_api,
  }
}

// The server remains the authority. A live-valid directory only makes the disabled
// affordance discoverable; starting still requires the saved, freshly re-checked
// configuration because the no-payload POST reads that configuration.
export function comfyuiLauncherState(step, configPersisted, liveDirValid = false) {
  // Docker either owns the integrated process or connects to a process on the
  // host. In neither case can a button inside the app safely launch a Windows
  // portable executable. Hide it for the whole deployment lifetime, including
  // the brief window where runtime readiness is ahead of the full caps refresh.
  if (step && ['integrated', 'external-host'].includes(step.managedMode)) {
    return { visible: false, enabled: false, reason: '' }
  }
  // Never offer a second process while the saved probe says one is answering (or
  // still answering slowly). This comes before the live-directory affordance.
  if (!step || step.reachable || step.connectionStatus === 'slow') {
    return { visible: false, enabled: false, reason: '' }
  }
  // The live verdict is deliberately an UNSAVED-only affordance. Once the config
  // is saved, only the server's re-check (`step.dirValid`) can satisfy this gate.
  if (!step.dirValid && !(liveDirValid && !configPersisted)) {
    return { visible: false, enabled: false, reason: '' }
  }
  if (!configPersisted) {
    return { visible: true, enabled: false, reason: 'Save & re-check the ComfyUI settings before starting it from LDS.' }
  }
  if (!step.portableLauncherSupported) {
    return { visible: true, enabled: false, reason: 'This button supports only the NVIDIA portable ComfyUI install. Your usual launcher is unchanged.' }
  }
  if (!step.portableLauncherLocalApi) {
    return { visible: true, enabled: false, reason: 'Set ComfyUI to its local address on port 8188 before starting it from LDS.' }
  }
  return { visible: true, enabled: true, reason: '' }
}

// What "continue without ComfyUI" costs vs keeps — shown in the skip-confirmation
// panel BEFORE the user commits. Sourced from the real capability gates (n'invente
// rien): studio_visible / engines.klein / watermark_klein key on ComfyUI being
// reachable with its models; the training base listers and the LoRA preset picker
// resolve from comfyui.base_dir. Everything under KEPT is independent of ComfyUI.
export const COMFYUI_SKIP_LOST = [
  'Local Klein generation, including the uncensored (NSFW) local lane',
  'Watermark cleaning with Klein (LaMa inpainting and crop still work)',
  'Test Studio (comparing checkpoints, every model family)',
  'Training on your own ComfyUI base models (built-in and cloud bases still work)',
  'Picking LoRA presets from what is on disk (free-text entry still works)',
]
export const COMFYUI_SKIP_KEPT = [
  'Scraping and dataset curation',
  'Captioning (Ollama vision model or the API engines)',
  'Nano Banana, ChatGPT and OpenRouter image engines',
  'LoRA training — local ai-toolkit and cloud (vast.ai)',
  'Publishing datasets and LoRAs to Hugging Face',
]

// What "continue without Ollama" costs vs keeps. Same rule as the ComfyUI lists
// above: every line is sourced from a real gate, nothing is invented.
//   LOST  — classifyFramingGate.js ("Ollama is the only backend for this pass"),
//           detect_head_bbox, lora_test_studio.describe/enhance_test_prompt,
//           bank_filter_translator.translate, watermark_detect.backend='vision',
//           and the caption_short derivation (text-only, Ollama-only).
//   KEPT  — captioning itself, because JoyCaption writes the SAME prompt: the
//           caption style is chosen by train_type (prose for Z-Image, booru for
//           SDXL) and handed to BOTH engines, so JoyCaption is not an SDXL-only
//           fallback. The wizard used to claim otherwise and gated on it.
export const OLLAMA_SKIP_LOST = [
  'Auto-classify framing (📐) on datasets and the bank',
  'Auto head-crop when a dataset is built from a reference photo',
  'Test Studio 🔎 Describe and ✨ Enhance',
  'The bank’s “Describe filter” natural-language search',
  'Watermark detection through the vision route (the detector engine still works)',
  'Short captions derived from long ones',
]
export const OLLAMA_SKIP_KEPT = [
  'Captioning with JoyCaption — prose or booru tags, matched to what you train',
  'Scraping, dataset curation and the bank',
  'Local generation, Test Studio comparisons and the Canvas (ComfyUI)',
  'LoRA training — local ai-toolkit and cloud (vast.ai)',
  'Nano Banana, ChatGPT and OpenRouter image engines',
]

// The KEPT list as THIS machine may claim it. Every other line is true of any
// install; the captioning one is not — ticking it where JoyCaption is absent would
// promise a captioner that isn't there. The panel's amber note names the fix, so
// the line is withheld rather than reworded into a half-promise.
export function ollamaSkipKept(joycaptionReady) {
  return joycaptionReady
    ? OLLAMA_SKIP_KEPT
    : OLLAMA_SKIP_KEPT.filter((t) => !/^Captioning with JoyCaption/.test(t))
}

// Why the wizard would keep you on the Ollama step, or `null` when it would not.
//
// This step used to HARD BLOCK, on the premise that "JoyCaption only covers SDXL
// booru tags" — which the code it guarded contradicts: the caption STYLE follows
// the TRAIN TYPE (face_dataset_service picks prose unless sdxl) and the resulting
// prompt is handed to BOTH engines, so JoyCaption writes the very prose Z-Image
// wants. A ready JoyCaption therefore lifts the gate outright, and a conscious
// skip lifts it too; with neither, SetupPage offers the skip panel rather than a
// wall. What Ollama alone still unlocks — framing, head-crop, Describe/Enhance,
// the bank's NL filter, short captions — stays listed, counted and honestly
// absent (OLLAMA_SKIP_LOST, and the capability summary below).
//
// Pure, and living here rather than inside the page, so it can be re-evaluated
// against FRESH capabilities after a save — and so `node --test` can hold every
// branch of it, which is what the page's own closures can never offer.
export function ollamaGateReason(s) {
  if (!s || s.status === 'ready' || s.disabled) return null
  // Asked BEFORE the lifts below: an unconfigured Docker install is not a capability
  // question but an unanswered one — nothing starts until a card is picked, and the
  // 'No Ollama' card is itself one of the answers. Having a captioner elsewhere must
  // not wave that choice through, or the companion container is never brought up.
  if (s.unconfigured) {
    return 'Choose No Ollama, Existing host Ollama or Docker Ollama on this page before continuing.'
  }
  if (s.skipped || s.joycaptionReady) return null
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
  if (!s.visionModelReady) return 'Pull the vision model below to continue — with no JoyCaption installed, it is the only captioner this install has.'
  return 'Finish this step to continue.'
}

// Map a /api/setup/comfyui-dir verdict to the wizard's inline feedback: a tone
// (drives the colour) and an actionable message. `suggestion` is carried through so
// the caller can render an "adopt this folder" button for the launcher-folder case.
// Pure + exhaustive so node --test can lock every branch. `checking` is the UI's own
// in-flight state; `empty` (nothing typed) renders nothing here — the skip panel owns it.
export function comfyuiDirVerdict(check) {
  const c = check || {}
  const resolved = c.resolved || ''
  const suggestion = c.suggestion || ''
  const note = inputFolderNote(c.input_check)
  switch (c.status) {
    case 'valid':
      return { tone: 'ok', suggestion: '', note,
        message: resolved ? `ComfyUI found at ${resolved}.` : 'ComfyUI found.' }
    case 'nested':
      return { tone: 'warn', suggestion, note,
        message: `This looks like the launcher/parent folder — did you mean ${suggestion}?` }
    case 'missing':
      return { tone: 'warn', suggestion: '', note: '',
        message: "That folder doesn't exist yet — check the path." }
    case 'empty_dir':
      return { tone: 'warn', suggestion: '', note: '',
        message: 'That folder is empty — point at the folder that holds main.py and a models/ folder.' }
    case 'not_comfyui':
      return { tone: 'warn', suggestion: '', note: '',
        message: "This folder isn't a ComfyUI install — it must contain main.py and a models/ folder. "
          + 'For the portable build, point at the inner …\\ComfyUI_windows_portable\\ComfyUI.' }
    default:
      return { tone: 'muted', suggestion: '', note: '', message: '' }
  }
}

// The SECOND half of "is this ComfyUI usable": every local engine hands its source
// image over by COPYING it into ComfyUI's input/ folder. A URL that answers proves
// nothing about that — with ComfyUI in another container the copy lands nowhere
// ComfyUI can see, and the first generation used to die on a detail-free 500
// (reported on Discord by nofaceman). So the wizard says it here, at configuration
// time. Deliberately a NOTE, never a blocker: mounting the volumes afterwards is a
// perfectly normal order of operations. Empty string = nothing to say (not probed,
// or fine). The backend already redacted the path.
function inputFolderNote(inputCheck) {
  const c = inputCheck || {}
  if (c.ok !== false) return ''
  return c.problem || 'The app cannot write into ComfyUI’s input folder.'
}

// ── ai-toolkit: what the wizard says about the folder it was pointed at ───────
// The old copy said one thing — "set up its Python venv per the README" — which
// names a CAUSE the app never verified and a remedy that is a dead end for whole
// families of installs: conda, uv, the system Python, and the portable/embedded
// bundles that ship a `python_embeded\python.exe` instead of a venv. Someone
// running one of those reasonably concluded the app REQUIRED a venv and went
// asking in public (reported on Reddit by Psyko_2000) instead of filling in a
// setting that was three clicks away. So the copy below states the OBSERVATION
// ("no interpreter found here") and opens both doors with equal weight — make
// one, or name the one you already have. Pure strings + a shape: node --test
// locks the wording, which is the part that was wrong.
const AITOOLKIT_PYTHON_SETTING = 'Settings ▸ Local tools ▸ ai-toolkit Python interpreter'

// The install path, before any folder is configured. Same rule: the venv is ONE
// way to give ai-toolkit a Python, not the definition of a working install.
export const AITOOLKIT_INSTALL_STEPS = [
  { text: 'Clone ai-toolkit, or install it with the script of your choice.',
    command: 'git clone https://github.com/ostris/ai-toolkit' },
  { text: 'Give it a Python. Its README walks through creating a venv in that '
      + 'folder — or, if you already run it with a conda or uv environment, the '
      + 'system Python, or a portable/embedded build, keep that one and name it '
      + `in ${AITOOLKIT_PYTHON_SETTING}.` },
  { text: 'Point the app at the folder that holds run.py, below.' },
]

export function aitoolkitVerdict(step, dir) {
  const s = step || {}
  const path = (dir || '').trim()
  const base = { candidates: [], settingsSection: 'local-tools' }
  if (s.valid) {
    return { ...base, kind: 'ready', tone: 'ok',
      headline: `ai-toolkit is set up at ${path}.`, body: 'Nothing to do here.' }
  }
  if (!path) return { ...base, kind: 'unconfigured', tone: 'muted', headline: '', body: '' }
  if (!s.dirValid) {
    return { ...base, kind: 'not_a_checkout', tone: 'warn',
      headline: `No run.py in ${path}, so this isn't an ai-toolkit folder.`,
      body: "Point at the folder ai-toolkit was installed into — the one holding run.py." }
  }
  return {
    ...base,
    kind: 'no_interpreter',
    tone: 'warn',
    // The finding, not a diagnosis. True for every install shape.
    headline: `ai-toolkit is here, but no Python interpreter was found in ${path}.`,
    body: "The app doesn't know which Python to run it with. Two ways forward, "
      + "both fine: create a venv inside that folder (ai-toolkit's README walks "
      + 'through it), or keep the Python you already run ai-toolkit with — a conda '
      + 'or uv environment, your system Python, or the python.exe of a portable / '
      + 'embedded build (python_embeded) — and tell the app where it is.',
    action: `Set the interpreter in ${AITOOLKIT_PYTHON_SETTING}`,
    // The action names ONE field, so the link lands on it (SettingsLink focus →
    // SettingsPage's ?focus= deep link). Only this verdict carries an action, so
    // only this one needs a target; the DOM id is LocalToolsSection's.
    settingsFocus: 'aitoolkit-python',
    candidates: s.pythonCandidates || [],
  }
}

function ollamaStep(caps, runtimeReadiness) {
  const o = caps.ollama || {}
  const managed = (runtimeReadiness && runtimeReadiness.ollama) || {}
  const deploymentMode = managed.mode || 'local'
  const deploymentState = managed.state || ''
  const dockerManaged = ['unconfigured', 'none', 'host', 'docker'].includes(deploymentMode)
  const deploymentConfigured = deploymentMode !== 'unconfigured'
  const deploymentUrl = deploymentMode === 'host'
    ? 'http://host.docker.internal:11434'
    : deploymentMode === 'docker' ? 'http://ollama:11434' : ''
  const normalizedCapabilityUrl = String(o.url || '').replace(/\/+$/, '')
  // Lightweight runtime readiness owns reachability in Docker. The full caps
  // snapshot may still describe the previous endpoint while its refresh retries;
  // never transfer a model-ready verdict across that endpoint switch.
  const capabilityMatchesDeployment = !dockerManaged || !deploymentUrl
    || normalizedCapabilityUrl === deploymentUrl
  const reachable = dockerManaged ? !!managed.ready : !!o.reachable
  const visionModelReady = reachable && capabilityMatchesDeployment
    && !!o.vision_model_ready
  const disabled = deploymentMode === 'none'
  const unconfigured = deploymentMode === 'unconfigured'
  const managedInitializing = deploymentMode === 'docker'
    && managed.state === 'starting' && !managed.ready
  // Conscious "continue without Ollama". The backend derives it (stored flag AND
  // not reachable), so a reachable Ollama can never read as skipped — its real
  // state, model gap included, always wins. A Docker deployment set to 'none'
  // reaches the same neutral status by its own route.
  const skipped = !dockerManaged && !!o.skipped
  // JoyCaption covers captioning on its own — the caption style follows the
  // TRAIN TYPE (prose for Z-Image, booru for SDXL) and the same prompt goes to
  // both engines, so its presence is what turns this step from a gate into a
  // recommendation. Everything else Ollama unlocks stays genuinely unavailable.
  const joycaptionReady = !!((caps.captioners || {}).joycaption)
  const status = unconfigured
    ? 'available'
    : (disabled || skipped)
    ? 'skipped'
    : (managedInitializing
        ? 'initializing'
        : gateStatus(reachable, visionModelReady))
  return {
    id: 'ollama', title: 'Ollama — captioning & auto-framing', recommended: false,
    unlocks: ['Captioning', 'Auto-classify framing', 'Auto head-crop'],
    status, reachable, visionModelReady, skipped, joycaptionReady,
    deploymentMode, deploymentState, deploymentConfigured, deploymentUrl,
    dockerManaged, disabled, unconfigured, managedInitializing,
    url: deploymentUrl || o.url || '', visionModel: o.vision_model || '',
    // Execution-independent install signal (binary on disk) vs `reachable` (server
    // answering): installed && !reachable -> "installed but stopped", offer a Start.
    installed: !!o.installed, binaryPath: o.binary_path || '',
  }
}

function qualityStep(caps) {
  // Four required scoped ML capabilities (face scoring, masks, watermark
  // inpainting, bank scoring) each install/repair on their own. SigLIP2 and the
  // watermark detector are explicit optional downloads: their cards stay visible,
  // but neither turns an existing healthy install back to "partial" after update.
  const parts = [!!caps.face_scoring, !!caps.masks, !!caps.watermark_inpaint,
    !!caps.bank_scoring]
  const ready = parts.every(Boolean)
  const partial = parts.some(Boolean)
  return {
    id: 'quality', title: 'Quality tools (ML extras)', recommended: false,
    unlocks: ['Face-similarity scoring', 'Person masks', 'Watermark inpainting',
      'Bank scoring (aesthetic · NSFW · style)',
      'SigLIP2 Bank semantics (optional)',
      'Watermark detector (optional)', 'Scraping extras (optional)'],
    status: ready ? 'ready' : (partial ? 'partial' : 'available'),
    faceScoring: !!caps.face_scoring, masks: !!caps.masks,
    watermarkInpaint: !!caps.watermark_inpaint,
    bankScoring: !!caps.bank_scoring,
    // Optional like the watermark accelerator: its card remains visible, but an
    // existing CLIP-ready install does not become "partial" after this update.
    bankSiglip2: !!caps.bank_siglip2,
    watermarkDetect: !!caps.watermark_detect,
    // Also optional and also install-from-here (its own Setup card lives in
    // this step, see mlInstallCards.js) — same non-gating treatment.
    scrapeDeps: !!caps.scrape_deps,
  }
}

function trainingStep(caps) {
  const a = caps.aitoolkit || {}
  return {
    id: 'training', title: 'LoRA training — ai-toolkit', recommended: false,
    unlocks: ['LoRA training', 'JoyCaption captioning (bonus)'],
    status: a.valid ? 'ready' : 'available',
    valid: !!a.valid,
    // dirValid = run.py is there. Told apart from `valid` so the wizard can say
    // WHICH of the two problems it hit instead of one blanket sentence.
    dirValid: !!a.dir_valid,
    pythonCandidates: Array.isArray(a.python_candidates) ? a.python_candidates : [],
  }
}

export function deriveSetupSteps(caps, runtimeReadiness = null) {
  const c = caps || {}
  return [imageStep(c), comfyuiStep(c, runtimeReadiness),
    ollamaStep(c, runtimeReadiness), qualityStep(c), trainingStep(c)]
}

// The user's live capability checklist (Summary card). Watermark inpainting is a
// distinct ML extra (simple-lama-inpainting) — an existing install that never ran
// it must SEE it as still missing here, not be told "everything's ready".
export function deriveCapabilitySummary(caps) {
  const c = caps || {}
  const e = c.engines || {}
  const o = c.ollama || {}
  const cap = c.captioners || {}
  // "Configured but ComfyUI isn't running" is NOT a missing capability: the
  // install is fine, the process just isn't up. Those rows show as OK with a
  // "launch ComfyUI to enable" note instead of a discouraging ✗.
  const cu = c.comfyui || {}
  const comfyOff = !!cu.dir_valid && !cu.reachable
  const NOTE = 'launch ComfyUI to enable'
  // `topic` = the help-registry topic that OWNS the control turning this
  // capability on. It is not a second navigation table: the route + focus id
  // are read back from the registry (capabilityDestination below), so a field
  // that moves section moves the tile with it. `waitingTopic` is the door for
  // the pending state — "the install is fine, the process isn't up" is not the
  // same problem as "it isn't installed", so it must not lead to the same page.
  const WAITING = 'comfyui.api_url'
  // Krea's two distinct "not ready" causes, kept apart because the actions are
  // different: something is genuinely absent from disk (install it) vs everything
  // is installed and ComfyUI simply has not loaded the node pack yet (restart).
  const kreaDiskGap = !!(Array.isArray(cu.krea_missing) && cu.krea_missing.length)
    || !cu.krea_nodes_installed
  const kreaRestartPending = kreaNeedsComfyuiRestart(c)
  return [
    { label: 'Nano Banana (Gemini)', ok: !!e.nanobanana, topic: 'GEMINI_API_KEY' },
    { label: 'ChatGPT (gpt-image-2)', ok: !!e.chatgpt, topic: 'engines.chatgpt_auth' },
    { label: 'OpenRouter', ok: !!e.openrouter, topic: 'OPENROUTER_API_KEY' },
    { label: 'Klein (local)', ok: !!e.klein,
      topic: 'setup-comfyui', waitingTopic: WAITING,
      ...(!e.klein && comfyOff ? { pending: true, note: NOTE } : {}) },
    // Krea 2 Edit is COUNTED even though it is optional, and even though nothing
    // else in Setup used to mention it. Leaving it out of the list was worse than
    // showing it red: the final screen certified "11 of 11 capabilities ready" on
    // a machine where a whole engine was missing, so the user finished setup
    // believing there was nothing left — and met a dark engine card weeks later.
    // A capability that is absent must be VISIBLE and counted, never removed from
    // the denominator.
    { label: 'Krea 2 Edit (local)', ok: !!e.krea,
      topic: 'setup-krea-install', waitingTopic: WAITING,
      // Two different "not yet": nothing is on disk (a real install to do, so a
      // plain ✗ pointing at the install screen), or everything is there and only
      // ComfyUI is down/not restarted — the pending note, which must not read as
      // "install something".
      ...(!e.krea && !kreaDiskGap && kreaRestartPending
        ? { pending: true, note: 'restart ComfyUI to load its nodes' }
        : !e.krea && !kreaDiskGap && comfyOff ? { pending: true, note: NOTE } : {}) },
    // 📷 Same counting rule as Krea, for the same reason: the Gallery ships
    // this verb to every install, so a machine without the weights must read
    // "not ready, here is the install" — never a shorter list that certifies
    // completeness by omission. `camera_ready` is asset-only (no node pack, no
    // per-run process), so unlike the two engines above there is no restart
    // state — just installed or not, plus the shared "ComfyUI is off" note.
    { label: '📷 Camera angles (local)', ok: !!cu.camera_ready,
      topic: 'setup-camera-install', waitingTopic: WAITING,
      ...(!cu.camera_ready && !(Array.isArray(cu.camera_missing) && cu.camera_missing.length)
        && comfyOff ? { pending: true, note: NOTE } : {}) },
    { label: 'Captioning', ok: !!(cap.joycaption || cap.ollama), topic: 'setup-ollama' },
    { label: 'Auto-framing & head-crop', ok: !!(o.reachable && o.vision_model_ready),
      topic: 'setup-ollama' },
    { label: 'Face-similarity scoring', ok: !!c.face_scoring, topic: 'setup-quality' },
    { label: 'Person masks', ok: !!c.masks, topic: 'setup-quality' },
    { label: 'Watermark inpainting', ok: !!c.watermark_inpaint, topic: 'setup-quality' },
    // Counted for the same reason Krea is (see above): the final screen used to
    // certify "12 of 12 ready" on a machine whose video lane could not open one
    // file. A capability that is absent must be visible and counted, never
    // removed from the denominator.
    { label: 'Video bank — reading files', ok: !!c.video_decode, topic: 'setup-quality' },
    { label: 'Video bank — shot detection', ok: !!c.video_detect, topic: 'setup-quality' },
    // The THIRD video piece, and the one that was still silently missing from
    // this list. capabilities.probe_video() reports decode / detect / encode
    // apart on purpose ("a single boolean would be a lie here"), and the encoder
    // fails on its own for a documented reason: imageio-ffmpeg answers with a
    // path whether or not its binary download ever finished, so `av` can import
    // (decode ✓) on a machine where no ffmpeg exists. That machine could scan,
    // detect and triage — and certified "N of N ready" while it could not cut or
    // export one clip. Same install action as decoding (`video` = PyAV +
    // imageio-ffmpeg), so the fix is one ↻ Reinstall on the quality step, but it
    // is a separate row because a green "reading files" is not the answer to it.
    { label: 'Video bank — clip encoding', ok: !!c.video_encode, topic: 'setup-quality' },
    // Four more that were installable (INSTALL_ACTIONS: bank_scoring, bank_siglip2,
    // watermark_detect, scrape_extras; capabilities.py: bank_scoring, bank_siglip2,
    // watermark_detect, scrape_deps) and had a working Setup card, yet never had a
    // row here — the exact defect the comments above name, just for four different
    // engines. A machine missing all four still certified "14 of 14 ready".
    { label: 'Bank scoring (aesthetic · NSFW · style)', ok: !!c.bank_scoring,
      topic: 'setup-quality' },
    { label: 'SigLIP2 Bank semantics (optional)', ok: !!c.bank_siglip2,
      topic: 'setup-quality' },
    { label: 'Watermark detector (optional)', ok: !!c.watermark_detect,
      topic: 'setup-quality' },
    { label: 'Scraping extras (optional)', ok: !!c.scrape_deps, topic: 'setup-quality' },
    { label: 'LoRA training', ok: !!c.training_visible, topic: 'setup-training' },
    { label: 'Test Studio', ok: !!c.studio_visible,
      topic: 'setup-comfyui', waitingTopic: WAITING,
      ...(!c.studio_visible && comfyOff ? { pending: true, note: NOTE } : {}) },
  ]
}

// Human name of the screen a capability route lands on — derived, never typed
// twice: the Settings rail owns its section titles, the wizard owns its own.
function destinationName(route) {
  const id = (route.match(/^\/settings\/([a-z0-9-]+)/) || [])[1]
  if (id) {
    const s = SETTINGS_SECTIONS.find((x) => x.id === id)
    return s ? s.title : null
  }
  return route.startsWith('/setup') ? 'Setup wizard' : null
}

/* Resolve ONE capability row to the door that turns it on:
     { topic, href, where, announce }
   - href     in-app path, focus hint appended so the field is scrolled to and
              ringed on arrival (SettingsPage's ?focus= deep-link);
   - where    the screen's human name, for the visible/accessible wording;
   - announce the full accessible label — state FIRST (the sr-only "(ready)" /
              "(not available)" this replaces must not be lost), then where the
              row leads, because a link whose name hides its destination is a
              trap for anyone not looking at the grid.
   `getTopic` is injectable so the contract test can drive it explicitly. */
export function capabilityDestination(entry, getTopic = getHelpTopic) {
  if (!entry) return null
  const id = (entry.pending && entry.waitingTopic) ? entry.waitingTopic : entry.topic
  const t = id ? getTopic(id) : null
  if (!t || !t.app || !t.app.route) return null
  const { route, focus } = t.app
  const href = focus ? `${route}${route.includes('?') ? '&' : '?'}focus=${focus}` : route
  const where = destinationName(route)
  if (!where) return null
  // Ready rows still lead somewhere — that screen is where the capability is
  // managed (re-test a key, reinstall a helper), so "manage in" not "fix in".
  const state = entry.pending ? (entry.note || 'waiting') : (entry.ok ? 'ready' : 'not available')
  const verb = entry.pending || entry.ok ? 'manage in' : 'configure in'
  return { topic: id, href, where, announce: `${entry.label} — ${state}, ${verb} ${where}` }
}

export function recommendedMet(caps) {
  const e = (caps && caps.engines) || {}
  return !!(e.nanobanana || e.chatgpt || e.openrouter || e.klein)
}

// --- "Install everything" plan -------------------------------------------------
// Mirror of setup_installer.install_all_plan (backend — the AUTHORITY that the
// POST /api/setup/install-all recomputes and queues). Kept here so the Setup page can
// show the plan and an accurate "X / N" reactively from caps, without a round-trip, the
// same way deriveSetupSteps mirrors the backend gates. Both MUST stay in step: the set
// is the MISSING components the app can install ITSELF right now — the ML extras, the
// Ollama vision model when Ollama is already up, and the Klein weights when a valid
// ComfyUI folder is set. It never installs ComfyUI/Ollama themselves or pastes API keys
// (those are external tools / credentials), so those stay on the step-by-step path.

/** The install actions a component needs: absent from disk, OR present but not
 *  loadable. Mirrors setup_installer._needs_install — the backend authority, which
 *  now also REPLACES a blocking-invalid file instead of returning "already
 *  present" and doing nothing (the trap that made "download it again" a no-op). */
function brokenOrMissing(missing, invalid) {
  const out = Array.isArray(missing) ? [...missing] : []
  blockingInvalid(invalid).forEach((i) => { if (!out.includes(i.asset)) out.push(i.asset) })
  return out
}

// Setup-installer action -> the short human label shown in the Install-everything list.
export const INSTALL_ALL_ACTION_LABELS = {
  face_scoring: 'Face-similarity scoring',
  masks: 'Person masks',
  watermark_inpaint: 'Watermark inpainting',
  watermark_detect: 'Watermark detector',
  video: 'Video decoding (Video bank)',
  shot_detect: 'Shot detection (Video bank)',
  video_text: 'Burned-in text (Video bank)',
  ollama_model: 'Vision model (captioning)',
  klein_model: 'Klein model (local generation)',
  klein_text_encoder: 'Klein text encoder',
  klein_vae: 'Klein VAE',
  klein_lora: 'Klein consistency LoRA',
  klein_enhancement_lora: 'Klein enhancement LoRA (✨ improve detail)',
  krea_nodes: 'Krea 2 Edit node pack',
  krea_model: 'Krea 2 base model (Turbo)',
  krea_text_encoder: 'Krea 2 text encoder',
  krea_vae: 'Krea 2 VAE',
  krea_identity_lora: 'Krea 2 Identity Edit LoRA',
  seedvr2_model: 'SeedVR2 model (3B FP8)',
  seedvr2_vae: 'SeedVR2 VAE',
  lanpaint_nodes: 'LanPaint sampler (masked Repair)',
  // 📷 Camera angles. The lane's VAE has no row of its own on purpose: it is
  // the Krea 2 VAE (same file, same destination), listed once above.
  camera_model: 'Camera angles model (Qwen-Image-Edit 2511)',
  camera_lora: 'Camera angles LoRA (96 positions)',
  camera_speed_lora: 'Camera angles speed LoRA (4-step)',
  camera_text_encoder: 'Camera angles text encoder (Qwen 2.5-VL)',
}

// The Krea 2 Edit engine, installable in ONE click but deliberately NOT part of
// "Install everything": a second local engine is ~20 GB, and downloading it for
// someone who never picked it would be hostile. Mirrors the backend's
// setup_installer._INSTALL_GROUPS['krea'] — node pack first (a ~1 MB clone, so
// the only thing left to wait for is bytes), then the four weights.
export const KREA_INSTALL_ORDER = [
  'krea_nodes', 'krea_model', 'krea_text_encoder', 'krea_vae', 'krea_identity_lora',
]

/** What the "Install Krea 2 Edit" button would queue for these capabilities —
 *  the missing pieces only, so a user who already placed some files by hand sees
 *  a shorter list instead of a re-download. Mirror of
 *  setup_installer.install_group_plan('krea', caps), which stays the authority.
 *
 *  The node pack is queued only when it is neither loaded NOR on disk: on disk
 *  but not loaded is a ComfyUI RESTART, not an install. */
export function kreaInstallPlan(caps) {
  const cu = (caps || {}).comfyui || {}
  if (!cu.dir_valid) return []
  // A file on disk that cannot be LOADED needs the same install as one that is
  // absent — otherwise the one-click button silently plans nothing and the user is
  // left with a green screen and a dark engine.
  const missing = brokenOrMissing(cu.krea_missing, cu.krea_invalid)
  // On disk -> nothing to install (missing nodes then mean a RESTART). Reported
  // missing -> install. Reported present -> already there under another folder
  // name, never clone a duplicate. No answer at all (ComfyUI stopped: the node
  // probe fails OPEN) -> install it, rather than silently dropping it.
  const nodesMissing = Array.isArray(cu.krea_nodes_missing) ? cu.krea_nodes_missing : []
  const needsPack = cu.krea_nodes_installed ? false
    : (nodesMissing.length ? true : !cu.reachable)
  return KREA_INSTALL_ORDER.filter(
    (a) => (a === 'krea_nodes' ? needsPack : missing.includes(a)))
}

/** The one thing an install cannot do for the user. ComfyUI registers custom
 *  nodes at STARTUP only, so a pack that is on disk but absent from /object_info
 *  means "restart ComfyUI" — never "install it", which is what the app said
 *  before it could install the pack itself. */
export function kreaNeedsComfyuiRestart(caps) {
  const cu = (caps || {}).comfyui || {}
  return !!(cu.krea_nodes_installed
    && Array.isArray(cu.krea_nodes_missing) && cu.krea_nodes_missing.length)
}

/** Same restart rule for the LanPaint pack (masked Repair's sampler): on disk
 *  but absent from /object_info means "restart ComfyUI", never "install it". */
function lanpaintNeedsComfyuiRestart(caps) {
  const cu = (caps || {}).comfyui || {}
  return !!(cu.lanpaint_nodes_installed
    && Array.isArray(cu.lanpaint_nodes_missing) && cu.lanpaint_nodes_missing.length)
}

/** What the "Install SeedVR2" button would queue — the missing weights only.
 *  Mirror of setup_installer.install_group_plan('seedvr2', caps), which stays the
 *  authority.
 *
 *  There is NO node-pack action here, and that is the difference from Krea: this
 *  pack declares thirteen pip dependencies that belong in ComfyUI's own
 *  interpreter, which the app does not own and must never pip into. Cloning it
 *  alone would land a pack that fails to import — so the pack is explained, and
 *  only the weights are installed. */
const SEEDVR2_INSTALL_ORDER = ['seedvr2_model', 'seedvr2_vae']

export function seedvr2InstallPlan(caps) {
  const cu = (caps || {}).comfyui || {}
  if (!cu.dir_valid) return []
  const missing = brokenOrMissing(cu.seedvr2_missing, cu.seedvr2_invalid)
  return SEEDVR2_INSTALL_ORDER.filter((a) => missing.includes(a))
}

// 📷 Camera angles — weights only, like SeedVR2 (the graph is stock ComfyUI
// nodes, so there is no pack to clone and no restart state). Mirrors the
// backend's setup_installer._INSTALL_GROUPS['camera'], which stays the
// authority. `krea_vae` is a member on purpose: the lane runs on the Krea 2
// VAE, and camera_missing reports that file under the action that installs it —
// one file, one button, whichever engine asked first.
export const CAMERA_INSTALL_ORDER = [
  'camera_model', 'camera_lora', 'camera_speed_lora', 'camera_text_encoder',
  'krea_vae',
]

export function cameraInstallPlan(caps) {
  const cu = (caps || {}).comfyui || {}
  if (!cu.dir_valid) return []
  const missing = brokenOrMissing(cu.camera_missing, cu.camera_invalid)
  return CAMERA_INSTALL_ORDER.filter((a) => missing.includes(a))
}

/** The one thing an install cannot do: ComfyUI registers custom nodes at STARTUP
 *  only, so a pack on disk but absent from /object_info means "restart ComfyUI".
 *  Same rule as Krea's — here it also covers the case where the pack's Python
 *  dependencies failed to install, which looks identical from outside. */
export function seedvr2NeedsComfyuiRestart(caps) {
  const cu = (caps || {}).comfyui || {}
  return !!(cu.seedvr2_nodes_installed
    && Array.isArray(cu.seedvr2_nodes_missing) && cu.seedvr2_nodes_missing.length)
}

// Grouped by capability area (ML extras → Klein weights). The backend
// serializes pip and parallelizes downloads regardless of fire order, so this order
// only drives the progress list; it must match the backend's _INSTALL_ALL_ORDER.
export const INSTALL_ALL_ORDER = [
  'face_scoring', 'masks', 'watermark_inpaint',
  'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora',
]

export function installAllPlan(caps) {
  const c = caps || {}
  // face_scoring/masks install into the app's OWN Python, so they need it inside the ML
  // wheel range; on a newer interpreter they'd only source-build and fail. Absent python
  // info => assume supported (older payloads). watermark_inpaint builds its own venv, so
  // it stays runnable regardless.
  const mlOk = !(c.python && c.python.ml_supported === false)
  const cu = c.comfyui || {}
  // Broken counts as missing: an interrupted download leaves a file that every
  // presence check calls installed and no loader can open.
  const kleinMissing = brokenOrMissing(cu.klein_missing, cu.klein_invalid)
  const needed = (a) => {
    if (a === 'face_scoring' || a === 'masks') return mlOk && !c[a]
    if (a === 'watermark_inpaint') return !c.watermark_inpaint
    // klein_* — only into a validated ComfyUI tree.
    return !!cu.dir_valid && kleinMissing.includes(a)
  }
  return INSTALL_ALL_ORDER.filter(needed)
}

// The FULL one-by-one install menu (Setup "install" step). Unlike installAllPlan — which
// lists only what's MISSING and satisfiable now, i.e. what the "Install everything" shortcut
// queues — this lists EVERY app-installable component with its live state, so the menu stays
// visible and each item can be (re)installed on its own even once green (repairing a broken
// venv is the whole point of the reinstall button). Per item:
//   present   — the capability is already in place (drives the ✓ Installed / ✗ badge)
//   available — can be (re)installed from HERE right now (its precondition is met): ML extras
//               need the app's Python in the wheel range OR an already-present env to repair;
//               the vision model needs Ollama reachable + a model name; Klein weights need a
//               validated ComfyUI tree. Unavailable items render their `hint` instead of a
//               button, pointing back at the config step that unblocks them.
export function installCatalog(caps) {
  const c = caps || {}
  const mlOk = !(c.python && c.python.ml_supported === false)
  const mlRange = (c.python && c.python.ml_range) || '3.10–3.12'
  const mlHint = `Needs Python ${mlRange} — install it into a separate 3.10–3.12 env and set its path in Settings.`
  const o = c.ollama || {}
  const modelName = (o.vision_model || '').trim()
  const cu = c.comfyui || {}
  const dirValid = !!cu.dir_valid
  const kleinMissing = Array.isArray(cu.klein_missing) ? cu.klein_missing : []
  const kleinHint = 'Point the app at a valid ComfyUI folder first (the ComfyUI step).'
  const kreaMissing = Array.isArray(cu.krea_missing) ? cu.krea_missing : []
  const cameraMissing = Array.isArray(cu.camera_missing) ? cu.camera_missing : []
  // action -> the blocking integrity verdict for a file that IS on disk. This row
  // used to read "✓ Installed" purely because the file existed — which is how a
  // truncated 9.5 GB UNET certified itself on the very screen the user opened to
  // check (zigzag4794, Discord). Presence is not readability, and the validator
  // that tells them apart already ran; this screen simply never asked it.
  // `gates` = does this unreadable file actually STOP its engine? Klein's required
  // trio does; Klein's recommended consistency LoRA does NOT — the backend never
  // counted it (klein_engine_ready reads KLEIN_REQUIRED only), so generation keeps
  // working. Every Krea asset is required (KREA_REQUIRED = all four). Without this
  // split the LoRA row wore the exact red badge of a dead UNET, so the install
  // screen read as blocked while nothing was blocked: a badge that overstates the
  // damage is the same class of lie as one that hides it.
  const brokenBy = {}
  blockingInvalid(cu.klein_invalid).forEach((i) => {
    brokenBy[i.asset] = { ...i, gates: kleinAssetBlocks(i.asset) }
  })
  blockingInvalid(cu.krea_invalid).forEach((i) => { brokenBy[i.asset] = { ...i, gates: true } })
  // Present = the folder is there, OR a REACHABLE ComfyUI reports the nodes (a
  // pack installed under another folder name). An unreachable ComfyUI proves
  // nothing — its node probe fails open — so it must not read as installed.
  const kreaNodesPresent = !!cu.krea_nodes_installed
    || !!(cu.reachable && !(Array.isArray(cu.krea_nodes_missing) && cu.krea_nodes_missing.length))
  const kreaRestart = kreaNeedsComfyuiRestart(c)
  const lanpaintNodesPresent = !!cu.lanpaint_nodes_installed
    || !!(cu.reachable
      && !(Array.isArray(cu.lanpaint_nodes_missing) && cu.lanpaint_nodes_missing.length))
  const lanpaintRestart = lanpaintNeedsComfyuiRestart(c)
  const item = (action, present, available, hint) => {
    const bad = brokenBy[action]
    // A THIRD state, like the Krea node pack's: neither ✓ (it cannot load) nor a
    // bare ✗ (the file is right there, and "Not installed" invites the user to
    // re-run a download that used to no-op on an existing file). It names the
    // fault and the action — and `present: false` is what puts it back into the
    // install plans so the button has something to do.
    if (bad) {
      // Two severities, because there are two different situations. A REQUIRED
      // weight that cannot load means the engine is down (red, "fix this"). An
      // optional one means a feature is degraded and everything else still runs
      // (amber, "worth fixing") — it stays fully VISIBLE and re-installable, it
      // just stops claiming the engine is broken. `present: false` in both cases,
      // which is what keeps the file in the install plans so the button has
      // something to do.
      const gates = bad.gates !== false
      return {
        action, label: INSTALL_ALL_ACTION_LABELS[action] || action,
        present: false, available: !!available, hint: available ? '' : hint,
        state: gates ? 'broken' : 'broken_optional',
        stateLabel: gates ? '⚠ On disk, unreadable' : '⚠ On disk, unreadable — optional',
        blocking: gates,
        brokenReason: `${bad.filename}: ${integrityCause(bad.verdict)}. `
          + (gates ? 'Reinstall replaces it.'
            : 'Generation still works without it — reinstall replaces it.'),
      }
    }
    return {
      action, label: INSTALL_ALL_ACTION_LABELS[action] || action,
      present: !!present, available: !!available, hint: available ? '' : hint,
    }
  }
  const mlItem = (action) => {
    const present = mlOk ? !!c[action] : !!c[action]
    // Install fresh only when the app's Python supports the wheels; ALWAYS allow a
    // repair of one that's already present (its install targets whatever env it lives in).
    return item(action, present, mlOk || present, mlHint)
  }
  return [
    mlItem('face_scoring'),
    mlItem('masks'),
    item('watermark_inpaint', c.watermark_inpaint, true, ''),   // auto-provisions its own venv
    // The video extras were installable through the API and NOWHERE on this
    // screen — the banner in the Video bank said "Install … from Setup" and this
    // menu had no such row (found live, the day after the lane shipped).
    // `video` (PyAV) goes into the app's own Python — no torch, always available;
    // `shot_detect` rides the scoring environment, exactly like the watermark
    // detector, and the runner refuses the app venv itself.
    // `video` installs BOTH halves (PyAV + imageio-ffmpeg), so it is "present"
    // only when both probes are green. Keying it on decoding alone badged the row
    // ✓ Installed on a machine whose ffmpeg download never finished, and left it
    // out of the install plans — the one button that would have fixed it.
    item('video', c.video_decode && c.video_encode, true, ''),
    item('shot_detect', c.video_detect, true, ''),
    // The safe-zone pass's OCR half. Into the app's own Python like `video`:
    // it is CPU onnxruntime and no torch, so unlike `shot_detect` it has no
    // reason to borrow the scoring environment. Listed here even though the
    // pass runs without it — a user who reads "bands only" on the button needs
    // a row to click, and an extra with no row is the dead end this menu exists
    // to close.
    item('video_text', c.video_text, true, ''),
    item('ollama_model', o.vision_model_ready, o.reachable && modelName,
      !o.reachable ? 'Start Ollama first (the Captioning step).'
        : !modelName ? 'Set a vision model name first (the Captioning step).' : ''),
    // klein_enhancement_lora included: it was installable through the improve
    // 409 for weeks and offered by NO surface — the first thing the setup
    // coverage contract caught on its first run. Optional (the improve pass
    // degrades without it), which is exactly why a silent gap hid so long.
    ...['klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora',
      'klein_enhancement_lora'].map(
      (a) => item(a, dirValid && !kleinMissing.includes(a), dirValid, kleinHint)),
    // Krea 2 Edit. Same per-component repair path as everything else — the group
    // button above installs the whole engine, this row is for fixing ONE piece.
    // These rows did not exist at all before: Setup listed four Klein weights and
    // not one Krea file, so a whole engine was invisible on the screen where the
    // user decides they are done.
    {
      ...item('krea_nodes', dirValid && kreaNodesPresent && !kreaRestart, dirValid, kleinHint),
      // The one component with a THIRD state. It is on disk but ComfyUI has not
      // loaded it, so neither badge is true: "✓ Installed" would recreate the very
      // lie this wave fixes (the app certifying something it cannot see), and
      // "✗ Not installed" would invite a pointless re-install of a folder the user
      // just watched appear. The row says what to DO instead.
      ...(kreaRestart ? { state: 'restart', stateLabel: '⟳ Restart ComfyUI' } : {}),
    },
    ...['krea_model', 'krea_text_encoder', 'krea_vae', 'krea_identity_lora'].map(
      (a) => item(a, dirValid && !kreaMissing.includes(a), dirValid, kleinHint)),
    // 📷 Camera angles — the Gallery's re-shoot lane. Four rows, not five: the
    // Qwen VAE is the krea_vae row above (one file, one button). These rows
    // exist for the same reason the Krea ones do — the weights were installable
    // through the 409 and NOWHERE on this screen, which is the exact "engine
    // invisible where the user decides they are done" gap the menu closes.
    ...['camera_model', 'camera_lora', 'camera_speed_lora', 'camera_text_encoder'].map(
      (a) => item(a, dirValid && !cameraMissing.includes(a), dirValid, kleinHint)),
    // LanPaint — the sampler the masked ✦ Repair lane runs on (a ~1 MB clone,
    // zero pip dependencies). Same present/restart logic as the Krea pack: a
    // reachable ComfyUI that exposes the node counts as present whatever the
    // folder is called, and on-disk-but-not-loaded is a restart, not an install.
    {
      ...item('lanpaint_nodes', dirValid && lanpaintNodesPresent && !lanpaintRestart,
        dirValid, kleinHint),
      ...(lanpaintRestart ? { state: 'restart', stateLabel: '⟳ Restart ComfyUI' } : {}),
    },
  ]
}
