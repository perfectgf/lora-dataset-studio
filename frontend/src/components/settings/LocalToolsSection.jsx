import { useEffect, useState } from 'react'
import { INPUT_CLASS, Card, TextField, TestResult, TestButton, SecretField } from './primitives'
import { SettingsGroup, SettingsGroupsToc, useSettingsGroupProps } from './SettingsGroupsView'
import { LOCAL_TOOLS_GROUPS } from './settingsGroups'
import { postJson, apiFetch } from '../../api/fetchClient'
import {
  COMFY_FOLDER_FIELDS, comfyFolderField, folderPlaceholder, folderEffective,
  folderEffectiveNote, folderWarning, detectedSuggestion, foldersQuery, hasAnyOverride,
} from './comfyFolders'
import ResetToDefault from './ResetToDefault'
import { defaultValueAt } from './settingDefaults.js'

/* HF token is for gated TRAINING bases (Krea 2 / FLUX.1 / FLUX.2 Klein) and reading
   your private custom-base repos — it lives with the ComfyUI card because that's
   where local training/generation is set up. The Klein generation download itself
   (9B KV) is public and needs no token. */
/* LM Studio can be configured to require a bearer token. It is a SECRET, so it
   lives in the secret store like every other credential here — never in
   config.json, where it would come back out of /api/settings in clear. Most local
   setups never need it, which is why it sits at the bottom of the card. */
const LMSTUDIO_SECRET = {
  key: 'LMSTUDIO_API_KEY', label: 'LM Studio API key (optional)', testTarget: null,
  help: 'Only if you turned on authentication in LM Studio. Left empty — the usual '
    + 'case for a local server — no Authorization header is sent at all.',
}

const HF_SECRET = {
  key: 'HF_TOKEN', label: 'Hugging Face token', testTarget: null,
  help: (
    <>
      Needed for gated training bases (Krea 2, FLUX.1, FLUX.2 Klein) and to read your private custom-base cloud repos — accept each model license, then create a token with the{' '}
      <strong className="font-semibold text-content">read</strong>
      {' role. Local Klein generation (9B KV) downloads without a token.'}
    </>
  ),
  guide: (
    <a
      href="https://huggingface.co/settings/tokens/new?tokenType=read"
      target="_blank"
      rel="noreferrer"
      className="mb-2 inline-block max-w-full text-xs font-medium text-sky-300 underline underline-offset-2 hover:text-sky-200"
    >
      Create a read token on Hugging Face ↗
    </a>
  ),
}

const HF_CLOUD_SECRET = {
  key: 'HF_CLOUD_TOKEN', label: 'Dedicated Hugging Face cloud token', testTarget: 'hf_cloud',
  help: (
    <>
      Required only for full-model Krea 2 cloud training. Recommended: create a separate fine-grained token with zero global permissions and grant{' '}
      <strong className="font-semibold text-content">repo.content.read exactly on krea/Krea-2-Raw</strong>
      {', then '}
      <strong className="font-semibold text-content">repo.content.read + repo.write on one dedicated HF user/org namespace that contains only LDS deliveries</strong>
      {'. A per-run repository does not exist yet when the token is created, so scope write access to that single dedicated namespace. A global write token is also accepted, but LDS will warn because it can modify every repository this account can write to.'}
    </>
  ),
  guide: (
    <div className="mb-2 flex flex-wrap gap-x-3 gap-y-1">
      <a
        href="https://huggingface.co/settings/tokens/new?tokenType=fineGrained"
        target="_blank"
        rel="noreferrer"
        className="inline-block max-w-full text-xs font-medium text-sky-300 underline underline-offset-2 hover:text-sky-200"
      >
        Create a fine-grained token on Hugging Face ↗
      </a>
      <a
        href="https://huggingface.co/settings/tokens/new?tokenType=write"
        target="_blank"
        rel="noreferrer"
        className="inline-block max-w-full text-xs font-medium text-sky-300 underline underline-offset-2 hover:text-sky-200"
      >
        Create a global write token on Hugging Face ↗
      </a>
    </div>
  ),
}

/* Ollama's three live states, from capabilities (installed + reachable):
     not installed   → install hint (the app can't start what isn't there)
     installed, down → "Installed but not running" + ▶ Start Ollama (starts the
                       detached server, polls readiness, then force-re-probes so
                       the card flips to green with no app restart)
     running         → confirmation, plus whether the vision model is pulled.
   Detecting the install independently of the server running is the whole point:
   an installed-but-stopped Ollama used to read as simply "unreachable". */
function LmStudioStatus({ caps, active, refreshCaps, toast }) {
  const l = (caps && caps.lmstudio) || {}
  const [starting, setStarting] = useState(false)

  /* This card used to carry a written reason for having NO Start button: "LM
     Studio has no reliable way to be launched from here". That was wrong -- its
     CLI sits at a fixed per-user path and `lms server start` is a one-shot
     command that returns in well under a second, with the loaded model surviving
     the cycle. The button exists now, and only when the CLI was actually found. */
  const start = async () => {
    setStarting(true)
    try {
      const r = await postJson('/api/local-llm/start', {})
      if (r.reachable) {
        toast?.success(r.already_running ? 'LM Studio is already running.' : 'LM Studio is running.')
        await refreshCaps?.(true)   // force re-probe → the card flips, no app restart
      } else {
        toast?.error(r.error || 'LM Studio did not start — start its server from the app.')
      }
    } catch (e) {
      toast?.error(e.message || 'Could not start LM Studio.')
    } finally {
      setStarting(false)
    }
  }
  if (!active) {
    return (
      <p className="text-xs text-content-muted">
        <span aria-hidden="true">○</span> Not the provider in use — press Test to check it
        without switching.
      </p>
    )
  }
  if (l.reachable && l.model_ready) {
    return (
      <p className="text-xs text-emerald-400">
        <span aria-hidden="true">✓</span> Running · {l.detail}
      </p>
    )
  }
  if (l.reachable) {
    return (
      <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
        <p className="text-sm text-content"><span aria-hidden="true">●</span> Running, but not ready.</p>
        <p className="mt-1 text-xs text-content-muted">{l.detail}</p>
      </div>
    )
  }
  if (!l.probed) {
    // Selected here but not yet saved, so the status snapshot still describes the
    // OTHER provider. Announcing "not answering" for a server nobody has asked yet
    // is a scary sentence about nothing — say what is actually true instead.
    return (
      <p className="text-xs text-content-muted">
        <span aria-hidden="true">○</span> Selected — save to check it, or press Test now.
      </p>
    )
  }
  if (l.installed) {
    return (
      <div className="space-y-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
        <p className="text-sm text-content">
          <span aria-hidden="true">●</span> Installed but its server is not running.
        </p>
        <p className="text-xs text-content-muted">
          A model you had loaded stays loaded across this — measured, not assumed.
        </p>
        <button
          type="button"
          onClick={start}
          disabled={starting}
          className="inline-flex items-center gap-1.5 rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-50"
        >
          {starting ? 'Starting…' : '▶ Start LM Studio'}
        </button>
      </div>
    )
  }
  return (
    <p className="text-xs text-content-muted">
      <span aria-hidden="true">✗</span> Not answering at {l.url || 'the configured URL'} — open
      LM Studio, go to <span className="font-medium text-content">Developer</span> and press
      Start Server.
    </p>
  )
}

function OllamaStatus({ caps, refreshCaps, toast }) {
  const o = (caps && caps.ollama) || {}
  const [starting, setStarting] = useState(false)

  const start = async () => {
    setStarting(true)
    try {
      const r = await postJson('/api/ollama/start', {})
      if (r.reachable) {
        toast?.success('Ollama is running.')
        await refreshCaps?.(true)   // force re-probe → state flips to green, no restart
      } else {
        toast?.error(r.error || 'Ollama did not start — check the log or start it manually.')
      }
    } catch (e) {
      toast?.error(e.message || 'Could not start Ollama.')
    } finally {
      setStarting(false)
    }
  }

  if (o.reachable) {
    return (
      <p className="text-xs text-emerald-400">
        <span aria-hidden="true">✓</span> Running{o.vision_model_ready ? ' · vision model ready' : ''}
      </p>
    )
  }
  if (o.installed) {
    return (
      <div className="space-y-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
        <p className="text-sm text-content">
          <span aria-hidden="true">●</span> Installed but not running.
        </p>
        <button
          type="button"
          onClick={start}
          disabled={starting}
          className="inline-flex items-center gap-1.5 rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-50"
        >
          {starting && (
            <span aria-hidden="true"
              className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
          )}
          {starting ? 'Starting…' : '▶ Start Ollama'}
        </button>
      </div>
    )
  }
  return (
    <p className="text-xs text-content-muted">
      <span aria-hidden="true">✗</span> Not detected on this machine.{' '}
      <a href="https://ollama.com/download" target="_blank" rel="noreferrer"
        className="text-sky-300 underline hover:text-sky-200">Download Ollama →</a>
    </p>
  )
}

/* The four ComfyUI folder overrides. They have always been honoured by the backend but
   had no field anywhere, so a ComfyUI started with --input-directory/--output-directory
   looked like it was being ignored (reported on Discord by vykas22).

   Two things make them safe to use rather than a leap of faith: an empty field shows
   the DERIVED path it actually falls back to, and a path that isn't on disk says so —
   both resolved by the backend with the same function the app itself uses, so the
   preview cannot drift from the real behaviour. Debounced so typing a path doesn't
   fire a request per keystroke. */
function ComfyFolderOverrides({ config, setField }) {
  const comfy = config.comfyui
  const [state, setState] = useState({ folders: {}, detected: {} })
  const query = foldersQuery(comfy)

  useEffect(() => {
    let alive = true
    const t = setTimeout(async () => {
      try {
        const r = await apiFetch(`/api/setup/comfyui-folders?${query}`)
        if (alive) setState((prev) => ({ folders: r.folders || {}, detected: prev.detected }))
      } catch { /* preview only — never block the form on it */ }
    }, 350)
    return () => { alive = false; clearTimeout(t) }
  }, [query])

  // Ask the running ComfyUI what it was launched with, once per mount. It answers from
  // its own command line (/system_stats), so a hit is a fact, not a layout guess; no
  // answer simply means nothing is offered and the manual fields stand alone.
  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const r = await apiFetch(`/api/setup/comfyui-folders?${foldersQuery(comfy, { detect: true })}`)
        if (alive) setState((prev) => ({ ...prev, detected: r.detected || {} }))
      } catch { /* ComfyUI down: no suggestions, nothing broken */ }
    })()
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const anyDetected = COMFY_FOLDER_FIELDS.some((f) => detectedSuggestion(f, state.detected, comfy[f.key]))
  // The four ids below are written out literally, not mapped: the help-registry
  // contract test discovers Settings ids by scanning this file for id="…", and a
  // computed id would silently escape the "every setting has a help topic" check.
  const rowProps = { comfy, setField, state }

  return (
    <details className="rounded-lg border border-border p-3" open={hasAnyOverride(comfy) || anyDetected}>
      <summary className="cursor-pointer text-sm font-medium text-content-muted">
        Advanced: ComfyUI folder overrides
      </summary>
      <div className="mt-3 space-y-4">
        <p className="text-xs text-content-muted">
          Leave these empty unless ComfyUI was started with its own folders (
          <code>--output-directory</code>, <code>--input-directory</code>,{' '}
          <code>--models-directory</code>). Each field shows the folder the app uses
          while it is empty.
        </p>
        <ComfyFolderRow {...rowProps} fieldKey="output_dir" id="comfyui-output-dir" />
        <ComfyFolderRow {...rowProps} fieldKey="input_dir" id="comfyui-input-dir" />
        <ComfyFolderRow {...rowProps} fieldKey="models_dir" id="comfyui-models-dir" />
        <ComfyFolderRow {...rowProps} fieldKey="loras_dir" id="comfyui-loras-dir" />
      </div>
    </details>
  )
}

function ComfyFolderRow({ comfy, setField, state, fieldKey, id }) {
  const field = comfyFolderField(fieldKey)
  const info = state.folders[fieldKey]
  const suggestion = detectedSuggestion(field, state.detected, comfy[fieldKey])
  return (
    <TextField
      id={id}
      label={field.label}
      value={comfy[fieldKey]}
      onChange={(v) => setField('comfyui', fieldKey, v)}
      placeholder={folderPlaceholder(field)}
      help={field.help}
      warn={folderWarning(info)}
    >
      {folderEffective(info) && (
        <p className="mt-1 break-all text-xs text-content-muted">
          Currently using <code className="text-content">{folderEffective(info)}</code>
          {folderEffectiveNote(info) && <> ({folderEffectiveNote(info)})</>}
        </p>
      )}
      {suggestion && (
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          <span className="min-w-0 break-all text-xs text-content-muted">
            ComfyUI is running with <code className="text-sky-300">{suggestion}</code>
          </span>
          <button
            type="button"
            onClick={() => setField('comfyui', fieldKey, suggestion)}
            className="shrink-0 rounded-md border border-sky-400/40 px-2 py-1 text-xs font-medium text-sky-300 hover:bg-sky-400/10"
          >
            Use this
          </button>
        </div>
      )}
    </TextField>
  )
}

export default function LocalToolsSection(props) {
  const { config, setField, testResults, recordTestResult, saveConfigSection, caps, refreshCaps, toast,
          configDefaults } = props
  // Shipped values come from the server payload, never retyped here.
  const ollamaDefault = (key) => defaultValueAt(configDefaults, 'ollama', key)
  const lmstudioDefault = (key) => defaultValueAt(configDefaults, 'lmstudio', key)
  // Total: a config.json written before this setting existed has no local_llm
  // section at all, and every such install means Ollama.
  const provider = ((config.local_llm || {}).provider) || 'ollama'
  const comfyDefault = (key) => defaultValueAt(configDefaults, 'comfyui', key)
  // Summary + collapsible groups, one per tool — same shells as Image engines
  // (SettingsGroupsView), and the ?focus= reveal opens a collapsed group on
  // its own, so every deep-link and search result keeps landing.
  const [comfyGroup, ollamaGroup, aitkGroup] = LOCAL_TOOLS_GROUPS
  const groupProps = useSettingsGroupProps('local-tools')
  return (
    <div className="space-y-4">
      <SettingsGroupsToc sectionId="local-tools" groups={LOCAL_TOOLS_GROUPS} />

      <SettingsGroup {...groupProps(comfyGroup)}>
      <Card
        title="ComfyUI"
        help="Local (Klein) generation and the Test Studio. The API URL is where a running ComfyUI answers; the install directory is scanned for checkpoints and LoRAs."
      >
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
          <TestButton target="comfyui" beforeTest={() => saveConfigSection('comfyui')}
            onResult={(r) => recordTestResult('comfyui', r)} />
        </div>
        <TextField
          id="comfyui-base-dir"
          label="ComfyUI install directory"
          value={config.comfyui.base_dir}
          onChange={(v) => setField('comfyui', 'base_dir', v)}
          placeholder="C:\ComfyUI"
          help="Used to derive the output/input/models/loras folders unless overridden below."
        />
        <ComfyFolderOverrides config={config} setField={setField} />
        <div>
          <label htmlFor="comfyui-object-info-timeout" className="block text-sm font-medium text-content">
            ComfyUI response timeout
          </label>
          <select
            id="comfyui-object-info-timeout"
            value={String(config.comfyui.object_info_timeout_s ?? comfyDefault('object_info_timeout_s'))}
            onChange={(e) => setField('comfyui', 'object_info_timeout_s', Number(e.target.value))}
            className={INPUT_CLASS}
          >
            <option value="15">15 seconds</option>
            <option value="30">30 seconds</option>
            <option value="45">45 seconds — recommended</option>
            <option value="90">90 seconds</option>
            <option value="180">3 minutes — very large install</option>
          </select>
          <p className="mt-1 text-xs text-content-muted">
            How long ComfyUI may take to list its nodes and model files. That list
            grows with every custom-node pack and every weight you install, so a
            heavily-loaded ComfyUI can need 15 seconds or more — and when the app gave
            up too early it wrongly reported ComfyUI as not running. Raise this if you
            see “ComfyUI is answering too slowly”. A ComfyUI that is genuinely stopped
            is detected in a couple of seconds either way, so a high value here costs
            you nothing. Reported and measured by j_o_e_l. (Discord).
          </p>
          <ResetToDefault label="ComfyUI response timeout" section="comfyui" field="object_info_timeout_s"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <SecretField field={HF_SECRET} {...props} />
        <div className="rounded-lg border border-sky-400/25 bg-sky-400/5 p-3">
          <SecretField field={HF_CLOUD_SECRET} {...props} />
        </div>
      </Card>

      </SettingsGroup>

      <SettingsGroup {...groupProps(ollamaGroup)}>
      <Card
        title="Which local LLM"
        help="Captioning, framing, head-crop and the prompt helpers all run on one local model server. This is which one."
      >
        <div>
          <label htmlFor="local-llm-provider" className="block text-sm font-medium text-content">
            Local LLM provider
          </label>
          <select
            id="local-llm-provider"
            value={provider}
            onChange={(e) => setField('local_llm', 'provider', e.target.value)}
            className={INPUT_CLASS}
          >
            <option value="ollama">Ollama — the default</option>
            <option value="lmstudio">LM Studio</option>
          </select>
          <p className="mt-1 text-xs text-content-muted">
            Both cards below stay editable whichever you pick, so you can set the other one up
            and press Test before switching to it. Only the selected provider is used — and only
            it is checked when the app refreshes its status, so nothing pays for a server you are
            not running.
          </p>
        </div>
      </Card>
      <Card
        title="Ollama"
        help="Lightweight local vision backend — captioning, framing auto-classify and head-crop."
      >
        <OllamaStatus caps={caps} refreshCaps={refreshCaps} toast={toast} />
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
              placeholder="huihui_ai/qwen3-vl-abliterated:8b-instruct"
            />
            <TestResult result={testResults.ollama} />
          </div>
          <TestButton target="ollama" beforeTest={() => saveConfigSection('ollama')}
            onResult={(r) => recordTestResult('ollama', r)} />
        </div>
        <div>
          <label htmlFor="ollama-vision-concurrency" className="block text-sm font-medium text-content">
            Images analysed at once
          </label>
          <select
            id="ollama-vision-concurrency"
            value={String(config.ollama.vision_concurrency ?? ollamaDefault('vision_concurrency'))}
            onChange={(e) => setField('ollama', 'vision_concurrency', Number(e.target.value))}
            className={INPUT_CLASS}
          >
            <option value="1">1 — one at a time (slowest, gentlest)</option>
            <option value="2">2</option>
            <option value="4">4 — recommended (about twice as fast)</option>
            <option value="6">6</option>
            <option value="8">8 — only if Ollama is set up for it</option>
          </select>
          <p className="mt-1 text-xs text-content-muted">
            Bank passes that read every image — watermark scan, framing, captions — send
            this many requests to Ollama at the same time. Most of each request is
            waiting, not computing, so overlapping them roughly halves a long pass.
            Raising it past 4 gains little unless your Ollama is configured for more
            parallel requests, and it makes Stop take a few seconds longer.
          </p>
          <ResetToDefault label="Images analysed at once" section="ollama" field="vision_concurrency"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="ollama-vision-keep-warm" className="block text-sm font-medium text-content">
            Keep the vision model warm
          </label>
          <select
            id="ollama-vision-keep-warm"
            value={String(config.ollama.vision_keep_warm_seconds ?? ollamaDefault('vision_keep_warm_seconds'))}
            onChange={(e) => setField('ollama', 'vision_keep_warm_seconds', Number(e.target.value))}
            className={INPUT_CLASS}
          >
            <option value="0">Off — unload after every single image</option>
            <option value="60">1 minute</option>
            <option value="120">2 minutes — recommended</option>
            <option value="300">5 minutes</option>
            <option value="600">10 minutes</option>
          </select>
          <p className="mt-1 text-xs text-content-muted">
            Loading the vision model takes about 13 seconds; describing an image once
            it's loaded takes half a second. One-off jobs — the automatic head crop on a
            reference photo, Describe in Test Studio — used to unload it straight away,
            so doing several in a row paid that load every time. The app now keeps it
            loaded for this long, but only while nothing else needs the graphics card,
            and hands the memory straight back the moment a generation or a training run
            starts. Set it to Off if your card is tight on memory.
          </p>
          <ResetToDefault label="Keep the vision model warm" section="ollama" field="vision_keep_warm_seconds"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
      </Card>

      <Card
        title={provider === 'lmstudio' ? 'LM Studio — in use' : 'LM Studio'}
        help="A local model server with a graphical app. Unlike Ollama it cannot be started from here, and it only serves a model that is already loaded."
      >
        <LmStudioStatus caps={caps} active={provider === 'lmstudio'}
          refreshCaps={refreshCaps} toast={toast} />
        <div className="flex items-end gap-3">
          <div className="flex-1 space-y-4">
            <TextField
              id="lmstudio-url"
              label="LM Studio URL"
              value={(config.lmstudio || {}).url || ''}
              onChange={(v) => setField('lmstudio', 'url', v)}
              placeholder={(caps?.lmstudio?.docker_runtime)
                ? 'http://host.docker.internal:1234' : 'http://127.0.0.1:1234'}
              help={(caps?.lmstudio?.docker_runtime)
                ? 'This app is running in a container, so 127.0.0.1 means the container itself. '
                  + 'LM Studio runs on your machine — use http://host.docker.internal:1234 and make '
                  + 'sure its server is reachable from Docker.'
                : "The server root. LM Studio's Developer tab shows it with /v1 on the end — either form is accepted."}
            />
            <TextField
              id="lmstudio-vision-model"
              label="LM Studio model"
              value={(config.lmstudio || {}).vision_model || ''}
              onChange={(v) => setField('lmstudio', 'vision_model', v)}
              placeholder="leave empty to use whatever is loaded"
              help="Left empty, the app uses whichever model LM Studio has loaded — usually what you want, since it only serves a loaded one."
            />
            <TestResult result={testResults.lmstudio} />
          </div>
          <TestButton target="lmstudio" beforeTest={() => saveConfigSection('lmstudio')}
            onResult={(r) => recordTestResult('lmstudio', r)} />
        </div>
        <div>
          <label htmlFor="lmstudio-vision-concurrency" className="block text-sm font-medium text-content">
            Images analysed at once
          </label>
          <select
            id="lmstudio-vision-concurrency"
            value={String((config.lmstudio || {}).vision_concurrency ?? lmstudioDefault('vision_concurrency'))}
            onChange={(e) => setField('lmstudio', 'vision_concurrency', Number(e.target.value))}
            className={INPUT_CLASS}
          >
            <option value="1">1 — one at a time (slowest, gentlest)</option>
            <option value="2">2</option>
            <option value="4">4 — recommended</option>
            <option value="6">6</option>
            <option value="8">8</option>
          </select>
          <p className="mt-1 text-xs text-content-muted">
            The same dial as Ollama's, kept separate because the two servers do not take the
            same load: LM Studio serves as many parallel requests as its own Parallel setting
            allows, and going wider here than that gains nothing.
          </p>
          <ResetToDefault label="Images analysed at once" section="lmstudio" field="vision_concurrency"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <div>
          <label htmlFor="lmstudio-vision-keep-warm" className="block text-sm font-medium text-content">
            Keep the vision model warm
          </label>
          <select
            id="lmstudio-vision-keep-warm"
            value={String((config.lmstudio || {}).vision_keep_warm_seconds
              ?? lmstudioDefault('vision_keep_warm_seconds'))}
            onChange={(e) => setField('lmstudio', 'vision_keep_warm_seconds', Number(e.target.value))}
            className={INPUT_CLASS}
          >
            <option value="0">Off — unload after every single image</option>
            <option value="60">1 minute</option>
            <option value="120">2 minutes — recommended</option>
            <option value="300">5 minutes</option>
            <option value="600">10 minutes</option>
          </select>
          <p className="mt-1 text-xs text-content-muted">
            Honoured differently from Ollama&rsquo;s, and worth knowing: Ollama takes a
            per-request keep-alive, while LM Studio has no expiry of its own and holds a
            loaded model until something unloads it. So this is how long the app waits
            before actively unloading &mdash; and unlike Ollama, that unload really does hand
            the memory back. A model you loaded yourself is never unloaded by this.
          </p>
          <ResetToDefault label="Keep the vision model warm" section="lmstudio" field="vision_keep_warm_seconds"
            config={config} configDefaults={configDefaults} setField={setField} />
        </div>
        <SecretField field={LMSTUDIO_SECRET} {...props} />
      </Card>
      </SettingsGroup>

      <SettingsGroup {...groupProps(aitkGroup)}>
      <Card
        title="ai-toolkit"
        help="The training engine. Point at the folder containing run.py — its venv/ or .venv/ is detected automatically."
      >
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
          <TestButton target="aitoolkit" beforeTest={() => saveConfigSection('aitoolkit')}
            onResult={(r) => recordTestResult('aitoolkit', r)} />
        </div>
        <TextField
          id="aitoolkit-python"
          label="Python interpreter (optional)"
          value={config.aitoolkit.python}
          onChange={(v) => setField('aitoolkit', 'python', v)}
          placeholder="Auto — only needed when ai-toolkit has no venv/.venv (conda, uv, system Python)"
          help="Full path to the python executable ai-toolkit should run with, e.g. C:\miniconda3\envs\aitk\python.exe."
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
      </SettingsGroup>
    </div>
  )
}
