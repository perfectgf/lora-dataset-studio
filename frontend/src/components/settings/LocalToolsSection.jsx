import { useState } from 'react'
import { Card, TextField, TestResult, TestButton, SecretField } from './primitives'
import { postJson } from '../../api/fetchClient'

/* HF token is for gated TRAINING bases (Krea 2 / FLUX.1 / FLUX.2 Klein) and reading
   your private custom-base repos — it lives with the ComfyUI card because that's
   where local training/generation is set up. The Klein generation download itself
   (9B KV) is public and needs no token. */
const HF_SECRET = {
  key: 'HF_TOKEN', label: 'Hugging Face token', testTarget: null,
  help: 'Needed for gated training bases (Krea 2, FLUX.1, FLUX.2 Klein) and to read your private custom-base cloud repos — accept each model license, then read a token from hf.co/settings/tokens. Local Klein generation (9B KV) downloads without a token.',
}

/* Ollama's three live states, from capabilities (installed + reachable):
     not installed   → install hint (the app can't start what isn't there)
     installed, down → "Installed but not running" + ▶ Start Ollama (starts the
                       detached server, polls readiness, then force-re-probes so
                       the card flips to green with no app restart)
     running         → confirmation, plus whether the vision model is pulled.
   Detecting the install independently of the server running is the whole point:
   an installed-but-stopped Ollama used to read as simply "unreachable". */
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
          className="inline-flex items-center gap-1.5 rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
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

export default function LocalToolsSection(props) {
  const { config, setField, testResults, recordTestResult, saveConfigSection, caps, refreshCaps, toast } = props
  return (
    <div className="space-y-6">
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
          help="Used to derive the output/input/models/loras folders unless overridden."
        />
        <SecretField field={HF_SECRET} {...props} />
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
      </Card>

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
    </div>
  )
}
