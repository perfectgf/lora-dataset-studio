import { useState } from 'react'
import { Card, TextField, TestResult, TestButton, SecretField } from './primitives'
import { postJson } from '../../api/fetchClient'
import { useI18n } from '../../i18n/I18nContext'

/* HF token is for gated TRAINING bases (Krea 2 / FLUX.1 / FLUX.2 Klein) and reading
   your private custom-base repos — it lives with the ComfyUI card because that's
   where local training/generation is set up. The Klein generation download itself
   (9B KV) is public and needs no token. */
const hfSecret = (t) => ({
  key: 'HF_TOKEN', label: t('settings.localTools.hfToken'), testTarget: null,
  help: t('settings.localTools.hfTokenHelp'),
})

/* Ollama's three live states, from capabilities (installed + reachable):
     not installed   → install hint (the app can't start what isn't there)
     installed, down → "Installed but not running" + ▶ Start Ollama (starts the
                       detached server, polls readiness, then force-re-probes so
                       the card flips to green with no app restart)
     running         → confirmation, plus whether the vision model is pulled.
   Detecting the install independently of the server running is the whole point:
   an installed-but-stopped Ollama used to read as simply "unreachable". */
function OllamaStatus({ caps, refreshCaps, toast }) {
  const { t } = useI18n()
  const o = (caps && caps.ollama) || {}
  const [starting, setStarting] = useState(false)

  const start = async () => {
    setStarting(true)
    try {
      const r = await postJson('/api/ollama/start', {})
      if (r.reachable) {
        toast?.success(t('settings.localTools.ollamaStarted'))
        await refreshCaps?.(true)   // force re-probe → state flips to green, no restart
      } else {
        toast?.error(r.error || t('settings.localTools.ollamaStartFailed'))
      }
    } catch (e) {
      toast?.error(e.message || t('settings.localTools.ollamaStartFailed'))
    } finally {
      setStarting(false)
    }
  }

  if (o.reachable) {
    return (
      <p className="text-xs text-emerald-400">
        <span aria-hidden="true">✓</span> {t('settings.localTools.running')}
        {o.vision_model_ready ? ` · ${t('settings.localTools.visionReady')}` : ''}
      </p>
    )
  }
  if (o.installed) {
    return (
      <div className="space-y-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
        <p className="text-sm text-content">
          <span aria-hidden="true">●</span> {t('settings.localTools.installedStopped')}
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
          {starting ? t('settings.localTools.starting') : t('settings.localTools.startOllama')}
        </button>
      </div>
    )
  }
  return (
    <p className="text-xs text-content-muted">
      <span aria-hidden="true">✗</span> {t('settings.localTools.notDetected')}{' '}
      <a href="https://ollama.com/download" target="_blank" rel="noreferrer"
        className="text-sky-300 underline hover:text-sky-200">{t('settings.localTools.downloadOllama')}</a>
    </p>
  )
}

export default function LocalToolsSection(props) {
  const { t } = useI18n()
  const { config, setField, testResults, recordTestResult, saveConfigSection, caps, refreshCaps, toast } = props
  return (
    <div className="space-y-6">
      <Card
        title="ComfyUI"
        help={t('settings.localTools.comfyHelp')}
      >
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <TextField
              id="comfyui-api-url"
              label={t('settings.localTools.comfyApiUrl')}
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
          label={t('settings.localTools.comfyDir')}
          value={config.comfyui.base_dir}
          onChange={(v) => setField('comfyui', 'base_dir', v)}
          placeholder="C:\ComfyUI"
          help={t('settings.localTools.comfyDirHelp')}
        />
        <SecretField field={hfSecret(t)} {...props} />
      </Card>

      <Card
        title="Ollama"
        help={t('settings.localTools.ollamaHelp')}
      >
        <OllamaStatus caps={caps} refreshCaps={refreshCaps} toast={toast} />
        <div className="flex items-end gap-3">
          <div className="flex-1 space-y-4">
            <TextField
              id="ollama-url"
              label={t('settings.localTools.ollamaUrl')}
              value={config.ollama.url}
              onChange={(v) => setField('ollama', 'url', v)}
              placeholder="http://127.0.0.1:11434"
            />
            <TextField
              id="ollama-vision-model"
              label={t('settings.localTools.ollamaModel')}
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
        help={t('settings.localTools.aitoolkitHelp')}
      >
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <TextField
              id="aitoolkit-dir"
              label={t('settings.localTools.aitoolkitDir')}
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
          label={t('settings.localTools.pythonOptional')}
          value={config.aitoolkit.python}
          onChange={(v) => setField('aitoolkit', 'python', v)}
          placeholder={t('settings.localTools.pythonPlaceholder')}
          help={t('settings.localTools.pythonHelp')}
        />

        <details className="rounded-lg border border-border p-3">
          <summary className="cursor-pointer text-sm font-medium text-content-muted">
            {t('settings.localTools.advanced')}
          </summary>
          <div className="mt-3 space-y-4">
            <TextField
              id="aitoolkit-datasets-dir"
              label={t('settings.localTools.datasetsDir')}
              value={config.aitoolkit.datasets_dir}
              onChange={(v) => setField('aitoolkit', 'datasets_dir', v)}
              placeholder={t('settings.localTools.datasetsDirPlaceholder')}
            />
            <TextField
              id="aitoolkit-output-dir"
              label={t('settings.localTools.outputDir')}
              value={config.aitoolkit.output_dir}
              onChange={(v) => setField('aitoolkit', 'output_dir', v)}
              placeholder={t('settings.localTools.outputDirPlaceholder')}
            />
            <TextField
              id="aitoolkit-hf-home"
              label={t('settings.localTools.hfCache')}
              value={config.aitoolkit.hf_home}
              onChange={(v) => setField('aitoolkit', 'hf_home', v)}
              placeholder={t('settings.localTools.hfCachePlaceholder')}
            />
          </div>
        </details>
      </Card>
    </div>
  )
}
