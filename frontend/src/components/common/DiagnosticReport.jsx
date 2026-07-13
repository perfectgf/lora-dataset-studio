import { useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { useToast } from './Toast'

/* Renders /api/diagnostic as a fenced-markdown block ready for Discord/GitHub.
   Keys/paths never appear — the backend only ships presence booleans. */
export function formatDiagnostic(d) {
  const yn = (v) => (v ? 'yes' : 'no')
  const c = d.capabilities || {}
  const e = c.engines || {}
  const cf = d.config || {}
  return [
    '```',
    `LoRA Dataset Studio diagnostic — v${d.app_version}${d.git_sha ? ` (${d.git_sha})` : ''}`,
    `OS: ${d.os} · Python ${d.python}`,
    `Engines: nanobanana=${yn(e.nanobanana)} chatgpt=${yn(e.chatgpt)} klein=${yn(e.klein)} (default: ${cf.default_engine})`,
    `ComfyUI: reachable=${yn(c.comfyui_reachable)} klein_model=${yn(c.klein_model)} · Ollama: reachable=${yn(c.ollama_reachable)} vision_model=${yn(c.vision_model_ready)}`,
    `ai-toolkit: ${yn(c.aitoolkit_valid)} · face scoring: ${yn(c.face_scoring)} · masks: ${yn(c.masks)} · cloud: ${yn(c.cloud_training)}`,
    `Captioning: ${cf.captioning_backend} · default family: ${cf.training_default_family} · LAN: ${yn(cf.lan_enabled)}`,
    `Keys set: ${Object.entries(d.secrets_present || {}).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none'}`,
    '--- last log lines ---',
    ...(d.log_tail || []).slice(-40),
    '```',
  ].join('\n')
}

export default function DiagnosticReport() {
  const toast = useToast()
  const [busy, setBusy] = useState(false)
  const copy = async () => {
    setBusy(true)
    try {
      const d = await apiFetch('/api/diagnostic')
      await navigator.clipboard.writeText(formatDiagnostic(d))
      toast.success('Diagnostic report copied — paste it into your bug report.')
    } catch (err) {
      toast.error(`Could not build the report: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-sm font-medium text-content">One-click bug report</p>
      <p className="mt-1 text-xs text-content-muted">
        Copies version, OS, capability status and the last log lines — no API keys, no folder
        paths. The log tail can still mention file names: skim it before posting.
      </p>
      <button type="button" onClick={copy} disabled={busy}
        className="mt-3 rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
        {busy ? 'Building…' : '📋 Copy diagnostic report'}
      </button>
    </div>
  )
}
