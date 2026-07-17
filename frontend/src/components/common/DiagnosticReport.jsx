import { useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { useToast } from './Toast'
import { formatDiagnostic } from './diagnosticFormat'

export { formatDiagnostic }

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
        Copies version, environment health (Python/Pillow/disk), per-engine status with the
        exact missing Klein assets, live ComfyUI GPU/VRAM/queue, the last generation failures
        and the last error tracebacks — no API keys, no folder paths (your home dir is redacted
        to ~). The log/error lines can still mention file names: skim before posting.
      </p>
      <button type="button" onClick={copy} disabled={busy}
        className="mt-3 rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
        {busy ? 'Building…' : '📋 Copy diagnostic report'}
      </button>
    </div>
  )
}
