import { useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { useToast } from './Toast'
import { formatDiagnostic } from './diagnosticFormat'
import { copyWhenReady } from './clipboardWhenReady'

export { formatDiagnostic }

export default function DiagnosticReport() {
  const toast = useToast()
  const [busy, setBusy] = useState(false)
  // The report, kept on screen when the browser refused the copy, to copy by hand.
  const [refused, setRefused] = useState('')
  const copy = async () => {
    setBusy(true)
    setRefused('')
    const report = apiFetch('/api/diagnostic').then(formatDiagnostic)
    // Before any await: Firefox only lets the click that asked write the clipboard.
    const copied = copyWhenReady(report)
    copied.catch(() => {})
    try {
      let text
      try {
        text = await report
      } catch (err) {
        toast.error(`Could not build the report: ${err.message}`)
        return
      }
      try {
        await copied
        toast.success('Diagnostic report copied — paste it into your bug report.')
      } catch {
        setRefused(text)
        toast.error('Your browser blocked the copy. The report is shown below: copy it from there.')
      }
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
        className="mt-3 rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50">
        {busy ? 'Building…' : '📋 Copy diagnostic report'}
      </button>
      {refused && (
        <textarea readOnly value={refused} rows={8} aria-label="Diagnostic report"
          onFocus={(e) => e.target.select()}
          className="mt-3 w-full resize-y rounded-md border border-border bg-app/60 p-2 font-mono text-[0.6875rem] text-content" />
      )}
    </div>
  )
}
