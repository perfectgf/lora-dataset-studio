import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'

/* ⏬ Download an LM Studio model from inside LDS — the missing half of the
 * Ollama pull, asked for in exactly those words.
 *
 * ONE component, mounted on BOTH surfaces that own the provider (the Settings
 * card and the Setup step), so the two can never drift apart in behaviour or
 * wording. It talks to the provider-routed /api/local-llm/pull, whose answer
 * keeps the Ollama pull shape ({state, model, progress, error}).
 *
 * Two properties come straight from what was measured on the server:
 *  - The job runs INSIDE LM Studio, so navigating away, reloading the page or
 *    restarting LDS loses nothing. On mount this re-attaches to whatever is
 *    already running and simply resumes showing it.
 *  - `progress` is honest and sometimes absent (bytes-on-disk over the job's
 *    total; LM Studio can be configured to store models elsewhere) — a missing
 *    number renders as "downloading…", never as a fake 0%.
 */
export default function LmStudioDownload({ refreshCaps, toast }) {
  const [name, setName] = useState('')
  const [job, setJob] = useState(null)      // last poll payload, or null
  const [busy, setBusy] = useState(false)   // a POST in flight
  const timerRef = useRef(null)
  const aliveRef = useRef(true)

  const stop = () => { clearTimeout(timerRef.current); timerRef.current = null }

  const poll = async () => {
    let s = null
    try { s = await apiFetch('/api/local-llm/pull', { background: true }) } catch { /* keep the last state */ }
    if (!aliveRef.current) return
    if (s) setJob(s)
    if (s && s.state === 'running') {
      timerRef.current = setTimeout(poll, 1500)
    } else if (s && s.state === 'done') {
      toast?.success(`Model downloaded — ${s.model}.`)
      await refreshCaps?.(true)
    }
  }

  useEffect(() => {
    aliveRef.current = true
    // Re-attach: a download started on the other surface (or before a reload)
    // is still this install's one download, and it should be visible here.
    apiFetch('/api/local-llm/pull', { background: true })
      .then((s) => {
        if (!aliveRef.current || !s || s.state === 'idle') return
        setJob(s)
        if (s.state === 'running') timerRef.current = setTimeout(poll, 1500)
      })
      .catch(() => {})
    return () => { aliveRef.current = false; stop() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = async () => {
    const model = name.trim()
    if (!model || busy) return
    setBusy(true)
    try {
      const r = await postJson('/api/local-llm/pull', { model })
      setJob(r)
      if (r.state === 'running') { stop(); timerRef.current = setTimeout(poll, 1500) }
      else if (r.state === 'done') { toast?.success(`Already downloaded — ${r.model}.`); await refreshCaps?.(true) }
      else if (r.error) toast?.error(r.error)
    } catch (e) {
      toast?.error(e.message || 'The download could not start.')
    } finally { setBusy(false) }
  }

  const running = job?.state === 'running'
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text" value={name} onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); start() } }}
          placeholder="qwen/qwen3-vl-4b — or a huggingface.co model URL"
          aria-label="LM Studio model to download"
          disabled={running}
          className="w-full min-w-[12rem] flex-1 rounded-md border border-border bg-app/60 px-2 py-1.5 text-xs text-content"
        />
        <button type="button" onClick={start} disabled={busy || running || !name.trim()}
          className="min-h-10 rounded-md border border-border px-3 py-1.5 text-xs font-semibold text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50 lg:min-h-0">
          {running ? 'Downloading…' : '⏬ Download'}
        </button>
      </div>
      {running && (
        <p role="status" aria-live="polite" className="text-xs text-content-muted">
          Downloading <span className="font-mono">{job.model}</span>
          {Number.isFinite(job.progress) ? ` — ${job.progress}%` : '…'} The download runs
          inside LM Studio, so leaving this page does not stop it.
        </p>
      )}
      {job?.state === 'error' && job.error && (
        <p className="text-xs text-rose-300">{job.error}</p>
      )}
      {job?.state === 'done' && (
        <p className="text-xs text-emerald-400">
          ✓ <span className="font-mono">{job.model}</span> is downloaded — it loads by
          itself the first time a pass needs it.
        </p>
      )}
    </div>
  )
}
