import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import CopyCommand from './CopyCommand'

const POLL_MS = 1200
const MAX_POLL_FAILURES = 5

function fmtSize(b) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`
  return `${Math.max(0, Math.round(b / 1e3))} KB`
}

export default function InstallRunner({ action, buttonLabel, manualCommand, onDone }) {
  const toast = useToast()
  const [state, setState] = useState('idle')  // idle|running|success|error
  const [log, setLog] = useState([])
  const [returncode, setReturncode] = useState(null)
  const [progress, setProgress] = useState(null)  // {done,total,pct} for streaming downloads
  // The manual fallback command comes from the BACKEND (scoped to sys.executable,
  // absolute + quoted) so copy-pasting it hits the app's own interpreter, not a
  // stray `pip` on PATH. The prop is only a fallback until the first fetch lands.
  const [manualCmd, setManualCmd] = useState(manualCommand)
  const timer = useRef(null)
  const mountedRef = useRef(true)
  const fails = useRef(0)

  const apply = (s) => {
    setState(s.state); setLog(s.log || []); setReturncode(s.returncode)
    setProgress(s.progress || null)
    if (s.manual_command) setManualCmd(s.manual_command)
  }

  const poll = async () => {
    try {
      const s = await apiFetch(`/api/setup/install/${action}/status`)
      if (!mountedRef.current) return
      fails.current = 0
      apply(s)
      if (s.state === 'running') {
        timer.current = setTimeout(poll, POLL_MS)
      } else if (s.state === 'success') {
        toast.success('Installed.'); onDone?.()
      } else if (s.state === 'error') {
        toast.error('Install failed — see the log or run the command manually.')
      }
    } catch {
      if (!mountedRef.current) return
      fails.current += 1
      if (fails.current >= MAX_POLL_FAILURES) {
        // Stop hammering a down backend; tell the user and fall back to manual.
        setState('error')
        toast.error('Lost contact with the installer — check the server, then run the command manually.')
      } else {
        timer.current = setTimeout(poll, POLL_MS)   // transient poll error — retry
      }
    }
  }

  // Re-attach on mount to an install that may already be running or finished
  // (e.g. the user left this page mid-install and came back). Idle -> stay ready.
  useEffect(() => {
    mountedRef.current = true
    apiFetch(`/api/setup/install/${action}/status`).then((s) => {
      if (!mountedRef.current) return
      if (s.manual_command) setManualCmd(s.manual_command)   // even when idle
      if (s.state === 'idle') return
      apply(s)
      if (s.state === 'running') timer.current = setTimeout(poll, POLL_MS)
    }).catch(() => { /* not attached; leave the button idle */ })
    return () => { mountedRef.current = false; clearTimeout(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [action])

  const start = async () => {
    setLog([]); setReturncode(null); setProgress(null); setState('running'); fails.current = 0
    try {
      await postJson(`/api/setup/install/${action}`, {})
      poll()
    } catch (e) {
      setState('error')
      toast.error(e.message || 'Could not start install.')
    }
  }

  const running = state === 'running'
  return (
    <div className="space-y-2">
      <button type="button" onClick={start} disabled={running}
        className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50">
        {running ? 'Installing…' : buttonLabel}
      </button>
      {running && progress && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px] text-content-muted tabular-nums">
            <span>{progress.pct != null ? `Downloading ${progress.pct}%` : 'Downloading…'}</span>
            <span>{fmtSize(progress.done)}{progress.total ? ` / ${fmtSize(progress.total)}` : ' downloaded'}</span>
          </div>
          {progress.pct != null && (
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
              <div className="h-full rounded-full bg-gradient-primary transition-[width] duration-300"
                style={{ width: `${progress.pct}%` }} />
            </div>
          )}
        </div>
      )}
      {(log.length > 0 || running) && (
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-surface-raised p-2 text-[11px] text-content-muted">
          {log.slice(-40).join('\n') || 'starting…'}
        </pre>
      )}
      {state === 'error' && (
        <p className="text-xs text-rose-400">
          {returncode != null
            ? `Exit code ${returncode}. Run this manually instead:`
            : 'Could not start the install. Run this manually instead:'}
        </p>
      )}
      <CopyCommand command={manualCmd} />
    </div>
  )
}
