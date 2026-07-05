import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import CopyCommand from './CopyCommand'

const POLL_MS = 1200

export default function InstallRunner({ action, buttonLabel, manualCommand, onDone }) {
  const toast = useToast()
  const [state, setState] = useState('idle')  // idle|running|success|error
  const [log, setLog] = useState([])
  const [returncode, setReturncode] = useState(null)
  const timer = useRef(null)
  const mountedRef = useRef(true)

  useEffect(() => () => { mountedRef.current = false; clearTimeout(timer.current) }, [])

  const poll = async () => {
    try {
      const s = await apiFetch(`/api/setup/install/${action}/status`)
      if (!mountedRef.current) return
      setState(s.state); setLog(s.log || []); setReturncode(s.returncode)
      if (s.state === 'running') {
        timer.current = setTimeout(poll, POLL_MS)
      } else if (s.state === 'success') {
        toast.success('Installed.'); onDone?.()
      } else if (s.state === 'error') {
        toast.error('Install failed — see the log or run the command manually.')
      }
    } catch {
      if (mountedRef.current) timer.current = setTimeout(poll, POLL_MS)   // transient poll error — retry
    }
  }

  const start = async () => {
    setLog([]); setReturncode(null); setState('running')
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
      <CopyCommand command={manualCommand} />
    </div>
  )
}
