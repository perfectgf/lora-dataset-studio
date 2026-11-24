import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { fmtSize } from './fmtSize'
import ScoringPythonDialog from '../bank/ScoringPythonDialog'
import { PICKER_PROFILES } from '../bank/scoringPython'
import { installCompletion, installRuntimeNotice } from './installRuntimeNotice'

const POLL_MS = 1200
const MAX_POLL_FAILURES = 5

export function InstallRuntimeNotice({ result, onChoose }) {
  if (!result) return null
  return (
    <div role="status" className={`space-y-2 rounded-md border p-3 text-xs ${result.warn
      ? 'border-amber-400/50 bg-amber-500/10 text-amber-200' : 'border-border bg-surface-raised text-content'}`}>
      <p className="font-semibold">{result.title}</p>
      <p>{result.detail}</p>
      {result.validation && <p>{result.validation}</p>}
      <dl className="space-y-1 text-content-muted">
        <div><dt>Installed into</dt><dd className="break-all font-mono">{result.managedPython}</dd></div>
        <div><dt>Selected at the end of this repair</dt><dd className="break-all font-mono">{result.effectivePython}</dd></div>
      </dl>
      <button type="button" onClick={onChoose}
        className="min-h-10 rounded-md border border-current px-3 py-1.5 font-medium hover:bg-surface-raised">
        Choose Python / test calculation
      </button>
    </div>
  )
}

// One-click installer. Install failures retry in place; a repaired Bank environment
// with a broken external selection opens the shared Python picker instead. There
// is deliberately NO "run this pip command by hand" path here. Pip
// installs are serialized by the backend: a second one requested while one runs
// comes back 'queued' and starts on its own, so this shows an honest "waiting"
// state instead of a dead-looking button.
// `body` is posted with the start request (the AI Toolkit install sends its
// folder); `cancellable` shows a Cancel button for actions the backend can stop.
// The one by-hand path: a node pack that adds Python packages, on a ComfyUI
// whose Python LDS does not manage, ends with nothing installed and the exact
// steps (`manual_steps`), shown instead of a failure.
export default function InstallRunner({ action, buttonLabel, busyLabel = 'Installing…', onDone,
  body = null, disabled = false, cancellable = action === 'ollama_model',
  cancelLabel = 'Cancel download', cancelledNote = '', successMessage = '' }) {
  const toast = useToast()
  const [state, setState] = useState('idle')  // idle|queued|running|success|error|cancelled
  const [cancelling, setCancelling] = useState(false)
  const [log, setLog] = useState([])
  const [returncode, setReturncode] = useState(null)
  const [progress, setProgress] = useState(null)  // {done,total,pct} for streaming downloads
  const [step, setStep] = useState(null)  // {index,total,label} for multi-step installs
  const [runtimeNotice, setRuntimeNotice] = useState(null)
  const [manualSteps, setManualSteps] = useState([])
  const [pythonPicker, setPythonPicker] = useState(null)
  const timer = useRef(null)
  const mountedRef = useRef(true)
  const fails = useRef(0)

  const apply = (s) => {
    setState(s.state); setLog(s.log || []); setReturncode(s.returncode)
    setProgress(s.progress || null)
    setStep(s.step || null)
    setRuntimeNotice(s.runtime_notice || null)
    setManualSteps(Array.isArray(s.manual_steps)
      ? s.manual_steps.filter((line) => typeof line === 'string') : [])
    if (s.state !== 'running') setCancelling(false)
  }

  const poll = async () => {
    try {
      const s = await apiFetch(`/api/setup/install/${action}/status`)
      if (!mountedRef.current) return
      fails.current = 0
      apply(s)
      if (s.state === 'running' || s.state === 'queued') {
        timer.current = setTimeout(poll, POLL_MS)   // keep polling while queued too
      } else if (s.state === 'success') {
        const completion = installCompletion(s.state, s.runtime_notice)
        toast[completion.tone](successMessage && completion.tone === 'success'
          ? successMessage : completion.message); onDone?.()
      } else if (s.state === 'error' && s.manual_steps?.length) {
        toast.info('Nothing was installed. Follow the steps shown to install it yourself.')
      } else if (s.state === 'error') {
        const completion = installCompletion(s.state, s.runtime_notice)
        toast[completion.tone](completion.message)
        if (s.runtime_notice?.managed_installed) onDone?.()
      } else if (s.state === 'cancelled') {
        toast.info(action === 'ollama_model' ? 'Download cancelled.' : 'Install cancelled.')
      }
    } catch {
      if (!mountedRef.current) return
      fails.current += 1
      if (fails.current >= MAX_POLL_FAILURES) {
        // Stop hammering a down backend; tell the user to retry.
        setState('error')
        toast.error('Lost contact with the installer — check the server, then click to try again.')
      } else {
        timer.current = setTimeout(poll, POLL_MS)   // transient poll error — retry
      }
    }
  }

  // Re-attach on mount to an install that may already be running/queued/finished
  // (e.g. the user left this page mid-install and came back). Idle -> stay ready.
  useEffect(() => {
    mountedRef.current = true
    setRuntimeNotice(null); setPythonPicker(null)
    apiFetch(`/api/setup/install/${action}/status`).then((s) => {
      if (!mountedRef.current) return
      if (s.state === 'idle') return
      apply(s)
      if (s.state === 'running' || s.state === 'queued') timer.current = setTimeout(poll, POLL_MS)
    }).catch(() => { /* not attached; leave the button idle */ })
    return () => { mountedRef.current = false; clearTimeout(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [action])

  const start = async () => {
    setLog([]); setReturncode(null); setProgress(null); setStep(null); setState('running')
    setRuntimeNotice(null); setPythonPicker(null); setManualSteps([])
    setCancelling(false); fails.current = 0
    try {
      const s = await postJson(`/api/setup/install/${action}`, body || {})
      if (mountedRef.current && s && s.state) apply(s)  // immediate queued/running feedback
      poll()
    } catch (e) {
      setState('error')
      toast.error(e.message || 'Could not start install.')
    }
  }

  const cancel = async () => {
    if (!cancellable || state !== 'running' || cancelling) return
    setCancelling(true)
    try {
      const s = await postJson(`/api/setup/install/${action}/cancel`, {})
      if (!mountedRef.current) return
      apply(s)
      clearTimeout(timer.current)
      timer.current = setTimeout(poll, 0)
    } catch (error) {
      setCancelling(false)
      toast.error(error.message || 'Could not cancel.')
    }
  }

  const running = state === 'running'
  const busy = running || state === 'queued'
  const repairResult = installRuntimeNotice(runtimeNotice)
  const byHand = state === 'error' && manualSteps.length > 0
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={start} disabled={busy || disabled}
          className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-50">
          {state === 'queued' ? 'Queued…' : running ? busyLabel : buttonLabel}
        </button>
        {cancellable && running && (
          <button type="button" onClick={cancel} disabled={cancelling}
            className="rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised disabled:opacity-50">
            {cancelling ? 'Cancelling…' : cancelLabel}
          </button>
        )}
      </div>
      {state === 'queued' && (
        <p className="text-[11px] text-content-muted">
          Another install is running — this one starts automatically when it finishes.
        </p>
      )}
      {running && step && (
        <div className="space-y-1" role="status">
          <p className="text-[11px] text-content-muted">
            Step {step.index} of {step.total} · {step.label}
          </p>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
            <div className="h-full rounded-full bg-gradient-primary transition-[width] duration-300"
              style={{ width: `${Math.round(((step.index - 1) / step.total) * 100)}%` }} />
          </div>
        </div>
      )}
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
      {!busy && <InstallRuntimeNotice result={repairResult}
        onChoose={() => setPythonPicker(repairResult.profile)} />}
      {byHand && (
        <div role="status" className="space-y-2 rounded-md border border-amber-400/50 bg-amber-500/10 p-3 text-xs text-amber-200">
          <p className="font-semibold">Install it yourself</p>
          <p>This node pack adds Python packages. LDS installs those only on a ComfyUI Windows portable,
            where it manages ComfyUI’s Python, so nothing was installed here.</p>
          <ol className="list-decimal space-y-1 pl-5">
            {manualSteps.map((line, index) => (
              <li key={index} className="select-text break-all font-mono text-[11px]">{line}</li>
            ))}
          </ol>
        </div>
      )}
      {(log.length > 0 || running) && !byHand && (
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-surface-raised p-2 text-[11px] text-content-muted">
          {log.slice(-40).join('\n') || 'starting…'}
        </pre>
      )}
      {state === 'error' && !byHand && !repairResult?.errorIsSelection && (
        <p className="text-xs text-rose-400">
          {returncode != null
            ? `Install failed (exit ${returncode}). Click "${buttonLabel}" to try again — it repairs in place.`
            : 'Could not start the install. Click to try again.'}
        </p>
      )}
      {state === 'cancelled' && (
        <p className="text-xs text-content-muted">
          {cancelledNote || <>Download cancelled. The partial Ollama transfer can be resumed safely by clicking
          “{buttonLabel}” again.</>}
        </p>
      )}
      {pythonPicker && (
        <ScoringPythonDialog profile={PICKER_PROFILES[pythonPicker]}
          onClose={() => setPythonPicker(null)}
          onChanged={() => {
            setRuntimeNotice(null); setState('idle'); setReturncode(null); onDone?.()
          }} />
      )}
    </div>
  )
}
