import { useEffect, useRef, useState } from 'react'
import { fmtSize } from '@lds/plugin-sdk/setup'
import { installStatus, startInstall } from '../lib/api.js'

const POLL = 1500
const ended = state => state === 'success' || state === 'error'

/** The model files one choice needs, with a button that fetches exactly those. */
export default function ModelInstall({ models, slots, onDone }) {
  const [statuses, setStatuses] = useState({})
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const timer = useRef(0)
  const alive = useRef(true)
  const done = useRef(onDone)
  done.current = onDone
  const actions = slots.map(slot => models[slot].action)
  const key = actions.join(',')

  const watch = list => {
    installStatus(list).then(data => {
      if (!alive.current) return
      const found = data.statuses || {}
      setStatuses(found)
      if (list.every(action => ended(found[action]?.state))) {
        setRunning(false)
        const failed = list.find(action => found[action]?.state === 'error')
        if (failed) setError([].concat(found[failed].log || []).slice(-1)[0] || 'A download failed. Try again.')
        done.current?.()
      } else timer.current = setTimeout(() => watch(list), POLL)
    }).catch(() => { if (alive.current) timer.current = setTimeout(() => watch(list), POLL * 2) })
  }

  // A download started earlier, here or in Settings, is picked up where it is.
  useEffect(() => {
    alive.current = true
    const list = key ? key.split(',') : []
    if (list.length) installStatus(list).then(data => {
      if (!alive.current) return
      const busy = list.filter(action => ['running', 'queued'].includes(data.statuses?.[action]?.state))
      if (busy.length) { setStatuses(data.statuses); setRunning(true); watch(busy) }
    }).catch(() => { /* not attached: the button stays */ })
    return () => { alive.current = false; clearTimeout(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const start = async () => {
    setError('')
    setRunning(true)
    try {
      const data = await startInstall(actions)
      setStatuses(data.statuses || {})
      watch(data.plan || actions)
    } catch (problem) {
      setRunning(false)
      setError(problem.message || 'The download could not start.')
    }
  }

  if (!slots.length) return null
  const total = slots.reduce((sum, slot) => sum + models[slot].bytes, 0)
  const licences = [...new Map(slots.map(slot => [models[slot].license_url, models[slot]])).values()]
  return <div className="space-y-2 rounded-lg border border-border p-3 text-sm">
    <ul className="m-0 list-none space-y-1 p-0 text-content-muted">
      {slots.map(slot => {
        const { action, name, bytes } = models[slot]
        const { state, progress } = statuses[action] || {}
        return <li key={slot} className="flex items-baseline justify-between gap-3">
          <span className="min-w-0 break-words">{name}</span>
          <span className="shrink-0 text-xs tabular-nums text-content-subtle">
            {running && state === 'running' && progress?.total ? `${Math.floor(progress.done * 100 / progress.total)}% of ${fmtSize(progress.total)}`
              : running && state === 'success' ? 'Installed' : running && state === 'queued' ? 'Waiting' : fmtSize(bytes)}
          </span>
        </li>
      })}
    </ul>
    <p className="m-0 text-xs text-content-subtle">
      {licences.map((model, i) => <span key={model.license_url}>{i > 0 && ' · '}
        <a href={model.license_url} target="_blank" rel="noreferrer" className="underline">{model.name}: {model.license}</a></span>)}
    </p>
    {error && <p role="alert" className="m-0 break-words text-xs text-red-300">{error}</p>}
    <button type="button" onClick={start} disabled={running}
      className="min-h-10 w-full rounded-lg border border-primary/50 bg-primary/10 px-3 py-2 text-sm font-semibold text-primary disabled:opacity-60">
      {running ? 'Installing…' : `Install (${fmtSize(total)})`}
    </button>
  </div>
}
