import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { HelpBadge, useToast } from '@lds/plugin-sdk'
import { saveUrlAsFile } from '@lds/plugin-sdk/files'
import { SettingsLink } from '@lds/plugin-sdk/ui'
import { addCandidate, addToDataset, cancelCut, cutoutUrl, isRunning, maskUrl, readDatasets, readPicture,
  readStatus, startCut } from '../lib/api.js'
import { BACKGROUNDS, KEEPS, KINDS, blocked, missingModels, notReadyToCut, readOptions, runLabel, saveOptions } from '../lib/options.js'
import ModelInstall from './ModelInstall.jsx'
import Stage from './Stage.jsx'

const POLL = 1200
const TABS = [
  { id: 'choose', label: 'Original' },
  { id: 'result', label: 'Result' },
  { id: 'touch', label: 'Touch up' },
]
const tool = 'inline-flex min-h-10 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-45'
const quiet = `${tool} border-border text-content hover:border-border-strong`
const strong = `${tool} border-primary/50 bg-primary/10 font-semibold text-primary hover:bg-primary/20`

function Choices({ label, items, value, onChange, disabled = false, columns = 'grid-cols-2' }) {
  return <div role="radiogroup" aria-label={label} className={`grid gap-1.5 ${columns}`}>
    {items.map(item => <button key={item.id} type="button" role="radio" aria-checked={value === item.id} disabled={disabled}
      title={item.hint} onClick={() => onChange(item.id)}
      className={`min-h-10 rounded-lg border px-2.5 py-1.5 text-left text-sm disabled:opacity-50 ${value === item.id
        ? 'border-primary bg-primary/10 text-content' : 'border-border text-content-muted hover:border-border-strong'}`}>
      <span className="block font-medium">{item.label}</span>
      {item.model && <span className="block text-xs text-content-subtle">{item.model}</span>}
    </button>)}
  </div>
}

/**
 * The whole tool for one picture: what stays, the edge, the touch-up and where
 * the result goes. The page and the viewers' window show the same editor.
 */
export default function Editor({ picture, onChange, className = '' }) {
  const toast = useToast()
  const stage = useRef(null)
  const saved = useMemo(() => readOptions(globalThis.window?.localStorage), [])
  const [status, setStatus] = useState(null)
  const [keep, setKeep] = useState('subject')
  const [kind, setKind] = useState(saved.kind)
  const [phrase, setPhrase] = useState('')
  const [points, setPoints] = useState([])
  const [hasSelection, setHasSelection] = useState(false)
  const [leaveOut, setLeaveOut] = useState(false)
  const [background, setBackground] = useState(saved.background)
  const [clean, setClean] = useState(saved.clean)
  const [tab, setTab] = useState(picture.mask ? 'result' : 'choose')
  const [brush, setBrush] = useState({ keep: true, size: 32 })
  const [strokes, setStrokes] = useState({ count: 0, painted: false })
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [datasets, setDatasets] = useState(null)
  const [target, setTarget] = useState('')
  const change = useRef(onChange)
  change.current = onChange
  const current = useRef(picture)
  current.current = picture
  const running = isRunning(picture)
  const home = picture.origin?.kind === 'dataset'

  const refresh = useCallback(() => readStatus().then(setStatus).catch(problem => setError(problem.message || 'The models could not be checked.')), [])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { saveOptions(globalThis.window?.localStorage, { kind, background, clean }) }, [kind, background, clean])

  // Follow the running cut-out; the picture on screen changes only when something did.
  const runId = picture.run?.id
  useEffect(() => {
    if (!running) return undefined
    let timer = 0
    let alive = true
    const look = () => readPicture(current.current, { background: true }).then(next => {
      if (!alive) return
      change.current(next)
      if (isRunning(next)) timer = setTimeout(look, POLL)
      else if (next.run?.status === 'done') setTab('result')
    }).catch(() => { if (alive) timer = setTimeout(look, POLL * 3) })
    timer = setTimeout(look, POLL)
    return () => { alive = false; clearTimeout(timer) }
  }, [running, runId, picture.id])

  const choose = value => {
    setKeep(value)
    setLeaveOut(false)
    if (value === 'selection' || value === 'points') setTab('choose')
  }
  const missing = missingModels(status, keep, kind)
  const obstacle = blocked(status, keep, kind)
  const notYet = notReadyToCut({ keep, phrase, points, hasSelection })

  const cut = async () => {
    setError('')
    setSending(true)
    try {
      await stage.current?.flush()
      onChange(await startCut(picture, { keep, kind, phrase, points,
        selection: keep === 'selection' ? await stage.current.selection() : null }))
    } catch (problem) {
      setError(problem.message || 'The cut-out could not start.')
      refresh()
    } finally { setSending(false) }
  }
  const stop = async () => {
    try { onChange(await cancelCut(picture)) } catch (problem) { setError(problem.message || 'It could not be cancelled.') }
  }

  /** Run `work` on the mask as it is on screen, its last strokes included. */
  const withResult = async work => {
    setError('')
    setSending(true)
    try { await work((await stage.current?.flush()) || current.current) }
    catch (problem) { setError(problem.message || 'This did not work. Try again.') }
    finally { setSending(false) }
  }
  const download = () => withResult(next => saveUrlAsFile(`${cutoutUrl(next, { background, clean })}&download=1`,
    { fallbackName: 'cutout.png', failure: 'The cut-out could not be downloaded.' }))
  const downloadMask = () => withResult(next => saveUrlAsFile(`${maskUrl(next)}&download=1`,
    { fallbackName: 'mask.png', failure: 'The mask could not be downloaded.' }))
  const toHome = () => withResult(async next => {
    await addCandidate(next, { background, clean })
    toast.success('The cut-out is on its way to this dataset, as a candidate to keep or reject.')
  })
  const openDatasets = () => withResult(async () => {
    const list = await readDatasets()
    setDatasets(list)
    setTarget(String(picture.origin?.dataset_id || list[0]?.id || ''))
  })
  const toDataset = () => withResult(async next => {
    const answer = await addToDataset(next, target, { background, clean })
    if (answer.imported) toast.success(`Added to ${datasets.find(item => String(item.id) === target)?.name || 'the dataset'}.`)
    else throw new Error(answer.duplicates ? 'This dataset already has a picture that looks the same.' : 'The dataset could not read this picture.')
  })

  const shownTab = picture.mask ? tab : 'choose'
  const mode = shownTab === 'result' ? 'look' : shownTab === 'touch' ? 'brush'
    : keep === 'selection' ? 'select' : keep === 'points' ? 'points' : 'look'
  const result = picture.mask && shownTab === 'result' ? cutoutUrl(picture, { clean, max: 2048 }) : ''
  const plain = background !== 'transparent'
  const failed = picture.run?.status === 'failed' ? picture.run.error : ''
  const onError = useCallback(message => setError(message), [])
  const onPainted = useCallback(next => change.current(next), [])

  return <div className={`grid min-h-0 min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_21rem] ${className}`}>
    <section aria-label="Picture" className="flex min-h-0 min-w-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {picture.mask && <div role="tablist" aria-label="What is shown" className="flex rounded-lg border border-border p-0.5">
          {TABS.map(item => <button key={item.id} type="button" role="tab" aria-selected={shownTab === item.id}
            onClick={() => { if (item.id === 'result') stage.current?.flush().catch(() => {}); setTab(item.id) }}
            className={`min-h-9 rounded-md px-2 text-sm sm:px-3 ${shownTab === item.id ? 'bg-primary/15 font-semibold text-primary' : 'text-content-muted hover:text-content'}`}>{item.label}</button>)}
        </div>}
        <div className="ml-auto flex shrink-0 gap-1">
          {[['Zoom out', '−', () => stage.current?.zoom(0.8)], ['Fit the whole picture', '⤢', () => stage.current?.fit()],
            ['Zoom in', '+', () => stage.current?.zoom(1.25)]].map(([label, sign, run]) => <button key={label} type="button" title={label}
            aria-label={label} onClick={run} className="size-8 rounded-lg border border-border text-content hover:border-border-strong sm:size-9">{sign}</button>)}
        </div>
      </div>
      {mode === 'brush' && <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Brush">
        <div role="radiogroup" aria-label="What the brush does" className="flex rounded-lg border border-border p-0.5">
          {[[true, 'Keep'], [false, 'Remove']].map(([value, label]) => <button key={label} type="button" role="radio" aria-checked={brush.keep === value}
            onClick={() => setBrush(item => ({ ...item, keep: value }))}
            className={`min-h-9 rounded-md px-3 text-sm ${brush.keep === value ? 'bg-primary/15 font-semibold text-primary' : 'text-content-muted'}`}>{label}</button>)}
        </div>
        <label className="flex items-center gap-2 text-sm text-content-muted">Size
          <input type="range" min="6" max="160" step="2" value={brush.size} aria-label="Brush size"
            onChange={event => setBrush(item => ({ ...item, size: Number(event.target.value) }))} className="w-28 accent-primary" /></label>
        <button type="button" className={quiet} disabled={!strokes.count} onClick={() => stage.current?.undo()}>Undo</button>
        <button type="button" className={quiet} disabled={!strokes.count && !strokes.painted} onClick={() => stage.current?.resetMask()}
          title="Go back to the mask the models made">Reset</button>
      </div>}
      {(mode === 'select' || mode === 'points') && <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="What stays">
        <div role="radiogroup" aria-label="What a press means" className="flex rounded-lg border border-border p-0.5">
          {[[false, 'Keep'], [true, 'Leave out']].map(([value, label]) => <button key={label} type="button" role="radio" aria-checked={leaveOut === value}
            onClick={() => setLeaveOut(value)}
            className={`min-h-9 rounded-md px-3 text-sm ${leaveOut === value ? 'bg-primary/15 font-semibold text-primary' : 'text-content-muted'}`}>{label}</button>)}
        </div>
        <button type="button" className={quiet} disabled={mode === 'points' ? !points.length : !hasSelection}
          onClick={() => mode === 'points' ? setPoints([]) : stage.current?.clearSelection()}>Clear</button>
      </div>}
      <Stage ref={stage} key={picture.id} picture={picture} mode={mode} result={result}
        background={BACKGROUNDS.find(item => item.id === background)?.css || ''}
        points={points} onPoints={setPoints} leaveOut={leaveOut} onSelection={setHasSelection} brush={brush}
        onPainted={onPainted} onStrokes={setStrokes} onError={onError}
        className="min-h-[16rem] flex-1 rounded-xl border border-border bg-black/40" />
    </section>

    <section aria-label="Remove the background" className="min-w-0 space-y-4 lg:overflow-y-auto lg:pr-1">
      <div className="space-y-2">
        <h2 className="m-0 text-sm font-semibold text-content">What stays<HelpBadge topic="remove-background" className="ml-1" /></h2>
        <Choices label="What stays" items={KEEPS} value={keep} onChange={choose} disabled={running} />
        <p className="m-0 text-xs text-content-subtle">{KEEPS.find(item => item.id === keep).hint}</p>
        {keep === 'describe' && <input type="text" value={phrase} maxLength={300} disabled={running} aria-label="What stays, in a few words"
          placeholder="woman in red, dog" onChange={event => setPhrase(event.target.value)}
          onKeyDown={event => { if (event.key === 'Enter' && !notYet && !missing.length && !obstacle && !running && !sending) cut() }}
          className="min-h-10 w-full rounded-lg border border-border-strong bg-surface-raised px-3 py-2 text-sm text-content placeholder:text-content-subtle" />}
      </div>
      <div className="space-y-2">
        <h2 className="m-0 text-sm font-semibold text-content">Kind of picture</h2>
        <Choices label="Kind of picture" items={KINDS} value={kind} onChange={setKind} disabled={running} columns="grid-cols-1" />
        <p className="m-0 text-xs text-content-subtle">{KINDS.find(item => item.id === kind).hint}</p>
      </div>
      {obstacle ? <p role="status" className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 p-3 text-sm text-amber-100">{obstacle}{' '}
        {status && !status.configured && <SettingsLink section="local-tools" focus="comfyui-base-dir">Open the ComfyUI settings</SettingsLink>}</p>
        : missing.length > 0 && <ModelInstall models={status.models} slots={missing} onDone={refresh} />}
      {running ? <div className="flex items-center gap-2">
        <p role="status" aria-live="polite" className="m-0 flex-1 text-sm text-content">{runLabel(picture.run)}</p>
        <button type="button" className={quiet} onClick={stop}>Cancel</button>
      </div> : <button type="button" className={`${strong} w-full`} title={notYet}
        disabled={sending || Boolean(obstacle) || missing.length > 0 || Boolean(notYet)} onClick={cut}>
        {picture.mask ? 'Remove the background again' : 'Remove the background'}</button>}
      {!running && notYet && !obstacle && !missing.length && <p className="m-0 text-xs text-content-subtle">{notYet}</p>}
      {failed && !running && <p role="alert" className="m-0 rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{failed}</p>}
      {error && <p role="alert" className="m-0 rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</p>}

      {picture.mask && <div className="space-y-3 border-t border-border pt-4">
        {!picture.mask.matted && <p className="m-0 text-xs text-amber-200">The edge comes from SAM 3.1 alone: the matting model was looking at something else.</p>}
        <h2 className="m-0 text-sm font-semibold text-content">Result</h2>
        <Choices label="Background of the result" items={BACKGROUNDS} value={background} onChange={setBackground} columns="grid-cols-2" />
        <label className="flex min-h-10 items-center gap-2 text-sm text-content-muted">
          <input type="checkbox" checked={clean} onChange={event => setClean(event.target.checked)} className="accent-primary" />
          <span>Clean edge colours<span className="block text-xs text-content-subtle">Takes the old background&apos;s colour out of hair and soft edges.</span></span></label>
        <div className="grid grid-cols-2 gap-2">
          <button type="button" className={strong} disabled={sending || running} onClick={download}>Download</button>
          <button type="button" className={quiet} disabled={sending || running} onClick={downloadMask}>Download mask</button>
        </div>
        {home && <button type="button" className={`${quiet} w-full`} disabled={sending || running || !plain} onClick={toHome}>Add to this dataset</button>}
        {!home && !datasets && <button type="button" className={`${quiet} w-full`} disabled={sending || running || !plain} onClick={openDatasets}>Add to a dataset…</button>}
        {!home && datasets && (datasets.length ? <div className="flex gap-2">
          <select value={target} onChange={event => setTarget(event.target.value)} aria-label="Dataset"
            className="min-h-10 min-w-0 flex-1 rounded-lg border border-border-strong bg-surface-raised px-2 text-sm text-content">
            {datasets.map(item => <option key={item.id} value={String(item.id)}>{item.name}</option>)}
          </select>
          <button type="button" className={quiet} disabled={sending || running || !plain || !target} onClick={toDataset}>Add</button>
        </div> : <p className="m-0 text-xs text-content-subtle">There is no dataset yet.</p>)}
        {!plain && <p className="m-0 text-xs text-content-subtle">A training picture cannot be transparent: choose a plain background to add this to a dataset.</p>}
      </div>}
    </section>
  </div>
}
