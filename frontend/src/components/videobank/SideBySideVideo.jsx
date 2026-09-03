import { useEffect, useRef, useState } from 'react'
import { syncActions, sidesFor } from './videoSync'
import { isAbort, saveUrlAsFile } from '../../utils/fileSave'

/** ⇔ The original and its neural render, side by side and in step.
 *
 * ONE component for both surfaces (a rendered dataset clip, a studio clip and
 * its source): the hosts hand over two URLs and a title, nothing else. The left
 * player is the LEADER — its controls are the only controls — and the right one
 * follows: play, pause, seek and rate are mirrored, and drift past two frames
 * is corrected on every timeupdate (see videoSync.js for the rules).
 *
 * Two <video> elements, deliberately and knowingly: this is a layer opened for
 * one comparison and closed after, never a grid — the ~60-player ceiling that
 * keeps <video> out of the grids does not apply to a pair.
 *
 * Narrow screens stack the two under each other (a phone cannot show two
 * portrait clips side by side at any useful size); `swap` puts the render first
 * for whoever reads left to right and wants the render as the reference.
 */
export default function SideBySideVideo({ originalSrc, renderSrc, title, exportHref, onClose }) {
  const leader = useRef(null)
  const follower = useRef(null)
  const [swapped, setSwapped] = useState(false)
  // 1:1 shows the pixels the render changed; fitted to the pane they vanish.
  // At 1:1 both panes scroll, and one pane's scroll drives the other.
  const [oneToOne, setOneToOne] = useState(false)
  const panes = useRef([])
  const syncingScroll = useRef(false)
  const followScroll = (from) => (e) => {
    if (syncingScroll.current) return
    const other = panes.current[1 - from]
    if (!other) return
    syncingScroll.current = true
    other.scrollLeft = e.currentTarget.scrollLeft
    other.scrollTop = e.currentTarget.scrollTop
    syncingScroll.current = false
  }
  const [failed, setFailed] = useState({ original: false, render: false })
  // ⬇ The same picture as ONE file. The server encodes it, so the wait is real
  // and the button says so; the error goes on screen rather than into a console
  // nobody has open.
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState('')
  const sides = sidesFor(swapped)
  const srcFor = (key) => (key === 'original' ? originalSrc : renderSrc)

  // Closing the layer hangs up on an export in flight. The server finishes the
  // encode either way (measured), so this is about the browser: no megabytes
  // held for a window that is gone, and no state set on a dead component.
  const exportAbort = useRef(null)
  useEffect(() => () => exportAbort.current?.abort(), [])

  const exportFile = async () => {
    if (exporting) return
    setExporting(true)
    setExportError('')
    const controller = new AbortController()
    exportAbort.current = controller
    try {
      await saveUrlAsFile(exportHref, {
        fallbackName: 'comparison.mp4',
        failure: 'The comparison could not be built.',
        signal: controller.signal,
      })
    } catch (err) {
      if (!isAbort(err)) setExportError(err.message || 'The comparison could not be built.')
    } finally {
      exportAbort.current = null
      setExporting(false)
    }
  }

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.()
      if (e.key === ' ' && leader.current && e.target === document.body) {
        e.preventDefault()
        if (leader.current.paused) leader.current.play?.()
        else leader.current.pause()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // The follower is driven from the leader's events. A snapshot is taken on
  // each, the pure rules decide, and the actions are applied in their order.
  useEffect(() => {
    const a = leader.current
    const b = follower.current
    if (!a || !b) return undefined
    const apply = () => {
      const actions = syncActions(
        { currentTime: a.currentTime, paused: a.paused, playbackRate: a.playbackRate },
        { currentTime: b.currentTime, paused: b.paused, playbackRate: b.playbackRate })
      for (const act of actions) {
        if (act.type === 'rate') b.playbackRate = act.value
        else if (act.type === 'seek') b.currentTime = act.value
        else if (act.type === 'pause') b.pause()
        else if (act.type === 'play') b.play?.()?.catch?.(() => {})
      }
    }
    const events = ['play', 'pause', 'seeked', 'ratechange', 'timeupdate']
    events.forEach((ev) => a.addEventListener(ev, apply))
    return () => events.forEach((ev) => a.removeEventListener(ev, apply))
  }, [swapped, originalSrc, renderSrc])

  const player = (side, isLeader) => (
    <figure key={side.key} className="flex min-w-0 flex-col gap-1">
      <figcaption className="flex items-center justify-between text-xs font-semibold text-content">
        <span>{side.label}</span>
        {isLeader && <span className="font-normal text-content-subtle">controls</span>}
      </figcaption>
      {failed[side.key] ? (
        <p className="rounded-md border border-border bg-surface-raised px-3 py-6 text-center text-xs text-content-muted">
          This side could not be loaded — the file may have been restored or removed.
        </p>
      ) : (
        <div ref={(el) => { panes.current[isLeader ? 0 : 1] = el }} onScroll={followScroll(isLeader ? 0 : 1)}
          className={oneToOne ? 'max-h-[70vh] overflow-auto rounded-md bg-black' : ''}>
          <video ref={isLeader ? leader : follower} src={srcFor(side.key)}
            controls={isLeader} muted={!isLeader} preload="metadata" playsInline
            onError={() => setFailed((f) => ({ ...f, [side.key]: true }))}
            aria-label={side.label}
            className={oneToOne ? 'max-w-none' : 'max-h-[70vh] w-full rounded-md bg-black object-contain'} />
        </div>
      )}
    </figure>
  )

  return (
    <div role="dialog" aria-modal="true" aria-label={`Compare: ${title || 'original and neural render'}`}
      data-probe-layer
      className="fixed inset-0 z-[60] flex flex-col bg-black/90 p-2 sm:p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose?.() }}>
      <div className="mx-auto flex w-full max-w-6xl min-w-0 flex-1 flex-col gap-2 overflow-y-auto rounded-xl border border-border bg-surface-overlay p-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-content">⇔ {title || 'Original vs neural render'}</h2>
          {exportHref && (
            <button type="button" onClick={exportFile} disabled={exporting}
              title="Save the two clips as one video, side by side — labelled, in step, ready to send"
              className="min-h-10 rounded-md border border-border px-2 py-1 text-xs text-content-muted hover:text-content disabled:opacity-60 lg:min-h-0">
              {exporting ? 'Building…' : '⬇ Export'}
            </button>
          )}
          <button type="button" onClick={() => setOneToOne((z) => !z)} aria-pressed={oneToOne}
            title="Show the pixels at their real size — the detail the render adds is invisible once the frame is shrunk to fit"
            className={`min-h-10 rounded-md border px-2 py-1 text-xs lg:min-h-0 ${oneToOne ? 'border-border-strong bg-surface-raised text-content' : 'border-border text-content-muted hover:text-content'}`}>
            1:1
          </button>
          <button type="button" onClick={() => setSwapped((s) => !s)}
            className="min-h-10 rounded-md border border-border px-2 py-1 text-xs text-content-muted hover:text-content lg:min-h-0">
            Swap sides
          </button>
          <button type="button" onClick={onClose} aria-label="Close the comparison"
            className="min-h-10 rounded-md border border-border px-3 py-1 text-sm text-content hover:bg-surface-raised lg:min-h-0">
            ✕
          </button>
        </div>
        <p className="text-[0.6875rem] text-content-subtle">
          The left player leads: play, pause and seek there and the right one follows in step (the right one is muted).
          {oneToOne ? ' At 1:1, scrolling one pane scrolls the other.' : ' Press 1:1 to see the pixels at their real size.'}
          {exportHref ? ' ⬇ Export saves both as one labelled video.' : ''}
        </p>
        {exportError && (
          <p role="alert" className="text-[0.6875rem] text-red-300">{exportError}</p>
        )}
        <div className="grid min-w-0 grid-cols-1 gap-3 md:grid-cols-2">
          {player(sides[0], true)}
          {player(sides[1], false)}
        </div>
      </div>
    </div>
  )
}
