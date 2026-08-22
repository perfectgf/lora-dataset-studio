import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from './Toast'
import {
  elapsedLabel, hasQueue, jobLabel, jobOrigin, pausedReason, promoteBlockedReason,
  rowNote, summarize,
} from '../../utils/queuePanel'
import { dockBottomClass } from '../../utils/dockPlacement'

/**
 * ⏳ The generation queue, where you can see it.
 *
 * LDS has always HAD a queue — `job_queue` is a FIFO worker over
 * `image_generation_queue` and it takes one job at a time — but it never had a
 * face. Each surface showed only its own slice: the tiles a dataset is waiting
 * on, the cells of a Studio run. So work queued from one screen was invisible
 * from every other, and the two questions a queue exists to answer ("what is
 * the GPU doing?", "is mine stuck or just behind something?") had no answer
 * anywhere in the app. That is what GitHub #44 asked for, in the words of
 * someone who had assumed there was no queue at all.
 *
 * Mounted once in the shell, like the recovery banner, and for the same reason:
 * one ComfyUI, one queue, fed by the dataset workspace, the Test Studio, the
 * ◉ Canvas and the Bank. A dock that only existed on one of them would rebuild
 * the very blind spot it is here to remove.
 *
 * It is silent when the queue is empty — which is most of the time.
 */
const IDLE_POLL_MS = 6000
const OPEN_POLL_MS = 2500

export default function GenerationQueueDock() {
  const toast = useToast()
  // Some screens own the bottom of the window with their own fixed bar; the dock
  // moves up rather than fighting them (see utils/dockPlacement.js).
  const { pathname } = useLocation()
  const [listing, setListing] = useState(null)
  const [open, setOpen] = useState(false)
  const [pending, setPending] = useState(null)   // job_id of the action in flight
  const aliveRef = useRef(true)

  const poll = useCallback(async () => {
    try {
      // background: this ticks forever on every screen, so a server blink must
      // not toast. The offline banner already owns that story.
      const data = await apiFetch('/api/system/queue', { background: true })
      if (aliveRef.current) setListing(data)
      return data
    } catch {
      // An older backend without the route, or a server that is down: the dock
      // stays quiet. It is not the outage reporter.
      return null
    }
  }, [])

  useEffect(() => {
    aliveRef.current = true
    poll()
    const timer = setInterval(poll, open ? OPEN_POLL_MS : IDLE_POLL_MS)
    return () => { aliveRef.current = false; clearInterval(timer) }
  }, [poll, open])

  // An empty queue collapses the dock as well as hiding it: reopening on the
  // next job, still expanded from an hour ago, would cover the app unasked.
  useEffect(() => { if (!hasQueue(listing) && open) setOpen(false) }, [listing, open])

  const act = useCallback(async (job, action) => {
    setPending(job.job_id)
    try {
      await postJson(`/api/system/queue/${encodeURIComponent(job.job_id)}/${action}`, {})
      toast.success(action === 'next'
        ? `${jobLabel(job)} runs next.`
        : `${jobLabel(job)} cancelled.`)
    } catch (e) {
      // The server refuses what it cannot do safely and its message says which
      // — surface it verbatim rather than a generic failure.
      toast.error(e?.message || 'That job could not be updated.')
    } finally {
      // Clear only OUR job. A flat `setPending(null)` let the first action to
      // return unlock a row whose own request was still in flight, re-enabling
      // its buttons for a second click.
      if (aliveRef.current) setPending((cur) => (cur === job.job_id ? null : cur))
      await poll()
    }
  }, [poll, toast])

  if (!hasQueue(listing)) return null
  return (
    <QueueDockBody listing={listing} open={open} pending={pending}
      bottomClass={dockBottomClass(pathname)}
      onToggle={() => setOpen((v) => !v)}
      onPromote={(job) => act(job, 'next')}
      onCancel={(job) => act(job, 'cancel')} />
  )
}

/**
 * The pixels alone — no polling, no fetching — so a test can render every state
 * the dock can be in. Splitting this out is not cosmetic: the polling component
 * returns `null` until an effect has answered, and `node --test` never runs
 * effects, so mounting the default export would prove nothing about what a user
 * with a full queue actually sees.
 */
export function QueueDockBody({ listing, open = false, pending = null,
                                bottomClass = 'bottom-4',
                                onToggle, onPromote, onCancel }) {
  if (!hasQueue(listing)) return null
  const jobs = listing.jobs
  const paused = pausedReason(listing)
  return (
    <div className={`fixed ${bottomClass} left-3 z-40 w-[min(23rem,calc(100vw-1.5rem))]`}>
      {open && (
        <div className="mb-2 max-h-[min(60vh,28rem)] overflow-y-auto rounded-xl border border-indigo-400/40 bg-surface-overlay/95 shadow-lg backdrop-blur">
          {/* Above the list, because it explains the whole list: nothing here is
              moving, and this is what is holding the GPU instead. */}
          {paused && (
            <p className="border-b border-border bg-amber-400/10 px-3 py-2 text-content text-[0.6875rem] leading-snug">
              {paused}
            </p>
          )}
          <ul className="divide-y divide-border">
            {jobs.map((job) => (
              <QueueRow key={job.job_id} job={job} busy={pending === job.job_id}
                onPromote={onPromote} onCancel={onCancel} />
            ))}
          </ul>
        </div>
      )}
      {/* The hold reason belongs in the COLLAPSED name too. A pill reading "4
          queued" that never moves is the question; making the user open the dock
          to find the answer would be the same silence, one click deep. */}
      <button type="button" onClick={onToggle}
        aria-expanded={open}
        aria-label={paused
          ? `Generation queue — ${summarize(listing)}, on hold: ${paused}`
          : `Generation queue — ${summarize(listing)}`}
        title={paused || undefined}
        className="flex w-full items-center gap-2 rounded-xl border border-indigo-400/40 bg-surface-overlay/95 px-3 py-2 text-left shadow-lg backdrop-blur hover:border-indigo-400/70">
        <span aria-hidden="true"
          className={listing.generating && !paused ? 'animate-pulse' : undefined}>
          {paused ? '⏸' : '⏳'}
        </span>
        <span className="text-content text-sm font-semibold">{summarize(listing)}</span>
        <span aria-hidden="true" className="ml-auto text-content-subtle text-xs">
          {open ? '▾' : '▴'}
        </span>
      </button>
    </div>
  )
}

function QueueRow({ job, busy, onPromote, onCancel }) {
  const note = rowNote(job)
  const promoteBlocked = promoteBlockedReason(job)
  return (
    <li className="flex flex-col gap-1 px-3 py-2">
      <div className="flex items-start gap-2">
        <span aria-hidden="true" className="mt-0.5 shrink-0 text-xs">
          {job.status === 'generating' ? '▶' : job.status === 'stalled' ? '⏸' : `${job.position}.`}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-content text-sm">{jobLabel(job)}</p>
          <p className="truncate text-content-subtle text-[0.6875rem]">
            {jobOrigin(job)}
            {job.since ? ` · ${elapsedLabel(job.since)}` : ''}
            {job.promoted ? ' · moved up' : ''}
          </p>
        </div>
        <div className="flex shrink-0 gap-1">
          {/* ↑ rather than a nicer-looking ⤒ (U+2912): at 11px in the app's
              stack that glyph renders as a bare tick, indistinguishable from a
              stray mark — measured in the headless proof, not guessed.
              It is off for two DIFFERENT reasons and each says its own —
              "already next" is not a failure, and a mute grey button is what
              sent people to the issue tracker in the first place. */}
          <button type="button" onClick={() => onPromote?.(job)}
            disabled={busy || !job.promotable || !!promoteBlocked}
            title={promoteBlocked || 'Run this one next'}
            aria-label={promoteBlocked || `Run ${jobLabel(job)} next`}
            className="grid min-h-7 min-w-7 place-items-center rounded bg-app/60 text-[11px] text-content disabled:cursor-not-allowed disabled:opacity-40">
            <span aria-hidden="true">↑</span>
          </button>
          <button type="button" onClick={() => onCancel?.(job)}
            disabled={busy || !job.cancellable}
            title={job.cancellable
              ? 'Cancel this job — its tile is left marked failed, and Retry re-queues it'
              : note || 'This job cannot be cancelled from here'}
            aria-label={job.cancellable ? `Cancel ${jobLabel(job)}` : note || 'Cannot be cancelled here'}
            className="grid min-h-7 min-w-7 place-items-center rounded bg-app/60 text-[11px] text-content disabled:cursor-not-allowed disabled:opacity-40">
            <span aria-hidden="true">✕</span>
          </button>
        </div>
      </div>
      {note && <p className="text-content-subtle text-[0.625rem] leading-snug">{note}</p>}
    </li>
  )
}
