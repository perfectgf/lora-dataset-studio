import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import {
  videoBankUrl, videoClipsUrl, videoPassUrl, videoSourceClipsUrl,
  videoSourceRecutUrl, videoSourceSingleShotUrl,
} from './videoBankApi'
import { retouchToast } from './videoClipEdit'
import { passBlockedBy } from './videoCapability'
import {
  countsSummary, countsProblems, activityLine, activityPercent, isBusy,
  resumeSafetyNote,
  announcement, nextStep, passLabel, PASS_LABELS,
} from './videoBankStatus'
import {
  STATUS_FILTERS, statusFilterCount, toggleSelection, selectRange,
  triagePayload, triageAllPayload, triageAllConfirmation, emptyGridMessage,
  hasMore,
} from './videoTriage'
import VideoCapabilityStrip from './VideoCapabilityStrip'
import VideoShotCutsPanel from './VideoShotCutsPanel'
import VideoThresholdsPanel from './VideoThresholdsPanel'
import VideoSourceList from './VideoSourceList'
import VideoClipGrid from './VideoClipGrid'
import VideoClipLightbox from './VideoClipLightbox'
import VideoClipSearchBox from './VideoClipSearchBox'
import { matchLine, captionStyleLabel } from './videoClipSearch'
import { filterByFlag, flagChips, flagFilterNote } from './videoMetricsFilter'
import PromoteVideoDialog from './PromoteVideoDialog'

const PAGE = 120
const POLL_MS = 2000

/** 🎬 One video bank — sources, passes, and the shot gallery.
 *
 * A DELIBERATELY SEPARATE COMPONENT TREE from the image bank's workspace. The
 * two look alike from a distance and are not the same job: this one cuts one
 * file into hundreds of shots, stores BOUNDS rather than files, and encodes only
 * at promotion. Folding it into the image workspace (already 2500 lines) would
 * have bought shared chrome at the price of every conditional in it.
 *
 * The poll is the same 2 s contract as the image lane, on the payload that
 * already carries the live job — so a running pass costs one request, not two.
 */
export default function VideoBankWorkspace({ bankId, onBack, onGone }) {
  const toast = useToast()
  const [bank, setBank] = useState(null)
  const [clips, setClips] = useState([])
  const [total, setTotal] = useState(0)
  const [status, setStatus] = useState('all')
  const [sourceId, setSourceId] = useState(null)
  const [selected, setSelected] = useState([])
  const [anchor, setAnchor] = useState(null)
  const [openIndex, setOpenIndex] = useState(null)
  const [promoting, setPromoting] = useState(false)
  const [loadingClips, setLoadingClips] = useState(false)
  // A retouch ADDED a shot (a split, or a hand-made cut). The gallery is reloaded
  // when the player closes rather than under it: `openIndex` addresses the list by
  // POSITION, so inserting a row mid-session moves the player onto another shot.
  const [pendingRefresh, setPendingRefresh] = useState(false)
  // 🔎 A search REPLACES what the grid shows, and does not touch the filters.
  // Two reasons it is held here rather than inside the box: the grid and the
  // lightbox both read it (the ranking is an order, and the matched second is
  // where the player opens), and a triage action has to be able to update the
  // ranked rows in place — a search result that stops reflecting a Keep the user
  // just pressed is worse than no search at all.
  // 🗣 Which prompt the NEXT caption run uses. Per-run, seeded from the config
  // default the server reports: captioning one bank plainly must not silently
  // re-point every other bank, and the measurement says the prompt matters more
  // than the checkpoint.
  const [captionStyle, setCaptionStyle] = useState(null)
  const [search, setSearch] = useState(null)
  const [searching, setSearching] = useState(false)
  // ⚑ Which verdict the grid is narrowed to, or null. Client-side over the shots
  // already loaded — the status filter is a server-side query and this is not,
  // which is a real difference and the reason the chip row carries a note.
  const [flag, setFlag] = useState(null)
  // The last job we announced, so a finished pass is toasted ONCE instead of on
  // every poll for as long as the server keeps its snapshot.
  const announced = useRef(null)

  const loadBank = useCallback(async (refresh = false) => {
    try {
      const d = await apiFetch(videoBankUrl(bankId, { refresh }),
        refresh ? {} : { background: true })
      setBank(d)
      return d
    } catch (e) {
      if (e?.status === 404) { onGone?.(); return null }
      // A failed POLL says nothing (apiFetch already owns the offline banner);
      // a failed OPEN is worth a line.
      if (refresh) toast.error(e?.message || 'Could not load this bank.')
      return null
    }
  }, [bankId, onGone, toast])

  const loadClips = useCallback(async (append = false) => {
    setLoadingClips(true)
    try {
      const d = await apiFetch(videoClipsUrl(bankId, {
        status, sourceId, offset: append ? clips.length : 0, limit: PAGE,
      }), { background: true })
      setClips((prev) => (append ? [...prev, ...(d.clips || [])] : (d.clips || [])))
      setTotal(d.total || 0)
    } catch {
      if (!append) { setClips([]); setTotal(0) }
    } finally {
      setLoadingClips(false)
    }
  }, [bankId, status, sourceId, clips.length])

  // Open: one refreshing read (the folder is LIVE — people keep dropping files
  // into it) plus the first page of shots.
  useEffect(() => { loadBank(true) }, [bankId])          // eslint-disable-line react-hooks/exhaustive-deps
  // Changing the filter clears the search. A ranking computed over one bucket
  // has nothing to say about another, and leaving it on screen while the chips
  // moved would show "keep only" over shots the search found in every bucket.
  useEffect(() => {
    // The flag chip goes with them: it was computed over the previous bucket's
    // clips, and a chip left pressed over a page it never counted narrows the
    // grid to something the user did not ask for.
    setSelected([]); setAnchor(null); setSearch(null); setOpenIndex(null); setFlag(null)
    loadClips(false)
  }, [bankId, status, sourceId])                          // eslint-disable-line react-hooks/exhaustive-deps

  // The 2 s poll. Never sends refresh=1: that re-walks the whole source tree
  // server-side, and doing it every two seconds on a folder of rushes is a
  // directory scan per tick for an answer that changes once an hour.
  useEffect(() => {
    const t = setInterval(() => { loadBank(false) }, POLL_MS)
    return () => clearInterval(t)
  }, [loadBank])

  // A pass that ended: say what it produced ONCE, and refresh the gallery it
  // just changed. `announcement` owns both halves of that (see its docstring —
  // the naive version either repeats on every poll or swallows a second
  // identical run).
  const activity = bank?.activity || null
  useEffect(() => {
    const { announce, marker, outcome } = announcement(announced.current, activity)
    announced.current = marker
    if (!announce) return
    if (outcome) toast[outcome.tone](outcome.text)
    loadClips(false)
  }, [activity])                                          // eslint-disable-line react-hooks/exhaustive-deps

  // The grid draws the ranking while a search is on, the filtered page
  // otherwise. Paging is hidden under a search on purpose: the ranking is a
  // fixed top-N, and a "Load more" that re-ran the filter would silently replace
  // the ranking with something ordered by file and start time.
  // The flag filter applies to a ranking too: "which of these results do I
  // already have twice" is a question about the ranking, and a chip row that
  // went inert under a search would be the one place it is most useful.
  const baseClips = search ? (search.clips || []) : clips
  const shownClips = filterByFlag(baseClips, flag)
  const chips = flagChips(baseClips)
  const flagNote = search ? '' : flagFilterNote(clips.length, total)
  const matchLines = search
    ? Object.fromEntries((search.results || []).map((r) => [r.clip_id, matchLine(r)]))
    : null
  const counts = bank?.counts || {}
  const capability = bank?.capability || null
  const busy = isBusy(activity)
  const step = nextStep(counts, capability, passBlockedBy)

  const startPass = async (pass, body = {}) => {
    const blocked = passBlockedBy(capability, pass)
    if (blocked) { toast.warning(blocked.why); return }
    try {
      await postJson(videoPassUrl(bankId, pass),
        pass === 'caption' ? { ...body, style: captionStyle || undefined } : body)
      loadBank(false)
    } catch (e) {
      // 409 carries busy_kind — name the pass that owns the bank rather than
      // repeating "busy", which does not tell you what to wait for.
      if (e?.status === 409 && e.body?.busy_kind) {
        toast.warning(`${passLabel(e.body.busy_kind)} is already running on this bank.`)
      } else {
        toast.error(e?.message || `Could not start ${passLabel(pass)}.`)
      }
    }
  }

  const cancel = async () => {
    try {
      const d = await postJson(videoPassUrl(bankId, 'cancel'), {})
      // `cancelled: false` just means nothing was running — not an error, and
      // certainly not a red toast.
      toast.info(d.cancelled ? 'Stopping…' : 'Nothing was running.')
      loadBank(false)
    } catch (e) {
      toast.error(e?.message || 'Could not stop the pass.')
    }
  }

  const rescan = async () => {
    try {
      const d = await postJson(videoPassUrl(bankId, 'refresh'), {})
      toast.success(d.added
        ? `${d.added} new file(s) inventoried.`
        : 'No new file — the bank already knows this folder.')
      if (d.missing) toast.warning(`${d.missing} file(s) are no longer where the bank left them.`)
      loadBank(false)
    } catch (e) {
      toast.error(e?.message || 'Could not rescan the folder.')
    }
  }

  const triage = async (ids, next) => {
    const body = triagePayload(ids, next)
    // null = empty selection. Posting it would retag EVERY clip in the bank.
    if (!body) { toast.info('Select some shots first.'); return }
    await applyTriage(body, ids.length)
  }

  const triageEverything = async (next) => {
    // eslint-disable-next-line no-alert
    if (!window.confirm(triageAllConfirmation(next, counts.clips || 0))) return
    await applyTriage(triageAllPayload(next), counts.clips || 0)
  }

  const applyTriage = async (body, howMany) => {
    try {
      const d = await postJson(videoPassUrl(bankId, 'triage'), body)
      setBank((b) => (b ? { ...b, counts: d.counts || b.counts } : b))
      const touched = new Set(body.ids?.length ? body.ids : null)
      const retag = (list) => list.map((c) => (
        !body.ids?.length || touched.has(c.id) ? { ...c, status: body.status } : c))
      setClips(retag)
      // The ranked rows are a SEPARATE list; leaving them stale would show a
      // shot the user just kept still wearing its old badge, under a search that
      // is still on screen.
      setSearch((r) => (r ? { ...r, clips: retag(r.clips || []) } : r))
      setSelected([])
      toast.success(`${d.updated ?? howMany} shot(s) → ${body.status}.`)
    } catch (e) {
      toast.error(e?.message || 'Could not save that decision.')
    }
  }

  const onToggle = (id, e) => {
    if (e?.shiftKey && anchor != null) {
      setSelected((s) => selectRange(s, shownClips.map((c) => c.id), anchor, id))
    } else {
      setSelected((s) => toggleSelection(s, id))
    }
    setAnchor(id)
  }

  const openAt = (clip) => setOpenIndex(shownClips.findIndex((c) => c.id === clip.id))
  const openClip = openIndex != null ? shownClips[openIndex] : null
  // The second the search matched inside the OPEN shot, so the player lands on
  // it instead of on the shot's first frame. Null outside a search.
  const openAtSecond = openClip
    ? (search?.results || []).find((r) => r.clip_id === openClip.id)?.frame_s ?? null
    : null
  const triageOpen = async (next) => {
    if (!openClip) return
    await triage([openClip.id], next)
    setOpenIndex((i) => (i != null && i + 1 < shownClips.length ? i + 1 : i))
  }

  /** A shot was re-cut, split, or drawn by hand.
   *
   * The retouched row is swapped IN PLACE rather than triggering a reload: the
   * user is inside the lightbox, `openIndex` addresses the list by position, and
   * a reload would either close the player or move it onto a different shot.
   *
   * A split's new half and a hand-made shot are NOT inserted into the open list.
   * The list is one page of a filter they may not even satisfy (a 'pending' new
   * shot under the 'keep' filter), and quietly inserting a row would shift every
   * index under the player. The counters move immediately, so the grid says three
   * where it said two the next time it is loaded, and the next-step line asks for
   * the thumbnails pass on its own because `counts.thumbs` just fell.
   */
  const onRetouched = (payload, kind) => {
    if (payload?.counts) {
      setBank((b) => (b ? { ...b, counts: payload.counts } : b))
    }
    if (payload?.clip) {
      const patch = (list) => list.map((c) => (
        c.id === payload.clip.id ? { ...c, ...payload.clip } : c))
      setClips(patch)
      // The row is patched, its MATCH is dropped. A re-cut shot loses its search
      // vectors server-side (they described three instants of the old span), so
      // "matched at 12.5 s" under the new bounds would be a claim nothing backs
      // — and the second could now sit outside the shot entirely.
      setSearch((r) => (r ? {
        ...r,
        clips: patch(r.clips || []),
        results: (r.results || []).filter((x) => x.clip_id !== payload.clip.id),
      } : r))
    }
    if (kind !== 'bounds') setPendingRefresh(true)
  }

  /** ✂ The FIRST shot of a file, from the Files list.
   *
   * Reloading the gallery here is right where swapping a row in place was right
   * above: nothing is open, and the new shot has to appear — it is the thing the
   * user is about to click on to trim it.
   */
  const cutByHand = async (src, bounds) => {
    try {
      const d = await postJson(videoSourceClipsUrl(bankId, src.id), bounds)
      toast.success(retouchToast('create'))
      setBank((b) => (b && d.counts ? { ...b, counts: d.counts } : b))
      loadClips(false)
    } catch (e) {
      if (e?.status === 409) {
        toast.warning('A pass is running on this bank — stop it before cutting.')
      } else {
        toast.error(e?.message || 'Could not add that shot.')
      }
    }
  }

  /** ▣ "This file is one take" and ↻ "re-detect this one file".
   *
   * Both replace every shot of ONE file, so both confirm first — and the
   * confirmation for ↻ names the thing the user could not get back: a re-cut of
   * a single file replaces hand-made cuts too. That asymmetry with the
   * bank-wide re-cut is deliberate (it is what makes ↻ the way back from ▣) and
   * it is exactly the kind of asymmetry that has to be said out loud.
   */
  const perSource = async (src, url, question, done) => {
    if (!window.confirm(question)) return
    try {
      const d = await postJson(url, {})
      toast.success(done(d))
      loadBank(false)
      loadClips(false)
    } catch (e) {
      if (e?.status === 409) {
        toast.warning('A pass is running on this bank — stop it first.')
      } else if (e?.status === 503) {
        toast.warning(e?.body?.error || 'This file has no cached measurement yet.')
      } else {
        toast.error(e?.body?.error || e?.message || 'That did not go through.')
      }
    }
  }

  const singleShot = (src) => perSource(
    src, videoSourceSingleShotUrl(bankId, src.id),
    `Replace every shot of ${src.relpath} with one full-length shot?\n\n`
    + 'Shots already promoted into a dataset are kept. Bulk passes will leave '
    + 'this file alone afterwards.',
    () => 'One shot now covers the whole file.')

  const recutSource = (src) => perSource(
    src, videoSourceRecutUrl(bankId, src.id),
    `Find the shots in ${src.relpath} again?\n\n`
    + 'This replaces every shot on this file, INCLUDING any you cut by hand. '
    + 'Shots already promoted into a dataset are kept.',
    (d) => `${d.clips} shots`
      + (d.replaced_manual ? `, replacing ${d.replaced_manual} hand-made.` : '.'))

  if (!bank) return <p className="text-sm text-content-muted">Loading…</p>
  const problems = countsProblems(counts)

  return (
    <div className="space-y-4">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <button type="button" onClick={onBack}
          className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-sm text-content hover:bg-surface">
          ← Banks
        </button>
        <h1 className="min-w-0 truncate text-lg font-bold text-content">🎬 {bank.name}</h1>
        <HelpBadge topic="page-video-bank" />
        <button type="button" onClick={rescan}
          title="Re-walk the folder and inventory anything new"
          className="ml-auto rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs font-semibold text-content hover:bg-surface">
          ↻ Rescan folder
        </button>
      </div>
      <p className="truncate font-mono text-xs text-content-subtle" title={bank.source_path}>
        {bank.source_path}
      </p>

      <VideoCapabilityStrip capability={capability} />

      <p className="text-sm text-content-muted">{countsSummary(counts)}</p>
      {problems.map((p) => (
        <p key={p} className="text-xs text-amber-300">⚠ {p}</p>
      ))}

      {/* What to do next, as ONE sentence. Four equal buttons and no order is
          how a user runs detection before the probe, gets "0 shots" and a green
          success, and concludes the app cannot read their files. */}
      <div className="rounded-lg border border-border bg-surface p-3 text-sm">
        <p className="text-content">{step.text}</p>
        {step.blocked && (
          <p className="mt-1 text-xs text-amber-300">⚠ {step.blocked.why}</p>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {/* ✂ and 🔖 sit AFTER embed on purpose: duplicates reuse the vectors
            that pass caches, so running them before it is the one order that
            produces an honest-looking empty answer. */}
        {/* 🩻 sits last: it is the only pass here that needs the ENCODER rather
            than the decoder, so on an install without ffmpeg it is the one grey
            button in a row of working ones — and the tooltip says which binary
            rather than "unavailable". */}
        {/* 🤖 sits at the end because it is the slowest button in the row —
            about 0.8 s per shot on the CPU, measured — and because it is the
            only one whose result is a hedge rather than a measurement. */}
        {['pipeline', 'probe', 'detect', 'thumbs', 'measure', 'embed',
          'caption', 'dedup', 'watermark', 'safezone', 'defects',
          'aicheck'].map((pass) => {
          const blocked = passBlockedBy(capability, pass)
          const primary = pass === 'pipeline'
          // 🔳 is the one pass that runs with HALF its dependencies: no OCR
          // engine still measures the bands. So it is never disabled for that —
          // the tooltip says what the run will and will not include instead,
          // which is the difference between a working button and a dead one.
          const partial = pass === 'safezone' && capability
            && !capability.video_text
          return (
            <button key={pass} type="button" onClick={() => startPass(pass)}
              disabled={busy || !!blocked}
              title={blocked ? blocked.why
                : partial ? `Bands only — ${capability.video_text_detail
                    || 'the burned-in text extra is not installed'}`
                : undefined}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold disabled:opacity-40 ${
                primary
                  ? 'bg-gradient-primary text-white'
                  : 'border border-border bg-surface-raised text-content hover:bg-surface'}`}>
              {primary ? '▶ ' : ''}{PASS_LABELS[pass]}
            </button>
          )
        })}
        <button type="button" onClick={() => setPromoting(true)}
          disabled={busy || !counts.keep}
          title={!counts.keep ? 'Keep some shots first' : undefined}
          className="rounded-md border border-indigo-500/60 bg-indigo-500/15 px-3 py-1.5 text-xs font-semibold text-indigo-200 hover:bg-indigo-500/25 disabled:opacity-40">
          🎬 {PASS_LABELS.promote}
        </button>
        {busy && (
          <button type="button" onClick={cancel}
            className="rounded-md border border-rose-500/60 bg-rose-500/10 px-3 py-1.5 text-xs font-semibold text-rose-200 hover:bg-rose-500/20">
            ⏹ Stop
          </button>
        )}
      </div>

      {busy && (
        <div role="status" className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-2.5">
          <p className="text-sm text-amber-100">⏳ {activityLine(activity, counts)}</p>
          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-black/30">
            <div className={`h-full bg-amber-400 ${activityPercent(activity, counts) == null ? 'w-1/3 animate-pulse' : ''}`}
              style={activityPercent(activity, counts) == null ? undefined
                : { width: `${activityPercent(activity, counts)}%` }} />
          </div>
          {/* A resumed pass counts only what is LEFT, so it honestly reports "3 of
              117" while most of the bank is already cut. Saying what is kept is
              what makes stopping a one-hour pass feel allowed. */}
          {resumeSafetyNote(activity, counts) && (
            <p className="mt-1.5 text-xs text-amber-200/80">
              ↩ {resumeSafetyNote(activity, counts)}
            </p>
          )}
        </div>
      )}

      {/* The cuts panel sits between the passes and the grid: it only means
          something once Measure has run, and it changes what the grid shows. */}
      {/* Above the quality cuts on purpose: this panel decides which shots
          EXIST, the other decides which of them get flagged. */}
      {(counts.sources || 0) > 0 && (
        <VideoShotCutsPanel bankId={bankId} shotDetect={bank?.shot_detect}
          onChanged={() => { loadBank(false); loadClips(false) }} />
      )}

      {(counts.clips || 0) > 0 && (
        <VideoThresholdsPanel bankId={bankId} saved={bank?.thresholds}
          totalClips={counts.clips || 0} onApplied={() => loadBank(false)} />
      )}

      <details className="rounded-lg border border-border bg-surface">
        <summary className="cursor-pointer px-3 py-2 text-sm font-semibold text-content">
          Files ({counts.sources || 0})
        </summary>
        <div className="border-t border-border p-3">
          <VideoSourceList sources={bank.sources || []} activeSourceId={sourceId}
            onFilter={setSourceId} onCut={cutByHand}
            onSingleShot={singleShot} onRecut={recutSource} />
        </div>
      </details>

      {/* 🗣 The caption pass's own option, next to the button that runs it.
          The prompt decides how plainly a caption describes its footage — more
          than the checkpoint does — so it is a choice, not a constant. */}
      {(counts.clips || 0) > 0 && (bank?.caption_model?.styles || []).length > 1 && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <label htmlFor="video-caption-style" className="font-medium text-content">
            Caption wording
          </label>
          <select id="video-caption-style"
            value={captionStyle || bank?.caption_model?.style || 'standard'}
            onChange={(e) => setCaptionStyle(e.target.value)}
            className="rounded-md border border-border bg-surface-raised px-2 py-1 text-content">
            {(bank?.caption_model?.styles || []).map((s) => (
              <option key={s.key} value={s.key}>{s.label}</option>
            ))}
          </select>
          <span className="min-w-0 flex-1 text-content-subtle">
            {captionStyleLabel(bank?.caption_model?.styles,
              captionStyle || bank?.caption_model?.style || 'standard')}
          </span>
        </div>
      )}

      {/* 🔎 Above the gallery, below the passes: it is a way of LOOKING at the
          shots, not a pass — and what it changes is the grid underneath it. */}
      {(counts.clips || 0) > 0 && (
        <VideoClipSearchBox bankId={bankId} counts={counts} busy={busy}
          result={search} searching={searching}
          onRunPass={() => startPass('embed')}
          onResult={(r, pending) => {
            setSearching(!!pending)
            if (!pending) { setSearch(r); setOpenIndex(null); setSelected([]) }
          }}
          captionModel={bank?.caption_model}
          onClear={() => { setSearch(null); setOpenIndex(null) }} />
      )}

      {/* --- the gallery ------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-1.5">
        {STATUS_FILTERS.map((f) => (
          <button key={f.key} type="button" onClick={() => setStatus(f.key)}
            aria-pressed={status === f.key}
            className={`rounded-full border px-2.5 py-1 text-[0.6875rem] font-semibold transition-colors ${
              status === f.key
                ? 'border-primary/60 bg-primary/15 text-content'
                : 'border-border bg-surface text-content-muted hover:bg-surface-raised'}`}>
            {f.label} ({statusFilterCount(counts, f.key)})
          </button>
        ))}
        {sourceId && (
          <button type="button" onClick={() => setSourceId(null)}
            className="rounded-full border border-indigo-500/60 bg-indigo-500/15 px-2.5 py-1 text-[0.6875rem] font-semibold text-indigo-200">
            one file only ✕
          </button>
        )}
      </div>

      {/* ⚑ The verdicts, as chips you can act on. Every flag in this lane used
          to be a badge you read one shot at a time — which is fine for "too much
          motion" and useless for "you already have this shot", where the whole
          point is rejecting the pile in one gesture. */}
      {chips.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[0.6875rem] font-semibold text-content-muted">⚑ Flagged:</span>
          {chips.map((c) => (
            <button key={c.flag} type="button"
              onClick={() => setFlag((f) => (f === c.flag ? null : c.flag))}
              aria-pressed={flag === c.flag}
              className={`rounded-full border px-2.5 py-1 text-[0.6875rem] font-semibold transition-colors ${
                flag === c.flag
                  ? 'border-amber-400/70 bg-amber-500/20 text-amber-100'
                  : 'border-border bg-surface text-content-muted hover:bg-surface-raised'}`}>
              {c.label} ({c.count})
            </button>
          ))}
          {flag && (
            <button type="button" onClick={() => setFlag(null)}
              className="rounded-full border border-border bg-surface px-2.5 py-1 text-[0.6875rem] text-content-muted hover:bg-surface-raised">
              show all ✕
            </button>
          )}
        </div>
      )}
      {/* The counts cover the LOADED page, and a chip that read like a bank-wide
          total would be a wrong number rather than a filter. */}
      {flagNote && <p className="text-[0.6875rem] text-content-subtle">{flagNote}</p>}

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-content-muted">
          {selected.length
            ? `${selected.length} selected`
            : (flag
              ? `${shownClips.length} flagged`
              : (search ? `${shownClips.length} found` : `${clips.length} of ${total} shown`))}
        </span>
        <button type="button" onClick={() => triage(selected, 'keep')} disabled={!selected.length}
          className="rounded-md bg-emerald-600/80 px-2.5 py-1 font-semibold text-white hover:bg-emerald-600 disabled:opacity-30">
          ✓ Keep
        </button>
        <button type="button" onClick={() => triage(selected, 'reject')} disabled={!selected.length}
          className="rounded-md bg-rose-600/80 px-2.5 py-1 font-semibold text-white hover:bg-rose-600 disabled:opacity-30">
          ✕ Reject
        </button>
        <button type="button" onClick={() => setSelected(shownClips.map((c) => c.id))}
          disabled={!shownClips.length}
          className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-content hover:bg-surface disabled:opacity-30">
          Select page
        </button>
        {selected.length > 0 && (
          <button type="button" onClick={() => setSelected([])}
            className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-content hover:bg-surface">
            Clear
          </button>
        )}
        {/* "Everything" is a separate control on purpose — it is the one action
            here that also hits shots you cannot see. */}
        <button type="button" onClick={() => triageEverything('reject')}
          disabled={!counts.clips}
          className="ml-auto rounded-md border border-border bg-surface-raised px-2.5 py-1 text-content-muted hover:bg-surface disabled:opacity-30">
          Reject all…
        </button>
      </div>

      <VideoClipGrid bankId={bankId} clips={shownClips} selected={selected}
        onToggle={onToggle} onOpen={openAt} matchLines={matchLines}
        emptyMessage={search
          ? `Nothing came back for “${search.query}”.`
          : emptyGridMessage({
            status,
            sourceName: bank.sources?.find((s) => s.id === sourceId)?.relpath,
            counts,
          })} />

      {!search && hasMore({ loaded: clips.length, total }) && (
        <div className="flex justify-center">
          <button type="button" onClick={() => loadClips(true)} disabled={loadingClips}
            className="rounded-md border border-border bg-surface-raised px-4 py-1.5 text-sm font-semibold text-content hover:bg-surface disabled:opacity-40">
            {loadingClips ? 'Loading…' : `Load more (${total - clips.length} left)`}
          </button>
        </div>
      )}

      {openClip && (
        <VideoClipLightbox bankId={bankId} clip={openClip}
          // The SOURCE's own facts, which the clip row does not carry: the file's
          // duration is the wall a bound cannot cross, and `fps_native` is what
          // "one frame" means for a nudge. Reading the TARGET's rate there is
          // exactly the mistake that turns a 16 fps profile into fast motion.
          source={bank.sources?.find((s) => s.id === openClip.source_id) || null}
          onRetouched={onRetouched}
          playFrom={openAtSecond}
          hasPrev={openIndex > 0} hasNext={openIndex < shownClips.length - 1}
          onClose={() => {
            setOpenIndex(null)
            if (pendingRefresh) { setPendingRefresh(false); loadClips(false) }
          }}
          onPrev={() => setOpenIndex((i) => Math.max(0, i - 1))}
          onNext={() => setOpenIndex((i) => Math.min(shownClips.length - 1, i + 1))}
          onKeep={() => triageOpen('keep')}
          onReject={() => triageOpen('reject')} />
      )}

      {promoting && (
        <PromoteVideoDialog bankId={bankId} capability={capability}
          keepCount={counts.keep || 0} selectedIds={selected}
          onClose={() => setPromoting(false)}
          onDone={() => { setSelected([]); loadBank(false) }} />
      )}
    </div>
  )
}
