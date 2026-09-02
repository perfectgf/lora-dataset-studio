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
  countsProblems, activityLine, activityPercent, isBusy,
  resumeSafetyNote,
  announcement, nextStep, passLabel, PASS_LABELS,
} from './videoBankStatus'
import {
  statusFilterCount, toggleSelection, selectRange,
  triagePayload, triageAllPayload, triageAllConfirmation, emptyGridMessage,
  hasMore,
} from './videoTriage'
import {
  burstKeyAction, clipIndex, firstPendingIndex, stepIndex, afterDecision,
  undoEntry, pushUndo, popUndo,
  createQueue, queueDecision, startBatch, finishBatch, queueDepth,
  loadBurstPrefs, saveBurstPrefs,
} from './videoBurstTriage'
import VideoBurstBar from './VideoBurstBar'
import VideoClipGrid from './VideoClipGrid'
import VideoClipLightbox from './VideoClipLightbox'
import VideoFilterRail from './VideoFilterRail'
import VideoPassesPanel from './VideoPassesPanel'
import RunEverythingDialog from './RunEverythingDialog'
import { matchLine } from './videoClipSearch'
import { filterByFlag, flagChips, flagFilterNote } from './videoMetricsFilter'
import { cameraChips, filterByCamera } from './videoCameraMotion'
import PromoteVideoDialog from './PromoteVideoDialog'
import DescribeShotsDialog from './DescribeShotsDialog'
import { GuideInfoDot } from '../common/GuideSectionModal'
import { VIDEO_PASS_TOPICS } from './videoPassTopics'
import { Stat } from '../bank/BankAtoms.jsx'
import {
  loadRailOpen, passesButtonLabel, railIsColumn, saveRailOpen,
} from '../bank/bankLayout.js'

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
  // 🗣 The Describe launch window (wording + redo scope). The pass button opens
  // it; only its own Describe button actually starts the pass.
  const [describing, setDescribing] = useState(false)
  const [search, setSearch] = useState(null)
  const [searching, setSearching] = useState(false)
  // ⚑ Which verdict the grid is narrowed to, or null. Client-side over the shots
  // already loaded — the status filter is a server-side query and this is not,
  // which is a real difference and the reason the chip row carries a note.
  const [flag, setFlag] = useState(null)
  // Its own state and not a second value of `flag`: the two filters COMPOSE —
  // "shots I flagged as shaky that also pan right" is a real question, and one
  // shared slot would make picking either clear the other.
  const [camera, setCamera] = useState(null)
  // The last job we announced, so a finished pass is toasted ONCE instead of on
  // every poll for as long as the server keeps its snapshot.
  const announced = useRef(null)
  // ⌨ Burst mode — one keystroke per shot on the grid. The mode and the
  // auto-advance are screen preferences and survive a reload; the cursor,
  // the undo net and the send queue are the run itself and do not.
  const [burst, setBurst] = useState(() => loadBurstPrefs())
  const [cursorId, setCursorId] = useState(null)
  const [undoStack, setUndoStack] = useState([])
  const [helpOpen, setHelpOpen] = useState(false)
  // ── The Encre shell, shared with the image lane. bankLayout.js makes every
  // decision (when the rail is a column, the persisted preference, what the ⚙
  // button says) so the two lanes cannot drift apart on it again.
  const viewportWidth = () => (typeof window === 'undefined' ? undefined : window.innerWidth)
  const [railIsColumnNow, setRailIsColumnNow] = useState(() => railIsColumn(viewportWidth()))
  const [railOpen, setRailOpen] = useState(() => loadRailOpen(viewportWidth()))
  const [passesOpen, setPassesOpen] = useState(false)
  // ▶ The pipeline's launch window: which preparation passes the chain runs.
  const [runningAll, setRunningAll] = useState(false)
  const passesAutoOpened = useRef(false)
  useEffect(() => {
    const onResize = () => setRailIsColumnNow(railIsColumn(window.innerWidth))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  const setRail = (open) => {
    setRailOpen(open)
    if (railIsColumn(viewportWidth())) saveRailOpen(open)
  }
  // Decisions typed but not yet acknowledged by the server. A ref and not state
  // because a burst run mutates it faster than React re-renders, and the count
  // that the bar shows is derived from it after every step.
  const queue = useRef(createQueue())
  const flushing = useRef(false)
  const [saving, setSaving] = useState(0)

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
    setSelected([]); setAnchor(null); setSearch(null); setOpenIndex(null)
    setFlag(null); setCamera(null)
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
  // The two filters compose, flag first. The camera CHIPS are counted over the
  // flag-filtered set rather than over `shownClips`, so picking one does not
  // make the other ten vanish — the same reason `chips` counts over `baseClips`.
  const flagged = filterByFlag(baseClips, flag)
  const shownClips = filterByCamera(flagged, camera)
  const cameraOptions = cameraChips(flagged)
  const chips = flagChips(baseClips)
  const flagNote = search ? '' : flagFilterNote(clips.length, total)
  const matchLines = search
    ? Object.fromEntries((search.results || []).map((r) => [r.clip_id, matchLine(r)]))
    : null
  const counts = bank?.counts || {}
  // An empty bank's next move lives in the passes panel, so it starts open —
  // once, on arrival; the image lane makes the same call on its own panel.
  useEffect(() => {
    if (!bank || passesAutoOpened.current) return
    passesAutoOpened.current = true
    if (!(counts.clips > 0)) setPassesOpen(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bank])
  const capability = bank?.capability || null
  const busy = isBusy(activity)
  const step = nextStep(counts, capability, passBlockedBy)

  const startPass = async (pass, body = {}) => {
    const blocked = passBlockedBy(capability, pass)
    if (blocked) { toast.warning(blocked.why); return }
    try {
      await postJson(videoPassUrl(bankId, pass),
        // An explicit style in the body (the launch window's) wins over the
        // remembered one: setCaptionStyle is async and the remembered value is
        // one render behind at exactly the moment the window launches.
        pass === 'caption' ? { style: captionStyle || undefined, ...body } : body)
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

  /* ── ⌨ Burst mode ─────────────────────────────────────────────────────────
     The keyboard run. Every decision is applied to the rows IMMEDIATELY and
     sent afterwards — the hand never waits for the network, which is the whole
     measured gain — and the queue in videoBurstTriage keeps exactly one request
     in flight so a fast run cannot land its decisions out of order.

     Every body still goes through `triagePayload`, which refuses an empty id
     list. That is not ceremony: an empty `ids` means EVERY CLIP IN THE BANK on
     this endpoint, and a burst run posts constantly. */

  /* The rows and the cursor as the KEY HANDLER must see them.
   *
   * Measured, on this screen: twenty-four keydowns delivered inside ONE task
   * landed one decision and swallowed twenty-three — every handler after the
   * first read the same render-time `shownClips` and the same `cursorId`, and
   * re-decided the shot that was already decided. A real keyboard cannot do
   * that (each keydown is its own task and React commits in between — the same
   * twenty-four keys at 190/s all landed), but anything that dispatches events
   * in a loop can, and "the second key did nothing" is not a failure anyone
   * would think to look for.
   *
   * So the handler reads this ref instead. It is refreshed on every render —
   * the derive-during-render pattern VideoClipLightbox already uses for its
   * player key — and each decision writes the new truth into it straight away,
   * so between two renders the ref, not the stale closure, is the authority. */
  const live = useRef({ clips: [], cursorId: null })
  live.current = { clips: shownClips, cursorId }

  const retagRow = useCallback((id, status) => {
    const retag = (rows) => rows.map((c) => (c.id === id ? { ...c, status } : c))
    setClips(retag)
    setSearch((r) => (r ? { ...r, clips: retag(r.clips || []) } : r))
  }, [])

  /** Send what is queued, one request at a time, until the queue is empty. */
  const flushBurst = useCallback(async () => {
    if (flushing.current) return
    flushing.current = true
    try {
      for (;;) {
        queue.current = startBatch(queue.current)
        const batch = queue.current.inflight
        if (!batch) break
        const body = triagePayload(batch.ids, batch.status)
        if (!body) { queue.current = finishBatch(queue.current); continue }
        try {
          const d = await postJson(videoPassUrl(bankId, 'triage'), body)
          setBank((b) => (b ? { ...b, counts: d.counts || b.counts } : b))
          queue.current = finishBatch(queue.current)
        } catch (e) {
          // A failed batch makes the rows on screen a claim nothing backs, and
          // it makes the undo net describe a state that never existed. So we do
          // not guess our way back: the grid is reloaded from the bank, the net
          // is dropped, and the message says how many decisions did not land.
          const lost = queueDepth(queue.current)
          queue.current = createQueue()
          setUndoStack([])
          toast.error(`${lost} decision(s) did not save — the grid now shows what the `
            + `bank actually holds. (${e?.message || 'the request failed'})`)
          loadClips(false)
          break
        }
        setSaving(queueDepth(queue.current))
      }
    } finally {
      flushing.current = false
      setSaving(queueDepth(queue.current))
    }
  }, [bankId, toast, loadClips])

  /** Move the cursor in BOTH the ref the handler reads and the state the grid
   * draws, so a second keystroke in the same task sees where the first left it. */
  const placeCursor = useCallback((id) => {
    live.current.cursorId = id ?? null
    setCursorId(id ?? null)
  }, [])

  const burstDecide = useCallback((status) => {
    const rows = live.current.clips
    const at = clipIndex(rows, live.current.cursorId)
    const clip = at >= 0 ? rows[at] : null
    if (!clip) return
    const before = clip.status || 'pending'
    // Pressing K on an already-kept shot must still MOVE — a key that does
    // nothing reads as a dropped keystroke — but it owes the server nothing and
    // it is not something to offer an undo for.
    if (before !== status) {
      setUndoStack((s) => pushUndo(s, undoEntry(clip, status)))
      retagRow(clip.id, status)
      queue.current = queueDecision(queue.current, clip.id, status)
      setSaving(queueDepth(queue.current))
      flushBurst()
    }
    const next = rows.map((c) => (c.id === clip.id ? { ...c, status } : c))
    live.current.clips = next
    const to = afterDecision({ clips: next, index: at, autoAdvance: burst.autoAdvance })
    placeCursor(next[to]?.id ?? clip.id)
  }, [burst.autoAdvance, retagRow, flushBurst, placeCursor])

  const burstUndo = useCallback(() => {
    const { entry, stack } = popUndo(undoStack)
    if (!entry) return
    setUndoStack(stack)
    retagRow(entry.id, entry.from)
    live.current.clips = live.current.clips.map((c) => (
      c.id === entry.id ? { ...c, status: entry.from } : c))
    // The cursor goes ONTO the shot that was put back. An undo you cannot see
    // is indistinguishable from an undo that did nothing.
    placeCursor(entry.id)
    queue.current = queueDecision(queue.current, entry.id, entry.from)
    setSaving(queueDepth(queue.current))
    flushBurst()
  }, [undoStack, retagRow, flushBurst, placeCursor])

  const burstMove = useCallback((delta) => {
    const rows = live.current.clips
    const at = clipIndex(rows, live.current.cursorId)
    const to = stepIndex(rows, at < 0 ? 0 : at, delta)
    if (to >= 0) placeCursor(rows[to].id)
  }, [placeCursor])

  const burstFirst = useCallback(() => {
    const rows = live.current.clips
    const i = firstPendingIndex(rows)
    if (i >= 0) placeCursor(rows[i].id)
  }, [placeCursor])

  const setBurstPref = useCallback((patch) => {
    setBurst((b) => saveBurstPrefs({ ...b, ...patch }))
  }, [])

  // Where the cursor sits. Re-resolved by ID rather than by position, because a
  // filter change replaces the whole list — and kept where it is as long as the
  // shot is still on screen, which is what makes a decision not move it.
  useEffect(() => {
    if (!burst.on) {
      if (cursorId !== null) placeCursor(null)
      return
    }
    if (cursorId != null && shownClips.some((c) => c.id === cursorId)) return
    const i = firstPendingIndex(shownClips)
    placeCursor(shownClips[i >= 0 ? i : 0]?.id ?? null)
  }, [burst.on, shownClips, cursorId, placeCursor])

  useEffect(() => {
    if (!burst.on) return undefined
    const onKey = (e) => {
      // The player and the promote dialog own the keyboard while they are open:
      // the lightbox has its own K/R/←/→ over the shot it is showing, and a
      // second handler would decide TWO different shots on one keystroke.
      if (openIndex != null || promoting) return
      const action = burstKeyAction(e)
      if (!action) return
      e.preventDefault()
      // 'keep' | 'reject' | 'pending' ARE the three triage statuses, by name.
      if (action === 'keep' || action === 'reject' || action === 'pending') burstDecide(action)
      else if (action === 'skip') burstMove(1)
      else if (action === 'back') burstMove(-1)
      else if (action === 'undo') burstUndo()
      else if (action === 'first') burstFirst()
      else if (action === 'help') setHelpOpen((v) => !v)
      else if (action === 'exit') setBurstPref({ on: false })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [burst.on, openIndex, promoting, burstDecide, burstMove, burstUndo, burstFirst,
    setBurstPref])

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
    + 'Shots whose bounds do not change keep their triage and captions. The '
    + 'others are replaced, INCLUDING any you cut by hand. Shots already '
    + 'promoted into a dataset are kept.',
    (d) => `${d.clips} shot(s)`
      + (d.kept ? `, ${d.kept} unchanged (triage and captions kept)` : '')
      + (d.replaced_manual ? `, replacing ${d.replaced_manual} hand-made.` : '.'))

  if (!bank) return <p className="text-sm text-content-muted">Loading…</p>
  const problems = countsProblems(counts)

  return (
    <div className="space-y-4">
      <header data-probe-chrome="header"
        className="space-y-2 rounded-xl border border-border bg-surface p-3 [@media(max-height:500px)]:space-y-1 [@media(max-height:500px)]:p-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <button type="button" onClick={onBack}
            className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-2.5 py-1 text-sm text-content hover:bg-surface">
            ← Banks
          </button>
          <h1 className="min-w-0 truncate text-lg font-bold text-content">🎬 {bank.name}</h1>
          <HelpBadge topic="page-video-bank" />
          <button type="button" onClick={rescan}
            title="Re-walk the folder and inventory anything new"
            className="min-h-10 lg:min-h-0 ml-auto rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs font-semibold text-content hover:bg-surface">
            ↻ Rescan folder
          </button>
        </div>
        {/* The path is desktop reading — on a phone it was a quarter of the
            fold saying what the title already names. */}
        <p className="hidden truncate font-mono text-xs text-content-subtle lg:block" title={bank.source_path}>
          {bank.source_path}
        </p>
        {problems.map((p) => (
          <p key={p} className="text-xs text-amber-300">⚠ {p}</p>
        ))}
        {/* The strip a bank is read by — the same Stat atom and the same order
            of concern as the image lane: what exists, then how triage stands. */}
        <div className="flex flex-nowrap items-baseline gap-x-4 gap-y-1 overflow-x-auto border-t border-border pt-2 text-sm sm:flex-wrap sm:overflow-visible [@media(max-height:500px)]:hidden">
          <Stat label="files" value={counts.sources || 0} />
          <Stat label="shots" value={counts.clips || 0} />
          <Stat label="to triage" value={statusFilterCount(counts, 'pending')} />
          <Stat label="kept" value={statusFilterCount(counts, 'keep')} tone="emerald" />
          <Stat label="rejected" value={statusFilterCount(counts, 'reject')} tone="rose" />
        </div>
        {/* The decisive actions. ⚙ Passes opens the analysis panel; ▶ and 🎬
            are the two that change what leaves this bank — top-bar residents,
            like the image lane's Launch all and Promote. */}
        <div className="flex min-w-0 flex-nowrap items-center gap-2 overflow-x-auto border-t border-border pt-2 sm:flex-wrap sm:overflow-visible [@media(max-height:500px)]:border-t-0 [@media(max-height:500px)]:pt-0">
          {/* ☰ exists only where the rail cannot sit beside the grid — there it
              is the ONLY way back to the filters, so it is a real button and
              never a CSS-hidden one. */}
          {!railIsColumnNow && (
            <button type="button" onClick={() => setRail(true)}
              aria-expanded={railOpen} aria-controls="video-filter-rail"
              className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content hover:bg-surface">
              ☰ Filters
            </button>
          )}
          <button type="button" onClick={() => setPassesOpen((v) => !v)}
            aria-expanded={passesOpen} aria-controls="video-passes-panel"
            title="Open the analysis passes — probe, find shots, thumbnails, measure, embeddings, captions, duplicates, watermarks, safe zones, defects, camera and AI check."
            className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content hover:bg-surface">
            {passesButtonLabel(busy)}
          </button>
          <span className="inline-flex items-center gap-1">
            <button type="button" onClick={() => setRunningAll(true)}
              disabled={busy || !!passBlockedBy(capability, 'pipeline')}
              title={passBlockedBy(capability, 'pipeline')?.why
                || 'Chain the preparation passes — scan, find shots, thumbnails, and whichever of measure, embeddings, duplicates and camera you tick. Start it and walk away.'}
              className="min-h-10 lg:min-h-0 rounded-md bg-gradient-primary px-4 py-2 text-sm font-bold text-gray-950 shadow disabled:opacity-50">
              ▶ {PASS_LABELS.pipeline}
            </button>
            <GuideInfoDot topic={VIDEO_PASS_TOPICS.pipeline} label={PASS_LABELS.pipeline} />
          </span>
          <span className="ml-auto" />
          <span className="inline-flex items-center gap-1">
            <button type="button" onClick={() => setPromoting(true)}
              disabled={busy || !counts.keep}
              title={!counts.keep ? 'Keep some shots first' : undefined}
              className="min-h-10 lg:min-h-0 rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50">
              🎬 {PASS_LABELS.promote}
            </button>
            <GuideInfoDot topic={VIDEO_PASS_TOPICS.promote} label={PASS_LABELS.promote} />
          </span>
          {busy && (
            <button type="button" onClick={cancel}
              className="min-h-10 lg:min-h-0 rounded-md border border-rose-500/60 bg-rose-500/10 px-3 py-1.5 text-xs font-semibold text-rose-200 hover:bg-rose-500/20">
              ⏹ Stop
            </button>
          )}
        </div>
      </header>

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

      {passesOpen && (
        /* data-probe-reading: the panel is what you asked for when you pressed
           ⚙, one tap puts it away, and nothing in it is used against the grid
           behind it — the same charging rule as the image lane's panel. Plain
           comment: this is an EXPRESSION position. */
        <div id="video-passes-panel" data-probe-chrome="passes" data-probe-panel="passes" data-probe-reading>
          <VideoPassesPanel bankId={bankId} bank={bank} counts={counts}
            capability={capability} step={step} busy={busy}
            startPass={startPass} onDescribe={() => setDescribing(true)}
            onCutsChanged={() => { loadBank(false); loadClips(false) }} />
        </div>
      )}

      {/* ── The rail, beside the grid it filters ─────────────────────────
          Below the column width it becomes a drawer OVER the grid instead of
          squeezing it — bankLayout.js decides, for both lanes at once. */}
      <div className={`grid gap-3 ${railOpen && railIsColumnNow
        ? 'lg:grid-cols-[17rem_minmax(0,1fr)]' : 'grid-cols-1'}`}>
        {railOpen && !railIsColumnNow && (
          <div className="fixed inset-0 z-40 bg-black/60" onClick={() => setRail(false)} aria-hidden />
        )}
        {railOpen && (
          /* Sticky only as a column: the rail is ~600 px and the gallery is
             thousands — unpinned, it scrolls away after one screen and the
             round trip this layout removes comes straight back. As a drawer it
             is `fixed` already. Plain comment: EXPRESSION position. */
          <div id="video-filter-rail"
            className={railIsColumnNow
              ? 'min-w-0 lg:sticky lg:top-[calc(var(--app-header-h)+0.75rem)] lg:max-h-[calc(100vh-var(--app-header-h)-1.5rem)] lg:overflow-y-auto lg:self-start'
              : 'min-w-0'}>
            <VideoFilterRail bankId={bankId} isDrawer={!railIsColumnNow}
              onClose={() => setRail(false)}
              counts={counts} status={status} setStatus={setStatus}
              sourceId={sourceId} setSourceId={setSourceId}
              sources={bank.sources || []}
              chips={chips} flag={flag} setFlag={setFlag} flagNote={flagNote}
              cameraOptions={cameraOptions} camera={camera} setCamera={setCamera}
              busy={busy} search={search} searching={searching}
              captionModel={bank?.caption_model}
              onRunEmbed={() => startPass('embed')}
              onSearchResult={(r, pending) => {
                setSearching(!!pending)
                if (!pending) { setSearch(r); setOpenIndex(null); setSelected([]) }
              }}
              onSearchClear={() => { setSearch(null); setOpenIndex(null) }}
              thresholds={bank?.thresholds} totalClips={counts.clips || 0}
              onThresholdsApplied={() => loadBank(false)}
              onCut={cutByHand} onSingleShot={singleShot} onRecut={recutSource} />
          </div>
        )}
        <div className="min-w-0 space-y-3">

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-content-muted">
          {/* The camera branch comes FIRST, and it is a fix rather than an
              ordering preference: with only a camera chip pressed this line fell
              through to "9 of 9 shown" while the grid was showing one tile.
              A count that contradicts the grid is worse than no count. It also
              wins when BOTH filters are on, because `shownClips` is the composed
              result and "flagged" would name only half of what narrowed it. */}
          {selected.length
            ? `${selected.length} selected`
            : (camera
              ? `${shownClips.length} shown`
              : (flag
                ? `${shownClips.length} flagged`
                : (search ? `${shownClips.length} found` : `${clips.length} of ${total} shown`)))}
        </span>
        <button type="button" onClick={() => triage(selected, 'keep')} disabled={!selected.length}
          className="rounded-md bg-emerald-600/80 px-2.5 py-1 font-semibold text-white hover:bg-emerald-600 disabled:opacity-30">
          ✓ Keep
        </button>
        <button type="button" onClick={() => triage(selected, 'reject')} disabled={!selected.length}
          className="rounded-md bg-rose-600/80 px-2.5 py-1 font-semibold text-white hover:bg-rose-600 disabled:opacity-30">
          ✕ Reject
        </button>
        {/* The third verb of the same endpoint. Without it a mis-kept shot could
            only move to the OTHER decision, never back to undecided — the image
            bank has had this exit all along, and a decision you cannot take back
            is exactly what makes people afraid to triage fast. */}
        <button type="button" onClick={() => triage(selected, 'pending')} disabled={!selected.length}
          className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-content hover:bg-surface disabled:opacity-30">
          ↩ To triage
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

      {/* ⌨ Directly above the grid it drives — the cursor it moves is a tile
          down there, and a control for it anywhere else would be a control for
          something off screen. */}
      {(counts.clips || 0) > 0 && (
        <VideoBurstBar on={burst.on} autoAdvance={burst.autoAdvance}
          clips={shownClips} index={clipIndex(shownClips, cursorId)}
          hasMore={!search && hasMore({ loaded: clips.length, total })}
          undoStack={undoStack} saving={saving}
          helpOpen={helpOpen} onHelp={() => setHelpOpen((v) => !v)}
          onToggle={() => setBurstPref({ on: !burst.on })}
          onAutoAdvance={(v) => setBurstPref({ autoAdvance: v })}
          onUndo={burstUndo} />
      )}

      <VideoClipGrid bankId={bankId} clips={shownClips} selected={selected}
        onToggle={onToggle} onOpen={openAt} matchLines={matchLines}
        cursorId={burst.on ? cursorId : null}
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

        </div>
      </div>

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

      {runningAll && (
        <RunEverythingDialog capability={capability}
          onClose={() => setRunningAll(false)}
          onLaunch={async (steps) => {
            await startPass('pipeline', { steps })
            setRunningAll(false)
          }} />
      )}

      {describing && (
        <DescribeShotsDialog captionModel={bank?.caption_model}
          initialStyle={captionStyle}
          onClose={() => setDescribing(false)}
          onLaunch={(opts) => {
            setDescribing(false)
            // Remember the wording for the next open; the pass itself reads the
            // explicit opts, not this state (one render behind by design).
            setCaptionStyle(opts.style)
            startPass('caption', opts)
          }} />
      )}
    </div>
  )
}
