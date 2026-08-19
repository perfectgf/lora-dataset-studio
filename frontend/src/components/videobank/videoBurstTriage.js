/** ⌨ Burst triage — one keystroke per shot, on the GRID.
 *
 * The lightbox already judged shots one at a time (K/R/←/→/Esc). This is the
 * same job without the round trip: the grid stays on screen, a cursor sits on
 * one tile, and a single key decides it and moves on. Three gestures per shot
 * (click the tile, click ✓ or ✕, come back) become one.
 *
 * PURE: no JSX, no fetch, no DOM. `node --test` cannot parse JSX, so everything
 * worth being sure about — which keystroke means what, where the cursor lands,
 * what a fast run does to a slow network — lives here and the components stay
 * renderers.
 *
 * ── The four decisions this file makes, and why ──────────────────────────────
 *
 * 1. THE KEYS COME FROM THE SHARED GRAMMAR. `../shared/reviewShortcuts.js` is
 *    the one place K keep · R reject · S skip · ← back · Esc is written down;
 *    the Bank's ▶ Review and the dataset lightbox already read it. A third
 *    hand-written copy is how ✓ ends up on K in one screen and on Enter in the
 *    next. This lane adds only what it genuinely owns: P (back to untriaged —
 *    the video bank has three statuses where the image review has two), U
 *    (undo), Home (jump to the first untriaged) and ? (the shortcut panel).
 *
 * 2. AUTOMATIC MOVEMENT IS SMART, EXPLICIT MOVEMENT IS LITERAL. After a
 *    decision the cursor jumps to the next UNTRIAGED shot — on a half-triaged
 *    bank that is half the gain, and it is the difference between a burst run
 *    and a walk past everything you already judged. But ← and → move by exactly
 *    one tile, decided or not, because a user who presses an arrow is asking to
 *    go THERE. Same doctrine as bankReview.back: navigation the user typed is
 *    never rerouted.
 *
 * 3. THE RUN DOES NOT WRAP. When nothing untriaged is left ahead, the cursor
 *    STAYS on the shot just decided and the bar says so. A silent wrap to the
 *    top re-proposes shots in an order the user cannot predict, and at one
 *    keystroke a second that is how a decision lands on the wrong shot. Home is
 *    the explicit way back to the first untriaged one.
 *
 * 4. THE QUEUE IS OPTIMISTIC AND SINGLE-FLIGHT. See the queue section below —
 *    it is the part that decides what happens when a user out-types the network.
 */

import { ownsTypedKeys, reviewKeyAction } from '../shared/reviewShortcuts.js'
import { TRIAGE_STATUSES } from './videoTriage.js'
import { clipLabel } from './videoClipFragment.js'

/* ── Keys ─────────────────────────────────────────────────────────────────── */

/** What the ? panel prints, and the only list of shortcuts in this lane. The
 * bar's one-line hint is derived from it, so what the UI PROMISES cannot drift
 * from what the handler does. */
export const BURST_SHORTCUTS = [
  { keys: 'K', what: 'Keep this shot' },
  { keys: 'R', what: 'Reject this shot' },
  { keys: 'P', what: 'Put it back to untriaged' },
  { keys: 'S  or  →', what: 'Move on without deciding' },
  { keys: '←', what: 'Move back one shot' },
  { keys: 'U', what: 'Undo the last decision (and go to that shot)' },
  { keys: 'Home', what: 'Jump to the first untriaged shot' },
  { keys: '?', what: 'Show or hide this panel' },
  { keys: 'Esc', what: 'Leave burst mode' },
]

/** The one-liner under the burst bar. Short enough for 400 px. */
export const BURST_HINT = 'K keep · R reject · P untriaged · S skip · U undo · ? help'

/**
 * What this keystroke means in burst mode:
 * 'keep' | 'reject' | 'pending' | 'skip' | 'back' | 'undo' | 'first' | 'help'
 * | 'exit' | null.
 *
 * `null` is "not mine" — the caller leaves the event alone, which is what keeps
 * ⌘R a reload and ⇧→ a text selection.
 *
 * The typing guard is `ownsTypedKeys` and not a blanket "is it an input": this
 * screen is full of checkboxes and range sliders (the quality cuts, the flag
 * chips), and a blanket guard once killed K/R/S outright in the image Bank
 * because the focus had landed on a checkbox. Only real text entry — a text
 * input, a textarea, a select, a contenteditable — eats letters.
 *
 * Escape answers before the typing guard, deliberately: a user who has clicked
 * into the search field must always be one Escape from being out of the mode.
 */
export function burstKeyAction(event) {
  if (!event) return null
  if (event.metaKey || event.ctrlKey || event.altKey) return null
  if (event.key === 'Escape') return 'exit'
  // '?' is Shift+/ on most layouts, and the shared grammar refuses every
  // shifted keystroke — so it is read here, before delegating.
  if (event.key === '?') return ownsTypedKeys(event.target) ? null : 'help'
  const shared = reviewKeyAction(event)
  if (shared) return shared === 'close' ? 'exit' : shared
  if (event.shiftKey || ownsTypedKeys(event.target)) return null
  const key = typeof event.key === 'string' ? event.key : ''
  const letter = key.toLowerCase()
  if (letter === 'p') return 'pending'
  if (letter === 'u') return 'undo'
  if (key === 'Home') return 'first'
  return null
}

/* ── The cursor ───────────────────────────────────────────────────────────────
   Everything below reads the clip rows the grid is CURRENTLY showing. That list
   is stable during a run — a triage decision is patched into the rows in place
   rather than triggering a reload (VideoBankWorkspace.applyTriage), so a shot
   does not vanish from under the cursor the moment it is judged, even under the
   "To triage" filter. That stability is what lets a position be a position. */

const isPending = (clip) => (clip?.status || 'pending') === 'pending'

/** Where a clip id sits in the shown rows, or -1. */
export function clipIndex(clips, id) {
  if (!Array.isArray(clips) || id == null) return -1
  return clips.findIndex((c) => c?.id === id)
}

/** The next untriaged shot STRICTLY after `from`, or -1. `from` below zero
 * scans the whole list, which is what "the first untriaged one" means. */
export function nextPendingIndex(clips, from) {
  if (!Array.isArray(clips)) return -1
  const start = Number.isFinite(from) ? Math.max(-1, from) : -1
  for (let i = start + 1; i < clips.length; i += 1) {
    if (isPending(clips[i])) return i
  }
  return -1
}

/** The first untriaged shot on the page, or -1 — what Home lands on. */
export function firstPendingIndex(clips) {
  return nextPendingIndex(clips, -1)
}

/** ← and →: exactly one tile, clamped at both ends, decided or not. */
export function stepIndex(clips, index, delta) {
  const list = Array.isArray(clips) ? clips : []
  if (!list.length) return -1
  const at = Number.isFinite(index) ? index : 0
  return Math.min(list.length - 1, Math.max(0, at + (Number(delta) || 0)))
}

/**
 * Where the cursor goes once a decision has been applied to the rows.
 *
 * `clips` must ALREADY carry the new status — the caller patches its state
 * optimistically and then asks this, so "the next untriaged" cannot answer with
 * the shot that was just judged.
 *
 * With auto-advance off the cursor does not move at all: pressing K then R on
 * the same shot is the correction gesture, and a cursor that ran away would
 * make the second key land on the neighbour.
 */
export function afterDecision({ clips, index, autoAdvance = true }) {
  const list = Array.isArray(clips) ? clips : []
  if (!list.length) return -1
  const at = Math.min(Math.max(Number.isFinite(index) ? index : 0, 0), list.length - 1)
  if (!autoAdvance) return at
  const next = nextPendingIndex(list, at)
  return next >= 0 ? next : at
}

/** How much is left, over the rows on screen. */
export function burstTally(clips) {
  const list = Array.isArray(clips) ? clips : []
  const pending = list.filter(isPending).length
  return { total: list.length, pending, triaged: list.length - pending }
}

/** The counter in the bar: "14 / 120 triaged · 106 left". */
export function burstProgressLine(clips) {
  const { total, pending, triaged } = burstTally(clips)
  if (!total) return 'Nothing to triage here.'
  return `${triaged} / ${total} triaged · ${pending} left`
}

/**
 * The end-of-run sentence, or null while there is still work ahead of the
 * cursor. Three different endings, because they call for three different next
 * moves and one vague "done" would hide two of them.
 */
export function burstEndNote({ clips, index, hasMore = false }) {
  const list = Array.isArray(clips) ? clips : []
  const { total, pending } = burstTally(list)
  if (!total) return 'No shot on this page to triage.'
  if (pending === 0) {
    return hasMore
      ? '✔ Every shot loaded here is triaged — Load more to keep going.'
      : '✔ Every shot on this page is triaged.'
  }
  if (nextPendingIndex(list, index) >= 0) return null
  return `End of the page — ${pending} untriaged shot${pending === 1 ? '' : 's'} `
    + 'sit before the cursor. Home goes back to the first one.'
}

/* ── Undo ─────────────────────────────────────────────────────────────────────
   The brief's real question: at one keystroke a second, how far back should a
   single press reach? A stack that unwinds twenty decisions is as dangerous as
   no undo at all, because by the twentieth you no longer know what is being put
   back.

   The answer here is a BOUNDED stack whose every step is addressed: each entry
   remembers the shot AND the status it had before, pressing U reverts exactly
   one step and MOVES THE CURSOR ONTO that shot, and the bar always names the
   step it is offering. So three fast mistakes are three presses of U, each one
   showing you what it just fixed rather than silently rewinding a run.

   And the offer is a LINE IN THE BAR, not a toast that fades. A four-second
   toast under a one-keystroke-a-second burst is a stroboscope: it is replaced
   before it is read, and the one time it matters it has already gone. The bar
   is always there, never steals focus, and costs no timer. */

/** How many steps back the net reaches. Ten is about what a user can still
 * account for; past that "undo" is really "reload the page". */
export const UNDO_DEPTH = 10

const VERB = { keep: '✓ Keep', reject: '✕ Reject', pending: '↺ Untriaged' }

/** One reversible step. `from` is what the shot was BEFORE, which is what U
 * restores — reverting a Reject on an already-kept shot must put the Keep back,
 * not blank it to untriaged. */
export function undoEntry(clip, to) {
  if (!clip || clip.id == null || !TRIAGE_STATUSES.includes(to)) return null
  const from = TRIAGE_STATUSES.includes(clip.status) ? clip.status : 'pending'
  return { id: clip.id, from, to, label: clipLabel(clip.start_s, clip.end_s) }
}

/** Push, keeping the newest UNDO_DEPTH. A null entry changes nothing. */
export function pushUndo(stack, entry) {
  const list = Array.isArray(stack) ? stack : []
  if (!entry) return list
  return [...list, entry].slice(-UNDO_DEPTH)
}

/** The top step and the stack without it. */
export function popUndo(stack) {
  const list = Array.isArray(stack) ? stack : []
  if (!list.length) return { entry: null, stack: list }
  return { entry: list[list.length - 1], stack: list.slice(0, -1) }
}

/** What the bar offers, or null. Says WHAT it takes back and HOW FAR the net
 * still reaches — an undo that does not say how many steps it has left is one
 * nobody dares lean on. */
export function undoLine(stack) {
  const list = Array.isArray(stack) ? stack : []
  if (!list.length) return null
  const top = list[list.length - 1]
  const rest = list.length - 1
  return `↩ U undoes ${VERB[top.to] || top.to} on ${top.label}`
    + (rest ? ` · ${rest} more step${rest === 1 ? '' : 's'} back` : '')
}

/* ── The queue ────────────────────────────────────────────────────────────────
   What happens when the user out-types the network — the question that decides
   whether a burst run is trustworthy.

   NOT chosen: firing one request per keystroke. Twenty keys in twenty seconds
   is twenty overlapping POSTs whose completion order is the network's business,
   and a Keep that lands after the Reject that replaced it is a lost decision
   with no error anywhere.

   NOT chosen: a lock that swallows keystrokes while a request is in flight. The
   whole gain is that the hand never waits.

   CHOSEN: apply the decision to the rows immediately (the tile flips at once,
   the cursor moves at once), enqueue it, and let ONE request be in flight at a
   time. A run of identical decisions collapses into a single batch, so twenty
   R's in a row cost one POST rather than twenty. Re-deciding the same shot
   drops the earlier queued entry, and the serialisation is what makes that safe:
   an entry already in flight lands before the one that replaces it, so the last
   key pressed is the status the server ends up holding.

   The bodies are still built by videoTriage.triagePayload, which REFUSES an
   empty id list — the one footgun of this API, and the reason a batch is never
   assembled here as a bare `{ids, status}` the caller could post as-is. */

export function createQueue() {
  return { pending: [], inflight: null }
}

/** Enqueue one decision. Re-deciding a shot that is still waiting replaces its
 * queued entry rather than adding a second one. */
export function queueDecision(queue, id, status) {
  const q = queue || createQueue()
  if (id == null || !TRIAGE_STATUSES.includes(status)) return q
  return {
    ...q,
    pending: [...q.pending.filter((e) => e.id !== id), { id, status }],
  }
}

/**
 * Take the next batch: the LEADING run of same-status entries, moved into
 * flight. Answers the queue unchanged when something is already in flight or
 * there is nothing to send — so the caller's flush loop is a plain
 * "start, await, finish, start again".
 *
 * A batch is never empty: `startBatch` only ever builds one from entries that
 * exist, which is what stops an empty `ids` from ever reaching the server's
 * "empty means EVERY clip in the bank" convention.
 */
export function startBatch(queue) {
  const q = queue || createQueue()
  if (q.inflight || !q.pending.length) return q
  const { status } = q.pending[0]
  let i = 0
  const ids = []
  while (i < q.pending.length && q.pending[i].status === status) {
    ids.push(q.pending[i].id)
    i += 1
  }
  return { pending: q.pending.slice(i), inflight: { ids, status } }
}

/** The flight is over (landed or failed — the caller owns the difference). */
export function finishBatch(queue) {
  return { pending: (queue && queue.pending) || [], inflight: null }
}

/** How many decisions are not yet acknowledged by the server. What the bar
 * prints as "saving 12…" so a run that ends is never mistaken for a run that
 * is saved. */
export function queueDepth(queue) {
  if (!queue) return 0
  return (queue.pending?.length || 0) + (queue.inflight?.ids?.length || 0)
}

/* ── Settings ─────────────────────────────────────────────────────────────────
   Two screen preferences, in one key. `lds.videobank.burst` follows the
   convention bankLayout.RAIL_STORAGE_KEY set, and the key and the field names
   are STORED IDENTIFIERS: renaming either without an alias path silently resets
   everyone's preference (the repo's rule on stored ids). */

export const BURST_STORAGE_KEY = 'lds.videobank.burst'

export const BURST_DEFAULTS = { on: false, autoAdvance: true }

/** localStorage, or nothing at all: this module is imported by node tests, and
 * a private-mode browser throws on ACCESS, not only on read. */
function defaultStore() {
  try {
    return globalThis.localStorage || null
  } catch { return null }
}

/** The remembered preferences. Anything unreadable, malformed or of the wrong
 * type falls back FIELD BY FIELD to the default — a corrupt half must not be
 * able to take the working half down with it. */
export function loadBurstPrefs(store = defaultStore()) {
  let raw = null
  try {
    raw = store?.getItem(BURST_STORAGE_KEY)
  } catch { return { ...BURST_DEFAULTS } }
  if (!raw) return { ...BURST_DEFAULTS }
  let saved = null
  try {
    saved = JSON.parse(raw)
  } catch { return { ...BURST_DEFAULTS } }
  if (!saved || typeof saved !== 'object') return { ...BURST_DEFAULTS }
  return {
    on: typeof saved.on === 'boolean' ? saved.on : BURST_DEFAULTS.on,
    autoAdvance: typeof saved.autoAdvance === 'boolean'
      ? saved.autoAdvance : BURST_DEFAULTS.autoAdvance,
  }
}

/** Remember them. A full quota must not break a toggle that otherwise worked. */
export function saveBurstPrefs(prefs, store = defaultStore()) {
  const next = {
    on: !!(prefs?.on),
    autoAdvance: prefs?.autoAdvance !== false,
  }
  try {
    store?.setItem(BURST_STORAGE_KEY, JSON.stringify(next))
  } catch { /* private mode / quota — the session still has the right state */ }
  return next
}
