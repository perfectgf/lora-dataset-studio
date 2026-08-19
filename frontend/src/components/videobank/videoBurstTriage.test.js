import test from 'node:test'
import assert from 'node:assert/strict'

import {
  BURST_SHORTCUTS, BURST_HINT, BURST_STORAGE_KEY, BURST_DEFAULTS, UNDO_DEPTH,
  burstKeyAction, clipIndex, nextPendingIndex, firstPendingIndex, stepIndex,
  afterDecision, burstTally, burstProgressLine, burstEndNote,
  undoEntry, pushUndo, popUndo, undoLine,
  createQueue, queueDecision, startBatch, finishBatch, queueDepth,
  loadBurstPrefs, saveBurstPrefs,
} from './videoBurstTriage.js'
import { triagePayload } from './videoTriage.js'
import { reviewKeyAction } from '../shared/reviewShortcuts.js'

const press = (key, extra = {}) => ({ key, target: null, ...extra })

/** A page of shots. `s` is the status; ids are 1-based to keep the maths
 * readable when a test says "the cursor lands on 4". */
const page = (statuses) => statuses.map((s, i) => ({
  id: i + 1, status: s, start_s: i * 10, end_s: i * 10 + 3,
}))

/* ── the keys ─────────────────────────────────────────────────────────────── */

test('the three verdicts sit on the SAME keys as the image lane', () => {
  assert.equal(burstKeyAction(press('k')), 'keep')
  assert.equal(burstKeyAction(press('r')), 'reject')
  assert.equal(burstKeyAction(press('s')), 'skip')
  assert.equal(burstKeyAction(press('ArrowRight')), 'skip')
  assert.equal(burstKeyAction(press('ArrowLeft')), 'back')
  // and they are the SHARED grammar's answers, not a second copy of them
  for (const key of ['k', 'r', 's', 'ArrowRight', 'ArrowLeft']) {
    assert.equal(burstKeyAction(press(key)), reviewKeyAction(press(key)),
      `${key} drifted from components/shared/reviewShortcuts.js`)
  }
})

test('the video lane\'s own keys: P untriaged, U undo, Home, ? and Esc', () => {
  assert.equal(burstKeyAction(press('p')), 'pending')
  assert.equal(burstKeyAction(press('P')), 'pending')
  assert.equal(burstKeyAction(press('u')), 'undo')
  assert.equal(burstKeyAction(press('Home')), 'first')
  assert.equal(burstKeyAction(press('?')), 'help')
  assert.equal(burstKeyAction(press('Escape')), 'exit')
})

test('? is read even though it arrives shifted', () => {
  // Shift+/ on most layouts — the shared grammar refuses every shifted key, so
  // delegating first would have swallowed the help panel.
  assert.equal(burstKeyAction(press('?', { shiftKey: true })), 'help')
})

test('a modified keystroke is never ours', () => {
  assert.equal(burstKeyAction(press('r', { ctrlKey: true })), null)
  assert.equal(burstKeyAction(press('r', { metaKey: true })), null)
  assert.equal(burstKeyAction(press('u', { altKey: true })), null)
  assert.equal(burstKeyAction(press('ArrowRight', { shiftKey: true })), null)
  assert.equal(burstKeyAction(null), null)
  assert.equal(burstKeyAction(press('q')), null)
})

/* ── the safety that matters most on this screen ──────────────────────────── */

test('NOTHING fires while the focus is in a text field', () => {
  const typing = [
    { tagName: 'INPUT', type: 'text' },
    { tagName: 'INPUT', type: 'search' },
    { tagName: 'TEXTAREA' },
    { tagName: 'SELECT' },
    { isContentEditable: true },
  ]
  for (const target of typing) {
    for (const key of ['k', 'r', 'p', 's', 'u', '?', 'Home', 'ArrowRight', 'ArrowLeft']) {
      assert.equal(burstKeyAction(press(key, { target })), null,
        `${key} fired inside ${target.tagName || 'contenteditable'}`)
    }
  }
})

test('but a checkbox, a radio, a button and a slider do NOT eat letters', () => {
  // The workspace is full of them (flag chips, quality cuts, Select page) and a
  // blanket "is it an input" guard once killed K/R/S outright in the image Bank.
  for (const type of ['checkbox', 'radio', 'button', 'submit', 'range']) {
    assert.equal(burstKeyAction(press('r', { target: { tagName: 'INPUT', type } })), 'reject')
  }
})

test('Escape gets out even from inside a text field', () => {
  assert.equal(burstKeyAction(press('Escape', { target: { tagName: 'INPUT', type: 'text' } })),
    'exit')
})

test('the printed hint names every key the handler answers', () => {
  const answered = ['keep', 'reject', 'pending', 'skip', 'back', 'undo', 'first', 'help', 'exit']
  const seen = new Set()
  for (const key of ['k', 'r', 'p', 's', 'ArrowLeft', 'u', 'Home', '?', 'Escape']) {
    seen.add(burstKeyAction(press(key)))
  }
  assert.deepEqual([...seen].sort(), [...answered].sort())
  // and the panel documents all nine, so ? cannot go stale against the handler
  assert.equal(BURST_SHORTCUTS.length, 9)
  for (const row of BURST_SHORTCUTS) {
    assert.ok(row.keys && row.what, 'a shortcut row must say the key AND what it does')
  }
  for (const bit of ['K keep', 'R reject', 'U undo', '? help']) {
    assert.ok(BURST_HINT.includes(bit), `the hint line dropped "${bit}"`)
  }
})

/* ── the cursor and the auto-advance ──────────────────────────────────────── */

test('a decision jumps to the next UNTRIAGED shot, not simply the next one', () => {
  //           0        1       2         3         4
  const clips = page(['keep', 'keep', 'pending', 'keep', 'pending'])
  // decided index 0; 1 and 3 are already judged, so the cursor lands on 2
  assert.equal(afterDecision({ clips, index: 0, autoAdvance: true }), 2)
  assert.equal(afterDecision({ clips, index: 2, autoAdvance: true }), 4)
})

test('with auto-advance off the cursor does not move at all', () => {
  const clips = page(['keep', 'pending', 'pending'])
  assert.equal(afterDecision({ clips, index: 0, autoAdvance: false }), 0)
})

test('the run does NOT wrap — it stops on the shot just decided', () => {
  const clips = page(['pending', 'keep', 'keep'])
  // everything after index 0 is judged: no wrap back to the top
  assert.equal(afterDecision({ clips, index: 2, autoAdvance: true }), 2)
  assert.equal(afterDecision({ clips, index: 1, autoAdvance: true }), 1)
})

test('a full run of decisions walks every untriaged shot exactly once', () => {
  // The whole state machine, end to end: K on each, auto-advance on.
  const clips = page(['pending', 'keep', 'pending', 'reject', 'pending', 'pending'])
  let index = firstPendingIndex(clips)
  const visited = []
  for (let guard = 0; guard < 50; guard += 1) {
    if (index < 0) break
    visited.push(index)
    clips[index] = { ...clips[index], status: 'keep' }
    const next = afterDecision({ clips, index, autoAdvance: true })
    if (next === index) break              // nothing untriaged left ahead
    index = next
  }
  assert.deepEqual(visited, [0, 2, 4, 5])
  assert.equal(burstTally(clips).pending, 0)
})

test('← and → move by exactly one tile, decided or not, and never fall off', () => {
  const clips = page(['pending', 'keep', 'keep', 'pending'])
  assert.equal(stepIndex(clips, 0, 1), 1)        // lands on a KEPT shot on purpose
  assert.equal(stepIndex(clips, 1, -1), 0)
  assert.equal(stepIndex(clips, 0, -1), 0)       // clamped, no wrap
  assert.equal(stepIndex(clips, 3, 1), 3)
  assert.equal(stepIndex([], 0, 1), -1)
})

test('Home goes to the first untriaged shot, wherever the cursor was', () => {
  const clips = page(['keep', 'pending', 'reject', 'pending'])
  assert.equal(firstPendingIndex(clips), 1)
  assert.equal(firstPendingIndex(page(['keep', 'keep'])), -1)
})

test('clipIndex and nextPendingIndex survive junk', () => {
  assert.equal(clipIndex(null, 3), -1)
  assert.equal(clipIndex(page(['pending']), null), -1)
  assert.equal(nextPendingIndex(null, 0), -1)
  assert.equal(afterDecision({ clips: [], index: 0 }), -1)
  // a row with no status at all counts as untriaged, like the server's default
  assert.equal(nextPendingIndex([{ id: 1, status: 'keep' }, { id: 2 }], 0), 1)
})

/* ── what the bar says at the end ─────────────────────────────────────────── */

test('the end note tells the three endings apart', () => {
  const clips = page(['keep', 'pending', 'keep'])
  // still work ahead → nothing to say
  assert.equal(burstEndNote({ clips, index: 0 }), null)
  // past the last untriaged one, with some left BEHIND
  const behind = burstEndNote({ clips, index: 2 })
  assert.match(behind, /1 untriaged shot/)
  assert.match(behind, /Home/)
  // all judged, page complete
  assert.match(burstEndNote({ clips: page(['keep', 'reject']), index: 1 }),
    /Every shot on this page is triaged/)
  // all judged, but the bank has more to load
  assert.match(burstEndNote({ clips: page(['keep']), index: 0, hasMore: true }),
    /Load more/)
  assert.match(burstEndNote({ clips: [], index: 0 }), /No shot on this page/)
})

test('the progress line counts what is done and what is left', () => {
  assert.equal(burstProgressLine(page(['keep', 'pending', 'reject', 'pending'])),
    '2 / 4 triaged · 2 left')
  assert.equal(burstProgressLine([]), 'Nothing to triage here.')
})

/* ── undo ─────────────────────────────────────────────────────────────────── */

test('an undo step restores what the shot WAS, not a blank', () => {
  const clip = { id: 7, status: 'keep', start_s: 12, end_s: 15 }
  const e = undoEntry(clip, 'reject')
  assert.deepEqual({ id: e.id, from: e.from, to: e.to }, { id: 7, from: 'keep', to: 'reject' })
  // an untriaged shot reverts to untriaged
  assert.equal(undoEntry({ id: 8, start_s: 0, end_s: 1 }, 'keep').from, 'pending')
  assert.equal(undoEntry(null, 'keep'), null)
  assert.equal(undoEntry({ id: 1, start_s: 0, end_s: 1 }, 'nonsense'), null)
})

test('undoing after three fast decisions walks back one shot at a time', () => {
  // The brief's case: K, R, R at speed, then "no wait, that second one".
  let stack = []
  stack = pushUndo(stack, undoEntry({ id: 1, status: 'pending', start_s: 0, end_s: 2 }, 'keep'))
  stack = pushUndo(stack, undoEntry({ id: 2, status: 'pending', start_s: 2, end_s: 4 }, 'reject'))
  stack = pushUndo(stack, undoEntry({ id: 3, status: 'keep', start_s: 4, end_s: 6 }, 'reject'))
  assert.equal(stack.length, 3)

  // the bar names the step it is offering AND how far the net still reaches
  assert.match(undoLine(stack), /✕ Reject/)
  assert.match(undoLine(stack), /2 more steps back/)

  let popped = popUndo(stack)
  assert.equal(popped.entry.id, 3)
  assert.equal(popped.entry.from, 'keep')       // it was kept before, it goes back to kept
  popped = popUndo(popped.stack)
  assert.equal(popped.entry.id, 2)
  assert.equal(popped.entry.from, 'pending')
  popped = popUndo(popped.stack)
  assert.equal(popped.entry.id, 1)
  assert.deepEqual(popUndo(popped.stack), { entry: null, stack: [] })
})

test('the last step drops the "more steps back" tail, and an empty stack offers nothing', () => {
  const one = pushUndo([], undoEntry({ id: 1, status: 'pending', start_s: 0, end_s: 2 }, 'keep'))
  assert.doesNotMatch(undoLine(one), /more step/)
  assert.equal(undoLine([]), null)
  assert.equal(undoLine(null), null)
  assert.equal(pushUndo([], null).length, 0)
})

test('the net is bounded — it never grows past UNDO_DEPTH, and keeps the NEWEST', () => {
  let stack = []
  for (let i = 1; i <= UNDO_DEPTH + 5; i += 1) {
    stack = pushUndo(stack, undoEntry({ id: i, status: 'pending', start_s: i, end_s: i + 1 }, 'reject'))
  }
  assert.equal(stack.length, UNDO_DEPTH)
  assert.equal(stack[stack.length - 1].id, UNDO_DEPTH + 5)
  assert.equal(stack[0].id, 6)
})

/* ── the queue: what a fast hand does to a slow network ───────────────────── */

test('a run of identical decisions collapses into ONE batch', () => {
  let q = createQueue()
  for (const id of [1, 2, 3, 4, 5]) q = queueDecision(q, id, 'reject')
  assert.equal(queueDepth(q), 5)
  q = startBatch(q)
  assert.deepEqual(q.inflight, { ids: [1, 2, 3, 4, 5], status: 'reject' })
  assert.equal(q.pending.length, 0)
  assert.equal(queueDepth(q), 5)                 // still unacknowledged
  q = finishBatch(q)
  assert.equal(queueDepth(q), 0)
})

test('a change of verdict starts a new batch — order is never scrambled', () => {
  let q = createQueue()
  q = queueDecision(q, 1, 'reject')
  q = queueDecision(q, 2, 'reject')
  q = queueDecision(q, 3, 'keep')
  q = queueDecision(q, 4, 'reject')
  q = startBatch(q)
  assert.deepEqual(q.inflight, { ids: [1, 2], status: 'reject' })
  q = startBatch(q)                              // in flight → unchanged
  assert.deepEqual(q.inflight, { ids: [1, 2], status: 'reject' })
  q = startBatch(finishBatch(q))
  assert.deepEqual(q.inflight, { ids: [3], status: 'keep' })
  q = startBatch(finishBatch(q))
  assert.deepEqual(q.inflight, { ids: [4], status: 'reject' })
  q = finishBatch(q)
  assert.equal(startBatch(q).inflight, null)
})

test('re-deciding a queued shot replaces it — the LAST key wins, once', () => {
  let q = createQueue()
  q = queueDecision(q, 1, 'keep')
  q = queueDecision(q, 2, 'keep')
  q = queueDecision(q, 1, 'reject')              // changed my mind about 1
  assert.equal(queueDepth(q), 2)
  q = startBatch(q)
  assert.deepEqual(q.inflight, { ids: [2], status: 'keep' })
  q = startBatch(finishBatch(q))
  assert.deepEqual(q.inflight, { ids: [1], status: 'reject' })
})

test('a shot re-decided while its first decision is IN FLIGHT lands after it', () => {
  // Serialisation is what makes the replace-in-place rule safe: the flight
  // that already left is acknowledged before the entry that supersedes it.
  let q = queueDecision(createQueue(), 1, 'keep')
  q = startBatch(q)
  assert.deepEqual(q.inflight, { ids: [1], status: 'keep' })
  q = queueDecision(q, 1, 'reject')              // hammered while in flight
  assert.deepEqual(q.inflight, { ids: [1], status: 'keep' })   // untouched
  assert.deepEqual(q.pending, [{ id: 1, status: 'reject' }])
  q = startBatch(finishBatch(q))
  assert.deepEqual(q.inflight, { ids: [1], status: 'reject' }) // the last key wins
})

test('nothing is lost when twenty keystrokes race one slow request', () => {
  // Twenty decisions typed while a single flight is out; every one of them must
  // reach the server exactly once, in order, and none may be dropped.
  let q = queueDecision(createQueue(), 0, 'keep')
  q = startBatch(q)
  const sent = [...q.inflight.ids]
  for (let i = 1; i <= 20; i += 1) q = queueDecision(q, i, i % 2 ? 'reject' : 'keep')
  let guard = 0
  while (queueDepth(finishBatch(q)) > 0 && guard < 100) {
    q = startBatch(finishBatch(q))
    sent.push(...q.inflight.ids)
    guard += 1
  }
  q = finishBatch(q)
  assert.equal(queueDepth(q), 0)
  assert.deepEqual(sent, Array.from({ length: 21 }, (_, i) => i))
})

test('the queue REFUSES what would become an empty-ids bulk overwrite', () => {
  // The one footgun of this API: POST /triage with empty ids means EVERY clip
  // in the bank. A burst run posts constantly, so this is the exact context in
  // which an empty batch would silently retag hundreds of decisions.
  let q = createQueue()
  q = queueDecision(q, null, 'reject')
  q = queueDecision(q, undefined, 'reject')
  q = queueDecision(q, 3, 'not-a-status')
  assert.equal(queueDepth(q), 0)
  assert.equal(startBatch(q).inflight, null)     // no batch at all — nothing to post

  // and every batch this file DOES produce still goes through triagePayload,
  // which is the thing that refuses an empty list
  q = startBatch(queueDecision(createQueue(), 9, 'reject'))
  assert.ok(q.inflight.ids.length > 0)
  assert.deepEqual(triagePayload(q.inflight.ids, q.inflight.status),
    { ids: [9], status: 'reject' })
  assert.equal(triagePayload([], 'reject'), null)
})

test('the queue survives junk without throwing', () => {
  assert.equal(queueDepth(null), 0)
  assert.deepEqual(finishBatch(null), { pending: [], inflight: null })
  assert.deepEqual(startBatch(null), createQueue())
  assert.deepEqual(queueDecision(null, 1, 'keep').pending, [{ id: 1, status: 'keep' }])
})

/* ── the remembered preferences ───────────────────────────────────────────── */

const fakeStore = (initial = {}) => {
  const data = { ...initial }
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = String(v) },
  }
}

test('burst mode and auto-advance are remembered across sessions', () => {
  const store = fakeStore()
  saveBurstPrefs({ on: true, autoAdvance: false }, store)
  assert.deepEqual(loadBurstPrefs(store), { on: true, autoAdvance: false })
  assert.equal(store.data[BURST_STORAGE_KEY], '{"on":true,"autoAdvance":false}')
})

test('a fresh install gets the defaults: burst off, auto-advance on', () => {
  assert.deepEqual(loadBurstPrefs(fakeStore()), BURST_DEFAULTS)
  assert.equal(BURST_DEFAULTS.on, false)
  assert.equal(BURST_DEFAULTS.autoAdvance, true)
})

test('a corrupt or half-written preference degrades FIELD BY FIELD', () => {
  assert.deepEqual(loadBurstPrefs(fakeStore({ [BURST_STORAGE_KEY]: 'not json' })), BURST_DEFAULTS)
  assert.deepEqual(loadBurstPrefs(fakeStore({ [BURST_STORAGE_KEY]: 'null' })), BURST_DEFAULTS)
  assert.deepEqual(loadBurstPrefs(fakeStore({ [BURST_STORAGE_KEY]: '[1,2]' })),
    { on: false, autoAdvance: true })
  // the good half survives the bad half
  assert.deepEqual(loadBurstPrefs(fakeStore({ [BURST_STORAGE_KEY]: '{"on":true,"autoAdvance":"yes"}' })),
    { on: true, autoAdvance: true })
})

test('a private-mode browser that throws on ACCESS never breaks the toggle', () => {
  const hostile = {
    getItem() { throw new Error('SecurityError') },
    setItem() { throw new Error('SecurityError') },
  }
  assert.deepEqual(loadBurstPrefs(hostile), BURST_DEFAULTS)
  assert.deepEqual(saveBurstPrefs({ on: true, autoAdvance: true }, hostile),
    { on: true, autoAdvance: true })
  assert.deepEqual(loadBurstPrefs(null), BURST_DEFAULTS)
})

test('the storage key is a STORED identifier and must not drift', () => {
  // Renaming it silently resets everyone's preference — the repo's rule on ids
  // kept in user databases and localStorage.
  assert.equal(BURST_STORAGE_KEY, 'lds.videobank.burst')
})
