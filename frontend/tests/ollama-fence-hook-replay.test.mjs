/**
 * The local-LLM fence guard, RUN — clicks, timers and the fence state, with
 * the hook's own code. `ollamaFenceWiring.test.js` reads the hook as text and
 * proves the vigil is written; this proves what it does with a second click.
 *
 * The defect these tests exist for: a click refused by the fence armed a
 * vigil; a LATER click on the same surface went through by itself (the model
 * had been freed) — and the vigil, never stopped, fired a moment later and
 * ran that later click AGAIN. One click, two model calls, the field written
 * twice; and a later click that FAILED for another reason was toasted by its
 * own catch and then again by the replay. Invisible to a render (effects
 * never run) and to a regex (nothing is missing from the text).
 */
import assert from 'node:assert/strict'
import test, { mock } from 'node:test'

import { fake, flush, mountHook } from './support/hookRuntime.mjs'

const { default: useOllamaFence } = await import('../src/hooks/useOllamaFence.js')
const { AUTO_RETRY_CAP_MS, OLLAMA_FENCE_CODE, fenceNoticeModel } =
  await import('../src/utils/ollamaFence.js')

const fenced = () => Object.assign(new Error('A local model is in use outside LDS.'),
  { body: { code: OLLAMA_FENCE_CODE } })

/* A fence state the test moves by hand; the hook polls it. */
function scene() {
  const fence = { blocked: true, models: ['other:8b'], provider: 'ollama' }
  const polls = []
  fake.apiFetch = async (url) => { polls.push(url); return { ...fence } }
  return { fence, polls }
}

/* One click: the action runs; `calls` counts how many times. `fail` is the
   error the action throws — or a function of the moment, for an action that
   is refused only while the fence is up, the way a real request is. */
function click(handle, name, calls, fail) {
  return handle.read().runGuarded(async () => {
    calls.push(name)
    const err = typeof fail === 'function' ? fail() : fail
    if (err) throw err
  })
}

test.beforeEach(() => { mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 10_000 }) })
test.afterEach(() => { mock.timers.reset() })

test('a refused click waits, and runs by itself once — the vigil as designed', async () => {
  const { fence, polls } = scene()
  const calls = []
  const h = mountHook(useOllamaFence, {})
  const refusedWhileBlocked = () => (fence.blocked ? fenced() : null)
  assert.equal(await click(h, 'a', calls, refusedWhileBlocked), false)
  assert.equal(h.read().fence.phase, 'waiting')

  mock.timers.tick(2_000); await flush()
  assert.equal(polls.length, 1)
  assert.equal(h.read().fence.phase, 'waiting')

  fence.blocked = false
  mock.timers.tick(2_000); await flush(); await flush()
  assert.deepEqual(calls, ['a', 'a'])
  assert.equal(h.read().fence, null)

  // Nothing armed after: more time changes nothing.
  mock.timers.tick(60_000); await flush()
  assert.deepEqual(calls, ['a', 'a'])
  assert.equal(polls.length, 2)
  h.unmount()
})

test('a later click that goes through supersedes the vigil — it is not run a second time', async () => {
  const { fence } = scene()
  const calls = []
  const h = mountHook(useOllamaFence, {})
  await click(h, 'a', calls, fenced())
  assert.equal(h.read().fence.phase, 'waiting')

  // The model was freed; the user clicks again before the poll sees it.
  fence.blocked = false
  assert.equal(await click(h, 'b', calls), true)
  assert.equal(h.read().fence, null)

  mock.timers.tick(2_000); await flush(); await flush()
  mock.timers.tick(5_000); await flush(); await flush()
  assert.deepEqual(calls, ['a', 'b'], 'the vigil replayed the click that had already gone through')
  assert.equal(h.read().fence, null)
  h.unmount()
})

test('a later click that fails for another reason is reported once, by its own catch', async () => {
  const { fence } = scene()
  const calls = []
  const errors = []
  const h = mountHook(useOllamaFence, { onError: (e) => errors.push(e) })
  await click(h, 'a', calls, fenced())

  fence.blocked = false
  await assert.rejects(click(h, 'b', calls, new Error('Ollama died')), /Ollama died/)
  assert.equal(h.read().fence, null)

  mock.timers.tick(2_000); await flush(); await flush()
  assert.deepEqual(calls, ['a', 'b'])
  assert.deepEqual(errors, [], 'the replay ran the failed click again and reported it a second time')
  h.unmount()
})

test('a poll already past its fetch when the vigil is stopped acts on nothing', async () => {
  // The narrow race: the poll is awaiting the fence state when the user
  // clicks again (or stops waiting). It must not replay when it comes back.
  const calls = []
  let release
  fake.apiFetch = () => new Promise((resolve) => { release = () => resolve({ blocked: false, models: [] }) })
  const h = mountHook(useOllamaFence, {})
  await click(h, 'a', calls, fenced())
  mock.timers.tick(2_000); await flush()
  assert.equal(typeof release, 'function', 'the poll is in flight')

  assert.equal(await click(h, 'b', calls), true)
  release(); await flush(); await flush()
  assert.deepEqual(calls, ['a', 'b'])
  assert.equal(h.read().fence, null)
  h.unmount()
})

test('a click made during the unload is not run a second time by the resume', async () => {
  // "Unload it and continue" is a POST that takes a moment; the buttons are
  // live meanwhile. A click made then runs itself — the resume that follows
  // the unload must not run it again.
  const { fence } = scene()
  const calls = []
  let finishUnload
  fake.postJson = () => new Promise((resolve) => { finishUnload = () => resolve({ ok: true }) })
  const h = mountHook(useOllamaFence, {})
  await click(h, 'a', calls, fenced())
  const unload = h.read().unloadAndRetry()
  await flush()
  assert.equal(h.read().fence.phase, 'unloading')
  assert.equal(typeof finishUnload, 'function', 'the unload is in flight')

  // The model frees up; the user clicks again, and that click is still
  // running when the unload comes back.
  fence.blocked = false
  const bResolvers = []
  const bDone = h.read().runGuarded(async () => {
    calls.push('b')
    await new Promise((resolve) => { bResolvers.push(resolve) })
  })
  await flush()
  assert.deepEqual(calls, ['a', 'b'])
  finishUnload(); await flush(); await flush()
  bResolvers.forEach((resolve) => resolve()); await flush()
  assert.equal(await unload, true)
  assert.equal(await bDone, true)
  await flush()
  assert.deepEqual(calls, ['a', 'b'], 'the resume ran the click that was already running')
  assert.equal(h.read().fence, null)
  h.unmount()
})

test('a replay that ends after a newer click was fenced leaves that click its notice and its turn', async () => {
  // The replay of an old click is still running when the user clicks again
  // and is fenced. The old replay then fails for another reason: it must not
  // clear the new click's notice, drop its kept action, or toast — the new
  // click owns the vigil now, and gets replayed when the model frees up.
  const { fence } = scene()
  const calls = []
  const errors = []
  let failA
  let aRuns = 0
  const h = mountHook(useOllamaFence, { onError: (e) => errors.push(e) })
  await h.read().runGuarded(async () => {
    calls.push('a')
    aRuns += 1
    if (aRuns === 1) throw fenced()
    await new Promise((_, reject) => { failA = () => reject(new Error('died mid-replay')) })
  })
  assert.equal(h.read().fence.phase, 'waiting')

  fence.blocked = false
  mock.timers.tick(2_000); await flush(); await flush()
  assert.equal(h.read().fence.phase, 'retrying')
  assert.deepEqual(calls, ['a', 'a'])

  fence.blocked = true
  assert.equal(await click(h, 'b', calls, () => (fence.blocked ? fenced() : null)), false)
  assert.equal(h.read().fence.phase, 'waiting')

  failA(); await flush(); await flush()
  assert.deepEqual(errors, [], 'the superseded replay toasted its failure')
  assert.equal(h.read().fence.phase, 'waiting', 'the superseded replay cleared the newer notice')

  fence.blocked = false
  mock.timers.tick(2_000); await flush(); await flush()
  assert.deepEqual(calls, ['a', 'a', 'b', 'b'], 'the newer click was not replayed')
  assert.equal(h.read().fence, null)
  h.unmount()
})

test('a click superseded while it runs is not put on watch when refused — the refusal reaches its caller', async () => {
  // The Video Test Studio stops waiting when the mode or the start frame
  // changes; a click still running under the old setup that then comes back
  // refused must not open a vigil that would replay it under the new one —
  // and must not vanish either: its caller's catch gets the refusal.
  const { polls } = scene()
  const calls = []
  let refuse
  const h = mountHook(useOllamaFence, {})
  const a = h.read().runGuarded(async () => {
    calls.push('a')
    await new Promise((_, reject) => { refuse = () => reject(fenced()) })
  })
  await flush()
  h.read().stopWaiting()
  refuse()
  await assert.rejects(a, (e) => e?.body?.code === OLLAMA_FENCE_CODE)
  assert.equal(h.read().fence, null)

  mock.timers.tick(30_000); await flush(); await flush()
  assert.deepEqual(calls, ['a'])
  assert.equal(polls.length, 0, 'a vigil was opened for a click nobody wants replayed')
  h.unmount()
})

test('"stop waiting" abandons the click for good', async () => {
  const { fence } = scene()
  const calls = []
  const h = mountHook(useOllamaFence, {})
  await click(h, 'a', calls, fenced())
  h.read().stopWaiting()
  assert.equal(h.read().fence, null)

  fence.blocked = false
  mock.timers.tick(2_000); await flush(); await flush()
  assert.deepEqual(calls, ['a'])
  h.unmount()
})

test('the notice names the server the fence state reports — LM Studio, not Ollama by default', async () => {
  const { fence } = scene()
  fence.provider = 'lmstudio'
  fence.models = ['qwen/qwen3-vl-4b']
  const h = mountHook(useOllamaFence, {})
  await click(h, 'a', [], fenced())

  mock.timers.tick(2_000); await flush()
  let m = fenceNoticeModel(h.read().fence)
  assert.match(m.detail, /qwen\/qwen3-vl-4b in LM Studio/)
  assert.doesNotMatch(m.detail, /Ollama/)

  // Still named when the patience runs out.
  mock.timers.tick(AUTO_RETRY_CAP_MS); await flush()
  assert.equal(h.read().fence.phase, 'gave-up')
  m = fenceNoticeModel(h.read().fence)
  assert.match(m.detail, /in LM Studio/)
  h.unmount()
})

test('unmounting stops the vigil', async () => {
  const { fence, polls } = scene()
  const calls = []
  const h = mountHook(useOllamaFence, {})
  await click(h, 'a', calls, fenced())
  h.unmount()
  fence.blocked = false
  mock.timers.tick(30_000); await flush(); await flush()
  assert.deepEqual(calls, ['a'])
  assert.equal(polls.length, 0)
})

/* ── a reply that arrives after the user moved on ──────────────────────────
   The hook can stop a vigil; it cannot stop a request in flight. The reply
   comes back to the action, which is what writes into the field — so the
   action is handed a `run` and asks it. These actions do what the panels do:
   a request in flight, then a write, guarded — and unguarded when no `run`
   is handed (the hook before it), which is the defect these prove gone. */
const defer = () => { let r; const p = new Promise((a) => { r = a }); return { p, r } }

function writer(writes, name, reply, { refusedOnce = false } = {}) {
  let first = true
  return async (run) => {
    if (refusedOnce && first) { first = false; throw fenced() }
    await reply.p
    if (!run?.current || run.current()) writes.push(`${name}:written`)
    else writes.push(`${name}:set aside${run.mounted() ? ', toast' : ', quiet'}`)
  }
}

test('an answer that arrives after "stop waiting" is set aside, and the surface is told', async () => {
  const { fence } = scene()
  const writes = []
  const reply = defer()
  const h = mountHook(useOllamaFence, { onError: () => {} })
  assert.equal(await h.read().runGuarded(writer(writes, 'a', reply, { refusedOnce: true })), false)

  fence.blocked = false
  mock.timers.tick(60_000); await flush()
  assert.deepEqual(writes, [], 'the replay is in flight')
  // The user changes the frame: the panel's effect stops the vigil.
  h.read().stopWaiting()
  assert.equal(h.read().fence, null)

  reply.r(); await flush(); await flush()
  assert.deepEqual(writes, ['a:set aside, toast'])
  assert.equal(h.read().fence, null)
  h.unmount()
})

test('an answer that arrives after a newer click is set aside; the newer click is written', async () => {
  const { fence } = scene()
  const writes = []
  const replyA = defer()
  const replyB = defer()
  const h = mountHook(useOllamaFence, { onError: () => {} })
  await h.read().runGuarded(writer(writes, 'a', replyA, { refusedOnce: true }))

  fence.blocked = false
  mock.timers.tick(60_000); await flush()
  // A's replay is in flight when the user clicks again.
  const b = h.read().runGuarded(writer(writes, 'b', replyB))
  replyA.r(); await flush(); await flush()
  assert.deepEqual(writes, ['a:set aside, toast'])
  replyB.r(); await flush(); await flush()
  assert.equal(await b, true)
  assert.deepEqual(writes, ['a:set aside, toast', 'b:written'])
  assert.equal(h.read().fence, null)

  mock.timers.tick(60_000); await flush(); await flush()
  assert.deepEqual(writes, ['a:set aside, toast', 'b:written'])
  h.unmount()
})

test('an answer that arrives after the panel went away is set aside, quietly', async () => {
  const { fence } = scene()
  const writes = []
  const reply = defer()
  const h = mountHook(useOllamaFence, { onError: () => {} })
  await h.read().runGuarded(writer(writes, 'a', reply, { refusedOnce: true }))

  fence.blocked = false
  mock.timers.tick(60_000); await flush()
  h.unmount()
  reply.r(); await flush(); await flush()
  assert.deepEqual(writes, ['a:set aside, quiet'])
})

test('a setup change during the FIRST click sets its answer aside too', async () => {
  // No refusal at all: the click is simply slow, and the user re-aims the
  // panel before it comes back.
  scene()
  const writes = []
  const reply = defer()
  const h = mountHook(useOllamaFence, { onError: () => {} })
  const a = h.read().runGuarded(writer(writes, 'a', reply))
  h.read().stopWaiting()
  reply.r(); await flush(); await flush()
  assert.equal(await a, true)
  assert.deepEqual(writes, ['a:set aside, toast'])
  assert.equal(h.read().fence, null)
  h.unmount()
})

test('a click made while the notice is still up takes it down, and the unload has nothing to resume', async () => {
  // The mirror of "a click made during the unload": here the click comes
  // FIRST — a's notice is up, b is clicked and still running — and the user
  // clicks Unload. With a's notice left up under b, the resume that followed
  // the unload ran b again on top of itself: two model calls, and b's own
  // answer set aside with a note that told the user nothing true.
  const { fence } = scene()
  const writes = []
  const replyB = defer()
  const unloads = []
  fake.postJson = async (url) => { unloads.push(url); return { ok: true } }
  const h = mountHook(useOllamaFence, { onError: () => {} })
  await h.read().runGuarded(writer(writes, 'a', defer(), { refusedOnce: true }))
  assert.equal(h.read().fence.phase, 'waiting')

  fence.blocked = false
  const b = h.read().runGuarded(writer(writes, 'b', replyB))
  await flush()
  const noticeUnderB = h.read().fence
  const unload = h.read().unloadAndRetry()
  await flush()
  replyB.r(); await flush(); await flush()
  assert.equal(await b, true)
  assert.deepEqual(writes, ['b:written'], 'b ran once and was written once')
  assert.equal(noticeUnderB, null, 'a newer click takes the notice down')
  assert.equal(await unload, false)
  assert.deepEqual(unloads, [], 'nothing waiting, nothing to unload for')
  assert.equal(h.read().fence, null)

  mock.timers.tick(60_000); await flush(); await flush()
  assert.deepEqual(writes, ['b:written'])
  h.unmount()
})
