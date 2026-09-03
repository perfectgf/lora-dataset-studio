import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ENRICH_UNKNOWN, addFrames, failureNotice, generateLabel, perImagePrompts, queueClips, queuedNotice, releasePreview,
  removeFrame, uploadKey,
} from './videoStartFrames.js'

const frame = (key, image = `${key}.png`, ratio = 1.5) => ({ key, image, ratio, preview: null })

test('frames are appended in pick order and a frame already held is skipped, by its ORIGIN', () => {
  const a = frame('bank:3:41')
  const { frames, skipped } = addFrames([a], [frame('gallery:88'), { ...a, image: 'lds_vstudio_other.png' }])
  assert.deepEqual(frames.map((f) => f.key), ['bank:3:41', 'gallery:88'])
  // The same portrait staged again has a NEW staged name — the key, not the
  // name, is what says it is the same picture.
  assert.equal(skipped.length, 1)
  assert.equal(skipped[0].key, 'bank:3:41')
  // Nothing to stage is nothing to add: a reply without an image is dropped.
  assert.deepEqual(addFrames([], [{ key: 'x', image: null }, null]).frames, [])
  assert.deepEqual(addFrames(undefined, undefined).frames, [])
  assert.deepEqual(removeFrame(frames, 'bank:3:41').map((f) => f.key), ['gallery:88'])
  assert.equal(uploadKey({ name: 'a.png', size: 10, lastModified: 5 }), 'upload:a.png:10:5')
})

test('letting go of a preview only touches a blob URL the picker minted', () => {
  // A server URL is not ours; a frame without a preview has nothing to let go
  // of; a blob URL is revoked (and revoking twice is a no-op, so a frame let
  // go of by ✕ and again by the parent cannot throw).
  const blob = URL.createObjectURL(new Blob(['x']))
  assert.doesNotThrow(() => releasePreview({ preview: blob }))
  assert.doesNotThrow(() => releasePreview({ preview: blob }))
  assert.doesNotThrow(() => releasePreview({ preview: '/api/bank/3/thumb/41' }))
  assert.doesNotThrow(() => releasePreview({ preview: null }))
  assert.doesNotThrow(() => releasePreview(null))
})

test('one launch per frame, in order, and what the server ran first — seed AND prompt — is what the rest run', async () => {
  const bodies = []
  const post = async (body) => {
    bodies.push(body)
    return { seed: 4242, frames: 56, prompt: 'HEADER\n\nShe turns, slowly, toward the lens.' }
  }
  const progress = []
  const out = await queueClips(
    [frame('a', 'a.png', 1.5), frame('b', 'b.png', 0.5625), frame('c', 'c.png', 1)],
    { mode: 'i2v', prompt: 'she turns', seed: '', frames: 56, enhance: true },
    post, (done, total) => progress.push(`${done}/${total}`))
  assert.equal(out.failed, null)
  assert.equal(out.queued.length, 3)
  assert.equal(out.total, 3)
  assert.equal(out.seed, 4242)
  assert.deepEqual(bodies.map((b) => b.image), ['a.png', 'b.png', 'c.png'])
  // Each frame's OWN shape sizes its latent upscale — never the first one's.
  assert.deepEqual(bodies.map((b) => b.ratio), [1.5, 0.5625, 1])
  // Random on the first launch (no seed key at all — the server's definition
  // of random), then the server's number on every later one.
  assert.equal('seed' in bodies[0], false)
  assert.deepEqual(bodies.slice(1).map((b) => b.seed), [4242, 4242])
  // ✨ Enrich at launch is asked of the FIRST clip only; the rest run the
  // rewrite the server named, with `enhance` dropped — a clip in the queue
  // shuts the vision window, so asking again would give clips 2 and 3 the
  // prompt as typed (refuted 2026-09-02).
  assert.equal(bodies[0].enhance, true)
  assert.equal(bodies[0].prompt, 'she turns')
  assert.ok(bodies.slice(1).every((b) => !('enhance' in b) && b.prompt === 'HEADER\n\nShe turns, slowly, toward the lens.'))
  assert.equal(out.enrichSkipped, null)
  // The rest of the launch travels unchanged into every clip.
  assert.ok(bodies.every((b) => b.frames === 56))
  assert.deepEqual(progress, ['1/3', '2/3', '3/3'])
  assert.equal(queuedNotice(out), 'Queued 3 clips — seed 4242, 56 frames.')
})

test('enrichment: the server\'s refusal is kept as said; a reply that names no prompt is said too; no enrich, no carry', async () => {
  // The fence refused the first clip: every clip ran the prompt as typed,
  // consistently, and the toast carries the server's own sentence.
  let bodies = []
  const fenced = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', enhance: true },
    async (b) => { bodies.push(b); return { seed: 1, prompt: 'HEADER\n\np', enrich_skipped: 'the fence is up' } })
  assert.equal(fenced.enrichSkipped, 'the fence is up')
  assert.deepEqual(bodies.map((b) => b.prompt), ['p', 'HEADER\n\np'])
  assert.deepEqual(bodies.map((b) => 'enhance' in b), [true, false])
  // A server that does not say which prompt ran (older than this reply):
  // the rest run the text as typed and the batch says so — but one clip has
  // no "rest", and says nothing.
  bodies = []
  const mute = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', enhance: true },
    async (b) => { bodies.push(b); return { seed: 1 } })
  assert.equal(mute.enrichSkipped, ENRICH_UNKNOWN)
  assert.deepEqual(bodies.map((b) => b.prompt), ['p', 'p'])
  // A prompt that is nothing but blanks is "no prompt" too: carried as is, it
  // would send the rest with an empty prompt and a 400 each (the mutation
  // that drops the trim() survived the tests above — refuted 2026-09-02).
  bodies = []
  const blank = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', enhance: true },
    async (b) => { bodies.push(b); return { seed: 1, prompt: '   \n ' } })
  assert.equal(blank.enrichSkipped, ENRICH_UNKNOWN)
  assert.deepEqual(bodies.map((b) => b.prompt), ['p', 'p'])
  assert.deepEqual(bodies.map((b) => 'enhance' in b), [true, false])
  const single = await queueClips([frame('a')], { mode: 'i2v', prompt: 'p', enhance: true }, async () => ({ seed: 1 }))
  assert.equal(single.enrichSkipped, null)
  // Without the checkbox nothing is asked and nothing is carried: the prompt
  // as typed, every time, even when the server echoes one.
  bodies = []
  await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', enhance: false },
    async (b) => { bodies.push(b); return { seed: 1, prompt: 'HEADER\n\np' } })
  assert.ok(bodies.every((b) => !('enhance' in b) && b.prompt === 'p'))
})

test('a seed typed by hand is sent as typed to every clip; a reply without a seed leaves the dial alone AND says so', async () => {
  const bodies = []
  const post = async (body) => { bodies.push(body); return { frames: 39 } }
  const typed = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', seed: '7' }, post)
  assert.deepEqual(bodies.map((b) => b.seed), [7, 7])
  assert.equal(typed.seed, '7')
  bodies.length = 0
  // A negative seed is the server's other word for random (it draws on it):
  // the number it drew is the batch's seed, and the notice names THAT one —
  // not "-1" over three clips that ran three draws (refuted 2026-09-02).
  const negative = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', seed: '-1' },
    async (b) => { bodies.push(b); return { seed: 555, frames: 39 } })
  assert.deepEqual(bodies.map((b) => b.seed), [-1, 555])
  assert.equal(queuedNotice(negative), 'Queued 2 clips — seed 555, 39 frames.')
  bodies.length = 0
  const random = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', seed: '' }, post)
  // No seed came back: nothing to carry, and nothing invented — the later
  // launches stay random rather than sending `seed: undefined`. And the
  // notice cannot stay quiet about it: the strip promised "on one seed"
  // (refuted 2026-09-02 — three clips, three draws, a notice that said
  // nothing). A seed of 0 is a seed, not an absence.
  assert.ok(bodies.every((b) => !('seed' in b)))
  assert.equal(queuedNotice(random), 'Queued 2 clips — independent seeds (the server did not return the first), 39 frames.')
  const single = await queueClips([frame('a')], { mode: 'i2v', prompt: 'p', seed: '' }, post)
  assert.equal(queuedNotice(single), 'Queued — 39 frames.', 'one clip has nothing to share')
  const zero = await queueClips([frame('a'), frame('b')], { mode: 'i2v', prompt: 'p', seed: '' },
    async () => ({ seed: 0, frames: 39 }))
  assert.equal(queuedNotice(zero), 'Queued 2 clips — seed 0, 39 frames.')
})

test('the first refusal ends the walk, keeps what queued, and the notice counts it', async () => {
  const bodies = []
  const post = async (body) => {
    bodies.push(body)
    if (bodies.length === 3) throw new Error('ComfyUI is not reachable')
    return { seed: 9, frames: 56 }
  }
  const progress = []
  const out = await queueClips([frame('a'), frame('b'), frame('c'), frame('d')],
    { mode: 'i2v', prompt: 'p', seed: '' }, post, (done, total) => progress.push(`${done}/${total}`))
  assert.equal(out.queued.length, 2)
  assert.equal(out.total, 4)
  assert.equal(bodies.length, 3, 'the fourth frame is never sent')
  // The counter never advances on the launch that was refused.
  assert.deepEqual(progress, ['1/4', '2/4'])
  assert.equal(out.failed.message, 'ComfyUI is not reachable')
  assert.equal(failureNotice(out), 'Queued 2 of 4 — ComfyUI is not reachable')
  // Nothing went through: the sentence alone, as the single-clip panel said it.
  const none = await queueClips([frame('a')], { mode: 'i2v', prompt: 'p' },
    async () => { throw new Error('no LoRA') })
  assert.equal(failureNotice(none), 'no LoRA')
  assert.equal(failureNotice({ queued: [], total: 1, failed: {} }), 'The clip could not be queued.')
})

test('text-only is one launch without a picture, whatever the strip holds', async () => {
  const bodies = []
  const post = async (body) => { bodies.push(body); return { seed: 1, frames: 56, enrich_skipped: 'fence' } }
  const out = await queueClips([null], { mode: 't2v', prompt: 'p', aspect: 'portrait', seed: '' }, post)
  assert.equal(bodies.length, 1)
  assert.equal('image' in bodies[0], false)
  assert.equal(bodies[0].aspect, 'portrait')
  assert.equal(queuedNotice(out), 'Queued — seed 1, 56 frames.')
  // An empty list is the same single launch — the panel never sends nothing.
  bodies.length = 0
  await queueClips([], { mode: 't2v', prompt: 'p' }, post)
  assert.equal(bodies.length, 1)
})

test('"queued without enrichment" is kept once for the batch, not once per clip', async () => {
  const post = async () => ({ seed: 1, frames: 56, enrich_skipped: 'the fence is up' })
  const out = await queueClips([frame('a'), frame('b'), frame('c')], { mode: 'i2v', prompt: 'p', enhance: true }, post)
  assert.equal(out.enrichSkipped, 'the fence is up')
  const clean = await queueClips([frame('a')], { mode: 'i2v', prompt: 'p' }, async () => ({ seed: 1 }))
  assert.equal(clean.enrichSkipped, null)
  assert.equal(queuedNotice(clean), 'Queued — seed 1.')
})

test('the button counts the clips a click queues, and where the walk is while it queues', () => {
  assert.equal(generateLabel({ mode: 'i2v', count: 1, busy: false }), 'Generate clip')
  assert.equal(generateLabel({ mode: 'i2v', count: 0, busy: false }), 'Generate clip')
  assert.equal(generateLabel({ mode: 'i2v', count: 3, busy: false }), 'Generate 3 clips')
  // Text-only ignores the strip: it is one clip however many frames wait.
  assert.equal(generateLabel({ mode: 't2v', count: 3, busy: false }), 'Generate clip')
  assert.equal(generateLabel({ mode: 'i2v', count: 1, busy: true, done: 0, total: 1 }), 'Queueing…')
  assert.equal(generateLabel({ mode: 'i2v', count: 3, busy: true, done: 0, total: 3 }), 'Queueing 1 of 3…')
  assert.equal(generateLabel({ mode: 'i2v', count: 3, busy: true, done: 2, total: 3 }), 'Queueing 3 of 3…')
  // Never "4 of 3" between the last reply and the state reset.
  assert.equal(generateLabel({ mode: 'i2v', count: 3, busy: true, done: 3, total: 3 }), 'Queueing 3 of 3…')
})

test('a frame with its own prompt launches as written, never enriched again; a continuation names its clip', async () => {
  const bodies = []
  const post = async (body) => { bodies.push(body); return { seed: 7, prompt: 'rewritten' } }
  const frames = [
    { ...frame('a'), prompt: 'she waves' },
    frame('b'),
    { ...frame('c'), continues: 41 },
  ]
  await queueClips(frames, { mode: 'i2v', prompt: 'typed', enhance: true }, post)
  assert.equal(bodies[0].prompt, 'she waves')
  assert.equal(bodies[0].enhance, undefined, 'its own prompt: nothing to enrich')
  assert.equal(bodies[1].prompt, 'typed')
  assert.equal(bodies[1].enhance, true, 'the first launch WITHOUT its own prompt is the one enriched')
  assert.equal(bodies[2].prompt, 'rewritten', 'and the rest run the rewrite')
  assert.equal(bodies[2].continues, 41)
  assert.equal(bodies[0].continues, undefined)
})

test('the per-picture mode writes one prompt per frame first, falls back to the typed one, and names who fell', async () => {
  const asked = []
  const ask = async (f, typed) => {
    asked.push([f.key, typed])
    if (f.key === 'b') throw new Error('fence')
    if (f.key === 'c') return '   '
    return `motion for ${f.key}`
  }
  const steps = []
  const { frames, fallen, error } = await perImagePrompts([frame('a'), frame('b'), frame('c')], 'typed',
    ask, (d, t) => steps.push([d, t]))
  assert.deepEqual(frames.map((f) => f.prompt), ['motion for a', 'typed', 'typed'])
  assert.deepEqual(fallen.map((f) => [f.key, f.index]), [['b', 1], ['c', 2]], 'which pictures, by place in the strip')
  assert.equal(error?.message, 'fence', 'the first refusal travels, so the notice can say why')
  assert.equal(fallen[0].error?.message, 'fence')
  assert.deepEqual(steps, [[1, 3], [2, 3], [3, 3]])
  assert.deepEqual(asked.map((a) => a[1]), ['typed', 'typed', 'typed'])
  assert.deepEqual((await perImagePrompts([], 'x', ask)).frames, [])
})

test('the button says it is writing prompts while it writes them', () => {
  assert.equal(generateLabel({ mode: 'i2v', count: 3, busy: true, done: 1, total: 3, phase: 'writing' }), 'Writing prompt 2 of 3…')
  assert.equal(generateLabel({ mode: 'i2v', count: 3, busy: true, done: 1, total: 3 }), 'Queueing 2 of 3…')
})
