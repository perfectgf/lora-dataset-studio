import test from 'node:test'
import assert from 'node:assert/strict'
import { clipTags } from './videoClipTags.js'

test('clipTags carries the same facts as the summary, one per pill, no emoji', () => {
  const tags = clipTags({ eros: true, lora: 'h3/lds/j.safetensors', lora_strength: 1.3,
    turbo: true, sparse: 'max', latent_upscale: true, steps: 6, seed: 42 })
  assert.deepEqual(tags, ['10Eros base', 'j @ 1.3', 'turbo', 'sparse max', 'upscale ×2', '6 steps', 'seed 42'])
  assert.ok(tags.every((t) => !/\p{Extended_Pictographic}/u.test(t)))
  assert.deepEqual(clipTags({ steps: 30 }), ['no LoRA', '30 steps'])
  assert.deepEqual(clipTags(null), [])
})

test('a smoothed clip is never mistaken for the one it came from', () => {
  // ↗ VFI copies every setting of its source, so without this tag the pair is
  // two identical-looking cards — and the whole point is comparing them.
  const tags = clipTags({ turbo: true, fps: 48, vfi_of: 12, steps: 6 })
  assert.ok(tags.some((t) => /smoothed/.test(t)))
  assert.ok(tags.some((t) => /48 fps/.test(t)))
  // The source keeps its own tags and gains nothing.
  assert.ok(!clipTags({ turbo: true, fps: 24 }).some((t) => /smoothed/.test(t)))
  // ⚡ The other two accelerations carry their own name; larryvrh's keeps 'turbo'.
  assert.ok(clipTags({ accel: 'parasyte', steps: 6 }).includes('Parasyte Turbo'))
  assert.ok(clipTags({ accel: 'dareties', steps: 6 }).includes('DARE-TIES merge'))
  assert.ok(clipTags({ accel: 'turbo', turbo: true, steps: 6 }).includes('turbo'))
  // ⏭ A continuation says which clip it follows, and whether the join happened.
  assert.ok(clipTags({ continues_of: 41, joined: true, steps: 6 }).includes('continues #41'))
  assert.ok(clipTags({ continues_of: 41, joined: false, steps: 6 }).includes('continues #41 (not joined)'))
  // Still rendering: no verdict, so no "(not joined)" for the length of the render.
  const pending = clipTags({ continues_of: 41, joined: null, steps: 6 })
  assert.ok(pending.includes('continues #41') && !pending.some((t) => t.includes('not joined')))
})
