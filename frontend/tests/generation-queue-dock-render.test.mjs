/**
 * What the queue dock actually SHOWS (GitHub #44).
 *
 * Its wording is a tested pure module, but the things that make it usable are
 * properties of the markup: that the running job is what you read first, that a
 * button which refuses says so in its accessible name rather than going grey,
 * and — the one that matters most — that an empty queue renders NOTHING. A dock
 * that renders an empty shell would sit permanently over the corner of an app
 * whose queue is empty most of the time, which is a worse bug than the one this
 * whole change is fixing.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { QueueDockBody } = await import(
  '../src/components/common/GenerationQueueDock.jsx')

const job = (patch = {}) => ({
  job_id: 'j1', title: 'Generation', surface: '📁 Dataset', engine: 'Klein',
  status: 'queued', position: 1, raw_status: 'pending', promoted: false,
  promotable: true, cancellable: true, blocked_by: null, dataset_name: null,
  since: null, ...patch,
})

const render = (listing, props = {}) => renderToStaticMarkup(
  createElement(QueueDockBody, { listing, open: true, ...props }))

test('an empty queue renders nothing at all', () => {
  for (const listing of [null, { jobs: [] }, { jobs: [], queued: 0 }])
    assert.equal(render(listing), '')
})

test('collapsed, it says what the GPU is doing without listing anything', () => {
  const listing = { jobs: [job({ status: 'generating' }), job({ job_id: 'j2' })],
    generating: 1, queued: 1, stalled: 0 }
  const html = render(listing, { open: false })
  assert.match(html, /1 generating · 1 queued/)
  assert.match(html, /aria-expanded="false"/)
  // The list is the expanded half; collapsed it must not be in the markup.
  assert.doesNotMatch(html, /<ul/)
})

test('expanded, the running job is the first thing on screen', () => {
  const listing = {
    jobs: [job({ job_id: 'run', title: 'Upscale & improve', status: 'generating' }),
      job({ job_id: 'wait', position: 1 })],
    generating: 1, queued: 1, stalled: 0,
  }
  const html = render(listing)
  assert.ok(html.indexOf('Upscale &amp; improve') < html.indexOf('Generation'),
    'the job on the GPU must be read before the ones waiting for it')
  assert.match(html, /<ul/)
})

test('a job names the dataset it belongs to, so two feeding one queue are told apart', () => {
  const html = render({ jobs: [job({ dataset_name: 'Faces' })], queued: 1 })
  assert.match(html, /📁 Dataset · Faces/)
})

// The heart of the change: a control that refuses must say why and where the
// thing that does work lives. Grey and mute is what produced the issue.
test('a job owned by a blocking pass explains itself instead of going grey', () => {
  const html = render({
    jobs: [job({ title: 'Watermark inpaint', engine: null, cancellable: false,
      blocked_by: 'the 🧽 Clean watermarks pass' })],
    queued: 1,
  })
  assert.match(html, /Owned by the 🧽 Clean watermarks pass — stop it from there\./)
  assert.match(html, /aria-label="Owned by the 🧽 Clean watermarks pass[^"]*"/)
})

test('a paused job points at the recovery banner', () => {
  const html = render({ jobs: [job({ status: 'stalled' })], stalled: 1 })
  assert.match(html, /recovery banner/)
})

test('“run next” is refused with its own reason, not the cancel one', () => {
  const first = render({ jobs: [job({ position: 1 })], queued: 1 })
  assert.match(first, /aria-label="Already next in line\."/)
  const running = render({ jobs: [job({ status: 'generating', promotable: false })],
    generating: 1 })
  assert.match(running, /nothing left to re-order/)
})

test('cancelling names what it will leave behind, not just "cancel"', () => {
  const html = render({ jobs: [job()], queued: 1 })
  assert.match(html, /Retry re-queues it/)
})

test('a queue held by a training run says so, collapsed and expanded', () => {
  const listing = { jobs: [job(), job({ job_id: 'j2', position: 2 })], queued: 2,
    paused_reason: 'LoRA training in progress - the studio is unavailable (GPU busy).' }
  const expanded = render(listing)
  assert.match(expanded, /LoRA training in progress/)
  // Collapsed, the same answer must be reachable without opening anything: a
  // pill that counts a line going nowhere and stays mute is the original bug.
  const collapsed = render(listing, { open: false })
  assert.match(collapsed, /aria-label="[^"]*on hold: LoRA training in progress[^"]*"/)
  assert.match(collapsed, /⏸/)
  assert.doesNotMatch(collapsed, /animate-pulse/)
})

test('the dock takes the offset its screen asks for', () => {
  // On /studio the bar underneath is opaque and z-[9960]: at bottom-4 the dock
  // is invisible AND unclickable, on one of the four surfaces that feed the
  // queue. The rule itself lives in utils/dockPlacement.js.
  const listing = { jobs: [job()], queued: 1 }
  assert.match(render(listing, { bottomClass: 'bottom-20' }), /fixed bottom-20 left-3/)
  assert.match(render(listing), /fixed bottom-4 left-3/)
})

test('the dock stays inside a narrow window', () => {
  // 400px is the width the project tests every surface at. A fixed panel that
  // is wider than the viewport pushes the page sideways instead of docking.
  const html = render({ jobs: [job()], queued: 1 })
  assert.match(html, /w-\[min\(23rem,calc\(100vw-1\.5rem\)\)\]/)
})
