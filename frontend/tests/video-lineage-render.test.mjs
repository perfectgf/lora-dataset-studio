/**
 * 🌳 The ◉ Graph of a video dataset, RENDERED from a tree of the shape the
 * server answers — with the image lane's cards, pills and edges, verbatim.
 * What is pinned: a card per run labelled "Video", a pill per STEP (a Wan
 * pair is one pill), an edge per continuation anchored on the resumed pill,
 * a still on a pill that has a sample and none on one that has not, and no
 * `<video>` in the graph itself (the lightbox is the one player).
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { default: VideoLineageGraph, VideoCheckpointPopover } =
  await import('../src/components/videobank/VideoLineageGraph.jsx')
const { default: VideoSampleLightbox } = await import('../src/components/videobank/VideoSampleLightbox.jsx')
const { pillActionModel } = await import('../src/components/videobank/videoLineage.js')

const file = (filename, extra = {}) => ({ filename, size: 314572800, deployed_as: null, undeployable: false, ...extra })
const POSTER = '/api/video-dataset/9/train/sample/poster?run_id=12&filename=1725__000000100_0.mp4'
const TREE = {
  root_id: null, current_id: null, single: false,
  nodes: [
    { record_id: 12, run_id: 12, source: 'cloud', parent_record_id: null, resumed_from: null, origin_unknown: false,
      dataset_id: 9, dataset_name: 'City', train_type: 'video', variant: 'wan22_14b', base_model: '', version: null,
      steps: 2000, config: {}, note: '', has_note: false, is_current: false, created_at: '2026-09-01T10:00:00',
      status: 'done', active: false, training_mode: 'lora', saves: 3, checkpoint_ready: true,
      checkpoints: [
        { step: 100, final: false, filename: 'a_000000100_high_noise.safetensors', present: true, testable: false,
          deployed_filename: null, preview_url: POSTER, preview_status: 'ready', preview_count: 2,
          download_url: '/api/video-dataset/9/train/cloud/checkpoint?run_id=12&filename=a_000000100_high_noise.safetensors',
          files: [file('a_000000100_high_noise.safetensors'), file('a_000000100_low_noise.safetensors')] },
        { step: 2000, final: true, filename: 'a.safetensors', present: true, testable: false, deployed_filename: null,
          preview_url: null, preview_status: null, preview_count: 0,
          download_url: '/api/video-dataset/9/train/cloud/checkpoint?run_id=12&filename=a.safetensors',
          files: [file('a.safetensors')] },
      ] },
    { record_id: 13, run_id: 13, source: 'cloud', parent_record_id: 12, resumed_from: 100, origin_unknown: false,
      dataset_id: 9, dataset_name: 'City', train_type: 'video', variant: 'wan22_14b', base_model: '', version: null,
      steps: 600, config: {}, note: '', has_note: false, is_current: false, created_at: '2026-09-02T10:00:00',
      status: 'training', active: true, training_mode: 'lora', saves: 1, checkpoint_ready: true,
      checkpoints: [
        { step: 200, final: false, filename: 'a_000000200.safetensors', present: true, testable: false, deployed_filename: null,
          preview_url: null, preview_status: null, preview_count: 0,
          download_url: '/api/video-dataset/9/train/cloud/checkpoint?run_id=13&filename=a_000000200.safetensors',
          files: [file('a_000000200.safetensors')] },
      ] },
    { record_id: -9, run_id: null, source: 'local', parent_record_id: null, resumed_from: null, origin_unknown: false,
      dataset_id: 9, dataset_name: 'City', train_type: 'video', variant: 'wan22_14b', base_model: '', version: null,
      steps: null, config: {}, note: '', has_note: false, is_current: false, created_at: null,
      status: null, active: false, training_mode: 'lora', run_name: 'video_city_ds9', saves: 1, checkpoint_ready: true,
      checkpoints: [
        { step: null, final: true, filename: 'v.safetensors', present: true, testable: true,
          deployed_filename: 'h3/lds/v.safetensors', preview_url: null, preview_status: null, preview_count: 0,
          download_url: '/api/video-dataset/9/train/checkpoint?filename=v.safetensors',
          files: [file('v.safetensors', { deployed_as: 'h3/lds/v.safetensors', undeployable: true })] },
      ] },
  ],
  edges: [{ parent: 12, child: 13, resumed_from: 100, superseded: false }],
}
const html = (props) => renderToStaticMarkup(createElement(VideoLineageGraph,
  { datasetId: 9, tree: TREE, onPlaySample: () => {}, ...props }))

test('a card per run, labelled Video, and one pill per STEP — a Wan pair is one pill', () => {
  const h = html()
  assert.equal((h.match(/class="lds-gcard/g) || []).length, 3, 'three run cards')
  assert.equal((h.match(/lds-ckpill/g) || []).length >= 4, true, 'four pills: 100, 2000, 200, final')
  assert.ok(h.includes('>Video<') || h.includes('Video ·') || /Video/.test(h), 'the family reads "Video", not the raw key')
  assert.ok(!/>video</.test(h), 'the raw family key must not show')
  assert.ok(h.includes('aria-label="Lineage graph: 3 runs"'))
  assert.ok(!h.includes('<video'), 'the graph mounts no player')
})

test('a continuation draws one edge, anchored on the pill it resumed from', () => {
  const h = html()
  const paths = (h.match(/<path[^>]*class="[^"]*lds-ledge[^"]*"/g) || []).length
    || (h.match(/<path /g) || []).length
  assert.ok(paths >= 1, 'at least one edge path')
  // The resumed pill is marked as a resume source by the layout (ring class).
  assert.ok(h.includes('ring-indigo-400/60'), 'the step-100 pill is the anchor of the continuation')
})

test('compact: a pill with samples carries their COUNT, worded as samples; big previews: the still itself', () => {
  const compact = html()
  assert.ok(compact.includes('aria-label="Open the 2 samples of step 100"'), 'the count badge says samples, not images')
  assert.ok(compact.includes('🎬'), 'and wears the clip icon')
  assert.ok(!compact.includes('<img '), 'compact pills carry no thumbnail (illegible at that size)')
  assert.ok(!/Generate will deploy/.test(compact), 'no image-lane sentence about the 🎨 Generate bar')
  assert.ok(compact.includes('Deploy from its actions'), "the title points at the pill's own verbs")
  // Big previews on: the graph reads the toggle from localStorage.
  globalThis.localStorage = { getItem: (k) => (k === 'lds.videoGraphBigPreviews' ? '1' : null), setItem() {} }
  try {
    const big = html()
    assert.ok(big.includes(POSTER.replace(/&/g, '&amp;')), 'the poster of step 100')
    assert.equal((big.match(/<img /g) || []).length, 1, 'exactly one still — the other pills have no sample')
  } finally {
    delete globalThis.localStorage
  }
})

test('the popover offers the list\'s verbs for the same save, one ⬇ per file', () => {
  const node = TREE.nodes[0]
  const pill = node.checkpoints[0]
  const h = renderToStaticMarkup(createElement(VideoCheckpointPopover, {
    node, pill, a: pillActionModel(9, node, pill, { canDeploy: true }), onClose: () => {},
  }))
  const links = [...h.matchAll(/<a [^>]*href="([^"]+)"[^>]*download/g)].map((m) => m[1].replace(/&amp;/g, '&'))
  assert.deepEqual(links, [
    '/api/video-dataset/9/train/cloud/checkpoint?run_id=12&filename=a_000000100_high_noise.safetensors',
    '/api/video-dataset/9/train/cloud/checkpoint?run_id=12&filename=a_000000100_low_noise.safetensors',
  ])
  assert.ok(h.includes('Play samples (2)'))
  assert.ok(h.includes('Continue from here') && h.includes('Deploy → h3/lds') && h.includes('Details'))
  assert.ok(h.includes('Delete the training saves'))
  // The active run's pill: refusals with their sentence, no Continue button.
  const active = TREE.nodes[1]
  const h2 = renderToStaticMarkup(createElement(VideoCheckpointPopover, {
    node: active, pill: active.checkpoints[0], a: pillActionModel(9, active, active.checkpoints[0]), onClose: () => {},
  }))
  assert.ok(!h2.includes('Continue from here') && h2.includes('still on its pod'))
  assert.ok(!h2.includes('Play sample'))
})

test('the sample lightbox is a layer with ONE player slot and a close control', () => {
  const h = renderToStaticMarkup(createElement(VideoSampleLightbox, {
    datasetId: 9, target: { node: TREE.nodes[0], pill: TREE.nodes[0].checkpoints[0] }, onClose: () => {},
  }))
  assert.ok(h.includes('data-probe-chrome="sample-lightbox" data-probe-layer'))
  assert.ok(h.includes('aria-label="Close"'))
  assert.ok(h.includes('Step 100'))
  assert.ok(!h.includes('<video'), 'no player before the samples are read (effects do not run here)')
  assert.equal(renderToStaticMarkup(createElement(VideoSampleLightbox, { datasetId: 9, target: null, onClose: () => {} })), '')
})
