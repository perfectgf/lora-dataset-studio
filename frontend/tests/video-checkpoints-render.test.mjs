/**
 * 📦 The Checkpoints & LoRAs section of a video dataset, RENDERED from a
 * payload — the states a maintainer would otherwise only see with a trained
 * set behind the page. `renderToStaticMarkup` runs no effect, so the list is
 * fed directly; what is pinned is that the markup carries what the model
 * decided: a verb as a real <button>, a refusal as its sentence, one ⬇ per
 * file, and never "Final (step null)".
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { VideoCheckpointList } = await import('../src/components/videobank/VideoCheckpointManager.jsx')
const {
  ACTIVE_CLOUD_REASON, CONTINUE_LOCAL_REASON, EMPTY_NOTE, HAND_PLACED_REASON, NO_LORAS_ROOT_REASON,
} = await import('../src/components/videobank/videoCheckpoints.js')

const file = (filename, extra = {}) => ({ filename, size: 314572800, deployed_as: null, undeployable: false, ...extra })
const PAYLOAD = {
  can_deploy: true, deploy_folder: 'h3/lds', delete_mode: 'app_trash',
  local: {
    run_name: 'video_city_ds9', folder: 'X:/out/video_city_ds9', active: false,
    steps: [
      { step: 50, final: false, deployed: false,
        files: [file('video_city_000000050_high_noise.safetensors'), file('video_city_000000050_low_noise.safetensors')] },
      { step: null, final: true, deployed: true,
        files: [file('video_city.safetensors', { deployed_as: 'h3/lds/video_city.safetensors', undeployable: true })] },
    ],
  },
  cloud: [
    { run_id: 13, status: 'training', active: true, gpu: 'A100', price_per_hour: 1.2, parent_run_id: null,
      created_at: '2026-09-02T10:00:00', finished_at: null,
      steps: [{ step: 200, final: false, deployed: false, files: [file('video_city_000000200.safetensors')] }] },
    { run_id: 12, status: 'done', active: false, gpu: 'RTX 4090', price_per_hour: 0.5, parent_run_id: 7,
      created_at: '2026-09-01T10:00:00', finished_at: '2026-09-01T12:00:00',
      steps: [{ step: 100, final: false, deployed: true,
        files: [file('video_city_000000100.safetensors', { deployed_as: 'h3/video_city_000000100.safetensors', undeployable: false })] }] },
  ],
}
const html = (props) => renderToStaticMarkup(createElement(VideoCheckpointList, { datasetId: 9, payload: PAYLOAD, ...props }))
const rowOf = (h, key) => {
  const at = h.indexOf(`data-step-key="${key}"`)
  assert.notEqual(at, -1, `no row for ${key}`)
  const from = h.lastIndexOf('<li', at)
  return h.slice(from, h.indexOf('</li>', at))
}
const esc = (s) => s.replace(/&/g, '&amp;').replace(/'/g, '&#x27;')

test('an empty payload is one sentence, not a header over nothing', () => {
  const h = html({ payload: { local: null, cloud: [] } })
  assert.ok(h.includes(EMPTY_NOTE))
  assert.ok(!h.includes('<ul'))
})

test('a local Wan step: one ⬇ per file on the LOCAL route, 📦, the continue refusal, no ⓘ', () => {
  const row = rowOf(html(), 'local:50')
  assert.ok(row.includes('Step 50 — 2 files (both experts)'))
  const links = [...row.matchAll(/<a [^>]*href="([^"]+)"[^>]*download/g)].map((m) => m[1])
  assert.deepEqual(links, [
    '/api/video-dataset/9/train/checkpoint?filename=video_city_000000050_high_noise.safetensors',
    '/api/video-dataset/9/train/checkpoint?filename=video_city_000000050_low_noise.safetensors',
  ])
  assert.ok(row.includes('high noise') && row.includes('low noise') && row.includes('300 MB'))
  assert.match(row, /<button[^>]*>[^<]*<span aria-hidden="true">📦<\/span> Deploy → h3\/lds<\/button>/)
  assert.ok(row.includes(esc(CONTINUE_LOCAL_REASON)))
  assert.ok(!row.includes('Continue from here'))
  assert.ok(!row.includes('Details'))
  assert.match(row, /<button[^>]*>.*Delete the training saves<\/button>/s)
})

test('the local FINAL save reads "Final", carries the Deployed badge and a real ⏏ Undeploy button', () => {
  const row = rowOf(html(), 'local:final')
  assert.ok(row.includes('>Final</span>'))
  assert.ok(!row.includes('null'))
  assert.ok(row.includes('>Deployed</span>'))
  assert.match(row, /<button[^>]*>[^<]*<span aria-hidden="true">⏏<\/span> Undeploy<\/button>/)
  assert.ok(!row.includes('📦'))
})

test('a terminal cloud step: ▶ Continue, ⓘ Details, and a hand-placed copy that is deployed but not ours to undeploy', () => {
  const h = html()
  const row = rowOf(h, 'cloud-12:100')
  assert.match(row, /<button[^>]*>[^<]*<span aria-hidden="true">▶<\/span> Continue from here<\/button>/)
  assert.match(row, /<button[^>]*>[^<]*<span aria-hidden="true">ⓘ<\/span> Details<\/button>/)
  assert.ok(row.includes('>Deployed</span>'))
  assert.ok(row.includes(esc(HAND_PLACED_REASON)))
  assert.ok(!row.includes('Undeploy</button>'))
  assert.ok(row.includes('href="/api/video-dataset/9/train/cloud/checkpoint?run_id=12&amp;filename=video_city_000000100.safetensors"'))
  // The run-level 🗑 and the genealogy line live on the group header.
  assert.ok(h.includes('aria-label="Delete run 12 and its checkpoints"'))
  assert.ok(h.includes('Cloud run #12 — continued from #7'))
})

test('an ACTIVE cloud run: no run-level 🗑, and ▶ / 🗑 are refusals with their sentence', () => {
  const h = html()
  const row = rowOf(h, 'cloud-13:200')
  assert.ok(!h.includes('aria-label="Delete run 13 and its checkpoints"'))
  assert.ok(!row.includes('Continue from here'))
  assert.ok(!row.includes('Delete the training save'))
  assert.equal((row.match(new RegExp(esc(ACTIVE_CLOUD_REASON), 'g')) || []).length, 2)
})

test('no loras root on this install: every 📦 is the refusal, no deploy button anywhere', () => {
  const h = html({ payload: { ...PAYLOAD, can_deploy: false } })
  assert.ok(!h.includes('Deploy → '))
  assert.ok(h.includes(esc(NO_LORAS_ROOT_REASON)))
})

test('ⓘ rows render under their run, and the ▶ form under its step', () => {
  const h = html({
    details: { run_id: 12, rows: [['GPU', 'RTX 4090 · $0.50/h'], ['Steps', '2000']] },
    continueTarget: 'cloud-12:100', extraSteps: 1500,
  })
  assert.ok(h.includes('aria-label="Run 12 details"'))
  assert.ok(h.includes('<dt class="text-content-subtle">GPU</dt>'))
  const row = rowOf(h, 'cloud-12:100')
  assert.ok(row.includes('aria-label="Extra steps"') && row.includes('value="1500"'))
  // Native constraint validation: with min=1, a step of 100 would make every
  // round number (500, 1000, 2000) invalid and the form would never submit.
  assert.match(row, /<input type="number" min="1" step="1"/, 'the extra-steps input must accept any integer')
  assert.ok(row.includes('▶ Train further'))
  assert.ok(!rowOf(h, 'cloud-13:200').includes('Extra steps'))
})

test('busy rows are disabled, and the verb says what it is doing', () => {
  const row = rowOf(html({ busy: 'local:50:deploy' }), 'local:50')
  assert.ok(row.includes('Deploying…'))
  assert.ok((row.match(/<button[^>]*disabled=""/g) || []).length >= 2)
})
