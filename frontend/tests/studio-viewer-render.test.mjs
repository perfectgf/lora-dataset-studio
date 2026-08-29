/* 🔍 The Studio viewer, EXECUTED: the adapter mounts the unified viewer with
   the Studio's extras — and the facts the backend now serves actually render.

   A source-text pin can prove the adapter exists; only rendering proves the
   wrapper's props survive contact with GeneratedImageLightbox (a renamed prop
   would parse fine and show nothing). Same harness as the other render tests. */
import assert from 'node:assert/strict'
import test from 'node:test'

import { render } from './support/mountJsx.mjs'

const { default: StudioResultViewer } =
  await import('../src/components/dataset/studio/StudioResultViewer.jsx')

const row = (over = {}) => ({
  id: 41, dataset_id: 7, url: '/api/dataset/7/img/cell.png', rating: 0,
  prompt: 'a probe scene', seed: 424242, strength: 0.9,
  checkpoint: 'z image\\lola_2000.safetensors', base_model: 'zimage_turbo.safetensors',
  sampler: 'euler', cfg: 1, steps: 8, inject_trigger: false, ...over,
})

test('the studio viewer renders the shared facts and its own verdict', () => {
  const html = render(StudioResultViewer, {
    img: row(), items: [row(), row({ id: 42 })],
    onRate: () => {}, onNavigate: () => {}, onClose: () => {},
  })
  // The unified viewer, not a fifth lightbox.
  assert.match(html, /data-testid="generated-image-lightbox"/)
  // The facts the comparison used to hide: seed, checkpoint, trigger state.
  assert.match(html, /424242/)
  assert.match(html, /lola_2000/)
  assert.match(html, /not injected/)
  // The viewer's own verbs arrived for free.
  assert.match(html, /data-testid="lightbox-camera-angles"/)
  assert.match(html, /data-testid="lightbox-repair"/)
  assert.match(html, /data-testid="lightbox-download"/)
  // The Studio's extras: the verdict pair and the loop counter.
  assert.match(html, /👍 Like/)
  assert.match(html, /👎 Not a fan/)
  assert.match(html, /1 \/ 2/)
  // Both chevrons: the comparison set wraps, so neither end loses one.
  assert.match(html, /aria-label="Previous image"/)
  assert.match(html, /aria-label="Next image"/)
})

test('a lone image drops the loop and the counter, keeps the verdict', () => {
  const html = render(StudioResultViewer, {
    img: row(), items: [row()], onRate: () => {}, onNavigate: () => {}, onClose: () => {},
  })
  assert.doesNotMatch(html, /1 \/ 1/)
  assert.doesNotMatch(html, /aria-label="Previous image"/)
  assert.match(html, /👍 Like/)
})
