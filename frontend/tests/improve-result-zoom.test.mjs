/**
 * ✨ The improve result can be magnified — proved on the rendered element, and
 * on the wiring that puts it in front of the reader.
 *
 * An upscale is judged on detail that fit-to-dialog hides, so a dialog that can
 * only show the whole picture answers "did it run?" and not "is it better?".
 * The gestures themselves belong to useImageZoomPan (covered by
 * imageZoomPan.test.js, geometry and all); what is pinned here is that this
 * view mounts on that engine rather than growing a second one, and that the
 * modal actually renders it.
 *
 * ⚠️ mountJsx runs no effects and fires no events: the wheel listener and every
 * gesture are attached in effects, so what a render can prove is the frame, the
 * picture, and the state the view starts in. That is stated rather than
 * implied — a green run here is not a zoom that works, it is a zoom that is
 * wired and cannot throw.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { readSource } from './support/readSource.mjs'
import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

const { default: ImproveResultView, ZoomResetPill } =
  await import('../src/components/shared/ImproveResultView.jsx')

const render = (props) => renderToStaticMarkup(createElement(ImproveResultView, props))

test('the result is drawn inside a frame the gestures can own', () => {
  const html = render({ url: '/api/image/42.png' })
  assert.match(html, /data-testid="improve-result-frame"/)
  assert.match(html, /src="\/api\/image\/42\.png"/)
  // Without touch-none the browser claims the pinch and zooms the whole PAGE
  // over a dialog that was already zooming the picture — and the modal's own
  // vertical scroll eats the drag on a phone.
  assert.match(html, /touch-none/)
  assert.match(html, /overflow-hidden/,
    'a magnified picture must be clipped by its frame, not push the dialog wide')
})

test('at fit there is no way-back pill — it appears only when it can do something', () => {
  const html = render({ url: '/api/image/42.png' })
  assert.doesNotMatch(html, /data-testid="improve-result-zoom-reset"/)
})

test('no result, no frame', () => {
  for (const url of [null, undefined, '']) {
    assert.equal(render({ url }), '', `${JSON.stringify(url)} should render nothing`)
  }
})

test('it runs on the lightbox engine, not a second implementation', () => {
  const src = readSource('src/components/shared/ImproveResultView.jsx')
  assert.match(src, /useImageZoomPan/,
    'wheel, pinch, drag and double-tap already exist once in this app')
  assert.match(src, /resetKey/,
    'a second run in the same dialog must not inherit the first view, panned '
    + 'to a corner of a picture that is gone')
  assert.doesNotMatch(src, /addEventListener\(\s*['"]wheel/,
    'the wheel is the hook’s, attached non-passive there for a reason')
})

test('the modal shows the result through that view', () => {
  const src = readSource('src/components/shared/ImproveModal.jsx')
  assert.match(src, /import ImproveResultView/)
  assert.match(src, /<ImproveResultView\s+url=\{result\.url\}/)
  assert.doesNotMatch(src, /<img\s+src=\{result\.url\}/,
    'the flat <img> is what left people closing the dialog to inspect the render')
})

test('the way back out renders, and says how far in you are', () => {
  // The zoomed branch of the view itself is unreachable in `node --test` (no
  // gesture ever fires), so the pill is its own component and IS executed here.
  const html = renderToStaticMarkup(createElement(ZoomResetPill, { scale: 2.5 }))
  assert.match(html, /data-testid="improve-result-zoom-reset"/)
  assert.match(html, /250%/, '"reset" alone never says how far in you are')
  assert.match(html, /aria-label="Reset the zoom"/)
  // 40 px below lg, like every other control of this dialog.
  assert.match(html, /min-h-10/)
})

test('the pill is a SIBLING of the capturing frame, never a child of it', () => {
  /* Load-bearing, and measured: the frame calls setPointerCapture on itself, and
     a captured pointer retargets the compatibility click to the capturing
     element — a button inside the frame is dead to a mouse (a touch tap still
     works, which is how it looked fine on a phone). The lightbox has always
     drawn its reset pill outside its pane. */
  const src = readSource('src/components/shared/ImproveResultView.jsx')
  const frame = src.split('data-testid="improve-result-frame"')[1].split('</div>')[0]
  assert.doesNotMatch(frame, /ZoomResetPill/,
    'inside the capturing frame, the pill never receives a mouse click')
  assert.match(src.split('</div>')[1] || src, /ZoomResetPill/)
  assert.match(src, /zoom\.zoomed && <ZoomResetPill/)
})

test('the result phase gets the height, and stops the dialog scrolling', () => {
  /* A fixed 62vh frame inside a scrollable body was the worst of both on a
     phone held sideways: part of the render below the fold, and `touch-none`
     had taken away the drag that would have scrolled to it. */
  const view = readSource('src/components/shared/ImproveResultView.jsx')
  assert.doesNotMatch(view, /h-\[62vh\]/)
  assert.match(view, /flex min-h-0 w-full flex-1/)
  const modal = readSource('src/components/shared/ImproveModal.jsx')
  assert.match(modal, /phase === 'done' \? 'flex overflow-hidden' : 'overflow-y-auto'/)
  assert.match(modal, /phase === 'done' \? 'h-full max-h-full' : 'max-h-full'/)
})

test('a run starts on the dials the panel is SHOWING, and never on a dead candidate', () => {
  const modal = readSource('src/components/shared/ImproveModal.jsx')
  const generate = modal.split('const generate =')[1].split('const close =')[0]
  // The chain is resolved server-side at enqueue time: a slider dropped a beat
  // ago has to have landed before the POST, or the render uses the old value.
  assert.match(generate, /flushImproveSettings\(\)/)
  assert.match(generate, /await whenImproveSettingsSettled\(\)/)
  assert.ok(generate.indexOf('whenImproveSettingsSettled') < generate.indexOf('postJson'),
    'settling AFTER the POST would settle nothing')
  // A candidate left over from a failed attempt answers "failed" forever, and
  // the poll restarts on it the moment the phase flips.
  assert.match(generate, /setCandidateId\(null\)/)
  assert.match(modal, /setCandidateId\(null\); setResult\(null\); setError\(null\); setPhase\('settings'\)/)
})
