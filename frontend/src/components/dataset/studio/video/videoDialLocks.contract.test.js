/* 🔒 Every dial in the video studio is guarded, and the guard is the app's own.
 *
 * Asked for from a phone (2026-09-02): scrolling the render rail with a thumb
 * dragged whichever slider it crossed, silently — the clip then rendered on a
 * length or a step count nobody chose.
 *
 * This file does not assert "the three sliders are locked": it FINDS every
 * range input in these panels and requires each one to carry the lock. A
 * fourth dial added tomorrow without one fails here rather than on somebody's
 * phone — the number is derived, never written down.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')
const PANELS = {
  'VideoOptionsPanel.jsx': read('./VideoOptionsPanel.jsx'),
  'VideoLoraPicker.jsx': read('./VideoLoraPicker.jsx'),
}

/** Every `<input type="range" … />` tag in a source, whole. */
function ranges(src) {
  const out = []
  let i = src.indexOf('<input type="range"')
  while (i !== -1) {
    out.push(src.slice(i, src.indexOf('/>', i) + 2))
    i = src.indexOf('<input type="range"', i + 1)
  }
  return out
}

test('every slider in the video studio wears the shared lock', () => {
  let total = 0
  for (const [name, src] of Object.entries(PANELS)) {
    const found = ranges(src)
    assert.ok(found.length > 0, `${name}: no range input found — has the panel moved?`)
    for (const tag of found) {
      assert.match(tag, /Lock\.rangeProps/,
        `${name}: a range input with no lock:\n${tag}`)
      // rangeProps carries `disabled` and the touch guard; the className must
      // be composed AFTER the spread or the lock's dimming is overwritten.
      assert.ok(tag.indexOf('rangeProps}') < tag.lastIndexOf('className'),
        `${name}: className must come after the spread:\n${tag}`)
    }
    total += found.length
  }
  assert.ok(total >= 4, `expected the studio's dials, found ${total}`)
})

test('the picker’s Preview size is the one dial that goes without a lock — on purpose', () => {
  // The lock exists because a dial dragged by a scrolling thumb changes what
  // RENDERS without anyone noticing. The picker's preview size changes only
  // how big the tiles are — a drift is seen at once and undone with the same
  // thumb, and no clip reads it. So it stays bare, and this pins that the
  // exemption is exactly one range wide: a second dial in the picker, or a
  // render setting moved there, fails here.
  const src = read('./VideoSourcePicker.jsx')
  const found = ranges(src)
  assert.equal(found.length, 1, `the picker holds one range input, found ${found.length}`)
  assert.match(found[0], /aria-label="Preview size"/, `not the preview size:\n${found[0]}`)
  assert.doesNotMatch(found[0], /Lock\.rangeProps/, 'a lock here would guard a cosmetic dial')
})

test('the lock is the app’s one implementation, not a second one', () => {
  // The video lane wearing its own padlock is how the two lanes drift apart.
  for (const [name, src] of Object.entries(PANELS)) {
    assert.match(src, /import SliderLock, \{ useSliderLock \} from '\.\.\/\.\.\/\.\.\/shared\/SliderLock'/,
      `${name} does not use the shared lock`)
    assert.doesNotMatch(src, /localStorage/,
      `${name} keeps its own lock memory instead of the shared one`)
  }
  const shared = read('../../../shared/LockableSlider.jsx')
  assert.match(shared, /useSliderLock/,
    'LockableSlider still holds a second copy of the lock')
})

test('a vertical swipe scrolls the page rather than dragging a dial', () => {
  /* The floor under every slider in the app, locked or not: without
     `touch-action: pan-y` a range input keeps the gesture that merely crossed
     it. Asserted on the stylesheet because that is where it applies to the 27
     ranges this app renders, not only to the four above. */
  const css = read('../../../../index.css')
  const rule = css.slice(css.indexOf('input[type="range"]'))
  assert.match(rule.slice(0, 120), /touch-action:\s*pan-y/)
})
