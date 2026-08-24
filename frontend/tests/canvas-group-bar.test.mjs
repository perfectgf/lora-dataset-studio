/**
 * 🖼🖼 A group of pinned images must stay MOVABLE and CLOSABLE.
 *
 * The bug this pins down: a group's title bar is drawn on board space ABOVE its
 * own strip, and `LaneImages` renders its entries as absolutely-positioned
 * siblings with no z-index. As an ordinary descendant of the strip, the bar was
 * therefore painted over by any picture the board placed on that space — and it
 * carries ALL THREE of a group's affordances, so one picture above a strip left
 * it impossible to move, to export AND to close at once.
 *
 * Measured on the real DOM before the fix: a picture pinned flush above a
 * two-image strip made 5 of 11 points sampled along the bar hand the pointer to
 * the picture instead of the bar.
 *
 * Two halves, both asserted here without a browser:
 *   · RENDER ORDER — the bars are a layer drawn after every picture and strip;
 *   · PLACEMENT — the placers know the bar occupies board space, so ✦ Tidy up
 *     and 📌 Pin all cannot create the overlap on their own.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'

import { layoutBoxes, layoutImageNodes, occupiedBox } from '../src/utils/canvasImageGroups.js'
import { groupBarHeight, groupBarMaxHeight, isNodeControlTarget, nodePointerIntent }
  from '../src/utils/canvasNodeChrome.js'

const read = (rel) => readSource(`src/${rel}`)
const CANVAS = read('components/canvas/LineageCanvas.jsx')
const BAR = read('components/canvas/CanvasGroupBar.jsx')
const GROUP = read('components/canvas/CanvasImageGroup.jsx')

const img = (id, x, y, w, h, extra = {}) => ({
  imageId: id, x, y, w, h, visible: true, groupId: null, groupPos: null,
  image: { url: 'u', record_id: 1, step: 1000 }, ...extra,
})

const STRIP = [
  img(1, 100, 300, 200, 150, { groupId: 'g1', groupPos: 0 }),
  img(2, 300, 300, 200, 150, { groupId: 'g1', groupPos: 1 }),
]

/* --- the invariant, in as many words -------------------------------------- */

test('✕ IS ALWAYS REACHABLE: the bars are drawn after every picture and strip', () => {
  // A node that can be neither moved nor removed is the worst state this board
  // can reach, and closing is the way out of every other one. So the bars are a
  // LAYER: their block comes after the layout block that draws the pictures.
  const pictures = CANVAS.indexOf('{layout.map((r) => (r.kind === \'group\' ? (')
  const bars = CANVAS.indexOf('<CanvasGroupBar')
  assert.ok(pictures > 0, 'the layout pass must exist')
  assert.ok(bars > pictures, 'the bar layer must be rendered AFTER the pictures')
  // …and it must be a pass of its own, not a child of the strip again.
  assert.doesNotMatch(GROUP, /data-canvas-group-bar/)
  assert.match(BAR, /data-canvas-group-bar/)
  assert.match(BAR, /z-10/)
})

test('the bar still answers the frame the way the pointer handler reads it', () => {
  // Moving a group acts on the ANCHOR, and the handler reads that off the
  // closest [data-canvas-group]. Moving the bar out of the strip must not have
  // taken those attributes with it.
  for (const attr of ['data-canvas-group=""', 'data-dataset-id', 'data-anchor-id']) {
    assert.ok(BAR.includes(attr), `the bar must carry ${attr}`)
  }
  // And the two chrome rules still classify it: a click on its buttons is a
  // control (never a drag), the bar itself is the group's grip.
  const inBar = (sel) => ({ closest: (q) => (sel.includes(q) ? {} : null) })
  assert.equal(isNodeControlTarget(inBar(['[data-canvas-group-bar] button'])), true)
  assert.equal(nodePointerIntent(inBar(['[data-canvas-group-bar] button']), 'mouse'), 'control')
  assert.equal(nodePointerIntent(inBar(['[data-canvas-group-bar]']), 'touch'), 'group-move')
})

/* --- the placement half --------------------------------------------------- */

test('a group OCCUPIES the board space its bar needs, not just its strip', () => {
  const [group] = layoutImageNodes(STRIP)
  assert.equal(group.kind, 'group')
  const box = occupiedBox(group)
  const bar = groupBarMaxHeight(group.h)
  assert.ok(bar > 0)
  assert.equal(box.y, group.y - bar, 'the reserved box starts above the strip')
  assert.equal(box.h, group.h + bar)
  assert.equal(box.x, group.x)
  assert.equal(box.w, group.w)
  // A lone picture occupies exactly itself — nothing else changes shape.
  const [single] = layoutImageNodes([img(9, 0, 0, 10, 10)])
  assert.deepEqual(occupiedBox(single), { x: 0, y: 0, w: 10, h: 10 })
})

test('the reservation is the WORST case, because the bar grows as you zoom OUT', () => {
  // This is the detail that explains why it was met on a real board: the bar is
  // counter-scaled, so a gap that clears it at 100 % does not clear it at 40 %.
  const h = 150
  const at100 = groupBarHeight(1, h)
  const at40 = groupBarHeight(0.4, h)
  assert.ok(at40 > at100, 'the bar must be taller when zoomed out')
  assert.equal(at40, groupBarMaxHeight(h), 'and it saturates at the reserved max')
  // Reserving the current height would be a bug that only appears on zoom-out.
  assert.ok(groupBarMaxHeight(h) >= at100)
})

test('the placers are handed the reserved boxes, so Tidy up cannot create the overlap', () => {
  const layout = layoutImageNodes([...STRIP, img(9, 100, 100, 200, 130)])
  const boxes = layoutBoxes(layout)
  const group = layout.find((r) => r.kind === 'group')
  const reserved = boxes.find((b) => b.y === group.y - groupBarMaxHeight(group.h))
  assert.ok(reserved, 'layoutBoxes must report the group with its bar');
  // Both placement call sites go through the same helper — two answers to "how
  // much board does this take" is how they would drift apart again. ✦ Tidy up
  // asks it through `tidyGroupRows` (which brings a strayed strip home and
  // hands back the footprints it landed on, bar included — see
  // utils/canvasPinBatch.test.js), and feeds exactly those to the placer that
  // puts the lone pictures down. That sequence lives in `tidyLaneRows` because
  // it has a SECOND caller: the lane stack, which has to reserve the room the
  // tidy layout will need before the button is ever pressed.
  const batch = read('utils/canvasPinBatch.js')
  assert.match(batch, /tidyGroupRows/)
  assert.match(batch, /existing:\s*strips\.boxes/)
  assert.match(read('pages/CanvasPage.jsx'), /tidyLaneRows/)
  assert.match(CANVAS, /tidyLaneReach/)
  assert.match(CANVAS, /layoutBoxes\(layoutImageNodes\(visibleImageNodes\(laneMap\)\)\)/)
})
