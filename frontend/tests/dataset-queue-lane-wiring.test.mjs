/**
 * The wiring of the two queue-lane gates, pinned by grep.
 *
 * `activityLanes.js` decides WHAT blocks what, and its own tests cover that. The
 * bugs this file exists for were one level down, in the plumbing, where no unit
 * test could see them — both shipped green:
 *
 *   * the tiles were handed `ds.improveBusy` for their 🔄 / ✏️ retries, which
 *     enqueue a plain 'generate'. `improveBusy` is true for the whole length of
 *     an ✨ improve batch (deliberately — the backend refuses a second one), so
 *     retrying a tile during a batch stayed impossible: GitHub #44's exact
 *     symptom, on the surface its own release note promised;
 *   * `onRegenerate` was still withheld on the OLD lock while the button's
 *     `disabled` had moved to the new one, so the 🔄 lit up and did nothing.
 *
 * Both are properties of which prop goes where. A render test would not catch
 * them either (it mounts the tile directly), so the contract is read from the
 * source, the way the repo already pins `improveRerun`'s tile wiring.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'

const read = readSource
// Comments in these files quote the very patterns under test (one explains that
// reads carry NO `disabled={busy}`), so a naive grep reports its own
// documentation as a violation. Match the code only.
const code = (src) => src.replace(/^\s*\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '')
const workspace = read('src/components/dataset/DatasetWorkspace.jsx')
const grid = read('src/components/dataset/DatasetGrid.jsx')
const tile = read('src/components/dataset/DatasetGridItem.jsx')
const lightbox = read('src/components/dataset/DatasetLightbox.jsx')

test('the workspace hands the grid BOTH lane gates, from their own sources', () => {
  assert.match(workspace, /improveBusy=\{ds\.improveBusy\}/)
  assert.match(workspace, /generateBusy=\{ds\.generationBusy\}/)
  // The single merged flag is what caused the bug; it must not come back.
  assert.doesNotMatch(workspace, /queueBusy=\{/)
  assert.doesNotMatch(workspace, /generateBusy=\{ds\.improveBusy\}/)
})

test('the grid forwards the GENERATE gate to the tiles, not the improve one', () => {
  assert.match(grid, /improveBusy=\{improveLaunchBusy\} generateBusy=\{generateLaunchBusy\}/)
  assert.match(grid, /const improveLaunchBusy = \(improveBusy \?\? busy\)/)
  assert.match(grid, /const generateLaunchBusy = \(generateBusy \?\? busy\)/)
})

test('the ✨ improve batch button keeps the IMPROVE gate', () => {
  // It is the one launch the backend really does refuse twice (409).
  assert.match(grid, /disabled=\{improveLaunchBusy \|\| !!improveLabel \|\| !!blocked\}/)
})

test('no handler is withheld behind a lock its button no longer reads', () => {
  // The trap that produced a lit-up, inert 🔄. onView was fixed for this once
  // already; onRegenerate and onReimprove now travel the same way.
  for (const handler of ['onRegenerate', 'onReimprove', 'onView'])
    assert.match(grid, new RegExp(`${handler}=\\{${handler}\\}`),
      `${handler} must be handed over unconditionally`)
  assert.doesNotMatch(grid, /onRegenerate=\{bulkBusy \? undefined : onRegenerate\}/)
})

test('the tile reads each button against its own lane', () => {
  // 🔄 and ✏️ enqueue a 'generate'; 🔄✨ is improve work.
  assert.match(tile, /const generateRefused = \(generateBusy \?\? busy\);/)
  assert.match(tile, /const improveRefused = \(improveBusy \?\? busy\);/)
  assert.match(tile, /disabled=\{generateRefused\}[\s\S]{0,200}?Regenerate this variation/)
  assert.match(tile, /disabled=\{improveRefused \|\| !rerunImprove\.enabled\}/)
})

test('curating an image reads its own gate, and no write is left on the blanket', () => {
  // Keep/reject, caption, delete, score, watermark: queued work is not a reason
  // to refuse them (every one is defended server-side), a pass that owns the
  // ROWS still is. Verified in the code before unblocking: `delete_image`
  // cancels the in-flight job and refuses when it cannot prove it,
  // `gpu_exclusive_vision_window` is fail-closed, `crop_image` needs a file.
  assert.match(tile, /const curationRefused = \(curationBusy \?\? busy\);/)
  assert.match(workspace, /curationBusy=\{ds\.curationBusy\}/)
  assert.match(grid, /const curationWriteBusy = \(curationBusy \?\? busy\)/)
  // Nothing may be left reading the old blanket: a write still on `busy` would
  // be grey while its neighbours are live, with no way to tell why.
  const stragglers = code(tile).match(/disabled=\{busy[^}]*\}/g) || []
  assert.deepEqual(stragglers, [],
    `these writes were left on the old blanket: ${stragglers.join(', ')}`)
})

// The one case queued work really does make awkward — and it is per TILE, not
// global: the upscale copied its source at enqueue, so it would come back as an
// upscale of the version from before the edit.
test('editing the pixels waits for an upscale of THAT image, and says so', () => {
  for (const source of [tile, lightbox]) {
    assert.match(source, /const pixelEditRefused = curationRefused \|\| upscaleRendering;/)
    assert.match(source, /upscale of the version from before your edit/)
  }
  // Crop, mirror and rotate read it; keep/reject and captions do not — they
  // touch no pixels.
  assert.match(tile, /disabled=\{pixelEditRefused\}/)
  assert.match(lightbox, /onClick=\{mirror\} disabled=\{pixelEditRefused \|\| mirrorBusy\}/)
})

// H2's trap, on the lightbox this time: a handler cut on a different lock than
// the button's own `disabled` gives a live button that does nothing.
test('no lightbox handler guards on a lock its button no longer reads', () => {
  assert.match(lightbox, /if \(!onMirror \|\| pixelEditRefused \|\| mirrorBusy\) return;/)
  assert.match(lightbox, /if \(!onRotate \|\| pixelEditRefused \|\| mirrorBusy\) return;/)
  assert.doesNotMatch(lightbox, /if \(!onMirror \|\| busy \|\| mirrorBusy\) return;/)
})
