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
import test from 'node:test'
import { readFileSync } from 'node:fs'

const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8')
const workspace = read('../src/components/dataset/DatasetWorkspace.jsx')
const grid = read('../src/components/dataset/DatasetGrid.jsx')
const tile = read('../src/components/dataset/DatasetGridItem.jsx')

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

test('every write to the image itself still reads the conservative gate', () => {
  // Keep / reject / crop / delete / mirror / caption own the pixels and the row.
  // They must NOT have followed the retries onto a queue gate.
  assert.match(tile, /const refused = busy \? busyReason : null;/)
  const writes = tile.match(/disabled=\{busy[^}]*\}/g) || []
  assert.ok(writes.length >= 8,
    `expected the image writes to stay on \`busy\`, found ${writes.length}`)
})
