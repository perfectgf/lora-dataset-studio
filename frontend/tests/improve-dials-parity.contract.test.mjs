/**
 * ✨ Upscale & improve offers the SAME dials on both surfaces.
 *
 * The repo's standing rule is full parity between Bank and Dataset — the dials,
 * the launch window and the counts, not just the pass underneath. Improve had
 * drifted the other way for months and this wave widened it by one notch: the
 * dataset/gallery window gained the preset's LoRA strengths while the bank's
 * pass still answered "the improve instruction (Settings ▸ Engines)" — a list of
 * things it obeys and a pointer somewhere else to change them.
 *
 * It is the SAME pass, which is what makes the divergence indefensible rather
 * than merely untidy: image_bank_service._improve_job enqueues through the
 * dataset's own _enqueue_improve / _improve_enqueue_profile, so a bank improve
 * reads the very same instruction, preset, strengths and output size.
 *
 * Ported with the maintainer's explicit sign-off (2026-09-01), which his own
 * rule requires for any difference between the two surfaces.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { readSource } from './support/readSource.mjs'

/** Every surface that lets someone START an improve, and how it mounts the
 *  shared panel. A new one must be added here — that is the point of the list. */
const SURFACES = [
  { file: 'src/components/shared/ImproveModal.jsx',
    what: 'the ✨ window opened from any lightbox (gallery, canvas, studio, dataset)' },
  { file: 'src/components/dataset/DatasetGrid.jsx',
    what: "the dataset's bulk toolbar, which improves a whole selection" },
  { file: 'src/components/bank/BankEditPanel.jsx',
    what: "the bank's ✨ launch window (PassDialog), a batch over a whole bank" },
]

for (const { file, what } of SURFACES) {
  test(`${file} carries the improve dials — ${what}`, () => {
    const src = readSource(file)
    assert.match(src, /import KleinImproveNote/,
      'the panel is shared, never restated: one instruction, one preset, one truth')
    assert.match(src, /<KleinImproveNote\b/)
  })
}

test('the bank mounts them Klein-only, and inside the launch window', () => {
  const src = readSource('src/components/bank/BankEditPanel.jsx')
  // SeedVR2 is a restoration: no instruction, no LoRA chain. Offering the dials
  // beside it would name settings that run cannot use.
  assert.match(src, /\{kleinPicked && <KleinImproveNote/)
  // In the dialog's per-run block, not loose under the panel — that block is
  // where every other pass puts what belongs to THIS run.
  const dialog = src.split('<PassDialog passId="improve"')[1]
  assert.ok(dialog, 'the improve PassDialog is gone')
  assert.match(dialog.split('</PassDialog>')[0], /<KleinImproveNote/)
})

test('the bank stops pointing at Settings for what it now offers', () => {
  const passes = readSource('src/components/bank/bankPasses.js')
  const improve = passes.split('  improve: {')[1].split('\n  },')[0]
  assert.match(improve, /editable in this window/)
  assert.doesNotMatch(improve, /The improve instruction \(Settings/,
    'the line that sent the reader away is the one this port replaces')
})

test('the app-wide sentence is true on a bank, not only in a dataset', () => {
  // The panel states its own reach unconditionally; on a bank the old wording
  // ("in every dataset") described a scope the reader is not even in.
  const editor = readSource('src/components/dataset/kleinImproveEditor.js')
  const scope = editor.split('export const IMPROVE_SCOPE_NOTE =')[1].split(';')[0]
  assert.match(scope, /every dataset/)
  assert.match(scope, /every bank/)
})
