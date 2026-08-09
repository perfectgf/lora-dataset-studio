/**
 * 🧬 Blend SWEEP — plusieurs poids cochés par LoRA → un lot de combinaisons.
 *
 * Le curseur donnait un poids par LoRA, donc une pile = une image : comparer
 * « 0.8/0.6 » à « 0.6/0.8 » coûtait deux lancements. Les cases de poids en font
 * un balayage, et le produit cartésien part en un seul run.
 *
 * La logique est PURE et vit dans studio/loraStack.js — les DEUX surfaces
 * (Test Studio et ◉ LoRA Canvas) l'importent, donc elle est testée ici une fois.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import {
  BLEND_WARN_CELLS, BLEND_WEIGHT_CHIPS, blendComboLabel, blendCombinations,
  blendConfigCount, blendSweepCost, buildSelectionsPayload, cellCount,
  stackKey, stackWeightList, stackWeightSet,
} from '../src/components/dataset/studio/loraStack.js'

const A = { dataset_id: 1, checkpoint: 'z image\\a.safetensors', lora_label: 'margot', family: 'zimage' }
const B = { dataset_id: 2, checkpoint: 'z image\\b.safetensors', lora_label: 'telegram', family: 'zimage' }
const SEL = [A, B]
const kA = stackKey(A)
const kB = stackKey(B)

test('no box ticked = the slider governs, exactly as before the boxes existed', () => {
  // THE invariant of this feature: adding the boxes must not move a single
  // existing run. An untouched panel sweeps one weight per LoRA — the slider's.
  const weights = { [kA]: 0.85, [kB]: 0.35 }
  assert.deepEqual(stackWeightList(weights, {}, A), [0.85])
  assert.deepEqual(stackWeightList(weights, {}, B), [0.35])
  assert.deepEqual(blendCombinations(SEL, { weights }), [[0.85, 0.35]])
  assert.equal(blendConfigCount(SEL, { weights }), 1)
})

test('one box ticked is still exactly one configuration', () => {
  const sets = { [kA]: [0.8], [kB]: [0.6] }
  assert.deepEqual(blendCombinations(SEL, { sets }), [[0.8, 0.6]])
  assert.equal(blendConfigCount(SEL, { sets }), 1)
  // …and the boxes WIN over the slider once any is ticked, so the panel never
  // shows a ticked 0.8 while launching the slider's 0.85.
  assert.deepEqual(stackWeightList({ [kA]: 0.85 }, sets, A), [0.8])
})

test('the launch is the cartesian product, last LoRA varying fastest', () => {
  const sets = { [kA]: [0.6, 0.8], [kB]: [0.4, 1.0] }
  assert.deepEqual(blendCombinations(SEL, { sets }), [
    [0.6, 0.4], [0.6, 1.0], [0.8, 0.4], [0.8, 1.0],
  ])
  assert.equal(blendConfigCount(SEL, { sets }), 4)
  // 2 × 3 = 6, the brief's own example.
  assert.equal(blendConfigCount(SEL, { sets: { [kA]: [0.4, 0.6, 0.8], [kB]: [0.6, 1.0] } }), 6)
})

test('a mixed panel sweeps the ticked LoRA and pins the other to its slider', () => {
  const sets = { [kA]: [0.4, 0.6, 0.8] }
  const weights = { [kB]: 0.9 }
  assert.deepEqual(blendCombinations(SEL, { weights, sets }), [
    [0.4, 0.9], [0.6, 0.9], [0.8, 0.9],
  ])
})

test('ticked weights are clamped, rounded, de-duplicated and sorted', () => {
  const sets = { [kA]: [1.0, 0.4, 9, -3, 0.4, 0.5555, 'x', null] }
  // 9 → 5 (the ceiling, 2 until 2026-08-08), -3 → 0, 0.5555 → 0.56, the
  // duplicate 0.4 collapses, junk is dropped.
  assert.deepEqual(stackWeightSet(sets, A), [0, 0.4, 0.56, 1, 5])
  assert.deepEqual(stackWeightSet({}, A), [])
  assert.deepEqual(stackWeightSet({ [kA]: 'nope' }, A), [])
})

test('the cell count multiplies the configurations, and a lone LoRA is still no blend', () => {
  // 4 combinations × 2 images per seed = 8 cells.
  assert.equal(cellCount({ selectionCount: 2, count: 2, combine: true, configCount: 4 }), 8)
  // The ⚖ batch axis still doubles it.
  assert.equal(cellCount({ selectionCount: 2, count: 2, batchMult: 2, combine: true, configCount: 4 }), 16)
  // No configCount given = one configuration = the pre-sweep behaviour, byte for byte.
  assert.equal(cellCount({ selectionCount: 2, count: 2, combine: true }), 2)
  assert.equal(cellCount({ selectionCount: 1, count: 2, combine: true, configCount: 4 }), 0)
  // Compare mode is untouched by any of this.
  assert.equal(cellCount({ selectionCount: 2, strengthCount: 3, count: 2 }), 12)
})

test('the cost is announced before launch and WARNS instead of refusing', () => {
  // The Studio has no hard cell cap on purpose (build_matrix says so in as many
  // words: the queue is serial and the user sees the count first). A sweep is a
  // sweep — same rule, not a second one. So: warn, never block.
  const small = blendSweepCost({ configCount: 4, count: 1 })
  assert.equal(small.cells, 4)
  assert.equal(small.warn, false)
  const big = blendSweepCost({ configCount: 9, count: 4 })
  assert.equal(big.cells, 36)
  assert.equal(big.warn, true)
  assert.ok(big.minutes > 0)
  // Exactly at the threshold is not yet a warning; one over is.
  assert.equal(blendSweepCost({ configCount: BLEND_WARN_CELLS, count: 1 }).warn, false)
  assert.equal(blendSweepCost({ configCount: BLEND_WARN_CELLS + 1, count: 1 }).warn, true)
})

test('every cell can name its own combination', () => {
  assert.equal(blendComboLabel(SEL, [0.8, 0.6]), 'margot 0.8 × telegram 0.6')
  // Trailing zeros do not survive: "1" reads as a weight, "1.00" reads as noise.
  assert.equal(blendComboLabel(SEL, [1.0, 0.4]), 'margot 1 × telegram 0.4')
  assert.equal(blendComboLabel(SEL, [0.8]), 'margot 0.8 × telegram ?')
})

test('the payload carries the whole weight list, and a scalar for an older backend', () => {
  const sets = { [kA]: [0.6, 0.8] }
  const weights = { [kB]: 0.9 }
  assert.deepEqual(buildSelectionsPayload(SEL, { combine: true, weights, sets }), [
    { dataset_id: 1, checkpoint: 'z image\\a.safetensors', weight: 0.6, weights: [0.6, 0.8] },
    { dataset_id: 2, checkpoint: 'z image\\b.safetensors', weight: 0.9, weights: [0.9] },
  ])
  // Compare mode keeps the historical shape, still with no weight key at all.
  for (const entry of buildSelectionsPayload(SEL, { combine: false })) {
    assert.deepEqual(Object.keys(entry).sort(), ['checkpoint', 'dataset_id'])
  }
})

test('the offered boxes are round, usable weights inside the allowed range', () => {
  assert.ok(BLEND_WEIGHT_CHIPS.length >= 3)
  for (const w of BLEND_WEIGHT_CHIPS) assert.ok(w >= 0 && w <= 2, `${w} out of range`)
  assert.deepEqual([...BLEND_WEIGHT_CHIPS].sort((a, b) => a - b), BLEND_WEIGHT_CHIPS)
})
