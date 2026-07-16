import assert from 'node:assert/strict'
import test from 'node:test'

import {
  compatibleTrainingPresetSelection,
  filterTrainingPresets,
  isTrainingPresetCompatible,
  trainingPresetApplyPayload,
  trainingPresetDatasetKind,
  trainingPresetSnapshotScope,
} from './trainingPresets.js'

const presets = [
  { id: 'krea-style', train_type: 'krea', dataset_kind: 'style', variants: ['base', 'raw'] },
  { id: 'krea-character', train_type: 'krea', dataset_kind: 'character', variants: ['base'] },
  { id: 'z-style', train_type: 'zimage', kind: 'style', variants: ['base'] },
  { id: 7, train_type: 'krea', settings: {} }, // legacy DB preset: no kind metadata
]

test('preset scope reads dataset_kind and only dataset-kind values from legacy kind', () => {
  assert.equal(trainingPresetDatasetKind(presets[0]), 'style')
  assert.equal(trainingPresetDatasetKind(presets[2]), 'style')
  assert.equal(trainingPresetDatasetKind({ kind: 'training-preset' }), null)
})

test('preset list is exact by family and explicit dataset kind', () => {
  assert.deepEqual(
    filterTrainingPresets(presets, { trainType: 'krea', datasetKind: 'style' }).map((p) => p.id),
    ['krea-style', 7],
  )
  assert.equal(isTrainingPresetCompatible(presets[0], { trainType: 'zimage', datasetKind: 'style' }), false)
  assert.equal(isTrainingPresetCompatible(presets[1], { trainType: 'krea', datasetKind: 'style' }), false)
})

test('selection is cleared when a family or dataset-kind switch hides it', () => {
  assert.equal(compatibleTrainingPresetSelection('krea-style', presets,
    { trainType: 'krea', datasetKind: 'style' }), 'krea-style')
  assert.equal(compatibleTrainingPresetSelection('krea-style', presets,
    { trainType: 'zimage', datasetKind: 'style' }), '')
  assert.equal(compatibleTrainingPresetSelection('krea-style', presets,
    { trainType: 'krea', datasetKind: 'character' }), '')
})

test('variant-scoped preset is hidden and cannot apply on a different recipe', () => {
  const preset = { id: 'builtin-style-zimage-base', train_type: 'zimage', dataset_kind: 'style', variants: ['base'] }
  assert.equal(isTrainingPresetCompatible(preset,
    { trainType: 'zimage', datasetKind: 'style', variant: 'turbo' }), false)
  assert.equal(trainingPresetApplyPayload(preset,
    { trainType: 'zimage', datasetKind: 'style', variant: 'turbo' }), null)
})

test('apply plan uses preset id and the explicitly selected supported variant', () => {
  assert.deepEqual(trainingPresetApplyPayload(
    { id: 'builtin-style-zimage-base', train_type: 'zimage', dataset_kind: 'style', variants: ['base'] },
    { trainType: 'zimage', datasetKind: 'style', variant: 'base' },
  ), {
    preset_id: 'builtin-style-zimage-base',
    train_type: 'zimage',
    variant: 'base',
  })
})

test('mismatch produces no payload, so the caller performs no request', () => {
  let requests = 0
  const payload = trainingPresetApplyPayload(presets[0],
    { trainType: 'krea', datasetKind: 'character', variant: 'base' })
  if (payload) requests += 1
  assert.equal(payload, null)
  assert.equal(requests, 0)
})

test('save-current scope omits synthetic variants for single-recipe families', () => {
  assert.deepEqual(trainingPresetSnapshotScope(
    { trainType: 'flux', datasetKind: 'style', variant: 'turbo' },
  ), { train_type: 'flux', dataset_kind: 'style' })
  assert.deepEqual(trainingPresetSnapshotScope(
    { trainType: 'sdxl', datasetKind: 'style', variant: 'turbo' },
  ), { train_type: 'sdxl', dataset_kind: 'style' })
  assert.deepEqual(trainingPresetSnapshotScope(
    { trainType: 'zimage', datasetKind: 'style', variant: 'base' },
  ), { train_type: 'zimage', dataset_kind: 'style', variant: 'base' })
})
