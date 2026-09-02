import test from 'node:test'
import assert from 'node:assert/strict'

import { VIDEO_DATASET_SECTIONS, isVideoDatasetSection } from './videoDatasetSections.js'
import {
  PANEL_STATUS, visibleVideoDatasetSections, getVideoDatasetPanel,
  getVideoDatasetPanelStatus, getVideoDatasetPanels,
  resolveVideoDatasetLocation, withVideoDatasetLocation,
} from './videoDatasetNavigation.js'

// A dataset with everything switched on, so a test can turn ONE thing off.
const full = {
  selected: 2, clips: 6, requiresReferences: true, checkpointGroups: 1,
}
const params = (qs) => new URLSearchParams(qs)

// ---- the rail is data, and every entry of it must be answerable --------------

test('every section and every panel declares a predicate the code really has', () => {
  // Exercised rather than counted: each `when` is looked up through the same
  // path the rail uses, and UNKNOWN is what a missing predicate returns.
  for (const section of VIDEO_DATASET_SECTIONS) {
    assert.ok(isVideoDatasetSection(section.id))
    assert.ok(section.title && section.icon && section.eyebrow && section.description,
      `${section.id} is missing a rail field`)
    assert.ok(section.panels.length > 0, `${section.id} has no destination`)
    for (const panel of section.panels) {
      assert.notEqual(getVideoDatasetPanelStatus(section.id, panel.id, full),
        PANEL_STATUS.UNKNOWN,
        `${section.id}/${panel.id} names a predicate nothing implements`)
      assert.ok(panel.targetId.startsWith('vds-'),
        `${section.id}/${panel.id} has no anchor to scroll to`)
    }
  }
  assert.equal(isVideoDatasetSection('curation'), false)
  assert.equal(getVideoDatasetPanel('clips', 'nope'), null)
  assert.equal(getVideoDatasetPanelStatus('clips', 'nope', full), PANEL_STATUS.UNKNOWN)
})

test('every anchor id in the rail is unique - two panels cannot share a target', () => {
  const seen = new Set()
  for (const section of VIDEO_DATASET_SECTIONS) {
    for (const panel of section.panels) {
      assert.equal(seen.has(panel.targetId), false, `${panel.targetId} is declared twice`)
      seen.add(panel.targetId)
    }
  }
})

// ---- References comes and goes, and that is the whole point ------------------

test('References is absent for a target that does not train on control images', () => {
  const withRefs = visibleVideoDatasetSections(full).map((s) => s.id)
  assert.deepEqual(withRefs, ['clips', 'captions', 'references', 'training', 'checkpoints', 'studio'])
  const without = visibleVideoDatasetSections({ ...full, requiresReferences: false })
    .map((s) => s.id)
  assert.deepEqual(without, ['clips', 'captions', 'training', 'checkpoints', 'studio'])
  // A section with no `when` is never hidden, whatever the context says.
  assert.deepEqual(visibleVideoDatasetSections({}).map((s) => s.id),
    ['clips', 'captions', 'training', 'checkpoints', 'studio'])
})

// ---- panels appear on the state they point at, not on hope -------------------

test('Bulk actions needs a selection; Caption tools needs clips, not captions', () => {
  assert.deepEqual(getVideoDatasetPanels('clips', full).map((p) => p.id), ['review', 'bulk'])
  assert.deepEqual(getVideoDatasetPanels('clips', { ...full, selected: 0 }).map((p) => p.id),
    ['review'])
  assert.deepEqual(getVideoDatasetPanels('captions', full).map((p) => p.id), ['list', 'tools'])
  // Gated on CLIPS, not on captions: `prefix` is written for the silent ones, so
  // an entirely uncaptioned set is exactly when the tools are wanted.
  assert.deepEqual(getVideoDatasetPanels('captions', { ...full, captioned: 0 }).map((p) => p.id),
    ['list', 'tools'])
  assert.deepEqual(getVideoDatasetPanels('captions', { ...full, clips: 0 }).map((p) => p.id),
    ['list'])
  assert.deepEqual(getVideoDatasetPanels('nope', full), [])
})

test('Checkpoints & LoRAs and Studio are sections of their own, always in the rail', () => {
  // Checkpoints used to be a jump destination inside Training, shown only once
  // a run had brought files back. It is a section now, like its image twin: an
  // empty one says "no checkpoints yet", a vanished entry says nothing.
  assert.deepEqual(getVideoDatasetPanels('training', full).map((p) => p.id), ['launch'])
  assert.deepEqual(getVideoDatasetPanels('checkpoints', {}).map((p) => p.id), ['manager'])
  assert.deepEqual(getVideoDatasetPanels('studio', {}).map((p) => p.id), ['launcher'])
})

// ---- deep links: a stale one must degrade, never break ------------------------

test('no section asked for lands on Clips, because a video set is already full', () => {
  assert.deepEqual(resolveVideoDatasetLocation(params(''), full),
    { section: 'clips', panel: null, needsNormalization: true })
  assert.deepEqual(resolveVideoDatasetLocation(params('section=nonsense'), full),
    { section: 'clips', panel: null, needsNormalization: true })
})

test('a link to References opened on a dataset that has none falls back to Clips', () => {
  assert.deepEqual(
    resolveVideoDatasetLocation(params('section=references'), full),
    { section: 'references', panel: null, needsNormalization: false })
  assert.deepEqual(
    resolveVideoDatasetLocation(params('section=references'), { ...full, requiresReferences: false }),
    { section: 'clips', panel: null, needsNormalization: true })
})

test('a panel that is not available drops to its section rather than 404-ing', () => {
  assert.deepEqual(
    resolveVideoDatasetLocation(params('section=checkpoints&panel=manager'), full),
    { section: 'checkpoints', panel: 'manager', needsNormalization: false })
  // The old deep link (Checkpoints as a panel of Training) degrades to Training.
  assert.deepEqual(
    resolveVideoDatasetLocation(params('section=training&panel=checkpoints'), full),
    { section: 'training', panel: null, needsNormalization: true })
  assert.deepEqual(
    resolveVideoDatasetLocation(params('section=clips&panel=invented'), full),
    { section: 'clips', panel: null, needsNormalization: true })
})

test('writing a location keeps the rest of the query string', () => {
  const next = withVideoDatasetLocation(params('section=clips&panel=bulk&foo=1'), 'training')
  assert.equal(next.get('section'), 'training')
  assert.equal(next.get('panel'), null)        // a section change drops the panel
  assert.equal(next.get('foo'), '1')
  const withPanel = withVideoDatasetLocation(params('foo=1'), 'captions', 'tools')
  assert.equal(withPanel.get('panel'), 'tools')
})
