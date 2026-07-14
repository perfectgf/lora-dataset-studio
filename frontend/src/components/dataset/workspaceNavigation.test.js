import test from 'node:test';
import assert from 'node:assert/strict';
import { WORKSPACE_SECTIONS } from './workspaceSections.js';
import {
  getWorkspacePanelStatus,
  getWorkspacePanels,
  resolveWorkspaceLocation,
  withWorkspaceLocation,
} from './workspaceNavigation.js';

const BASE = Object.freeze({
  kind: 'character',
  hasSelectableImages: true,
  hasKeptImages: true,
  hasCaptionedKept: true,
  hasLeakMetadata: true,
  watermarkDetected: 0,
  unused: 0,
  hfPublish: false,
  trainingVisible: true,
  trainingStatusReady: true,
  trainingQueueCount: 0,
  studioVisible: false,
});

const ids = (section, overrides = {}) =>
  getWorkspacePanels(section, { ...BASE, ...overrides }).map((panel) => panel.id);

test('registry ids are unique within sections and target ids are globally unique', () => {
  const targets = [];
  for (const section of WORKSPACE_SECTIONS) {
    const panelIds = section.panels.map((panel) => panel.id);
    assert.equal(new Set(panelIds).size, panelIds.length, section.id);
    targets.push(...section.panels.map((panel) => panel.targetId));
  }
  assert.equal(new Set(targets).size, targets.length);
});

test('character destinations expose real character-only panels', () => {
  assert.deepEqual(ids('add'), ['reference', 'generate', 'import', 'scraper']);
  assert.deepEqual(ids('curation'), ['face-analysis', 'watermarks']);
  assert.deepEqual(ids('captions'), ['generate', 'leak-review', 'tools']);
});

test('concept and style destinations omit character-only or inapplicable panels', () => {
  assert.deepEqual(ids('add', { kind: 'concept' }), ['import', 'scraper']);
  assert.deepEqual(ids('curation', { kind: 'concept' }), ['watermarks']);
  assert.deepEqual(ids('captions', { kind: 'concept' }), ['generate', 'leak-review', 'tools']);
  assert.deepEqual(ids('captions', { kind: 'style' }), ['generate', 'tools']);
});

test('data and capability predicates expose only destinations that currently exist', () => {
  assert.deepEqual(ids('images', { hasSelectableImages: false }), ['review']);
  assert.deepEqual(ids('curation', { watermarkDetected: 2, unused: 3 }),
    ['face-analysis', 'watermarks', 'review-flagged', 'rejected-cleanup']);
  assert.deepEqual(ids('captions', { hasKeptImages: false, hasCaptionedKept: false }), ['generate']);
  assert.deepEqual(ids('export', { hfPublish: true, hasKeptImages: true }),
    ['import', 'training-zip', 'backup', 'hugging-face']);
  assert.deepEqual(ids('training', { studioVisible: true }),
    ['launch', 'advanced', 'checkpoints', 'studio']);
});

test('queue availability remains pending until the first truthful status response', () => {
  const pending = { ...BASE, trainingStatusReady: false };
  assert.equal(getWorkspacePanelStatus('training', 'queue', pending), 'pending');
  assert.equal(getWorkspacePanelStatus('training', 'queue', BASE), 'unavailable');
  assert.equal(getWorkspacePanelStatus('training', 'queue', { ...BASE, trainingQueueCount: 1 }), 'available');
});

test('invalid sections and panels normalize to truthful parent state', () => {
  assert.deepEqual(resolveWorkspaceLocation(new URLSearchParams('section=nope&panel=scraper'), BASE), {
    section: 'images', panel: null, pending: false, needsNormalization: true,
  });
  assert.deepEqual(resolveWorkspaceLocation(new URLSearchParams('section=add&panel=nope'), BASE), {
    section: 'add', panel: null, pending: false, needsNormalization: true,
  });
  assert.deepEqual(resolveWorkspaceLocation(new URLSearchParams('section=curation&panel=review-flagged'), BASE), {
    section: 'curation', panel: null, pending: false, needsNormalization: true,
  });
});

test('a pending queue URL is preserved until training status resolves', () => {
  assert.deepEqual(resolveWorkspaceLocation(
    new URLSearchParams('section=training&panel=queue'),
    { ...BASE, trainingStatusReady: false },
  ), {
    section: 'training', panel: 'queue', pending: true, needsNormalization: false,
  });
});

test('query updates preserve unrelated keys and clear panel on parent navigation', () => {
  const child = withWorkspaceLocation(new URLSearchParams('foo=bar&section=images'), 'add', 'scraper');
  assert.equal(child.toString(), 'foo=bar&section=add&panel=scraper');
  const parent = withWorkspaceLocation(child, 'captions', null);
  assert.equal(parent.toString(), 'foo=bar&section=captions');
});
