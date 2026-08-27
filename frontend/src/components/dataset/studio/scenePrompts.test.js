import test from 'node:test';
import assert from 'node:assert/strict';
import { combinedPromptBatch, joinScenePrompt, sceneSource, sceneThumbUrl,
  scenePromptList, toggleSceneIndex } from './scenePrompts.js';

const SCENES = [
  { label: 'Scene 1', prompt: 'street at dawn' },
  { label: 'Scene 2', prompt: 'close on her face' },
  { label: 'Scene 3', prompt: 'rooftop chase' },
];

test('scenePromptList keeps SCENE order, never tick order', () => {
  assert.deepEqual(scenePromptList(SCENES, [2, 0]),
    ['street at dawn', 'rooftop chase']);
});

test('scenePromptList ignores junk indices and promptless rows', () => {
  const scenes = [...SCENES, { label: 'Scene 4' }, null];
  assert.deepEqual(scenePromptList(scenes, [3, 4, 99, 1]), ['close on her face']);
  assert.deepEqual(scenePromptList(null, [0]), []);
  assert.deepEqual(scenePromptList(SCENES, null), []);
});

test('toggleSceneIndex adds then removes, returning new arrays', () => {
  const a = toggleSceneIndex([], 1);
  assert.deepEqual(a, [1]);
  assert.deepEqual(toggleSceneIndex(a, 1), []);
  assert.deepEqual(a, [1]);
});

test('combinedPromptBatch puts the history batch first, scenes after, in order', () => {
  assert.deepEqual(combinedPromptBatch(['from history'], SCENES, [1, 0]),
    ['from history', 'street at dawn', 'close on her face']);
  assert.deepEqual(combinedPromptBatch(null, SCENES, []), []);
});

test('joinScenePrompt appends the custom text after the caption', () => {
  assert.equal(joinScenePrompt('street at dawn', 'red dress'),
    'street at dawn, red dress');
  // Nothing typed (or only spaces): the caption runs untouched.
  assert.equal(joinScenePrompt('street at dawn', ''), 'street at dawn');
  assert.equal(joinScenePrompt('street at dawn', '   '), 'street at dawn');
  assert.equal(joinScenePrompt('street at dawn', undefined), 'street at dawn');
});

test('joinScenePrompt drops the caption’s closing punctuation before the join', () => {
  // Captioners end sentences; "…dawn., red dress" must never reach the sampler.
  assert.equal(joinScenePrompt('street at dawn.', 'red dress'),
    'street at dawn, red dress');
  assert.equal(joinScenePrompt('street at dawn, ', 'red dress'),
    'street at dawn, red dress');
  // The typed text keeps its own punctuation as typed.
  assert.equal(joinScenePrompt('street at dawn', 'red dress.'),
    'street at dawn, red dress.');
});

test('scenePromptList folds each ticked scene’s custom text in, by index', () => {
  assert.deepEqual(
    scenePromptList(SCENES, [2, 0], { 0: 'red dress', 2: 'at night' }),
    ['street at dawn, red dress', 'rooftop chase, at night']);
});

test('custom text on an UNTICKED scene changes nothing until it is ticked', () => {
  assert.deepEqual(scenePromptList(SCENES, [1], { 0: 'red dress' }),
    ['close on her face']);
  // …and the batch end-to-end: history first, extras only on ticked scenes.
  assert.deepEqual(
    combinedPromptBatch(['from history'], SCENES, [1], { 0: 'red dress', 1: 'smiling' }),
    ['from history', 'close on her face, smiling']);
});

test('a custom text alone cannot resurrect a promptless scene', () => {
  // Row 3 has no caption: it is skipped at load and must stay skipped even if
  // an index in extras points at it.
  const scenes = [...SCENES, { label: 'Scene 4' }, { label: 'Scene 5', prompt: '  ' }];
  assert.deepEqual(scenePromptList(scenes, [3, 4], { 3: 'red dress', 4: 'red dress' }), []);
});

test('sceneSource normalises the two payloads into ONE descriptor', () => {
  assert.deepEqual(sceneSource('bank', { bank_id: 7, bank_name: 'Chapter 1' }),
    { kind: 'bank', id: 7, name: 'Chapter 1' });
  assert.deepEqual(sceneSource('dataset', { dataset_id: 4, dataset_name: 'Lola' }),
    { kind: 'dataset', id: 4, name: 'Lola' });
  // A payload from the OTHER route is not silently accepted: loading a bank id
  // as a dataset would address every thumbnail at the wrong table.
  assert.equal(sceneSource('dataset', { bank_id: 7 }), null);
  assert.equal(sceneSource('bank', {}), null);
});

test('sceneThumbUrl addresses each surface the way that surface serves thumbs', () => {
  const bank = { kind: 'bank', id: 7, name: 'Chapter 1' };
  const dataset = { kind: 'dataset', id: 4, name: 'Lola' };
  assert.equal(sceneThumbUrl(bank, { image_id: 41 }), '/api/bank/7/thumb/41');
  assert.equal(sceneThumbUrl(dataset, { filename: 'p 001+a.png' }),
    '/api/dataset/4/thumb/p%20001%2Ba.png?s=256');
});

test('sceneThumbUrl returns nothing rather than a URL that cannot resolve', () => {
  // A bank card from before thumbnails, a dataset image still rendering, and a
  // panel with no source loaded: each must draw NO <img>, not a broken one.
  assert.equal(sceneThumbUrl({ kind: 'bank', id: 7 }, { filename: 'p000.jpg' }), '');
  assert.equal(sceneThumbUrl({ kind: 'dataset', id: 4 }, { image_id: 41 }), '');
  assert.equal(sceneThumbUrl(null, { image_id: 41 }), '');
  assert.equal(sceneThumbUrl({ kind: 'bank' }, { image_id: 41 }), '');
});
