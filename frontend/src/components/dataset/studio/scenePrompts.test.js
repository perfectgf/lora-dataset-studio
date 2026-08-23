import test from 'node:test';
import assert from 'node:assert/strict';
import { combinedPromptBatch, sceneSource, sceneThumbUrl, scenePromptList,
  toggleSceneIndex } from './scenePrompts.js';

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
