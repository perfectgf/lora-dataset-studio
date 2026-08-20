import test from 'node:test';
import assert from 'node:assert/strict';
import { combinedPromptBatch, scenePromptList, toggleSceneIndex } from './scenePrompts.js';

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
