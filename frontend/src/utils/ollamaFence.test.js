import test from 'node:test';
import assert from 'node:assert/strict';
import {
  AUTO_RETRY_CAP_MS, OLLAMA_FENCE_CODE, blockingModelsLabel, fenceNoticeModel,
  isOllamaFenceError, keepAnswer, nextPollDelay, waitedLabel,
} from './ollamaFence.js';

test('only the structured code identifies a fence refusal', () => {
  assert.equal(isOllamaFenceError({ body: { code: OLLAMA_FENCE_CODE } }), true);
  // The raw-Response path (Describe) carries the code on the error itself.
  assert.equal(isOllamaFenceError({ code: OLLAMA_FENCE_CODE }), true);
  assert.equal(isOllamaFenceError({ body: { code: 'comfyui_recovery_required' } }), false);
  assert.equal(isOllamaFenceError(null), false);
  // A message that merely QUOTES the fence must not earn the unload button:
  // matching on prose would offer to evict a model over a coincidence.
  assert.equal(isOllamaFenceError(
    new Error('A local Ollama model is already in use outside LDS.')), false);
});

test('an answer is kept while its click is current, set aside once the user moved on', () => {
  // Every surface stands between the reply and the field with this; a panel
  // that answered `true` on both sides would write a stale answer with a
  // note beside it — so the helper is run, not read.
  const handle = (current, mounted = true) => ({ current: () => current, mounted: () => mounted });
  let told = 0;
  const tell = () => { told += 1; };
  // No handle (an action run outside the guard) keeps every answer.
  assert.equal(keepAnswer(undefined, tell), true);
  assert.equal(keepAnswer({}, tell), true);
  assert.equal(keepAnswer(handle(true), tell), true);
  assert.equal(told, 0, 'a current click is written without a word');
  assert.equal(keepAnswer(handle(false), tell), false);
  assert.equal(told, 1, 'a superseded click is set aside, and the surface is told');
  assert.equal(keepAnswer(handle(false, false), tell), false);
  assert.equal(told, 1, 'nobody is told once the panel went away');
  // The quiet surfaces (🔎 Describe) pass nothing to say.
  assert.equal(keepAnswer(handle(false)), false);
});

test('the vigil backs off instead of hammering a local daemon for ten minutes', () => {
  assert.equal(nextPollDelay(0), 2000);
  assert.equal(nextPollDelay(29_000), 2000);
  assert.equal(nextPollDelay(30_000), 5000);
  assert.equal(nextPollDelay(119_000), 5000);
  assert.equal(nextPollDelay(120_000), 15_000);
  assert.equal(nextPollDelay(undefined), 2000);
  // The cap sits past Ollama's own ~5 min idle unload, so the commonest way
  // out of this block happens well inside the wait.
  assert.ok(AUTO_RETRY_CAP_MS > 5 * 60 * 1000);
});

test('the waited counter reads as a duration, not a millisecond count', () => {
  assert.equal(waitedLabel(0), '0s');
  assert.equal(waitedLabel(12_400), '12s');
  assert.equal(waitedLabel(65_000), '1m 05s');
  assert.equal(waitedLabel(200_000), '3m 20s');
});

test('the blocking models are named, not dumped', () => {
  assert.equal(blockingModelsLabel([]), null);
  assert.equal(blockingModelsLabel(['a:8b']), 'a:8b');
  assert.equal(blockingModelsLabel(['a', 'b']), 'a and b');
  assert.equal(blockingModelsLabel(['a', 'b', 'c', 'd']), 'a, b and 2 more');
  assert.equal(blockingModelsLabel(null), null);
});

test('nothing is drawn when nothing is blocked', () => {
  assert.equal(fenceNoticeModel(null), null);
  assert.equal(fenceNoticeModel({}), null);
});

test('waiting says what holds the model, how long, and offers both ways out', () => {
  const m = fenceNoticeModel({ phase: 'waiting', models: ['other:8b'], elapsedMs: 12_000 });
  assert.match(m.headline, /Waiting for the model to be released/);
  assert.match(m.headline, /\(12s\)/);
  assert.match(m.detail, /other:8b/);
  // The promise the whole feature rests on: it resumes by itself.
  assert.match(m.detail, /on its own the moment it is free/);
  assert.equal(m.canUnload, true);
  assert.equal(m.canCancel, true);
  assert.equal(m.busy, false);
});

test('the consent click and the automatic resume both read as work in progress', () => {
  const unloading = fenceNoticeModel({ phase: 'unloading', models: ['other:8b'] });
  assert.equal(unloading.busy, true);
  assert.equal(unloading.canUnload, false);
  const retrying = fenceNoticeModel({ phase: 'retrying', models: ['other:8b'] });
  assert.equal(retrying.busy, true);
  assert.match(retrying.headline, /picking up where you left off/);
});

test('giving up hands the decision back instead of spinning forever', () => {
  const m = fenceNoticeModel({ phase: 'gave-up', models: ['other:8b'], elapsedMs: AUTO_RETRY_CAP_MS });
  assert.match(m.headline, /Still in use after 10 minutes/);
  assert.equal(m.canUnload, true);
  assert.equal(m.canCancel, false);
});

test('a block with no model name still says something true', () => {
  const m = fenceNoticeModel({ phase: 'waiting', models: [], elapsedMs: 0,
                              message: 'A local Ollama model is already in use outside LDS.' });
  assert.match(m.detail, /already in use outside LDS/);
});

test('the notice names the server the fence state reports', () => {
  // Someone on LM Studio told to look in Ollama looks at a daemon that holds
  // nothing. The provider comes from the fence state, not from the refusal.
  const lm = fenceNoticeModel({ phase: 'waiting', models: ['qwen/qwen3-vl-4b'], elapsedMs: 0,
                               provider: 'lmstudio' });
  assert.match(lm.detail, /qwen\/qwen3-vl-4b in LM Studio/);
  assert.doesNotMatch(lm.detail, /Ollama/);
  const gaveUp = fenceNoticeModel({ phase: 'gave-up', models: [], elapsedMs: AUTO_RETRY_CAP_MS,
                                   provider: 'lmstudio' });
  assert.match(gaveUp.detail, /a model in LM Studio/);
  // No provider yet (the first poll has not answered): Ollama, the default.
  const first = fenceNoticeModel({ phase: 'waiting', models: ['other:8b'], elapsedMs: 0 });
  assert.match(first.detail, /other:8b in Ollama/);
});
