import assert from 'node:assert/strict';
import test from 'node:test';

import { filterSavedPrompts, normalizeSavedPrompt } from './savedPrompts.js';

test('an entry is read the same whether the API sent a string or an object', () => {
  // `/api/studio/recent-prompts` answers with strings before a Flask restart and
  // with objects after. Both must reach the UI as the same shape, or a freshly
  // updated install shows a list that behaves differently from a restarted one.
  assert.deepEqual(normalizeSavedPrompt('a woman on a couch'), {
    prompt: 'a woman on a couch', thumbnail: null, thumbDatasetId: null,
    liked: false, count: 0,
  });
  assert.deepEqual(normalizeSavedPrompt({
    prompt: 'a woman on a couch', thumbnail: 'x.png', thumb_dataset_id: 7,
    thumb_rating: 1, count: 12,
  }), {
    prompt: 'a woman on a couch', thumbnail: 'x.png', thumbDatasetId: 7,
    liked: true, count: 12,
  });
});

test('a thumbs-DOWN thumbnail is not reported as liked', () => {
  // thumb_rating is -1/0/1; only 1 means "an image you liked". Treating any
  // non-zero as liked would put a 👍 on a picture the user rejected.
  assert.equal(normalizeSavedPrompt({ prompt: 'p', thumb_rating: -1 }).liked, false);
  assert.equal(normalizeSavedPrompt({ prompt: 'p', thumb_rating: 0 }).liked, false);
});

test('a dataset id of 0 survives normalisation', () => {
  // `??` and not `||`: id 0 is a real dataset, and `|| null` would send the
  // thumbnail request to /api/dataset/null/img/…
  assert.equal(normalizeSavedPrompt({ prompt: 'p', thumb_dataset_id: 0 }).thumbDatasetId, 0);
});

test('an empty search shows the whole history, in order', () => {
  const items = ['b', 'a', 'c'];
  assert.deepEqual(filterSavedPrompts(items, ''), items);
  assert.deepEqual(filterSavedPrompts(items, '   '), items);
  assert.deepEqual(filterSavedPrompts(items, null), items);
});

test('every word must match, however far apart they sit', () => {
  // The measured shape of a real history: prompts are ~500 characters and open
  // the same way, so the words that tell them apart are deep in the sentence and
  // rarely adjacent. A single-substring search would find neither of these.
  const items = [
    { prompt: 'Photograph of a young woman with brown hair, standing in a bathroom, holding a mirror' },
    { prompt: 'Photograph of a young woman with brown hair, sitting on a couch near a window' },
  ];
  assert.deepEqual(filterSavedPrompts(items, 'bathroom mirror').map((i) => i.prompt),
    [items[0].prompt]);
  assert.deepEqual(filterSavedPrompts(items, 'mirror bathroom').map((i) => i.prompt),
    [items[0].prompt],
    'word order must not change the result — nobody remembers the order');
  assert.deepEqual(filterSavedPrompts(items, 'couch window').map((i) => i.prompt),
    [items[1].prompt]);
});

test('search is case-insensitive and can find nothing', () => {
  const items = ['A Woman In A BATHROOM'];
  assert.equal(filterSavedPrompts(items, 'bathroom').length, 1);
  assert.equal(filterSavedPrompts(items, 'BaThRoOm').length, 1);
  assert.equal(filterSavedPrompts(items, 'kitchen').length, 0);
  assert.equal(filterSavedPrompts(items, 'woman kitchen').length, 0,
    'one word missing is no match — the filter is an AND');
});

test('a missing or malformed list never throws', () => {
  assert.deepEqual(filterSavedPrompts(null, 'x'), []);
  assert.deepEqual(filterSavedPrompts(undefined, ''), []);
  assert.equal(normalizeSavedPrompt(null).prompt, '');
  assert.equal(filterSavedPrompts([{ }, null], 'x').length, 0);
});
