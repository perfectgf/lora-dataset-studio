import assert from 'node:assert/strict';
import test from 'node:test';
import { launchSettings, launchText, mergeBatches, promptTexts, visibleBatch } from './promptBatch.js';

test('the history is read in both shapes the API has ever returned', () => {
  assert.deepEqual(
    promptTexts(['a string one', { prompt: 'an object one', count: 3 }, { count: 1 }, null]),
    ['a string one', 'an object one'],
  );
  assert.deepEqual(promptTexts(null), []);
});

test('a prompt deleted from the history leaves the batch by itself', () => {
  // The 🗑 button removes a prompt AND its images. A batch still holding it
  // would launch on a line the screen no longer shows.
  const history = [{ prompt: 'kept' }, { prompt: 'also kept' }];
  assert.deepEqual(visibleBatch(['kept', 'gone', 'also kept'], history),
    ['kept', 'also kept']);
  assert.deepEqual(visibleBatch([], history), []);
  assert.deepEqual(visibleBatch(['kept'], null), []);
});

test('nothing ticked sends the body it has always sent — same object, no key', () => {
  const gen = { negative: 'blurry', sampler: 'euler' };
  assert.equal(launchSettings(gen, []), gen);          // identity, not a copy
  assert.equal(launchSettings(gen, null), gen);
  assert.ok(!('prompts' in launchSettings(gen, [])));
});

test('N ticked prompts travel as N entries alongside the global settings', () => {
  const gen = { negative: 'blurry' };
  const body = launchSettings(gen, ['on a rooftop', 'in the snow', 'at night']);
  assert.deepEqual(body.prompts, ['on a rooftop', 'in the snow', 'at night']);
  // The batch rides WITH the settings, it never replaces them: the run must
  // still be the one the panel was showing.
  assert.equal(body.negative, 'blurry');
  // …and the caller's object is not mutated (it is component state).
  assert.ok(!('prompts' in gen));
});

test('one ticked prompt is still a batch of one — that is what was ticked', () => {
  assert.deepEqual(launchSettings({}, ['only this']).prompts, ['only this']);
});

test('the button keeps its surface’s verb and adds what the batch changes', () => {
  assert.equal(launchText(null, []), null);            // Test Studio, untouched
  assert.equal(launchText('Deploy 2 checkpoints, then generate', []),
    'Deploy 2 checkpoints, then generate');            // canvas, untouched
  assert.equal(launchText(null, ['a', 'b', 'c']), 'Run test · 3 prompts');
  assert.equal(launchText('Deploy 2 checkpoints, then generate', ['a', 'b']),
    'Deploy 2 checkpoints, then generate · 2 prompts');
  // One prompt is not announced as a batch: the button would be shouting about
  // a run that looks exactly like the ordinary one.
  assert.equal(launchText(null, ['a']), null);
});

test('🌐 Civitai picks join the batch after the history picks, once each', async () => {
  const { mergeBatches } = await import('./promptBatch.js');
  assert.deepEqual(mergeBatches(['a', 'b'], ['c', 'b', 'd']), ['a', 'b', 'c', 'd'],
    'a prompt ticked in both places is ONE pass');
  assert.deepEqual(mergeBatches([], ['x']), ['x']);
  assert.deepEqual(mergeBatches(['x'], []), ['x']);
  assert.deepEqual(mergeBatches(null, null), []);
  assert.deepEqual(mergeBatches(['a', ''], [42, 'a']), ['a'], 'only real prompt texts travel');
});

test('the merge compares TRIMMED but sends the original string', () => {
  // Le moteur (`_prompt_axis`) strippe puis dédoublonne. Une règle différente
  // ici ferait annoncer au compteur une cellule que le run ne rendra jamais.
  assert.deepEqual(mergeBatches(['a prompt'], ['  a prompt  ']), ['a prompt']);
  // …et la chaîne d'origine part telle quelle quand elle est seule.
  assert.deepEqual(mergeBatches(['a prompt\n'], []), ['a prompt\n']);
  // Les vides et les non-chaînes n'atteignent jamais le corps du lancement.
  assert.deepEqual(mergeBatches(['keep', '', '   ', null, 42], [undefined]), ['keep']);
});
