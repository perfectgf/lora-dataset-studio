/* The pass has to be able to say WHO wrote it — including when two engines shared
   the work, which is exactly what the default 'auto' backend does without saying so. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { captionEngineBreakdown, captionEnginesSummary, captionResultSuffix,
  captionSkippedSuffix, CAPTION_WRITERS, CAPTION_ENGINE_WHY } from './captionEngines.js';

test('one engine wrote everything -> a sentence naming it', () => {
  assert.equal(captionEnginesSummary({ joycaption: 12 }), 'Written by JoyCaption.');
  assert.equal(captionEnginesSummary({ ollama: 12 }),
    'Written by the Ollama vision model.');
  // The chained case is ONE writer, not two: the stored text is Ollama's rewrite of
  // a JoyCaption draft, and naming either alone would be false.
  assert.equal(captionEnginesSummary({ joycaption_refined: 12 }),
    'Drafted by JoyCaption, rewritten by the Ollama vision model.');
});

test('the silent fallback is what this exists for: both engines, both counted', () => {
  // 'auto' with JoyCaption covering part of the batch — the case where captions come
  // out in two visibly different styles and nothing used to say why.
  assert.equal(captionEnginesSummary({ joycaption: 8, ollama: 4 }),
    '8 by JoyCaption · 4 by Ollama');
  // Canonical order, never the payload's key order.
  assert.equal(captionEnginesSummary({ ollama: 4, joycaption: 8 }),
    '8 by JoyCaption · 4 by Ollama');
  assert.equal(
    captionEnginesSummary({ ollama: 1, joycaption_refined: 2, joycaption: 3 }),
    '3 by JoyCaption · 2 by JoyCaption + Ollama · 1 by Ollama');
});

test('nothing to say stays silent — no invented author', () => {
  for (const empty of [undefined, null, {}, { joycaption: 0, ollama: 0 }, 'nope', 42]) {
    assert.equal(captionEnginesSummary(empty), '', `${JSON.stringify(empty)} must be silent`);
    assert.equal(captionResultSuffix(empty), '');
    assert.deepEqual(captionEngineBreakdown(empty), []);
  }
});

test('an engine this build never heard of is listed, not dropped', () => {
  // A future backend key must degrade to "we do not know that name" and stay
  // COUNTED. Swallowing it would rebuild the blind spot this module closes.
  const rows = captionEngineBreakdown({ joycaption: 2, florence: 5 });
  assert.deepEqual(rows.map((r) => r.key), ['joycaption', 'florence']);
  assert.equal(captionEnginesSummary({ florence: 5 }), 'Written by florence.');
  assert.equal(captionEnginesSummary({ joycaption: 2, florence: 5 }),
    '2 by JoyCaption · 5 by florence');
});

test('the toast suffix concatenates cleanly, or vanishes', () => {
  assert.equal(`12 captioned${captionResultSuffix({ joycaption: 12 })}`,
    '12 captioned · Written by JoyCaption.');
  assert.equal(`12 captioned${captionResultSuffix({})}`, '12 captioned');
});

test('the writer keys are the backend contract, and the copy is complete', () => {
  assert.deepEqual(CAPTION_WRITERS.map((w) => w.key),
    ['joycaption', 'joycaption_refined', 'ollama']);
  for (const w of CAPTION_WRITERS) {
    assert.ok(w.short && w.solo, `${w.key} needs both wordings`);
    assert.ok(w.solo.endsWith('.'), `${w.key} solo copy is a sentence`);
  }
  // The explanation names the mechanism AND the lever, so it is not a dead end.
  assert.match(CAPTION_ENGINE_WHY, /JoyCaption/);
  assert.match(CAPTION_ENGINE_WHY, /Ollama/);
  assert.match(CAPTION_ENGINE_WHY, /Options/);
});

/* A pass can FINISH having refused images — the engine says so per image and the
   batch keeps going. The result line has to carry that, or the only visible trace
   is a count smaller than the dataset. */
test('refused images are named with the reason the engine gave', () => {
  const reason = 'bank image rejects images above 8192 px per side or 16777216 pixels (got 3840x5760)';
  assert.equal(`37 captioned${captionSkippedSuffix({ skipped: 52, skipped_reason: reason })}`,
    `37 captioned — 52 skipped: ${reason}`);
});

test('a skip with no reason still gets counted, and a clean pass adds nothing', () => {
  assert.equal(captionSkippedSuffix({ skipped: 2, skipped_reason: '' }), ' — 2 skipped');
  assert.equal(captionSkippedSuffix({ skipped: 0, skipped_reason: 'ignored' }), '');
  assert.equal(captionSkippedSuffix({}), '');
  // An older backend that sends no counts at all must not print "NaN skipped".
  assert.equal(captionSkippedSuffix(undefined), '');
  assert.equal(captionSkippedSuffix({ skipped: 'lots' }), '');
});
