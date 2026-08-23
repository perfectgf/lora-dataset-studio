// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
// 🔤 Push down — the wording of an exclusion CLIP can only approximate.
//
// The failure this whole feature answers is silent: CLIP ignores "without", so
// "a woman without a hat" returns hats, confidently, with a full result list and
// no signal that anything went wrong (measured: 60% of the top 60 carried a
// bikini for "a woman without a bikini", against a 10.1% base rate). Every test
// here defends one of the two halves of the fix — catching the phrasing before
// it fails, and never letting the replacement over-promise.

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CLIP_LIMITS, PUSH_DOWN_DEFAULT_STRENGTH, PUSH_DOWN_STRENGTHS, pushDownCaveat,
  pushDownNote, limitsSentence, suggestPushDown, withoutNegation,
} from './bankTextSearch.js';

// --- catching the phrasing that silently inverts -----------------------------
test('the phrasings a user actually reaches for are all caught', () => {
  assert.equal(suggestPushDown('a woman without a hat'), 'a hat');
  assert.equal(suggestPushDown('a woman with no glasses'), 'glasses');
  assert.equal(suggestPushDown('a woman, not blonde'), 'blonde');
  assert.equal(suggestPushDown('portrait excluding sunglasses'), 'sunglasses');
  assert.equal(suggestPushDown('a woman WITHOUT A HAT'), 'a hat');
});

test('a query with nothing to negate suggests nothing', () => {
  assert.equal(suggestPushDown('brunette outdoors, wide shot'), '');
  assert.equal(suggestPushDown(''), '');
  assert.equal(suggestPushDown(null), '');
});

test('a long clause is not mistaken for a trait', () => {
  // A trait is one to three words. Past that it is a sentence, and offering to
  // push down half a clause is worse than staying quiet.
  assert.equal(suggestPushDown('a woman without anything covering her face'), '');
});

test('accepting the suggestion leaves a clean positive query', () => {
  assert.equal(withoutNegation('a woman without a hat'), 'a woman');
  assert.equal(withoutNegation('a woman with no glasses'), 'a woman');
  // Nothing to strip ⇒ the query is returned untouched, never mangled.
  assert.equal(withoutNegation('brunette outdoors'), 'brunette outdoors');
});

// --- never promising an absence ----------------------------------------------
test('the caveat promises a push-down and refuses to promise absence', () => {
  const c = pushDownCaveat();
  assert.match(c, /cannot guarantee/i);
  assert.match(c, /Nothing is filtered out/i);
  assert.doesNotMatch(c, /\bremoves\b|\bexcludes\b|\bhides\b/i);
});

test('the limits copy sends the user to the field instead of to "without"', () => {
  const s = limitsSentence();
  assert.match(s, /Push down/);
  assert.match(s, /without/);
  const negation = CLIP_LIMITS.find((l) => /Negation/.test(l));
  assert.match(negation, /Push down/);
});

// --- reporting what it did, measured on this bank ----------------------------
test('a push-down that worked says what it moved and how it compares', () => {
  const s = pushDownNote({
    push_down: 'a hat', push_down_moved: 7, image_ids: [1, 2, 3, 4, 5, 6, 7, 8],
    push_down_median: { pool: 0.21, results: 0.14 },
  });
  assert.match(s, /changed 7 places of 8/);
  assert.match(s, /0\.14/);
  assert.match(s, /0\.21/);
  assert.match(s, /below/);
  assert.match(s, /not a filter/);
});

test('a push-down that changed NOTHING says so — the outcome nobody would spot', () => {
  const s = pushDownNote({
    push_down: 'a hat', push_down_moved: 0,
    push_down_median: { pool: 0.21, results: 0.21 },
  });
  assert.match(s, /changed nothing/);
  assert.match(s, /stronger push/);
});

test('two tangled phrases are named as tangled, not as a failure to try harder', () => {
  // Measured case: excluding "a bikini" from "a woman at the beach" still
  // returned 66.7% bikinis at every usable weight — the trait IS most of what
  // the query means, and no strength setting fixes that.
  const s = pushDownNote({
    push_down: 'a bikini', push_down_moved: 3,
    push_down_median: { pool: 0.19, results: 0.19 },
  });
  assert.match(s, /too tangled/);
});

test('no exclusion, no note', () => {
  assert.equal(pushDownNote({ query: 'a woman' }), '');
  assert.equal(pushDownNote(null), '');
});

test('one place is singular', () => {
  assert.match(pushDownNote({ push_down: 'a hat', push_down_moved: 1 }),
    /changed 1 place\b/);
});

// --- the strengths are the measured ones -------------------------------------
test('three strengths, ordered, with the calibrated default in the middle', () => {
  assert.equal(PUSH_DOWN_STRENGTHS.length, 3);
  const values = PUSH_DOWN_STRENGTHS.map((s) => s.value);
  assert.deepEqual(values, [...values].sort((a, b) => a - b));
  assert.ok(values.includes(PUSH_DOWN_DEFAULT_STRENGTH));
  assert.equal(PUSH_DOWN_DEFAULT_STRENGTH, 0.6);
  // Strong is honest about what it costs — that trade is the whole reason the
  // default is not simply the strongest setting available.
  const strong = PUSH_DOWN_STRENGTHS[2];
  assert.match(strong.hint, /loosens/);
});

// --- wiring contract on the JSX ----------------------------------------------
const ws = bankTreeSource();

test('the push-down field is labelled and sent to the backend', () => {
  assert.match(ws, /htmlFor="bank-text-exclude"/);
  assert.match(ws, /id="bank-text-exclude"/);
  assert.match(ws, /push_down: textExclude\.trim\(\) \|\| null/);
  assert.match(ws, /push_down_weight: textExcludeW/);
});

test('the panel calls it "Push down", never "Exclude"', () => {
  // The tag filter owns the word "Exclude" and it means a guaranteed absence.
  // Two different promises must not wear the same label in one workspace.
  assert.match(ws, /Push down \(optional\)/);
  const panel = ws.slice(ws.indexOf('bank-text-search'), ws.indexOf('How many'));
  assert.doesNotMatch(panel, />\s*Exclude/);
});

test('the suggestion is offered, never applied on its own', () => {
  assert.match(ws, /suggestPushDown\(textQuery\)/);
  // It lives behind a button the user has to press.
  assert.match(ws, /Push “\{suggestPushDown\(textQuery\)\}” down instead\?/);
});

test('the wording comes from the tested helpers, not re-typed in JSX', () => {
  assert.match(ws, /pushDownCaveat\(\)/);
  assert.match(ws, /pushDownNote\(textResult\)/);
  assert.match(ws, /PUSH_DOWN_STRENGTHS\.map/);
});
