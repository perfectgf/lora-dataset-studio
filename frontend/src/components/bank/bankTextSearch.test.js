// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CLIP_LIMITS, clampN, limitsSentence, pendingLabel, readinessHint,
  spreadLabel, summarize, unsearchableNote,
} from './bankTextSearch.js';

// --- the ranking must never read as a filter ---------------------------------
test('the summary says "closest", gives the range, and refuses to claim a match', () => {
  const s = summarize({
    query: 'brunette outdoors, wide shot', pool: 120, filtered: 120, unscored: 0,
    image_ids: [1, 2, 3], score_range: { top: 0.22, bottom: 0.19 },
  });
  assert.match(s, /3 closest of 120/);
  assert.match(s, /0\.22/);
  assert.match(s, /0\.19/);
  // The framing that stops a ranking reading as a selection.
  assert.match(s, /does not select them/);
  assert.match(s, /Every image scores something/);
  // It must never assert that the results "match" the query.
  assert.doesNotMatch(s, /\bmatch(es|ing)? the\b/);
});

// --- MEASURED: the usable cosine band is ≈0.09–0.23 on this model -------------
// A correct top-1 match peaked at 0.20–0.23; unrelated pairs sat at 0.09–0.16.
// Any absolute quality scale calibrated above that band silently declares every
// perfect result a failure. These two tests exist to stop one being reintroduced.
test('a MEASURED-perfect match (0.22) is never described as weak or poor', () => {
  const s = summarize({
    query: 'a cat', pool: 50, filtered: 50, unscored: 0, image_ids: [1, 2],
    score_range: { top: 0.22, bottom: 0.21 },
  });
  assert.doesNotMatch(s, /weak|poor|bad|low/i);
  // 0.22 vs 0.21 is a tight set: the tail is as good as the head.
  assert.match(s, /all about equally close/);
});

test('similarity is never rendered as a percentage', () => {
  const s = summarize({
    query: 'a cat', pool: 50, filtered: 50, unscored: 0, image_ids: [1],
    score_range: { top: 0.22, bottom: 0.22 },
  });
  // "22% match" on the best achievable result reads as a failure.
  assert.doesNotMatch(s, /%/);
});

test('strength is measured against the bank\'s OWN median, not an absolute band', () => {
  // Measured baseline: median 0.112, correct hits 0.177-0.233. lift = 0.10.
  // A tight set (spread 0.02) is a fifth of the lift -> the tail is as good.
  assert.equal(spreadLabel({ top: 0.22, bottom: 0.20 }, 0.12),
    'all about equally close');
  // Same numbers, different bank: a corpus whose cosines live high is described
  // identically, because only the ratio is used.
  assert.equal(spreadLabel({ top: 0.90, bottom: 0.88 }, 0.80),
    'all about equally close');
  assert.equal(spreadLabel({ top: 0.22, bottom: 0.15 }, 0.12),
    'the last ones are noticeably looser');
  assert.equal(spreadLabel({ top: 0.22, bottom: 0.10 }, 0.12),
    'the tail is much weaker than the top');
  assert.equal(spreadLabel({ top: undefined, bottom: 1 }, 0.1), '');
});

test('a single-subject bank — the MAIN use case — trips the "no separation" warning', () => {
  // Measured: on a bank dominated by one subject the image-to-image cosine is
  // 0.60-0.89 and a query's discriminating gap compresses by 30-70%. The best
  // result then sits barely above what every image scores, and the ranking stops
  // meaning anything. That must be SAID, and an absolute band would never say it.
  const flat = spreadLabel({ top: 0.204, bottom: 0.201 }, 0.198);
  assert.match(flat, /barely above what any image here scores/);
  // The same absolute numbers on a discriminating bank say something different —
  // proof that the verdict comes from the measured baseline, not the values.
  assert.doesNotMatch(spreadLabel({ top: 0.204, bottom: 0.201 }, 0.09),
    /barely above/);
});

test('the summary carries the "brings to the front, does not select" framing', () => {
  const s = summarize({
    query: 'a cat', pool: 50, filtered: 50, unscored: 0, image_ids: [1],
    score_range: { top: 0.22, bottom: 0.21 }, pool_median: 0.12,
  });
  assert.match(s, /does not select/);
});

// --- images that CANNOT be found must be named -------------------------------
test('unscored images are reported, never silently dropped', () => {
  const note = unsearchableNote({ unscored: 3, filtered: 27 });
  assert.match(note, /3 of 27/);
  assert.match(note, /no CLIP semantic embedding/);
  assert.match(note, /run ✨ Score/i);
  // …and it rides along in the announced summary.
  assert.match(summarize({
    query: 'x', pool: 24, filtered: 27, unscored: 3, image_ids: [1],
    score_range: { top: 0.3, bottom: 0.2 },
  }), /3 of 27/);
});

test('SigLIP2 search accepts the new unindexed field and names its own fix', () => {
  const note = unsearchableNote({
    engine: 'siglip2', unindexed: 4, unscored: 0, filtered: 30,
  });
  assert.match(note, /4 of 30/);
  assert.match(note, /no SigLIP 2 semantic embedding/);
  assert.match(note, /build or complete the SigLIP 2 semantic index/i);
  assert.doesNotMatch(note, /run ✨ Score/i);
});

test('a fully scored bank shows no scary note', () => {
  assert.equal(unsearchableNote({ unscored: 0, filtered: 27 }), '');
});

test('an empty result still explains itself instead of showing nothing', () => {
  const s = summarize({ query: 'x', pool: 0, filtered: 5, unscored: 5, image_ids: [] });
  assert.match(s, /No CLIP-indexed image/);
  assert.match(s, /5 of 5/);
});

// --- the latency has to be announced BEFORE the click ------------------------
test('a cold first search is announced, a warm one is called instant', () => {
  assert.match(readinessHint({ available: true, warm: false }), /First search loads/);
  assert.match(readinessHint({ available: true, warm: false }), /10 seconds/);
  assert.match(readinessHint({ available: true, warm: true }), /instant/);
  assert.equal(pendingLabel({ warm: false }), 'Loading the search model…');
  assert.equal(pendingLabel({ warm: true }), 'Searching…');
});

test('an unavailable install is told why, not left with a dead field', () => {
  const hint = readinessHint({ available: false, reason: 'torch is not installed' });
  assert.match(hint, /torch is not installed/);
});

test('a possible 1.6 GB download outranks the normal hint', () => {
  const hint = readinessHint({
    available: true, warm: false,
    weights_warning: 'the first search may download ~1.6 GB',
  });
  assert.match(hint, /1\.6 GB/);
});

// --- honesty about what CLIP cannot do ---------------------------------------
test('the known CLIP failure modes are stated in the UI, not just the docs', () => {
  assert.equal(CLIP_LIMITS.length, 3);
  assert.match(CLIP_LIMITS.join(' '), /Counting/);
  assert.match(CLIP_LIMITS.join(' '), /Negation/);
  assert.match(CLIP_LIMITS.join(' '), /Spatial/);
});

test('the negation warning states the INVERSION, not a vague imprecision', () => {
  // Measured: on a helmeted astronaut, "without a helmet" (0.217) outscored
  // "with a helmet" (0.212). A user typing "without glasses" gets glasses and
  // no signal that anything is wrong — so the copy has to say what happens.
  const negation = CLIP_LIMITS.find((l) => /Negation/.test(l));
  assert.match(negation, /returns glasses|ignored/);
  const s = limitsSentence();
  assert.match(s, /^CLIP is best/);
  assert.match(s, /cannot count/);
  assert.match(s, /without/);
  // …and it gives the usable workaround rather than only a disclaimer.
  assert.match(s, /describe what IS in the shot/);
  assert.match(limitsSentence('siglip2'), /^SigLIP 2 is best/);
});

test('clampN matches the sibling selectors', () => {
  assert.equal(clampN(0), 1);
  assert.equal(clampN(99999), 2000);
  assert.equal(clampN('60'), 60);
  assert.equal(clampN('nonsense'), 1);
});

// --- wiring contract on the JSX ----------------------------------------------
const ws = bankTreeSource();

test('the search field is labelled, announced, and feeds the shared selection view', () => {
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/search-text/);
  // Ranked ids go through the SAME "show what you selected" helper as the other
  // two selectors — no separate screen.
  assert.match(ws, /showCuratedSelection\(d\.image_ids\)/);
  // Accessibility: a real label, and results announced politely.
  assert.match(ws, /htmlFor="bank-text-search"/);
  assert.match(ws, /id="bank-text-search"/);
  assert.match(ws, /aria-live="polite"/);
});

test('the summary and limits are rendered from the tested helpers, not re-typed', () => {
  assert.match(ws, /from '\.\/bankTextSearch\.js'/);
  assert.match(ws, /summarize\(/);
  assert.match(ws, /readinessHint\(/);
  assert.match(ws, /pendingLabel\(/);
  assert.match(ws, /limitsSentence\(semanticState\.engine\)/);
});

test('closing the panel releases the encoder memory', () => {
  assert.match(ws, /\/api\/bank\/text-search\/release/);
});
