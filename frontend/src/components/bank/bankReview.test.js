import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createSession, currentId, isFinished, progress, advance, back, decide, skip,
  setShuffle, shuffled,
} from './bankReview.js';

// Deterministic PRNG so shuffles are reproducible in the assertions below.
const lcg = (seed = 42) => () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};
const IDS = [1, 2, 3, 4, 5];

// Walk a session to the end, always deciding, and return the ids in the order
// they were proposed.
const walkAll = (s, act = (x) => decide(x, 'keep')) => {
  const seen = [];
  let cur = s;
  while (!isFinished(cur)) { seen.push(currentId(cur)); cur = act(cur); }
  return { seen, session: cur };
};

test('sequential session walks the snapshot in order, once each', () => {
  const { seen, session } = walkAll(createSession(IDS));
  assert.deepEqual(seen, IDS);
  assert.ok(isFinished(session));
  assert.equal(currentId(session), null);
});

test('shuffled session is a permutation — every id exactly once, no repeat', () => {
  const { seen } = walkAll(createSession(IDS, { shuffle: true, rand: lcg(7) }));
  assert.equal(seen.length, IDS.length);
  assert.deepEqual([...seen].sort((a, b) => a - b), IDS);
});

test('shuffled order actually differs from sequential (with this seed)', () => {
  const s = createSession([1, 2, 3, 4, 5, 6, 7, 8], { shuffle: true, rand: lcg(3) });
  assert.notDeepEqual(s.order, [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('duplicate and null ids are dropped from the snapshot', () => {
  const s = createSession([1, 1, 2, null, 3, 2]);
  assert.deepEqual(s.pool, [1, 2, 3]);
  assert.equal(progress(s).total, 3);
});

test('startId RESUMES from that image instead of restarting the pool', () => {
  // Reversed on a user report (nofaceman, Discord): this used to pull the id to
  // the front, so the next → after the clicked shot went back to image #1 and a
  // triage could not be resumed at all. "Start here" now means "continue here".
  const s = createSession(IDS, { startId: 4 });
  assert.equal(currentId(s), 4);
  assert.equal(progress(s).position, 4, 'the readout must say where we are');
  const { seen } = walkAll(s);
  assert.deepEqual(seen, [4, 5]);
});

test('a resumed session can still step BACK into what it skipped over', () => {
  // Nothing is lost by resuming: ← walks back down the untouched order.
  const s = back(back(createSession(IDS, { startId: 4 })));
  assert.equal(currentId(s), 2);
});

test('shuffle keeps pulling startId to the front (a position means nothing there)', () => {
  const s = createSession(IDS, { shuffle: true, startId: 4, rand: lcg(7) });
  assert.equal(currentId(s), 4);
  const { seen } = walkAll(s);
  assert.equal(seen.length, IDS.length, 'the whole pool is still covered');
  assert.equal(seen[0], 4);
});

test('startId not in the pool is ignored (no crash, normal start)', () => {
  const s = createSession(IDS, { startId: 999 });
  assert.equal(currentId(s), 1);
  assert.equal(progress(s).total, 5);
});

test('empty pool is finished immediately', () => {
  const s = createSession([]);
  assert.ok(isFinished(s));
  assert.equal(currentId(s), null);
  assert.deepEqual(progress(s), { position: 0, total: 0, kept: 0, rejected: 0, skipped: 0, remaining: 0 });
});

test('progress counts position honestly and tallies the decisions', () => {
  let s = createSession(IDS);
  assert.deepEqual(progress(s).position, 1);
  assert.equal(progress(s).total, 5);
  s = decide(s, 'keep');
  s = decide(s, 'reject');
  s = skip(s);
  const p = progress(s);
  assert.equal(p.position, 4);
  assert.equal(p.kept, 1);
  assert.equal(p.rejected, 1);
  assert.equal(p.skipped, 1);
  assert.equal(p.remaining, 2);
});

test('skip records the id and never re-proposes it', () => {
  const { seen, session } = walkAll(createSession(IDS), skip);
  assert.deepEqual(seen, IDS);
  assert.deepEqual(session.skipped, IDS);
  assert.deepEqual(session.decisions, {});   // skip decides nothing
});

test('advance/decide/skip stop at the end instead of running past it', () => {
  let s = createSession([1]);
  s = decide(s, 'keep');
  assert.ok(isFinished(s));
  assert.equal(advance(s).pos, s.pos);
  assert.equal(skip(s), s);
  assert.equal(decide(s, 'reject'), s);
});

test('back steps to the previous image and stops at the first', () => {
  let s = createSession(IDS);
  s = decide(s, 'keep');
  s = decide(s, 'keep');
  assert.equal(currentId(s), 3);
  s = back(s);
  assert.equal(currentId(s), 2);
  s = back(back(back(s)));
  assert.equal(currentId(s), 1);
  assert.equal(s.pos, 0);
});

test('back from the end-of-pool screen returns to the last image', () => {
  let s = createSession([1, 2]);
  s = decide(decide(s, 'keep'), 'reject');
  assert.ok(isFinished(s));
  s = back(s);
  assert.equal(currentId(s), 2);
});

test('re-deciding an image after back overwrites, never double-counts', () => {
  let s = createSession([1, 2]);
  s = decide(s, 'keep');
  s = back(s);
  s = decide(s, 'reject');
  const p = progress(s);
  assert.equal(p.kept, 0);
  assert.equal(p.rejected, 1);
});

test('turning shuffle ON mid-session only re-orders what is still unseen', () => {
  let s = createSession([1, 2, 3, 4, 5, 6, 7, 8]);
  s = decide(s, 'keep');           // 1 judged, cursor on 2
  s = setShuffle(s, true, lcg(11));
  assert.deepEqual(s.order.slice(0, 2), [1, 2], 'seen + current stay put');
  const tail = s.order.slice(2);
  assert.deepEqual([...tail].sort((a, b) => a - b), [3, 4, 5, 6, 7, 8]);
  const { seen } = walkAll(s);
  assert.deepEqual([...seen].sort((a, b) => a - b), [2, 3, 4, 5, 6, 7, 8]);
});

test('turning shuffle OFF mid-session restores the snapshot order for the rest', () => {
  let s = createSession([1, 2, 3, 4, 5, 6], { shuffle: true, rand: lcg(5) });
  const first = currentId(s);
  s = decide(s, 'keep');
  const second = currentId(s);
  s = setShuffle(s, false);
  assert.deepEqual(s.order.slice(0, 2), [first, second]);
  const rest = s.order.slice(2);
  assert.deepEqual(rest, [1, 2, 3, 4, 5, 6].filter((id) => id !== first && id !== second));
});

test('toggling shuffle repeatedly never re-shows a judged or skipped image', () => {
  let s = createSession([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const rand = lcg(99);
  const seen = [];
  let flip = true;
  while (!isFinished(s)) {
    seen.push(currentId(s));
    s = flip ? decide(s, 'keep') : skip(s);
    s = setShuffle(s, !s.shuffle, rand);
    flip = !flip;
  }
  assert.equal(seen.length, 10);
  assert.equal(new Set(seen).size, 10, 'no id proposed twice across toggles');
});

test('setShuffle to the current mode is a no-op (same object)', () => {
  const s = createSession(IDS);
  assert.equal(setShuffle(s, false), s);
});

test('setShuffle after the end keeps the whole order frozen', () => {
  let s = createSession([1, 2]);
  s = decide(decide(s, 'keep'), 'keep');
  const order = s.order;
  s = setShuffle(s, true, lcg(1));
  assert.deepEqual(s.order, order);
});

test('shuffled() leaves the input array untouched', () => {
  const src = [1, 2, 3, 4, 5];
  const out = shuffled(src, lcg(2));
  assert.deepEqual(src, [1, 2, 3, 4, 5]);
  assert.deepEqual([...out].sort((a, b) => a - b), src);
});
