import test from 'node:test';
import assert from 'node:assert/strict';
import {
  COMPARE_HINT, COMPARE_SHORTCUTS, appendGroups, bestReasonText, bestWins,
  compareFacts, compareProgress, createCompare, currentGroup, currentMember,
  dupCompareKeyAction, isExhausted, liveCount, memberKeyIndex, moveMember,
  nextGroup, pickBest, pickMember, prevGroup, rejectMember, resolveGroup,
  skipGroup, startingLayout, twinPositions, vetoGroup,
} from './dupCompare.js';

const img = (id, extra = {}) => ({
  id, name: `img${id}.jpg`, width: 1000, height: 1000, status: 'pending', ...extra,
});
const group = (gid, ids, bestId = null, extra = {}) => ({
  group: gid,
  best_id: bestId == null ? ids[0] : bestId,
  images: ids.map((i) => (typeof i === 'object' ? i : img(i))),
  ...extra,
});

const key = (k, mods = {}) => ({ key: k, target: { tagName: 'DIV' }, ...mods });

/* ── keys ─────────────────────────────────────────────────────────────────── */

test('K, R and S keep the meaning they have in every other review surface', () => {
  assert.equal(dupCompareKeyAction(key('k')), 'keep');
  assert.equal(dupCompareKeyAction(key('R')), 'reject');
  assert.equal(dupCompareKeyAction(key('s')), 'skip');
  assert.equal(dupCompareKeyAction(key('Escape')), 'close');
});

test('the bare arrows walk the copies, shifted arrows walk the groups', () => {
  assert.equal(dupCompareKeyAction(key('ArrowLeft')), 'prev-member');
  assert.equal(dupCompareKeyAction(key('ArrowRight')), 'next-member');
  assert.equal(dupCompareKeyAction(key('ArrowLeft', { shiftKey: true })), 'prev-group');
  assert.equal(dupCompareKeyAction(key('ArrowRight', { shiftKey: true })), 'next-group');
});

test('B accepts the app’s pick, N vetoes the group, F switches the layout', () => {
  assert.equal(dupCompareKeyAction(key('b')), 'best');
  assert.equal(dupCompareKeyAction(key('n')), 'distinct');
  assert.equal(dupCompareKeyAction(key('F')), 'layout');
});

test('a modified keystroke and a typed field are never ours', () => {
  assert.equal(dupCompareKeyAction(key('k', { ctrlKey: true })), null);
  assert.equal(dupCompareKeyAction(key('r', { metaKey: true })), null);
  assert.equal(dupCompareKeyAction({ key: 'k', target: { tagName: 'INPUT', type: 'text' } }), null);
  // …but Escape gets you out of that field's overlay all the same.
  assert.equal(dupCompareKeyAction({ key: 'Escape', target: { tagName: 'TEXTAREA' } }), 'close');
});

test('1-9 answer as copy indices, and nothing else does', () => {
  assert.equal(memberKeyIndex(key('1')), 0);
  assert.equal(memberKeyIndex(key('9')), 8);
  assert.equal(memberKeyIndex(key('0')), null);
  assert.equal(memberKeyIndex(key('k')), null);
  assert.equal(memberKeyIndex(key('2', { ctrlKey: true })), null);
  // A digit is not claimed by the action reader — the two questions never
  // answer the same keystroke.
  assert.equal(dupCompareKeyAction(key('3')), null);
});

test('the printed hint names every shortcut the handler answers', () => {
  // The hint is prose, the list is the contract: each entry's first key must
  // appear in the line the user reads, or the UI promises something else than
  // it does.
  for (const { keys } of COMPARE_SHORTCUTS) {
    const first = keys.split(/\s+/)[0];
    assert.ok(COMPARE_HINT.includes(first), `hint is missing ${first}`);
  }
});

test('a narrow screen opens on the full frame, a desktop side by side', () => {
  assert.equal(startingLayout(1280), 'side');
  assert.equal(startingLayout(1024), 'side');
  assert.equal(startingLayout(412), 'single');
  assert.equal(startingLayout(844), 'single');
  // No width to go on: the desktop layout is the one that degrades gracefully,
  // same call the rail makes.
  assert.equal(startingLayout(undefined), 'side');
});

/* ── the session ──────────────────────────────────────────────────────────── */

test('a run opens on the app’s pick, not on the first copy', () => {
  const s = createCompare([group(7, [10, 11, 12], 12)]);
  assert.equal(currentMember(s).id, 12);
});

test('the clicked group is where the run starts, and it walks forward', () => {
  const s = createCompare([group(1, [1, 2]), group(2, [3, 4]), group(3, [5, 6])],
    { startGroup: 2 });
  assert.equal(currentGroup(s).group, 2);
  assert.equal(currentGroup(nextGroup(s)).group, 3);
  assert.equal(currentGroup(prevGroup(s)).group, 1);
});

test('an unknown startGroup starts at the top rather than emptying the run', () => {
  const s = createCompare([group(1, [1, 2])], { startGroup: 99 });
  assert.equal(currentGroup(s).group, 1);
});

test('the cursor wraps inside the group and never lands on a rejected copy', () => {
  const s = createCompare([group(1, [img(1), img(2, { status: 'reject' }), img(3)], 1)]);
  assert.equal(currentMember(s).id, 1);
  assert.equal(currentMember(moveMember(s, 1)).id, 3, 'skips the rejected copy');
  assert.equal(currentMember(moveMember(s, -1)).id, 3, 'wraps backwards past it too');
});

test('a group whose pick is already rejected opens on a copy still standing', () => {
  const s = createCompare([group(1, [img(1, { status: 'reject' }), img(2)], 1)]);
  assert.equal(currentMember(s).id, 2);
});

test('pickMember ignores what it cannot honour', () => {
  const s = createCompare([group(1, [img(1), img(2, { status: 'reject' }), img(3)], 1)]);
  assert.equal(currentMember(pickMember(s, 2)).id, 3);
  assert.equal(currentMember(pickMember(s, 1)).id, 1, 'a rejected copy is not selectable');
  assert.equal(currentMember(pickMember(s, 9)).id, 1, 'out of range moves nothing');
});

test('B puts the cursor back on the app’s pick', () => {
  const s = createCompare([group(1, [10, 11, 12], 12)]);
  const moved = moveMember(s, 1);
  assert.notEqual(currentMember(moved).id, 12);
  assert.equal(currentMember(pickBest(moved)).id, 12);
});

test('resolving a group advances and records it', () => {
  const s = createCompare([group(1, [1, 2]), group(2, [3, 4])]);
  const after = resolveGroup(s);
  assert.equal(currentGroup(after).group, 2);
  assert.deepEqual(after.resolved, [1]);
});

test('rejecting one copy keeps the group open while two are still standing', () => {
  const s = createCompare([group(1, [1, 2, 3], 1), group(2, [4, 5])]);
  const after = rejectMember(s, 1);
  assert.equal(currentGroup(after).group, 1, 'still the same group');
  assert.equal(liveCount(after), 2);
  assert.ok(after.rejected.includes(1));
  assert.notEqual(currentMember(after).id, 1, 'the cursor left the copy just rejected');
});

test('rejecting a copy the cursor is NOT on leaves the cursor alone', () => {
  // Side by side, the ✕ under a tile rejects that tile. Moving the cursor off a
  // picture the user never touched is how the next keystroke lands wrong.
  const s = createCompare([group(1, [1, 2, 3], 1)]);
  assert.equal(currentMember(s).id, 1);
  const after = rejectMember(s, 3);
  assert.equal(currentMember(after).id, 1);
  assert.ok(after.rejected.includes(3));
});

test('rejecting down to one copy settles the group and moves on', () => {
  const s = createCompare([group(1, [1, 2], 1), group(2, [3, 4])]);
  const after = rejectMember(s, 1);
  assert.equal(currentGroup(after).group, 2, 'nothing left to choose between');
  assert.deepEqual(after.resolved, [1]);
});

test('a Keep records the losers, so walking back cannot empty the group', () => {
  /* The server rejects every other copy in the same call. Without recording it,
     ⇧← into a settled group showed live-looking tiles, and a second K there sent
     keep_ids for a copy whose rivals were already rejected — resolve_dups then
     rejected the lone survivor too and the whole group was gone. */
  const s = createCompare([group(1, [1, 2, 3], 1), group(2, [4, 5])]);
  const after = resolveGroup(s, 1, 2);          // the server elected copy 2
  assert.deepEqual(after.resolved, [1]);
  assert.deepEqual(after.rejected, [1, 3], 'the losers are known to be gone');
  const back = prevGroup(after);
  assert.equal(liveCount(back), 1, 'only the elected copy still stands');
  assert.equal(currentMember(back).id, 2, 'and the cursor sits on it, not on a corpse');
});

test('a resolve BY ELIMINATION leaves the survivor standing', () => {
  // rejectMember resolves a group by knocking copies out; the last one must not
  // be swept up with them, which is why keptId is optional.
  const s = createCompare([group(1, [1, 2], 1), group(2, [3, 4])]);
  const after = rejectMember(s, 1);
  assert.deepEqual(after.resolved, [1]);
  assert.deepEqual(after.rejected, [1], 'only the copy actually rejected');
});

test('≠ settles a group without rejecting anything, and counts apart', () => {
  const s = createCompare([group(1, [1, 2]), group(2, [3, 4])]);
  const after = vetoGroup(s);
  assert.equal(currentGroup(after).group, 2, 'the run moves on');
  assert.deepEqual(after.vetoed, [1]);
  assert.deepEqual(after.resolved, [], 'a veto is not a resolve');
  assert.deepEqual(after.rejected, [], 'and it rejects nothing');
  const p = compareProgress(after);
  assert.equal(p.vetoed, 1);
  assert.equal(p.resolved, 0);
});

test('a vetoed group never comes back in a refill', () => {
  // The server stops listing it, but a refill that arrived in flight must not
  // queue it either — same promise as a skip, for a stronger reason: the user
  // did not say "not now", they said "never".
  const s = vetoGroup(createCompare([group(1, [1, 2]), group(2, [3, 4])]));
  const refilled = appendGroups(s, [group(1, [1, 2]), group(9, [7, 8])]);
  assert.deepEqual(refilled.groups.map((g) => g.group), [1, 2, 9]);
});

test('a skipped group is remembered so a refill cannot hand it back', () => {
  const s = skipGroup(createCompare([group(1, [1, 2]), group(2, [3, 4])]));
  assert.deepEqual(s.skipped, [1]);
  const refilled = appendGroups(s, [group(1, [1, 2]), group(5, [9, 10])]);
  assert.deepEqual(refilled.groups.map((g) => g.group), [1, 2, 5],
    'group 1 is not queued a second time');
});

test('a refill drops what the run already walked or settled', () => {
  const s = resolveGroup(createCompare([group(1, [1, 2]), group(2, [3, 4])]));
  const refilled = appendGroups(s, [group(1, [1, 2]), group(2, [3, 4])]);
  assert.equal(refilled, s, 'nothing new — the same object, so the shell knows it is done');
});

test('a refill that lands past the end seats the cursor on the first new group', () => {
  let s = createCompare([group(1, [1, 2])]);
  s = resolveGroup(s);
  assert.ok(isExhausted(s));
  s = appendGroups(s, [group(4, [7, 8], 8)]);
  assert.equal(isExhausted(s), false);
  assert.equal(currentGroup(s).group, 4);
  assert.equal(currentMember(s).id, 8, 'and on that group’s own pick');
});

test('a refill arriving mid-run does not move the cursor', () => {
  const s = createCompare([group(1, [1, 2]), group(2, [3, 4])]);
  const refilled = appendGroups(s, [group(3, [5, 6])]);
  assert.equal(currentGroup(refilled).group, 1);
  assert.equal(refilled.groups.length, 3);
});

test('the progress readout counts what the run walked, not the bank', () => {
  let s = createCompare([group(1, [1, 2]), group(2, [3, 4]), group(3, [5, 6])]);
  s = resolveGroup(s);
  s = skipGroup(s);
  const p = compareProgress(s);
  assert.equal(p.position, 3);
  assert.equal(p.loaded, 3);
  assert.equal(p.resolved, 1);
  assert.equal(p.skipped, 1);
});

/* ── reading the copies against each other ────────────────────────────────── */

test('a fact is lit on whichever copies hold the group’s top value', () => {
  const images = [
    img(1, { width: 2000, height: 1000, blur_score: 100, file_size: 500_000 }),
    img(2, { width: 1000, height: 1000, blur_score: 100, file_size: 900_000 }),
  ];
  const first = compareFacts(images, images[0]);
  const pixels = first.find((f) => f.key === 'pixels');
  const sharp = first.find((f) => f.key === 'sharp');
  const bytes = first.find((f) => f.key === 'bytes');
  assert.equal(pixels.win, true, 'twice the pixels');
  assert.equal(sharp.win, true, 'a tie is still the top — both copies light up');
  assert.equal(bytes.win, false, 'the other copy is the heavier one');
  assert.equal(pixels.text, '2000×1000');
});

test('the copy that loses on everything is the one with nothing lit', () => {
  // The commonest group there is: a file, its exact copy, and a re-compressed
  // third. The two twins must not both go blank — that reads as "nothing was
  // measured" when the truth is "these two are top and the third is worse".
  const images = [
    img(1, { width: 1000, height: 1500, blur_score: 700, file_size: 2_000_000 }),
    img(2, { width: 1000, height: 1500, blur_score: 700, file_size: 2_000_000 }),
    img(3, { width: 500, height: 750, blur_score: 300, file_size: 40_000 }),
  ];
  assert.ok(compareFacts(images, images[0]).every((f) => f.win), 'the twin is top on all of them');
  assert.ok(compareFacts(images, images[2]).every((f) => !f.win), 'the loser lights nothing');
});

test('byte-identical copies are named by position, and only when weighed', () => {
  const images = [
    img(1, { width: 1000, height: 1500, file_size: 2_000_000 }),
    img(2, { width: 1000, height: 1500, file_size: 2_000_000 }),
    img(3, { width: 500, height: 750, file_size: 40_000 }),
  ];
  const twins = twinPositions(images);
  assert.deepEqual(twins[1], [2]);
  assert.deepEqual(twins[2], [1]);
  assert.equal(twins[3], undefined, 'the odd one out has no twin');
  // A never-scanned group is all zeros — "0 = 0" must not declare it identical.
  assert.deepEqual(twinPositions([img(1, { width: 0, height: 0 }), img(2, { width: 0, height: 0 })]), {});
});

test('a metric no copy carries is absent, not printed as a question mark', () => {
  const images = [img(1), img(2)];
  const facts = compareFacts(images, images[0]);
  assert.equal(facts.find((f) => f.key === 'aesthetic'), undefined);
  assert.equal(facts.find((f) => f.key === 'sharp'), undefined);
  assert.ok(facts.find((f) => f.key === 'pixels'), 'the dimensions are always there');
});

test('file size is read in the unit that fits', () => {
  const small = compareFacts([img(1, { file_size: 40_000 })], img(1, { file_size: 40_000 }));
  assert.equal(small.find((f) => f.key === 'bytes').text, '39 kB');
  const big = compareFacts([img(1, { file_size: 3_145_728 })], img(1, { file_size: 3_145_728 }));
  assert.equal(big.find((f) => f.key === 'bytes').text, '3.0 MB');
});

test('the pick is explained by what it visibly wins', () => {
  const images = [
    img(1, { width: 2000, height: 2000, blur_score: 200 }),
    img(2, { width: 1000, height: 1000, blur_score: 100 }),
  ];
  assert.deepEqual(bestWins(images, 1), ['resolution', 'sharpness']);
  assert.match(bestReasonText(images, 1), /resolution and sharpness/);
});

test('when nothing measured separates the copies, the badge says so', () => {
  const images = [img(1), img(2)];
  assert.deepEqual(bestWins(images, 1), []);
  assert.match(bestReasonText(images, 1), /coin toss/);
});
