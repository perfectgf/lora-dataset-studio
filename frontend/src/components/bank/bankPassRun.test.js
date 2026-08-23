// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ENDPOINT_JOB_KIND, JOB_LABELS, bankIsBusy, busyLine, busyRefusal,
  busyRefusalLive, jobLabel,
  jobProgress, passButtonState, passOutcome, passSettled, settledActivity,
  summaryKeyFor,
} from './bankPassRun.js';

/* The exact shape the bank payload embeds while ✨ Score is walking a bank —
   the situation the report came from: an Analyze-all pass was in flight, the
   re-group button still looked clickable, and the click came back with the
   server's own sentence in a red toast. */
const scoring = { kind: 'score', done: 137, total: 412, finished: false, detail: null };

// ── 1. A button that cannot work must not look like one ─────────────────────

test('a re-run button is DISABLED while any pass holds the bank', () => {
  const s = passButtonState({ activity: scoring });
  assert.equal(s.disabled, true);
});

test('the disabled button says WHICH pass is holding the bank, and where it is', () => {
  const { reason } = passButtonState({ activity: scoring });
  assert.match(reason, /✨ Score pass/);   // which
  assert.match(reason, /137 \/ 412/);      // where it is
  assert.match(reason, /Stop/);            // what to do about it
});

test('the button is enabled again once the job finishes', () => {
  const finished = { ...scoring, finished: true };
  assert.equal(passButtonState({ activity: finished }).disabled, false);
  assert.equal(passButtonState({ activity: null }).disabled, false);
});

test('losing contact does NOT re-enable the button — the pass keeps running server-side', () => {
  // PROGRESS_STALE: the snapshot is old but the bank is still occupied, so a
  // click would still 409. Re-enabling on a failed poll would invite exactly
  // the refusal this whole change exists to prevent.
  assert.equal(passButtonState({ activity: scoring, offline: true }).disabled, true);
});

test('offline with no snapshot at all leaves the button usable', () => {
  // We do not KNOW the bank is busy; disabling on ignorance would block a bank
  // that is perfectly free. The network layer reports the failed click.
  assert.equal(passButtonState({ activity: null, offline: true }).disabled, false);
});

test('a click in flight disables the button and says so', () => {
  const s = passButtonState({ activity: null, pending: true });
  assert.equal(s.disabled, true);
  assert.equal(s.pending, true);
  assert.match(s.reason, /Starting/);
});

test('bankIsBusy covers running and stale, not finished or absent', () => {
  assert.equal(bankIsBusy(scoring), true);
  assert.equal(bankIsBusy(scoring, true), true);
  assert.equal(bankIsBusy({ ...scoring, finished: true }), false);
  assert.equal(bankIsBusy(null), false);
});

// ── 2. A 409 never reaches the user as server text ──────────────────────────

const SERVER_TEXT = 'a score job is already running on this bank';

test('the refusal never repeats the raw server sentence', () => {
  const refusal = busyRefusal({ kind: 'score', activity: scoring });
  assert.ok(!refusal.includes(SERVER_TEXT),
    `the refusal must be rewritten, got: ${refusal}`);
  assert.ok(!/\bjob is already running\b/.test(refusal),
    `no server phrasing may survive, got: ${refusal}`);
});

test('the refusal states the blocker, its progress AND the remedy', () => {
  const refusal = busyRefusal({ kind: 'score', activity: scoring });
  assert.match(refusal, /✨ Score pass/);
  assert.match(refusal, /137 \/ 412/);
  assert.match(refusal, /Wait for it to finish, or press Stop/);
});

test('the refusal still works with only the 409 body — no snapshot', () => {
  // The poll may not have landed yet when the click is refused. Naming the pass
  // from `busy_kind` alone is why the route returns it.
  const refusal = busyRefusal({ kind: 'semantic_dedup' });
  assert.match(refusal, /✂ Crops & variants/);
  assert.match(refusal, /Stop/);
  assert.ok(!refusal.includes('undefined'), refusal);
});

/* ── 2b. The refusal reads the LIVE registry, not the 25-second payload ──────
   Measured on a 36 921-image bank: the payload takes ~25 s at rest and longer
   while a pass writes, so at refusal time `payload.activity` is stale or null —
   the toast said "Framing pass is running… press Stop in the progress bar"
   while no bar was on screen, because the bar reads the same stale payload.
   The registry read behind /activity is in-memory and instant; the refusal must
   prefer it, and the caller merges it into the payload so the bar appears. */

test('the refusal quotes the FRESH snapshot when the live read lands', async () => {
  const fresh = { kind: 'framing', done: 1968, total: 34922, finished: false, detail: null };
  const refusal = await busyRefusalLive({
    kind: 'framing',
    fetchActivity: async () => fresh,
    fallback: null,
  });
  assert.match(refusal, /📐 Framing pass/);
  assert.match(refusal, /1968 \/ 34922/);
  assert.match(refusal, /press Stop/);
});

test('a failed live read falls back to the last known snapshot', async () => {
  const refusal = await busyRefusalLive({
    kind: 'score',
    fetchActivity: async () => { throw new Error('offline') },
    fallback: scoring,
  });
  assert.match(refusal, /✨ Score pass/);
  assert.match(refusal, /137 \/ 412/);
});

test('no live read and no fallback still names the pass from the 409 body', async () => {
  const refusal = await busyRefusalLive({ kind: 'watermark' });
  assert.match(refusal, /🚩 Watermark scan/);
  assert.match(refusal, /press Stop/);
});

test('a live read that answers "nothing running" does not erase the 409 kind', async () => {
  // The job can finish between the 409 and the read — the refusal was still
  // true when the click was refused, so it still names the blocker.
  const refusal = await busyRefusalLive({
    kind: 'framing',
    fetchActivity: async () => null,
    fallback: null,
  });
  assert.match(refusal, /📐 Framing pass/);
});

test('an unknown job kind degrades to a neutral phrase, never to an internal id', () => {
  const refusal = busyRefusal({ kind: 'some_future_pass' });
  assert.ok(!refusal.includes('some_future_pass'), refusal);
  assert.match(refusal, /Another pass/);
});

test('busyLine carries the running detail when the pass has no total', () => {
  const line = busyLine({ activity: { kind: 'scan', done: 12, total: 0, detail: 'grouping duplicates' } });
  assert.match(line, /🔎 Quality scan/);
  assert.match(line, /12/);
  assert.match(line, /grouping duplicates/);
});

test('a detail that only repeats the pass name is dropped, not stuttered', () => {
  // Measured on a live bank: start_scan sets detail='quality scan', which gave
  // "🔎 Quality scan is running on this bank — 1996 / 5425 · quality scan".
  const line = busyLine({ activity: { kind: 'scan', done: 1996, total: 5425, detail: 'quality scan' } });
  assert.equal(line, '🔎 Quality scan is running on this bank — 1996 / 5425');
});

test('a finished job contributes no stale detail to the line', () => {
  const line = busyLine({ activity: { kind: 'scan', done: 9, total: 9, finished: true, detail: 'done — 3 duplicate group(s)' } });
  assert.ok(!line.includes('done — 3'), line);
});

test('jobProgress: total when there is one, bare count when there is not', () => {
  assert.equal(jobProgress({ done: 5, total: 20 }), '5 / 20');
  assert.equal(jobProgress({ done: 5, total: 0 }), '5');
  assert.equal(jobProgress(null), '');
  assert.equal(jobProgress({}), '');
});

test('every job kind bank_jobs can report has a human label', () => {
  // Mirrors the `kind` strings passed to bank_jobs.start() in image_bank_service.
  for (const kind of ['scan', 'faces', 'score', 'semantic_index', 'semantic_dedup', 'watermark',
    'framing', 'caption', 'promote', 'bank_promote', 'pipeline']) {
    assert.ok(JOB_LABELS[kind], `no label for job kind ${kind}`);
    assert.ok(!jobLabel(kind).includes('_'), `label for ${kind} leaks an internal id`);
  }
});

// ── 3. A successful pass reports its NUMBERS ────────────────────────────────

test('a successful re-group shows the figures it produced', () => {
  const out = passOutcome({
    endpoint: 'scan',
    before: { groups: 9, images: 26 },
    after: { groups: 12, images: 34 },
    activity: { kind: 'scan', finished: true },
  });
  assert.equal(out.tone, 'ok');
  assert.match(out.text, /12 duplicate groups/);
  assert.match(out.text, /34 images/);
  assert.match(out.text, /was 9 · 26/);
});

test('a pass that changed nothing says so instead of looking like it never ran', () => {
  const same = { groups: 9, images: 26 };
  const out = passOutcome({ endpoint: 'scan', before: same, after: { ...same } });
  assert.match(out.text, /9 duplicate groups/);
  assert.match(out.text, /Unchanged/i);
});

test('singulars are singular', () => {
  const out = passOutcome({ endpoint: 'scan', after: { groups: 1, images: 1 } });
  assert.match(out.text, /1 duplicate group · 1 image\./);
});

test('the semantic pass counts its own groups, not the exact-duplicate ones', () => {
  assert.equal(summaryKeyFor('scan'), 'dup');
  assert.equal(summaryKeyFor('semantic-dedup'), 'semantic_dup');
  assert.equal(summaryKeyFor('faces'), null);
  const out = passOutcome({ endpoint: 'semantic-dedup', after: { groups: 4, images: 11 } });
  assert.match(out.text, /4 groups of the same shot · 11 images/);
});

test('a pass with no summary of its own quotes what the job itself reported', () => {
  const out = passOutcome({
    endpoint: 'faces',
    activity: { kind: 'faces', finished: true, detail: 'done — 7 person cluster(s)' },
  });
  assert.match(out.text, /7 person cluster/);
});

test('a crashed pass reports the failure, not a success line', () => {
  const out = passOutcome({
    endpoint: 'scan',
    after: { groups: 0, images: 0 },
    activity: { kind: 'scan', finished: true, error: 'source folder is gone' },
  });
  assert.equal(out.tone, 'error');
  assert.match(out.text, /source folder is gone/);
});

test('a stopped pass is not reported as a result', () => {
  const out = passOutcome({
    endpoint: 'scan',
    after: { groups: 3, images: 8 },
    activity: { kind: 'scan', finished: true, cancelled: true },
  });
  assert.equal(out.tone, 'warn');
  assert.match(out.text, /Stopped/);
});

// ── 4. Knowing when the pass we launched has come back ──────────────────────

test('the endpoint→kind table matches the routes the buttons POST to', () => {
  assert.equal(ENDPOINT_JOB_KIND['semantic-dedup'], 'semantic_dedup');
  assert.equal(ENDPOINT_JOB_KIND['semantic-index'], 'semantic_index');
  assert.equal(ENDPOINT_JOB_KIND.scan, 'scan');
});

test('our pass is unsettled only while ITS kind is live', () => {
  assert.equal(passSettled({ kind: 'scan', finished: false }, 'scan'), false);
  assert.equal(passSettled({ kind: 'scan', finished: true }, 'scan'), true);
  // Someone else's job is holding the bank — ours never started.
  assert.equal(passSettled({ kind: 'score', finished: false }, 'scan'), true);
  // A pass fast enough to have been purged before the first poll: settled.
  assert.equal(passSettled(null, 'scan'), true);
});

test('only OUR finished snapshot is quoted in the outcome', () => {
  const mine = { kind: 'semantic_dedup', finished: true, detail: 'done — 4 group(s)' };
  assert.equal(settledActivity(mine, 'semantic-dedup'), mine);
  assert.equal(settledActivity(mine, 'scan'), null);
  assert.equal(settledActivity({ kind: 'scan', finished: false }, 'scan'), null);
  assert.equal(settledActivity(null, 'scan'), null);
});

test('the progress bar names the destructive pass instead of "Job running"', () => {
  // The bar keeps its OWN kind->label map (it renders shorter, un-emoji'd
  // wording); a kind missing from it falls through to the anonymous fallback,
  // which is the last thing a delete should do.
  const src = bankTreeSource();
  assert.match(src, /delete_rejected: 'Deleting rejected files'/);
  assert.equal(jobLabel('delete_rejected'), '\u{1F5D1} Delete rejected');
});
