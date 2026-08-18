import test from 'node:test';
import assert from 'node:assert/strict';
import { faceAnalysisState, faceAnalysisLabel, autoTriageAvailable,
  autoTriageEmptyReason } from './faceScoringGate.js';

const BLOCKED = 'Face similarity needs a photographic face; it cannot read a drawn one. '
  + 'Set the subject type to Human if this dataset is photographic.';

test('a blocked dataset disables the pass AND carries the reason', () => {
  const s = faceAnalysisState({ blockedReason: BLOCKED, hasRef: true, busy: false });
  assert.equal(s.disabled, true);
  // "Disappears in silence" is the bug, not the fix: the reason IS the title.
  assert.equal(s.title, BLOCKED);
  assert.equal(s.blocked, true);
});

test('the reason wins over "set a reference photo first"', () => {
  // A drawn dataset with no reference must not be sent off to add one: it would
  // not have helped.
  const s = faceAnalysisState({ blockedReason: BLOCKED, hasRef: false, busy: false });
  assert.equal(s.title, BLOCKED);
});

test('a photographic dataset is strictly unchanged', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: true, busy: false });
  assert.equal(s.disabled, false);
  assert.equal(s.blocked, false);
  assert.match(s.title, /facial resemblance/i);

  const noRef = faceAnalysisState({ blockedReason: null, hasRef: false, busy: false });
  assert.equal(noRef.disabled, true);
  assert.equal(noRef.title, 'Set a reference photo first');

  const busy = faceAnalysisState({ blockedReason: null, hasRef: true, busy: true });
  assert.equal(busy.disabled, true);
  assert.match(busy.title, /facial resemblance/i);
});

test('the gate is the SERVER string — the UI never re-derives the rule', () => {
  // Any non-empty reason blocks, whatever it says: when the server grows a second
  // reason, the UI already honours it. Nothing here mentions "anime".
  const s = faceAnalysisState({ blockedReason: 'some future reason', hasRef: true });
  assert.equal(s.disabled, true);
  assert.equal(s.title, 'some future reason');
  assert.doesNotMatch(faceAnalysisState.toString(), /anime/i);
});

// ── Missing install: name the gap, never an inert button ────────────────────

test('without insightface the button says what is missing AND where to get it', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: true, capable: false });
  assert.equal(s.disabled, true);
  assert.equal(s.blocked, true, 'blocked => the reason is rendered in place, not only in a tooltip');
  assert.match(s.reason, /not installed/i);
  assert.match(s.reason, /Setup/i, 'a gap with no way out is just a dead end');
  assert.equal(s.setupRoute, '#/setup?step=quality');
});

test('the server refusal outranks a missing install', () => {
  // Installing InsightFace would not make a drawn face readable: do not send
  // anyone off to install something that would not have helped.
  const s = faceAnalysisState({ blockedReason: BLOCKED, hasRef: true, capable: false });
  assert.equal(s.reason, BLOCKED);
  assert.equal(s.setupRoute, null);
});

test('a missing install outranks "set a reference photo first"', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: false, capable: false });
  assert.match(s.reason, /not installed/i);
});

test('capabilities still loading does not flash "not installed"', () => {
  const s = faceAnalysisState({ blockedReason: null, hasRef: true, capable: false, capsLoading: true });
  assert.equal(s.blocked, false);
  assert.equal(s.reason, null);
});

// ── The button names its own scope ──────────────────────────────────────────

test('the label states how many images the pass will actually score', () => {
  // A mystery pass on an unknown set is what "🎭 Analyze faces" used to be.
  assert.equal(faceAnalysisLabel({ total: 42, unscored: 42 }), '🎭 Analyze faces (42)');
  assert.equal(faceAnalysisLabel({ total: 42, unscored: 7 }), '🎭 Analyze faces (42 · 7 new)');
  assert.equal(faceAnalysisLabel({ total: 42, unscored: 0 }), '🎭 Analyze faces (42)');
  // An older payload has no scope: fall back to the bare label rather than
  // inventing a count.
  assert.equal(faceAnalysisLabel(null), '🎭 Analyze faces');
  assert.equal(faceAnalysisLabel(undefined), '🎭 Analyze faces');
  assert.equal(faceAnalysisLabel({ total: 0, unscored: 0 }), '🎭 Analyze faces');
  assert.equal(faceAnalysisLabel({ total: 'x' }), '🎭 Analyze faces');
});

test('the tooltip says the triage pile is in scope', () => {
  // The scope change is the whole point: generated variations sit at 'pending',
  // and until now the pass never touched them.
  const s = faceAnalysisState({ blockedReason: null, hasRef: true });
  assert.match(s.title, /✓\/✕/, 'the undecided pile must be named, not implied');
});

test('auto-triage is withheld when scoring is blocked', () => {
  // Auto-triage BATCHES keep/reject from face_score. Letting it act on scores from
  // a pass that cannot read this dataset is the one place stale numbers do damage.
  assert.equal(autoTriageAvailable(BLOCKED), false);
  assert.equal(autoTriageAvailable(null), true);
  assert.equal(autoTriageAvailable(undefined), true);
  assert.equal(autoTriageAvailable(''), true);
});

// ── The panel explains its own emptiness ────────────────────────────────────
// Reported by a user: "sometimes auto-triage shows, sometimes it doesn't." It
// was returning null whenever its replay set was empty, so four unrelated
// situations all rendered as the same silent absence.

const img = (over = {}) => ({ id: 1, filename: 'a.png', status: 'pending',
  face_state: 'scorable', face_score: 0.6, ...over });

test('a panel with work to do says nothing (it renders its controls instead)', () => {
  assert.equal(autoTriageEmptyReason([img()]), null);
});

test('a filter hiding the undecided images says so, and counts them', () => {
  // The one case where the user is ONE CLICK from the work rather than a whole
  // pass away — so it has to outrank every other reason.
  const all = [img({ id: 1 }), img({ id: 2 }), img({ id: 3, status: 'keep' })];
  const visible = [all[2]];                       // filter = "Kept"
  const r = autoTriageEmptyReason(visible, all);
  assert.equal(r.kind, 'hidden_by_filter');
  assert.equal(r.count, 2);
  assert.match(r.message, /filter hides them/);
  // Singular is not "1 scored images".
  assert.match(autoTriageEmptyReason([], [img()]).message, /^1 scored image is/);
});

test('a dataset that was never analysed names the pass to run', () => {
  // Same contract as the greyed-out Sort menu on this screen.
  const rows = [img({ face_state: null, face_score: null })];
  const r = autoTriageEmptyReason(rows, rows);
  assert.equal(r.kind, 'never_scored');
  assert.match(r.message, /🎭 Analyze faces/);
});

test('a set the pass could not score at all says THAT, not "run the pass"', () => {
  // The Discord report: wide shots came back entirely unscorable, so the tool
  // that could have triaged them did not just empty — it vanished. Telling this
  // user to "run 🎭 Analyze faces" would send them to redo what they just did.
  const rows = [img({ face_state: 'too_small', face_score: null }),
                img({ id: 2, face_state: 'extreme_pose', face_score: null })];
  const r = autoTriageEmptyReason(rows, rows);
  assert.equal(r.kind, 'none_scorable');
  assert.match(r.message, /judge those by eye/);
});

test('a fully curated dataset says the work is done', () => {
  const rows = [img({ status: 'keep' }), img({ id: 2, status: 'reject' })];
  assert.equal(autoTriageEmptyReason(rows, rows).kind, 'all_decided');
});

test('without the unfiltered list the answer degrades, never lies', () => {
  // `all` is optional (older callers, tests). The reason must then be whatever
  // the visible list can PROVE — never an invented "your filter hides them".
  const rows = [img({ status: 'keep' })];
  assert.equal(autoTriageEmptyReason(rows).kind, 'all_decided');
  assert.equal(autoTriageEmptyReason(rows, null).kind, 'all_decided');
  assert.equal(autoTriageEmptyReason([]).kind, 'never_scored');
});

test('a scorable row with no score is not treated as scored', () => {
  // face_state without face_score is a half-written row; auto-triage compares
  // numbers, so it must not count as something it could have ruled on.
  const rows = [img({ face_score: null })];
  assert.equal(autoTriageEmptyReason(rows, rows).kind, 'none_scorable');
});

test('rows with no file are ignored rather than counted as unscored', () => {
  const rows = [img({ filename: null, face_state: null }), img({ id: 2, status: 'keep' })];
  assert.equal(autoTriageEmptyReason(rows, rows).kind, 'all_decided');
});
