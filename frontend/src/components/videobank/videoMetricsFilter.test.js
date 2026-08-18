import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FLAG_LABELS, cutSummary, draftThresholds, editThreshold, filterByFlag,
  flagChips, flagCounts, payloadFromDraft, thresholdFields, audioState,
  audioNote, audioSummary,
} from './videoMetricsFilter.js';

const CLIPS = [
  { id: 1, flags: ['still'], metrics: { motion_mean: 0.0001 } },
  { id: 2, flags: [], metrics: { motion_mean: 0.004 } },
  { id: 3, flags: ['black', 'still'], metrics: { motion_mean: 0.0002 } },
  { id: 4, flags: [], metrics: null },          // not measured yet
];

// --- counting ----------------------------------------------------------------

test('each flag counts the clips that carry it', () => {
  const counts = flagCounts(CLIPS);
  assert.equal(counts.still, 2);
  assert.equal(counts.black, 1);
});

test('unmeasured clips are counted as such, never as clean', () => {
  // "No flags because nothing was measured" and "no flags because the clip is
  // fine" are different facts; showing them as one would make a half-scanned
  // bank look healthy.
  assert.equal(flagCounts(CLIPS).unmeasured, 1);
});

test('a clip with two flags is one clip in the total, not two', () => {
  assert.equal(flagCounts(CLIPS).flagged, 2);
});

// --- filtering ---------------------------------------------------------------

test('filtering by a flag keeps exactly the carriers', () => {
  assert.deepEqual(filterByFlag(CLIPS, 'still').map(c => c.id), [1, 3]);
});

test('the unmeasured pseudo-flag selects clips with no metrics', () => {
  assert.deepEqual(filterByFlag(CLIPS, 'unmeasured').map(c => c.id), [4]);
});

test('no filter returns everything', () => {
  assert.equal(filterByFlag(CLIPS, null).length, 4);
});

// --- the threshold panel -----------------------------------------------------

test('every threshold field is described for the panel, exactly once', () => {
  // The panel renders from this table; a cut the backend supports but the table
  // omits would be configurable only by editing config.json, invisibly. The
  // list itself is no longer hard-coded HERE — it is checked against the
  // backend's own THRESHOLD_KEYS in tests/video-thresholds-contract.test.mjs,
  // because a copy of the list in the test is what let the two drift in the
  // first place while every test stayed green.
  const keys = thresholdFields().map(f => f.key);
  assert.equal(new Set(keys).size, keys.length, 'a cut is listed twice');
  assert.ok(keys.length >= 5);
});

test('each field says which flag it feeds and which way the cut points', () => {
  for (const f of thresholdFields()) {
    assert.ok(FLAG_LABELS[f.flag], `${f.key} names an unknown flag`);
    assert.ok(['below', 'above'].includes(f.direction));
  }
});

// --- the dry-run sentence ----------------------------------------------------

test('the dry-run summary is a sentence with real numbers', () => {
  const text = cutSummary({ still: 31, black: 4, total_flagged: 33 }, 470);
  assert.match(text, /33/);
  assert.match(text, /470/);
  assert.match(text, /still: 31/);
});

test('an empty dry run says nothing would be removed', () => {
  assert.match(cutSummary({ total_flagged: 0 }, 470), /nothing|no clips/i);
});

test('a dry run that would flag most of the bank warns instead of celebrating', () => {
  // The Hugging Face failure mode: a mis-set threshold kept 47 of 1493 and
  // nobody noticed until after. Above half the bank, the sentence changes tone.
  const text = cutSummary({ still: 400, total_flagged: 400 }, 470);
  assert.match(text, /⚠|most of/i);
});

// --- the panel's draft state -------------------------------------------------
// The panel edits a DRAFT and the dry run previews the draft; nothing reaches
// config until Apply. Editing live thresholds would re-flag the grid on every
// keystroke — including through the states the user is merely passing through.

test('a draft starts from the saved cuts and tracks edits', () => {
  const d = draftThresholds({ motion_floor: 0.001, luma_floor: null });
  assert.equal(d.motion_floor, 0.001);
  const edited = editThreshold(d, 'luma_floor', '0.05');
  assert.equal(edited.luma_floor, 0.05);
  assert.equal(d.luma_floor, null);          // the original is not mutated
});

test('clearing a field disables that cut rather than making it zero', () => {
  // Zero is a real threshold (flag everything below 0 = nothing, above = all);
  // an empty input means "no cut", and those must not be confused.
  const d = editThreshold(draftThresholds({}), 'motion_floor', '');
  assert.equal(d.motion_floor, null);
});

test('garbage input leaves the previous draft value in place', () => {
  const d = editThreshold(draftThresholds({ motion_floor: 0.002 }), 'motion_floor', 'abc');
  assert.equal(d.motion_floor, 0.002);
});

test('only the fields the backend knows ever leave the draft', () => {
  const d = editThreshold(draftThresholds({}), 'motion_floor', '0.001');
  d.garbage = 42;
  assert.deepEqual(Object.keys(payloadFromDraft(d)),
                   ['motion_floor']);
});

test('a draft with no active cut yields an empty payload and no dry run', () => {
  assert.deepEqual(payloadFromDraft(draftThresholds({})), {});
});

// --- audio (wave 4) -----------------------------------------------------------

test('the audio cuts have a row in the panel, or they are invisible', () => {
  // The exact failure this table was written to prevent, and which happened
  // anyway to `first_frame_floor`: a cut the backend honours, named nowhere the
  // user can reach, configurable only by hand-editing config.json.
  const keys = thresholdFields().map((f) => f.key);
  assert.ok(keys.includes('silence_max'), 'silence_max has no panel row');
  assert.ok(keys.includes('audio_floor'), 'audio_floor has no panel row');
});

test('silent and quiet are different flags with different labels', () => {
  // A quiet clip can be normalised; a silent one cannot be rescued. Same split
  // as freeze vs still, and the labels have to carry it.
  assert.ok(FLAG_LABELS.silent);
  assert.ok(FLAG_LABELS.quiet);
  assert.notEqual(FLAG_LABELS.silent, FLAG_LABELS.quiet);
});

test('audio has THREE states and none of them is the others', () => {
  // "No track", "silent" and "nobody measured it" are three different facts
  // with three different remedies. Any two collapsed produce a bank that lies.
  assert.equal(audioState({ metrics: { audio_state: 'ok', silence_ratio: 0 } }), 'ok');
  assert.equal(audioState({ metrics: { audio_state: 'none' } }), 'none');
  // Measured before the metric existed: the summary carries NO audio keys.
  assert.equal(audioState({ metrics: { metrics_state: 'ok', motion_mean: 0.1 } }),
    'unmeasured');
  // Not measured at all.
  assert.equal(audioState({}), 'unmeasured');
});

test('the audio note tells the user what to DO about each state', () => {
  assert.match(audioNote({ metrics: { metrics_state: 'ok' } }), /re-?measure/i);
  assert.match(audioNote({ metrics: { audio_state: 'none' } }), /no (sound|audio)/i);
  assert.equal(audioNote({ metrics: { audio_state: 'ok', silence_ratio: 0.0,
    rms_dbfs: -14 } }), '');
});

test('a level is shown in dBFS and a share as a percentage', () => {
  // Raw floats are not readable: "-14.2 dBFS" is a level an audio person knows,
  // "0.42" is not a share anybody reads as 42%.
  assert.match(audioSummary({ audio_state: 'ok', rms_dbfs: -14.23,
    silence_ratio: 0.42 }), /-14\.2 dBFS/);
  assert.match(audioSummary({ audio_state: 'ok', rms_dbfs: -14.23,
    silence_ratio: 0.42 }), /42%/);
  assert.equal(audioSummary({ audio_state: 'none' }), '');
});

// --- 🎨 the look score -------------------------------------------------------

test('the look cut has a panel row that feeds a labelled flag', () => {
  // The whole reason this table exists: a cut the backend honours and the panel
  // omits is settable only by hand-editing config.json — invisibly.
  const field = thresholdFields().find((f) => f.key === 'aesthetic_floor');
  assert.ok(field, 'aesthetic_floor has no panel row');
  assert.equal(field.flag, 'low_aesthetic');
  assert.equal(field.direction, 'below');
  assert.ok(FLAG_LABELS.low_aesthetic);
});

test('the look flag wears the same words as the image bank does', () => {
  // Same model, same 1–10 scale, same finding. A user who learned "Low
  // aesthetic" on a still must not have to learn a second name for a shot.
  assert.match(FLAG_LABELS.low_aesthetic, /low aesthetic/i);
});

test('the look hint carries the published reference, not a default', () => {
  // The numbers are a REFERENCE for a user choosing their own cut — the field
  // ships empty on purpose. Dropping them from the hint would leave the only
  // 1–10 field in the panel with nothing to anchor a first guess to.
  const { hint } = thresholdFields().find((f) => f.key === 'aesthetic_floor');
  assert.match(hint, /LAION/);
  assert.match(hint, /4\.75/);
  assert.match(hint, /Find scenes/);
});

test('the look chip counts shots and offers itself only when some are flagged', () => {
  const clips = [
    { id: 1, flags: ['low_aesthetic'], metrics: { aesthetic_score: 2.9 } },
    { id: 2, flags: [], metrics: { aesthetic_score: 6.2 } },
  ];
  assert.equal(flagCounts(clips).low_aesthetic, 1);
  const chip = flagChips(clips).find((c) => c.flag === 'low_aesthetic');
  assert.equal(chip.count, 1);
  assert.equal(filterByFlag(clips, 'low_aesthetic').map((c) => c.id).join(), '1');
  assert.ok(!flagChips([{ id: 3, flags: [], metrics: {} }])
    .some((c) => c.flag === 'low_aesthetic'));
});

test('an unrated shot is not a low-aesthetic shot', () => {
  // "Nobody rated it" and "it rated badly" are different facts with different
  // remedies — one is fixed by running 🔎 Find scenes, the other by dropping the
  // shot. The backend never emits the flag without a score; this pins that the
  // UI never invents one either.
  const clips = [{ id: 1, flags: [], metrics: { metrics_state: 'ok' } }];
  assert.equal(flagCounts(clips).low_aesthetic, undefined);
  assert.equal(filterByFlag(clips, 'low_aesthetic').length, 0);
});
