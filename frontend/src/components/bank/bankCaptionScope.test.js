import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CAPTION_SCOPE_OPTIONS, captionAssertedCount, captionButtonLabel, captionCountsKnown,
  captionExistingCount, captionForcePileSize, captionGeneratedCount,
  captionIncludeAssertedLabel, captionOverwriteCount, captionProvenanceKnown,
  captionRecaptionConfirmation,
  captionRecaptionDisabledReason, captionRecaptionLabel, captionRecaptionNote,
  captionRecaptionRunSize, captionScopeCount, captionScopeDisabledReason,
  captionScopeNote, captionScopeStatuses, captionUnrecordedCount,
} from './bankCaptionScope.js';

/* The bank caption scope: three options, two vocabularies, and one number that has to
   be the number the pass moves. */

const counts = { keep: 40, pending: 900, reject: 60, caption_todo_keep: 12,
  caption_todo_pending: 300 };

test('the bin is offered, and it is never the default', () => {
  const ids = CAPTION_SCOPE_OPTIONS.map((o) => o.id);
  assert.deepEqual(ids, ['', 'keep', 'pending', 'reject', 'all']);
  // THE CHANGE OF PRINCIPLE, pinned. The bin used to be unreachable and the server
  // answered 400 for it; the maintainer asked to be able to aim a pass at it. What
  // must NOT change is where a click lands by accident:
  assert.equal(CAPTION_SCOPE_OPTIONS[0].id, '', 'the default must come first');
  assert.equal(captionScopeStatuses(''), null,
    'the default still sends nothing at all');
  assert.ok(!CAPTION_SCOPE_OPTIONS[0].piles.includes('reject'),
    'the default scope must never include the rejected pile');
  assert.deepEqual(captionScopeStatuses('reject'), ['reject'],
    'the bin is reachable only by naming it');
});

test('the wire carries the stored column values, the labels carry the human words', () => {
  assert.deepEqual(captionScopeStatuses('keep'), ['keep']);
  assert.deepEqual(captionScopeStatuses('pending'), ['pending']);
  const labels = CAPTION_SCOPE_OPTIONS.map((o) => o.label);
  assert.ok(labels.some((l) => /Kept/.test(l)));
  assert.ok(labels.some((l) => /Undecided/.test(l)));
  // …and never the raw column values, which are an implementation detail.
  assert.ok(!labels.some((l) => /\bkeep\b|\bpending\b/.test(l)));
});

test('the DEFAULT scope sends nothing at all', () => {
  // The byte-identical contract: a run that leaves the select alone must post the
  // same body the pass posted before this control existed. `null` is what the caller
  // spreads away; anything truthy would add a key.
  assert.equal(captionScopeStatuses(''), null);
  assert.equal(captionScopeStatuses(undefined), null);
  assert.equal(captionScopeStatuses('nonsense'), null);
});

test('the count is the UNCAPTIONED rows of the scope, not the size of the pile', () => {
  assert.equal(captionScopeCount(counts, 'keep'), 12);        // not 40
  assert.equal(captionScopeCount(counts, 'pending'), 300);    // not 900
  assert.equal(captionScopeCount(counts, ''), 312);
});

test('the rejected pile is in no count', () => {
  assert.equal(captionScopeCount({ ...counts, reject: 99999 }, ''), 312);
});

test('the button quotes the number it will move', () => {
  assert.equal(captionButtonLabel(0, counts, ''), 'Caption 312 images');
  assert.equal(captionButtonLabel(0, counts, 'keep'), 'Caption 12 kept');
  assert.equal(captionButtonLabel(0, counts, 'pending'), 'Caption 300 undecided');
});

test('a selection overrides the scope in the label, and says "up to"', () => {
  // The server intersects the selection with the non-rejected, uncaptioned set, so a
  // selection of 12 can run on 6. "up to" is the bound the pass really honours; a bare
  // "12" would be the same broken promise the scope counts exist to end.
  assert.equal(captionButtonLabel(7, counts, 'keep'), 'Caption up to 7 selected');
});

test('"not measured yet" is not rendered as zero', () => {
  // Before the first payload lands there is no count. Showing "Caption 0 images"
  // then would be a lie the user cannot distinguish from an empty bank.
  assert.equal(captionCountsKnown(null), false);
  assert.equal(captionCountsKnown({ keep: 3 }), false);
  assert.equal(captionCountsKnown(counts), true);
  assert.equal(captionButtonLabel(0, null, ''), 'Caption all');
  assert.equal(captionButtonLabel(0, {}, 'keep'), 'Caption kept');
});

test('the scope goes inert while a selection is live, and says why', () => {
  assert.equal(captionScopeDisabledReason(0, false), '');
  const why = captionScopeDisabledReason(5, false);
  assert.match(why, /selection/i);
  assert.match(why, /5/);
  assert.match(captionScopeDisabledReason(0, true), /already running/i);
});

test('the note names the number, the skip rule and the bin — every time', () => {
  const note = captionScopeNote(0, counts, 'keep');
  assert.match(note, /12/);
  assert.match(note, /no caption yet/i);
  assert.match(note, /Rejected images are never captioned/i);
});

test('the note switches to the selection when there is one', () => {
  const note = captionScopeNote(9, counts, 'keep');
  assert.match(note, /9 selected/);
  assert.ok(!/12/.test(note), 'a selection must not quote a status count');
});

test('an empty scope says so instead of offering a pass that does nothing', () => {
  const note = captionScopeNote(0, { caption_todo_keep: 0, caption_todo_pending: 3 }, 'keep');
  assert.match(note, /Nothing to caption/i);
});

/* RE-CAPTION — the forced pass. Two numbers, and neither may be the other:
   what it REWRITES (the whole pile) and what it DESTROYS (the captions already there). */

test('the forced run size is the PILE, not the uncaptioned part of it', () => {
  assert.equal(captionForcePileSize(counts, 'keep'), 40);        // not 12
  assert.equal(captionForcePileSize(counts, 'pending'), 900);    // not 300
  assert.equal(captionForcePileSize(counts, ''), 940);           // keep + pending
});

test('the rejected pile is in no forced run either', () => {
  assert.equal(captionForcePileSize({ ...counts, reject: 99999 }, ''), 940);
});

test('the overwrite count is the pile minus what has no caption yet', () => {
  assert.equal(captionOverwriteCount(counts, 'keep'), 28);       // 40 - 12
  assert.equal(captionOverwriteCount(counts, 'pending'), 600);   // 900 - 300
  assert.equal(captionOverwriteCount(counts, ''), 628);
});

test('the overwrite count never goes negative on an inconsistent payload', () => {
  // Two numbers polled a moment apart can disagree; a "-3 captions overwritten" is
  // worse than a stale 0, and clamping keeps the confirmation readable.
  assert.equal(captionOverwriteCount(
    { keep: 2, caption_todo_keep: 5, pending: 0, caption_todo_pending: 0 }, 'keep'), 0);
});

test('the re-caption button quotes the number it REWRITES', () => {
  assert.equal(captionRecaptionLabel(counts, 'keep'), 'Re-caption 40 kept');
  assert.equal(captionRecaptionLabel(counts, 'pending'), 'Re-caption 900 undecided');
  assert.equal(captionRecaptionLabel(counts, ''), 'Re-caption 940 images');
  // …and never a number nobody measured.
  assert.equal(captionRecaptionLabel(null, ''), 'Re-caption');
});

test('an inert re-caption button quotes NO number at all', () => {
  // A greyed button still offering "24 images" is the same defect one rung down: a
  // figure on screen that no click will act on. The reason is in the tooltip; the
  // label goes back to the bare verb.
  const why = captionRecaptionDisabledReason(5, false, counts, 'keep');
  assert.notEqual(why, '');
  assert.equal(captionRecaptionLabel(counts, 'keep', why), 'Re-caption');
  assert.equal(captionRecaptionLabel(counts, 'keep', ''), 'Re-caption 40 kept');
});

test('the two buttons quote different numbers, on purpose', () => {
  // Caption fills 12; Re-caption rewrites 40 of which 28 already had a caption.
  // If these ever collapsed into one figure, one of the two buttons would be lying.
  assert.equal(captionScopeCount(counts, 'keep'), 12);
  assert.equal(captionForcePileSize(counts, 'keep'), 40);
  assert.equal(captionOverwriteCount(counts, 'keep'), 28);
});

test('re-caption refuses to run on a selection, and says why', () => {
  const why = captionRecaptionDisabledReason(5, false, counts, 'keep');
  assert.match(why, /selection/i);
  assert.match(why, /5/);
  // The REASON matters: a selection can span pages that were never loaded, so the
  // overwrite count is unknowable client-side — and this button never runs on a
  // number it cannot state.
  assert.match(why, /cannot be counted|pile/i);
});

test('re-caption is inert while a pass runs, and before the counts land', () => {
  assert.match(captionRecaptionDisabledReason(0, true, counts, ''), /already running/i);
  assert.match(captionRecaptionDisabledReason(0, false, null, ''), /counts/i);
  assert.match(captionRecaptionDisabledReason(0, false, {}, ''), /counts/i);
});

test('re-caption is inert when there is no caption to overwrite', () => {
  const fresh = { keep: 40, pending: 900, caption_todo_keep: 40,
    caption_todo_pending: 900 };
  assert.match(captionRecaptionDisabledReason(0, false, fresh, 'keep'),
    /Nothing to re-caption/i);
  // …and it points at the button that CAN do something, instead of dead-ending.
  assert.match(captionRecaptionDisabledReason(0, false, fresh, 'keep'), /Caption/);
  assert.equal(captionRecaptionDisabledReason(0, false, counts, 'keep'), '');
});

test('with nothing recorded, the warning says so instead of claiming "generated"', () => {
  // This payload carries NO provenance breakdown, which is exactly what a bank
  // captioned before the column looks like. The honest reading of that is "nobody
  // recorded who wrote these", never "a model wrote these".
  const note = captionRecaptionNote(0, false, counts, 'keep');
  assert.match(note, /40/);                       // what it rewrites
  assert.match(note, /28/);                       // what it destroys
  assert.match(note, /never recorded/i);
  assert.ok(!/a model wrote/i.test(note), 'unknown authorship must not read as generated');
  assert.match(note, /no undo/i);
});

test('the warning is silent whenever the button cannot run', () => {
  // A warning about an impossible action is what teaches people to skip warnings.
  assert.equal(captionRecaptionNote(5, false, counts, 'keep'), '');
  assert.equal(captionRecaptionNote(0, true, counts, 'keep'), '');
  assert.equal(captionRecaptionNote(0, false, null, 'keep'), '');
  assert.equal(captionRecaptionNote(
    0, false, { keep: 4, caption_todo_keep: 4, pending: 0, caption_todo_pending: 0 },
    'keep'), '');
});

test('the confirmation is the Dataset\'s sentence, plus the two bank facts', () => {
  const q = captionRecaptionConfirmation(counts, 'keep');
  // Same opening and same closing word as dataset/captionCategory.js — the app asks
  // this question one way, not two.
  assert.match(q, /^Re-captioning overwrites the 28 existing caption\(s\)/);
  assert.match(q, /Continue\?$/);
  // …and the two things the dataset never has to say.
  assert.match(q, /40 kept image\(s\)/);
  // The bank's own admission: this payload records no author for any of them, and the
  // question says that rather than pretending the losses are all machine-written.
  assert.match(q, /no recorded author/i);
  assert.match(q, /cannot be undone/i);
});

test('the confirmation names the right pile for each scope', () => {
  assert.match(captionRecaptionConfirmation(counts, 'pending'),
    /900 undecided image\(s\)/);
  assert.match(captionRecaptionConfirmation(counts, ''),
    /940 kept and undecided image\(s\)/);
});

/* 🔒 PROVENANCE — three counts that must never be folded into two.

   The fixture below is the interesting bank: of 40 kept images, 12 have no caption,
   5 carry one the user wrote, 9 carry one whose author was never recorded, and the
   remaining 14 were written by a model. Every assertion here is arithmetic on that
   partition, so a helper that started double-counting would break several at once. */
const mixed = {
  keep: 40, pending: 900, reject: 60,
  caption_todo_keep: 12, caption_todo_pending: 300,
  caption_asserted_keep: 5, caption_asserted_pending: 100,
  caption_unrecorded_keep: 9, caption_unrecorded_pending: 200,
};

test('the four provenance buckets partition the pile exactly', () => {
  // blank + yours + unknown + machine = the pile. If these ever stopped adding up,
  // one of the sentences on screen would be describing images that are not there.
  const sum = captionScopeCount(mixed, 'keep')
    + captionAssertedCount(mixed, 'keep')
    + captionUnrecordedCount(mixed, 'keep')
    + captionGeneratedCount(mixed, 'keep');
  assert.equal(sum, captionForcePileSize(mixed, 'keep'));
  assert.equal(captionGeneratedCount(mixed, 'keep'), 14);
});

test('the run size is the pile MINUS what the pass spares', () => {
  // The number on the button has to be the number of rows that change. Quoting the
  // pile while sparing 5 of it is the same defect as quoting 5 930 and moving 0.
  assert.equal(captionRecaptionRunSize(mixed, 'keep'), 35);
  assert.equal(captionRecaptionLabel(mixed, 'keep'), 'Re-caption 35 kept');
  // …and with the opt-out, the pile again — because that is what will really run.
  assert.equal(captionRecaptionRunSize(mixed, 'keep', true), 40);
  assert.equal(captionRecaptionLabel(mixed, 'keep', '', true), 'Re-caption 40 kept');
});

test('the overwrite count drops the captions the pass keeps', () => {
  assert.equal(captionExistingCount(mixed, 'keep'), 28);   // 40 - 12
  assert.equal(captionOverwriteCount(mixed, 'keep'), 23);  // 28 - 5 spared
  assert.equal(captionOverwriteCount(mixed, 'keep', true), 28);
});

test('the warning names all three things, and never merges two of them', () => {
  const note = captionRecaptionNote(0, false, mixed, 'keep');
  assert.match(note, /rewrites 35 of the 40/);
  assert.match(note, /keeps the 5 caption\(s\) you wrote/i);
  assert.match(note, /overwrites 9 caption\(s\) whose origin was never recorded/i);
  assert.match(note, /overwrites 14 caption\(s\) a model wrote/i);
  assert.match(note, /no undo/i);
});

test('ticking the opt-out changes the warning instead of hiding it', () => {
  const note = captionRecaptionNote(0, false, mixed, 'keep', true);
  assert.match(note, /rewrites 40 of the 40/);
  assert.match(note, /because you ticked the box/i);
  assert.ok(!/keeps the 5/i.test(note), 'it must not still promise to keep them');
});

test('the confirmation says what is kept, and what has no known author', () => {
  const q = captionRecaptionConfirmation(mixed, 'keep');
  assert.match(q, /^Re-captioning overwrites the 23 existing caption\(s\)/);
  assert.match(q, /5 caption\(s\) you wrote or corrected by hand are kept/i);
  assert.match(q, /9 of them have no recorded author/i);
  const opted = captionRecaptionConfirmation(mixed, 'keep', true);
  assert.match(opted, /^Re-captioning overwrites the 28 existing caption\(s\)/);
  assert.match(opted, /are overwritten too/i);
});

test('the opt-out is offered only when there is something to protect', () => {
  // A tick box that would change nothing is a control that teaches people to tick
  // boxes. '' is the signal not to render it at all.
  assert.equal(captionIncludeAssertedLabel(mixed, 'keep'),
    'Also rewrite the 5 caption(s) I wrote');
  assert.equal(captionIncludeAssertedLabel(counts, 'keep'), '');
  assert.equal(captionIncludeAssertedLabel(null, 'keep'), '');
});

test('a pile whose only captions are yours says so, and points at the way out', () => {
  // Two different zeros. "Nothing is captioned here" sends you to Caption;
  // "the only captions here are yours" sends you to the tick box. Rendering both
  // as one message hides the protection at the moment it does all the work.
  const mineOnly = { keep: 10, pending: 0, caption_todo_keep: 7,
    caption_todo_pending: 0, caption_asserted_keep: 3, caption_asserted_pending: 0,
    caption_unrecorded_keep: 0, caption_unrecorded_pending: 0 };
  const why = captionRecaptionDisabledReason(0, false, mineOnly, 'keep');
  assert.match(why, /only 3 caption\(s\)/i);
  assert.match(why, /Also rewrite/i);
  // …and ticking it makes the button live again.
  assert.equal(captionRecaptionDisabledReason(0, false, mineOnly, 'keep', true), '');
});

test('a payload with no provenance at all degrades to "never recorded"', () => {
  // A server that predates the breakdown is not a bank of machine-written captions.
  // Reading its silence as "generated" would put an attribution on screen that
  // nothing measured, on the exact side that costs the user work.
  assert.equal(captionProvenanceKnown(counts), false);
  assert.equal(captionUnrecordedCount(counts, 'keep'), 28);
  assert.equal(captionGeneratedCount(counts, 'keep'), 0);
  // …and nothing is spared, so the button quotes the whole pile exactly as before.
  assert.equal(captionRecaptionLabel(counts, 'keep'), 'Re-caption 40 kept');
});
