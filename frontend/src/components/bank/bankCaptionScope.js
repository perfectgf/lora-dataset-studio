/* 🏷️ WHICH pile a caption pass is aimed at, and how many images that really is.
   Pure so the arithmetic behind the button label is unit-tested rather than eyeballed
   in a running app.

   TWO VOCABULARIES, on purpose. The user reads "Kept" and "Undecided"; the wire and the
   database carry 'keep' and 'pending' — the values stored in the status column since the
   bank existed. Renaming either side would break stored filters and saved queries, so the
   translation lives here and nowhere else.

   THE BIN IS NO LONGER OUT OF REACH — and that is a change of principle, held on purpose.
   'reject' used to be absent here and refused by the server ("you curate from what you
   might keep, never from what you threw away"). The maintainer asked to be able to aim a
   pass at the rejected pile, so it is now an explicit choice: never the default, never
   part of the default, and the launch window states what it costs (a full captioning call
   per rejected image — the slowest pass there is) next to the option itself.

   THE SCOPES THEMSELVES LIVE IN bankPassScope.js, shared with every other pass. One list
   means the word "Undecided" cannot come to mean two different piles on two screens. */
import { PASS_SCOPE_OPTIONS, passScopeOption } from './bankPassScope.js';

/** The scopes, in the order the window shows them — the shared list, re-exported under the
 *  name the caption row has always used so no caller has to be rewritten to follow it. */
export const CAPTION_SCOPE_OPTIONS = PASS_SCOPE_OPTIONS;

function captionScopeOption(scopeId) {
  return passScopeOption(scopeId);
}

/** The `statuses` value to POST, or null when the key must be left out entirely. */
export function captionScopeStatuses(scopeId) {
  return captionScopeOption(scopeId).statuses;
}

/** Sum the per-pile figures a scope covers. ONE reader for every scope-shaped count in
 *  this file, so adding a pile can never leave one of them behind. */
function pileSum(counts, scopeId, prefix) {
  return captionScopeOption(scopeId).piles
    .reduce((n, pile) => n + (Number(counts?.[`${prefix}${pile}`]) || 0), 0);
}

/** How many images the pass would ACTUALLY caption for this scope.
 *
 *  NOT counts.keep / counts.pending. The pass skips rows that already carry a caption,
 *  so the pile size is not the run size, and quoting the pile would advertise work that
 *  never happens — the mistake the 🧹 Auto-reject counter already paid for once, where a
 *  button offering "5 930 flagged" rejected 0 and read as a broken feature.
 *  The server computes these two numbers with the same filter the job uses. */
export function captionScopeCount(counts, scopeId) {
  return pileSum(counts, scopeId, 'caption_todo_');
}

/** Has the server told us the per-scope run sizes yet?
 *
 *  "Not polled yet" is not "zero", and the two must never render as the same 0 — the
 *  rule this payload already follows for `unscanned`. Until the first payload lands the
 *  button keeps its old wording and stays clickable, rather than greying itself out on a
 *  count nobody has measured. */
export function captionCountsKnown(counts) {
  return counts != null && counts.caption_todo_keep !== undefined;
}

/** The button's own words. It states the number it is about to move — a button that
 *  announces one figure and acts on another is the specific misunderstanding this whole
 *  control exists to end. A selection OVERRIDES the scope (see captionScopeDisabled), so
 *  the label switches to the selection and stops quoting a status count. */
export function captionButtonLabel(selectedSize, counts, scopeId) {
  // "up to": the server INTERSECTS the selection with the non-rejected set, so a
  // selection holding rejected or already-captioned images runs shorter than its
  // own size. The pass cannot exceed it, and saying so is the honest bound —
  // quoting a bare `12` for a run of 6 is the same lie the scope counts exist to
  // end, just on the other axis.
  if (selectedSize > 0) return `Caption up to ${selectedSize} selected`;
  const opt = captionScopeOption(scopeId);
  if (!captionCountsKnown(counts)) return `Caption ${opt.short === 'images' ? 'all' : opt.short}`;
  return `Caption ${captionScopeCount(counts, scopeId)} ${opt.short}`;
}

/** Is the scope select inert right now, and why?
 *
 *  A SELECTION WINS. The server intersects the two — "kept only" plus a selection of
 *  undecided images would caption fewer than the button promises — so rather than let a
 *  user build that contradiction, the scope goes inert while a selection is live and the
 *  request omits `statuses` entirely. Returns '' when the control is live. */
export function captionScopeDisabledReason(selectedSize, live) {
  if (live) return 'A pass is already running on this bank.';
  if (selectedSize > 0) {
    return `Your selection decides what gets captioned (${selectedSize} image(s)). `
      + 'Clear it to caption by status instead.';
  }
  return '';
}

/** The sentence under the row: what this run will do, in full, including the two things
 *  a count alone never says — that already-captioned images are skipped, and that the
 *  rejected pile is out of reach whatever is chosen. */
export function captionScopeNote(selectedSize, counts, scopeId) {
  if (selectedSize > 0) {
    return `Captions up to ${selectedSize} selected image(s) that have no caption yet. `
      + 'Rejected images are never captioned.';
  }
  const opt = captionScopeOption(scopeId);
  if (!captionCountsKnown(counts)) {
    return 'Captions the kept and undecided images that have no caption yet. '
      + 'Rejected images are never captioned.';
  }
  const n = captionScopeCount(counts, scopeId);
  if (n === 0) {
    return `Nothing to caption — every ${opt.noun} image already has one.`;
  }
  const what = `${opt.nounAll} images`;
  return `Captions the ${n} ${what} that have no caption yet. `
    + 'Rejected images are never captioned.';
}

/* 🔄 RE-CAPTION — the same pass, forced, and the only DESTRUCTIVE thing this row can do.
 *
 * Why it has to exist at all: 🏷️ Caption skips rows that already carry a caption, so on a
 * fully captioned bank its run size is 0 and the button greys out — taking the engine, the
 * model and the pile selects down with it. The dials added beside them would be unreachable
 * on exactly the bank whose captions you want to redo with a better model.
 *
 * WHAT IT NOW SPARES. Captions carry an origin (backend services/caption_origin.py, the
 * 'asserted' token BankImage.face_cluster_origin already uses), so the pass SKIPS the ones a
 * human wrote or corrected instead of warning that it cannot tell them apart. That warning
 * used to be the honest thing to say; it is now the wrong thing to do.
 *
 * THREE COUNTS, THREE SENTENCES, and they are never merged:
 *   - what the run REWRITES — the number the button quotes, and the number the job walks;
 *   - what it KEEPS BECAUSE YOU WROTE IT — the protection, stated as a measured figure;
 *   - what it rewrites WHOSE ORIGIN WAS NEVER RECORDED — every caption written before the
 *     column existed. Those are re-captioned (their authorship cannot be recovered, and
 *     sparing them would make this button inert on every bank that exists today) but they
 *     are NOT "generated captions", and calling them that would be the app claiming to know
 *     something it does not. On a bank that predates the column this figure is the whole
 *     overwrite, which is exactly the warning the user needs.
 *
 * THE WAY OUT is a separate tick (`includeAsserted`), offered only when there is something
 * to protect, never pre-ticked, and named in its own confirmation. Someone who deliberately
 * wants their own captions redone by a better model must be able to; they must not get there
 * by leaving a key out of a request. bank_undo covers (status, reject_reason) and says in its
 * own header that it offers undo for nothing else — there is no undo to promise here either. */

/** How many images a FORCED run of this scope really walks — captioned or not.
 *
 *  This is NOT captionScopeCount: with force the pass drops its "no caption yet" filter,
 *  so the run size becomes the pile itself (server: _caption_scope_q with no extra filter).
 *  Quoting the uncaptioned count on a re-caption button would understate the run. */
export function captionForcePileSize(counts, scopeId) {
  // The default scope is the server's `status != 'reject'` set. keep + pending is that
  // set exactly — the three piles partition the bank, a fact a backend test pins so an
  // extra status value could never make this number drift in silence.
  return pileSum(counts, scopeId, '');
}

/** How many EXISTING captions this scope holds, whoever wrote them.
 *
 *  The pile minus the part of it that has no caption yet. Both terms come from the same
 *  payload, computed by the same server filter the job uses, so this is arithmetic on
 *  measured numbers rather than a second definition of "captioned". */
export function captionExistingCount(counts, scopeId) {
  return Math.max(
    0, captionForcePileSize(counts, scopeId) - captionScopeCount(counts, scopeId));
}

/** Per-scope reader for the two provenance figures the server sends.
 *
 *  Both DEFAULT TO 0 when the keys are absent, which is what a server that predates them
 *  sends — and 0 is the truthful reading there: nothing is marked, so nothing is spared,
 *  and every sentence below degrades to what it said before this existed. */
function scopedProvenance(counts, scopeId, prefix) {
  return pileSum(counts, scopeId, `${prefix}_`);
}

/** Captions a human wrote or corrected — what a forced run KEEPS. */
export function captionAssertedCount(counts, scopeId) {
  return scopedProvenance(counts, scopeId, 'caption_asserted');
}

/** Has the server sent the provenance breakdown at all?
 *
 *  A payload that predates it is not a bank of machine-written captions — it is a bank
 *  whose captions have no recorded author, which is the SAME state the column stores as
 *  NULL. Reading the absence as "generated" would put an attribution on screen that
 *  nothing measured, on the exact side that costs the user work. */
export function captionProvenanceKnown(counts) {
  return counts != null && counts.caption_unrecorded_keep !== undefined;
}

/** Captions whose author was never recorded — rewritten, and counted apart.
 *
 *  With no breakdown at all, EVERY existing caption falls here: unknown is what we
 *  actually know. That also makes the fallback the safe one — the sentence the user
 *  reads is the cautious one, not the confident one. */
export function captionUnrecordedCount(counts, scopeId) {
  if (!captionProvenanceKnown(counts)) {
    return Math.max(0, captionExistingCount(counts, scopeId)
      - captionAssertedCount(counts, scopeId));
  }
  return scopedProvenance(counts, scopeId, 'caption_unrecorded');
}

/** Captions a model wrote, on the record — rewritten without ceremony.
 *
 *  A REMAINDER, deliberately: the pile is partitioned into blank / asserted / unrecorded /
 *  the rest, so these four always add up to it. Asking the server for a fifth number that
 *  could disagree with the other four is how two counts of the same thing start to drift. */
export function captionGeneratedCount(counts, scopeId) {
  return Math.max(0, captionExistingCount(counts, scopeId)
    - captionAssertedCount(counts, scopeId)
    - captionUnrecordedCount(counts, scopeId));
}

/** How many images the forced run REALLY walks — the pile, minus what it spares.
 *
 *  This is the number the button quotes, and it must equal the number the server's own
 *  filter selects (image_bank_service.start_caption). `includeAsserted` is the opt-out:
 *  with it the run is the whole pile again. */
export function captionRecaptionRunSize(counts, scopeId, includeAsserted = false) {
  const pile = captionForcePileSize(counts, scopeId);
  if (includeAsserted) return pile;
  return Math.max(0, pile - captionAssertedCount(counts, scopeId));
}

/** How many existing captions the run DESTROYS — what it rewrites, minus the blanks. */
export function captionOverwriteCount(counts, scopeId, includeAsserted = false) {
  const existing = captionExistingCount(counts, scopeId);
  if (includeAsserted) return existing;
  return Math.max(0, existing - captionAssertedCount(counts, scopeId));
}

/** The re-caption button's words: the number of images it will REWRITE — the pile minus
 *  what it spares, never the pile itself once a protection exists. A button that says 40
 *  and moves 37 is the same defect as one that says 5 930 and moves 0, just smaller.
 *
 *  AND IT DROPS THE NUMBER WHEN IT CANNOT RUN. An inert button still quoting "24 images"
 *  is that defect one rung down: a figure on screen that no click will act on. Pass
 *  the inert reason (or '') and the label falls back to the bare verb. */
export function captionRecaptionLabel(counts, scopeId, inertReason = '',
                                      includeAsserted = false) {
  if (inertReason || !captionCountsKnown(counts)) return 'Re-caption';
  const opt = captionScopeOption(scopeId);
  return `Re-caption ${captionRecaptionRunSize(counts, scopeId, includeAsserted)} ${opt.short}`;
}

/** Is the re-caption button inert right now, and why? '' when it is live.
 *
 *  THE SELECTION CASE IS THE INTERESTING ONE. A selection can span pages that were never
 *  loaded (⬚ Select all fetches ids, not rows), so the client cannot know how many of the
 *  selected images already carry a caption. For a destructive button, "I cannot give you
 *  the number" means "I do not run": re-caption works by pile, and says so. 🏷️ Caption
 *  still honours the selection, so nothing the user could do before is lost. */
export function captionRecaptionDisabledReason(selectedSize, live, counts, scopeId,
                                               includeAsserted = false) {
  if (live) return 'A pass is already running on this bank.';
  if (selectedSize > 0) {
    return 'Re-caption works on a whole pile, not on a selection: how many of the '
      + `${selectedSize} selected image(s) already have a caption cannot be counted `
      + 'without loading every one of them, and this button never runs on a number it '
      + 'cannot state. Clear the selection to re-caption by pile.';
  }
  if (!captionCountsKnown(counts)) return 'Waiting for this bank\'s counts.';
  if (captionOverwriteCount(counts, scopeId, includeAsserted) === 0) {
    const opt = captionScopeOption(scopeId);
    const pile = opt.noun;
    const mine = captionAssertedCount(counts, scopeId);
    // TWO different zeros, and telling them apart is the whole point of the column:
    // "there is nothing captioned here" sends you to 🏷️ Caption, "the only captions
    // here are yours and I am keeping them" sends you to the tick box. Rendering
    // both as "nothing to re-caption" would hide the protection at the one moment
    // it is doing all the work.
    if (mine > 0 && !includeAsserted) {
      return `Nothing left to re-caption — the only ${mine} caption(s) in this pile are `
        + 'ones you wrote or corrected, and Re-caption keeps those. Tick '
        + '"Also rewrite the ones I wrote" to redo them anyway.';
    }
    return `Nothing to re-caption — no ${pile} image has a caption yet. `
      + 'Use Caption to write them first.';
  }
  return '';
}

/** The warning under the row: THREE facts, never folded into one number.
 *
 *  '' when the button is inert, because a warning about an action that cannot happen is
 *  noise that teaches people to skip warnings. */
export function captionRecaptionNote(selectedSize, live, counts, scopeId,
                                     includeAsserted = false) {
  if (captionRecaptionDisabledReason(selectedSize, live, counts, scopeId, includeAsserted)) {
    return '';
  }
  const opt = captionScopeOption(scopeId);
  const pile = captionForcePileSize(counts, scopeId);
  const run = captionRecaptionRunSize(counts, scopeId, includeAsserted);
  const mine = captionAssertedCount(counts, scopeId);
  const unknown = captionUnrecordedCount(counts, scopeId);
  const generated = captionGeneratedCount(counts, scopeId);
  const what = opt.nounAll;
  const parts = [`Re-caption rewrites ${run} of the ${pile} ${what} image(s) with the `
    + 'engine and model picked here.'];
  // What it KEEPS comes first when it keeps anything: the reassurance is the news.
  if (mine > 0 && !includeAsserted) {
    parts.push(`It keeps the ${mine} caption(s) you wrote or corrected by hand.`);
  } else if (mine > 0) {
    parts.push(`Including the ${mine} caption(s) you wrote or corrected by hand, because `
      + 'you ticked the box.');
  }
  if (unknown > 0) {
    parts.push(`It overwrites ${unknown} caption(s) whose origin was never recorded — `
      + 'written before this app tracked who writes a caption, so anything you typed back '
      + 'then is among them.');
  }
  if (generated > 0) {
    parts.push(`It overwrites ${generated} caption(s) a model wrote.`);
  }
  parts.push('No undo covers captions.');
  return parts.join(' ');
}

/** The confirmation, worded on the Dataset's own re-caption prompt
 *  (dataset/captionCategory.js: "Re-captioning overwrites the N existing caption(s).
 *  <rule> Continue?") so the app asks this question one way, not two. The bank adds what
 *  the dataset does not have to state: which pile, what is spared, and which part of the
 *  loss is the app admitting it does not know who wrote what. */
export function captionRecaptionConfirmation(counts, scopeId, includeAsserted = false) {
  const opt = captionScopeOption(scopeId);
  const pile = captionForcePileSize(counts, scopeId);
  const n = captionOverwriteCount(counts, scopeId, includeAsserted);
  const mine = captionAssertedCount(counts, scopeId);
  const unknown = captionUnrecordedCount(counts, scopeId);
  const what = opt.nounAll;
  let out = `Re-captioning overwrites the ${n} existing caption(s) among the ${pile} `
    + `${what} image(s).`;
  if (mine > 0 && !includeAsserted) {
    out += ` The ${mine} caption(s) you wrote or corrected by hand are kept.`;
  } else if (mine > 0) {
    out += ` The ${mine} caption(s) you wrote or corrected by hand are overwritten too — `
      + 'that is what the box you ticked does.';
  }
  if (unknown > 0) {
    out += ` ${unknown} of them have no recorded author and may include captions you `
      + 'wrote before this app started keeping track.';
  }
  return `${out} This cannot be undone. Continue?`;
}

/** The opt-out's own label, and whether to offer it at all.
 *
 *  Returns '' when there is nothing to protect — a tick box that would change nothing is
 *  a control that teaches people to tick boxes. It also quotes the number, so the gesture
 *  and its cost are read in one line rather than one being buried in a warning. */
export function captionIncludeAssertedLabel(counts, scopeId) {
  const mine = captionAssertedCount(counts, scopeId);
  if (!captionCountsKnown(counts) || mine <= 0) return '';
  return `Also rewrite the ${mine} caption(s) I wrote`;
}
