/* WHERE a bank pass runs, and how many images that really is.
 *
 * THE ASK. "Every one of these passes should be settable to kept, undecided,
 * unkept or all, to choose where it applies — or individually. Each button, once
 * pressed, opens a window showing the available options, with the launch button
 * at the end."
 *
 * TWO VOCABULARIES, on purpose — the same split bankCaptionScope.js already
 * draws. The user reads "Kept", "Undecided", "Unkept"; the wire and the database
 * carry 'keep', 'pending', 'reject', the values stored in the status column since
 * the bank existed. Renaming either side would break stored filters and saved
 * queries, so the translation lives here and nowhere else.
 *
 * THE DEFAULT IS A FIFTH VALUE, and it has to be. "Kept + undecided" is what
 * every pass has always run on (`status != 'reject'` server-side), so it is the
 * option that sends NOTHING — a run that leaves it alone posts the byte-identical
 * request it posted before this control existed. The four the maintainer asked
 * for sit under it and each name themselves explicitly.
 *
 * UNKEPT IS NEVER THE DEFAULT AND NEVER PART OF ONE. Aiming a pass at the bin
 * spends real time (GPU, for most of them) on images you already decided against.
 * It is offered, because "un-reject and re-run" was the only way to caption or
 * measure something you had set aside; it is never where a click lands by
 * accident, and the dialog states the cost next to the choice.
 *
 * EVERY LINE CARRIES ITS OWN COUNT, and that count is the number the run walks —
 * read off `payload.pass_scopes`, which the server computes from the SAME clause
 * each pass's pool filters on. A line that announces one figure while the pass
 * acts on another is the defect this whole surface exists to end (the 🧹
 * Auto-reject counter that offered "5 930" and moved 0).
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */

/** The scope of a run, as the user picks it. Order = the order the dialog lists
 *  them: the default first, then the four explicit piles. */
export const PASS_SCOPE_OPTIONS = [
  {
    id: '',
    nounAll: 'kept and undecided',
    noun: 'kept or undecided',
    label: 'Kept + undecided',
    short: 'images',
    hint: 'What every pass has always run on.',
    statuses: null,
    piles: ['keep', 'pending'],
  },
  {
    id: 'keep',
    nounAll: 'kept',
    noun: 'kept',
    label: '✓ Kept only',
    short: 'kept',
    hint: 'The images you have already decided to keep.',
    statuses: ['keep'],
    piles: ['keep'],
  },
  {
    id: 'pending',
    nounAll: 'undecided',
    noun: 'undecided',
    label: 'Undecided only',
    short: 'undecided',
    hint: 'The images you have not ruled on yet.',
    statuses: ['pending'],
    piles: ['pending'],
  },
  {
    id: 'reject',
    nounAll: 'unkept',
    noun: 'unkept',
    label: '✕ Unkept only (the bin)',
    short: 'unkept',
    hint: 'Images you already rejected. Nothing here is deleted — but this run '
      + 'spends its time on shots you set aside.',
    statuses: ['reject'],
    piles: ['reject'],
    bin: true,
  },
  {
    id: 'all',
    nounAll: 'kept, undecided and unkept',
    noun: 'kept, undecided or unkept',
    label: 'All three, the bin included',
    short: 'images',
    hint: 'Kept, undecided and unkept together.',
    statuses: ['keep', 'pending', 'reject'],
    piles: ['keep', 'pending', 'reject'],
    bin: true,
  },
];

export const DEFAULT_PASS_SCOPE = '';

export function passScopeOption(scopeId) {
  return PASS_SCOPE_OPTIONS.find((o) => o.id === scopeId) || PASS_SCOPE_OPTIONS[0];
}

/** The `statuses` value to POST, or null when the key must be left out entirely. */
export function passScopeStatuses(scopeId) {
  return passScopeOption(scopeId).statuses;
}

/** Does this scope reach the rejected pile? Drives the warning, nothing else. */
function passScopeTouchesBin(scopeId) {
  return !!passScopeOption(scopeId).bin;
}

/** Has the server sent the per-pass run sizes yet?
 *
 *  "Not polled yet" is not "zero", and the two must never render as the same 0 —
 *  the rule this payload already follows for `unscanned` and for the caption
 *  counts. Until the first payload lands the dialog says so instead of quoting a
 *  number nobody measured. */
function passCountsKnown(payload, passId) {
  const p = payload?.pass_scopes?.[passId];
  return !!p && !!p.todo;
}

/**
 * How many images THIS pass would really touch, for this scope.
 *
 * @param payload the bank payload
 * @param passId  key into payload.pass_scopes
 * @param scopeId one of PASS_SCOPE_OPTIONS
 * @param redo    the "do it again anyway" lane (rescan / rescore / force). It is
 *                a different POOL, not a bigger number: a watermark rescan still
 *                skips the rows you dismissed, so its two figures are two
 *                measured queries and never one total reused twice.
 * @returns a number, or null when the server has not sent the figures yet.
 */
export function passScopeCount(payload, passId, scopeId, redo = false) {
  if (!passCountsKnown(payload, passId)) return null;
  const table = payload.pass_scopes[passId][redo ? 'all' : 'todo'];
  if (!table) return null;
  return passScopeOption(scopeId).piles
    .reduce((n, pile) => n + (Number(table[pile]) || 0), 0);
}


/**
 * How many images in THIS scope the pass will reach but cannot answer for,
 * because a PREREQUISITE pass never got to them. 0 when there is none, null when
 * the server sends no figure (an older build, or a pass with no prerequisite).
 *
 * WHY THIS EXISTS. 🎨 Medium runs no image inference at all — it multiplies the
 * CLIP embeddings ✨ Score cached. On a bank of 50 397 images where every
 * unclassified row but two sat in the bin, its window counted its pool honestly
 * ("2 images") and the run answered "0 classified, 2 skipped (not scored yet)":
 * those two rows had never been scored. The count was right about the POOL and
 * wrong about the WORK — on this screen, the same defect one step earlier.
 */
export function passScopeBlocked(payload, passId, scopeId, redo = false) {
  const table = payload?.pass_scopes?.[passId]?.[redo ? 'blocked_all' : 'blocked'];
  if (!table) return null;
  return passScopeOption(scopeId).piles
    .reduce((n, pile) => n + (Number(table[pile]) || 0), 0);
}

/** One scope line's words: "✓ Kept only — 412 images".
 *
 *  With no count yet it says so rather than showing a bare label that reads like
 *  a zero. When part of the line cannot be answered without ✨ Score, the line
 *  says so INSIDE itself: the user is choosing BETWEEN these lines, so the
 *  objection has to sit on the line it disqualifies. */
export function passScopeLineLabel(payload, passId, scopeId, redo = false) {
  const opt = passScopeOption(scopeId);
  const n = passScopeCount(payload, passId, scopeId, redo);
  if (n === null) return `${opt.label} — counting…`;
  const blocked = passScopeBlocked(payload, passId, scopeId, redo);
  // BOTH magnitudes, never one replacing the other. Showing only the feasible
  // count would trade a wrong number for a mute "0 images" — a dead button with
  // no explanation, which is the same defect wearing the opposite coat. So the
  // line reads "N in scope, M ready" and names the pass that closes the gap.
  const note = !blocked ? ''
    : ` in scope, ${Math.max(0, n - blocked)} ready — ✨ Score has not reached `
      + `${blocked} of them`;
  if (note) return `${opt.label} — ${n} ${n === 1 ? 'image' : 'images'}${note}`;
  return `${opt.label} — ${n} ${n === 1 ? 'image' : 'images'}`;
}

/** The launch button's words. It quotes the number it is about to move.
 *
 *  `selection` overrides everything: the server INTERSECTS a selection with the
 *  scope, so the run can only ever be SHORTER than the selection — "up to" is the
 *  honest bound, and quoting a bare number for a shorter run is the same lie the
 *  counts exist to end.
 *
 *  A pass with no countable pool (✂ Find crops & variants reads a cache only the
 *  server can size) gets the bare verb. A fabricated number would be worse than
 *  none: the rule is that every number on this surface is one somebody measured. */
export function passLaunchLabel({
  verb, payload, passId, scopeId, redo = false, selectionSize = 0, countable = true,
} = {}) {
  if (selectionSize > 0) return `${verb} up to ${selectionSize} selected`;
  if (!countable) return verb;
  const n = passScopeCount(payload, passId, scopeId, redo);
  if (n === null) return verb;
  return `${verb} ${n} ${n === 1 ? 'image' : 'images'}`;
}

/** Can this run start at all, and why not? '' when it can.
 *
 *  A zero-image run is refused HERE rather than started and reported as a success
 *  — the second defect this surface is built against: a button that does nothing
 *  and reports "scored N image(s)" from a count read back out of the database. */
export function passLaunchDisabledReason({
  payload, passId, scopeId, redo = false, selectionSize = 0, countable = true, live = false,
} = {}) {
  if (live) return 'A pass is already running on this bank.';
  if (selectionSize > 0) return '';
  if (!countable) return '';
  const n = passScopeCount(payload, passId, scopeId, redo);
  if (n === null) return '';        // unknown is not zero — stay clickable
  // A pool that exists but is entirely waiting on a prerequisite is a run that
  // provably writes nothing. Refused HERE, with the fix named, rather than
  // started and reported as "done — 0 classified, N skipped (not scored yet)"
  // seconds later, which is what read as "nothing happened".
  const blocked = passScopeBlocked(payload, passId, scopeId, redo);
  if (n > 0 && blocked >= n) {
    return `${n} image(s) in scope, 0 ready — this pass reads the embeddings `
      + '✨ Score computes and none of these has been scored, so it would write '
      + 'nothing. Run ✨ Score first, or pick a scope that has been scored.';
  }
  if (n === 0) {
    const opt = passScopeOption(scopeId);
    return `Nothing to do in this scope — 0 ${opt.short === 'images' ? 'image' : opt.short} `
      + 'would be touched. Pick another scope, or tick the line above to run it '
      + 'again on images that already have a result.';
  }
  return '';
}

/** The sentence a bin-bound run has to read before it starts.
 *
 *  '' for every other scope. Stated as a cost, not a scolding: the maintainer
 *  asked for this reach, and the one pass where it is provably free says so. */
export function passBinWarning(scopeId, { free = false, cost = '' } = {}) {
  if (!passScopeTouchesBin(scopeId)) return '';
  if (free) {
    return 'This run reaches the images you rejected. Nothing is deleted or '
      + 'un-rejected — and this pass costs no GPU time and re-reads no image, so '
      + 'including the bin is free.';
  }
  return 'This run reaches the images you rejected. Nothing is deleted or '
    + 'un-rejected, but the work is spent on shots you already set aside'
    + (cost ? ` — ${cost}.` : '.');
}
