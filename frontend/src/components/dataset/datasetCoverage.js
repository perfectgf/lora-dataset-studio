// 🔍 Coverage — turning the backend's per-axis counts into something readable.
//
// The composition meter directly above this panel counts face / bust / body /
// back against a target. It can go fully green on a set where every image is the
// same pose, one outfit, one light — and it will, because those are not things
// it counts. This panel is the second question: what did the set never show?
//
// Pure functions, no JSX, because `node --test` cannot parse JSX and the
// distribution logic is exactly the part worth proving. The component below is
// then only markup over these.
//
// Axis and bucket ids come from the server payload (`caption_coverage.py`) and
// are keyed on here — they are contract, never renamed without an alias.

/** Rows for one axis: every bucket, in lexicon order, with the state a reader
 *  needs to hear. `state` is 'ok' | 'thin' | 'gap' (a core bucket nobody
 *  mentioned) | 'none' (absent, but nobody needs it).
 *
 *  `imageIds` is the list of images the bucket counted, when the payload carries
 *  it (the dataset panel asks for it, the bank one does not) — it is what turns
 *  a number into somewhere to go. `null` means "this payload never said", which
 *  is NOT the same as an empty list, and the difference decides whether a chip
 *  can be clicked at all. */
export function axisRows(axis) {
  const buckets = (axis && axis.buckets) || [];
  return buckets.map((b) => ({
    id: b.id,
    label: b.label,
    count: b.count || 0,
    state: b.count > 0 ? (b.thin ? 'thin' : 'ok') : (b.core ? 'gap' : 'none'),
    imageIds: Array.isArray(b.image_ids) ? b.image_ids : null,
  }));
}

/** What the grid's filter bar calls a coverage pick. The axis is part of the
 *  name because the bucket alone is ambiguous across axes — "studio" is both a
 *  light and a place, and the two are different chips on the same panel. */
export function coverageFilterLabel(axis, row) {
  return `${row.label} — ${String(axis.label || '').toLowerCase()}`;
}

/** One line per axis, the thing a screen reader announces. Says what IS there
 *  first — a panel that only lists absences reads as an accusation. */
export function axisSummary(axis) {
  const rows = axisRows(axis);
  const seen = rows.filter((r) => r.count > 0);
  if (!seen.length) return `${axis.label}: nothing in the captions names one.`;
  const shown = seen.map((r) => `${r.label} ${r.count}`).join(', ');
  const gaps = rows.filter((r) => r.state === 'gap').map((r) => r.label);
  return gaps.length
    ? `${axis.label}: ${shown} — no ${gaps.join(', no ')}.`
    : `${axis.label}: ${shown}.`;
}

/** Whether the panel has anything real to say, and why not when it does not.
 *  Answered from the payload so the UI never renders an empty bar and lets the
 *  user read "no gaps" into what is actually "nothing was measured". */
export function coverageReadiness(coverage) {
  if (!coverage) return { ready: false, reason: 'Reading coverage…' };
  if (!coverage.total) {
    return { ready: false, reason: 'No images yet — add some and the variety read appears here.' };
  }
  if (!coverage.captioned) {
    return {
      ready: false,
      reason: 'No captions yet, so variety cannot be read. Run the caption pass '
        + 'and this panel fills in — the composition counts above are all that is known so far.',
    };
  }
  return { ready: true, reason: '' };
}

/** The one-line header: which images this is read from. Never "your dataset" —
 *  the pool is not the whole dataset (rejected and failed are out) and the panel
 *  that hides that is the panel people argue with. */
export function coverageScope(coverage) {
  if (!coverage) return '';
  const { total, captioned, uncaptioned } = coverage;
  // NOT "kept images": the pool is everything the composition bar counts, which
  // includes undecided ones and excludes only rejected and failed. Calling that
  // "kept" was wrong on any dataset mid-triage — and a panel about honesty
  // cannot mislabel the set it read.
  const base = `${captioned} of ${total} image${total === 1 ? '' : 's'} captioned`;
  return uncaptioned ? `${base} · ${uncaptioned} not read` : base;
}

/** Actionable one-liner for the "generate more" nudge: the concrete things to
 *  add, taken from the CORE gaps only. Empty string when there is nothing
 *  honest to suggest — a suggestion invented to fill a box is worse than none. */
export function generateMoreHint(coverage) {
  if (!coverage || !coverage.axes) return '';
  const wants = [];
  for (const axis of coverage.axes) {
    for (const b of axis.buckets || []) {
      if (b.core && !b.count) wants.push(b.label);
    }
  }
  if (!wants.length) return '';
  return `Generate or import more: ${wants.slice(0, 5).join(', ')}.`;
}
