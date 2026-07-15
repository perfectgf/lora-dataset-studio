export const SMALL_IMAGE_SOURCE = 'small_image_source';
export const KLEIN_SMALL_IMAGE = 'klein_small_image';

export const isSmallImageRescueRow = (image) =>
  image?.derivation_kind === SMALL_IMAGE_SOURCE
  || image?.derivation_kind === KLEIN_SMALL_IMAGE;

const countLabel = (count, singular, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`;

const SKIP_LABELS = {
  duplicates: ['duplicate', 'duplicates'],
  low_res: ['low-resolution image', 'low-resolution images'],
  extreme_ratio: ['extreme-ratio image', 'extreme-ratio images'],
  not_image: ['non-image', 'non-images'],
  errors: ['download error', 'download errors'],
};

/**
 * Honest, testable summary for a batched scrape import. Klein rescues are a
 * separate outcome: they must never be folded into the generic skipped count.
 */
export function summarizeScrapeImport({
  imported = 0,
  rescueQueued = 0,
  rescueFailed = 0,
  skipped = {},
} = {}) {
  const parts = [countLabel(imported, 'image imported', 'images imported')];
  if (rescueQueued) {
    parts.push(`${countLabel(rescueQueued, 'small image', 'small images')} queued for Klein review`);
  }
  if (rescueFailed) {
    parts.push(`${countLabel(rescueFailed, 'Klein rescue', 'Klein rescues')} failed`);
  }

  let skippedTotal = 0;
  for (const [reason, rawCount] of Object.entries(skipped || {})) {
    const count = Number(rawCount) || 0;
    if (!count) continue;
    skippedTotal += count;
    const [singular, plural] = SKIP_LABELS[reason]
      || [String(reason).replaceAll('_', ' '), `${String(reason).replaceAll('_', ' ')} items`];
    parts.push(`${countLabel(count, singular, plural)} skipped`);
  }

  return {
    message: parts.join(' · '),
    severity: rescueFailed || skippedTotal ? 'warning' : 'success',
  };
}

function resolvedChoice(original, candidate) {
  if (original.status === 'keep' && candidate.status === 'reject') return 'original';
  if (original.status === 'reject' && candidate.status === 'keep') return 'klein';
  if (original.status === 'reject' && candidate.status === 'reject') return 'reject';
  return null;
}

/** Build source/candidate pairs from the regular dataset payload. */
export function buildSmallImageRescuePairs(images = []) {
  const rows = Array.isArray(images) ? images : [];
  const byId = new Map(rows.map((image) => [image.id, image]));

  return rows
    .filter((image) => image.derivation_kind === KLEIN_SMALL_IMAGE)
    .map((candidate) => {
      const original = byId.get(candidate.parent_image_id);
      if (!original || original.derivation_kind !== SMALL_IMAGE_SOURCE) return null;
      const choice = resolvedChoice(original, candidate);
      const phase = candidate.status === 'failed'
        ? 'failed'
        : candidate.filename ? 'ready' : 'queued';
      return { original, candidate, phase, resolved: choice !== null, choice };
    })
    .filter(Boolean);
}

/**
 * Generic-grid projection: unresolved pairs live only in Curation; resolved
 * pairs expose only their kept winner. The rejected provenance counterpart is
 * intentionally retained in the payload/database but never offered to generic
 * bulk/status/delete controls.
 */
export function filterSmallImageRescueGrid(images = []) {
  const rows = Array.isArray(images) ? images : [];
  const winnerIds = new Set();
  for (const pair of buildSmallImageRescuePairs(rows)) {
    if (pair.choice === 'original') winnerIds.add(pair.original.id);
    if (pair.choice === 'klein') winnerIds.add(pair.candidate.id);
  }
  return rows.filter((image) => {
    return !isSmallImageRescueRow(image) || winnerIds.has(image.id);
  });
}
