/* WHERE the 🎲 caption shortcut draws from — a dataset, or a BANK.
 *
 * Banks were the obvious missing half and the bigger one: the 🏷️ Caption pass
 * captions a bank long before anything is promoted, so the richest pile of real
 * captions on the machine was the one pile this shortcut could not reach.
 * (Asked for by the maintainer.)
 *
 * THE ONE THING THIS FILE EXISTS TO PROTECT is the choice already stored in
 * people's browsers. `studioCaptionDataset_v1` holds `{id, name}` from before
 * banks were a source; that shape carries NO kind, and reading it as anything
 * other than a dataset would silently repoint a locked choice at a bank with the
 * same number. So the key is unchanged and a missing `kind` means 'dataset' —
 * that default IS the alias path, not a convenience.
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */

export const DATASET = 'dataset';
export const BANK = 'bank';

/** How each source names itself on screen. */
export const SOURCE_LABEL = { [DATASET]: 'Dataset', [BANK]: 'Bank' };
export const SOURCE_ICON = { [DATASET]: '📁', [BANK]: '🖼️' };

/** One stored/selected source, or null when the value cannot be trusted.
 *
 *  `kind` is normalised to the two known values and defaults to 'dataset' for
 *  anything older or unrecognised — see the header. */
export function normaliseSource(raw) {
  const id = Number(raw?.id);
  if (!Number.isInteger(id) || id <= 0) return null;
  const kind = raw?.kind === BANK ? BANK : DATASET;
  const fallback = kind === BANK ? 'Bank #' + id : 'Dataset #' + id;
  const name = typeof raw?.name === 'string' && raw.name.trim()
    ? raw.name.trim()
    : fallback;
  return { kind, id, name };
}

/** The POST body for /api/studio/random-caption.
 *
 *  A dataset produces `{dataset_id}` — byte-identical to the request that
 *  shipped before banks existed, so nothing about the old lane moved. */
export function captionSourceBody(source) {
  const s = normaliseSource(source);
  if (!s) return null;
  return s.kind === BANK ? { bank_id: s.id } : { dataset_id: s.id };
}

/** The subtitle under a row in the picker: what this source IS, in its own
 *  vocabulary. A dataset counts its images and states its kind; a bank counts
 *  what is KEPT, because kept is the only pile the draw reads. */
export function sourceMeta(row, kind) {
  const parts = [];
  if (kind === BANK) {
    const keep = Number(row?.keep);
    const total = Number(row?.total);
    if (Number.isFinite(keep)) parts.push(keep + ' kept');
    if (Number.isFinite(total)) parts.push('of ' + total);
  } else {
    if (row?.kind) parts.push(row.kind);
    const total = Number(row?.images_total);
    if (Number.isFinite(total)) parts.push(total + ' image' + (total === 1 ? '' : 's'));
  }
  return parts.join(' · ');
}

/** The chip/tooltip wording for the locked source. Names the KIND, because
 *  "locked: portraits" is ambiguous the moment a bank and a dataset can share a
 *  name — and on this machine they routinely do (a bank promotes into the
 *  dataset it is named after). */
export function lockedLabel(source) {
  const s = normaliseSource(source);
  if (!s) return '';
  return `${SOURCE_LABEL[s.kind]}: ${s.name}`;
}

/** The refusal wording when a draw comes back empty, in the source's own noun.
 *  The server sends its own sentence for 422; this is the client-side fallback
 *  for "200 but nothing usable". */
export function emptyDrawMessage(source) {
  const s = normaliseSource(source);
  const noun = s && s.kind === BANK ? 'bank' : 'dataset';
  return `This ${noun} did not return a usable caption. Caption some kept images, then try again.`;
}
