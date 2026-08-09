/* 💾 Named arrangements of the ◉ LoRA Canvas — the browser half.
 *
 * The board holds ONE arrangement, the live one. That is right until the board
 * is used for what it is good at: twenty minutes spent laying two datasets'
 * renders out to judge a likeness, and then the only two moves were "leave it
 * there forever" or ✦ Tidy up, which throws it away. A preset is the third one.
 *
 * Everything here is pure: what to SEND from the board's live state, and what
 * the picker should SAY about a preset. The requests live in the component.
 */

/** The board's current arrangement, as the POST body.
 *
 * `positions` is the page's override map ({datasetId: {recordId: {x,y}}}) and
 * `imageNodes` its pinned-node map ({datasetId: {imageId: node}}) — the exact
 * two structures CanvasPage already holds, so nothing is recomputed and the
 * preset can never disagree with the board it was taken from.
 *
 * CLOSED pictures travel too (`visible: false` with their geometry). A preset
 * that silently re-opened everything you had closed would not be putting your
 * board back, it would be putting a different board back.
 */
export function canvasLayoutSnapshot({ positions = {}, imageNodes = {}, datasetIds = null } = {}) {
  const wanted = datasetIds == null ? null : new Set(datasetIds.map(Number));
  const keep = (id) => wanted == null || wanted.has(Number(id));
  const out = { positions: {}, images: {} };
  for (const [dsId, map] of Object.entries(positions || {})) {
    if (!keep(dsId)) continue;
    const rows = Object.entries(map || {}).map(([recordId, p]) => ({
      record_id: Number(recordId), x: Number(p?.x), y: Number(p?.y),
    })).filter((r) => Number.isFinite(r.record_id)
      && Number.isFinite(r.x) && Number.isFinite(r.y));
    if (rows.length) out.positions[String(dsId)] = rows;
  }
  for (const [dsId, map] of Object.entries(imageNodes || {})) {
    if (!keep(dsId)) continue;
    const rows = Object.values(map || {}).map((n) => ({
      image_id: Number(n?.imageId), x: Number(n?.x), y: Number(n?.y),
      w: Number(n?.w), h: Number(n?.h),
      visible: n?.visible !== false,
      group_id: n?.groupId ?? null,
      group_pos: n?.groupPos ?? null,
    })).filter((r) => Number.isFinite(r.image_id));
    if (rows.length) out.images[String(dsId)] = rows;
  }
  return out;
}

/** Is there anything on this board worth keeping? A "Save layout" button that
 *  writes an empty preset is a button that teaches the feature does not work. */
export function canvasLayoutIsEmpty(snapshot) {
  return !Object.keys(snapshot?.positions || {}).length
    && !Object.keys(snapshot?.images || {}).length;
}

/** What the picker prints under a preset's name — its size, in the board's own
 *  vocabulary, so the right one can be picked without applying three of them. */
export function canvasPresetSummary(preset) {
  const bits = [];
  const lanes = Number(preset?.lanes) || 0;
  const cards = Number(preset?.cards) || 0;
  const images = Number(preset?.images) || 0;
  bits.push(`${lanes} lane${lanes === 1 ? '' : 's'}`);
  if (cards) bits.push(`${cards} card${cards === 1 ? '' : 's'}`);
  if (images) bits.push(`${images} picture${images === 1 ? '' : 's'}`);
  return bits.join(' · ');
}

/** What the app says after a restore, INCLUDING what it could not put back.
 *
 * A preset kept for three weeks routinely names a run that has been deleted
 * since. Silence there is the worst answer available: the board comes back
 * almost right and the user spends ten minutes looking for the card that is
 * missing. The counts come from the server, which is the only side that knows
 * what still exists. */
export function canvasPresetApplied(result, preset) {
  const cards = Number(result?.applied?.cards) || 0;
  const images = Number(result?.applied?.images) || 0;
  const wantCards = Number(preset?.cards) || 0;
  const wantImages = Number(preset?.images) || 0;
  const head = `Layout “${preset?.name || 'preset'}” restored — `
    + `${cards} card${cards === 1 ? '' : 's'}, ${images} picture${images === 1 ? '' : 's'}`;
  const lostCards = Math.max(0, wantCards - cards);
  const lostImages = Math.max(0, wantImages - images);
  if (!lostCards && !lostImages) return head;
  const gone = [];
  if (lostCards) gone.push(`${lostCards} run${lostCards === 1 ? '' : 's'}`);
  if (lostImages) gone.push(`${lostImages} picture${lostImages === 1 ? '' : 's'}`);
  return `${head}. ${gone.join(' and ')} no longer exist, so nothing was put back for them.`;
}

/** A name the server will accept, or null. Trimmed and capped at the column's
 *  own 80 — a name silently truncated by the database is a preset the picker
 *  and the user disagree about. */
export const PRESET_NAME_MAX = 80;
export function canvasPresetName(raw) {
  const clean = String(raw ?? '').trim().slice(0, PRESET_NAME_MAX);
  return clean || null;
}
