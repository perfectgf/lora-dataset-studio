/* ✂ Crop and ✨ Upscale & improve made IN THE BANK — everything the screens read
 * off a row or off the payload, with no JSX so `node --test` covers it.
 *
 * WHY THE BANK GOT THESE AT ALL. Asked for by nofaceman on Discord (backed by
 * mr.arrow): the Bank is where the filtering and the curation live, but it had
 * neither crop nor upscale — so reframing one shot meant Bank → Dataset →
 * export to a NEW bank → Dataset again. Cropping and upscaling here collapses
 * that into curate → edit → re-analyse → promote, in one place.
 *
 * THE ONE THING EVERY SURFACE HERE MUST GET RIGHT is that an edit changes the
 * BYTES BEHIND AN UNCHANGED URL. `/thumb/<id>` answers with max-age=3600, so a
 * cropped image whose URL did not move keeps showing its old framing for an
 * hour — indistinguishable from a button that did nothing. That is why the
 * server puts the generation in the blob's name and in the thumbnail's name, and
 * why `imageVersionQuery` exists: the last link of that chain is the query the
 * browser sees.
 */
import {
  availableImproveEngines, improveEngineBlockedReason,
} from '../../utils/improveEngines.js';

/** How each edit is named on screen. Method ids are stored in user databases
    (`bank_image.edit_method`) — the WORDS may change, the keys may not. */
const EDIT_LABEL = {
  crop: '✂ Cropped here',
  improve: '✨ Improved here',
};

/** The cache-busting query for one bank image's `/file/` and `/thumb/` URLs.
 *
 * Both transforms have to be in it, and for different reasons:
 *   · `r` — a turn changes the pixels at an unchanged URL (this one already
 *     shipped with the rotation lane);
 *   · `e` — the EDIT GENERATION, which is what makes a SECOND crop of the same
 *     image visible. Keying on "is edited" alone would serve the first crop
 *     forever, because re-cropping produces different pixels under an unchanged
 *     `edit_method`.
 * The server reads neither: they are cache keys, not parameters. */
export function imageVersionQuery(img) {
  const parts = [];
  if (img?.rotation) parts.push(`r=${img.rotation}`);
  if (img?.edit_generation) parts.push(`e=${img.edit_generation}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

/** The grid badge for a bank-side edit, or null. An edited image LOOKS like any
    other image, so without this the grid cannot answer "did my crop land?" —
    and, more importantly, cannot warn that ↩ Revert has something to take back. */
export function editBadge(img) {
  const method = img?.edit_method;
  if (!method) return null;
  return {
    text: method === 'improve' ? '✨' : '✂',
    label: EDIT_LABEL[method] || method,
    title: method === 'improve'
      ? 'Upscaled & improved in this Bank. Your own file was not modified — ↩ Revert brings the original back.'
      : 'Cropped in this Bank. Your own file was not modified — ↩ Revert brings the original back.',
  };
}

/** How many images this bank has edited, split by which edit made the blob.
    Two numbers rather than one: ↩ Revert undoes both, but "12 cropped" and "12
    upscaled" answer different questions, and a single total would let a run of
    GPU-minutes hide behind a hand crop.

    Read off `payload.counts`, where the server puts EVERY per-bank tally, and
    deliberately with NO fallback to the payload root: a lenient reader would
    have quietly kept answering 0 (which is what "nothing edited yet" looks like)
    instead of failing where the mistake is. */
export function editCounts(payload) {
  const counts = payload?.counts;
  const cropped = Number(counts?.cropped) || 0;
  const improved = Number(counts?.improved) || 0;
  return { cropped, improved, total: cropped + improved };
}

/** The panel header's state line — what is true right now, in a fragment. */
export function editSummary(payload) {
  const { cropped, improved, total } = editCounts(payload);
  if (!total) return 'nothing edited yet';
  const bits = [];
  if (cropped) bits.push(`${cropped} cropped`);
  if (improved) bits.push(`${improved} improved`);
  return bits.join(' · ');
}

/** The ✨ engine buttons for the BANK, in display order.
 *
 * The two engines and their one-line positioning are shared with the dataset
 * lane (`utils/improveEngines.js`) on purpose — they are the same two passes and
 * a second copy of "what Klein does to skin" would drift. What is NOT shared is
 * the promise: a dataset improve creates a separate candidate to review, a BANK
 * improve REPLACES the image the bank shows. Saying "a separate candidate is
 * created" here would be false, and it would be false about the one thing the
 * user needs to know before spending GPU-minutes. */
export function bankImproveEngines(caps, { engines = null, live = false, todo = null } = {}) {
  return availableImproveEngines(caps).map((engine) => {
    const blocked = improveEngineBlockedReason(engine.id, {
      caps, engines, eligibleCount: 1,
    });
    // `live` and an empty pool are not engine problems — they block both buttons
    // equally, and the sentence has to say which of the two it is.
    // `todo === null` is "not counted yet", NOT "nothing to do": a payload still
    // loading must not disable the pass with a sentence claiming the bank is
    // already fully improved.
    const reason = live
      ? 'A pass is already running on this bank.'
      : (blocked || (todo === 0 ? 'Every image in this scope has already been improved.' : null));
    return {
      id: engine.id,
      label: `${engine.emoji} ${engine.label}`,
      summary: engine.summary,
      disabled: !!reason,
      reason,
      title: reason ? `${reason} ${engine.summary}` : `${engine.summary} ${BANK_IMPROVE_PROMISE}`,
    };
  });
}

/** Said under both engine buttons, and the reason this module does not reuse the
    dataset's confirm text. */
export const BANK_IMPROVE_PROMISE =
  'The result replaces what this Bank shows for that image — your own file is never'
  + ' modified, and ↩ Revert brings the original back.';

/** What a finished crop should say. Quotes the NEW size: a crop that landed one
    pixel off its box is invisible in a green "done", and visible here. */
export function cropOutcomeMessage(state) {
  const w = Number(state?.width) || 0;
  const h = Number(state?.height) || 0;
  const size = w && h ? ` — now ${w}×${h}` : '';
  return `Cropped${size}. Your own file was not modified, and every measurement`
    + ' taken from the old framing was cleared, so the next pass re-reads this image.';
}

/** The ↩ Revert confirmation. `ids` empty = the whole bank, which is a different
    (and much larger) promise than reverting the image under the cursor. */
export function revertConfirmMessage(payload, ids = null) {
  const { total } = editCounts(payload);
  const n = ids && ids.length ? ids.length : total;
  const what = ids && ids.length
    ? `the ${n} selected image(s)`
    : `all ${n} image(s) edited in this bank`;
  return `Throw away the ✂ crops and ✨ upscales of ${what}?\n\n`
    + 'Only copies made by the app are deleted — your own files were never modified,'
    + ' so this simply brings the original framing back. Any rotation the edit'
    + ' absorbed is given back to the image.\n\n'
    + 'The measurements taken from the edited pixels go with them, so the analysis'
    + ' passes will cover those images again.';
}

/** What the revert actually did, from the route's own ledger. Zero is reported
    as zero rather than as a green success — a "0 reverted" success is the exact
    shape of a broken button. */
export function revertOutcomeMessage(result) {
  const n = Number(result?.reverted) || 0;
  if (!n) {
    return {
      type: 'info',
      text: 'Nothing to revert — none of those images carries a ✂ crop or a ✨ upscale'
        + ' made in this bank.',
    };
  }
  return {
    type: 'success',
    text: `${n} image(s) back to the version this bank started from.`,
  };
}

