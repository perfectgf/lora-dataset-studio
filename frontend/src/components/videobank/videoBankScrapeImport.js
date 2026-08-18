/**
 * 🕸 Scrape → VIDEO BANK — the client half of the scraper's third destination.
 *
 * Deliberately thin. The server answers the SAME shape as the image lane's
 * `/api/bank/scrape-import`, and the one rule worth pinning is the same too
 * (only the FIRST batch may create the bank; every later one has to resume into
 * the id that came back, or a 30-clip scrape quietly becomes five banks). So the
 * loop is `runBankScrapeImport`, reused, and what lives here is what genuinely
 * differs: the address, the batch size, and the sentence a user reads.
 *
 * WHY THE BATCH IS SO MUCH SMALLER. One image is capped at 12 MB and 20 s, one
 * video at 200 MB and 180 s. The server bounds a request at
 * `SCRAPE_VIDEO_IMPORT_MAX` for that reason; sending more would earn a 400, so
 * the client cuts the selection at the same number and sends batches — which is
 * exactly what it already does for a 200-image scrape.
 *
 * Pure logic (no JSX): `node --test` cannot parse JSX, and this is the part
 * worth pinning.
 */
// Extension included on purpose: `node --test` runs this file through the real
// ESM resolver, which does not do Vite's extensionless lookup.
import { runBankScrapeImport } from '../bank/bankScrapeImport.js';

export const VIDEO_BANK_SCRAPE_ENDPOINT = '/api/video-bank/scrape-import';

/** = SCRAPE_VIDEO_IMPORT_MAX server-side (video_bank_service.py). */
export const VIDEO_BANK_SCRAPE_BATCH = 6;

/** The destination as the server wants it, or null when it is not usable yet.
 *
 * Same two shapes as the image lane — {bank_id} resumes, {name} creates — with
 * one addition it does not need: a video bank may only receive downloads into a
 * folder the APP owns, because this lane promises never to write into the folder
 * you pointed a bank at. `scrapable` on the bank row is the server's answer to
 * that question, so a bank over someone's own rushes is not offered rather than
 * being offered and then refused. */
export function videoBankScrapeDestination({ mode, name, bankId, banks }) {
  if (mode === 'existing') {
    const id = Number(bankId);
    if (!Number.isInteger(id) || id <= 0) return null;
    const known = Array.isArray(banks) ? banks : null;
    if (known && !known.some((b) => Number(b?.id) === id && b?.scrapable)) return null;
    return { bank_id: id };
  }
  const clean = (name || '').trim();
  return clean ? { name: clean } : null;
}

/** Only the banks that can actually receive a scrape. */
export function scrapableVideoBanks(banks) {
  return (Array.isArray(banks) ? banks : []).filter((b) => b && b.scrapable);
}

/**
 * The server's skip reason for "it arrived, and this bank cannot hold it": a GIF
 * or an audio-only file the resolver was happy to keep, refused by the intake
 * because the bank's folder walk would never list it. Called out separately
 * below — lumping it in with the network failures would tell someone their clip
 * "could not be downloaded" when it downloaded perfectly well.
 */
const REFUSED_AT_INTAKE = 'not_video';

/**
 * One human sentence for a finished (or partly finished) run. `alreadyThere` is
 * NOT called a duplicate on purpose: it means the exact same bytes were already
 * in the folder — file identity, not a verdict about the footage, which only the
 * shot detection and the metrics pass produce.
 */
export function summarizeVideoBankScrapeImport(totals) {
  const { saved = 0, alreadyThere = 0, added = 0, skipped = {} } = totals || {};
  const bits = [`${saved} video(s) downloaded into the bank`];
  if (added && added !== saved) bits.push(`${added} inventoried`);
  if (alreadyThere) bits.push(`${alreadyThere} already in the folder`);
  const refused = Number(skipped[REFUSED_AT_INTAKE]) || 0;
  const failed = Object.entries(skipped)
    .reduce((n, [k, v]) => (k === REFUSED_AT_INTAKE ? n : n + (Number(v) || 0)), 0);
  if (refused) bits.push(`${refused} were not a video this bank can hold`);
  if (failed) bits.push(`${failed} could not be downloaded`);
  return bits.join(' · ');
}

/** The next step, said where the result is read: a scraped bank holds FILES and
 * no shots until the passes run, which is not obvious from a success toast. */
export function videoBankScrapeNextStep(totals) {
  return (totals?.added || 0) > 0
    ? 'Open the bank and run 🎬 Scan files → Find shots to cut them.'
    : '';
}

/** Run the whole import against the video lane's route. */
export function runVideoBankScrapeImport({ items, destination, post, onBatch }) {
  return runBankScrapeImport({
    items, destination, post, onBatch,
    endpoint: VIDEO_BANK_SCRAPE_ENDPOINT,
    batchSize: VIDEO_BANK_SCRAPE_BATCH,
  });
}
