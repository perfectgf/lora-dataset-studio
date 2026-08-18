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
 * The same two shapes as the image lane — {bank_id} resumes, {name} creates —
 * and now the same permissiveness: ANY bank can receive a scrape, including one
 * you pointed at your own folder. Picking it is the consent.
 *
 * `scrapable` is still consulted, because one refusal survives that rule: a bank
 * sitting on a dataset's own folder. The server is the authority; checking here
 * only means the user is not offered a choice that would come back a 400. */
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

/** The banks a scrape may be sent to — in practice all of them, minus any bank
 * whose folder belongs to a dataset. */
export function scrapableVideoBanks(banks) {
  return (Array.isArray(banks) ? banks : []).filter((b) => b && b.scrapable);
}

/**
 * The line to show under the chosen bank, or '' when there is nothing to say.
 *
 * The honesty moved here from a refusal. A bank the app created has a folder
 * nobody else looks at, so a download into it is unremarkable and saying so
 * would be noise. A bank pointed at a folder of your own is the case worth one
 * sentence BEFORE the click — with the path, because "this bank's folder" is
 * exactly the part someone would have to go and check.
 */
export function videoBankScrapeFolderNotice(bank) {
  if (!bank || bank.app_folder) return '';
  const path = typeof bank.source_path === 'string' ? bank.source_path.trim() : '';
  return path
    ? `Downloads will be added to this bank’s folder on disk: ${path}`
    : 'Downloads will be added to this bank’s folder on disk.';
}

/** The selected bank row, from the id the <select> holds (always a string). */
export function findVideoBank(banks, bankId) {
  const id = Number(bankId);
  if (!Number.isInteger(id) || id <= 0) return null;
  return (Array.isArray(banks) ? banks : []).find((b) => Number(b?.id) === id) || null;
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
/** Like `not_video`, a reason with its own sentence: on an install without the
 * scrape extras EVERY direct-file item fails, and "could not be downloaded"
 * sends someone to check their connection for a missing package. */
const MISSING_EXTRAS = 'no_curl';

export function summarizeVideoBankScrapeImport(totals) {
  const { saved = 0, alreadyThere = 0, added = 0, skipped = {},
          syncError = '' } = totals || {};
  const bits = [`${saved} video(s) downloaded into the bank`];
  if (added && added !== saved) bits.push(`${added} inventoried`);
  if (saved && !added) {
    // Downloaded-but-not-inventoried used to read as the perfect run: the
    // `added &&` guard above short-circuits at zero, which is exactly the case
    // that needs a sentence, not the one that can do without.
    bits.push(syncError
      ? `none inventoried — ${syncError}`
      : 'none inventoried yet — press ↻ Rescan folder in the bank');
  }
  if (alreadyThere) bits.push(`${alreadyThere} already in the folder`);
  const refused = Number(skipped[REFUSED_AT_INTAKE]) || 0;
  const missingExtras = Number(skipped[MISSING_EXTRAS]) || 0;
  const failed = Object.entries(skipped)
    .reduce((n, [k, v]) => (k === REFUSED_AT_INTAKE || k === MISSING_EXTRAS
      ? n : n + (Number(v) || 0)), 0);
  if (refused) bits.push(`${refused} were not a video this bank can hold`);
  if (missingExtras) {
    bits.push(`${missingExtras} need the scraper extras (Setup → Install scraper extras)`);
  }
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
