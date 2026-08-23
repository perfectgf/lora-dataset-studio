/**
 * 🕸 Scrape → BANK — the client half of the scraper's second destination.
 *
 * The server bounds one request (same cap as the dataset outlet), so a big
 * "Select all" is sent in sequential batches. The one thing that MUST NOT be
 * naive here: when the destination is a NEW bank, only the FIRST batch may
 * create it — every later batch has to resume into the bank id that came back,
 * or a 200-image scrape would silently end up as four banks of 60.
 *
 * Pure logic (no JSX): `node --test` cannot parse JSX, and this is exactly the
 * part worth pinning.
 */

export const BANK_SCRAPE_BATCH = 60;      // = SCRAPE_IMPORT_MAX server-side

export function bankScrapeBatches(items, size = BANK_SCRAPE_BATCH) {
  const list = Array.isArray(items) ? items : [];
  const out = [];
  for (let i = 0; i < list.length; i += size) out.push(list.slice(i, i + size));
  return out;
}

/** The destination as the server wants it, or null when it is not usable yet. */
export function bankScrapeDestination({ mode, name, bankId }) {
  if (mode === 'existing') {
    const id = Number(bankId);
    return Number.isInteger(id) && id > 0 ? { bank_id: id } : null;
  }
  const clean = (name || '').trim();
  return clean ? { name: clean } : null;
}

/**
 * One human sentence for a finished (or partly finished) run. `alreadyThere` is
 * NOT called a duplicate on purpose: it means the exact same bytes were already
 * in the folder — file identity, not the bank's duplicate verdict, which only
 * its own passes produce.
 */
export function summarizeBankScrapeImport(totals) {
  const { saved = 0, alreadyThere = 0, added = 0, skipped = {} } = totals || {};
  const bits = [`${saved} image(s) downloaded into the bank`];
  if (added && added !== saved) bits.push(`${added} inventoried`);
  if (alreadyThere) bits.push(`${alreadyThere} already in the folder`);
  const failed = Object.entries(skipped)
    .reduce((n, [, v]) => n + (Number(v) || 0), 0);
  if (failed) bits.push(`${failed} could not be downloaded`);
  return bits.join(' · ');
}

export const BANK_SCRAPE_ENDPOINT = '/api/bank/scrape-import';

/**
 * Run the whole import. `post(url, body)` is the caller's JSON POST (injected so
 * this is testable without a server). Returns
 * {ok, bankId, created, saved, alreadyThere, added, skipped, error}.
 *
 * `endpoint` and `batchSize` exist for the 🎬 video lane, whose route answers the
 * SAME shape under another address and bounds a request at fewer items (a clip is
 * two orders of magnitude bigger than a photo). Everything that makes this
 * function worth pinning — only the first batch may create the bank, every later
 * one resumes into the id that came back — is identical on both, so it is one
 * function rather than two that drift.
 */
export async function runBankScrapeImport({ items, destination, post, onBatch,
  endpoint = BANK_SCRAPE_ENDPOINT, batchSize = BANK_SCRAPE_BATCH }) {
  const batches = bankScrapeBatches(items, batchSize);
  const totals = { saved: 0, alreadyThere: 0, added: 0, skipped: {} };
  let bankId = destination?.bank_id ?? null;
  let created = false;
  if (!batches.length) return { ok: false, error: 'nothing selected', ...totals };
  if (!destination) return { ok: false, error: 'pick a destination first', ...totals };

  for (let i = 0; i < batches.length; i += 1) {
    // First batch obeys the caller's destination; every later one resumes into
    // whatever bank now exists — that is what makes a 200-image scrape ONE bank.
    const body = bankId ? { items: batches[i], bank_id: bankId }
      : { items: batches[i], ...destination };
    onBatch?.({ index: i, count: batches.length, total: items.length });
    let d;
    try {
      d = await post(endpoint, body);
    } catch (e) {
      // The REAL post (fetchClient.postJson) THROWS on any non-2xx — it never
      // resolves to {ok:false}. Without this catch every server refusal (a 400
      // "dataset folder", a 409 "bank busy") escaped as an unhandled rejection:
      // no toast, the batches already imported uncounted, and the panel's
      // carefully written error branch dead code. The refusal becomes the same
      // envelope a soft failure uses, with the totals the earlier batches
      // actually earned.
      return { ok: false, bankId, created,
        error: e?.body?.error || e?.message || 'Unexpected error',
        status: e?.status, ...totals };
    }
    if (!d?.ok) {
      return { ok: false, bankId, created,
        error: d?.error || 'Unexpected error', ...totals };
    }
    bankId = d.bank_id ?? bankId;
    created = created || !!d.created;
    totals.saved += d.saved || 0;
    totals.alreadyThere += d.already_there || 0;
    totals.added += d.added || 0;
    // Files stored but the folder walk failed — the batch is not a failure
    // (the bytes are on disk) but the summary must not read as the perfect
    // run. Last one wins; the video lane's summary turns it into a sentence.
    if (d.sync_error) totals.syncError = d.sync_error;
    for (const [k, v] of Object.entries(d.skipped || {})) {
      totals.skipped[k] = (totals.skipped[k] || 0) + v;
    }
  }
  return { ok: true, bankId, created, ...totals };
}
