/** Split a dropped stack of photos into the requests the server will take.
 *
 * A drop used to go up as ONE multipart request. The server caps a request at
 * 64 MiB (MAX_CONTENT_LENGTH) and an import at 20 files, and it answered the
 * first with a bare 413 "upload too large" — five to eight high-resolution
 * body shots were enough, and the dropzone only ever mentioned the per-image
 * pixel budget (_nofaceman, Discord, 2026-09-07). So the split happens here,
 * by BOTH figures, before anything is sent; the figures come from
 * GET /api/capabilities (`dataset_import.max_files_per_request` and
 * `.max_request_bytes`), never from a copy kept in the front — the fallbacks
 * below only cover a rolling update where the server has not published them yet.
 *
 * Its own .js module, like importPolicy.js: `node --test` parses .js and not
 * .jsx, and the packing rule is the whole point, so it gets locked by a test.
 */
export const IMPORT_FALLBACK_MAX_FILES = 20;                         // = IMPORT_MAX_FILES server-side
// The ceiling of a backend that does NOT publish its own — i.e. one from before
// the import route got a raised ceiling, where the generic 64 MiB applied. A
// current backend publishes `max_request_bytes` (512 MiB by default, because a
// photo drop is a local file copy, not a stranger's upload) and that is what
// gets used; this value only ever serves a rolling update.
export const IMPORT_FALLBACK_MAX_REQUEST_BYTES = 64 * 1024 * 1024;
// Multipart framing (boundaries, part headers, file names) rides in the same
// request: leave it room under the ceiling rather than measuring it.
export const IMPORT_MULTIPART_HEADROOM_BYTES = 1024 * 1024;

function positiveInt(value, fallback) {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

/** {maxFiles, maxBytes, budgetBytes}: the server's two limits, and the byte
 *  budget one batch may fill (the ceiling minus the multipart headroom). */
export function importBatchLimits(policy) {
  const maxFiles = positiveInt(policy?.max_files_per_request, IMPORT_FALLBACK_MAX_FILES);
  const maxBytes = positiveInt(policy?.max_request_bytes, IMPORT_FALLBACK_MAX_REQUEST_BYTES);
  const budgetBytes = Math.max(1024 * 1024, maxBytes - IMPORT_MULTIPART_HEADROOM_BYTES);
  return { maxFiles, maxBytes, budgetBytes };
}

/** Greedy, order-preserving packing: a batch closes when the next file would
 *  push it past `maxFiles` or past the byte budget. A file that alone exceeds
 *  the budget can be sent by no batch at all — it comes back in `oversized`
 *  so the caller can NAME it instead of letting the server refuse the lot.
 *  Returns {batches: File[][], oversized: File[], limits}. */
export function planImportBatches(files, policy) {
  const limits = importBatchLimits(policy);
  const batches = [];
  const oversized = [];
  let current = [];
  let currentBytes = 0;
  for (const file of Array.from(files || [])) {
    const size = Number(file?.size) || 0;
    if (size > limits.budgetBytes) { oversized.push(file); continue; }
    const closes = current.length >= limits.maxFiles || currentBytes + size > limits.budgetBytes;
    if (closes && current.length) { batches.push(current); current = []; currentBytes = 0; }
    current.push(file);
    currentBytes += size;
  }
  if (current.length) batches.push(current);
  return { batches, oversized, limits };
}

export function formatMiB(bytes) {
  const mib = (Number(bytes) || 0) / (1024 * 1024);
  return `${mib >= 10 ? Math.round(mib) : Math.round(mib * 10) / 10} MiB`;
}

/** 'Importing 1–20 of 57…' for the toast that keeps a long drop honest. */
export function importBatchProgress(offset, batchSize, total) {
  return `Importing ${offset + 1}–${Math.min(offset + batchSize, total)} of ${total}…`;
}

/** The sentence for files that no batch can carry: which ones, how big, what to do. */
export function oversizedFilesMessage(oversized, limits) {
  if (!oversized || !oversized.length) return '';
  const named = oversized.slice(0, 3)
    .map((f) => `${f.name || 'a file'} (${formatMiB(f.size)})`).join(', ');
  const more = oversized.length > 3 ? ` and ${oversized.length - 3} more` : '';
  const n = oversized.length;
  // "the app accepts", not "the server takes": this runs on the user's own
  // machine, and naming a server sends them looking for one.
  return `${n} file${n === 1 ? '' : 's'} larger than the ${formatMiB(limits.maxBytes)} the app accepts `
    + `at a time ${n === 1 ? 'was' : 'were'} not sent: ${named}${more}. `
    + 'Resize or re-encode them, then drop them again.';
}
