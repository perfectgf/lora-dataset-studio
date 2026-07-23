/* 🗃 Import from a bank — the DECIDABLE part, kept free of JSX so `node --test`
   can run it.

   The dataset side never re-implements the bank's image picker: it only picks a
   BANK, and the server's own promote path does the rest (normalize + perceptual
   dedup against the dataset). What has to be right here is the ARITHMETIC and
   the WORDING, because "0 promotable" has two completely different meanings:

     * the bank was never triaged — promote only ever copies KEPT images, so a
       bank full of undecided rows offers zero. The fix is to go triage it.
     * every kept image is ALREADY on this dataset — nothing to do, and the user
       should be told that instead of being sent back to the bank for nothing.

   A bare "nothing to promote" conflates the two and has cost real time before,
   hence the explicit reason codes below. */

/** GET route for the honest per-target count of ONE bank. dataset_id is
 *  REQUIRED: the count is per dataset (an image promoted to ANOTHER dataset
 *  still counts here). Used by the bank page's own promote dialog, which only
 *  ever asks about the bank it has open. */
export const promotableUrl = (bankId, datasetId) =>
  `/api/bank/${Number(bankId)}/promotable?dataset_id=${Number(datasetId)}`;

/** The chooser's ONE request: the bank list, with every bank's promotable count
 *  for this dataset embedded. Asking /promotable per bank instead would cost
 *  1 + N requests to open a panel — the counts are all read from the same table
 *  in one grouped query server-side, so they come back together or not at all. */
export const banksUrl = (datasetId) =>
  `/api/banks?dataset_id=${Number(datasetId)}`;

/** bank id → promotable count, read off a /api/banks?dataset_id= payload.
 *  A row WITHOUT the field is left out (not defaulted to 0): "we don't know" and
 *  "nothing to import" say different things to the user, and bankImportOption
 *  renders the first as "Counting…" rather than a wrong "no kept images". */
export function promotableCounts(banks) {
  const out = {};
  for (const b of banks || []) {
    if (b && b.promotable != null) out[b.id] = Number(b.promotable);
  }
  return out;
}

/** POST route that starts the background promote job. */
export const promoteUrl = (bankId) => `/api/bank/${Number(bankId)}/promote`;

/** Body for "import everything promotable": an EMPTY image_ids means "every kept
 *  image not already on this dataset" (the bank's own no-selection semantics). */
export const promoteAllBody = (datasetId) => ({
  dataset_id: Number(datasetId),
  image_ids: [],
});

/** One row of the bank chooser: what to show, whether it can be imported, and —
 *  when it can't — WHY, in words the user can act on.
 *  `promotable` is the /promotable count, or null/undefined while it loads. */
export function bankImportOption(bank, promotable) {
  const total = Number(bank?.total || 0);
  const keep = Number(bank?.keep || 0);
  const reject = Number(bank?.reject || 0);
  const base = { id: bank?.id, name: bank?.name || '', total, keep };

  if (promotable == null) {
    return { ...base, count: null, ready: false, reason: 'loading', hint: 'Counting…' };
  }
  const count = Number(promotable);
  if (count > 0) {
    return {
      ...base,
      count,
      ready: true,
      reason: null,
      hint: `${count} kept image${count === 1 ? '' : 's'} to import`,
    };
  }
  if (total === 0) {
    return { ...base, count: 0, ready: false, reason: 'empty', hint: 'This bank has no images yet.' };
  }
  if (keep === 0) {
    // Triaged to nothing vs never triaged — both give 0, but only one is a mistake.
    if (reject >= total) {
      return {
        ...base,
        count: 0,
        ready: false,
        reason: 'all-rejected',
        hint: `All ${total} image${total === 1 ? '' : 's'} in this bank are rejected — nothing kept to import.`,
      };
    }
    return {
      ...base,
      count: 0,
      ready: false,
      reason: 'untriaged',
      hint: `No kept images yet — ${total - reject} still undecided. Triage this bank first.`,
    };
  }
  return {
    ...base,
    count: 0,
    ready: false,
    reason: 'already-imported',
    hint: `All ${keep} kept image${keep === 1 ? '' : 's'} are already in this dataset.`,
  };
}

/** The whole chooser, in the order /api/banks returned (newest first).
 *  `counts` maps bank id → promotable count (missing = still loading). */
export function bankImportOptions(banks, counts = {}) {
  return (banks || []).map((b) => bankImportOption(b, counts?.[b?.id]));
}

/** A bank job is "live" while it exists and hasn't finished — same contract the
 *  bank page reads off the embedded snapshot (bank_jobs). */
export const isBankJobLive = (activity) => Boolean(activity && !activity.finished);

/** The embedded job snapshot of one bank inside a /api/banks payload. */
export function bankActivity(banks, bankId) {
  const row = (banks || []).find((b) => Number(b?.id) === Number(bankId));
  return row?.activity || null;
}

/** What to say when a promote job ends: the server's own detail when it has one,
 *  an error when it failed, so the toast never invents a number. */
export function promoteOutcome(activity) {
  if (!activity) return null;
  if (activity.error) return { kind: 'error', text: `Import failed — ${activity.error}` };
  if (activity.cancelled) return { kind: 'info', text: activity.detail || 'Import stopped.' };
  return { kind: 'success', text: activity.detail || 'Import finished.' };
}
