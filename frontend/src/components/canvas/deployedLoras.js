/* ⏏ The deployed-LoRA list behind the Canvas' bulk undeploy — grouping, the
 * request it sends, and what it reports back.
 *
 * WHY THE SCREEN EXISTS. Undeploying was one pill at a time, inside a node's
 * popover, and nothing anywhere said how many LoRAs were deployed at all. Asked
 * for by the maintainer: one button, the whole list, tick what goes.
 *
 * WHAT THIS FILE IS CAREFUL ABOUT. Every row here is a candidate for a DELETE,
 * so nothing in it is ever invented: the rows come from the server's own
 * attribution (only files this app deployed), and the request sends back exactly
 * the identity the server handed out. A row the server did not list can never be
 * built here.
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */

/** Family id → the words the board already uses for it. An unknown family keeps
    its id rather than being hidden: a LoRA in a folder we have no label for is
    still a LoRA the user may want gone. */
export const FAMILY_LABEL = {
  zimage: 'Z-Image',
  sdxl: 'SDXL',
  krea: 'Krea 2',
  flux: 'FLUX.1',
  flux2klein: 'FLUX.2 Klein',
  anima: 'Anima',
};

export function familyLabel(family) {
  return FAMILY_LABEL[family] || family || 'Unknown family';
}

/** A stable key for one deployed file. The pair (family, filename) is what the
    server de-duplicates on, so the UI keys on the same thing — anything else
    would let two rows disagree about being the same file. */
export function rowKey(row) {
  return `${row?.family || ''}::${row?.filename || ''}`;
}

/** The rows, grouped by DATASET and sorted for reading: datasets alphabetically,
    files by their label inside each.

    Grouped by dataset rather than by family because that is the question people
    actually ask of this list — "what is still deployed for Elsa?" — and because
    the family is already printed on every row. */
export function groupByDataset(rows) {
  const groups = new Map();
  for (const row of rows || []) {
    if (!row || !row.filename) continue;
    const id = row.dataset_id ?? null;
    if (!groups.has(id)) {
      groups.set(id, { datasetId: id, datasetName: row.dataset_name || `Dataset #${id}`, rows: [] });
    }
    groups.get(id).rows.push(row);
  }
  const out = [...groups.values()];
  for (const g of out) {
    g.rows.sort((a, b) => String(a.label || a.filename)
      .localeCompare(String(b.label || b.filename)));
  }
  out.sort((a, b) => String(a.datasetName).localeCompare(String(b.datasetName)));
  return out;
}

/** The POST body items for the ticked keys, built from the SERVER's rows.
    `keys` is a Set of rowKey() values. Rows the server never sent cannot appear,
    which is the point: this screen deletes files, and the only names it may name
    are the ones the server itself attributed to the app. */
export function undeployItems(rows, keys) {
  return (rows || [])
    .filter((row) => keys?.has?.(rowKey(row)))
    .map((row) => ({
      dataset_id: row.dataset_id,
      filename: row.filename,
      train_type: row.family,
    }));
}

/** The confirmation shown before a bulk removal. Says the COUNT, that the
    training saves are untouched (this is the reversible half — the checkpoint
    stays and can be deployed again), and where the files go. */
export function undeployConfirm(count) {
  return `Remove ${count} LoRA${count === 1 ? '' : 's'} from ComfyUI's loras folder?\n\n`
    + 'Your training saves are kept: each one can be deployed again from its'
    + ' checkpoint at any time.\n\n'
    + 'The files go to the trash, recoverable until you empty it in Settings.';
}

/** What the run actually did, from the server's ledger. Three outcomes, said
    apart — a flat "done" over twenty files would hide the two that did not go.
    Returns {type, text} for the toast. */
export function undeployOutcome(ledger) {
  const removed = (ledger?.removed || []).length;
  const missing = (ledger?.missing || []).length;
  const failed = (ledger?.failed || []).length;
  const bits = [];
  if (removed) bits.push(`${removed} removed from ComfyUI`);
  // Not an error: the file was already gone, which IS the outcome asked for.
  if (missing) bits.push(`${missing} already gone`);
  if (failed) bits.push(`${failed} refused`);
  if (!bits.length) {
    return { type: 'info', text: 'Nothing to undeploy — no file was selected.' };
  }
  const tail = removed
    ? ' Training saves are kept — you can deploy any of them again.'
    : '';
  return {
    type: failed ? 'warning' : 'success',
    text: `${bits.join(' · ')}.${tail}`,
  };
}

/** The label on the launch button: it QUOTES what it will move, so nobody
    presses it wondering. */
export function undeployButtonLabel(count) {
  return count ? `⏏ Undeploy ${count} selected` : '⏏ Undeploy';
}

/** The one-line state under the title: how much is deployed, and how much of it
    is ticked. Zero deployed is said plainly rather than drawn as an empty list. */
export function deployedSummary(rows, selectedCount) {
  const total = (rows || []).length;
  if (!total) return 'Nothing is deployed in ComfyUI right now.';
  const noun = `${total} LoRA${total === 1 ? '' : 's'} deployed`;
  return selectedCount ? `${noun} · ${selectedCount} selected` : noun;
}
