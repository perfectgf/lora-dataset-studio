/* The two pure bits of the ◉ LoRA Canvas' filter BAR, out of the JSX so
 * `node --test` can hold them (the runner does not parse JSX).
 *
 * Both used to be inline expressions inside the old panel: a chain of ternaries
 * for the status label, and no dataset search at all — the popover's "find a
 * dataset in this list of thirty" is new, and it is the one thing the bar owes
 * the panel it replaced. A list you had to scroll was fine when it was three
 * columns wide and permanently on screen; inside a 20-rem popover it is not.
 */

/** One run status, as the filter names it. `unknown` is a real state, not a
 *  gap: a run whose lineage the board could not read is filterable like any
 *  other, and calling it "Other" would hide that. */
export function statusLabel(status) {
  switch (status) {
    case 'active': return 'Active';
    case 'completed': return 'Completed';
    case 'error': return 'Errors';
    default: return 'Unknown';
  }
}

/**
 * Does this dataset match what was typed in the picker?
 *
 * Name first, because that is what people type. The MODEL FAMILY matches too
 * ("krea" brings up every Krea lane), which is the query the old three-column
 * list answered by eye and a popover cannot. An empty query matches
 * everything — a search box that hides the list until you type is a list you
 * cannot browse.
 */
export function matchesDatasetQuery(dataset, query) {
  const q = String(query ?? '').trim().toLowerCase();
  if (!q) return true;
  const name = String(dataset?.name ?? '').toLowerCase();
  if (name.includes(q)) return true;
  return (dataset?.families || []).some((f) => String(f).toLowerCase().includes(q));
}
