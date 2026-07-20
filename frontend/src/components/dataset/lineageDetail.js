/* Pure helpers for the Lab detail panel (LineageDetailPanel.jsx). Kept in a
   JSX-free module so `node --test` can import and exercise them directly — the
   panel component imports these same functions. */

/* Known config keys → friendly labels, in the order the inspector shows them.
   Unknown/empty keys are skipped so a run only lists what it actually recorded. */
const CONFIG_LABELS = [
  ['rank', 'Rank'], ['alpha', 'Alpha'], ['learning_rate', 'Learning rate'],
  ['optimizer', 'Optimizer'], ['timestep_weighting', 'Timestep weighting'],
  ['network', 'Network'], ['ema', 'EMA'], ['steps', 'Steps'],
  ['base_model', 'Base model'], ['dataset_version', 'Dataset version'],
];

/* One config value formatted the way the inspector shows it — or null when the
   run didn't record that key (undefined/null/'' all mean "absent"). Shared by
   the single-run inspector and the two-run diff so both read a value identically. */
function formatValue(config, key) {
  if (!config || typeof config !== 'object') return null;
  const v = config[key];
  if (v === undefined || v === null || v === '') return null;
  return typeof v === 'object' ? JSON.stringify(v) : String(v);
}

export function configRows(config) {
  const rows = [];
  for (const [key, label] of CONFIG_LABELS) {
    const value = formatValue(config, key);
    if (value === null) continue;
    rows.push({ label, value });
  }
  return rows;
}

/* Side-by-side diff of two runs' configs for the Lab compare panel. Returns one
   row per known key that AT LEAST ONE run recorded, in CONFIG_LABELS order:
   { key, label, a, b, changed } — a/b are the formatted values (null when that
   side didn't record it), changed is true when they differ. A key present on
   only one side counts as changed (null vs value). Two legacy runs that both
   recorded nothing yield [] so the panel can say so honestly. Pure/derived —
   no mutation, no backend. */
export function diffConfigs(aConfig, bConfig) {
  const rows = [];
  for (const [key, label] of CONFIG_LABELS) {
    const a = formatValue(aConfig, key);
    const b = formatValue(bConfig, key);
    if (a === null && b === null) continue;   // neither run recorded it — nothing to compare
    rows.push({ key, label, a, b, changed: a !== b });
  }
  return rows;
}

/* Reducer for the bounded-to-two "compare" selection on the graph. Toggling an
   already-picked run removes it; picking a run when two are already selected
   slides the window (drops the oldest) so a fresh pick always lands. Keeps the
   selection an array of at most two record ids, oldest first. */
export function toggleDiffSelection(selected, recordId) {
  const arr = selected || [];
  if (arr.includes(recordId)) return arr.filter((id) => id !== recordId);
  return [...arr, recordId].slice(-2);
}

/* True when a node carries any annotation — the run itself or any checkpoint —
   so the graph can mark it with a ● badge. */
export function noteBadge(node) {
  if (!node) return false;
  if (node.has_note) return true;
  return (node.checkpoints || []).some((c) => (c.note || '').trim());
}
