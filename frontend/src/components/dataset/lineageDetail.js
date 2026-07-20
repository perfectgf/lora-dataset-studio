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

export function configRows(config) {
  if (!config || typeof config !== 'object') return [];
  const rows = [];
  for (const [key, label] of CONFIG_LABELS) {
    const v = config[key];
    if (v === undefined || v === null || v === '') continue;
    rows.push({ label, value: typeof v === 'object' ? JSON.stringify(v) : String(v) });
  }
  return rows;
}

/* True when a node carries any annotation — the run itself or any checkpoint —
   so the graph can mark it with a ● badge. */
export function noteBadge(node) {
  if (!node) return false;
  if (node.has_note) return true;
  return (node.checkpoints || []).some((c) => (c.note || '').trim());
}
