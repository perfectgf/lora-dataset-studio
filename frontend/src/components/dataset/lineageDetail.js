/* Pure helpers for the Lab detail panel (LineageDetailPanel.jsx). Kept in a
   JSX-free module so `node --test` can import and exercise them directly — the
   panel component imports these same functions. */

/* Known config keys → friendly labels, in the order the inspector shows them.
   Unknown/empty keys are skipped so a run only lists what it actually recorded.

   Each entry lists ALIASES because the row identity and the stored key are not
   the same string. The launch snapshot writes `lr`, `timestep_type` and
   `network_type`; this table originally only knew `learning_rate`,
   `timestep_weighting` and `network`, so those three rows never matched anything
   and the compare panel quietly showed a handful of knobs instead of the recipe.
   The FIRST alias stays the row's `key` (it is what the tests and any stored UI
   state address); the others are simply where the value may be found. */
const CONFIG_LABELS = [
  [['rank'], 'Rank'],
  [['alpha'], 'Alpha'],
  [['network', 'network_type'], 'Network'],
  [['lokr_factor'], 'LoKr factor'],
  [['lokr_full_rank'], 'LoKr full rank'],
  [['learning_rate', 'lr'], 'Learning rate'],
  [['optimizer'], 'Optimizer'],
  [['lr_scheduler'], 'LR scheduler'],
  [['warmup'], 'Warmup'],
  [['grad_accum'], 'Grad accumulation'],
  [['batch_size'], 'Batch size'],
  [['ema'], 'EMA'],
  [['timestep_weighting', 'timestep_type'], 'Timestep weighting'],
  [['content_or_style'], 'Krea content / style'],
  [['do_differential_guidance'], 'Differential guidance'],
  [['differential_guidance_scale'], 'Differential guidance scale'],
  [['resolution'], 'Resolution'],
  [['dropout'], 'Dropout'],
  [['dual_captions'], 'Dual captions'],
  [['quantize'], 'Quantise base model'],
  [['quantize_te'], 'Quantise text encoder'],
  [['low_vram'], 'Low-VRAM loading'],
  [['trigger'], 'Trigger word'],
  [['style_mode'], 'Style mode'],
  [['slider_mode'], 'Slider mode'],
  [['save_every'], 'Save every'],
  [['max_step_saves'], 'Max saved checkpoints'],
  [['sample_every'], 'Sample every'],
  [['recipe_version'], 'Recipe version'],
  [['effective_base'], 'Effective base'],
  [['training_adapter'], 'Training adapter'],
  [['base_weights'], 'Custom weights'],
  [['acknowledged_not_ready'], 'Launched despite blocker'],
  [['steps'], 'Steps'],
  [['masked'], 'Masked training'],
  [['family'], 'Family'],
  [['variant'], 'Variant'],
  [['source'], 'Launched from'],
  [['base_model'], 'Base model'],
  [['dataset_version'], 'Dataset version'],
];

/* One config value formatted the way the inspector shows it — or null when the
   run didn't record any of that row's keys (undefined/null/'' all mean
   "absent"). `false` is a VALUE, not an absence: a run that recorded
   `dual_captions: false` must read "false", never "—". Shared by the single-run
   inspector and the two-run diff so both read a value identically. */
function formatValue(config, keys) {
  if (!config || typeof config !== 'object') return null;
  for (const key of keys) {
    const v = config[key];
    if (v === undefined || v === null || v === '') continue;
    return typeof v === 'object' ? JSON.stringify(v) : String(v);
  }
  return null;
}

export function configRows(config) {
  const rows = [];
  for (const [keys, label] of CONFIG_LABELS) {
    const value = formatValue(config, keys);
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
  for (const [keys, label] of CONFIG_LABELS) {
    const a = formatValue(aConfig, keys);
    const b = formatValue(bConfig, keys);
    if (a === null && b === null) continue;   // neither run recorded it — nothing to compare
    rows.push({ key: keys[0], label, a, b, changed: a !== b });
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
