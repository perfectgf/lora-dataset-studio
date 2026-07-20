/* Pure helpers for the Lab's inline generation (RunLineageGraph.jsx). JSX-free so
   `node --test` exercises the selection/enable logic directly; the graph imports
   these same functions. The flagship interaction: check 1..N checkpoints across
   the graph, share ONE prompt + seed, generate a strength-1.0 preview per
   checkpoint (reusing the Test-Studio engine) so a LoRA's evolution reads at a
   glance. Only DEPLOYED checkpoints (pill.testable) can be previewed. */

/* Stable key for a checkpoint across the graph — a run's record_id + its step,
   since checkpoints aren't their own node. */
export function checkpointKey(recordId, step) {
  return `${recordId}:${step}`;
}

/* Toggle a checkpoint in/out of the selection set (a Set of checkpointKey). Pure:
   returns a NEW Set, never mutates. */
export function toggleCheckpointSelection(selected, key) {
  const next = new Set(selected || []);
  if (next.has(key)) next.delete(key); else next.add(key);
  return next;
}

/* The {record_id, step} refs the generate endpoint expects, for the TESTABLE
   selected checkpoints only (a non-deployed pick can't be rendered, so it's
   dropped from the request rather than sent to fail). pillByKey maps a
   checkpointKey to its pill descriptor { record_id, step, testable }. */
export function selectedCheckpointRefs(selected, pillByKey) {
  const out = [];
  for (const key of (selected || [])) {
    const p = pillByKey.get(key);
    if (p && p.testable) out.push({ record_id: p.record_id, step: p.step });
  }
  return out;
}

/* Derive the Generate button's state from the current selection, so the UI never
   fires a doomed request and always says WHY it can't (the app's 'needs setup'
   honesty). Returns { count, testableCount, undeployedCount, enabled, hint }:
     - enabled: at least one selected checkpoint is deployable/testable;
     - hint: null when all good, else a short reason (nothing selected, none
       deployed, or "N will be skipped" when the set is mixed). */
export function describePreviewSelection(selected, pillByKey) {
  let count = 0, testableCount = 0;
  for (const key of (selected || [])) {
    const p = pillByKey.get(key);
    if (!p) continue;
    count += 1;
    if (p.testable) testableCount += 1;
  }
  const undeployedCount = count - testableCount;
  let hint = null;
  if (count === 0) hint = 'Check one or more checkpoints to generate previews';
  else if (testableCount === 0) hint = "These checkpoints aren't deployed yet — import a checkpoint for this family first";
  else if (undeployedCount > 0) hint = `${undeployedCount} not-deployed checkpoint${undeployedCount > 1 ? 's' : ''} will be skipped`;
  return { count, testableCount, undeployedCount, enabled: testableCount > 0, hint };
}

/* A pill is already deployed (its LoRA sits in ComfyUI, ready to test/generate)
   when the lineage marked it testable. The graph then shows "✓ Deployed" instead
   of an Import button — nothing to deploy twice. */
export function checkpointDeployed(pill) {
  return !!(pill && pill.testable === true);
}

/* The POST /api/dataset/<id>/train/import body for deploying ONE checkpoint
   straight from a graph pill — the EXACT shape the flat checkpoint list sends
   (base_model / train_type / variant from the run, filename from the pill). A
   cloud node additionally carries cloud_run_id, which makes the server replay the
   family/variant/base stamped at cloud launch (and tag the deployed name ☁ #N so
   the same step from two runs never overwrites). Returns null when there's no
   file to deploy, or a cloud node with no resolved run (nothing importable). */
export function lineageImportPayload(node, pill) {
  if (!node || !pill || !pill.filename) return null;
  const body = {
    filename: pill.filename,
    base_model: node.base_model ?? '',
    train_type: node.train_type,
    variant: node.variant,
  };
  if (node.source === 'cloud') {
    if (node.run_id == null) return null;
    body.cloud_run_id = node.run_id;
  }
  return body;
}

/* Parse the shared seed field: a blank/whitespace value means "let the engine
   pick one" (null); a valid non-negative integer is used as-is; anything else is
   rejected (returns { error }) so a typo never silently reseeds the comparison. */
export function parseSeedInput(raw) {
  const s = (raw == null ? '' : String(raw)).trim();
  if (s === '') return { seed: null };
  if (!/^\d+$/.test(s)) return { error: 'Seed must be a whole number' };
  return { seed: Number(s) };
}
