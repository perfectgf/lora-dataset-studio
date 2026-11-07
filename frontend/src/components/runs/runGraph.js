/*
 * Which Runs-page rows offer the ◉ Graph (🌳 Lineage once a run has a parent or a
 * branch). A cloud row says whether it holds saves (checkpoint_ready). A local row
 * is never scanned on this list, which polls every few seconds, so every local run
 * offers its graph: the graph lists the saves still on disk when it opens. Before,
 * a local run without a parent or a branch had no way to its checkpoints here.
 *
 * Kept in plain .js (no JSX) so `node --test` can exercise the decision itself.
 */
export function runOpensGraph(run, { fullModel = false } = {}) {
  if (fullModel || run?.record_id == null) return false;
  return Boolean(run.lineage || run.checkpoint_ready || run.source === 'local');
}
