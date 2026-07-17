/* Pure helpers behind the run-identity chips + Checkpoints↔Runs deep-links.
   Kept framework-free (no JSX) so they are unit-testable with node:test; the
   presentational chips live in components/dataset/RunIdentityBadges.jsx. */

// DOM id of a run's row on the Runs page — the deep-link target for
// "View in Runs ↗". Cloud rows key on the cloud run id, local rows on the
// TrainingRunRecord id. Keep in sync with the row `id=` on CloudRunsPage.
export function runRowDomId(source, id) {
  if (id == null) return null;
  return `run-${source === 'cloud' ? 'cloud' : 'local'}-${id}`;
}

// Resolve a Runs-page row (from /train/cloud/runs) into {source, id}: cloud rows
// expose run_id, local rows expose record_id. null when neither is known (a
// legacy row we cannot address yet — the caller falls back to a bare glyph).
export function runIdentityOf(run) {
  if (!run) return null;
  const cloud = run.source === 'cloud' || run.run_id != null;
  const id = cloud ? run.run_id : run.record_id;
  if (id == null) return null;
  return { source: cloud ? 'cloud' : 'local', id };
}

// Identity of the run behind the LOCAL active set — the provenance record of
// its newest checkpoint (all local checkpoints share one run dir). null before
// the provenance registry has a record (pre-feature datasets).
export function localRunIdentity(checkpoints) {
  const withRun = (checkpoints || []).filter((c) => c.run_id != null);
  if (!withRun.length) return null;
  const c = withRun.reduce((a, b) => ((b.step ?? 0) >= (a.step ?? 0) ? b : a));
  return { source: c.run_source === 'cloud' ? 'cloud' : 'local', id: c.run_id };
}

// Per-run grouped cloud checkpoints for the Checkpoints panel. Prefers the
// server's grouped payload; if only the legacy flat list is present, rebuilds
// groups by run_id so an older server still renders one header per run.
export function cloudGroupsFrom(data) {
  if (Array.isArray(data?.cloud_checkpoint_groups) && data.cloud_checkpoint_groups.length) {
    return data.cloud_checkpoint_groups;
  }
  const flat = Array.isArray(data?.cloud_checkpoints) ? data.cloud_checkpoints : [];
  const byRun = new Map();
  for (const c of flat) {
    const key = c.run_id ?? 'unknown';
    if (!byRun.has(key)) {
      byRun.set(key, {
        run_id: c.run_id ?? null, source: 'cloud', status: c.active ? 'training' : 'done',
        active: !!c.active, version: c.version, variant: c.variant,
        train_type: c.train_type, created_at: c.trained_at, finished_at: null,
        checkpoints: [],
      });
    }
    byRun.get(key).checkpoints.push(c);
  }
  return [...byRun.values()];
}
