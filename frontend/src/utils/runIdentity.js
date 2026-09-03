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
  const cloud = c.run_source === 'cloud';
  return {
    source: cloud ? 'cloud' : 'local',
    // `id` is the ADDRESS (Runs-row anchor key: cloud id for a cloud run) —
    // unchanged. The chip displays recordId: THE run number, stamped on every
    // registry-tagged save; a local tag already is one.
    id: c.run_id,
    recordId: c.record_id ?? (cloud ? null : c.run_id),
    cloudId: cloud ? c.run_id : null,
  };
}

/* ── ONE number for a run, across every lineage surface ──────────────────────
   A run has two identifiers: its provenance RECORD id (the stable key — every
   run has one, local or cloud) and, for a rented-GPU run, the CLOUD run id. The
   lineage surfaces disagreed: the graph card and the tree printed the cloud id
   ("#103") while the inspector printed the record id ("Run #107"), so clicking a
   card opened a panel bearing a different number with nothing tying them.

   The record id wins as THE run number: a local run has no cloud id at all, so
   the cloud id cannot BE a run's identity. The cloud id is not hidden — but it
   is CONTEXT, not identity: it lives in tooltips (runIdentityLabel) and in the
   run inspector, never printed beside the number on cards, chips or tree rows.
   (It used to be printed as a "· cloud #103" secondary; two numbers on every
   card read as noise, and the checkpoints chip printed the cloud id ALONE —
   the same run wore different numbers on different screens, and ⚙ Details
   chased the wrong one. Maintainer, 2026-08-29: one number, everywhere.)

   Display only: no stored key changes. What was keyed by record id (checkpoint
   selections, notes) or by cloud run id (Runs deep links, checkpoint replays)
   still addresses exactly what it did before. */

// The run's own number as text: `#107`. Unknown id → `#?`, never `#undefined`.
export function runNumber(node) {
  const id = node?.record_id;
  // A NEGATIVE id is the video lane's local run: it has no row anywhere (the
  // folder is the run), so its node borrows the dataset id negated to be
  // keyable. That is an address, not a number to print.
  if (typeof id === 'number' && id < 0) return 'local';
  return `#${id == null ? '?' : id}`;
}

// The cloud run id when this run trained in the cloud and has one, else null —
// a local run answers null, which is exactly why it cannot be the identity.
export function cloudNumber(node) {
  if (!node || node.source !== 'cloud') return null;
  return node.run_id == null ? null : node.run_id;
}

// Full identity for a heading or tooltip: `Run #107 · cloud #103`, or plain
// `Run #107` when no cloud run is behind it.
export function runIdentityLabel(node) {
  const cloud = cloudNumber(node);
  return `Run ${runNumber(node)}${cloud == null ? '' : ` · cloud #${cloud}`}`;
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
