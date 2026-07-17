/** A local run can be stopped from the global Runs hub only while the aggregate
 * status identifies both its dataset and its opaque launch token. The token
 * prevents a stale card from killing a newer run of the same dataset. */
export function canStopLocalRun(localActive) {
  return Boolean(localActive?.in_progress
    && localActive?.current?.dataset_id !== null
    && localActive?.current?.dataset_id !== undefined
    && localActive?.current?.run_token);
}

export function trainingRunVariantLabel(trainType, variant) {
  if (!variant) return null;
  if (variant === 'base') return trainType === 'krea' ? 'Raw' : 'Base';
  if (variant === 'deturbo') return 'De-Turbo';
  if (variant === 'turbo') return 'Turbo';
  return variant.toUpperCase();
}

/** A Retry click posts to a different endpoint depending on where the run ran:
 * a cloud run replays its pod launch params by `run_id`; a LOCAL run replays the
 * stamped provenance record by `record_id`. Returns null when the run can't be
 * addressed (no id) so the caller can no-op. */
export function retryRequest(run) {
  if (run?.source === 'local') {
    return run.record_id != null
      ? { url: '/api/dataset/train/retry', body: { record_id: run.record_id } }
      : null;
  }
  return run?.run_id != null
    ? { url: '/api/dataset/train/cloud/retry', body: { run_id: run.run_id } }
    : null;
}

/** Stable per-row key for the "retry in flight" map — cloud run_ids and local
 * record_ids live in separate namespaces, so they are prefixed to never collide. */
export function runRetryKey(run) {
  return run?.source === 'local' ? `l${run?.record_id}` : `c${run?.run_id}`;
}

const REPLAY_BLOCKING_RECIPE_STATUSES = new Set(['legacy_incompatible', 'incompatible']);

/** Retry/Continue must not replay a checkpoint whose stamped Z-Image recipe is
 * unsafe. `recipe_status` is the current API field; the fallbacks keep the UI
 * safe if an intermediate backend serialises the diagnostic as `{recipe}` or
 * directly as `status`. */
export function isTrainingRecipeReplayBlocked(run) {
  const status = run?.recipe_status ?? run?.recipe?.status ?? run?.status;
  return REPLAY_BLOCKING_RECIPE_STATUSES.has(status);
}
