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

const REPLAY_BLOCKING_RECIPE_STATUSES = new Set(['legacy_incompatible', 'incompatible']);

/** Retry/Continue must not replay a checkpoint whose stamped Z-Image recipe is
 * unsafe. `recipe_status` is the current API field; the fallbacks keep the UI
 * safe if an intermediate backend serialises the diagnostic as `{recipe}` or
 * directly as `status`. */
export function isTrainingRecipeReplayBlocked(run) {
  const status = run?.recipe_status ?? run?.recipe?.status ?? run?.status;
  return REPLAY_BLOCKING_RECIPE_STATUSES.has(status);
}
