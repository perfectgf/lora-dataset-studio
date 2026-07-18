import { trainFamilyLabel } from './checkpointBrowser.js';

/** Trailing filename/tag of a custom base's path or repo id — the segment after
 * the last slash (forward OR back). A local `…\merges\bigLove_zt3.safetensors`
 * shows as `bigLove_zt3.safetensors`, an HF repo `owner/lds-base-hxxxx` as
 * `lds-base-hxxxx`. Keeping only the leaf is what stops a card (or its title)
 * ever surfacing a parent path. */
function baseModelBasename(value) {
  const trimmed = String(value || '').replace(/[\\/]+$/, '');
  const leaf = trimmed.split(/[\\/]/).pop();
  return leaf || trimmed;
}

/** The real base model a run trained on, resolved for a run card, or null when
 * the run doesn't record enough to say (a legacy row → the card degrades to the
 * family badge alone, never an anxious "unknown"). `base_model` is the raw
 * launch selection stamped per run:
 *   - '' → the family's OFFICIAL base → the canonical name the UI already uses
 *          for it, family + variant (e.g. "Z-Image Turbo", "Krea 2 Raw").
 *   - a path / repo id → a CUSTOM base → its trailing filename/tag (`custom`),
 *          so the card can style it apart and truncate it with a full title.
 *   - null / undefined → not recorded → null (graceful degradation).
 * The returned name COMPLETES the family+variant already on the card; it never
 * stands in for the variant. */
export function runBaseModelLabel(run) {
  if (!run) return null;
  const raw = run.base_model;
  if (raw == null) return null;                       // legacy row: not recorded
  const custom = String(raw).trim();
  if (custom) {
    const name = baseModelBasename(custom);
    return { text: name, title: `Custom base: ${name}`, custom: true };
  }
  const family = trainFamilyLabel(run.train_type);
  const variant = trainingRunVariantLabel(run.train_type, run.variant);
  const text = variant ? `${family} ${variant}` : family;
  return { text, title: `Official base: ${text}`, custom: false };
}

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

/** Consecutive Runs-page history rows of the SAME dataset merge under one
 * group header ("Selfie · 3 runs" at a glance). Only ADJACENT rows merge: the
 * list stays reverse-chronological, so a dataset interleaved with another
 * starts a new group instead of silently reordering history. */
export function groupRunsByDataset(runs) {
  const groups = [];
  for (const run of runs || []) {
    const last = groups[groups.length - 1];
    if (last && last.datasetId === run.dataset_id) last.runs.push(run);
    else groups.push({ datasetId: run.dataset_id, runs: [run] });
  }
  return groups;
}

// Backend timestamps are naive UTC (isoformat of utcnow) — pin them to UTC
// before diffing, exactly like the Runs page's timeAgo does.
function parseNaiveUtcMs(iso) {
  if (!iso) return NaN;
  return new Date(/[Z+]/.test(iso) ? iso : `${iso}Z`).getTime();
}

/** Wall-clock duration of a FINISHED run in seconds. Cloud rows carry both
 * timestamps; local registry rows only record the launch → null (the card
 * simply drops the metric). */
export function runDurationSeconds(run) {
  const start = parseNaiveUtcMs(run?.created_at);
  const end = parseNaiveUtcMs(run?.finished_at);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  return (end - start) / 1000;
}

/** '48s', '42m', '1h 05m', '2d 3h' — compact human duration for run cards. */
export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${String(m % 60).padStart(2, '0')}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
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
