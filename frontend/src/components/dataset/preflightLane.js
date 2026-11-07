/* Which lane a pre-launch preflight is asked for, and the URL that asks it.
 *
 * The preflight mixes two kinds of row: properties of the DATASET (image count,
 * captions, identity leaks, near-duplicates, untriaged images, face masks) and
 * reads of THIS MACHINE (GPU memory, torch build). Only the first kind means
 * anything when the job runs on a rented pod, so the lane rides in the query
 * string and the server drops the machine-scope rows for `cloud`.
 *
 * Plain .js on purpose: node --test does not parse JSX, so the logic worth
 * testing lives outside TrainingPanel.jsx.
 */

import { normalizeTrainingMode } from '../../utils/trainingMode.js';

/** 'cloud' or 'local' — anything unrecognised falls back to 'local', which is
 * the server's default and the historical (unfiltered) payload. */
export function normalizeLane(lane) {
  return lane === 'cloud' ? 'cloud' : 'local';
}

/** The ▶ Continue dialog resolves to a payload carrying its own lane; the plain
 * Train button has none. Same normalisation either way. */
export function laneOfPayload(payload) {
  return normalizeLane(payload && payload.lane);
}

/** GET url for the preflight. `lane` is only sent for the cloud lane: a request
 * with no `lane` must stay byte-for-byte the historical one, so nothing that
 * already calls this route (the workspace readiness badge) changes behaviour.
 *
 * `masked` follows the same rule for the opposite reason. Whether the run wants
 * person masks is a localStorage preference the server cannot read, so a launch
 * states it and gets the "rembg is missing, this run trains unmasked" row; a
 * caller that has no opinion (the readiness badge) omits it and gets no row,
 * rather than a warning about a mask nobody asked for. */
export function preflightUrl(datasetId, {
  trainType, variant, baseModel, trainingMode, lane, masked,
} = {}) {
  const qs = [];
  if (trainType) qs.push(`train_type=${encodeURIComponent(trainType)}`);
  if (variant) qs.push(`variant=${encodeURIComponent(variant)}`);
  // Empty is meaningful: it selects the official base instead of allowing a
  // previously persisted custom base to leak into this preflight.
  if (baseModel !== undefined) qs.push(`base_model=${encodeURIComponent(baseModel ?? '')}`);
  if (trainingMode) qs.push(`training_mode=${encodeURIComponent(normalizeTrainingMode(trainingMode))}`);
  if (normalizeLane(lane) === 'cloud') qs.push('lane=cloud');
  if (masked !== undefined && masked !== null) qs.push(`masked=${masked ? '1' : '0'}`);
  return `/api/dataset/${datasetId}/train/preflight${qs.length ? `?${qs.join('&')}` : ''}`;
}

/** What the readiness badge above the Training panel asks, and how it reads a
 * clean verdict.
 *
 * A full-model recipe is cloud-only, so its LOCAL-lane preflight is always one
 * blocker long ("full_transformer training is cloud-only") — the badge said
 * "Not ready" over a dataset that was ready to launch. In full-model mode it
 * asks the cloud lane, with no variant and no base: the server then reads the
 * SAVED selection, where a variant from a dataset payload loaded earlier could
 * be stale. The launch still checks the picked base itself. Every other mode keeps
 * the historical request byte for byte, and `endpoint` (the video workspace's)
 * wins over both. */
export function readinessRequest({ datasetId, trainType, variant, trainingMode, endpoint = null } = {}) {
  if (endpoint) return { url: endpoint, readyLabel: 'Ready to train', lane: null };
  if (normalizeTrainingMode(trainingMode) === 'full_transformer') {
    return {
      url: preflightUrl(datasetId, { trainType, trainingMode, lane: 'cloud' }),
      readyLabel: 'Ready to launch in the cloud',
      lane: 'cloud',
    };
  }
  const params = new URLSearchParams();
  if (trainType) params.set('train_type', trainType);
  if (variant) params.set('variant', variant);
  const qs = params.size ? `?${params.toString()}` : '';
  return { url: `/api/dataset/${datasetId}/train/preflight${qs}`, readyLabel: 'Ready to train', lane: 'local' };
}

/** A report answered for the OTHER lane (the dataset has just switched mode and
 * its new preflight is still running) is not this one's verdict. The server
 * echoes its lane; a report without one is the historical local payload. */
export function reportFitsLane(report, lane) {
  return !lane || (report?.lane === 'cloud' ? 'cloud' : 'local') === lane;
}

/** Rows the modal must never show on a cloud lane, kept here as the client-side
 * mirror of the server filter — used by the test that guards the whole point of
 * this feature: a warning that fires wrongly teaches people to ignore all of
 * them. Belt and braces; the server remains authoritative. */
export const MACHINE_SCOPE_CHECKS = ['vram', 'torch_arch'];

/** Defensive filter over a preflight payload's `checks`, for the lane given. */
export function checksForLane(checks, lane) {
  const list = Array.isArray(checks) ? checks : [];
  if (normalizeLane(lane) !== 'cloud') return list;
  return list.filter((c) => c && c.scope !== 'machine'
    && !MACHINE_SCOPE_CHECKS.includes(c.id));
}
