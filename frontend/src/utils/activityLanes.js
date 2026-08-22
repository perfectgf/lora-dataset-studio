// Which live dataset activity blocks which new action.
//
// `dataset_activity` (backend) publishes ONE entry per dataset describing the
// batch currently running on it. The hook used to turn that entry into a single
// boolean — `busyLive = busy || !!activity` — and hand it to every action button
// in the workspace. So starting a ✨ Upscale & improve batch, or merely retrying
// ONE tile (the synchronous regenerate publishes a 'generate' activity of
// total=1), greyed out Generate, the watermark passes, captioning, everything,
// for as long as the batch ran. That is GitHub #44, reported by charlesangus:
// "kicking off any generation task should allow more generation tasks to be
// queued up; the user should not have to wait for one to finish before kicking
// off the next".
//
// The blanket was never load-bearing. `backend/app/job_queue.py` is a FIFO
// worker over `image_generation_queue` and `lora_test_studio.gpu_busy_reason`
// says so in as many words: "the queue itself serializes normal generations, so
// no further locking is needed". Work piled on top does not collide — it lines
// up, which is exactly what was asked for.
//
// So the rule is about WHICH RESOURCE an activity holds, in two lanes:
//
//   * QUEUE — work that becomes rows in the serialized image queue. Several of
//     these may be in flight at once; the worker takes them one at a time.
//   * EXCLUSIVE — a pass that walks the dataset's rows and rewrites them
//     (captioning, watermarks, face analysis, exports, backups), or that holds
//     the GPU vision window and therefore pauses ComfyUI. Two of those overlap
//     badly, and one of those overlapping a generation is what the vision fence
//     exists to prevent.
//
// Only QUEUE-on-QUEUE overlaps. Everything else keeps the behaviour it had.
//
// This is the same shape as `components/dataset/scraperState.js`, whose gates
// already carved out targeted exemptions by kind. `isDatasetImportBlocked` asks
// this very question and now reads `laneOf` rather than naming 'generate' by
// hand, so the lane list lives in ONE place. `isStopGenerationBlocked` stays on
// its own list on purpose: it answers a different question — which activities
// the Stop button EXISTS to end — and 'improve' is stoppable without being
// blocking.

// Activities whose work is a row in the serialized image queue.
export const QUEUE_KINDS = ['generate', 'improve', 'edit_reference'];

// Queue-lane kinds the BACKEND still refuses to run twice on one dataset, so
// the button that starts them must stay disabled while one is live even though
// the lane is otherwise open:
//   * 'improve'        -> start_bulk_improve: "an improvement batch is already
//                         running on this dataset" (409).
//   * 'edit_reference' -> a pending Before/After is a decision the user still
//                         owes; a second edit would silently replace it.
export const SELF_EXCLUSIVE_KINDS = ['improve', 'edit_reference'];

export function laneOf(kind) {
  return QUEUE_KINDS.includes(kind) ? 'queue' : 'exclusive';
}

// The two engines that render somewhere else. Everything not on this list is
// treated as LOCAL, including an unknown or legacy value: failing safe means
// assuming the GPU is taken.
const REMOTE_ENGINES = ['nanobanana', 'chatgpt'];

/**
 * Is this activity occupying the local ComfyUI?
 *
 * Separate question from `activityBlocks`, and it must stay separate. Queued
 * work does not stop you queueing more — but it DOES stop anything that wants
 * the exclusive GPU vision window, and the import dropzone's auto head-crop is
 * exactly that. Widening the import gate (`isDatasetImportBlocked`) without
 * widening this one shipped an import that opened, ran, and came back 503
 * "GPU busy" on the crop: the door was unlocked onto a wall.
 */
export function holdsLocalGpu(activity) {
  const kind = activity?.kind || null;
  if (!kind) return false;
  if (laneOf(kind) !== 'queue') return false;   // exclusive passes own their own gate
  return !REMOTE_ENGINES.includes(String(activity?.engine || '').toLowerCase());
}

/**
 * Does the live `activity` block starting `actionKind`?
 *
 * `activity` is the dataset payload's `activity` object (or null). `actionKind`
 * is the kind the user is about to start — one of `dataset_activity.KINDS`.
 */
export function activityBlocks(activity, actionKind) {
  const liveKind = activity?.kind || null;
  if (!liveKind) return false;
  // An exclusive pass is running, or an exclusive pass is being started on top
  // of queued work: both keep the old blanket.
  if (laneOf(liveKind) !== 'queue' || laneOf(actionKind) !== 'queue') return true;
  // Queue on queue — allowed, unless the backend refuses this exact repeat.
  return liveKind === actionKind && SELF_EXCLUSIVE_KINDS.includes(actionKind);
}
