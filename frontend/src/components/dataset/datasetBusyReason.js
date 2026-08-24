/* WHY A DATASET WRITE IS REFUSED RIGHT NOW — one sentence, not a grey button.
 *
 * THE COMPLAINT THIS ANSWERS. "During an image generation every action around
 * the dataset is blocked — opening the lightbox, deleting a photo…". One
 * `busy` flag guarded the WHOLE tile, so a pass that touches pixels also
 * switched off inspecting and ticking, which conflict with nothing. Reads are
 * unblocked in the components; this file is the other half — the writes that
 * legitimately stay refused now SAY SO instead of going quietly grey.
 *
 * NOTHING NEW IS COMPOSED HERE. The Bank solved this exact problem first
 * ("A Bank being built is inspectable, never mutable or deletable") and
 * `bank/bankPassRun.js::busyLine` already builds the sentence: blocker, then
 * progress, then time left, then the running phase. Only the VOCABULARY was
 * bank-specific, so that function took a `labels`/`subject` pair and this file
 * supplies the Dataset's. Both surfaces therefore gain a future improvement to
 * the sentence (the ETA clause landed that way) at the same time, and cannot
 * drift into two dialects of the same refusal.
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */
import { busyRefusal } from '../bank/bankPassRun.js';

/* Dataset activity kind (as `services/dataset_activity.py` stores it and the
   dataset payload publishes it in `activity.kind`) → how a human names that
   pass. Same emoji + words as the button that starts it and as the amber
   progress banner, so "🧽 Watermark scan is running" points at something
   visible on screen. An unknown kind falls back to "Another pass" rather than
   leaking an internal identifier. */
const DATASET_JOB_LABELS = {
  watermark_detect: '🧽 Watermark scan',
  watermark_clean: '🧽 Watermark cleaning',
  caption: '🏷️ Captioning',
  recaption: '🏷️ Re-captioning',
  analyze_faces: '🎭 Face analysis',
  classify: '📐 Framing classification',
  generate: '⚡ Variation generation',
  improve: '✨ Improvement batch',
  edit_reference: '🖌 Reference edit',
  bank_export: '🏦 Copy into a Bank',
  bank_import: '🏦 Copy from a Bank',
  training_export: '🎓 Training export',
  backup: '💾 Portable backup',
};

/* Passes the user can actually STOP from this screen (backend
   `STOPPABLE_KINDS`, plus 'generate' through ⏹ Stop generation). Only those
   get told about a Stop button: pointing at a control that is not on screen is
   advice you cannot act on, which is the failure this whole file exists to
   fix. */
const STOPPABLE = new Set(['caption', 'recaption', 'watermark_detect', 'improve', 'generate']);

const STOP_HINT = 'Wait for it to finish, or press ⏹ Stop in the banner at the top of the workspace.';
const WAIT_HINT = 'Wait for it to finish — inspecting and selecting stay available meanwhile.';

/* The one thing the workspace's amber progress banner does NOT say: what the
   pass is holding, and what it is not. Kept OUT of `datasetBusyReason` on
   purpose — that sentence is read on a button, one at a time, and the grid
   sits directly under the progress banner where repeating the pass name and
   its counter would print the same words twice (the mistake `withDetail:false`
   exists to avoid in the Bank). */
export const READS_STAY_OPEN =
  'Edits, captions and deletes wait for the pass above to finish — inspecting an image and ticking a selection still work.';

/**
 * The full sentence a refused write shows: blocker + progress + time left +
 * what to do about it. `activity` may be null — a locally-tracked action with
 * no server snapshot still gets an honest "Another pass is running on this
 * dataset", which beats a silent grey button.
 * @param {object|null} activity the dataset payload's `activity` snapshot
 * @returns {string}
 */
export function datasetBusyReason(activity) {
  const kind = activity?.kind;
  return busyRefusal({
    activity,
    labels: DATASET_JOB_LABELS,
    subject: 'this dataset',
    stopHint: STOPPABLE.has(kind) ? STOP_HINT : WAIT_HINT,
  });
}
