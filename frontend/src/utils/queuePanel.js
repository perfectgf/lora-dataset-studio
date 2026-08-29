/** ⏳ The generation queue dock — everything about it that is not JSX.
 *
 * The queue itself is `backend/app/services/queue_view.py`; this is the wording
 * and the arithmetic the panel puts on screen, kept out of the component so it
 * can be tested for what it SAYS rather than for how it renders.
 *
 * The panel exists because the queue never had a face (GitHub #44). Every
 * surface showed its own slice — the tiles a dataset waits on, the cells of a
 * Studio run — so work queued from one screen was invisible from every other,
 * and "is it stuck, or is something else in front of it?" had no answer.
 */

/** Nothing to show is not the same as an empty list: don't dock over the app. */
export function hasQueue(listing) {
  return Array.isArray(listing?.jobs) && listing.jobs.length > 0;
}

/**
 * The collapsed pill's line. It names what is HAPPENING first, because that is
 * the question ("what is the GPU doing?"); the wait is the second half.
 */
export function summarize(listing) {
  if (!hasQueue(listing)) return '';
  const { generating = 0, queued = 0, stalled = 0 } = listing;
  const parts = [];
  if (generating) parts.push(`${generating} generating`);
  if (queued) parts.push(`${queued} queued`);
  if (stalled) parts.push(`${stalled} paused`);
  // A live row that is none of the three (an unknown future status) still has
  // to be counted, or the pill would claim an empty queue over a full one.
  if (!parts.length) parts.push(`${listing.jobs.length} in the queue`);
  return parts.join(' · ');
}

/**
 * Why the whole queue is standing still, or null when it is not.
 *
 * Training and the vision pass hold the GPU OUTSIDE this queue — the worker
 * claims nothing while either runs. A dock that counted a line which never
 * advanced, and said nothing about why, would rebuild the very confusion it is
 * here to remove, one level up.
 */
export function pausedReason(listing) {
  const reason = (listing?.paused_reason || '').trim();
  return reason || null;
}

/**
 * The ANSWER the server offers for a queue standing still, or null when waiting
 * is the only answer there is.
 *
 * Most holds end by themselves — a training run finishes, Ollama drops an idle
 * model — and those get a sentence and nothing else. One does not: a model
 * another app (or another LDS instance) holds on the shared card may never be
 * handed back, and some runners never unload at all. That hold used to be an
 * open wait with no way out but quitting the other program.
 *
 * Validated rather than trusted: an older backend sends no action at all, and a
 * future kind this dock does not know must not become a button that does
 * nothing. A button with no words on it is worse than no button.
 */
export function pausedAction(listing) {
  const action = listing?.paused_action;
  if (!action || action.kind !== 'share_gpu') return null;
  const label = (action.label || '').trim();
  const confirm = (action.confirm || '').trim();
  if (!label || !confirm) return null;
  return {
    kind: action.kind,
    label,
    confirm,
    models: Array.isArray(action.models) ? action.models : [],
  };
}

/** "Upscale & improve · Klein" — the engine only when there is one to name. */
export function jobLabel(job) {
  return job?.engine ? `${job.title} · ${job.engine}` : (job?.title || '');
}

/**
 * Where a job came from, and on what. "📁 Dataset · Faces" beats "📁 Dataset"
 * for the only question a queue view has to answer when two datasets are both
 * feeding it.
 */
export function jobOrigin(job) {
  if (!job) return '';
  return job.dataset_name ? `${job.surface} · ${job.dataset_name}` : (job.surface || '');
}

/** Compact elapsed time. `now` is injected so this is testable and pure. */
export function elapsedLabel(iso, now = Date.now()) {
  if (!iso) return '';
  const started = Date.parse(iso);
  if (Number.isNaN(started)) return '';
  const seconds = Math.max(0, Math.round((now - started) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min`;
}

/**
 * The one-line explanation under a row, or null when the row speaks for itself.
 *
 * A job the panel may not cancel must say WHY and WHERE its real Stop is —
 * a disabled button with no sentence is the failure mode this whole change is
 * about.
 */
export function rowNote(job) {
  if (!job) return null;
  // `status` collapses two different states onto the word "stalled" so the row
  // renders the same; `raw_status` is what tells them apart, and the server
  // emits it for exactly this. Reading `status` here announced "ComfyUI stopped
  // answering" about a job the user had just asked to cancel.
  if (job.raw_status === 'cancel_requested') {
    return 'Cancelling — waiting for ComfyUI to let go of it.';
  }
  if (job.status === 'stalled') {
    return 'Paused — ComfyUI stopped answering. Resolve it from the recovery banner.';
  }
  if (!job.cancellable && job.blocked_by) {
    return `Owned by ${job.blocked_by} — stop it from there.`;
  }
  return null;
}

/**
 * Why the ⤒ button is off, in words. Two different reasons, and collapsing them
 * into one grey button was not an option: "already running" is good news.
 */
export function promoteBlockedReason(job) {
  if (!job) return null;
  if (job.status !== 'queued') return 'Already running — there is nothing left to re-order.';
  if (job.position === 1) return 'Already next in line.';
  return null;
}
