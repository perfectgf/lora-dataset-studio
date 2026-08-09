/* Wording and arithmetic for the concept face-mask preview's progress.

   PURE (no React, no fetch) so the honest-message logic is testable on its own —
   node --test cannot parse JSX, so anything worth asserting lives here.

   The point of naming phases: the per-image counter alone LIES about the wait.
   InsightFace imports onnxruntime and prepares antelopev2 before image 1, which
   is tens of seconds, and on a fresh install it first downloads ~350 MB. A bar
   sitting at 0/40 through all of that is read as "it crashed" at exactly the
   moment nothing at all was wrong. So each stage says what it is, and only the
   stages that really cost are named. */

export const FACE_MASK_PHASES = ['starting', 'downloading', 'loading', 'detecting'];

const PHASE_LABELS = {
  starting: 'Starting the face detector…',
  // First run only, and worth its own sentence: several hundred megabytes over
  // the network is not a slow detection, and telling the user it is would earn
  // exactly the "is this broken?" it is meant to answer.
  downloading: 'Downloading the face-detection model (~350 MB, first run only)…',
  loading: 'Loading the face-detection model…',
  detecting: 'Analyzing images…',
};

/** The one line under the button. Never a bare spinner: it always names either a
 *  stage, a count, or a failure. */
export function previewStatusLabel(job) {
  if (!job) return '';
  // Before `error`, because a pass cut short by its own time budget is not a
  // failure and must not be worded as one: it was working, it ran out of the
  // time it was given, and everything it analysed is kept.
  if (job.interrupted) return 'Interrupted before the end.';
  if (job.error) return job.error;
  if (job.stopped) return 'Stopped.';
  if (job.finished) return 'Done.';
  // Between the click and the child actually winding up there is a real wait: it
  // only looks at the stop request between two images, and during the model load
  // it cannot look at all. Saying "Stopped" here would be the one lie the user
  // can catch — the counter is still moving in front of them.
  if (job.stopping) return 'Stopping — finishing the current image…';
  if (job.phase === 'detecting' && job.total > 0) {
    return `Analyzing image ${Math.min(job.done + 1, job.total)} of ${job.total}…`;
  }
  return PHASE_LABELS[job.phase] || 'Working…';
}

/** {done, total} once there is a real count to show, else null = indeterminate.
 *  Returning null rather than 0 matters: a determinate bar pinned at 0% for the
 *  whole model load is the misleading state this replaces. */
export function previewProgressValue(job) {
  if (!job || job.error || job.finished) return null;
  if (job.phase !== 'detecting' || !(job.total > 0)) return null;
  return { done: Math.max(0, Math.min(job.done || 0, job.total)), total: job.total };
}

/** Bar width, 0-100. Only meaningful when previewProgressValue is non-null. */
export function previewPercent(job) {
  const v = previewProgressValue(job);
  return v ? Math.round((v.done / v.total) * 100) : 0;
}

/** True while a pass is in flight — drives both the disabled button and the poll. */
export function previewRunning(job) {
  return Boolean(job) && !job.finished && !job.error;
}

/** How many images this pass has already banked — the count a stop would keep. */
function analyzed(job) {
  if (!job || job.phase !== 'detecting') return 0;
  return Math.max(0, Math.min(job.done || 0, job.total || 0));
}

/** What pressing Stop COSTS, said at the moment it is offered — never a generic
 *  "are you sure?".
 *
 *  Two different answers, and which one applies changes while you watch:
 *
 *  - during the load (and the first-run download) nothing has been analyzed yet,
 *    so the only thing given up is the load itself;
 *  - once images are being analyzed, every face already found is kept and only
 *    the load is re-paid on the way back.
 *
 *  The load is re-paid EITHER WAY, because the detector runs in a subprocess that
 *  exits with the pass — there is no warm model left behind. That is stated, not
 *  glossed: a Stop that quietly made the retry cost the whole run again would be
 *  a button that looks like it saves time and spends it. */
export function previewStopCost(job) {
  if (!previewRunning(job)) return '';
  const done = analyzed(job);
  if (!done) {
    return 'Nothing has been analyzed yet — stopping gives up the detector load only, '
      + 'and starting again pays it over.';
  }
  return `The ${done} image${done > 1 ? 's' : ''} already analyzed ${done > 1 ? 'are' : 'is'} kept. `
    + 'Starting again re-loads the detector, then carries on from where it stopped.';
}

/** The Stop button's own label. */
export function previewStopLabel(job) {
  return job && job.stopping ? 'Stopping…' : 'Stop';
}

/** The start button's label, so a resume ANNOUNCES its credit instead of looking
 *  like a fresh pass over the same images. `resume` is the server's
 *  {done, total} for the current kept set, or null when there is nothing banked. */
export function previewStartLabel(resume, hasPreview) {
  if (resume && resume.done > 0 && resume.total > resume.done) {
    return `▶ Resume — ${resume.done} of ${resume.total} already analyzed`;
  }
  if (resume && resume.done > 0) return '▶ Resume — finishing up';
  return hasPreview ? 'Refresh preview' : '👁 Preview the mask';
}

/** The line shown after a pass ended early, or ''. It exists to make the bargain
 *  visible AFTER the fact too: a user who stopped and sees nothing has no reason
 *  to believe the work survived.
 *
 *  Two ways to end early, and they need different sentences. A STOP was asked
 *  for, so the notice only has to account for what was kept. An INTERRUPTION was
 *  not: the pass ran out of the time budget (or the child died), so the notice
 *  also has to say what happened and that starting again is the way forward —
 *  otherwise the panel just goes quiet on a run the user was watching. */
export function previewStoppedNotice(job, resume) {
  if (job && job.interrupted) {
    const kept = resume && resume.done > 0
      ? `${resume.done} of ${resume.total} images are kept — starting again continues from there.`
      : 'Nothing had been analyzed yet, so there was nothing to keep.';
    return `${job.note || 'The pass was interrupted before the end.'} ${kept}`;
  }
  if (!job || !job.stopped) return '';
  if (resume && resume.done > 0 && resume.total > resume.done) {
    return `Stopped. ${resume.done} of ${resume.total} images are kept — `
      + 'starting again continues from there rather than beginning over.';
  }
  if (resume && resume.done > 0) return 'Stopped. Every image was analyzed — start again to draw the preview.';
  return 'Stopped before any image was analyzed, so there was nothing to keep.';
}

/** What went wrong, or ''. A finished job with an error is the failure case that
 *  used to look like an endless wait. */
export function previewError(job) {
  return (job && job.error) || '';
}
