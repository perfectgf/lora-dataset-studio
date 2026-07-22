/* ▶ "Continue from here" gate for a ◉ Graph checkpoint pill. JSX-free so
   `node --test` exercises the real rule (RunLineageGraph.jsx imports this exact
   function), and so both mounts of the graph share ONE definition of "this save
   can be resumed" instead of drifting apart.

   Two lanes, chosen by the mount:
     - 'cloud' (DEFAULT — the Runs hub): cloud runs only. That hub's Continue
       flow relaunches a cloud run, so a local run has nothing to offer there.
     - 'any' (the dataset ▸ Checkpoints panel): the same gesture, served by the
       panel's LOCAL resume flow, so a local run's save qualifies too.

   In both lanes the pill must be a save that really exists — the graph never
   offers an action the backend would refuse. */

// A cloud run that failed (e.g. 'pod did not become ready in time') can still
// hold a valid harvested save, hence the download check on the non-'done' path.
const TERMINAL_FAILED = ['error', 'error_pod_kept', 'stopped', 'failed'];

/* canContinueFromCheckpoint(node, pill, { continueSource, hasHandler }) -> bool.
   `hasHandler` is false when the mount passed no onContinueCheckpoint (the
   popover then shows Download/Import only). */
export function canContinueFromCheckpoint(node, pill, opts = {}) {
  const { continueSource = 'cloud', hasHandler = true } = opts;
  if (!hasHandler || !node) return false;
  if (node.source === 'cloud') {
    // The Runs hub's rule, unchanged: a terminal cloud run with a run id, and —
    // when it didn't finish cleanly — a pill that is actually downloadable.
    return node.run_id != null
      && (node.status === 'done'
          || (TERMINAL_FAILED.includes(node.status) && !!pill?.download_url));
  }
  // Local run: the lineage records no terminal status for it (only a currently
  // failed local run is flagged), so the SAVE decides — present and downloadable.
  // The panel wires the handler only when nothing is training and the checkpoint
  // selection matches, which is the in-flight gate.
  return continueSource === 'any' && pill?.present !== false && !!pill?.download_url;
}

/* Which checkpoint the ▶ Continue dialog opens ON. A pill click asks for a
   specific step (`requested`); honour it only when it is a REAL save of the run
   being resumed, otherwise keep the historical default — the newest. Never
   invents a step the run doesn't have. `steps` is ascending. */
export function initialResumeStep(requested, steps) {
  const list = Array.isArray(steps) ? steps : [];
  const latest = list.length ? list[list.length - 1] : 0;
  return requested != null && list.includes(requested) ? requested : latest;
}

/* Which lane the ▶ Continue dialog opens on when it offers the choice.
   `where` is the SOURCE run's lane (a local run defaults to Local, a cloud run to
   Cloud) — but a lane the user cannot use is never pre-selected: if the default
   one is unavailable and the other isn't, open on the one that works. Both
   unavailable → keep `where` (the dialog then shows each lane's reason and the
   submit button stays disabled). `lanes` is null when the mount doesn't offer the
   picker, in which case the answer is simply `where`. */
export function resolveInitialLane(where, lanes) {
  const lane = where === 'cloud' ? 'cloud' : 'local';
  if (!lanes) return lane;
  const other = lane === 'cloud' ? 'local' : 'cloud';
  if (lanes[lane]?.available === false && lanes[other]?.available) return other;
  return lane;
}

export default canContinueFromCheckpoint;
