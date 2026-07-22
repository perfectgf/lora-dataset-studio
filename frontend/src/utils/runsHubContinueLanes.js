/* ▶ Continue — which lanes the RUNS HUB can offer for one finished run.

   JSX-free so `node --test` exercises the real rule (CloudRunsPage.jsx imports
   this exact function), like lineageContinue.js does for the graph's gate.

   A checkpoint is just a file: a cloud run's epoch is mirrored into this
   machine's ai-toolkit run dir, so it can be finished HERE instead of on another
   rented pod. Both lanes are always returned — an unusable one carries its
   reason and the dialog shows it disabled, never hidden.

   The two guards are deliberately NOT symmetrical, and that is the whole
   difference with the dataset panel:
     - local training is single-flight for the WHOLE machine, so `localActive`
       closes the lane whatever dataset that run belongs to;
     - the cloud guard is per (dataset, family), and this hub lists runs of MANY
       datasets — it must be evaluated against the TARGET run's dataset, never
       page-wide. */

export function runsHubContinueLanes(run, opts = {}) {
  if (!run) return null;
  const {
    aitoolkitValid,          // caps.aitoolkit?.valid — undefined while caps load
    localActive = null,      // the hub payload's `local_active` (any dataset)
    actives = [],
    configured = false,      // a vast.ai API key is set
    limit = 1,               // max concurrent cloud runs
    familyLabel = (f) => f || 'LoRA',
  } = opts;

  const localReason =
    aitoolkitValid === false
      ? 'Local training needs ai-toolkit — set it up in Settings, or continue in the cloud.'
    // A legacy row with no dataset can't address a local run dir — say so
    // instead of firing a request that would 404.
    : run.dataset_id == null
      ? 'This run’s dataset is unknown, so it can only be continued in the cloud.'
    : localActive
      ? 'A training is already running on this machine — continue in the cloud, or wait for it to finish.'
    : null;

  // A run without train_type (older payload) matches any family, exactly like
  // the dataset panel's own cloudActiveHere lookup.
  const cloudActiveThere = actives.find((a) => a.dataset_id === run.dataset_id
    && (!a.train_type || a.train_type === run.train_type));
  const cloudReason =
    !configured
      ? 'Cloud training needs a vast.ai API key — add it in Settings.'
    : cloudActiveThere
      ? `A ${familyLabel(run.train_type)} cloud run is already active on this dataset`
    : actives.length >= limit
      ? `Cloud run limit reached (${actives.length}/${limit}) — stop one or raise the limit in Settings`
    : null;

  return {
    local: localReason ? { available: false, reason: localReason } : { available: true },
    cloud: cloudReason ? { available: false, reason: cloudReason } : { available: true },
  };
}

export default runsHubContinueLanes;
