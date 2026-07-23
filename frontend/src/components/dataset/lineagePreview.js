/* Pure helpers for the Lab's inline generation (RunLineageGraph.jsx). JSX-free so
   `node --test` exercises the selection/enable logic directly; the graph imports
   these same functions. The flagship interaction: check 1..N checkpoints across
   the graph, share ONE prompt + seed, generate a strength-1.0 preview per
   checkpoint (reusing the Test-Studio engine) so a LoRA's evolution reads at a
   glance. Only DEPLOYED checkpoints (pill.testable) can be previewed. */

/* Stable key for a checkpoint across the graph — a run's record_id + its step,
   since checkpoints aren't their own node. */
export function checkpointKey(recordId, step) {
  return `${recordId}:${step}`;
}

/* Toggle a checkpoint in/out of the selection set (a Set of checkpointKey). Pure:
   returns a NEW Set, never mutates. */
export function toggleCheckpointSelection(selected, key) {
  const next = new Set(selected || []);
  if (next.has(key)) next.delete(key); else next.add(key);
  return next;
}

/* The {record_id, step} refs the generate endpoint expects, for the TESTABLE
   selected checkpoints only (a non-deployed pick can't be rendered, so it's
   dropped from the request rather than sent to fail). pillByKey maps a
   checkpointKey to its pill descriptor { record_id, step, testable }. */
export function selectedCheckpointRefs(selected, pillByKey) {
  const out = [];
  for (const key of (selected || [])) {
    const p = pillByKey.get(key);
    if (p && p.testable) out.push({ record_id: p.record_id, step: p.step });
  }
  return out;
}

/* Derive the Generate button's state from the current selection, so the UI never
   fires a doomed request and always says WHY it can't (the app's 'needs setup'
   honesty). Returns { count, testableCount, undeployedCount, enabled, hint }:
     - enabled: at least one selected checkpoint is deployable/testable;
     - hint: null when all good, else a short reason (nothing selected, none
       deployed, or "N will be skipped" when the set is mixed). */
export function describePreviewSelection(selected, pillByKey) {
  let count = 0, testableCount = 0;
  for (const key of (selected || [])) {
    const p = pillByKey.get(key);
    if (!p) continue;
    count += 1;
    if (p.testable) testableCount += 1;
  }
  const undeployedCount = count - testableCount;
  let hint = null;
  if (count === 0) hint = 'Check one or more checkpoints to generate previews';
  else if (testableCount === 0) hint = "These checkpoints aren't deployed yet — import a checkpoint for this family first";
  else if (undeployedCount > 0) hint = `${undeployedCount} not-deployed checkpoint${undeployedCount > 1 ? 's' : ''} will be skipped`;
  return { count, testableCount, undeployedCount, enabled: testableCount > 0, hint };
}

/* A pill is already deployed (its LoRA sits in ComfyUI, ready to test/generate)
   when the lineage marked it testable. The graph then shows "✓ Deployed" instead
   of an Import button — nothing to deploy twice. */
export function checkpointDeployed(pill) {
  return !!(pill && pill.testable === true);
}

/* The POST /api/dataset/<id>/train/import body for deploying ONE checkpoint
   straight from a graph pill — the EXACT shape the flat checkpoint list sends
   (base_model / train_type / variant from the run, filename from the pill). A
   cloud node additionally carries cloud_run_id, which makes the server replay the
   family/variant/base stamped at cloud launch (and tag the deployed name ☁ #N so
   the same step from two runs never overwrites). Returns null when there's no
   file to deploy, or a cloud node with no resolved run (nothing importable). */
export function lineageImportPayload(node, pill) {
  if (!node || !pill || !pill.filename) return null;
  const body = {
    filename: pill.filename,
    base_model: node.base_model ?? '',
    train_type: node.train_type,
    variant: node.variant,
  };
  if (node.source === 'cloud') {
    if (node.run_id == null) return null;
    body.cloud_run_id = node.run_id;
  }
  return body;
}

/* The POST /api/dataset/<id>/train/run-checkpoint/delete body for trashing ONE
   save straight from a graph pill. Deliberately the SAME shape as the import
   payload (that's what the flat checkpoint list sends), because the server
   resolves the file the same way: a cloud pill by cloud_run_id + filename, a
   local pill by base_model/train_type/variant + filename. Returns null when
   there's no file to delete, or a cloud node whose run isn't resolved — sending
   it without cloud_run_id would make the server look for a LOCAL file of that
   name, i.e. delete the wrong thing (or nothing).

   ⚠ This route trashes the RUN's save. The LoRA already imported into ComfyUI is
   a separate file removed by a separate route (train/checkpoint/delete). */
export function lineageDeletePayload(node, pill) {
  if (!node || !pill || !pill.filename) return null;
  const body = {
    filename: pill.filename,
    base_model: node.base_model ?? '',
    train_type: node.train_type,
    variant: node.variant,
  };
  if (node.source === 'cloud') {
    if (node.run_id == null) return null;
    body.cloud_run_id = node.run_id;
  }
  return body;
}

/* A cloud run still going: its pod keeps syncing epochs down, so trashing one of
   its saves would be undone (or race the sync). The flat checkpoint list hides
   its 🗑 for exactly these runs — mirrored here so both views agree. */
const CLOUD_ACTIVE_STATES = ['preparing', 'provisioning', 'uploading', 'training',
  'downloading', 'terminating'];

/* The ONE delete action a pill offers, and WHICH of the two files it aims at.
   Deletion is progressive, and the target follows what the pill currently shows:

     deployed pill  → the COPY IN COMFYUI (train/checkpoint/delete). The run's
                      save is untouched, so no run-folder space is freed and the
                      pill falls back to "not deployed" — where the same action
                      then aims at the save.
     plain pill     → the TRAINING SAVE in the run folder
                      (train/run-checkpoint/delete), a cloud pill by its
                      cloud_run_id.

   "Deployed" is read from checkpointDeployed(pill) — the SAME source of truth
   that decides between "✓ Deployed" and "Import → loras/…", never a second one.
   The deployed target needs the deployed copy's own name (`deployed_filename`,
   which the lineage resolves from the very map that sets `testable`): without it
   the route would reject an unknown checkpoint, so the action is withheld.

   Returns null when there is nothing to delete: a `gone` pill, an unresolvable
   body, or a cloud run still in flight (its pod keeps syncing epochs down, so a
   deletion would race the sync — the flat checkpoint list hides its 🗑 for
   exactly those runs). The server keeps the authoritative guards (it refuses
   while THIS dataset trains locally, and only accepts filenames it listed
   itself); this is the honest front gate, not a substitute. */
export function checkpointDeleteTarget(node, pill) {
  if (!node || !pill) return null;
  if (node.source === 'cloud' && CLOUD_ACTIVE_STATES.includes(node.status)) return null;
  if (checkpointDeployed(pill)) {
    if (!pill.deployed_filename) return null;
    return {
      kind: 'deployed',
      path: 'train/checkpoint/delete',
      body: { filename: pill.deployed_filename, train_type: node.train_type },
      filename: pill.deployed_filename,
      label: 'Remove from ComfyUI',
      title: 'Move the imported copy out of ComfyUI\'s loras folder. The training save in the run folder is kept — this pill will then offer to delete that save.',
    };
  }
  if (pill.present === false) return null;
  const body = lineageDeletePayload(node, pill);
  if (!body) return null;
  return {
    kind: 'save',
    path: 'train/run-checkpoint/delete',
    body,
    filename: pill.filename,
    label: 'Delete the training save',
    title: 'Move this run\'s checkpoint file to the trash (recoverable until you empty it in Settings). This checkpoint is not imported in ComfyUI.',
  };
}

/* Is this checkpoint the one pinned as the dataset's ★ best settings in the Test
   Studio? Compared on the BASENAME, exactly like the flat list's guard-rail: the
   pin stores the deployed LoRA's path, the pill stores the run-dir filename, and
   the import keeps the name. Unknown pin (not loaded on this mount) → false. */
export function checkpointIsBestSettings(pill, bestSettingsLora) {
  if (!pill || !bestSettingsLora) return false;
  const tail = (s) => String(s).split(/[\\/]/).pop();
  const pin = tail(bestSettingsLora);
  return [pill.deployed_filename, pill.filename]
    .filter(Boolean).some((f) => tail(f) === pin);
}

/* The confirmation text, which must NAME THE TARGET OF THE MOMENT — the two
   deletions look identical from the popover and are not the same loss. It also
   says what SURVIVES (deleting the ComfyUI copy frees nothing in the run folder;
   the save is what actually holds the disk), that everything goes to the TRASH
   (recoverable until emptied in Settings), and opens with a ⚠ header when the
   checkpoint is the pinned ★ best settings. Pure, so the wording is unit-tested
   rather than eyeballed in a popover. */
export function describeCheckpointDelete(node, pill, { bestSettingsLora = null } = {}) {
  const target = checkpointDeleteTarget(node, pill);
  if (!target) return null;
  const isBest = checkpointIsBestSettings(pill, bestSettingsLora);
  const step = pill?.step != null ? ` (step ${pill.step})` : '';
  const lines = [];
  if (isBest) {
    lines.push(target.kind === 'deployed'
      ? '⚠ This is the LoRA pinned as this dataset\'s ★ BEST SETTINGS in the Test Studio — the saved combo will stop working.'
      : '⚠ This save is the one pinned as this dataset\'s ★ BEST SETTINGS in the Test Studio.', '');
  }
  if (target.kind === 'deployed') {
    lines.push(`REMOVE FROM COMFYUI — « ${target.filename} »${step}?`, '',
      'Only the copy imported into ComfyUI goes to the trash (recoverable until you empty it in Settings).',
      'The training save in the run folder is KEPT — this frees no space there. Once removed, this checkpoint offers to delete that save instead.');
  } else {
    lines.push(`DELETE THE TRAINING SAVE — « ${target.filename} »${step}?`, '',
      'This is the run\'s own checkpoint file, not a ComfyUI copy (this checkpoint isn\'t imported).',
      'It goes to the trash — recoverable until you empty it in Settings.');
  }
  return { isBest, kind: target.kind, message: lines.join('\n') };
}

/* Parse the shared seed field: a blank/whitespace value means "let the engine
   pick one" (null); a valid non-negative integer is used as-is; anything else is
   rejected (returns { error }) so a typo never silently reseeds the comparison. */
export function parseSeedInput(raw) {
  const s = (raw == null ? '' : String(raw)).trim();
  if (s === '') return { seed: null };
  if (!/^\d+$/.test(s)) return { error: 'Seed must be a whole number' };
  return { seed: Number(s) };
}
