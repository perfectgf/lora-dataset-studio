/* What the ⏹ Stop dialog of a cloud run SAYS — the consequence of stopping, and
 * the optional "and do not rent this machine again".
 *
 * WHY THE BAN IS A TICK BOX AND NOT A RULE. The app already blacklists a vast
 * host on its own when it can classify a failure: a boot that never completes,
 * a pod that stops making progress, a checkpoint it cannot serve. A machine that
 * boots fine and then simply trains at half speed produces no failure at all, so
 * nothing automatic will ever ban it — and, symmetrically, stopping because you
 * changed your mind says nothing about the box. Only the person watching the
 * throughput knows, so only they can say. (Asked for by mr.arrow on Discord.)
 *
 * Plain .js (no JSX) so `node --test` can execute all of it.
 */

/** What stopping costs, which is not the same sentence for the two run kinds:
    a LoRA run keeps every checkpoint already downloaded, a full-model run can
    lose the latest one outright because the Hub upload only happens on a clean
    finish. */
export function stopConsequence(fullModel) {
  return fullModel
    ? 'AI Toolkit uploads the full model to Hugging Face only when the run finishes cleanly. '
      + 'The latest checkpoint can be permanently lost if it has not been uploaded yet, '
      + 'even if an older checkpoint is already available on the Hub.'
    : 'The pod is terminated. Any LoRA checkpoint reached so far is still downloaded '
      + 'and importable — you only lose the remaining steps.';
}

/** The run, named the way the user recognises it. */
export function stopTitle(run) {
  const who = run?.dataset_name || run?.run_name || `run #${run?.run_id}`;
  return `Stop the cloud run for “${who}”?`;
}

/** The machine, named from what the row actually shows. Returns '' when there is
    nothing identifiable — the tick box is then not offered at all rather than
    promising to ban something it cannot name. */
export function machineLabel(run) {
  const bits = [];
  if (run?.gpu) bits.push(run.gpu);
  if (run?.vast_instance_id) bits.push(`instance ${run.vast_instance_id}`);
  return bits.join(' · ');
}

export function canBanHost(run) {
  return !!machineLabel(run);
}

/** The tick box, and the sentence under it. Deliberately says when NOT to tick
    it: a ban costs the user the next launch on that box, and a stop made for any
    other reason should leave a good machine alone. */
export function banHostLabel(run) {
  const machine = machineLabel(run);
  return {
    label: 'Do not rent this machine again',
    detail: `${machine ? `${machine} — ` : ''}skipped by the next launches for as long as`
      + ' Settings ▸ Cloud keeps a bad host out (3 days by default). Tick this only if the'
      + ' machine itself was the problem — slow, unstable, misbehaving. A stop for any other'
      + ' reason says nothing about it.',
  };
}
