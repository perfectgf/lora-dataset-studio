/* What the LoRA picker SHOWS, as pure functions — so `node --test` can hold
 * the two decisions that turned a 20-row wall back into a choice:
 *
 *  - a trained run is ONE row, its checkpoints are pills inside it. Two rows
 *    for "run #174 final" and "run #174 step 1000" read as two LoRAs; they are
 *    one training at two moments;
 *  - the files ComfyUI's h3 folder holds are not all LoRAs to TEST. The turbo
 *    distillations, the camera-motion adapter and its re-keyed twin are engine
 *    parts the graph grafts itself — offering them here invites a comparison
 *    that means nothing. They stay reachable, folded, and named for what they
 *    are.
 */

const ENGINE_PART = /turbo|lightx2v|camera_motion|_h3keys|ref2v/i;

/** Engine machinery, not a candidate: the grafted turbo LoRAs, the camera
 * adapter, the ref2v variants. Matched on the FILENAME because that is the
 * only thing a file in a folder carries. */
export function isEnginePart(filename) {
  return ENGINE_PART.test(String(filename || ''));
}

/** `lds174_video_Harbour___stills_000001000.safetensors` → `Harbour — stills`.
 * The run prefix is the row's subtitle, the step suffix is the pill, the
 * folder and the extension are noise. Triple underscores were an em dash on
 * the way in (the export flattens names). */
export function shortLoraName(filename) {
  const base = String(filename || '').replace(/\\/g, '/').split('/').pop()
    .replace(/\.safetensors$/i, '')
    .replace(/^lds\d+_video_/, '')
    .replace(/_\d{6,}$/, '');
  return base.replace(/___/g, ' — ').replace(/_/g, ' ').trim() || String(filename || '');
}

/** The training step a checkpoint filename encodes: `..._000001000` → 1000,
 * no suffix → null (the final save carries the run's target in its name only
 * through the training config, which the file does not know). */
export function checkpointStep(filename) {
  const s = String(filename || '').replace(/\.safetensors$/i, '');
  const m = /_(\d{6,})$/.exec(s);
  return m ? Number(m[1]) : null;
}

/** [{run_id, dataset_id, name, checkpoints: [{filename, step, final, deployed_as, label}]}]
 * newest run first, pills ordered final first then descending step — the way
 * someone reads a run: "the result, and the moments before it". */
export function groupTrained(trained) {
  const byRun = new Map();
  for (const t of trained || []) {
    const key = t.run_id;
    if (!byRun.has(key)) {
      byRun.set(key, {
        run_id: t.run_id, dataset_id: t.dataset_id,
        name: shortLoraName(t.filename), checkpoints: [],
      });
    }
    const step = checkpointStep(t.filename);
    byRun.get(key).checkpoints.push({
      filename: t.filename, deployed_as: t.deployed_as || null, label: t.label,
      step, final: step === null,
    });
  }
  const groups = [...byRun.values()];
  for (const g of groups) {
    g.checkpoints.sort((a, b) => (Number(b.final) - Number(a.final)) || ((b.step || 0) - (a.step || 0)));
  }
  groups.sort((a, b) => (b.run_id || 0) - (a.run_id || 0));
  return groups;
}

/** Split the folder list into candidates and engine parts. */
export function splitDeployed(deployed) {
  const candidates = [];
  const parts = [];
  for (const d of deployed || []) (isEnginePart(d.filename) ? parts : candidates).push(d);
  return { candidates, parts };
}
