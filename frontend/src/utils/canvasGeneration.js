/* Picking checkpoints ON THE BOARD, and deciding what a launch from there means.

   The LoRA Canvas mounts the very same Test-Studio engine as the Test Studio
   page — the same hooks, the same settings panel, the same POST. Exactly one
   thing differs: the checkpoints are not chosen in a picker, they are chosen by
   clicking the pills under the run cards, possibly across several datasets.

   Everything that follows from THAT lives here, JSX-free so `node --test` can
   exercise it without a browser:

     · the selection itself (a list, order = the order they were clicked);
     · whether a launch is possible, and if not, in the user's words WHY;
     · the request body the shared engine expects.

   Two refusals matter and they are not the same kind of thing:

     mixed FAMILIES — an impossibility. Krea and Z-Image do not share a base
       model or a workflow; one run cannot render both. The button goes dead and
       says which two families are in the way.
     not DEPLOYED — merely something to do first. The button offers to do it:
       "Deploy 2 checkpoints, then generate". Nothing is written to the user's
       ComfyUI folder until they click a button that said it would. */

// Explicit extension: this module is imported by `node --test` (which does not
// resolve extensionless specifiers) as well as by Vite.
import { famLabel } from './familyLabels.js';
import {
  blendConfigCount, combineBlocker, stackKey, stackWeight,
  stackWeightList, stackWeightSet,
} from '../components/dataset/studio/loraStack.js';

/** Stable identity of one pick. The canvas holds several datasets at once, so a
 *  (record, step) pair is not enough on its own to be unique across the board —
 *  and being able to say WHICH lane a pick came from is what makes the
 *  cross-dataset launch legible. */
export function canvasCheckpointKey(datasetId, recordId, step) {
  return `${datasetId}:${recordId}:${step}`;
}

const keyOf = (e) => canvasCheckpointKey(e.datasetId, e.recordId, e.step);

/** Add / remove a checkpoint. Pure: returns a NEW array, order preserved, so the
 *  first pick stays the first pick (it is the one whose dataset anchors the
 *  settings panel — a selection that reordered itself would move the panel's
 *  ground under the user mid-edit). */
export function toggleCanvasCheckpoint(selection, entry) {
  const list = selection || [];
  const k = keyOf(entry);
  return list.some((e) => keyOf(e) === k)
    ? list.filter((e) => keyOf(e) !== k)
    : [...list, entry];
}

export function isCanvasCheckpointSelected(selection, datasetId, recordId, step) {
  const k = canvasCheckpointKey(datasetId, recordId, step);
  return (selection || []).some((e) => keyOf(e) === k);
}

/** Which dataset the shared Test-Studio settings hang off. The FIRST pick: its
 *  payload supplies the model / aspect / cfg / steps choices and the recent
 *  prompts, and it is the one the user started from. */
export function anchorDataset(selection) {
  return (selection || [])[0]?.datasetId ?? null;
}

/** The one family a launch can have, or null when the selection is empty or
 *  straddles several. */
export function canvasFamily(selection) {
  const fams = [...new Set((selection || []).map((e) => e.family).filter(Boolean))];
  return fams.length === 1 ? fams[0] : null;
}

/* 🧬 BLEND — the board's side of the Test Studio's Blend mode.
   ---------------------------------------------------------------------------
   Compare (the default, and everything above) runs one pass PER ticked
   checkpoint. Blend loads them ALL into a single generation, each at its own
   weight, with every dataset's trigger word injected into the prompt.

   None of that logic is re-written here. The weights, their 0..2 clamp and the
   "at least two LoRAs, all of one family" rule are the Test Studio's, imported
   from studio/loraStack.js — one implementation, so the two screens cannot
   disagree about what a blend is. What this section adds is the shim between
   the board's camelCase picks and the shape that module reads.

   The shim is built so that `stackKey` of a shimmed pick IS
   `canvasCheckpointKey` of that pick: a weight is keyed on the PILL, never on
   the deployed filename. That filename is null while a checkpoint is on disk
   only, and it APPEARS when "Deploy, then generate" runs — keying on it would
   have silently reset every slider the user moved before launching. */

/** The picks, in the shape studio/loraStack.js reads. */
function canvasStackSelection(selection) {
  return (selection || []).map((e) => ({
    dataset_id: e.datasetId,
    checkpoint: `${e.recordId}:${e.step}`,
    family: e.family,
    // Le nom que `blendComboLabel` mettra sur la cellule. Le Test Studio y met un
    // nom de LoRA, le board y met sa lane et sa pastille — d'où le champ ici
    // plutôt qu'un second fabricant d'étiquettes.
    lora_label: `${e.datasetName || `Dataset ${e.datasetId}`} #${e.recordId}·${e.step}`,
  }));
}

/** Les poids COCHÉS d'une pastille (balayage 🧬), et ceux qu'elle balayera
 *  réellement — cases si elle en a, curseur sinon. La règle est celle du Test
 *  Studio, importée, pas réécrite. */
export const canvasStackWeightSet = (sets, entry) =>
  stackWeightSet(sets, canvasStackSelection([entry])[0]);
const canvasStackWeightList = (weights, sets, entry) =>
  stackWeightList(weights, sets, canvasStackSelection([entry])[0]);

/** Combien de combinaisons ce board va lancer. */
export const canvasBlendConfigCount = (selection, opts = {}) =>
  blendConfigCount(canvasStackSelection(selection), opts);

/** Weight-map key of ONE pick — equal to `canvasCheckpointKey` by construction. */
export const canvasStackKey = (entry) => stackKey(canvasStackSelection([entry])[0]);

/** This pick's weight in the blend: the Studio's clamp, 1 by default. */
export const canvasStackWeight = (weights, entry) =>
  stackWeight(weights, canvasStackSelection([entry])[0]);

/** Why this selection cannot be blended, or null. The Studio's own rule. */
export const canvasBlendBlocker = (selection) =>
  combineBlocker(canvasStackSelection(selection));

/** The trigger words a blend will prepend to the prompt, in pick order and
 *  de-duplicated case-insensitively — the same collapse the backend does, so
 *  what the panel promises is what the prompt gets. */
export function canvasStackTriggers(selection) {
  const out = [];
  for (const e of (selection || [])) {
    const t = String(e.triggerWord || '').trim();
    if (t && !out.some((v) => v.toLowerCase() === t.toLowerCase())) out.push(t);
  }
  return out;
}

/** Picks that will load with no trigger word of their own. Shown, not hidden:
 *  a blend member whose dataset never got a trigger contributes its weights but
 *  nothing the prompt can call on, and that is worth knowing before the GPU. */
export const canvasStackWithoutTrigger = (selection) =>
  (selection || []).filter((e) => !String(e.triggerWord || '').trim());

/** The `selections` array POST /api/studio/run expects — plus the exact origin
 *  of each pick, which the canvas KNOWS (the user clicked that pill) and which
 *  the engine stamps on every image it produces. Only DEPLOYED picks make it in:
 *  a checkpoint with no LoRA in ComfyUI has nothing to load.
 *
 *  In a 🧬 blend each entry also carries its `weight`; in Compare the body stays
 *  byte for byte what it always was, so existing runs change in nothing. */
export function canvasRunSelections(selection, { blend = false, weights = {}, sets = {} } = {}) {
  return (selection || [])
    .filter((e) => e.deployed && e.filename)
    .map((e) => ({ dataset_id: e.datasetId,
      checkpoint: e.filename,
      record_id: e.recordId,
      step: e.step,
      ...(blend ? {
        // Comme le Studio : `weight` scalaire EN PLUS de la liste, pour qu'un
        // backend qui ignore encore le balayage rende une image sensée.
        weight: canvasStackWeightList(weights, sets, e)[0],
        weights: canvasStackWeightList(weights, sets, e),
      } : {}) }));
}

/**
 * ◉ The BASE MODEL axis of a launch from the board.
 *
 * The panel offers « BASE MODEL (MULTI) » and the engine has always swept a
 * `z_models` list — but the board sent `zModels[0]` and dropped the rest in
 * silence, so three ticked bases produced one generation and nothing said the
 * other two had been left behind. A limit nobody is told about is indistinguish-
 * able from a run that did what was asked.
 *
 * The scalar rides along with the list, exactly like the blend weights above:
 * a backend that does not know `z_models` yet still renders a sensible image
 * instead of falling back to whatever base happened to be first on disk.
 */
export function canvasBaseModelAxis(zModels) {
  const list = [];
  for (const m of (Array.isArray(zModels) ? zModels : [])) {
    if (!list.includes(m)) list.push(m);
  }
  return { z_model: list[0] ?? null, z_models: list };
}

/** The checkpoints that would have to be deployed into ComfyUI before this
 *  launch can run. */
export function canvasUndeployed(selection) {
  return (selection || []).filter((e) => !e.deployed);
}

/** Everything the launch bar needs to render itself honestly.
 *
 *  Returns { count, datasets, families, family, undeployed, needsDeploy,
 *            blocked, reason, label }:
 *    blocked — the launch cannot happen at all (nothing picked, or families
 *              mixed). `reason` is shown, always; a dead button that does not
 *              say why is the thing this replaces.
 *    needsDeploy — it can happen, after deploying. `label` announces it. */
export function describeCanvasLaunch(selection) {
  const list = selection || [];
  const families = [...new Set(list.map((e) => e.family).filter(Boolean))];
  const datasets = [...new Set(list.map((e) => e.datasetId))];
  const undeployed = canvasUndeployed(list);
  const base = {
    count: list.length, datasets, families, undeployed,
    family: families.length === 1 ? families[0] : null,
  };
  if (!list.length) {
    return { ...base, needsDeploy: false, blocked: true, label: 'Generate',
      reason: 'Tick the ✓ box on a checkpoint to add it to this run.' };
  }
  if (families.length > 1) {
    const named = families.map(famLabel).join(' + ');
    return { ...base, needsDeploy: false, blocked: true, label: 'Generate',
      reason: `${named} cannot run together — they use different base models and `
        + 'workflows, so a single run has no engine that can render both. '
        + 'Unpick one family.' };
  }
  if (undeployed.length) {
    const n = undeployed.length;
    return { ...base, needsDeploy: true, blocked: false,
      label: `Deploy ${n} checkpoint${n > 1 ? 's' : ''}, then generate`,
      reason: `${n} of your picks ${n > 1 ? 'are' : 'is'} not in ComfyUI yet. `
        + 'This copies them there first, then launches.' };
  }
  return { ...base, needsDeploy: false, blocked: false, label: 'Generate',
    reason: datasets.length > 1
      ? `${list.length} checkpoints across ${datasets.length} datasets, one shared prompt and seed.`
      : null };
}

/** A one-line recap of the picks for the panel header, e.g.
 *  "3 checkpoints · 2 datasets · Krea 2". */
export function canvasSelectionSummary(selection) {
  const d = describeCanvasLaunch(selection);
  if (!d.count) return 'No checkpoint picked';
  const bits = [`${d.count} checkpoint${d.count > 1 ? 's' : ''}`];
  if (d.datasets.length > 1) bits.push(`${d.datasets.length} datasets`);
  if (d.families.length) bits.push(d.families.map(famLabel).join(' + '));
  return bits.join(' · ');
}

/** Drop the picks that are no longer on the board (their dataset was unticked in
 *  the filter, their run was deleted). A selection that outlived what it points
 *  at would launch on a checkpoint the user can no longer see. */
export function pruneCanvasSelection(selection, liveKeys) {
  const live = liveKeys instanceof Set ? liveKeys : new Set(liveKeys || []);
  return (selection || []).filter((e) => live.has(keyOf(e)));
}

/** Re-read each pick's DEPLOYED state (and its deployed filename) off a freshly
 *  fetched board, keeping the user's picks exactly as they are. This is what
 *  makes "deploy, then generate" a single gesture: after the imports the lineage
 *  is refetched, and the same picks come back testable. */
export function refreshCanvasSelection(selection, pillLookup) {
  return (selection || []).map((e) => {
    const fresh = pillLookup(e);
    if (!fresh) return e;
    return { ...e, deployed: !!fresh.deployed, filename: fresh.filename ?? e.filename };
  });
}
