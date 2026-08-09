import assert from 'node:assert/strict';
import test from 'node:test';
import {
  anchorDataset, canvasBaseModelAxis, canvasBlendBlocker, canvasCheckpointKey, canvasFamily,
  canvasRunSelections, canvasSelectionSummary, canvasStackKey, canvasStackTriggers,
  canvasStackWeight, canvasStackWithoutTrigger, canvasUndeployed,
  describeCanvasLaunch, isCanvasCheckpointSelected,
  pruneCanvasSelection, refreshCanvasSelection, toggleCanvasCheckpoint,
} from './canvasGeneration.js';

const pick = (datasetId, recordId, step, extra = {}) => ({
  datasetId, recordId, step, family: 'krea', deployed: true,
  filename: `krea\\lora_x_${step}.safetensors`, ...extra,
});

test('picking is a toggle that keeps the order it was clicked in', () => {
  let sel = [];
  sel = toggleCanvasCheckpoint(sel, pick(1, 10, 1000));
  sel = toggleCanvasCheckpoint(sel, pick(2, 20, 2000));
  assert.deepEqual(sel.map((e) => e.step), [1000, 2000]);
  // Re-clicking the same pill removes it, and nothing else moves.
  sel = toggleCanvasCheckpoint(sel, pick(1, 10, 1000));
  assert.deepEqual(sel.map((e) => e.step), [2000]);
  assert.equal(isCanvasCheckpointSelected(sel, 2, 20, 2000), true);
  assert.equal(isCanvasCheckpointSelected(sel, 1, 10, 1000), false);
});

test('the same step of two DIFFERENT runs are two different picks', () => {
  // The board holds several datasets: (record, step) alone is not an identity.
  let sel = toggleCanvasCheckpoint([], pick(1, 10, 1000));
  sel = toggleCanvasCheckpoint(sel, pick(2, 11, 1000));
  assert.equal(sel.length, 2);
});

test('the FIRST pick anchors the settings panel', () => {
  // The panel's model/aspect/cfg choices come from one dataset's payload; it must
  // not move under the user because a later pick came from elsewhere.
  const sel = [pick(7, 1, 500), pick(9, 2, 500)];
  assert.equal(anchorDataset(sel), 7);
  assert.equal(anchorDataset([]), null);
});

test('nothing picked → blocked, and it says what to do', () => {
  const d = describeCanvasLaunch([]);
  assert.equal(d.blocked, true);
  assert.match(d.reason, /Tick the ✓ box/);
});

test('MIXED FAMILIES are refused, naming the two families in the way', () => {
  // Not a comfort restriction: krea and zimage share no engine. The refusal has
  // to be legible, or the dead button reads as a bug.
  const d = describeCanvasLaunch([pick(1, 10, 1000), pick(2, 20, 2000, { family: 'zimage' })]);
  assert.equal(d.blocked, true);
  assert.equal(d.family, null);
  assert.match(d.reason, /Krea 2 \+ Z-Image/);
  assert.match(d.reason, /different base models and workflows/);
});

test('several datasets of the SAME family launch together — that is the point', () => {
  const sel = [pick(1, 10, 1000), pick(2, 20, 2000), pick(2, 21, 500)];
  const d = describeCanvasLaunch(sel);
  assert.equal(d.blocked, false);
  assert.equal(d.needsDeploy, false);
  assert.equal(d.family, 'krea');
  assert.deepEqual(d.datasets, [1, 2]);
  assert.match(d.reason, /3 checkpoints across 2 datasets/);
  assert.deepEqual(canvasRunSelections(sel), [
    { dataset_id: 1, checkpoint: 'krea\\lora_x_1000.safetensors', record_id: 10, step: 1000 },
    { dataset_id: 2, checkpoint: 'krea\\lora_x_2000.safetensors', record_id: 20, step: 2000 },
    { dataset_id: 2, checkpoint: 'krea\\lora_x_500.safetensors', record_id: 21, step: 500 },
  ]);
});

test('a launch carries the EXACT origin of every pick, never a filename to re-parse', () => {
  const [one] = canvasRunSelections([pick(4, 12, 3000)]);
  assert.equal(one.record_id, 12);
  assert.equal(one.step, 3000);
});

test('a NOT-DEPLOYED pick makes the button announce the deploy it would do', () => {
  const sel = [pick(1, 10, 1000), pick(1, 10, 2000, { deployed: false })];
  const d = describeCanvasLaunch(sel);
  assert.equal(d.blocked, false);          // possible — just not yet
  assert.equal(d.needsDeploy, true);
  assert.equal(d.label, 'Deploy 1 checkpoint, then generate');
  assert.equal(canvasUndeployed(sel).length, 1);
  // …and the request only ever carries what ComfyUI can actually load.
  assert.deepEqual(canvasRunSelections(sel).map((s) => s.step), [1000]);
});

test('the deploy label counts, so the user knows how much is about to be written', () => {
  const sel = [pick(1, 10, 1000, { deployed: false }), pick(1, 10, 2000, { deployed: false })];
  assert.equal(describeCanvasLaunch(sel).label, 'Deploy 2 checkpoints, then generate');
});

test('a mixed-family selection is refused even when everything is deployed', () => {
  // Family beats every other consideration: there is no engine for it.
  const d = describeCanvasLaunch([pick(1, 10, 1000, { deployed: false }),
    pick(2, 20, 2000, { family: 'sdxl' })]);
  assert.equal(d.blocked, true);
  assert.match(d.reason, /Krea 2 \+ SDXL/);
});

test('the summary line reads as a sentence', () => {
  assert.equal(canvasSelectionSummary([]), 'No checkpoint picked');
  assert.equal(canvasSelectionSummary([pick(1, 10, 1000)]), '1 checkpoint · Krea 2');
  assert.equal(canvasSelectionSummary([pick(1, 10, 1000), pick(2, 20, 500)]),
    '2 checkpoints · 2 datasets · Krea 2');
});

test('picks that left the board are dropped, not launched blind', () => {
  const sel = [pick(1, 10, 1000), pick(2, 20, 2000)];
  const kept = pruneCanvasSelection(sel, ['1:10:1000']);
  assert.deepEqual(kept.map((e) => e.datasetId), [1]);
});

test('after a deploy the SAME picks come back deployed', () => {
  const sel = [pick(1, 10, 1000, { deployed: false, filename: null })];
  const after = refreshCanvasSelection(sel,
    (e) => (e.step === 1000 ? { deployed: true, filename: 'krea\\deployed.safetensors' } : null));
  assert.equal(after[0].deployed, true);
  assert.equal(after[0].filename, 'krea\\deployed.safetensors');
  assert.equal(describeCanvasLaunch(after).needsDeploy, false);
  // A pick whose pill vanished from the refetched board keeps what it had rather
  // than silently flipping state.
  const stale = refreshCanvasSelection(sel, () => null);
  assert.equal(stale[0].deployed, false);
});

test('canvasFamily is null the moment the picks disagree', () => {
  assert.equal(canvasFamily([pick(1, 1, 1)]), 'krea');
  assert.equal(canvasFamily([pick(1, 1, 1), pick(1, 2, 2, { family: 'sdxl' })]), null);
  assert.equal(canvasFamily([]), null);
});

/* --- 🧬 Blend ------------------------------------------------------------- */

test('a blend weight is keyed on the PILL, so deploying does not reset the sliders', () => {
  // THE reason the shim does not key on the checkpoint filename: that filename is
  // null while the save is on disk only, and it APPEARS when "Deploy, then
  // generate" runs. Keying on it would silently throw away every slider the user
  // moved before launching.
  const before = pick(3, 42, 1500, { deployed: false, filename: null });
  const after = { ...before, deployed: true, filename: 'krea\\deployed.safetensors' };
  assert.equal(canvasStackKey(before), canvasStackKey(after));
  assert.equal(canvasStackKey(before), canvasCheckpointKey(3, 42, 1500));

  const weights = { [canvasStackKey(before)]: 0.65 };
  assert.equal(canvasStackWeight(weights, after), 0.65);
});

test('a blend weight uses the Test Studio clamp, not a second one', () => {
  const e = pick(1, 10, 1000);
  const k = canvasStackKey(e);
  assert.equal(canvasStackWeight({}, e), 1);
  assert.equal(canvasStackWeight({ [k]: 9 }, e), 5);   // the ceiling, 2 until 2026-08-08
  assert.equal(canvasStackWeight({ [k]: -4 }, e), 0);
  assert.equal(canvasStackWeight({ [k]: 0.5555 }, e), 0.56);
  assert.equal(canvasStackWeight({ [k]: 'nope' }, e), 1);
});

test('blending needs two picks of ONE family, and says which rule is in the way', () => {
  assert.match(canvasBlendBlocker([pick(1, 10, 1000)]), /at least two/i);
  assert.equal(canvasBlendBlocker([pick(1, 10, 1000), pick(2, 20, 2000)]), null);
  const mixed = [pick(1, 10, 1000), pick(2, 20, 2000, { family: 'sdxl' })];
  const why = canvasBlendBlocker(mixed);
  assert.match(why, /krea/);
  assert.match(why, /sdxl/);
});

test('a blend payload carries one weight per pick; Compare stays byte for byte what it was', () => {
  const sel = [pick(1, 10, 1000), pick(2, 20, 2000)];
  const weights = { [canvasStackKey(sel[0])]: 0.9, [canvasStackKey(sel[1])]: 0.55 };
  // Depuis le balayage, chaque pastille porte sa LISTE de poids ; sans case
  // cochee elle vaut un element (le curseur), et `weight` reste envoye pour un
  // backend qui ignorerait encore le balayage.
  assert.deepEqual(canvasRunSelections(sel, { blend: true, weights }), [
    { dataset_id: 1, checkpoint: 'krea\\lora_x_1000.safetensors', record_id: 10, step: 1000, weight: 0.9, weights: [0.9] },
    { dataset_id: 2, checkpoint: 'krea\\lora_x_2000.safetensors', record_id: 20, step: 2000, weight: 0.55, weights: [0.55] },
  ]);
  // No `blend` → not one extra key. The comparison runs must not change shape.
  for (const entry of canvasRunSelections(sel)) {
    assert.deepEqual(Object.keys(entry).sort(),
      ['checkpoint', 'dataset_id', 'record_id', 'step']);
  }
});

test('the triggers a blend will inject are listed in pick order, de-duplicated', () => {
  const sel = [
    pick(1, 10, 1000, { triggerWord: 'aaa' }),
    pick(2, 20, 2000, { triggerWord: 'BBB' }),
    pick(2, 21, 3000, { triggerWord: 'bbb' }),   // same token, collapsed like the backend does
    pick(3, 30, 4000),                           // no trigger at all
  ];
  assert.deepEqual(canvasStackTriggers(sel), ['aaa', 'BBB']);
  assert.deepEqual(canvasStackWithoutTrigger(sel).map((e) => e.recordId), [30]);
  assert.deepEqual(canvasStackTriggers([]), []);
});

/* ◉ L'axe des BASES. Le panneau du board dit « BASE MODEL (MULTI) » et laissait
   cocher plusieurs bases ; le lancement n'en envoyait qu'une (`zModels[0]`) et
   jetait les autres sans un mot — trois bases cochées, une seule génération.
   Ce qui est épinglé ici, c'est la VALEUR envoyée, pas la forme du code. */

test('every ticked base model is sent, not just the first', () => {
  assert.deepEqual(canvasBaseModelAxis(['base_one', 'base_two', 'base_three']), {
    z_model: 'base_one',
    z_models: ['base_one', 'base_two', 'base_three'],
  });
});

test('nothing ticked stays the run it always was', () => {
  // `z_model: null` est ce que la route lisait déjà — un backend plus ancien
  // (ou plus récent) qui ignore `z_models` retrouve exactement l'ancien corps.
  assert.deepEqual(canvasBaseModelAxis([]), { z_model: null, z_models: [] });
  assert.deepEqual(canvasBaseModelAxis(null), { z_model: null, z_models: [] });
  assert.deepEqual(canvasBaseModelAxis(undefined), { z_model: null, z_models: [] });
});

test('the axis keeps the order it was ticked in, without duplicates', () => {
  // L'ordre est la lecture : le balayage rend les bases dans cet ordre-là.
  assert.deepEqual(canvasBaseModelAxis(['b', 'a', 'b']),
    { z_model: 'b', z_models: ['b', 'a'] });
});
