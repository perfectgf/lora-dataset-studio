import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deployedSummary, familyLabel, groupByDataset, rowKey, undeployButtonLabel,
  undeployConfirm, undeployItems, undeployOutcome,
} from './deployedLoras.js';

const row = (over = {}) => ({
  dataset_id: 1, dataset_name: 'Elsa', family: 'krea',
  filename: 'krea\\lora_elsa_1000.safetensors', label: 'elsa · step 1000', ...over,
});

test('rows are keyed on the pair the SERVER de-duplicates on', () => {
  // Anything else lets two rows disagree about being the same file.
  assert.equal(rowKey(row()), 'krea::krea\\lora_elsa_1000.safetensors');
  assert.notEqual(rowKey(row()), rowKey(row({ family: 'flux' })));
});

test('an unknown family keeps its id instead of vanishing', () => {
  // A LoRA in a folder we have no label for is still a LoRA the user may want
  // gone — hiding the row would make it unremovable from this screen.
  assert.equal(familyLabel('krea'), 'Krea 2');
  assert.equal(familyLabel('some_new_family'), 'some_new_family');
  assert.equal(familyLabel(''), 'Unknown family');
});

test('grouping answers the question people actually ask of this list', () => {
  const groups = groupByDataset([
    row({ dataset_name: 'Zoe', dataset_id: 2, label: 'zoe · 2000' }),
    row({ label: 'elsa · b' }),
    row({ label: 'elsa · a', filename: 'krea\\other.safetensors' }),
  ]);
  assert.deepEqual(groups.map((g) => g.datasetName), ['Elsa', 'Zoe']);
  assert.deepEqual(groups[0].rows.map((r) => r.label), ['elsa · a', 'elsa · b']);
});

test('a row with no filename is dropped rather than shown unremovable', () => {
  assert.deepEqual(groupByDataset([{ dataset_id: 1, dataset_name: 'X' }]), []);
  assert.deepEqual(groupByDataset(null), []);
});

test('the request can only ever name rows the SERVER listed', () => {
  /* This screen deletes files. The server decides which files belong to the app
     (a Civitai download in the same folder does not), so the request is built by
     FILTERING its rows — a name that never came from it cannot be constructed. */
  const rows = [row(), row({ family: 'flux', filename: 'flux\\a.safetensors' })];
  const keys = new Set([rowKey(rows[0]), 'krea::a_lora_i_downloaded.safetensors']);
  const items = undeployItems(rows, keys);
  assert.equal(items.length, 1);
  assert.deepEqual(items[0], {
    dataset_id: 1, filename: 'krea\\lora_elsa_1000.safetensors', train_type: 'krea',
  });
});

test('nothing ticked sends nothing', () => {
  assert.deepEqual(undeployItems([row()], new Set()), []);
  assert.deepEqual(undeployItems([row()], undefined), []);
});

test('the confirmation states the count and what SURVIVES', () => {
  const msg = undeployConfirm(12);
  assert.match(msg, /12 LoRAs/);
  assert.match(msg, /training saves are kept/i);
  assert.match(msg, /deployed again/i);
  assert.match(msg, /trash/i);
  assert.match(undeployConfirm(1), /1 LoRA\b/);
});

test('the ledger keeps its three outcomes apart', () => {
  const ok = undeployOutcome({ removed: ['a', 'b'], missing: [], failed: [] });
  assert.equal(ok.type, 'success');
  assert.match(ok.text, /2 removed from ComfyUI/);
  assert.match(ok.text, /deploy any of them again/);

  // "already gone" is NOT an error: the user asked for it out of ComfyUI, and
  // it is out of ComfyUI.
  const gone = undeployOutcome({ removed: ['a'], missing: ['b'], failed: [] });
  assert.equal(gone.type, 'success');
  assert.match(gone.text, /1 already gone/);

  // A refusal must not hide behind the successes.
  const bad = undeployOutcome({ removed: ['a'], missing: [], failed: [{ filename: 'x' }] });
  assert.equal(bad.type, 'warning');
  assert.match(bad.text, /1 refused/);

  const nothing = undeployOutcome({ removed: [], missing: [], failed: [] });
  assert.equal(nothing.type, 'info');
});

test('the button and the summary quote what they will move', () => {
  assert.equal(undeployButtonLabel(0), '⏏ Undeploy');
  assert.equal(undeployButtonLabel(3), '⏏ Undeploy 3 selected');
  assert.equal(deployedSummary([], 0), 'Nothing is deployed in ComfyUI right now.');
  assert.equal(deployedSummary([row()], 0), '1 LoRA deployed');
  assert.equal(deployedSummary([row(), row()], 1), '2 LoRAs deployed · 1 selected');
});
