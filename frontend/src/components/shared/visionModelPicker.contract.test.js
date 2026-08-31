/**
 * The vision route's model is chosen IN the scan window — named, listed,
 * pullable — on BOTH surfaces through the one shared engine choice. `node
 * --test` parses no JSX, so what is mounted where is pinned by reading the
 * source: a picker that quietly moved to one surface, or a pull that talked to
 * a provider-pinned endpoint, would leave every behavioural test green.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const picker = read('./VisionModelPicker.jsx');
const choice = read('./WatermarkEngineChoice.jsx');
const dialog = read('../dataset/WatermarkScanDialog.jsx');
const bank = read('../bank/BankWorkspace.jsx');

test('the picker lives under the shared engine choice, on its vision branch only', () => {
  assert.match(choice, /import VisionModelPicker from '\.\/VisionModelPicker\.jsx'/);
  assert.match(choice, /status\.runs === 'vision' && \(\s*<VisionModelPicker/);
  // ONE mount per surface — the dataset dialog and the bank panel inherit it
  // from the shared choice; neither grows a second, drifting copy.
  for (const [name, src] of [['dataset dialog', dialog], ['bank panel', bank]]) {
    assert.equal((src.match(/<WatermarkEngineChoice\b/g) || []).length, 1, name);
    assert.doesNotMatch(src, /<VisionModelPicker\b/, `${name} mounts the picker directly`);
  }
});

test('it names the model, lists the installed ones and pulls through the routed endpoint', () => {
  assert.match(picker, /aria-label="Watermark vision model"/);
  assert.match(picker, /apiFetch\('\/api\/local-llm\/models'\)/);
  assert.match(picker, /postJson\('\/api\/local-llm\/pull'/);
  assert.match(picker, /putJson\('\/api\/settings', \{ config: visionModelSetting\(provider, value\) \}\)/);
  // A provider-pinned pull would answer about the wrong server under LM Studio.
  assert.doesNotMatch(picker, /\/api\/ollama\/pull/);
});

test('a finished pull selects what it fetched instead of leaving it in a list', () => {
  assert.match(picker, /state === 'done'\) \{\s*await loadModels\(\);\s*if \(s\.model\) await save\(s\.model\)/);
});

test('the sentence above the picker follows the pick at once', () => {
  assert.match(choice, /withVisionModel\(caps, activeLocalLlm\(caps\)\.provider, visionModel\)/);
  assert.match(choice, /watermarkEngineStatus\(value, capsView\)/);
});
