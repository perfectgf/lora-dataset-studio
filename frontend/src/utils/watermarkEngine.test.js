/* The 🚩 engine selector's pure rules — every branch mirrors a branch of the
 * backend's `watermark_detector.resolve_backend`, because the sentence shown
 * BEFORE the run must describe the route the server will actually take. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  DETECTOR_INSTALL_ROUTE, WATERMARK_ENGINES, normalizeEngine, pullCopy, visionModelSetting,
  watermarkEngineStatus, withVisionModel,
} from './watermarkEngine.js';

const CAPS_LMS = {
  watermark_detect: false,
  local_llm: { provider: 'lmstudio' },
  lmstudio: { reachable: true, model_ready: true, vision_model: 'qwen/qwen3-vl-4b' },
};

test('the three choices are the backend’s three values, auto first', () => {
  assert.deepEqual(WATERMARK_ENGINES.map((e) => e.id), ['auto', 'detector', 'vision']);
  // A hand-edited config reads as auto rather than crashing a dialog.
  for (const junk of ['', null, 'AUTO', ' detector ', 'surprise']) {
    assert.ok(['auto', 'detector'].includes(normalizeEngine(junk)));
  }
  assert.equal(normalizeEngine('surprise'), 'auto');
});

test('auto runs the detector when installed, the vision model otherwise', () => {
  const withDetector = watermarkEngineStatus('auto', { ...CAPS_LMS, watermark_detect: true });
  assert.equal(withDetector.runs, 'detector');
  assert.equal(withDetector.warn, false);
  assert.match(withDetector.line, /SigLIP2/);

  const without = watermarkEngineStatus('auto', CAPS_LMS);
  assert.equal(without.runs, 'vision');
  assert.equal(without.warn, false, 'auto falling to vision is the design, not a warning');
});

test('vision pinned always runs the vision model, and names it', () => {
  const s = watermarkEngineStatus('vision', { ...CAPS_LMS, watermark_detect: true });
  assert.equal(s.runs, 'vision');
  assert.match(s.line, /qwen\/qwen3-vl-4b/);
  assert.match(s.line, /LM Studio/);
});

test('detector pinned but missing is the ONE amber case, and it hands back the remedy', () => {
  // Mirror of the backend's fell_back: the run still happens, on the vision
  // route, and the sentence says where the detector installs from.
  const s = watermarkEngineStatus('detector', CAPS_LMS);
  assert.equal(s.runs, 'vision');
  assert.equal(s.warn, true);
  assert.ok(s.line.includes(DETECTOR_INSTALL_ROUTE));
});

test('the vision sentence follows the ACTIVE provider', () => {
  const ollama = watermarkEngineStatus('vision', {
    watermark_detect: false,
    ollama: { installed: true, reachable: true, vision_model: 'qwen3-vl:8b' },
  });
  assert.match(ollama.line, /qwen3-vl:8b/);
  assert.match(ollama.line, /Ollama/);
  assert.doesNotMatch(ollama.line, /LM Studio/);
});

/* --- parity: one selector, both scan windows ------------------------------- */
test('both scan windows mount the engine choice', () => {
  const read = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');
  for (const [name, rel] of [
    ['dataset', 'components/dataset/WatermarkScanDialog.jsx'],
    ['bank', 'components/bank/BankWorkspace.jsx'],
  ]) {
    const src = read(rel);
    assert.match(src, /<WatermarkEngineChoice caps=/, `the ${name} window lost the selector`);
    // ...and the threshold row keys on the RESOLVED route, so pinning vision
    // hides a dial that would apply to nothing.
    assert.match(src, /watermarkEngineStatus\((wmEngine|engine), caps\)\.runs === 'detector'/,
      `the ${name} threshold no longer follows the resolved route`);
  }
  // The component saves the SAME key both scan routes read — write-through,
  // like the threshold beside it.
  const comp = read('components/shared/WatermarkEngineChoice.jsx');
  assert.match(comp, /watermark_detect: \{ backend: engine \}/);
});

// --- the vision model, picked IN the scan window ------------------------------

test('picking a vision model writes the key the scan reads, per provider', () => {
  assert.deepEqual(visionModelSetting('ollama', ' llava:13b '), { ollama: { vision_model: 'llava:13b' } });
  assert.deepEqual(visionModelSetting('lmstudio', 'qwen/qwen3-vl-4b'),
    { lmstudio: { vision_model: 'qwen/qwen3-vl-4b' } });
  // An unknown provider falls back to the historical key rather than inventing one.
  assert.deepEqual(visionModelSetting('surprise', 'x'), { ollama: { vision_model: 'x' } });
});

test('the status sentence names the model just picked, before any caps refresh', () => {
  const lms = withVisionModel(CAPS_LMS, 'lmstudio', 'google/gemma-3-12b');
  assert.match(watermarkEngineStatus('vision', lms).line, /google\/gemma-3-12b via LM Studio/);
  const oll = withVisionModel({ local_llm: { provider: 'ollama' }, ollama: { reachable: true } },
    'ollama', 'llava:13b');
  assert.match(watermarkEngineStatus('vision', oll).line, /llava:13b via Ollama/);
  // The overlay is a copy — the caps object handed in is not mutated.
  assert.equal(CAPS_LMS.lmstudio.vision_model, 'qwen/qwen3-vl-4b');
});

test('the pull affordance speaks each provider’s language', () => {
  assert.equal(pullCopy('ollama').button, '⏬ Pull');
  assert.equal(pullCopy('lmstudio').button, '⏬ Download');
  assert.match(pullCopy('lmstudio').placeholder, /huggingface/);
});
