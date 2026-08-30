/* The ⚙️ Enhance model choice, pinned at the seams a screenshot cannot see.
 *
 * Two families of assertion:
 * — enhanceGate.js is plain JS, so its rules are EXECUTED;
 * — the JSX files are read as TEXT (node --test parses no JSX), so the wiring
 *   that keeps the pick riding every request — and the chain that carries the
 *   same button to the Canvas — cannot be silently dropped by a refactor.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { enhanceBlocker } from './enhanceGate.js';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), 'utf8');
const button = read('EnhancePromptButton.jsx');

test('the picked model rides the request spread-if-set — a no-pick request stays byte-identical', () => {
  assert.match(button, /\.\.\.\(model \? \{ ollama_model: model \} : \{\}\)/);
});

test('the pick is remembered under ONE key for every surface that mounts the button', () => {
  assert.match(button, /const MODEL_KEY = 'studioEnhanceModel'/);
  assert.match(button, /localStorage\.getItem\(MODEL_KEY\)/);
  assert.match(button, /localStorage\.setItem\(MODEL_KEY, m\)/);
});

test('the ⚙️ lists the actually-pulled models, and keeps an unlisted pick selectable', () => {
  assert.match(button, /'\/api\/local-llm\/models'/);
  assert.match(button, /\[model, \.\.\.models\]/);
});

test('a custom model lifts ONLY the default-model block — the Ollama blocks stay', () => {
  const down = { installed: true, reachable: false, vision_model_ready: false };
  assert.match(enhanceBlocker(down, { customModel: 'llama3.1:8b' }), /not running/);
  const noDefault = {
    installed: true, reachable: true, vision_model_ready: false, vision_model: 'x',
  };
  assert.match(enhanceBlocker(noDefault, {}), /not downloaded/);
  assert.equal(enhanceBlocker(noDefault, { customModel: 'llama3.1:8b' }), null);
});

test('the same button (gear included) reaches the Canvas through the shared panel chain', () => {
  // Canvas → RunSetupPanel → PromptField → EnhancePromptButton: the parity the
  // panels promise ("a setting added to the Test Studio appears here without
  // anyone touching this file"). Break a link and the ⚙️ silently becomes
  // Studio-only — exactly the drift this repo forbids.
  const promptField = read('PromptField.jsx');
  const runSetup = read('RunSetupPanel.jsx');
  const canvasPanel = read('../../canvas/CanvasGenerationPanel.jsx');
  assert.match(promptField, /<EnhancePromptButton/);
  assert.match(runSetup, /<PromptField/);
  assert.match(canvasPanel, /RunSetupPanel/);
});
