import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

/* THE GUARD FOR A DEFECT THAT SHIPPED TWICE AND WAS NEVER SEEN.
 *
 * The Launch-all report renders `STEP_LABEL[s.step] || s.step`. That fallback is
 * the whole problem: a step added to PIPELINE_STEPS without a label does not
 * break anything, it just prints its own internal identifier and looks like a
 * feature nobody named. Two of the eight steps lived that way — `semantic_dedup`
 * and `framing` — until the maintainer read his own report and reported that he
 * "didn't really know what semantic_dedup was". He did: it is the ✂ button in the
 * pass row. The report was speaking a different language from the buttons.
 *
 * So this test reads the SERVER's tuple, not a copy of it. A copy would have to be
 * kept in sync by the same person who forgot the label, which is no guard at all.
 */

const SERVICE = new URL(
  '../../../../backend/app/services/image_bank_service.py', import.meta.url);
const REPORT = new URL('./PipelineReport.jsx', import.meta.url);

/** PIPELINE_STEPS = ('scan', 'auto_reject', …) — read straight out of the Python. */
function serverSteps() {
  const src = readFileSync(SERVICE, 'utf8');
  const m = src.match(/^PIPELINE_STEPS\s*=\s*\(([\s\S]*?)\)/m);
  assert.ok(m, 'PIPELINE_STEPS not found in image_bank_service.py — '
    + 'if it was renamed, this guard has to follow it, not be deleted.');
  return [...m[1].matchAll(/'([a-z_]+)'/g)].map((x) => x[1]);
}

/** STEP_LABEL = { scan: '…', … } — the map the report renders from. */
function reportLabels() {
  const src = readFileSync(REPORT, 'utf8');
  const m = src.match(/export const STEP_LABEL = \{([\s\S]*?)\n\}/);
  assert.ok(m, 'STEP_LABEL not found in PipelineReport.jsx');
  return [...m[1].matchAll(/([a-z_]+)\s*:\s*'/g)].map((x) => x[1]);
}

test('every pipeline step the server can report has a label the user recognises', () => {
  const steps = serverSteps();
  const labelled = new Set(reportLabels());
  assert.ok(steps.length >= 8, `expected the full step list, got ${steps.length}`);
  const missing = steps.filter((s) => !labelled.has(s));
  assert.deepEqual(missing, [],
    `these steps would render as raw identifiers in the report: ${missing.join(', ')}. `
    + 'Add them to STEP_LABEL with the words on their BUTTON.');
});

test('semantic and framing steps carry their button wording, not raw identifiers', () => {
  const src = readFileSync(REPORT, 'utf8');
  // The exact strings the pass row and the Launch all dialog use. A report that
  // paraphrases a button is the same defect one notch quieter.
  assert.match(src, /semantic_dedup: '✂ Find crops & variants'/);
  assert.match(src, /semantic_index: 'Build semantic index'/);
  assert.match(src, /framing: 'Classify framing'/);
});

test('no label is left blank', () => {
  const src = readFileSync(REPORT, 'utf8');
  const m = src.match(/export const STEP_LABEL = \{([\s\S]*?)\n\}/);
  assert.ok(!/:\s*''/.test(m[1]), 'an empty label reads exactly like a missing one');
});
