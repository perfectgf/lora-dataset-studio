/** A finished run writes its checkpoints to disk, but the panel never re-read the
 * list: both status polls only refreshed their own state, so a LoRA that had just
 * finished training stayed invisible until the browse filter changed or the page
 * was reloaded ("sometimes I have to refresh the page to see the LoRAs"). */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');

test('a run finishing on this dataset re-reads the checkpoints it produced', () => {
  assert.match(panel, /const runActiveHere = Boolean\(/);
  // both lanes count: a cloud run produces checkpoints exactly like a local one
  assert.match(panel, /status\.in_progress && status\.current\?\.dataset_id === ds\.currentId\)\s*\|\|\s*cloudActiveHere/);
  // acts on the FALLING edge only — "was active, now is not" means new files exist
  assert.match(panel, /if \(runWasActiveHere\.current && !runActiveHere\) \{/);
  assert.match(panel, /runWasActiveHere\.current = runActiveHere;/);
  assert.match(panel, /loadCheckpoints\(\);/);
});

test('the graph view is refreshed too, so the two views cannot disagree', () => {
  assert.match(panel,
    /if \(checkpointsView === 'graph' && checkpointManagerOpen\) loadDatasetGraph\(\);/);
});

test('the watcher depends on a boolean, not on the polled cloud object', () => {
  // The cloud poll returns a fresh object every 5s; depending on it directly would
  // re-run the effect constantly instead of only on the transition that matters.
  assert.match(panel, /\}, \[runActiveHere\]\);/);
});
