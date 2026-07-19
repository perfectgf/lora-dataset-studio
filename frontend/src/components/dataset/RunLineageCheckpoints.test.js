import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const graph = fs.readFileSync(new URL('./RunLineageGraph.jsx', import.meta.url), 'utf8');
const cloud = fs.readFileSync(new URL('../../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');

test('the graph draws checkpoint pills with a download link (reused endpoint)', () => {
  assert.match(graph, /CheckpointPill/);
  // download reuses the server-provided url — no url built in the component
  assert.match(graph, /href=\{openCk\.pill\.download_url\}/);
  // the popover uses the OPAQUE overlay surface, never the see-through surface
  assert.match(graph, /bg-surface-overlay/);
  assert.doesNotMatch(graph, /lds-ck-popover[^]*bg-surface\b(?!-overlay)/);
});

test('the graph opens for any run with a checkpoint, not only 2+ run lineages', () => {
  // button + body both gate on lineage OR a saved checkpoint
  assert.match(cloud, /run\.lineage\s*\|\|\s*run\.checkpoint_ready/);
  // single-run graph is labelled ◉ Graph, a real lineage stays 🌳 Lineage
  assert.match(cloud, /run\.lineage \? '🌳 Lineage' : '◉ Graph'/);
});

test('continue-from-checkpoint is cloud-only, mirroring the per-run button', () => {
  assert.match(graph, /node\.source === 'cloud' && node\.run_id != null && node\.status === 'done'/);
});

test('the LoRA manager opens the same graph component for the whole dataset', () => {
  assert.match(panel, /import RunLineageGraph from '\.\/RunLineageGraph'/);
  assert.match(panel, /◉ Graph/);
  assert.match(panel, /train\/lineage\?/);
  assert.match(panel, /<RunLineageGraph tree=\{datasetGraph\.tree\}/);
});

test('the ◉ Graph modal portals to <body> so the hidden section never eats it', () => {
  // The Checkpoints & LoRAs manager portals into its OWN sidebar section; when
  // that section is active, TrainingPanel's home container carries `hidden`
  // (display:none). A modal rendered inline there inherits display:none and
  // never shows (fixed positioning does NOT escape an ancestor's display:none) —
  // the button looked dead. The dataset-graph dialog must therefore be portaled
  // to document.body, exactly like CaptionEditorDialog.
  assert.match(panel, /datasetGraph && createPortal\(/);
  assert.match(
    panel,
    /aria-label="Dataset run graph"[\s\S]*?\),\s*document\.body\)}/);
  assert.match(panel, /import \{ createPortal \} from 'react-dom'/);
});
