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

test('continue-from-checkpoint is cloud-only and allows terminal (done OR failed) runs', () => {
  // still cloud-only with a run id
  assert.match(graph, /node\.source === 'cloud' && node\.run_id != null/);
  // a 'done' run always; a failed/stopped run only when THIS pill is present
  assert.match(graph, /node\.status === 'done'/);
  assert.match(graph, /'error', 'error_pod_kept', 'stopped', 'failed'/);
  assert.match(graph, /pill\?\.download_url/);
});

test('the LoRA manager opens the same graph component for the whole dataset', () => {
  assert.match(panel, /import RunLineageGraph from '\.\/RunLineageGraph'/);
  assert.match(panel, /◉ Graph/);
  assert.match(panel, /train\/lineage\?/);
  assert.match(panel, /<RunLineageGraph tree=\{datasetGraph\.tree\}/);
});

test('the manager opens on the GRAPH by default, with a persisted List toggle', () => {
  // Default view is the graph — the showcase surface; the list stays available.
  assert.match(panel, /localStorage\.getItem\('lds\.checkpointsView'\) === 'list' \? 'list' : 'graph'/);
  assert.match(panel, /setItem\('lds\.checkpointsView'/);
  // The graph and the flat list are each gated on the current view.
  assert.match(panel, /checkpointsView === 'graph' &&/);
  assert.match(panel, /checkpointsView === 'list' &&/);
  // ☰ List toggle exists alongside ◉ Graph.
  assert.match(panel, /☰ List/);
});

test('the dataset graph renders INLINE inside the manager (no body-portal modal)', () => {
  // The graph now lives inside the CheckpointPortal'd manager, which itself
  // renders into the VISIBLE sidebar host — so it never inherits the hidden
  // home container's display:none that forced the old modal to portal to <body>.
  assert.doesNotMatch(panel, /aria-label="Dataset run graph"/);
  assert.doesNotMatch(panel, /datasetGraph && createPortal\(/);
  // createPortal stays imported — CheckpointPortal still uses it.
  assert.match(panel, /import \{ createPortal \} from 'react-dom'/);
});

test('a pill can be imported straight from the graph, deployed pills say so', () => {
  // 📦 Import → loras/<family> uses the CSRF-safe postJson and the list's exact
  // payload (via lineageImportPayload); an already-deployed pill shows ✓ Deployed.
  assert.match(graph, /lineageImportPayload/);
  assert.match(graph, /train\/import/);
  assert.match(graph, /postJson\(`\/api\/dataset\/\$\{datasetId\}\/train\/import`/);
  assert.match(graph, /checkpointDeployed\(openCk\.pill\)/);
  assert.match(graph, /✓ Deployed/);
  // after a successful import the lineage is refetched so the pill flips testable
  assert.match(graph, /refetchTree/);
});

test('a preview thumbnail opens LARGE in a lightbox, distinct from the popover', () => {
  assert.match(graph, /onZoomPreview/);
  // clicking the thumbnail must NOT open the popover (its own action)
  assert.match(graph, /e\.stopPropagation\(\); onZoomPreview/);
  assert.match(graph, /bigPreview/);
});

test('a persisted 🔍 Big-previews mode enlarges the generated tiles', () => {
  // Toggle + persistence in the graph, geometry threaded to the layout.
  assert.match(graph, /🔍 Big previews/);
  assert.match(graph, /localStorage\.getItem\('lds\.graphBigPreviews'\)/);
  assert.match(graph, /setItem\('lds\.graphBigPreviews'/);
  assert.match(graph, /buildLineageGraph\(shownTree, \{ bigPreviews \}\)/);
  // The pill sizes off the layout's per-mode geometry (pill.w/pill.h), not a const.
  assert.match(graph, /width: pill\.w, height: pill\.h/);
});

test('the ◉ Graph button is the prominent (accent) view control', () => {
  // On the Runs hub the graph toggle wears the indigo accent, not a bare grey.
  assert.match(cloud, /border-indigo-400\/40 bg-indigo-500\/10 text-indigo-200/);
});
