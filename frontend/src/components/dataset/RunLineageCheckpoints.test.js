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

test('continue-from-checkpoint is cloud-only by default and allows terminal (done OR failed) runs', () => {
  // The rule lives in the JSX-free helper (behaviour covered by
  // lineageContinue.test.js); the graph must USE it, not re-implement one.
  const rule = fs.readFileSync(new URL('./lineageContinue.js', import.meta.url), 'utf8');
  assert.match(graph, /import \{ canContinueFromCheckpoint \} from '\.\/lineageContinue\.js'/);
  assert.match(graph, /canContinueFromCheckpoint\(node, pill, \{/);
  // still cloud-only with a run id, unless the mount opted into 'any'
  assert.match(graph, /continueSource = 'cloud'/);
  assert.match(rule, /node\.source === 'cloud'/);
  assert.match(rule, /node\.run_id != null/);
  // a 'done' run always; a failed/stopped run only when THIS pill is present
  assert.match(rule, /node\.status === 'done'/);
  assert.match(rule, /'error', 'error_pod_kept', 'stopped', 'failed'/);
  assert.match(rule, /pill\?\.download_url/);
});

test('the Runs hub keeps the CLOUD gate — it passes no continueSource', () => {
  // The invariant: the hub's popover must not change. It wires the handler and
  // nothing else, so the graph falls back to its 'cloud' default.
  assert.match(cloud, /onContinueCheckpoint=\{continueFromCheckpoint\}/);
  assert.doesNotMatch(cloud, /continueSource/);
});

test('the dataset panel offers the SAME pill gesture through its local flow', () => {
  // continueSource="any" + a handler that reuses the existing local dialog —
  // no second continue path, no duplicated backend call.
  assert.match(panel, /continueSource="any"/);
  assert.match(panel, /onContinueCheckpoint=\{checkpointMatchesTraining/);
  assert.match(panel, /const continueFromGraphCheckpoint = \(node, pill\) =>/);
  assert.match(panel, /setContinueInitialStep\(step\);\s*setContinueOpen\(true\);/);
  assert.match(panel, /initialFromStep=\{continueInitialStep\}/);
  // the plain Continue button clears the pill pick, so it still opens on latest
  assert.match(panel, /setContinueInitialStep\(null\); setContinueOpen\(true\)/);
  // ONE continue call site (the guarded helper picks the lane's hook inside it),
  // never a second continue request assembled somewhere else in the panel
  assert.equal((panel.match(/payload\.extraSteps/g) || []).length, 1);
  assert.equal((panel.match(/await runConfirmableTrainingRequest\(/g) || []).length, 1);
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

test('the pill delete aims at what the pill SHOWS — deployed copy vs training save', () => {
  // The route is NOT hardcoded in the component: it comes from the target the
  // helper picks off the pill's deployed state (both routes live in the helper,
  // unit-tested in lineagePreview.test.js).
  assert.match(graph, /const target = checkpointDeleteTarget\(node, pill\);/);
  assert.match(graph, /postJson\(`\/api\/dataset\/\$\{datasetId\}\/\$\{target\.path\}`, target\.body\)/);
  const helpers = fs.readFileSync(new URL('./lineagePreview.js', import.meta.url), 'utf8');
  assert.match(helpers, /checkpointDeployed\(pill\)/);            // ONE source of truth for "deployed"
  assert.match(helpers, /path: 'train\/checkpoint\/delete'/);      // deployed → the ComfyUI copy
  assert.match(helpers, /path: 'train\/run-checkpoint\/delete'/);  // otherwise → the run's save
  // The BUTTON says which of the two it would delete, right now.
  assert.match(graph, /\{deleting \? 'Deleting…' : target\.label\}/);
  assert.match(graph, /title=\{target\.title\}/);
  // Confirmed, with the ★ best-settings pin reaching the confirmation text.
  assert.match(graph, /describeCheckpointDelete\(node, pill, \{ bestSettingsLora \}\)/);
  assert.match(graph, /if \(!window\.confirm\(message\)\) return;/);
  // postJson THROWS on 400/409 — the server's own message must be shown, not eaten.
  assert.match(graph, /catch \(e\) \{\s*toast\.error\(e\?\.message \|\| 'Delete failed'\);/);
  // The pill must stop lying: same refetch path the import success uses, so a
  // just-undeployed pill flips to "not deployed" (next click aims at the save).
  assert.match(graph, /Removed from ComfyUI \(training save kept\)[^]*?refetchTree\(\)/);
});

test('the lineage payload carries the deployed copy name from the testable map', () => {
  const svc = fs.readFileSync(new URL('../../../../backend/app/services/cloud_training.py', import.meta.url), 'utf8');
  // Same map that sets `testable` also names the deployed file — no second source.
  assert.match(svc, /_ck\['testable'\] = _step in _testable/);
  assert.match(svc, /_ck\['deployed_filename'\] = _deploy_names\.get\(/);
  // …resolved to the form the deployed-delete route whitelists.
  assert.match(svc, /def _deletable_deploy_names/);
  assert.match(svc, /lt\.list_imported_checkpoints\(cfg\.LOCAL_USER, dataset_id, family=family\)/);
});

test('the dataset panel feeds the graph the ★ best-settings pin', () => {
  assert.match(panel, /bestSettingsLora=\{ds\.data\?\.best_settings\?\.lora_filename \|\| null\}/);
  const tree = fs.readFileSync(new URL('./RunLineageTree.jsx', import.meta.url), 'utf8');
  assert.match(tree, /bestSettingsLora=\{bestSettingsLora\}/);
});
