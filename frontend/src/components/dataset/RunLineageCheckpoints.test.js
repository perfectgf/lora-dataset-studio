import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const graph = fs.readFileSync(new URL('./RunLineageGraph.jsx', import.meta.url), 'utf8');
const cloud = fs.readFileSync(new URL('../../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
// The card, the pill and the edges are no longer private to this component —
// they were extracted so the LoRA Canvas draws with the SAME ones (a lookalike
// would drift the first time either surface was tweaked).
const nodes = fs.readFileSync(new URL('./lineageNodes.jsx', import.meta.url), 'utf8');
const edges = fs.readFileSync(new URL('./lineageEdges.jsx', import.meta.url), 'utf8');
const canvas = fs.readFileSync(new URL('../canvas/LineageCanvas.jsx', import.meta.url), 'utf8');
// …and so are the ACTIONS. The popover (⬇ download, ▶ continue, 📦 deploy,
// ⏏ undeploy, 🗑 delete, ⓘ details) was ~90 lines inlined in the graph, which is
// why the canvas had none at all; it now lives in one component on one pure
// rule set, driven by one hook.
const popover = fs.readFileSync(new URL('./CheckpointActionsPopover.jsx', import.meta.url), 'utf8');
const popoverRules = fs.readFileSync(new URL('./checkpointPopover.js', import.meta.url), 'utf8');
const actionsHook = fs.readFileSync(new URL('../../hooks/useCheckpointActions.js', import.meta.url), 'utf8');

test('the drawing is shared with the canvas, not duplicated', () => {
  // Both surfaces IMPORT the card/pill/edges; neither declares its own.
  assert.match(graph, /import \{ GraphCard, CheckpointPill \} from '\.\/lineageNodes'/);
  assert.match(graph, /import \{ LineageEdgeDefs, LineageEdges \} from '\.\/lineageEdges'/);
  assert.match(canvas, /import \{ GraphCard, CheckpointPill \} from '\.\.\/dataset\/lineageNodes'/);
  assert.match(canvas, /import \{ LineageEdgeDefs, LineageEdges \} from '\.\.\/dataset\/lineageEdges'/);
  for (const src of [graph, canvas]) {
    assert.doesNotMatch(src, /function GraphCard\(/);
    assert.doesNotMatch(src, /function CheckpointPill\(/);
    assert.doesNotMatch(src, /linearGradient id="lds-edge-/);
  }
  assert.match(nodes, /export function GraphCard\(/);
  assert.match(nodes, /export function CheckpointPill\(/);
  assert.match(edges, /export function LineageEdgeDefs\(/);
  assert.match(edges, /export function LineageEdges\(/);
});

test('the canvas GENERATES through the Test Studio, it does not grow a second one', () => {
  // Slice 3 opened this up (the guard used to lock the canvas OUT of generation
  // so the earlier slices could not quietly grow half a feature). What is locked
  // now is the shape of the opening: the canvas mounts the Studio's panel and the
  // Studio's hooks. A canvas that re-declared a prompt field, a seed control or a
  // launch call would be the drift this whole design exists to prevent.
  const panel = fs.readFileSync(new URL('../canvas/CanvasGenerationPanel.jsx', import.meta.url), 'utf8');
  assert.match(panel, /import RunSetupPanel from '\.\.\/dataset\/studio\/RunSetupPanel'/);
  assert.match(panel, /useStudioForm/);
  assert.match(panel, /useCanvasStudio/);
  assert.doesNotMatch(panel, /<textarea/);          // the prompt field is PromptField's
  assert.doesNotMatch(panel, /rollSeed|nextSeed\(\)/);
  // The canvas ticks pills; the in-card graph keeps its own selection intact.
  assert.match(canvas, /onToggleSelect=\{\(\) => onTogglePick\(/);
  assert.match(graph, /onToggleSelect=\{\(pill\) => toggleCk\(/);
});

test('the canvas launch goes through ONE call site, so no setting can be dropped', () => {
  const hook = fs.readFileSync(new URL('../../hooks/useCanvasStudio.js', import.meta.url), 'utf8');
  const setup = fs.readFileSync(new URL('./studio/RunSetupPanel.jsx', import.meta.url), 'utf8');
  // RunSetupPanel owns genSettings in local state and passes it to studio.launch.
  // The canvas therefore swaps studio.launch — NOT the onLaunch handler, which
  // would have silently lost the global generation settings from a canvas run.
  assert.match(setup, /const onLaunch = async \(\) => \{/);
  assert.match(setup, /studio\.launch\(/);
  assert.doesNotMatch(setup, /onLaunchOverride/);
  assert.match(hook, /genSettings = \{\}/);
  assert.match(hook, /\.\.\.genSettings/);
  assert.match(hook, /\/api\/train\/canvas\/generate/);
});

test('the canvas refuses mixed families and deploys before generating — in the pure layer', () => {
  const rules = fs.readFileSync(new URL('../../utils/canvasGeneration.js', import.meta.url), 'utf8');
  // Both decisions are arithmetic a test can check without a browser (behaviour
  // covered in canvasGeneration.test.js); the component must USE them.
  assert.match(canvas, /from '\.\.\/\.\.\/utils\/canvasGeneration'/);
  assert.match(canvas, /describeCanvasLaunch\(picks\)/);
  assert.match(rules, /cannot run together/);
  assert.match(rules, /different base models/);
  assert.match(rules, /Deploy \$\{n\} checkpoint/);
  // A failed deploy ABORTS: half a comparison answers a different question.
  assert.match(canvas, /Deploy failed — nothing was generated/);
});

test('the canvas moves cards through the PURE placement layer, not by hand', () => {
  // The whole point of the layer is that "a new run moves nothing" is arithmetic
  // a test can check without a browser. A component that recomputed positions
  // inline would put that rule back out of reach.
  assert.match(canvas, /from '\.\.\/\.\.\/utils\/canvasPlacement'/);
  assert.match(canvas, /applyPlacement\(/);
  assert.match(canvas, /pinSnapshot\(/);
  assert.doesNotMatch(canvas, /function applyPlacement|function pinSnapshot/);
});

test('the canvas disambiguates the touch gesture with a long press', () => {
  // Dragging a card and panning the board are the same finger. Without an
  // explicit delay one of the two becomes impossible on a phone.
  assert.match(canvas, /LONG_PRESS_MS/);
  assert.match(canvas, /pointerType !== 'touch'/);
  assert.match(canvas, /cancelLongPress\(\)/);
});

test('✦ Tidy up exists and is wired to the page, not to a local reset', () => {
  const page = fs.readFileSync(new URL('../../pages/CanvasPage.jsx', import.meta.url), 'utf8');
  assert.match(canvas, /Tidy up/);
  // Through `handleTidyUp` since the frame stopped resizing itself: the button
  // arms a re-fit and then delegates, unchanged, to the page's handler. What is
  // pinned is that the page still owns the RESET — the local wrapper adds a
  // view decision on top of it and nothing else.
  assert.match(canvas, /onClick=\{handleTidyUp\}/);
  assert.match(canvas, /const handleTidyUp = useCallback\(\(\) => \{[^]{0,200}onTidyUp\?\.\(\);/);
  // It clears the SERVER's memory of the lane; a client-only reset would come
  // back on the next reload.
  assert.match(page, /del\(`\/api\/dataset\/\$\{id\}\/canvas\/positions`\)/);
});

test('the checkpoint popover is ONE component, hosted by BOTH surfaces', () => {
  // The rule this whole extraction exists for. A second popover would agree with
  // the first today and drift on the first change — exactly what happened to the
  // canvas, which shipped with no actions at all rather than a copy.
  assert.match(graph, /import CheckpointActionsPopover from '\.\/CheckpointActionsPopover'/);
  assert.match(canvas, /import CheckpointActionsPopover from '\.\.\/dataset\/CheckpointActionsPopover'/);
  assert.match(graph, /<CheckpointActionsPopover/);
  assert.match(canvas, /<CheckpointActionsPopover/);
  assert.match(popover, /export default function CheckpointActionsPopover\(/);
  for (const src of [graph, canvas]) {
    // No host may re-declare the popover, its rows, or its labels.
    assert.doesNotMatch(src, /function CheckpointActionsPopover\(/);
    assert.doesNotMatch(src, /<span aria-hidden>▶<\/span> Continue from here/);
    assert.doesNotMatch(src, /Deploy → \$\{/);
    assert.doesNotMatch(src, /✓<\/span> Deployed/);
    /* ⏏ ALONE is not the invariant, and asserting on it was too wide. The
       glyph is this app's word for "undeploy" wherever it appears, and the
       canvas has its own install-wide ⏏ Undeploy… — the panel listing every
       LoRA the app pushed into ComfyUI, a different action on a different
       scope, which moved onto the board's ⋯ shelf when the page header
       stopped being drawn on a phone. What may not be re-declared here is
       this popover's undeploy ROW, so that is what is pinned. */
    assert.doesNotMatch(src, /⏏<\/span> \{deleting/);
    assert.doesNotMatch(src, /className="lds-ck-popover/);
  }
  // download reuses the server-provided url — no url built in the component
  assert.match(popover, /href=\{a\.download\.url\} download/);
  // the popover uses the OPAQUE overlay surface, never the see-through one
  assert.match(popover, /lds-ck-popover[^\n]*bg-surface-overlay/);
});

test('both surfaces place the popover through the PURE geometry, and it never leaves the frame', () => {
  // Two spaces, one module: world units inside the graph's <svg>, screen pixels
  // over the zoomed board. Behaviour (flip above, clamp, narrow on a 400-px
  // screen) is covered in checkpointPopover.test.js — the hosts must USE it.
  assert.match(graph, /checkpointPopoverPlacement\(openCk\.pill, g\)/);
  assert.match(canvas, /clampPopoverToViewport\(openCk\.anchor/);
  assert.match(popoverRules, /export function checkpointPopoverPlacement\(/);
  assert.match(popoverRules, /export function clampPopoverToViewport\(/);
  for (const src of [graph, canvas]) assert.doesNotMatch(src, /const POP_W = |POP_H > g\.height/);
});

test('every popover action is live, or stated with its reason — never a silent dead button', () => {
  // The "works everywhere" rule, decided in the pure layer so it is testable:
  // no file on disk, an unlinked cloud run, a run still training, a host with no
  // continue flow. The component only renders the verdict.
  assert.match(popover, /checkpointActionModel\(node, pill, \{/);
  assert.match(popoverRules, /export function deployRefusal\(/);
  assert.match(popoverRules, /export function downloadRefusal\(/);
  assert.match(popoverRules, /cloud run is not linked/);
  // A refused action renders as TEXT, not as a greyed button inviting a click.
  assert.match(popover, /a\.download\.reason/);
  assert.match(popover, /a\.deploy\.reason/);
  assert.match(popover, /a\.continue\.reason/);
  // `continueReason` remains the escape hatch for a host with no resume flow —
  // the row is then a sentence rather than a dead button. No host uses it today:
  // the canvas grew its own flow (utils/canvasContinue.js) and now passes a
  // handler, so its ▶ Continue is live. The PROP must survive the change.
  assert.match(popoverRules, /continueReason \? \{ reason: continueReason \} : null/);
  assert.match(canvas, /onContinue=\{handleContinueCheckpoint\}/);
});

test('the graph opens for any run with a checkpoint, not only 2+ run lineages', () => {
  // button + body both gate on lineage OR a saved checkpoint
  assert.match(cloud, /run\.lineage\s*\|\|\s*run\.checkpoint_ready/);
  // single-run graph is labelled ◉ Graph, a real lineage stays 🌳 Lineage
  assert.match(cloud, /run\.lineage \? '🌳 Lineage' : '◉ Graph'/);
});

test('continue-from-checkpoint is cloud-only by default and allows terminal (done OR failed) runs', () => {
  // The rule lives in the JSX-free helper (behaviour covered by
  // lineageContinue.test.js); the popover must USE it, not re-implement one.
  const rule = fs.readFileSync(new URL('./lineageContinue.js', import.meta.url), 'utf8');
  assert.match(popoverRules, /import \{ canContinueFromCheckpoint \} from '\.\/lineageContinue\.js'/);
  assert.match(popoverRules, /canContinueFromCheckpoint\(node, pill, \{/);
  // still cloud-only with a run id, unless the mount opted into 'any'
  assert.match(graph, /continueSource = 'cloud'/);
  assert.match(popoverRules, /continueSource = 'cloud'/);
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
  // (a fresh open also clears the last refusal — the dialog now keeps it)
  assert.match(panel, /setContinueInitialStep\(step\);\s*setContinueError\(null\);\s*setContinueOpen\(true\);/);
  assert.match(panel, /initialFromStep=\{continueInitialStep\}/);
  // the plain Continue button clears the pill pick, so it still opens on latest
  assert.match(panel, /setContinueInitialStep\(null\); setContinueError\(null\); setContinueOpen\(true\)/);
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

test('a pill can be deployed straight from EITHER surface, deployed pills say so', () => {
  // 📦 Deploy → loras/<family> uses the CSRF-safe postJson and the list's exact
  // payload (via lineageImportPayload); an already-deployed pill shows ✓ Deployed.
  // One implementation, in the hook — the canvas is not a second one.
  assert.match(actionsHook, /lineageImportPayload/);
  assert.match(actionsHook, /postJson\(`\/api\/dataset\/\$\{datasetId\}\/train\/import`, body\)/);
  assert.match(popoverRules, /checkpointDeployed\(pill\)/);
  assert.match(popover, /✓<\/span> Deployed/);
  for (const src of [graph, canvas]) {
    assert.match(src, /useCheckpointActions\(\{/);
    // The POPOVER's deploy goes through the hook, never through a route the host
    // wrote itself. (The canvas keeps one other import call: the bulk
    // deploy-then-generate of the ticked picks, which is a different gesture.)
    assert.match(src, /deployCheckpoint\(/);
    assert.doesNotMatch(src, /postJson\([^\n]*train\/import`, body\)/);
  }
  // after a successful deploy the lineage is re-read so the pill flips testable
  assert.match(actionsHook, /await onChanged\?\.\(datasetId\)/);
  assert.match(graph, /refetchTree/);
  assert.match(canvas, /onRefetchDataset\?\.\(datasetId\)/);
});

test('the compact pill COUNTS results instead of showing an illegible thumbnail', () => {
  // A 14-px image on a 60×20 pill is a coloured smudge, not an image: it says
  // nothing about the framing or the face while eating the label's width. The
  // pill signals that results exist and how many; the gallery shows them.
  assert.doesNotMatch(nodes, /width=\{14\} height=\{14\}/);
  assert.doesNotMatch(nodes, /Click to view this preview large/);
  assert.match(nodes, /const count = Number\(preview\?\.count\) \|\| \(preview\?\.url \? 1 : 0\)/);
  assert.match(nodes, /const resultsChip = \(inline\) =>/);
  // The chip is INSIDE the pill. The old badge hung at -6 px off the corner, so
  // two neighbouring pills' badges overlapped each other at 100 % zoom.
  assert.doesNotMatch(nodes, /right: -6, bottom: -6/);
  // It always leads somewhere: the gallery when the host has one, else the
  // lightbox on the preview it holds.
  assert.match(nodes, /if \(typeof onOpenGallery === 'function'\) onOpenGallery\(pill\);/);
  assert.match(nodes, /else if \(preview\?\.url\) onZoomPreview\?\./);
  // Both surfaces now wire the gallery, so the count is never a dead chip.
  assert.match(graph, /onOpenGallery=\{\(pill\) => setGallery\(/);
  // On the board the handler is HOISTED out of the JSX (a new arrow per render
  // would disable the lanes' memo boundary and make every pan frame re-render
  // the whole board), so the wiring is checked where it now lives.
  assert.match(canvas, /const openGallery = useCallback\(\(recordId, step\) => setGallery\(/);
  assert.match(canvas, /onOpenGallery=\{openGallery\}/);
  // The big-preview tile KEEPS its image — that one is sized to be judged.
  assert.match(nodes, /h-full w-full cursor-zoom-in object-cover/);
});

test('a preview opens LARGE in a lightbox — shared, so the canvas is not a dead click', () => {
  // On the board the thumbnail was clickable and did nothing: the host passed no
  // handler. The lightbox is one component now, mounted by both.
  const lightbox = fs.readFileSync(new URL('./PreviewLightbox.jsx', import.meta.url), 'utf8');
  assert.match(lightbox, /export default function PreviewLightbox\(/);
  assert.match(nodes, /e\.stopPropagation\(\); onZoomPreview/);
  for (const src of [graph, canvas]) {
    assert.match(src, /<PreviewLightbox target=\{bigPreview\}/);
    assert.doesNotMatch(src, /aria-modal="true" aria-label=\{`Preview at step/);
  }
  assert.match(canvas, /onZoomPreview=\{zoomPreview\}/);
});

test('a persisted 🔍 Big-previews mode enlarges the generated tiles', () => {
  // Toggle + persistence in the graph, geometry threaded to the layout.
  assert.match(graph, /🔍 Big previews/);
  assert.match(graph, /localStorage\.getItem\('lds\.graphBigPreviews'\)/);
  assert.match(graph, /setItem\('lds\.graphBigPreviews'/);
  assert.match(graph, /buildLineageGraph\(shownTree, \{ bigPreviews \}\)/);
  // The pill sizes off the layout's per-mode geometry (pill.w/pill.h), not a const.
  assert.match(nodes, /width: pill\.w, height: pill\.h/);
});

test('the ◉ Graph button is the prominent (accent) view control', () => {
  // On the Runs hub the graph toggle wears the indigo accent, not a bare grey.
  assert.match(cloud, /border-indigo-400\/40 bg-indigo-500\/10 text-indigo-200/);
});

test('the pill delete aims at what the pill SHOWS — deployed copy vs training save', () => {
  // The route is NOT hardcoded in a component: it comes from the target the
  // helper picks off the pill's deployed state (both routes live in the helper,
  // unit-tested in lineagePreview.test.js), and ONE hook posts it for both
  // surfaces.
  assert.match(actionsHook, /const target = checkpointDeleteTarget\(node, pill\);/);
  assert.match(actionsHook, /postJson\(`\/api\/dataset\/\$\{datasetId\}\/\$\{target\.path\}`, target\.body\)/);
  const helpers = fs.readFileSync(new URL('./lineagePreview.js', import.meta.url), 'utf8');
  assert.match(helpers, /checkpointDeployed\(pill\)/);            // ONE source of truth for "deployed"
  assert.match(helpers, /path: 'train\/checkpoint\/delete'/);      // deployed → the ComfyUI copy
  assert.match(helpers, /path: 'train\/run-checkpoint\/delete'/);  // otherwise → the run's save
  // The BUTTON says which of the two it would delete, right now.
  assert.match(popover, /\{deleting \? 'Deleting…' : a\.del\.label\}/);
  assert.match(popover, /title=\{a\.del\.title\}/);
  // Confirmed, with the ★ best-settings pin reaching the confirmation text.
  assert.match(actionsHook, /describeCheckpointDelete\(node, pill, \{ bestSettingsLora \}\)/);
  assert.match(actionsHook, /if \(message && !window\.confirm\(message\)\) return false;/);
  assert.match(graph, /useCheckpointActions\(\{[^]*?bestSettingsLora/);
  // postJson THROWS on 400/409 — the server's own message must be shown, not eaten.
  assert.match(actionsHook, /catch \(e\) \{\s*toast\.error\(e\?\.message \|\| 'Delete failed'\);/);
  // The pill must stop lying: the same re-read the deploy success uses, so a
  // just-undeployed pill flips to "not deployed" (next click aims at the save).
  assert.match(actionsHook, /Undeployed from ComfyUI[^]*?onChanged\?\.\(datasetId\)/);
});

test('undeploy is EXPLICIT and symmetric with deploy — and never confusable with delete', () => {
  const helpers = fs.readFileSync(new URL('./lineagePreview.js', import.meta.url), 'utf8');
  // ⏏ Undeploy sits next to "✓ Deployed", where 📦 Deploy sits when it isn't —
  // no longer only reachable through the retreat row.
  assert.match(popover, /✓<\/span> Deployed/);
  assert.match(popover, /⏏<\/span> \{deleting \? 'Undeploying…' : a\.undeploy\.label\}/);
  assert.match(popoverRules, /undeploy: checkpointUndeployAction\(node, pill\)/);
  // It is DERIVED from the single delete target, so its label can never name one
  // file while the click posts another (the invariant of this popover).
  assert.match(helpers, /export function checkpointUndeployAction[^]*?checkpointDeleteTarget\(node, pill\)/);
  assert.match(helpers, /target\.kind !== 'deployed'\) return null;/);
  // The 🗑 retreat row is reserved for the one destructive action: the save.
  assert.match(popoverRules, /del: target && target\.kind === 'save' \? target : null/);
  // Undeploy is presented as REVERSIBLE — the save survives and can be re-deployed.
  assert.match(helpers, /Reversible: the training save is kept/);
});

test('the detail drawer opens from ⓘ Details, not from touching a card', () => {
  // It used to spring open on any click on the board, which turned a glance into
  // a panel to dismiss. It is now one of the popover's actions, filed with the rest.
  assert.match(popover, /<span aria-hidden>ⓘ<\/span> Details/);
  assert.match(popover, /onDetails\(node\)/);
  // On the canvas a card click opens the RUN GALLERY — its images by step, its
  // notes and its settings. It used to open the popover with a single ⓘ Details
  // row, which is how a card click read as doing nothing; that row's content now
  // lives IN the panel, and the drawer still waits to be asked (from the panel,
  // or from a checkpoint pill's popover).
  assert.match(canvas, /setGallery\(runGalleryTarget\(/);
  assert.doesNotMatch(canvas, /onOpenActions\(lane \|\| null, node, null, e\)/);
  assert.match(canvas, /onDetails=\{\(node\) => setOpenNode\(node\)\}/);
  assert.doesNotMatch(canvas, /if \(e && e\.shiftKey\)[^]*?\n    setOpenNode\(node\);/);
  // The pill body opens them too — it used to open the drawer instead.
  assert.match(canvas, /onOpen=\{\(pill, e\) => onOpenActions\(lane, n\.node, pill, e\)\}/);
});

test('📤 Civitai is a row of the shared popover, and BOTH hosts open the same dialog from it', () => {
  // The row lives in the popover (one set of actions, one set of labels)…
  assert.match(popover, /data-testid="checkpoint-civitai"/);
  assert.match(popover, /onPublish\(node, pill\)/);
  // …and never on a run card: a page is made from ONE save.
  assert.match(popover, /typeof onPublish === 'function' && !a\.isRun/);
  // Each host hands the checkpoint to the shared dialog — a host that forgot
  // would show the popover without the row, silently.
  assert.match(graph, /onPublish=\{handlePublish\}/);
  assert.match(canvas, /onPublish=\{\(node, pill\) => setPublishCk\(\{ node, pill \}\)\}/);
  assert.match(graph, /<CivitaiPublishModal context=\{\{ kind: 'checkpoint'/);
  assert.match(canvas, /<CivitaiPublishModal context=\{\{ kind: 'checkpoint'/);
});

test('a generation launched from the board is the BOARD’s state, recoverable', () => {
  // Reported from real use: closing the settings panel (or leaving the page) lost
  // the run in flight, because the run id lived in the panel's own hook.
  const runHook = fs.readFileSync(new URL('../../hooks/useCanvasRun.js', import.meta.url), 'utf8');
  const studio = fs.readFileSync(new URL('../../hooks/useCanvasStudio.js', import.meta.url), 'utf8');
  const tracker = fs.readFileSync(new URL('../canvas/CanvasRunTracker.jsx', import.meta.url), 'utf8');
  // The panel no longer owns the run: it receives the board's tracker.
  assert.doesNotMatch(studio, /useStudioRun\(/);
  assert.match(studio, /const runId = tracker\?\.runId \?\? null;/);
  assert.match(studio, /tracker\?\.adopt\?\.\(d\.run_id/);
  // …and the board remembers it across a reload, with the checkpoints it hit.
  assert.match(runHook, /readCanvasRun|writeCanvasRun/);
  assert.match(canvas, /const tracker = useCanvasRun\(\);/);
  assert.match(canvas, /<CanvasRunTracker/);
  // Progress is visible ON the board, in the Studio's own words, with its Stop.
  assert.match(tracker, /describeCanvasRun\(run\)/);
  assert.match(tracker, /Stop \(resumable\)/);
});

test('a finished generation SAYS where the images went, and the board re-reads itself', () => {
  // The other half of the same report: the images landed in the checkpoint's
  // gallery and nothing said so — and the board did not even refresh, so the
  // × N badge only appeared after a full reload.
  const rules = fs.readFileSync(new URL('../../utils/canvasRunResults.js', import.meta.url), 'utf8');
  const tracker = fs.readFileSync(new URL('../canvas/CanvasRunTracker.jsx', import.meta.url), 'utf8');
  assert.match(rules, /export function readyImageCount\(/);
  assert.match(rules, /export function canvasRunDatasetIds\(/);
  // Each finished run names its checkpoints, and each one opens its gallery.
  assert.match(tracker, /onOpenResult\(t\)/);
  assert.match(canvas, /onOpenResult=\{\(t\) => setGallery\(\{ recordId: t\.recordId, step: t\.step \}\)\}/);
  // New images ⇒ re-read the lanes they belong to, so the pills show them.
  assert.match(canvas, /const n = readyImageCount\(tracker\.run\.data\);/);
  // …through a THROTTLE (the poller ticks every 3 s for minutes; a full lineage
  // re-read per tick per lane is the board's most expensive background habit).
  // The targets are read from a ref because a trailing call may fire from a
  // timer armed several polls earlier.
  assert.match(canvas, /for \(const id of canvasRunDatasetIds\(trackerTargetsRef\.current\)\)/);
  assert.match(canvas, /REFETCH_MIN_MS/);
});

test('the lineage payload carries the deployed copy name from the testable map', () => {
  const svc = fs.readFileSync(new URL('../../../../backend/app/services/cloud_training.py', import.meta.url), 'utf8');
  // Same map that sets `testable` also names the deployed file — no second
  // source. Both now live in ONE annotator, shared with the Checkpoints panel,
  // so the two surfaces can never disagree on what is deployed.
  assert.match(svc, /def annotate_deployed_checkpoints\(/);
  assert.match(svc, /ck\['testable'\] = step in testable/);
  assert.match(svc, /ck\['deployed_filename'\] = names\.get\(/);
  assert.match(svc, /annotate_deployed_checkpoints\(rec\.dataset_id, rec\.family,/);
  // …resolved to the form the deployed-delete route whitelists.
  assert.match(svc, /def _deletable_deploy_names/);
  assert.match(svc, /lt\.list_imported_checkpoints\(cfg\.LOCAL_USER, dataset_id, family=family\)/);
});

test('the dataset panel feeds the graph the ★ best-settings pin', () => {
  // The FLATTENED list, not `best_settings.lora_filename`: the pin has been stored
  // per family for a while now, so that key only ever matched the legacy flat
  // shape and the ⚠ warning was silently dead on every modern pin.
  assert.match(panel, /bestSettingsLora=\{ds\.data\?\.best_settings_loras \|\| null\}/);
  assert.match(panel, /\(ds\.data\?\.best_settings_loras \|\| \[\]\)/);
  const tree = fs.readFileSync(new URL('./RunLineageTree.jsx', import.meta.url), 'utf8');
  assert.match(tree, /bestSettingsLora=\{bestSettingsLora\}/);
});

test('the canvas warns before deleting a ★ pinned checkpoint, with ITS lane pin', () => {
  // The board's dataset index publishes the pin…
  const svc = fs.readFileSync(new URL('../../../../backend/app/services/cloud_training.py', import.meta.url), 'utf8');
  assert.match(svc, /'best_settings_loras': studio\.best_settings_lora_filenames\(ds\)/);
  // …the page puts it on the lane…
  const page = fs.readFileSync(new URL('../../pages/CanvasPage.jsx', import.meta.url), 'utf8');
  assert.match(page, /bestSettingsLoras: row\?\.best_settings_loras \|\| \[\]/);
  // …and the board hands the hook the pin of the lane whose popover is OPEN,
  // never another dataset's.
  const canvasSrc = fs.readFileSync(new URL('../canvas/LineageCanvas.jsx', import.meta.url), 'utf8');
  assert.match(canvasSrc, /bestSettingsLora: openCk\?\.lane\?\.bestSettingsLoras \|\| null/);
});
