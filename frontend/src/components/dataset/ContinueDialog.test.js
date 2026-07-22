import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const dialog = fs.readFileSync(new URL('./ContinueDialog.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const cloud = fs.readFileSync(new URL('../../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');

test('the dialog resolves a flexible-continue payload (steps, checkpoint, overrides)', () => {
  // fromStep is null only when the newest checkpoint is chosen — the in-place resume.
  assert.match(dialog, /fromStep:\s*isEarlier\s*\?\s*fromStep\s*:\s*null/);
  assert.match(dialog, /extraSteps:\s*extraNum/);
  assert.match(dialog, /overrides:\s*Object\.keys\(overrides\)\.length/);
  // safe subset only — cadence + preview prompts, never rank/base/optimizer.
  assert.match(dialog, /overrides\.save_every/);
  assert.match(dialog, /overrides\.sample_every/);
  assert.match(dialog, /overrides\.sample_prompts/);
  // its own help topic (registered in helpRegistry)
  assert.match(dialog, /topic="continue-training"/);
});

test('both hubs open the shared ContinueDialog', () => {
  assert.match(panel, /import ContinueDialog from '\.\/ContinueDialog'/);
  assert.match(panel, /<ContinueDialog/);
  assert.match(cloud, /import ContinueDialog from '\.\.\/components\/dataset\/ContinueDialog'/);
  assert.match(cloud, /<ContinueDialog/);
});

test('local continue still routes through the guarded, accumulating request helper', () => {
  assert.match(panel, /runConfirmableTrainingRequest/);
  assert.match(panel, /\(continueOpts\) => \(inCloud \? ds\.continueTrainingInCloud : ds\.continueTraining\)\(/);
  assert.match(panel, /fromStep:\s*payload\.fromStep,\s*overrides:\s*payload\.overrides/);
  assert.match(panel, /confirmableRetryFlag\(error, 'Continue anyway \(force\)'\)/);
});

test('cloud continue posts the run, extra steps, chosen checkpoint and overrides', () => {
  assert.match(cloud, /from_step:\s*payload\.fromStep/);
  assert.match(cloud, /overrides:\s*payload\.overrides/);
  assert.match(cloud, /extra_steps:\s*payload\.extraSteps/);
});

test('the continue hook forwards from_step and overrides only when present', () => {
  assert.match(hook, /opts\.fromStep\s*!=\s*null\s*\?\s*\{\s*from_step:\s*opts\.fromStep\s*\}/);
  assert.match(hook, /opts\.overrides\s*\?\s*\{\s*overrides:\s*opts\.overrides\s*\}/);
});

test('the dialog offers the LR factor knob and sends it only as a real reduction', () => {
  // a factor selector in the safe-overrides section, with its resulting-value hint
  assert.match(dialog, /LR_FACTOR_CHOICES/);
  assert.match(dialog, /half \(polish\)/);
  assert.match(dialog, /tenth \(gentle finish\)/);
  assert.match(dialog, /aria-label="Learning rate for the continuation"/);
  // keep-current (1) and adaptive (Prodigy) runs never send lr_factor
  assert.match(dialog, /lrFactor\s*!==\s*1\s*&&\s*!isAdaptiveLR\).*overrides\.lr_factor\s*=\s*lrFactor/s);
  // Prodigy disables the control with a reason rather than hiding it silently
  assert.match(dialog, /isAdaptiveLR\s*=\s*String\(settings\.optimizer\s*\|\|\s*''\)\.startsWith\('prodigy'\)/);
  assert.match(dialog, /disabled=\{isAdaptiveLR\}/);
  // the hint shows the resulting rate (→ 5e-5) computed from the run's current LR
  assert.match(dialog, /fmtLR\(currentLR\s*\*\s*lrFactor\)/);
});

test('both hubs feed the dialog the run optimizer + current LR for the hint', () => {
  assert.match(panel, /optimizer:\s*adv\?\.optimizer,\s*learning_rate:\s*adv\?\.learning_rate/);
  assert.match(cloud, /optimizer:\s*continueRunTarget\.settings\?\.optimizer/);
  assert.match(cloud, /learning_rate:\s*continueRunTarget\.settings\?\.lr/);
});

test('the dialog can open on a specific checkpoint (◉ Graph "continue from here")', () => {
  // opt-in prop, resolved by the unit-tested rule (lineageContinue.test.js):
  // the requested step when it is a real save, else the newest.
  assert.match(dialog, /initialFromStep\s*=\s*null/);
  assert.match(dialog, /initialResumeStep\(initialFromStep, steps\)/);
  assert.match(dialog, /import \{ initialResumeStep, resolveInitialLane \} from '\.\/lineageContinue\.js'/);
});

test('the dialog can offer the LANE (local vs cloud), opt-in and reasoned', () => {
  // opt-in prop: absent → no picker at all (the Runs hub keeps today's dialog)
  assert.match(dialog, /lanes = null/);
  assert.match(dialog, /\{lanes && \(/);
  assert.match(dialog, /resolveInitialLane\(where, lanes\)/);
  // both lanes rendered as radios, a closed one disabled WITH its reason shown
  assert.match(dialog, /aria-label="Where to run the continuation"/);
  assert.match(dialog, /💻 Local/);
  assert.match(dialog, /☁ Cloud/);
  assert.match(dialog, /disabled=\{off\}/);
  assert.match(dialog, /laneState\(lane\)\.reason/);
  // the chosen lane rides the payload, and a blocked lane can't be submitted
  assert.match(dialog, /lane,\s*\}\)/);
  assert.match(dialog, /disabled=\{busy \|\| latest === 0 \|\| laneBlocked\}/);
});

test('the Runs hub offers the picker too, with its own lane rule', () => {
  // It used to pass no `lanes` (a deliberate scope choice) and silently
  // relaunched a pod — Continue opened from the Runs page gave no choice at all.
  assert.match(cloud, /lanes=\{continueLanes\}/);
  // the hub's guards differ from the panel's (many datasets, machine-wide local
  // single-flight), so they live in their own unit-tested rule
  assert.match(cloud, /runsHubContinueLanes\(continueRunTarget/);
});

test('the dataset panel routes the chosen lane to the matching call', () => {
  // ONE dialog, two hooks — no third resume path, same guarded request helper
  assert.match(panel, /const inCloud = payload\.lane === 'cloud'/);
  assert.match(panel, /\(inCloud \? ds\.continueTrainingInCloud : ds\.continueTraining\)\(/);
  assert.match(panel, /lanes=\{continueLanes\}/);
  assert.match(panel, /where=\{laneOfStep\(continueInitialStep\)\}/);
  // each lane carries its own honest reason, cloud reusing the app's single source
  assert.match(panel, /Cloud training needs a vast\.ai API key/);
  assert.match(panel, /Local training needs ai-toolkit/);
  assert.match(panel, /cloudDisabledReason\s*\?\s*\{ available: false, reason: cloudDisabledReason \}/);
});

test('the cloud lane posts the local checkpoint to the continue-local endpoint', () => {
  assert.match(hook, /train\/cloud\/continue-local/);
  // it reuses the local payload shape (selection + safe overrides + from_step)
  assert.match(hook, /const continueTrainingInCloud = useCallback/);
  assert.match(hook, /opts\.fromStep != null \? \{ from_step: opts\.fromStep \} : \{\}/);
  assert.match(hook, /continueTraining, continueTrainingInCloud,/);
});

test('a ◉ Graph checkpoint pill opens the cloud Continue dialog pre-filled', () => {
  const graph = fs.readFileSync(new URL('./RunLineageGraph.jsx', import.meta.url), 'utf8');
  const tree = fs.readFileSync(new URL('./RunLineageTree.jsx', import.meta.url), 'utf8');
  // the graph surfaces a "continue from here" action, threaded through the tree
  assert.match(graph, /onContinueCheckpoint\(openCk\.node,\s*openCk\.pill\)/);
  assert.match(tree, /onContinueCheckpoint=\{onContinueCheckpoint\}/);
  // the Runs page maps a pill to the run and opens the dialog on that step
  assert.match(cloud, /continueFromCheckpoint/);
  assert.match(cloud, /setContinueInitialStep\(pill\?\.step/);
  assert.match(cloud, /initialFromStep=\{continueInitialStep\}/);
});
