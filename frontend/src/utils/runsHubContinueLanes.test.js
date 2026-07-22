import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { runsHubContinueLanes } from './runsHubContinueLanes.js';

const page = fs.readFileSync(new URL('../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');

const RUN = { run_id: 7, dataset_id: 3, train_type: 'zimage', variant: 'turbo' };
const OK = { aitoolkitValid: true, configured: true, limit: 2, actives: [] };

test('both lanes are open when nothing is running and both are set up', () => {
  const lanes = runsHubContinueLanes(RUN, OK);
  assert.equal(lanes.local.available, true);
  assert.equal(lanes.cloud.available, true);
});

test('a blocked lane keeps its slot and states its reason — it never disappears', () => {
  const lanes = runsHubContinueLanes(RUN, { ...OK, aitoolkitValid: false });
  assert.equal(lanes.local.available, false);
  assert.match(lanes.local.reason, /ai-toolkit/);
  assert.equal(lanes.cloud.available, true);
  // the cloud key missing closes the OTHER lane the same way
  const noKey = runsHubContinueLanes(RUN, { ...OK, configured: false });
  assert.equal(noKey.cloud.available, false);
  assert.match(noKey.cloud.reason, /vast\.ai API key/);
});

test('local is single-flight for the WHOLE machine — a run on another dataset closes it', () => {
  // The hub lists runs of many datasets: unlike the per-dataset cloud guard,
  // the local one must not care WHICH dataset is currently training.
  const lanes = runsHubContinueLanes(RUN, {
    ...OK, localActive: { current: { dataset_id: 999 } },
  });
  assert.equal(lanes.local.available, false);
  assert.match(lanes.local.reason, /already running on this machine/);
  assert.equal(lanes.cloud.available, true);
});

test('the cloud guard is per dataset and family, not page-wide', () => {
  const otherDataset = runsHubContinueLanes(RUN, {
    ...OK, actives: [{ dataset_id: 42, train_type: 'zimage' }],
  });
  assert.equal(otherDataset.cloud.available, true, 'another dataset training must not block this run');

  const otherFamily = runsHubContinueLanes(RUN, {
    ...OK, actives: [{ dataset_id: 3, train_type: 'sdxl' }],
  });
  assert.equal(otherFamily.cloud.available, true, 'another family on the same dataset is allowed');

  const sameBoth = runsHubContinueLanes(RUN, {
    ...OK, actives: [{ dataset_id: 3, train_type: 'zimage' }],
    familyLabel: () => 'Z-Image',
  });
  assert.equal(sameBoth.cloud.available, false);
  assert.match(sameBoth.cloud.reason, /Z-Image cloud run is already active on this dataset/);
});

test('the concurrency limit closes the cloud lane and names the count', () => {
  const lanes = runsHubContinueLanes(RUN, {
    ...OK, limit: 1, actives: [{ dataset_id: 42, train_type: 'sdxl' }],
  });
  assert.equal(lanes.cloud.available, false);
  assert.match(lanes.cloud.reason, /limit reached \(1\/1\)/);
});

test('a run with no dataset can only go to the cloud', () => {
  const lanes = runsHubContinueLanes({ run_id: 9, train_type: 'zimage' }, OK);
  assert.equal(lanes.local.available, false);
  assert.match(lanes.local.reason, /only be continued in the cloud/);
  assert.equal(lanes.cloud.available, true);
});

test('no run open → no picker at all (the dialog stays single-lane)', () => {
  assert.equal(runsHubContinueLanes(null, OK), null);
});

test('the Runs hub actually offers the picker and routes the local lane', () => {
  // The hub used to mount ContinueDialog WITHOUT `lanes` (cloud-only by design)
  // — Jeremy opened Continue from the Runs page and had no local/pod choice.
  assert.match(page, /lanes=\{continueLanes\}/);
  assert.match(page, /runsHubContinueLanes\(continueRunTarget/);
  // and the local lane must reach the LOCAL endpoint, not the cloud one
  assert.match(page, /const local = payload\.lane === 'local';/);
  assert.match(page, /postJson\(`\/api\/dataset\/\$\{run\.dataset_id\}\/train\/continue`/);
  // addressed by the RUN's own base/family/variant, never the dataset's
  // persisted selection (which may point at another base entirely)
  assert.match(page, /run\.base_model != null \? \{ base_model: run\.base_model \}/);
  assert.match(page, /run\.train_type \? \{ train_type: run\.train_type \}/);
  assert.match(page, /run\.variant \? \{ variant: run\.variant \}/);
});

test('a local continuation loops on the confirmable refusals like the panel does', () => {
  // A resume re-exports the CURRENT dataset, so it hits the caption/quality
  // guards; without the loop the user just got a raw "MISMATCH_CAPTION: …".
  assert.match(page, /confirmableRetryFlag\(e\?\.message, 'Continue anyway \(force\)'\)/);
  assert.match(page, /body = \{ \.\.\.body, \[flag\]: true \}/);
  // and a refusal must surface: postJson THROWS on 400/409
  assert.match(page, /toast\.error\(e\?\.message \|\| 'Continue failed'\)/);
});

test('the confirmable refusal markers have ONE definition, shared by both mounts', () => {
  const util = fs.readFileSync(new URL('./trainingRefusals.js', import.meta.url), 'utf8');
  const panel = fs.readFileSync(
    new URL('../components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8');
  for (const marker of ['MISMATCH_CAPTION: ', 'UNCAPTIONED: ',
    'CAPTION_QUALITY: ', 'CUSTOM_WEIGHTS_UNVERIFIED: ']) {
    assert.ok(util.includes(marker), `${marker} must live in the shared util`);
  }
  assert.match(panel, /import \{ confirmableRetryFlag \} from '\.\.\/\.\.\/utils\/trainingRefusals'/);
  assert.match(page, /import \{ confirmableRetryFlag \} from '\.\.\/utils\/trainingRefusals'/);
  assert.doesNotMatch(panel, /const CONFIRMABLE_REFUSALS = \[/);
});
