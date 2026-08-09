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

test('an active run on the dataset no longer closes the cloud lane — the server confirm asks instead', () => {
  const otherDataset = runsHubContinueLanes(RUN, {
    ...OK, actives: [{ dataset_id: 42, train_type: 'zimage' }],
  });
  assert.equal(otherDataset.cloud.available, true, 'another dataset training must not block this run');

  const otherFamily = runsHubContinueLanes(RUN, {
    ...OK, actives: [{ dataset_id: 3, train_type: 'sdxl' }],
  });
  assert.equal(otherFamily.cloud.available, true, 'another family on the same dataset is allowed');

  // A same-family sibling used to hard-close the lane here, which made the
  // server's PARALLEL_RUN: confirm unreachable — the guard lives server-side
  // now, as a question ("second pod, billed separately — launch anyway?").
  const sameBoth = runsHubContinueLanes(RUN, {
    ...OK, actives: [{ dataset_id: 3, train_type: 'zimage' }],
    familyLabel: () => 'Z-Image',
  });
  assert.equal(sameBoth.cloud.available, true);
});

test('the hub cloud lane goes through the confirm loop, so PARALLEL_RUN: can be answered', () => {
  assert.match(page, /postWithConfirmations\(\s*\n?\s*\(b\) => postJson\('\/api\/dataset\/train\/cloud\/continue'/);
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
  // — Continue was opened from the Runs page and had no local/pod choice.
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
  // The loop itself now lives in the shared helper (one implementation, one
  // place where "never ask twice for the same flag" is enforced) — the contract
  // is that this lane goes through it, with its own label.
  assert.match(page, /postWithConfirmations\(\s*\n?\s*\(b\) => postJson\(`\/api\/dataset\/\$\{run\.dataset_id\}\/train\/continue`, b\),\s*\n?\s*body, 'Continue anyway \(force\)'\)/);
  // and a refusal must surface: postJson THROWS on 400/409. It is now shown
  // INSIDE the still-open dialog (the choices that caused it are still filled
  // in) instead of as a toast over a dialog that had already been destroyed.
  assert.match(page, /continueAttemptOutcome\(\{ thrown: e \}\)/);
  assert.match(page, /if \(!outcome\.close\) \{ setContinueError\(outcome\.error\); return; \}/);
});

test('the confirmable refusal markers have ONE definition, shared by both mounts', () => {
  const util = fs.readFileSync(new URL('./trainingRefusals.js', import.meta.url), 'utf8');
  const panel = fs.readFileSync(
    new URL('../components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8');
  for (const marker of ['MISMATCH_CAPTION: ', 'UNCAPTIONED: ',
    'CAPTION_QUALITY: ', 'CUSTOM_WEIGHTS_UNVERIFIED: ']) {
    assert.ok(util.includes(marker), `${marker} must live in the shared util`);
  }
  // The readiness floor is a marker too, and it also belongs to the util — the
  // Runs hub's ↻ Retry answers it (GitHub #23), the dataset panel's own
  // "Continue anyway" checkbox answers it there.
  assert.ok(util.includes('NOT_READY: '), 'NOT_READY: must live in the shared util');
  // The panel imports from the shared util — WHICH names it takes is not the
  // contract. It gained `postWithConfirmations` when the cloud launch stopped
  // hand-rolling a retry that could never run: that loop tested `d.ok === false`
  // around a client that THROWS on a 409, so a confirmable refusal reached the
  // user as an error toast asking a question it gave no way to answer. Pinning
  // the exact import line would have made that fix look like a violation.
  assert.match(panel, /import \{[^}]*confirmableRetryFlag[^}]*\} from '\.\.\/\.\.\/utils\/trainingRefusals'/);
  assert.match(page, /from '\.\.\/utils\/trainingRefusals'/);
  assert.doesNotMatch(panel, /const CONFIRMABLE_REFUSALS = \[/);
  assert.doesNotMatch(page, /const CONFIRMABLE_REFUSALS = \[/);
  // neither mount may hand-roll the confirm loop again
  assert.doesNotMatch(page, /\[flag\]: true \}/);
});
