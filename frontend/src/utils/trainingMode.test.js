import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { cloudTrainingLaunchPayload } from './checkpointBrowser.js';
import {
  TRAINING_MODE_FULL_TRANSFORMER,
  TRAINING_MODE_LORA,
  canRecheckFullTransformerDelivery,
  cloudTierEstimateView,
  denseContinueBlocker,
  fullTransformerArtifactView,
  fullTransformerRecheckOutcome,
  fullTransformerUnavailableReason,
  hfCloudTokenReadiness,
  isFullTransformerRun,
  denseTurboWarning,
  fullTransformerBaseLabel,
  isFullTransformerEligible,
  normalizeTrainingMode,
  trainingModeSettingsPayload,
  trainingModeLabel,
} from './trainingMode.js';
import { preflightUrl } from '../components/dataset/preflightLane.js';

// Slice 1 moved the dense recipe/picker and the cloud dialog to their own
// files; the mode contract spans all three, so it reads them as one text.
const panel = readFileSync(new URL('../components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8')
  + readFileSync(new URL('../components/dataset/FullTransformerRecipe.jsx', import.meta.url), 'utf8')
  + readFileSync(new URL('../components/dataset/CloudLaunchDialog.jsx', import.meta.url), 'utf8');
const datasetHook = readFileSync(new URL('../hooks/useDataset.js', import.meta.url), 'utf8');
const runsPage = readFileSync(new URL('../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');
const stopDialog = readFileSync(new URL('../pages/cloudStopDialog.js', import.meta.url), 'utf8');

test('training mode enum is exact and every legacy or invalid value falls back to LoRA', () => {
  assert.equal(TRAINING_MODE_LORA, 'lora');
  assert.equal(TRAINING_MODE_FULL_TRANSFORMER, 'full_transformer');
  assert.equal(normalizeTrainingMode('full_transformer'), 'full_transformer');
  assert.equal(normalizeTrainingMode('full_model'), 'lora');
  assert.equal(normalizeTrainingMode(undefined), 'lora');
  assert.equal(trainingModeLabel('lora'), 'LoRA');
  assert.equal(trainingModeLabel('full_transformer'), 'Full model');
});

test('dense eligibility covers the whole Krea 2 family, and picking Turbo no longer expels the user', () => {
  // The regression this replaces: choosing Turbo made Full model ineligible,
  // and the panel's fallback effect then SAVED LoRA behind the user's back —
  // which is why the owner could not find "where to put the turbo option".
  assert.equal(isFullTransformerEligible({ trainType: 'krea', variant: 'base', baseModel: '' }), true);
  assert.equal(isFullTransformerEligible({ trainType: 'krea', variant: 'turbo', baseModel: '' }), true);
  assert.equal(isFullTransformerEligible({
    trainType: 'krea', variant: 'turbo', baseModel: 'C:/w/custom.safetensors', customBase: true,
  }), true);
  assert.equal(isFullTransformerEligible({ trainType: 'zimage', variant: 'base', baseModel: '' }), false);
  assert.equal(fullTransformerUnavailableReason({ trainType: 'krea', variant: 'turbo' }), null);
  assert.match(fullTransformerUnavailableReason({ trainType: 'zimage' }), /Krea 2 family/);
});

test('the dense summary names the base the run will actually train on', () => {
  assert.equal(fullTransformerBaseLabel({ variant: 'base' }), 'official Krea 2 Raw');
  assert.equal(fullTransformerBaseLabel({ variant: 'turbo' }), 'official Krea 2 Turbo');
  assert.equal(fullTransformerBaseLabel({}), 'official Krea 2 Raw');
  // An unset variant means Raw, exactly like the backend default.
  assert.equal(fullTransformerBaseLabel({ variant: undefined }), 'official Krea 2 Raw');
  // A picked checkpoint wins over the variant, and never leaks its full path.
  assert.equal(
    fullTransformerBaseLabel({ variant: 'turbo', baseModel: 'C:/models/my-krea.safetensors' }),
    'custom: my-krea.safetensors');
  assert.equal(
    fullTransformerBaseLabel({ baseModel: 'C:/m/x.safetensors', baseLabel: 'Krea 2 Raw bf16' }),
    'Krea 2 Raw bf16');
});

test('the dense Turbo warning states what is unknown, without promising or predicting', () => {
  assert.equal(denseTurboWarning({ variant: 'base' }), null);
  assert.equal(denseTurboWarning({ variant: 'turbo', baseModel: 'C:/m/x.safetensors' }), null);
  const notice = denseTurboWarning({ variant: 'turbo' });
  assert.match(notice.title, /untested here/);
  assert.match(notice.body, /LoRA on Raw/);
  assert.match(notice.body, /have not measured/);
  assert.match(notice.body, /few-step/);
  assert.match(notice.body, /Nothing is blocked/);
  // No promise, no predicted failure.
  assert.doesNotMatch(notice.body, /will (?:work|fail|break|ruin)|guaranteed|corrupt/i);
});

test('cloud launch and preflight carry the mode while LoRA stays the default regression path', () => {
  assert.equal(cloudTrainingLaunchPayload({ trainType: 'krea', variant: 'base' }).training_mode, 'lora');
  assert.equal(cloudTrainingLaunchPayload({
    trainType: 'krea', variant: 'base', trainingMode: 'full_transformer', gpuName: 'H100',
  }).training_mode, 'full_transformer');
  assert.equal(preflightUrl(9, {
    trainType: 'krea', variant: 'base', baseModel: '',
    trainingMode: 'full_transformer', lane: 'cloud',
  }), '/api/dataset/9/train/preflight?train_type=krea&variant=base&base_model=&training_mode=full_transformer&lane=cloud');
  assert.deepEqual(trainingModeSettingsPayload('full_transformer', {
    trainType: 'krea', variant: 'base', baseModel: '',
    disableSliderForFullTransformer: true,
  }), {
    training_mode: 'full_transformer', train_type: 'krea', variant: 'base', base_model: '',
    disable_slider_for_full_transformer: true,
  });
});

test('the pending panel does not claim a model upload that has not started', () => {
  // Run #138: 'artifact_status' is stamped 'pending' at LAUNCH, so for the two
  // hours the run spent pushing its DATASET to the pod this panel announced
  // 'Uploading full model…' — a transfer that had not begun and could not,
  // next to a repository holding nothing but licence files. The run's phase is
  // what tells the two apart.
  const base = {
    training_mode: 'full_transformer', artifact_status: 'pending',
    hf_url: 'https://huggingface.co/me/private',
  };
  for (const status of ['preparing', 'provisioning', 'uploading']) {
    const view = fullTransformerArtifactView({ ...base, status });
    assert.equal(view.label, 'Full model not created yet', `status=${status}`);
    assert.match(view.detail, /Nothing is uploading to Hugging Face yet/);
    assert.equal(view.href, null);
    assert.equal(view.available, false);
  }

  const training = fullTransformerArtifactView({ ...base, status: 'training' });
  assert.equal(training.label, 'Full model not delivered yet');
  assert.match(training.detail, /delivered to Hugging Face at the end of the run/);

  // Once training is over, 'pending' does mean the weights are on their way.
  const delivering = fullTransformerArtifactView({ ...base, status: 'downloading' });
  assert.equal(delivering.label, 'Uploading full model…');

  // The worst version of the same lie, caught on the proof screenshot: a run
  // the supervisor had already terminated still announced an upload in flight
  // AND told the user to keep a pod alive that no longer existed.
  for (const status of ['error', 'stopped', 'error_pod_kept', 'done']) {
    const over = fullTransformerArtifactView({ ...base, status });
    assert.equal(over.label, 'Full model was never delivered', `status=${status}`);
    assert.match(over.detail, /ended before any weights reached Hugging Face/);
    assert.doesNotMatch(over.detail, /[Kk]eep the run and pod active/);
    assert.equal(over.tone, 'warning');
  }

  // No status at all (an older payload) is not evidence of anything: it keeps
  // the neutral wording rather than announcing a failed delivery.
  const unknown = fullTransformerArtifactView(base);
  assert.equal(unknown.label, 'Uploading full model…');
  assert.equal(unknown.tone, 'info');

  // Repository creation keeps its own label whatever the phase says, and a
  // detail the backend did send always wins over any of these fallbacks.
  assert.equal(fullTransformerArtifactView({
    ...base, artifact_status: 'creating_repository', status: 'preparing',
  }).label, 'Creating Hugging Face repository…');
  assert.equal(fullTransformerArtifactView({
    ...base, status: 'uploading', artifact_status_detail: 'from the backend',
  }).detail, 'from the backend');
});

test('a full artifact link exists only after verified availability', () => {
  const pending = fullTransformerArtifactView({
    training_mode: 'full_transformer', artifact_status: 'verification_pending',
    hf_url: 'https://huggingface.co/me/private', artifact_status_detail: 'token timed out',
  });
  assert.equal(pending.href, null);
  assert.equal(pending.repositoryHref, 'https://huggingface.co/me/private');
  assert.equal(pending.tone, 'warning');
  assert.equal(pending.label, 'Hugging Face verification pending');
  assert.match(pending.detail, /token timed out/);

  const missing = fullTransformerArtifactView({
    artifact_status: 'missing', hf_url: 'https://huggingface.co/me/private',
  });
  assert.equal(missing.href, null);
  assert.equal(missing.tone, 'error');
  assert.equal(missing.label, 'Full model not found');

  const available = fullTransformerArtifactView({
    artifact_status: 'available', hf_url: 'https://huggingface.co/me/private',
  });
  assert.equal(available.href, 'https://huggingface.co/me/private');
  assert.equal(available.available, true);
  // Past tense: `artifact_status` is stamped at delivery and never revisited,
  // so "available" is what WAS true. The label used to say "Full model
  // available" above a link that answered 404 for a deleted repository.
  assert.equal(available.label, 'Full model delivered');
  assert.equal(isFullTransformerRun({ training_mode: 'full_transformer' }), true);
});

test('a delivered model is dated and flagged unverified, not announced as present', () => {
  const view = fullTransformerArtifactView({
    artifact_status: 'available', hf_url: 'https://huggingface.co/me/private',
    verified_at: '2026-07-11T09:12:33',
    artifact_status_detail: 'Dense checkpoint and compliance metadata verified',
  });
  assert.equal(view.label, 'Full model delivered');
  assert.match(view.detail, /Delivered and verified on 2026-07-11 — not re-checked since/);
  assert.match(view.detail, /Open the repository to confirm it is still there/);
  // The stored detail is a present-tense claim about a past verification. It is
  // exactly what made this panel lie, so it does not get to speak here.
  assert.doesNotMatch(view.detail, /metadata verified/);
});

test('only a live Hub answer promotes the panel to the present tense', () => {
  const run = { artifact_status: 'available', hf_url: 'https://huggingface.co/me/private' };
  const fresh = fullTransformerArtifactView(run, { state: 'present' });
  assert.equal(fresh.label, 'Full model on Hugging Face');
  assert.match(fresh.detail, /Checked just now/);
  assert.equal(fresh.href, 'https://huggingface.co/me/private');
  // "Could not check" is OUR failure. Demoting the view on it would tell
  // someone offline that their eight hours of GPU are gone.
  const offline = fullTransformerArtifactView(run, {
    state: 'unknown', detail: 'Hugging Face could not be reached',
  });
  assert.equal(offline.label, 'Full model delivered');
  assert.equal(offline.tone, 'success');
  assert.equal(offline.href, 'https://huggingface.co/me/private');
});

test('a repository measured gone drops BOTH links — there is nothing behind them', () => {
  const view = fullTransformerArtifactView({
    artifact_status: 'available', hf_url: 'https://huggingface.co/me/private',
    verified_at: '2026-07-11T09:12:33',
  }, { state: 'gone' });
  assert.equal(view.label, 'Full model no longer on Hugging Face');
  assert.equal(view.tone, 'error');
  assert.equal(view.available, false);
  assert.equal(view.href, null);
  // Not even as "Inspect the repository (delivery unverified)": that label
  // would misname a 404, and the click is the dead end being fixed.
  assert.equal(view.repositoryHref, null);
  assert.match(view.detail, /no longer answers/);
  assert.match(view.detail, /check the Checkpoints panel for a copy on this computer/i);
});

test('a run that delivered nothing loses its inspect link too once the repo is gone', () => {
  const run = { artifact_status: 'missing', hf_url: 'https://huggingface.co/me/private' };
  // Without a live answer the link stays: a repository that exists but holds no
  // weights is worth opening, and that is what this branch was written for.
  assert.equal(fullTransformerArtifactView(run).repositoryHref,
    'https://huggingface.co/me/private');
  const gone = fullTransformerArtifactView(run, { state: 'gone' });
  assert.equal(gone.repositoryHref, null);
  assert.match(gone.detail, /nothing left here to inspect/);
  // And an unreachable Hub changes nothing: the link is still the right link.
  assert.equal(fullTransformerArtifactView(run, { state: 'unknown' }).repositoryHref,
    'https://huggingface.co/me/private');
});

test('Continue counts ROADS, and only a measured gone closes the Hub one', () => {
  // `resume_checkpoints[].source` is rebuilt from the disk on every payload —
  // that is why it, and not a delivery-time stamp, is what may take a button
  // away. Gating on `local_artifact_status` would have been this wave's own bug
  // mirrored: a button offered for a file deleted by hand.
  const hubOnly = { resume_checkpoints: [{ step: 3000, source: 'hub' }] };
  const localOnly = { resume_checkpoints: [{ step: 3000, source: 'local' }] };
  const both = { resume_checkpoints: [
    { step: 3000, source: 'local' }, { step: 3000, source: 'hub' }] };

  for (const run of [hubOnly, localOnly, both]) {
    assert.equal(denseContinueBlocker(run, null), null);
    assert.equal(denseContinueBlocker(run, { state: 'present' }), null);
    // A check we could not make never closes a road.
    assert.equal(denseContinueBlocker(run, { state: 'unknown' }), null);
  }
  // Gone shuts the Hub road ONLY. The direct road is exactly the way out of a
  // deleted repository, so greying the button there would hide it.
  assert.equal(denseContinueBlocker(localOnly, { state: 'gone' }), null);
  assert.equal(denseContinueBlocker(both, { state: 'gone' }), null);
  assert.match(denseContinueBlocker(hubOnly, { state: 'gone' }),
    /neither road is open/);
  // A missing source defaults to the local road — an older payload must not
  // lose its button.
  assert.equal(denseContinueBlocker(
    { resume_checkpoints: [{ step: 3000 }] }, { state: 'gone' }), null);
});

/* The value of a JSX attribute, brace-balanced — so an assertion about
   `disabled=` cannot be satisfied by something written inside `title=` further
   down. The obvious `/disabled=\{[\s\S]*?fn\(/` does exactly that: it spans
   attributes, and it passed on a file with the call deleted from `disabled`
   altogether. Measuring the reachable proxy instead of the property is the very
   mistake this wave exists to correct; it is not allowed in its own tests. */
function jsxAttr(source, name) {
  const at = source.indexOf(`${name}={`);
  if (at < 0) return '';
  let i = at + name.length + 1;
  let depth = 0;
  for (; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    else if (source[i] === '}') { depth -= 1; if (!depth) break; }
  }
  return source.slice(at, i + 1);
}

test('THE trap: a trashed master with a stale "available" stamp still disables Continue', () => {
  // The scenario, one click away from any user: they empty 26 GB with the card's
  // own "🧹 Clean" button, and the Hub copy is gone too.
  //
  // `local_artifact_status` is written in exactly three places, all at delivery
  // time (cloud_training.py :2384 pending, :4310 available, :4431 cancelled), and
  // `dense_artifacts.delete_artifact` — the trash path — does not touch it. So the
  // payload below is REAL: the stamp says the file is here, and it is not.
  //
  // Read the stamp and this run offers ▶ Continue, which then answers "this full
  // model has nothing left to continue from" — the exact defect this wave fixes
  // on the Hub side, rebuilt on the local side. `resume_checkpoints` cannot lie
  // the same way: _dense_resume_candidates stats every file and drops what OSError
  // says is gone.
  const trashedButStamped = {
    local_artifact_status: 'available',
    local_weight_filename: 'Krea_dense_000003000.safetensors',
    local_artifact_dir: 'anywhere',
    resume_checkpoints: [{ step: 3000, source: 'hub' }],   // the disk road is GONE
  };
  assert.match(denseContinueBlocker(trashedButStamped, { state: 'gone' }),
    /neither road is open/,
    'a delivery-time stamp must never keep this button alive');
  // The same payload with the Hub still there keeps the button: one road is enough.
  assert.equal(denseContinueBlocker(trashedButStamped, { state: 'present' }), null);
  assert.equal(denseContinueBlocker(trashedButStamped, { state: 'unknown' }), null);
  // And the mirror: the disk road alive, the stamp saying nothing at all.
  assert.equal(denseContinueBlocker(
    { resume_checkpoints: [{ step: 3000, source: 'local' }] }, { state: 'gone' }), null);
});

test('the Runs page reads the blocker for BOTH the disabled state and the reason', () => {
  // A greyed button with no explanation is the thing this replaces, and a
  // reason nobody is disabled by is decoration.
  const button = runsPage.slice(runsPage.indexOf('onClick={() => continueRun(run)}'));
  const call = 'denseContinueBlocker(run, hubPresence[run.run_id])';
  assert.ok(jsxAttr(button, 'disabled').includes(call),
    'the blocker must be inside the disabled expression itself');
  assert.ok(jsxAttr(button, 'title').includes(call),
    'the blocker must be inside the title expression itself');
});

test('an in-flight delivery is never demoted by a 404 — the repo may not exist yet', () => {
  // A pod that has not created the repository yet answers 404 by construction.
  // Only a run whose record says the delivery FINISHED may be read that way.
  for (const status of ['verification_pending', 'pending', 'uploading', 'creating_repository']) {
    const view = fullTransformerArtifactView(
      { artifact_status: status, status: 'training',
        hf_url: 'https://huggingface.co/me/private' }, { state: 'gone' });
    assert.notEqual(view.label, 'Full model no longer on Hugging Face');
  }
});

test('dense token readiness fails closed only when the backend explicitly reports a problem', () => {
  assert.deepEqual(hfCloudTokenReadiness({}), {
    signaled: false, ready: true, blocked: false, detail: null,
  });
  const offerFailure = hfCloudTokenReadiness({
    hf_cloud_token: {
      ok: false, configured: true,
      error: 'HF_CLOUD_TOKEN cannot write to the dedicated namespace',
    },
  });
  assert.equal(offerFailure.blocked, true);
  assert.match(offerFailure.detail, /dedicated namespace/);
  assert.equal(hfCloudTokenReadiness({
    hf_cloud_token: { ok: true, configured: true, namespace: 'lds-deliveries' },
  }).ready, true);
  assert.equal(hfCloudTokenReadiness({
    checks: [{ id: 'hf_cloud_token', status: 'fail', detail: 'fine-grained scope invalid' }],
    hf_cloud_token_status: { ok: false, configured: true },
  }).blocked, true);
});

test('kept dense runs can recheck verification or pending pod cleanup', () => {
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'verification_pending',
  }), true);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available',
  }), true);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available', artifact_cleanup_status: 'pending',
  }), true);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available', artifact_cleanup_status: 'complete',
  }), false);
  assert.equal(canRecheckFullTransformerDelivery({
    training_mode: 'lora', status: 'error_pod_kept', artifact_status: 'verification_pending',
  }), false);
});

test('legacy verified kept rows default to visible cleanup-pending state', () => {
  const legacy = fullTransformerArtifactView({
    training_mode: 'full_transformer', status: 'error_pod_kept',
    artifact_status: 'available',
    hf_url: 'https://huggingface.co/me/private',
  });
  assert.equal(legacy.available, true);
  assert.equal(legacy.cleanupPending, true);
  assert.equal(legacy.href, 'https://huggingface.co/me/private');
  assert.equal(legacy.tone, 'warning');
  assert.match(legacy.detail, /may still be billing/);
});

test('verified model and pending cleanup never produce a pod-released success', () => {
  const pending = fullTransformerRecheckOutcome({
    ok: true, delivery: 'available', cleanup_pending: true,
  });
  assert.equal(pending.kind, 'warning');
  assert.match(pending.text, /Hugging Face model verified and available/);
  assert.match(pending.text, /may still be billing/);

  const complete = fullTransformerRecheckOutcome({
    ok: true, delivery: 'available', cleanup_pending: false,
  });
  assert.equal(complete.kind, 'success');
  assert.match(complete.text, /pod cleanup is confirmed/);
});

test('dense offers never reuse an unlabelled or unavailable estimate', () => {
  assert.deepEqual(cloudTierEstimateView({
    est_minutes: 42, est_cost: 1.25, exceeds_cap: true,
  }, { fullMode: true }), {
    available: false, minutes: null, cost: null, exceedsCap: false, status: null,
  });
  assert.equal(cloudTierEstimateView({
    estimate_status: 'unavailable', est_minutes: 42, est_cost: 1.25, exceeds_cap: true,
  }, { fullMode: true }).available, false);
  assert.equal(cloudTierEstimateView({
    estimate_status: 'available', est_minutes: null, est_cost: null, exceeds_cap: true,
  }, { fullMode: true }).available, false);
  assert.deepEqual(cloudTierEstimateView({
    estimate_status: 'available', est_minutes: 42, est_cost: 1.25, exceeds_cap: true,
  }, { fullMode: true }), {
    available: true, minutes: 42, cost: 1.25, exceedsCap: true, status: 'available',
  });
  // Legacy LoRA offers remain usable while the backend rolls out estimate_status.
  assert.equal(cloudTierEstimateView({ est_minutes: 42 }, { fullMode: false }).available, true);
});

test('MVP copy and artifact actions distinguish a full model from a LoRA', () => {
  assert.match(panel, /LoRA/);
  assert.match(panel, /Full model/);
  assert.match(panel, /80 GB VRAM GPU/);
  assert.match(panel, /at least 200 GB disk/);
  assert.match(panel, /~26 GB/);
  assert.match(panel, /private Hugging Face repository/);
  assert.match(panel, /much larger, more diverse dataset/);
  assert.match(panel, /Open private model on Hugging Face/);
  assert.match(panel, /!fullMode && !cloudActiveHere/);
  assert.match(panel, /HF_CLOUD_TOKEN/);
  assert.match(panel, /ArrowLeft/);
  assert.match(panel, /tabIndex=\{!fullMode \|\| !fullTransformerEligible \? 0 : -1\}/);
  assert.match(panel, /aria-describedby/);
  assert.doesNotMatch(panel,
    /Modèle complet|Fine-tuning complet|Recette dense verrouillée|Lancer le fine-tuning complet|Choisir un GPU 80 Go|Livraison Hugging Face bloquée|estimation dense indisponible/);
});

test('dense Advanced exposes exactly the five unlocked values and states why the rest is locked', () => {
  const recipe = panel.slice(
    panel.indexOf('// FULL_TRANSFORMER_ADVANCED_RECIPE_START'),
    panel.indexOf('// FULL_TRANSFORMER_ADVANCED_RECIPE_END'),
  );
  assert.match(panel, /Full-model recipe · steps · prompts · LR · resolution · checkpoints/);
  // Computed, not a literal: this line used to say "Official Krea 2 Raw" over
  // a recipe that can now be Turbo or a local checkpoint.
  assert.match(recipe, /\{baseSummary\}[\s\S]{0,80}full transformer · unquantized/);
  assert.match(recipe, /80 GB VRAM GPU · at least 200 GB disk/);

  // The four values the 80 GB geometry depends on stay locked, and each says
  // WHY — a greyed-out control with no reason reads as an arbitrary limit.
  assert.match(recipe, /Locked · batch &amp; precision[\s\S]{0,200}Batch 1 · bf16/);
  assert.match(recipe, /Locked · optimizer[\s\S]{0,200}Adafactor/);
  assert.match(recipe, /Locked · memory[\s\S]{0,200}Gradient checkpointing/);
  assert.match(recipe, /would not fit in memory|has no room for more/);

  // The five unlocked ones, each wired to a real persistence call.
  assert.match(recipe, /setStepsOverride\(event\.target\.value\)/);
  assert.match(recipe, /patch\(\{ dense_lr: value \}\)/);
  assert.match(recipe, /patch\(\{ dense_resolution: Number\(event\.target\.value\) \}\)/);
  assert.match(recipe, /patch\(\{ dense_save_every: value \}\)/);
  assert.match(recipe, /patch\(\{ dense_max_step_saves: Number\(event\.target\.value\) \}\)/);
  assert.match(recipe, /saveSamplePrompts\?\.\(\)/);
  // Bounds come from the server payload, never from a second copy of the rules.
  assert.match(recipe, /adv\?\.dense_lr_min/);
  assert.match(recipe, /adv\?\.dense_save_every_max/);
  assert.match(recipe, /adv\?\.dense_max_step_saves_max/);

  // Keep x ~26 GB is stated BEFORE the launch, next to the control that sets it.
  assert.match(recipe, /dense_storage_plan/);
  assert.match(recipe, /fp8_typical_bytes/);
  assert.match(recipe, /PRIVATE Hugging Face storage/);

  // The dense artifact is a RAW checkpoint: how to test it must be visible here.
  assert.match(recipe, /dense_inference_hint/);


  // Still no LoRA CONTROL may leak in (naming them in prose is the point).
  assert.doesNotMatch(recipe, /CUSTOM_BASE_SENTINEL|advNetworkType|advEffRank|applyPreset|setBase\(|setVariant\(|setMasked\(/,
    'the dense recipe card must not grow an ignored LoRA control');
});

test('the full Advanced branch cannot render the unchanged LoRA controls', () => {
  const branch = panel.slice(
    panel.indexOf('{fullMode ? (', panel.indexOf('FULL_TRANSFORMER_ADVANCED_BRANCH_START')),
    panel.indexOf('FULL_TRANSFORMER_ADVANCED_BRANCH_END'),
  );
  const split = branch.indexOf(') : (<>');
  assert.ok(split > 0, 'Advanced must have explicit dense and LoRA render arms');
  const denseArm = branch.slice(0, split);
  const loraArm = branch.slice(split);
  assert.match(denseArm, /<FullTransformerAdvancedRecipe/);
  // LoRA-ONLY controls must not leak in. The base picker is deliberately NOT on
  // this list any more: a dense run can now be pointed at Raw, Turbo or a local
  // checkpoint, so the base/variant selectors are shared, not LoRA-only — and
  // the dense arm renders its own copy (the LoRA arm's lives below the split
  // and would otherwise be unreachable in full-model mode, which is exactly why
  // the Turbo option looked missing).
  assert.match(denseArm, /DENSE_BASE_PICKER_START/);
  assert.doesNotMatch(denseArm, /Presets|advNetworkType|Masked \(bg 10%\)|saveAdv\(/);
  assert.match(loraArm, /Presets/);
  assert.match(loraArm, /CUSTOM_BASE_SENTINEL/);
  assert.match(loraArm, /advNetworkType/);
  assert.match(loraArm, /Masked \(bg 10%\)/);
  assert.match(loraArm, /saveAdv\(/);
  assert.match(panel, /advancedOpen && trainingMode !== TRAINING_MODE_FULL_TRANSFORMER/);
});

test('local hook persists and launches with the canonical mode', () => {
  assert.match(datasetHook, /trainingModeSettingsPayload\(trainingMode, selection\)/);
  assert.match(datasetHook, /slider: d\.slider \?\? null/);
  assert.match(datasetHook, /training_mode: normalizeTrainingMode\(opts\.trainingMode\)/);
  assert.match(datasetHook, /catch \(error\)[\s\S]*return null/);
});

test('offers use the exact recipe and refetch when any recipe input changes', () => {
  assert.match(panel, /new URLSearchParams\(\{\s*train_type: trainType,\s*variant,\s*base_model: base \?\? '',\s*training_mode:/);
  assert.match(panel, /\[datasetId, trainType, variant, base, trainingMode, steps\]/);
  assert.match(panel, /full-model estimate unavailable — hourly price only/);
  assert.match(panel, /hasUsableEstimate/);
  assert.match(panel, /hfCloudTokenReadiness\(data \|\| \{\}\)/);
  assert.match(panel, /disabled=\{!selected \|\| launching \|\| !customBaseReady \|\| hfTokenBlocked\}/);
  assert.match(panel, /focus="HF_CLOUD_TOKEN"/);
  assert.match(panel, /checksDenseCloudToken/);
});

test('a verified cloud token stops asking the user to configure it', () => {
  // Regression 2026-08-01: the "configure it before renting the GPU" banner was
  // rendered on `fullMode` alone, so a token the backend had just verified was
  // still greeted with setup instructions on every launch.
  assert.doesNotMatch(panel, /\{fullMode && \(\s*\n\s*<p className="m-0 rounded-lg border border-amber-400\/35/);
  assert.match(panel, /\{fullMode && !hfTokenIssue && \(/);
  assert.match(panel, /const hfTokenVerified = offerTokenStatus\?\.ok === true/);
  assert.match(panel, /const hfTokenBroad = hfTokenVerified && offerTokenStatus\?\.code === 'broad_access'/);
  assert.match(panel, /Hugging Face delivery ready\./);
  assert.match(panel, /Hugging Face delivery ready \(broad token\)\./);
  // The setup instructions must survive for the case they were written for.
  assert.match(panel, /before renting the GPU/);
});

test('empty cloud offers preserve the cap message and link to its exact setting', () => {
  const emptyOffersStart = panel.indexOf('{!loading && !error && tiers.length === 0 && (');
  const populatedOffersStart = panel.indexOf('{tiers.length > 0 && (', emptyOffersStart);
  assert.ok(emptyOffersStart >= 0 && populatedOffersStart > emptyOffersStart,
    'the empty-offers branch must remain distinct from the populated offer list');

  const emptyOffersBranch = panel.slice(emptyOffersStart, populatedOffersStart);
  assert.match(emptyOffersBranch,
    /No GPU available under \$\{data\?\.max_price_per_hour\}\/h right now/);
  assert.match(emptyOffersBranch,
    /<SettingsLink section="training" focus="cloud-max-price-per-hour">\s*increase the price cap in Settings\s*<\/SettingsLink>/);
  assert.equal([...panel.matchAll(/focus="cloud-max-price-per-hour"/g)].length, 1,
    'the price-cap link must appear only in the tiers.length === 0 branch');
});

test('mode persistence is atomic and the incompatible fallback is not optimistic', () => {
  assert.match(panel, /setDatasetTrainingMode\?\.\(TRAINING_MODE_LORA, nextSelection\)/);
  assert.match(panel, /setDatasetTrainingMode\?\.\(\s*TRAINING_MODE_LORA,\s*fullTransformerSelection/);
  const fallback = panel.slice(panel.indexOf('// Family, Krea variant'), panel.indexOf('const toggleSliderMode'));
  const persistAt = fallback.indexOf('await ds.setDatasetTrainingMode');
  const showLoraAt = fallback.indexOf('setTrainingMode(TRAINING_MODE_LORA)');
  assert.ok(persistAt >= 0 && showLoraAt > persistAt,
    'the UI must not claim LoRA before the save resolves');
  assert.match(fallback, /const info = await ds\.trainBaseInfo/);
  const modeChange = panel.slice(panel.indexOf('const onTrainingModeChange'), panel.indexOf('const onTrainingModeKeyDown'));
  assert.match(modeChange, /disableSliderForFullTransformer: nextMode === TRAINING_MODE_FULL_TRANSFORMER/);
  assert.match(modeChange, /saved\.slider\?\.enabled !== false/);
  assert.doesNotMatch(modeChange, /saveSlider\(/,
    'switching to full must not issue or roll back a separate Slider request');
  assert.ok(modeChange.indexOf('setTrainingMode(canonicalMode)') > modeChange.indexOf('saved.slider?.enabled !== false'),
    'the UI must wait for the canonical mode + disabled Slider response');
});

test('full run cards surface Hub status and suppress LoRA-only actions', () => {
  assert.match(runsPage, /function FullArtifactStatus/);
  assert.match(runsPage, /!fullModel && run\.checkpoint_ready/);
  assert.match(runsPage, /!fullModel && run\.dataset_id != null/);
  assert.match(runsPage, /!fullModel && run\.record_id != null/);
  assert.match(runsPage, /isFullTransformerRun\(run\) && \([\s\S]*?<FullArtifactStatus run=\{run\}/);
  // The stop consequence moved into pages/cloudStopDialog.js when the confirm
  // became a dialog — same sentence, one testable home, still the sentence a
  // full-model run must show before it can lose its latest checkpoint.
  assert.match(stopDialog, /AI Toolkit uploads the full model to Hugging Face only when the run finishes cleanly/);
  assert.match(runsPage, /<CloudStopDialog run=\{stopTarget\}/);
  assert.match(runsPage, /fullModel=\{!!stopTarget && isFullTransformerRun\(stopTarget\)\}/);
  // The dataset panel used a window.confirm that could not carry the ban tick;
  // a full-model stop from there must go through the same dialog.
  assert.match(panel, /<CloudStopDialog run=\{cloudStopTarget\}/);
  assert.match(panel, /fullModel=\{!!cloudStopTarget && isFullTransformerRun\(cloudStopTarget\)\}/);
  assert.match(runsPage, /Verify Hugging Face delivery/);
  assert.match(runsPage, /Retry pod cleanup/);
  assert.match(runsPage, /\/api\/dataset\/train\/cloud\/recheck-delivery/);
  assert.match(runsPage, /Inspect Hugging Face repository \(delivery not verified\)/);
});

test('stopping a cloud run from either surface can ban the host', () => {
  // The ban lived on the Runs hub only, so a pod stuck on boot — watched from
  // the dataset training panel, next to "Stop cloud run" — could be killed
  // without ever offering "Do not rent this machine again".
  const banPost = /banHost \? \{ run_id: run\.run_id, ban_host: true \}/;
  assert.match(panel, banPost);
  assert.match(runsPage, banPost);
  assert.match(panel, /setCloudStopTarget\(cloudActiveHere\)/);
  assert.doesNotMatch(panel, /Stop this full-model training run/);
});

test('user-facing full-model recovery copy never falls back to dense terminology', () => {
  assert.match(panel, /the latest full-model checkpoint may not have reached Hugging Face/);
  assert.match(runsPage, /Verify or recover the full-model weights on Hugging Face/);
  assert.doesNotMatch(`${panel}\n${runsPage}`, /\bdense (?:checkpoint|weights)\b/i);
});
