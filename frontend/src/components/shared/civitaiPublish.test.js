import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CIVITAI_API, civitaiLinkLine, civitaiTarget, civitaiTargetKnown, civitaiVerbRefusal,
  draftFormFrom, draftFormRefusal, jobOutcome, jobPhaseLabel, pageVersionOptions, preselectVersion,
} from './civitaiPublish.js';

test('a looked-up page becomes a version pick: the address\'s version first, else the newest', () => {
  const page = { versions: [
    { id: 3274157, name: 'run 160 - step 2750', base_model: 'Krea 2' },
    { id: 3274154, name: 'run 156 - step 1500', base_model: 'Krea 2' },
  ] };
  assert.deepEqual(pageVersionOptions(page).map((o) => o.label),
    ['run 160 - step 2750 (#3274157) · Krea 2', 'run 156 - step 1500 (#3274154) · Krea 2']);
  assert.equal(preselectVersion(page, 3274154), 3274154);
  assert.equal(preselectVersion(page, '3274154'), 3274154, 'an id read from a URL is a string');
  // A version the page does not have is not an error here: the newest is offered.
  assert.equal(preselectVersion(page, 3274119), 3274157);
  assert.equal(preselectVersion(page, null), 3274157);
  assert.equal(preselectVersion({ versions: [] }, 1), null);
  assert.deepEqual(pageVersionOptions(null), []);
  assert.equal(CIVITAI_API.page('https://civitai.red/models/1?modelVersionId=2'),
    '/api/civitai/page?ref=https%3A%2F%2Fcivitai.red%2Fmodels%2F1%3FmodelVersionId%3D2');
});

test('the image door reads the checkpoint stamped on the row; the popover door reads node + pill', () => {
  // A picture has no file name of its own: it carries the DEPLOYED name it ran
  // with, which the server resolves to the save (its step block tells the
  // numbered save from the final).
  const fromImage = civitaiTarget({ kind: 'image', img: {
    id: 9, record_id: 4, step: 2500, dataset_id: 7, checkpoint: 'krea\\lora_x_000002500_Krea-2-Raw_rc158_v1.safetensors',
  } });
  assert.deepEqual(fromImage, {
    recordId: 4, step: 2500, datasetId: 7, filename: null,
    checkpoint: 'krea\\lora_x_000002500_Krea-2-Raw_rc158_v1.safetensors', imageId: 9,
  });
  const fromPill = civitaiTarget({
    kind: 'checkpoint', node: { record_id: 4, dataset_id: 7 }, pill: { step: 3000, filename: 'lora_x_000003000.safetensors' },
  });
  assert.deepEqual(fromPill, {
    recordId: 4, step: 3000, datasetId: 7, filename: 'lora_x_000003000.safetensors', checkpoint: null, imageId: null,
  });
  assert.equal(civitaiTarget(null), null);
});

test('a legacy row without a stamp stays UNKNOWN — never guessed', () => {
  const t = civitaiTarget({ kind: 'image', img: { id: 9, dataset_id: 7 } });
  assert.equal(t.recordId, null);
  assert.equal(civitaiTargetKnown(t), false);
  assert.equal(civitaiTargetKnown(civitaiTarget({ kind: 'image', img: { id: 9, record_id: 4, step: 0, dataset_id: 7 } })), true,
    'step 0 is a real step, not a missing one');
});

test('📤 is refused only for a picture with no library row', () => {
  assert.equal(civitaiVerbRefusal({ id: 12 }), null);
  assert.equal(civitaiVerbRefusal({ id: 12, record_id: null }), null, 'no stamp is answered by the picker, not by hiding the verb');
  assert.match(civitaiVerbRefusal({ url: '/x.png' }), /no library entry/);
  assert.match(civitaiVerbRefusal(null), /no library entry/);
});

test('a link is named by its model and version', () => {
  assert.equal(civitaiLinkLine({ model_name: 'Nova', version_name: 'v1 · step 2500' }), 'Nova · v1 · step 2500');
  assert.equal(civitaiLinkLine({ model_id: 42 }), 'model 42');
  assert.equal(civitaiLinkLine(null), '');
});

test('the create form starts as a DRAFT, from what the server derived', () => {
  const form = draftFormFrom({
    name: 'Nova (Krea 2)', version_name: 'v1 · step 2500', base_model: 'Krea 2',
    trained_words: ['nova'], tags: ['character', 'krea 2'], nsfw: false,
    file: { name: 'Nova_krea-2_v1_step2500.safetensors' },
  });
  assert.equal(form.publish, false);
  assert.equal(form.trained_words, 'nova');
  assert.equal(form.tags, 'character, krea 2');
  assert.equal(form.file_name, 'Nova_krea-2_v1_step2500.safetensors');
  assert.deepEqual(form.license, {
    allowNoCredit: true, allowCommercialUse: true, allowDerivatives: true, allowDifferentLicense: true,
  });
  // An empty base model is the server saying "I cannot name this lineage" —
  // it stays empty rather than defaulting to a wrong answer.
  assert.equal(draftFormFrom(null).base_model, '');
  assert.equal(draftFormFrom({ base_model: '' }).base_model, '');
});

test('the form is stopped by a missing name, an unnamed base, a missing file, or the file\'s own leak', () => {
  const ok = { name: 'Nova', base_model: 'Krea 2' };
  assert.equal(draftFormRefusal(ok, { file: { name: 'x' } }), null);
  assert.match(draftFormRefusal({ name: ' ', base_model: 'Krea 2' }, { file: { name: 'x' } }), /name/);
  assert.match(draftFormRefusal({ name: 'Nova', base_model: '' }, { file: { name: 'x' } }), /base model/);
  assert.match(draftFormRefusal(ok, { file: null }), /could not be found/);
  assert.equal(draftFormRefusal(ok, { file: { name: 'x' }, file_error: 'names this machine' }), 'names this machine');
});

test('a job phase reads as a sentence with its percentage', () => {
  assert.equal(jobPhaseLabel({ kind: 'model', phase: 'uploading', progress: 0.5 }), 'Uploading the checkpoint… 50%');
  assert.equal(jobPhaseLabel({ kind: 'post', phase: 'uploading', progress: 1 }), 'Uploading images… 100%');
  assert.equal(jobPhaseLabel({ kind: 'model', phase: 'creating' }), 'Creating the model page…');
  assert.equal(jobPhaseLabel({ kind: 'post', phase: 'creating' }), 'Creating the post…');
  assert.equal(jobPhaseLabel(null), '');
});

test('a finished job says what happened and where — draft vs published', () => {
  assert.equal(jobOutcome({ state: 'running' }), null);
  const draft = jobOutcome({ state: 'done', kind: 'model', result: { published: false, url: 'https://civitai.com/models/1/wizard?step=1' } });
  assert.match(draft.text, /Draft model page/);
  assert.equal(draft.url, 'https://civitai.com/models/1/wizard?step=1');
  const post = jobOutcome({ state: 'done', kind: 'post', result: { published: true, count: 1, url: 'https://civitai.com/posts/5' } });
  assert.equal(post.text, 'Posted 1 image on Civitai.');
  const failed = jobOutcome({ state: 'error', error: 'Civitai refused the API key', error_code: 'auth' });
  assert.deepEqual(failed, { ok: false, text: 'Civitai refused the API key', code: 'auth' });
});

test('every verb has ONE address', () => {
  assert.equal(CIVITAI_API.link(4, 2500), '/api/civitai/links/4/2500');
  // A pill names its file: two saves can share a step (the numbered one and the final).
  assert.equal(CIVITAI_API.link(4, 2500, 'lora_x.safetensors'),
    '/api/civitai/links/4/2500?filename=lora_x.safetensors');
  // A picture names the deployed LoRA it ran with instead.
  assert.equal(CIVITAI_API.link(4, 2500, null, 'krea\\lora_x_000002500_rc1_v1.safetensors'),
    '/api/civitai/links/4/2500?checkpoint=krea%5Clora_x_000002500_rc1_v1.safetensors');
  assert.equal(CIVITAI_API.draftDefaults(4, 2500, null, 'a b'),
    '/api/civitai/checkpoint/4/2500/draft-defaults?checkpoint=a+b');
  assert.equal(CIVITAI_API.draftDefaults(4, 2500, 'a b.safetensors'),
    '/api/civitai/checkpoint/4/2500/draft-defaults?filename=a+b.safetensors');
  assert.equal(CIVITAI_API.publishModel(4, 2500), '/api/civitai/checkpoint/4/2500/publish-model');
  assert.equal(CIVITAI_API.publishImages, '/api/civitai/images/publish');
});
