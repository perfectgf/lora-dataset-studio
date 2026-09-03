/* ⚡ The Video Test Studio's quick prompts — the contract, not the taste.
 *
 * What is worth pinning here is not "there are N chips" (a number a test can
 * assert without ever reading one) but the two things that break silently:
 * a preset that points at a start frame in a mode that has none, and a picker
 * that eats a half-written prompt. Both are enumerated over EVERY preset. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  VIDEO_QUICK_PROMPT_CATEGORIES, allQuickPrompts, promptForMode, appendQuickPrompt,
} from './videoPromptPresets.js';

test('every category carries an id, a label, an emoji and at least one preset', () => {
  const ids = new Set();
  for (const cat of VIDEO_QUICK_PROMPT_CATEGORIES) {
    assert.match(cat.id, /^[a-z]+$/, `category id: ${cat.id}`);
    assert.ok(!ids.has(cat.id), `duplicate category id: ${cat.id}`);
    ids.add(cat.id);
    assert.ok(cat.label && cat.emoji, `category ${cat.id} needs a label and an emoji`);
    assert.ok(cat.prompts.length > 0, `category ${cat.id} has no preset`);

    const labels = new Set();
    for (const p of cat.prompts) {
      assert.ok(p.label?.trim(), `a preset of ${cat.id} has no label`);
      assert.ok(p.prompt?.trim().length > 20, `preset ${cat.id}/${p.label} is too thin to be a prompt`);
      assert.ok(!labels.has(p.label), `duplicate label ${p.label} in ${cat.id}`);
      labels.add(p.label);
    }
  }
});

test('NO preset asks text-to-video to honour a start frame — every one of them checked', () => {
  /* The scenarios and the timeline point at the frame the way H3's template
     does ("the subject from <Picture 1>", "the frame opens on image 1"). In
     t2v there is no picture, and a prompt that names one asks the sampler for
     a frame nobody supplied. This walks the WHOLE set rather than the two
     presets that happen to have the reference today — a new scenario written
     in the same idiom is caught the day it is added, not the day a clip comes
     back wrong. */
  const all = allQuickPrompts();
  assert.ok(all.length > 0, 'no presets to check — the enumeration is broken');

  const referencesAPicture = /<Picture\s*\d+>|\bon image \d+\b/i;
  const offenders = all
    .map((p) => ({ ...p, t2v: promptForMode(p.prompt, 't2v') }))
    .filter((p) => referencesAPicture.test(p.t2v))
    .map((p) => `${p.category}/${p.label}`);
  assert.deepEqual(offenders, [], 'these presets still name a picture in t2v');

  // …and the ones that DO carry the reference must really be adapted, so the
  // test above cannot pass by the regex never matching anything in the first
  // place (that is how a guard goes quietly green forever).
  const carriers = all.filter((p) => referencesAPicture.test(p.prompt));
  assert.ok(carriers.length >= 2,
    'no preset references a start frame any more — is the t2v guard still needed?');
  for (const p of carriers) {
    assert.notEqual(promptForMode(p.prompt, 't2v'), p.prompt,
      `${p.category}/${p.label} carries a picture reference that t2v does not strip`);
  }
});

test('image-to-video gets the preset exactly as written', () => {
  for (const p of allQuickPrompts()) {
    assert.equal(promptForMode(p.prompt, 'i2v'), p.prompt, `${p.category}/${p.label} was rewritten`);
  }
});

test('a chip APPENDS on its own line and never eats what is written', () => {
  assert.equal(appendQuickPrompt('', 'Slow push-in.'), 'Slow push-in.');
  assert.equal(appendQuickPrompt('She turns.', 'Slow push-in.'), 'She turns.\nSlow push-in.');
  // the same chip twice is not a way to say it twice
  assert.equal(appendQuickPrompt('She turns.\nSlow push-in.', 'Slow push-in.'),
    'She turns.\nSlow push-in.');
  // an empty addition changes nothing, and trailing blank lines do not pile up
  assert.equal(appendQuickPrompt('She turns.', '   '), 'She turns.');
  assert.equal(appendQuickPrompt('She turns.\n\n', 'Slow push-in.'), 'She turns.\nSlow push-in.');
});

test('the picker is wired into the Motion field, and appends rather than replaces', () => {
  /* The component is JSX, which `node --test` cannot execute — so this reads
     the wiring as text, the same compromise the rest of the studio's contract
     tests make. What it guards is the promise: `setPrompt` receives the RESULT
     of appendQuickPrompt, not the chip alone. */
  const studio = fs.readFileSync(new URL('./VideoTestStudio.jsx', import.meta.url), 'utf8');
  assert.match(studio, /<VideoQuickPrompts mode=\{mode\}/);
  assert.match(studio, /onAppend=\{\(text\) => setPrompt\(\(p\) => appendQuickPrompt\(p, text\)\)\}/);

  const picker = fs.readFileSync(new URL('./VideoQuickPrompts.jsx', import.meta.url), 'utf8');
  // finger-sized below lg, on every control the picker draws (the responsive
  // contract: fixed with `min-h-10 lg:min-h-0`, never by exempting a chip)
  assert.equal((picker.match(/min-h-10 lg:min-h-0/g) || []).length, 2);
  // The section is measured by the probe as part of the `video` state (it sits
  // inside the Motion panel, which that state already opens). The testid is for
  // automation that needs to FIND it — it is not a probe marker, and calling it
  // one here would be a comment that lies.
  assert.match(picker, /data-testid="video-quick-prompts"/);
});
