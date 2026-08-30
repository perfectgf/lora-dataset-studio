import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  CAPTION_ORIGINS, CAPTION_ORIGIN_UNRECORDED, captionIsAsserted, captionOriginInfo,
  captionOriginTooltipLine,
} from './captionOrigin.js';

/* The values are STORED IN USER DATABASES (services/caption_origin.py says so in
   its header). This test is the alias rule, enforced: renaming one here without an
   alias would silently stop matching rows that already exist. */
test('the three stored values are the three this module knows, by key', () => {
  assert.deepEqual(CAPTION_ORIGINS.map((o) => o.key),
                   ['asserted', 'joycaption', 'ollama']);
  for (const o of CAPTION_ORIGINS) {
    assert.equal(typeof o.chip, 'string');
    assert.ok(o.chip.length, o.key);
    assert.ok(o.short.length, o.key);
    assert.ok(o.title.length > o.short.length, `${o.key}: the tooltip must explain`);
  }
});

test('a human-written caption is never folded into the two engines', () => {
  assert.equal(captionIsAsserted('asserted'), true);
  assert.equal(captionIsAsserted('joycaption'), false);
  assert.equal(captionIsAsserted('ollama'), false);
  assert.equal(captionIsAsserted(null), false);
  // …and its wording says WHO, not "manual" — the distinction the forced pass acts on.
  assert.match(captionOriginInfo('asserted').short, /you/i);
});

test('the two engines are named apart, and neither answers for the other', () => {
  // The claim is that the two are DISTINGUISHABLE, not that one of them is spelled
  // "Ollama". The stored value keeps that spelling (it is a frozen column on both
  // surfaces) but the label had to stop naming a product an LM Studio user does not
  // run — so the assertion moved onto the property, and off the brand.
  const joy = captionOriginInfo('joycaption').short;
  const llm = captionOriginInfo('ollama').short;
  assert.match(joy, /JoyCaption/);
  assert.doesNotMatch(llm, /JoyCaption/);
  assert.notEqual(joy, llm, 'the two engines wear the same label');
  assert.match(llm, /local LLM/i);
});

/* THE STATE THAT COSTS THE USER WORK IF IT IS READ WRONG. Every row written before
   the column existed carries NULL, and NULL is not "a model wrote it". */
test('no recorded origin reads as unrecorded, and says it is not an attribution', () => {
  for (const blank of [null, undefined, '', '   ']) {
    const info = captionOriginInfo(blank);
    assert.equal(info.known, false, JSON.stringify(blank));
    assert.equal(info.key, '');
    assert.equal(info.short, CAPTION_ORIGIN_UNRECORDED.short);
  }
  // The tooltip has to REFUSE the machine reading in words, not just omit it.
  assert.match(CAPTION_ORIGIN_UNRECORDED.title, /not the same as/i);
  assert.doesNotMatch(CAPTION_ORIGIN_UNRECORDED.short, /JoyCaption|Ollama/);
});

test('an origin this build does not know is shown under its own name, not dropped', () => {
  const info = captionOriginInfo('some-future-engine');
  assert.equal(info.known, true);
  assert.equal(info.key, 'some-future-engine');
  assert.match(info.short, /some-future-engine/);
  // The one thing it must NOT do is claim to be one of the engines we do know.
  assert.doesNotMatch(info.short, /JoyCaption|Ollama vision/);
});

test('stored values are matched case- and whitespace-insensitively', () => {
  assert.equal(captionOriginInfo(' OLLAMA ').key, 'ollama');
  assert.equal(captionOriginInfo('JoyCaption').key, 'joycaption');
});

/* An empty caption has no author whatever the column says — the same rule
   caption_origin.is_protected enforces server-side (text, not marker). */
test('a caption with no text is attributed to nobody', () => {
  assert.equal(captionOriginTooltipLine('', 'ollama'), '');
  assert.equal(captionOriginTooltipLine('   ', 'asserted'), '');
  assert.equal(captionOriginTooltipLine('a woman on a beach', 'ollama'),
               captionOriginInfo('ollama').short);
});

/* THE SURFACES, asserted against the module rather than described in prose: a
   feature that ships and is invisible is indistinguishable from one that did not
   ship, and that has already cost this app twice. */
test('the four reading surfaces really consume the origin', () => {
  const surfaces = {
    'components/bank/BankTile.jsx': /captionOriginTooltipLine\(/,
    'components/dataset/DatasetGridItem.jsx': /captionOriginInfo\(img\.caption_origin\)/,
    'components/dataset/CaptionEditorDialog.jsx': /captionOriginInfo\(/,
    'components/dataset/DatasetWorkspace.jsx': /captionOriginInfo\(img\.caption_origin\)/,
  };
  for (const [rel, pattern] of Object.entries(surfaces)) {
    const src = readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');
    assert.match(src, pattern, rel);
    assert.match(src, /from '\.\.\/\.\.\/utils\/captionOrigin\.js'/, `${rel}: import`);
  }
});

/* The expanded editor is the one surface with a DRAFT, so it is the one that can
   put a stale author on new words. It must send the saved text through, not the
   textarea's current value. */
test('the expanded editor compares the draft to the SAVED text before attributing', () => {
  const src = readFileSync(
    new URL('../components/dataset/CaptionEditorDialog.jsx', import.meta.url), 'utf8');
  assert.match(src, /authorshipNote\(draft, initialCaption, captionOrigin\)/);
  assert.match(src, /authorshipNote\(shortDraft, initialShortCaption, ?\s*shortCaptionOrigin\)/);
  // The short caption is attributed by its OWN column, never by the long one's.
  assert.doesNotMatch(src, /authorshipNote\(shortDraft, initialShortCaption, captionOrigin\)/);
});
