// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource, bankWorkspaceSource } from './bankTreeSource.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { markdownHeadingId } from '../../utils/headingId.js';

/* Source-text contract on the 🏷️ Caption options row, in the house style used by
   curation-ui.test.js and pipeline-ui.test.js: the file is read and asserted against,
   because these are wiring and layout facts that no pure function can hold. */

const ws = bankTreeSource();

test('only the model select is width-capped, and the other four are not truncated', () => {
  // A <select> sizes itself to its widest option and has min-width:auto, so it does not
  // shrink; the ① Analyze zone has no overflow-x container, so an unbounded one pushes
  // the whole page into a horizontal scroll on a phone. But the cap has a cost: at
  // 11rem the vocabulary and length selects render "Use default (Settings ▸ Cap⌄" and
  // "Standard — the prompt as⌄", and neither carried any bound before this row existed.
  // Only the MODEL select needs it — Ollama refs run long and it overflows on its own
  // (measured: 430 px asked at a 360 px width). The other four take max-w-full with a
  // 16rem ceiling from sm up: never wider than the column, readable everywhere else.
  // 'Caption scope' is no longer a <select>: it became the radio list in the
  // window's THIS RUN block, where each pile carries its own count. The other four
  // moved inside the window unchanged, keeping the width rules measured here.
  const capped = { 'Caption vision model': /max-w-\[11rem\]/ };
  for (const label of ['Caption engine', 'Caption vision model',
    'Caption vocabulary register', 'Caption length']) {
    const i = ws.indexOf(`aria-label="${label}"`);
    assert.ok(i > 0, `${label} select is missing`);
    // The className sits within the same element; look at the tag around it.
    const tagStart = ws.lastIndexOf('<select', i);
    const tagEnd = ws.indexOf('>', ws.indexOf('className=', i));
    const tag = ws.slice(tagStart, tagEnd);
    if (capped[label]) {
      assert.match(tag, capped[label], `${label} select lost its width bound`);
    } else {
      assert.ok(!/max-w-\[11rem\]/.test(tag),
        `${label} select is capped at 11rem and truncates its own options`);
      assert.match(tag, /sm:max-w-\[16rem\]/,
        `${label} select has no width bound at all`);
    }
  }
});

test('the caption options live INSIDE the caption window, not spread under the panel', () => {
  // WHERE THEY WENT, and why. They used to wrap onto their own row beneath the pass
  // buttons, where four <select>s and a destructive button read as bank-wide settings
  // rather than as one run's options — and where they greyed out with 🏷️ Caption on a
  // fully captioned bank. The maintainer's ask was literal: "this way we can gather
  // all the caption options". They are now the window's own block.
  const controls = ws.indexOf('const captionRunControls = (');
  assert.ok(controls > 0, 'the caption run controls block is missing');
  assert.ok(!ws.includes('<GroupLabel>Caption options</GroupLabel>'),
    'the old options row is still rendered under the panel');
  // The window is what the pass button opens…
  assert.match(ws, /onClick=\{\(\) => onPassOpen\('caption'\)\}/);
  // …and the controls are handed to PassDialog, not rendered on the panel.
  // (🔤 Find text hands its own controls through the same children seat, so
  // the ternary grew a second arm — the invariant is that caption's controls
  // are the caption window's children, wherever the chain ends.)
  assert.match(ws, /passOpen === 'caption' \? captionRunControls\s*\n?\s*:/);
});

test('every new option is spread-if-set, so an untouched run posts the old body', () => {
  // The caption request builder lives in useCaptionOptions.js since hook
  // wave 6; positional slices must read ONE file (see bankTreeSource.js),
  // so this one reads the hook directly. passBody below stays workspace.
  const hookFile = fs.readFileSync(
    new URL('./useCaptionOptions.js', import.meta.url), 'utf8');
  const wsFile = bankWorkspaceSource();
  const opts = hookFile.slice(hookFile.indexOf('const captionRunOptions'));
  assert.match(opts, /\.\.\.\(captionEngine \? \{ backend: captionEngine \} : \{\}\)/);
  assert.match(opts, /\.\.\.\(captionModel \? \{ ollama_model: captionModel \} : \{\}\)/);
  // The scope now rides through the SHARED body builder every pass uses, and it is
  // spread-if-set there — a run left on the default omits `statuses` entirely, and a
  // selection sends image_ids instead. The window never produces both, because the
  // selection and the piles are radio buttons in one group.
  const body = wsFile.slice(wsFile.indexOf('const passBody'), wsFile.indexOf('const runPass'));
  assert.match(body, /\.\.\.\(statuses \? \{ statuses \} : \{\}\)/);
  assert.match(body, /imageIds === 'selection' && selected\.size \? \{ image_ids/);
});

test('the engine picker never offers "none" — captioning with nothing is not a pass', () => {
  const i = ws.indexOf('aria-label="Caption engine"');
  const block = ws.slice(i, i + 600);
  assert.match(block, /ENGINE_OPTIONS\.filter\(\(o\) => o\.id !== 'none'\)/);
});

test('the model picker is inert unless the engine can reach Ollama, and keeps an unknown model', () => {
  assert.match(ws, /const ollamaPicksApply = OLLAMA_RELEVANT\.has\(captionEngine\)/);
  const i = ws.indexOf('aria-label="Caption vision model"');
  assert.match(ws.slice(i - 400, i + 400), /disabled=\{live \|\| !ollamaPicksApply\}/);
  // A model pulled elsewhere must stay selectable rather than be dropped in silence.
  assert.match(ws, /captionModelChoices = captionModel && !ollamaModels\.includes\(captionModel\)/);
});

test('the model list comes from its own always-200 endpoint, not from capabilities', () => {
  // caps.ollama carries the CONFIGURED model, never the installed list.
  assert.match(ws, /apiFetch\('\/api\/ollama\/models'\)\.catch\(\(\) => \(\{ models: \[\] \}\)\)/);
});

test('the explicit warning judges the model that will RUN, and points at a real place', () => {
  // Warning about the configured model while the run uses an override is worse than
  // not warning at all.
  assert.match(ws, /const visionModel = captionModel \|\| caps\.ollama\?\.vision_model \|\| ''/);
  // The old sentence sent people to Settings ▸ Captioning & quality, which holds the
  // ENGINE selector; the vision model field lives in Local tools.
  const warn = ws.slice(ws.indexOf("captionVocab === 'explicit'"),
    ws.indexOf("captionVocab === 'explicit'") + 1200);
  assert.ok(!/Captioning &amp; quality/.test(warn),
    'the explicit warning still points at the wrong Settings section');
  assert.match(warn, /section="local-tools" focus="ollama-vision-model"/);
});

test('no surface in the bank sends people to the wrong tab for the vision model', () => {
  const wm = fs.readFileSync(new URL('./bankWatermark.js', import.meta.url), 'utf8');
  for (const [name, src] of [['BankWorkspace.jsx', ws], ['bankWatermark.js', wm]]) {
    for (const m of src.matchAll(/[^\n]*vision model[^\n]*/gi)) {
      assert.ok(!/Settings ▸ Captioning/.test(m[0]),
        `${name}: "${m[0].trim()}" points at the engine section, not Local tools`);
    }
  }
});

/* 🔄 RE-CAPTION — the destructive twin. Wiring facts no pure function can hold. */

test('the normal caption pass never sends force', () => {
  // The whole safety story rests on this: 🏷️ Caption fills blanks and nothing else.
  // A stray `force` here would turn the everyday button into the destructive one.
  const call = ws.slice(ws.indexOf('const captionRunOptions'),
    ws.indexOf('const cancelJob'));
  assert.ok(!/force/.test(call), '🏷️ Caption must never post force');
});

test('re-caption posts force, and asks before it does', () => {
  const call = ws.slice(ws.indexOf('const startRecaption'),
    ws.indexOf('const startRecaption') + 2200);
  // The confirmation comes FIRST — after the post there is nothing left to confirm.
  const confirmAt = call.indexOf('window.confirm(captionRecaptionConfirmation');
  const postAt = call.indexOf('postJson');
  assert.ok(confirmAt > 0, 're-caption does not confirm');
  assert.ok(postAt > confirmAt, 'the request is built before the question is asked');
  assert.match(call, /force: true/);
  // …and it re-checks the inert reason itself, so a stale click cannot slip through.
  assert.match(call, /if \(captionRecaptionDisabledReason\(/);
});

test('re-caption never carries a selection', () => {
  // A selection can span pages that were never loaded, so the overwrite count is
  // unknowable client-side. This button goes inert instead of quoting a guess —
  // sending image_ids would be exactly the number-that-differs-from-the-action bug.
  const call = ws.slice(ws.indexOf('const startRecaption'),
    ws.indexOf('const startRecaption') + 2200);
  assert.ok(!/image_ids/.test(call), 're-caption must not post image_ids');
  assert.match(ws, /const recaptionInert = captionRecaptionDisabledReason\(\s*selected\.size/);
});

test('re-caption carries the same per-run options as the normal pass', () => {
  // Making the engine and model reachable on a finished bank IS the feature; a
  // re-caption that dropped them would redo the captions with the old model.
  const call = ws.slice(ws.indexOf('const startRecaption'),
    ws.indexOf('const startRecaption') + 2200);
  // The four per-run dials ride through the SAME helper the normal pass uses, so
  // they cannot come to disagree — which is exactly how a re-caption would quietly
  // redo everything with the old model.
  assert.ok(call.includes('...captionRunOptions()'), 're-caption drops the run options');
  assert.ok(call.includes('statuses: captionScopeStatuses(captionScope)'),
    're-caption drops the scope');
  // captionRunOptions lives in useCaptionOptions.js since hook wave 6; a
  // positional slice needs ONE file (see bankTreeSource.js), so read it.
  const hookSrc = fs.readFileSync(
    new URL('./useCaptionOptions.js', import.meta.url), 'utf8');
  const opts = hookSrc.slice(hookSrc.indexOf('const captionRunOptions'));
  for (const key of ['vocabulary: captionVocab', 'length: captionLength',
    'backend: captionEngine', 'ollama_model: captionModel']) {
    assert.ok(opts.includes(key), `the shared run options drop ${key}`);
  }
});

test('the button and the amber warning both live in the caption window', () => {
  // 🔄 Re-caption is a SECOND launch button in the window's footer, next to the
  // normal one — not a tenth button on the pass row, and not a pass of its own.
  const secondary = ws.indexOf('const captionSecondary = (');
  assert.ok(secondary > 0, 'the re-caption footer block is missing');
  const button = ws.indexOf('{captionRecaptionLabel(counts, captionScope, recaptionInert,');
  assert.ok(button > secondary, 're-caption must sit in the window footer');
  assert.ok(ws.indexOf('onClick={startRecaption}') > secondary);
  assert.match(ws, /secondary=\{passOpen === 'caption' \? captionSecondary : null\}/);
  // The warning is rendered, and only when the helper has something to say.
  assert.match(ws, /\{recaptionNote && \(/);
  assert.match(ws, /text-amber-400\/90">\{recaptionNote\}/);
  // …and when it CANNOT run, the reason is on screen rather than only in a tooltip:
  // on a bank whose only captions are hand-written, that sentence is the sole place
  // the protection is visible.
  assert.match(ws, /\{recaptionInert && \(/);
  // The label is handed the inert reason, so a button that cannot run stops quoting
  // a number — the tooltip carries the reason instead.
  // ...and the opt-out state, so the number on the button follows the tick box
  // rather than quoting the whole pile while the run spares part of it.
  assert.match(ws, /captionRecaptionLabel\(counts, captionScope, recaptionInert,\s+captionIncludeAsserted\)/);
  assert.match(ws, /title=\{recaptionInert \|\| recaptionNote\}/);
});

test('re-caption has a help topic pointing at a real guide anchor', () => {
  // Topic data lives in help/topics/* since the 2026-08-24 registry split;
  // a topic's section is layout, not contract, so read them all as one text.
  const registry = ['settingsSections', 'workspaceSections', 'pages',
    'videoLane', 'settingsFields', 'actions']
    .map((m) => fs.readFileSync(
      new URL(`../../help/topics/${m}.js`, import.meta.url), 'utf8'))
    .join(' ');
  const i = registry.indexOf("action('bank-recaption'");
  assert.ok(i > 0, 'bank-recaption has no help topic');
  const entry = registry.slice(i, i + 1200);
  const anchor = /'using-the-app', '([a-z0-9-]+)'\)/.exec(entry);
  assert.ok(anchor, 'the topic has no guide anchor');
  const guide = fs.readFileSync(
    new URL('../../../../docs/guide/using-the-app.md', import.meta.url), 'utf8');
  // Slugified with the SHARED function the Guide itself uses, not a look-alike:
  // a private copy is exactly how an anchor comes to pass its own test and 404 in
  // the app.
  const headings = [...guide.matchAll(/^## (.+)$/gm)].map((m) => markdownHeadingId(m[1]));
  assert.ok(headings.includes(anchor[1]),
    `no guide heading resolves to #${anchor[1]}`);
});

test('the opt-out is a separate, unticked gesture that only shows when it matters', () => {
  // The protection's way out. Three properties, and each one is a way this could
  // quietly become the default instead of an exception:
  //   - it starts false, so the destructive reading is never what you get by
  //     doing nothing;
  const boot = ws.slice(ws.indexOf('const [captionIncludeAsserted'),
    ws.indexOf('const [captionIncludeAsserted') + 120);
  assert.match(boot, /useState\(false\)/);
  //   - the key is spread in ONLY when ticked, so an untouched panel posts the
  //     same body it posted before this existed;
  const call = ws.slice(ws.indexOf('const startRecaption'),
    ws.indexOf('const startRecaption') + 2200);
  assert.match(call, /captionIncludeAsserted \? \{ include_asserted: true \} : \{\}/);
  //   - and the control is rendered only when the helper has a label, i.e. only
  //     when there is something to protect.
  assert.match(ws, /\{includeAssertedLabel && \(/);
  assert.match(ws, /const includeAssertedLabel = captionIncludeAssertedLabel\(/);
  // The confirmation is handed the same state, so what it says matches what runs.
  assert.match(call, /captionRecaptionConfirmation\(counts, captionScope,\s+captionIncludeAsserted\)/);
});
