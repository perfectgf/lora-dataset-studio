/**
 * The improve instruction is editable FROM the improve button — proved in the
 * DOM, not in a regex over the source.
 *
 * The pure decisions (what is read, what is written, when) are covered by
 * src/components/dataset/kleinImproveEditor.test.js. What that file cannot see
 * is whether the panel actually RENDERS: KleinImproveNote pulls in
 * PromptOverrideField, KleinModelSetting and SettingsLink, and the editor branch
 * is a branch no user click can reach inside `node --test`. mountJsx executes
 * the component, so a ReferenceError or a bad prop in that branch becomes a
 * failing test here instead of a blank rail on someone's screen.
 *
 * ⚠️ Effects never run under mountJsx and nothing fetches, so the loaded state
 * is reached by seeding the module cache (`_seedKleinImproveNoteCache`) exactly
 * as a settled /api/settings would have. Reset it between tests: it is shared.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'
import { readSource } from './support/readSource.mjs'

const { default: KleinImproveNote, _resetKleinImproveNoteCache, _seedKleinImproveNoteCache } =
  await import('../src/components/dataset/KleinImproveNote.jsx')
const { IMPROVE_SCOPE_NOTE, IMPROVE_OFF_NOTE } =
  await import('../src/components/dataset/kleinImproveEditor.js')

const SHIPPED = 'add detailed texture, add sharp details, add candid shot, add soft focus effect'

const payload = (identityPrompts = {}) => ({
  config: { identity_prompts: identityPrompts },
  identity_prompt_defaults: { klein_improve: SHIPPED },
})

/** Render the note as if /api/settings had answered with `identityPrompts`. */
const render = (identityPrompts, props = {}) => {
  if (identityPrompts === null) _resetKleinImproveNoteCache()
  else _seedKleinImproveNoteCache(payload(identityPrompts))
  const html = renderToStaticMarkup(
    createElement(KleinImproveNote, { subjectType: 'human', ...props }))
  _resetKleinImproveNoteCache()
  return html
}

/** The markup is HTML-escaped; assertions are about the TEXT the user reads. */
const text = (html) => html
  .replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&#39;/g, "'")
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#x2F;/g, '/')
  .replace(/&amp;/g, '&')

/** What a <textarea> actually holds, decoded. */
const textareaValue = (html) => {
  const m = html.match(/<textarea[^>]*>([\s\S]*?)<\/textarea>/)
  return m ? text(m[1]) : null
}

test('closed by default, but the way in is on screen and named', () => {
  const html = render({})
  assert.match(html, /data-testid="klein-improve-edit-toggle"/,
    'the editor has to be reachable from the note itself')
  assert.doesNotMatch(html, /data-testid="klein-improve-editor"/,
    'a note is a line, not a form: the box stays out of the rail until asked for')
  // Settings stays reachable: the per-subject identity prompts and the four
  // strength knobs are not in this panel and never will be.
  assert.match(html, /focus=identity-prompt-klein-improve/)
  assert.match(html, /focus=klein-improve-strength/)
})

test('the box holds the SHIPPED text when nothing is overridden', () => {
  const html = render({}, { defaultEditorOpen: true })
  assert.match(html, /data-testid="klein-improve-editor"/)
  assert.equal(textareaValue(html), SHIPPED,
    'an empty box would be indistinguishable from "no instruction"')
  // Following the default is not an override, so there is nothing to reset TO.
  assert.doesNotMatch(text(html), /Reset to default/,
    'the reset button is itself the "you changed this" marker')
  assert.match(text(html), /Following the built-in default/)
})

test('the box holds the OVERRIDE when there is one, and offers the way back', () => {
  const html = render({ klein_improve: 'keep it a flat drawing, no skin texture' },
    { defaultEditorOpen: true })
  assert.equal(textareaValue(html), 'keep it a flat drawing, no skin texture')
  assert.match(text(html), /Reset to default/)
  assert.match(text(html), /Custom override/)
})

test('turning the instruction off stays reachable here', () => {
  // Half of what the old "Edit or turn off this instruction →" link promised.
  const on = render({}, { defaultEditorOpen: true })
  assert.match(on, /type="checkbox"[^>]*checked/)
  assert.doesNotMatch(text(on), new RegExp(IMPROVE_OFF_NOTE.slice(0, 30)))

  const off = render({ klein_improve_enabled: false }, { defaultEditorOpen: true })
  assert.doesNotMatch(off, /type="checkbox"[^>]*checked/)
  assert.match(text(off), new RegExp(IMPROVE_OFF_NOTE.slice(0, 30)),
    'and it says what "off" actually does — upscale, nothing else')
  assert.match(off, /<textarea[^>]*disabled/,
    'editing a text that is not being sent would be a lie about the outcome')
})

test('the panel states, in the markup, that the change is app-wide', () => {
  // Not a hover and not a tooltip: a global control inside a dataset screen
  // reads as per-dataset until something says otherwise.
  const html = text(render({}, { defaultEditorOpen: true }))
  assert.ok(html.includes(IMPROVE_SCOPE_NOTE), 'the scope sentence must be rendered')
  assert.match(IMPROVE_SCOPE_NOTE, /every dataset/i)
})

test('the quoted line above the box shows the same text the box holds', () => {
  // Quoting one sentence while editing another would be worse than the link
  // this panel replaced.
  const html = text(render({ klein_improve: 'keep it a flat drawing' },
    { defaultEditorOpen: true }))
  assert.ok(html.includes('keep it a flat drawing'))
  assert.ok(html.includes('Improve asks Klein to:'))
  assert.equal(textareaValue(render({ klein_improve: 'keep it a flat drawing' },
    { defaultEditorOpen: true })), 'keep it a flat drawing')
})

test('with no settings answer yet, the panel refuses to open', () => {
  const html = render(null, { defaultEditorOpen: true })
  assert.doesNotMatch(html, /data-testid="klein-improve-editor"/,
    'an editor rendered before /api/settings answers would show a box whose '
    + 'content is a guess')
  assert.match(html, /data-testid="klein-improve-edit-toggle"[^>]*disabled/,
    'and the way in is disabled rather than opening onto nothing')
})

test('the note renders on every host that mounts it, open and closed', () => {
  // Four mount sites (dataset lightbox rail + bottom bar, bulk toolbar,
  // generated-image lightbox) and two of them pass no datasetId.
  for (const props of [
    {}, { datasetId: 7 }, { subjectType: 'anime' }, { subjectType: 'anime', datasetId: 7 },
    { className: 'w-full border-t' },
  ]) {
    for (const open of [false, true]) {
      assert.doesNotThrow(() => render({}, { ...props, defaultEditorOpen: open }),
        `threw for ${JSON.stringify(props)} open=${open}`)
    }
  }
})

test('the anime caution survives the editor being open', () => {
  const html = text(render({}, { subjectType: 'anime', defaultEditorOpen: true }))
  assert.match(html, /subject type is set to anime/,
    'the warning that explains WHY the box needs editing must stay above it')
})

/* ── The picked preset's chain, ON SCREEN ────────────────────────────────────
   The pure half (which rows, which index, what a slider writes) lives in
   kleinImproveEditor.test.js. What only a render can answer is whether the
   sliders are actually THERE, holding the stored strengths — the panel's whole
   claim is that you no longer have to open Settings to see them. */

const KLEIN = {
  improve_lora_preset: 'Real',
  consistency_lora: 'klein/Flux2-Klein-9B-consistency-V2.safetensors',
  generation_lora_presets: [
    { name: 'Real', loras: [
      { file: 'klein/details.safetensors', strength: 0.15 },
      { file: '', strength: 0.6 },
      { file: 'klein/realistic.safetensors', strength: 0.25 },
    ] },
    { name: 'Soft', loras: [] },
  ],
}

/** Render the note as if /api/settings had answered with this klein section. */
const renderKlein = (klein, props = {}) => {
  _seedKleinImproveNoteCache({
    config: { identity_prompts: {}, klein },
    identity_prompt_defaults: { klein_improve: SHIPPED },
  })
  const html = renderToStaticMarkup(
    createElement(KleinImproveNote, { subjectType: 'human', ...props }))
  _resetKleinImproveNoteCache()
  return html
}

/** Every rendered slider, as {label, value} — read from the markup, so a row
 *  that is not drawn cannot be asserted into existence. */
const sliders = (html) => [...html.matchAll(/<input[^>]*type="range"[^>]*>/g)]
  .map((m) => ({
    label: (m[0].match(/aria-label="([^"]*)"/) || [])[1],
    value: (m[0].match(/value="([^"]*)"/) || [])[1],
  }))

test('the picked preset shows its LoRAs, each with the strength it will run at', () => {
  const html = renderKlein(KLEIN)
  assert.match(html, /data-testid="klein-improve-lora-chain"/)
  const drawn = sliders(html)
  // EXERCISED, not counted: each slider is matched to the file it belongs to
  // and to the number stored for it.
  assert.deepEqual(drawn.map((s) => s.value), ['0.15', '0.25'])
  assert.match(drawn[0].label, /klein\/details\.safetensors/)
  assert.match(drawn[1].label, /klein\/realistic\.safetensors/)
  assert.equal(drawn.length, 2,
    'the blank third slot has nothing to tune and must not be drawn')
  assert.ok(text(html).includes('klein/details.safetensors'),
    'the file is named, not just a slider with no subject')
})

test('the chain says whose strengths those are, and where the list is built', () => {
  const html = text(renderKlein(KLEIN))
  assert.match(html, /Strengths belong to the preset/,
    'a dial inside a dataset screen reads as per-dataset until it says otherwise')
  assert.match(html, /Add or remove LoRAs/)
  assert.match(renderKlein(KLEIN), /focus=klein-generation-lora-presets/,
    'the link lands ON the preset card, not at the top of Engines')
})

test('a row the engine already loads says so where it is being tuned', () => {
  // The server DROPS that row; a slider that moves nothing, with nothing on
  // screen to explain it, is the silence the Settings card already warns about.
  const html = text(renderKlein({
    ...KLEIN,
    generation_lora_presets: [{ name: 'Real', loras: [
      { file: 'klein/Flux2-Klein-9B-consistency-V2.safetensors', strength: 0.5 },
    ] }],
  }))
  assert.match(html, /consistency/i)
  assert.match(html, /Ignored|already loads/i)
})

test('an empty preset says where its files come from instead of drawing nothing', () => {
  const html = renderKlein({ ...KLEIN, improve_lora_preset: 'Soft' })
  assert.equal(sliders(html).length, 0)
  assert.match(text(html), /chains no LoRA yet/)
})

test('no pick, and a pick that no longer exists, draw no chain at all', () => {
  for (const pick of ['', 'Deleted last week']) {
    const html = renderKlein({ ...KLEIN, improve_lora_preset: pick })
    assert.doesNotMatch(html, /data-testid="klein-improve-lora-chain"/,
      `pick=${JSON.stringify(pick)} must not grow rows`)
  }
  // …and the stale one is still visible in the picker, so it can be cleared.
  assert.match(text(renderKlein({ ...KLEIN, improve_lora_preset: 'Deleted last week' })),
    /Deleted last week \(missing — runs as None\)/)
})

test('the chain renders on every host, with and without a dataset', () => {
  for (const props of [{}, { datasetId: 7 }, { className: 'w-full border-t' }]) {
    for (const open of [false, true]) {
      assert.doesNotThrow(() => renderKlein(KLEIN, { ...props, defaultEditorOpen: open }))
    }
  }
})

test('a row is named by the part that tells it apart, at any width', () => {
  const html = text(renderKlein({ ...KLEIN, generation_lora_presets: [
    { name: 'Real', loras: [
      { file: 'klein/very-long-name-alpha.safetensors', strength: 0.5 },
      { file: 'klein/very-long-name-beta.safetensors', strength: 0.5 },
    ] },
  ] }))
  // The folder is the half that repeats, and `truncate` cuts the other end.
  assert.match(html, /very-long-name-alpha\.safetensors/)
  assert.match(html, /very-long-name-beta\.safetensors/)
  // The whole stored path stays on the row, for the case two folders collide.
  assert.match(renderKlein(KLEIN), /title="klein\/details\.safetensors"/)
})

test('the slider writes the row it belongs to, and lands when the finger lifts', () => {
  /* What a static render cannot execute: the handler. Pinned on the source,
     and it is worth pinning — the drawn position and the STORED index differ as
     soon as a preset holds an empty slot, and the 600 ms coalescing meant a
     drag followed by ✨ Generate rendered with the previous value. */
  const src = readSource('src/components/dataset/KleinImproveNote.jsx')
  assert.match(src, /onChange=\{\(e\) => setLoraStrength\(row\.index, e\.target\.value\)\}/,
    'writing by the drawn position would move the strength onto another LoRA')
  assert.match(src, /onPointerUp=\{\(\) => saver\.current\.flush\(\)\}/)
  assert.match(src, /onKeyUp=\{\(\) => saver\.current\.flush\(\)\}/)
  // A launcher can settle every mounted panel before it starts a run.
  assert.match(src, /export function flushImproveSettings/)
  assert.match(src, /export function whenImproveSettingsSettled/)
})

test('a publish LETS GO of the drafted preset list — the app-wide value cannot be reverted', () => {
  /* Two copies of this panel are mounted at once (the grid's bulk toolbar and
     the lightbox modal). The drafted value here is the WHOLE presets array, so
     a copy that kept its snapshot would rewrite the other copy's saved strength
     from stale data — silently, app-wide. */
  const src = readSource('src/components/dataset/KleinImproveNote.jsx')
  const receive = src.split('const receive =')[1].split('loadSettings()')[0]
  assert.match(receive, /saver\.current\?\.pending\?\.presets/,
    'not while a write is still coalescing here: that is a finger on a slider')
  assert.match(receive, /presets: undefined/)
})

test('a failed save is reported with the editor CLOSED, where the sliders are', () => {
  const src = readSource('src/components/dataset/KleinImproveNote.jsx')
  // From the chain block to the dial that follows it — the error line has to
  // live INSIDE that span, not in the instruction editor that starts closed.
  const chain = src.split('data-testid="klein-improve-lora-chain"')[1]
    .split('Output size, MP')[0]
  assert.match(chain, /error &&/,
    'a slider sitting on a value the server never stored must not stay silent')
})
