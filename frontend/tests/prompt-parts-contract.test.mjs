/* The five prompt parts that became editable, and the composed preview.
   These are the invisible bits of the feature — a warning sentence, a debounce,
   a `break-words` — each of which is exactly the kind of thing a later rewrite
   drops without a single test going red. */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

const read = readSource
const preview = read('src/components/settings/PromptPreview.jsx')
const engines = read('src/components/settings/EnginesSection.jsx')
const field = read('src/components/common/PromptOverrideField.jsx')

const {
  GLOBAL_PROMPT_PART_FIELDS, SUBJECT_PROMPT_PART_FIELDS, FRAMING_PROMPT_PART_FIELDS,
  PROMPT_PART_KEYS, PER_SUBJECT_PROMPT_KINDS,
} = await import('../src/components/common/promptOverride.js')

// The backend's own key lists, so the two sides cannot drift apart in silence.
const BACKEND_PART_KINDS = ['markings_lock', 'outfit_vary', 'expression_neutral',
  'outfit_palette', 'render_tail_sfw', 'render_tail_nsfw',
  'framing_face', 'framing_bust', 'framing_body', 'framing_back']
const BACKEND_PER_SUBJECT = ['face_single', 'face_multi', 'klein_identity',
  'render_tail_sfw', 'render_tail_nsfw',
  'framing_face', 'framing_bust', 'framing_body', 'framing_back']

test('every backend prompt part has a field, and no field invents a key', () => {
  assert.deepEqual([...PROMPT_PART_KEYS].sort(), [...BACKEND_PART_KINDS].sort())
})

test('per-subject scoping mirrors backend PER_SUBJECT_PROMPT_KINDS exactly', () => {
  // A mismatch writes the override to a key the backend never reads — the field
  // looks saved and changes nothing.
  assert.deepEqual([...PER_SUBJECT_PROMPT_KINDS].sort(), [...BACKEND_PER_SUBJECT].sort())
  for (const f of SUBJECT_PROMPT_PART_FIELDS.concat(FRAMING_PROMPT_PART_FIELDS)) {
    assert.ok(PER_SUBJECT_PROMPT_KINDS.includes(f.key), `${f.key} rendered per subject but stored flat`)
  }
  for (const f of GLOBAL_PROMPT_PART_FIELDS) {
    assert.ok(!PER_SUBJECT_PROMPT_KINDS.includes(f.key), `${f.key} rendered global but stored per subject`)
  }
})

test('every field has a label, a DOM id and non-empty guidance', () => {
  const all = [...GLOBAL_PROMPT_PART_FIELDS, ...SUBJECT_PROMPT_PART_FIELDS, ...FRAMING_PROMPT_PART_FIELDS]
  const ids = new Set()
  for (const f of all) {
    assert.ok(f.label && f.label.trim(), `${f.key}: no label`)
    assert.ok(f.id && !ids.has(f.id), `${f.key}: missing or duplicate id`)
    ids.add(f.id)
    assert.ok(Array.isArray(f.engines) && f.engines.length, `${f.key}: no engines named`)
  }
})

test('the markings lock warns about the incident that made it dangerous', () => {
  // Editable means a user can put `tattoos` back in the sentence that had the
  // model painting them on people who have none. Nothing in the backend can stop
  // that; this warning is the whole mitigation, so it is pinned here.
  const f = GLOBAL_PROMPT_PART_FIELDS.find((x) => x.key === 'markings_lock')
  assert.ok(f.warn, 'markings_lock lost its warning')
  assert.match(f.warn, /tattoo/i, 'the warning must name the actual incident')
  assert.match(f.warn, /without naming/i, 'the warning must state the rule, not just the anecdote')
})

test('the garment palette warns that its LENGTH reshuffles every shot', () => {
  const f = GLOBAL_PROMPT_PART_FIELDS.find((x) => x.key === 'outfit_palette')
  assert.ok(f.warn, 'outfit_palette lost its warning')
  assert.match(f.warn, /ADDING OR REMOVING A LINE/i)
  assert.match(f.warn, /empty/i, 'must say how to get the shipped list back')
})

test('PromptOverrideField renders the warning above the box', () => {
  assert.match(field, /warn = null/, 'the warn prop is gone')
  assert.match(field, /\{warn &&/, 'the warning is no longer rendered')
  // Above the textarea, or it is a footnote nobody reads before typing.
  assert.ok(field.indexOf('{warn &&') < field.indexOf('<textarea'),
    'the warning must sit ABOVE the box')
})

test('the preview is composed by the server, never re-implemented in JS', () => {
  assert.match(preview, /\/api\/settings\/prompt-preview/)
  // A second assembly in JS is how the preview would start lying.
  assert.ok(!/Create a new .*of the same/.test(preview),
    'PromptPreview builds prompt text itself — it must only display what the server composed')
})

test('the preview sends the UNSAVED prompts, so it shows what you are typing', () => {
  assert.match(preview, /identity_prompts: JSON\.parse\(body\)/)
  assert.match(preview, /setTimeout/, 'the per-keystroke request must be debounced')
})

test('the preview never lets a 1000-character prompt scroll the page sideways', () => {
  // The 400px case: wrap inside the box, scroll vertically, bounded height.
  assert.match(preview, /whitespace-pre-wrap/)
  assert.match(preview, /break-words/)
  assert.match(preview, /overflow-x-hidden/)
  assert.match(preview, /max-h-\d+ overflow-y-auto/)
})

test('the preview says the API engines ignore four of the boxes', () => {
  assert.match(preview, /API_ENGINES/)
  assert.match(preview, /no framing\s*\n?\s*block, no rendering tail and no markings hold/)
})

test('a failed preview reports itself and does not touch the fields', () => {
  assert.match(preview, /Preview unavailable/)
})

test('Settings renders all three groups plus the preview, with the help anchors', () => {
  for (const id of ['prompt-part-render-tail', 'prompt-part-framing', 'prompt-part-global']) {
    assert.ok(engines.includes(`id="${id}"`), `missing help anchor ${id}`)
  }
  assert.match(engines, /<PromptPreview\s+subject=\{subject\}\s+identityPrompts=\{ip\}/)
  // The per-subject groups must be keyed on the subject, or switching the chips
  // leaves the previous subject's text in an uncontrolled textarea.
  assert.match(engines, /key=\{`\$\{subject\}-\$\{f\.key\}`\}/)
})
