/* The three ways the generation-LoRA presets and the Krea base used to lie by
 * omission, pinned as contracts.
 *
 * 1. A configured preset could not apply on its own: the run panel opened on
 *    "None" on EVERY visit, so the request carried no preset name and the
 *    finished PNG's metadata showed no LoRA at all — indistinguishable, from the
 *    outside, from an app ignoring its settings.
 * 2. A preset row naming the LoRA the engine already loads is dropped by the
 *    server. That drop existed only in the server log.
 * 3. `krea.base_model` blank elects a base, and nothing named the winner.
 */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

import {
  resolveDefaultPresetName, generationLoraPresetPayload,
} from '../src/utils/generationLoras.js'
import {
  resolveKreaDefaultPresetName, kreaGenerationLoraPresetPayload,
} from '../src/utils/kreaGenerationLoras.js'
import {
  normalizeLoraRef, isFixedLoraDuplicate, fixedLoraDuplicateWarning,
} from '../src/utils/loraDuplicateGuard.js'
import { kreaBaseNote, KREA_BASE_NOTE_CLASS } from '../src/utils/kreaBaseNote.js'
import { helpTopics, searchHelpTopics } from '../src/help/helpRegistry.js'

const read = readSource

const KLEIN_PRESETS = [
  { name: 'Skin', loras: [{ file: 'klein/skin.safetensors', strength: 0.6 }] },
  { name: 'Empty', loras: [] },
]
const KREA_PRESETS = [
  { name: 'Skin', loras: [{ file: 'krea/other.safetensors', strength: 1 }] },
]

// ---- (1) the default preset ------------------------------------------------

test('a configured default preset is what the run panel starts on', () => {
  assert.equal(resolveDefaultPresetName('Skin', KLEIN_PRESETS), 'Skin')
  assert.equal(resolveKreaDefaultPresetName('Skin', KREA_PRESETS), 'Skin')
})

test('no default configured keeps the historical "None" start', () => {
  for (const v of ['', '   ', null, undefined, 42, {}]) {
    assert.equal(resolveDefaultPresetName(v, KLEIN_PRESETS), '')
    assert.equal(resolveKreaDefaultPresetName(v, KREA_PRESETS), '')
  }
})

test('a default naming no configured preset falls back to None, never blocks', () => {
  // Renamed in Settings, deleted, or hand-typed into config.json.
  assert.equal(resolveDefaultPresetName('Gone', KLEIN_PRESETS), '')
  assert.equal(resolveKreaDefaultPresetName('Gone', KREA_PRESETS), '')
  // …and no list at all is still an answer, not a crash.
  assert.equal(resolveDefaultPresetName('Skin', undefined), '')
  assert.equal(resolveKreaDefaultPresetName('Skin', undefined), '')
})

test('the two engines resolve their defaults against their OWN lists', () => {
  // Same NAME, two different chains — Klein's default must not select Krea's
  // preset and vice versa. Here 'Empty' exists only on the Klein side.
  assert.equal(resolveDefaultPresetName('Empty', KLEIN_PRESETS), 'Empty')
  assert.equal(resolveKreaDefaultPresetName('Empty', KREA_PRESETS), '')
})

test('the default flows into the SAME payload a manual pick produces', () => {
  const name = resolveDefaultPresetName('Skin', KLEIN_PRESETS)
  assert.deepEqual(
    generationLoraPresetPayload({ isKlein: true, presetName: name, presets: KLEIN_PRESETS }),
    { generation_lora_preset: 'Skin' })
  const kName = resolveKreaDefaultPresetName('Skin', KREA_PRESETS)
  assert.deepEqual(
    kreaGenerationLoraPresetPayload({ isKrea: true, presetName: kName, presets: KREA_PRESETS }),
    { krea_generation_lora_preset: 'Skin' })
})

test('a default naming an EMPTY preset still sends nothing (no dead name)', () => {
  const name = resolveDefaultPresetName('Empty', KLEIN_PRESETS)
  assert.equal(name, 'Empty')   // shown as picked, so the panel can say it is empty
  assert.deepEqual(
    generationLoraPresetPayload({ isKlein: true, presetName: name, presets: KLEIN_PRESETS }), {})
})

test('the workspace initialises both pickers from the config defaults', () => {
  const src = read('src/components/dataset/VariationCatalog.jsx')
  assert.match(src, /resolveDefaultPresetName\(\s*\n?\s*d\.config\?\.klein\?\.default_generation_lora_preset/)
  assert.match(src, /resolveKreaDefaultPresetName\(\s*\n?\s*d\.config\?\.krea\?\.default_generation_lora_preset/)
})

// ---- (2) the row the server will drop -------------------------------------

test('normalizeLoraRef mirrors os.path.normcase(os.path.normpath(...))', () => {
  assert.equal(normalizeLoraRef('klein\\A.safetensors'), 'klein/a.safetensors')
  assert.equal(normalizeLoraRef('klein//a.safetensors'), 'klein/a.safetensors')
  assert.equal(normalizeLoraRef('./klein/./a.safetensors'), 'klein/a.safetensors')
  assert.equal(normalizeLoraRef('klein/sub/../a.safetensors'), 'klein/a.safetensors')
  assert.equal(normalizeLoraRef('klein/a.safetensors/'), 'klein/a.safetensors')
  assert.equal(normalizeLoraRef('  klein/a.safetensors  '), 'klein/a.safetensors')
  assert.equal(normalizeLoraRef(''), '')
  assert.equal(normalizeLoraRef(null), '')
  assert.equal(normalizeLoraRef(7), '')
  // '..' pops, and a relative path keeps the leading ones python's normpath
  // keeps (verified against ntpath.normcase(ntpath.normpath(...)) on Windows
  // for all of these).
  assert.equal(normalizeLoraRef('../klein/a.safetensors'), '../klein/a.safetensors')
  assert.equal(normalizeLoraRef('klein/../../a.safetensors'), '../a.safetensors')
  assert.equal(normalizeLoraRef('C:/models/klein/a.safetensors'), 'c:/models/klein/a.safetensors')
  // An absolute path stays absolute — it must not compare equal to the bare tail.
  assert.notEqual(normalizeLoraRef('C:\\models\\loras\\klein\\a.safetensors'),
    normalizeLoraRef('klein/a.safetensors'))
})

test('the warning fires on every spelling the server also folds', () => {
  const fixed = 'klein/Flux2-Klein-9B-consistency-V2.safetensors'
  for (const row of [
    'klein/Flux2-Klein-9B-consistency-V2.safetensors',
    'klein\\Flux2-Klein-9B-consistency-V2.safetensors',   // the other separator
    'KLEIN/flux2-klein-9b-CONSISTENCY-v2.safetensors',    // case
    './klein/Flux2-Klein-9B-consistency-V2.safetensors',  // a '.' segment
    'klein//Flux2-Klein-9B-consistency-V2.safetensors',   // a doubled separator
    'klein/x/../Flux2-Klein-9B-consistency-V2.safetensors',
  ]) {
    assert.equal(isFixedLoraDuplicate(row, fixed), true, `should flag ${row}`)
  }
})

test('a different file is never flagged, and a blank side never matches', () => {
  const fixed = 'klein/Flux2-Klein-9B-consistency-V2.safetensors'
  assert.equal(isFixedLoraDuplicate('klein/other.safetensors', fixed), false)
  // A near-miss on the name is a DIFFERENT file, not a duplicate.
  assert.equal(isFixedLoraDuplicate('klein/Flux2-Klein-9B-consistency-V3.safetensors', fixed), false)
  assert.equal(isFixedLoraDuplicate('', fixed), false)
  assert.equal(isFixedLoraDuplicate('klein/a.safetensors', ''), false)
  assert.equal(isFixedLoraDuplicate('', ''), false)
})

test('each engine names its own fixed slot and the setting to change instead', () => {
  const klein = fixedLoraDuplicateWarning('klein')
  assert.match(klein, /consistency LoRA/)
  assert.match(klein, /Consistency strength/)
  const krea = fixedLoraDuplicateWarning('krea')
  assert.match(krea, /identity edit LoRA/)
  assert.match(krea, /Identity LoRA strength/)
  for (const w of [klein, krea]) {
    assert.match(w, /^Ignored:/)          // the verdict first, not buried
    assert.match(w, /trained for/)        // and WHY, not just "no"
  }
  assert.equal(fixedLoraDuplicateWarning('nanobanana'), '')
})

test('the preset editor feeds each card its OWN engine fixed slot', () => {
  const src = read('src/components/settings/EnginesSection.jsx')
  assert.match(src, /engineId="klein" fixedLora=\{config\.klein\?\.consistency_lora \|\| ''\}/)
  assert.match(src, /engineId="krea" fixedLora=\{config\.krea\?\.identity_lora \|\| ''\}/)
  // The warning is rendered on the row, not only computed.
  assert.match(src, /\{duplicate && \(/)
  assert.match(src, /fixedLoraDuplicateWarning\(engineId\)/)
})

// ---- (3) the elected Krea base is named ------------------------------------

test('blank field + a resolved base NAMES the file, instead of promising "auto"', () => {
  const note = kreaBaseNote('', 'Krea\\krea2_turbo_fp8_scaled.safetensors')
  assert.equal(note.tone, 'neutral')
  assert.match(note.text, /Krea\\krea2_turbo_fp8_scaled\.safetensors/)
  // The exact scenario behind this: two candidates, the community finetune won.
  assert.match(kreaBaseNote(null, 'Krea/finepornV31TURBOFP8_v3FIXFP8.safetensors').text,
    /finepornV31TURBOFP8_v3FIXFP8\.safetensors/)
})

test('blank field + nothing on disk says so rather than naming a file', () => {
  const note = kreaBaseNote('', '')
  assert.equal(note.tone, 'warn')
  assert.match(note.text, /No compatible Krea 2 base/)
})

test('a pin that resolved to itself is confirmed, whatever the spelling', () => {
  const note = kreaBaseNote('krea2_turbo_fp8_scaled.safetensors',
    'Krea\\krea2_turbo_fp8_scaled.safetensors')
  assert.equal(note.tone, 'ok')   // the server matches a pin on its BASENAME
  assert.match(note.text, /Currently loading/)
})

test('a pin that did NOT resolve says the engine is held, and never promises a substitute', () => {
  // The sibling chantier of this same wave made an unresolvable pin GATE the engine
  // (capabilities.krea_pin_gaps → krea_ready false). resolve_krea_unet still elects a
  // fallback, so `resolved` arrives populated — but nothing consumes it. Naming that
  // file as the one "runs load" would be the very silence this note exists to end,
  // pointed the other way, so the fallback name must NOT appear.
  const note = kreaBaseNote('krea2_turbo_typo.safetensors',
    'Krea\\krea2_turbo_fp8_scaled.safetensors')
  assert.equal(note.tone, 'warn')
  assert.match(note.text, /krea2_turbo_typo\.safetensors/)
  assert.doesNotMatch(note.text, /krea2_turbo_fp8_scaled\.safetensors/)
  assert.match(note.text, /will not run/)
  // …and a pin with nothing at all to fall back on says that too.
  assert.equal(kreaBaseNote('krea2_turbo_typo.safetensors', '').tone, 'warn')
})

test('every tone the note can produce has a colour', () => {
  for (const [pin, got] of [['', 'a'], ['', ''], ['a', 'a'], ['a', 'b'], ['a', '']]) {
    const { tone } = kreaBaseNote(pin, got)
    assert.ok(KREA_BASE_NOTE_CLASS[tone], `no class for tone ${tone}`)
  }
})

test('the Settings card reads the SERVER-resolved base, and ranks nothing itself', () => {
  const src = read('src/components/settings/EnginesSection.jsx')
  assert.match(src, /caps\?\.comfyui\?\.krea_base_resolved/)
  // Model resolution is a server decision: no second ranking in the browser.
  assert.doesNotMatch(src, /elect_krea_base|turboTier|rankKreaBase/)
})

test('the capabilities probe publishes the base the generation path would load', () => {
  const py = read('../backend/app/capabilities.py')
  assert.match(py, /krea_base_resolved = _krh\.resolve_krea_unet\(\) or ''/)
  assert.match(py, /'krea_base_resolved': krea_base_resolved,/)
})

// ---- the new settings reach all four surfaces ------------------------------

test('both default-preset keys have a control, a topic, a doc row and a shipped default', () => {
  const card = read('src/components/settings/EnginesSection.jsx')
  const guide = read('../docs/guide/settings-reference.md')
  const defaults = read('../backend/app/config.py')
  const topics = new Set(helpTopics.map((t) => t.id))
  for (const [section, domId] of [['klein', 'klein-default-lora-preset'],
                                  ['krea', 'krea-default-lora-preset']]) {
    const key = `${section}.default_generation_lora_preset`
    assert.ok(card.includes(`id="${domId}"`), `${key}: no control id="${domId}"`)
    assert.ok(card.includes(`setField('${section}', 'default_generation_lora_preset'`),
      `${key}: the control writes nothing`)
    assert.ok(topics.has(key), `${key}: no help topic (Help search cannot find it)`)
    assert.ok(guide.includes(`\`${key}\``), `${key}: absent from settings-reference.md`)
    assert.ok(defaults.includes("'default_generation_lora_preset': ''"),
      `${key}: not shipped as an empty default`)
  }
})

test('the help topics point at the controls that exist', () => {
  const byId = new Map(helpTopics.map((t) => [t.id, t]))
  assert.equal(byId.get('klein.default_generation_lora_preset').app.focus, 'klein-default-lora-preset')
  assert.equal(byId.get('krea.default_generation_lora_preset').app.focus, 'krea-default-lora-preset')
})

test('the dropped-row symptom is searchable in Help, in the words a user types', () => {
  // Someone whose preset produced nothing searches the symptom, not the cause.
  for (const q of ['ignored', 'double-stack', 'posterized']) {
    assert.ok(searchHelpTopics(q).some((t) => t.id === 'klein.generation_lora_presets'),
      `'${q}' should surface the Klein preset topic`)
  }
  for (const q of ['which model', 'wrong model']) {
    assert.ok(searchHelpTopics(q).some((t) => t.id === 'krea.base_model'),
      `'${q}' should surface the Krea base model topic`)
  }
})

test('nothing still promises the picker resets to None on every visit', () => {
  // That sentence was true and is now false in three places at once; a doc that
  // still says it is worse than no doc.
  const sources = [
    read('src/components/settings/EnginesSection.jsx'),
    read('src/components/dataset/VariationCatalog.jsx'),
    read('src/utils/generationLoras.js'),
    read('src/utils/kreaGenerationLoras.js'),
    read('../docs/guide/settings-reference.md'),
  ]
  for (const src of sources) {
    assert.doesNotMatch(src, /"None" by default\)/)
    assert.doesNotMatch(src, /defaults to \*None\* every visit/)
    assert.doesNotMatch(src, /None, the default on every visit/)
  }
})
