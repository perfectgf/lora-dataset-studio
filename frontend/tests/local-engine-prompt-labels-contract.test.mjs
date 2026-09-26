/* The identity lock named `klein_identity` is the LOCAL-ENGINE lock, not Klein's.
   Klein and Krea 2 Edit share one prompt assembly on the server
   (`_compose_edit_prompt`) and both read this one text — deliberately, so there
   is a single lock to keep in sync instead of two.

   WHY THIS FILE EXISTS
   --------------------
   Nothing tied the WORDING of a prompt box to the engines that consume it. The
   box shipped as "Klein — restage & face-identity block", described as what
   "Klein (local)" uses, for as long as Krea 2 Edit had been reading it — until a
   user asked on Discord whether the block applied to Krea 2 at all. The badge was
   right the whole time; only the words were wrong, and words are the entire
   feature here: a prompt a user believes is not theirs is a prompt they never
   edit.

   The drift was not a mistake at the keyboard, it was a second consumer arriving
   later. So the guard has to be mechanical: these assertions read the core engine catalog
   and fail when another core consumer is wired up without the label following.
   Plugin engines compose their own prompts and do not inherit this lock.
   The server half of the same contract — every core local engine really does send
   this text, and no API engine does — lives in
   backend/tests/test_identity_prompts_override.py. */
import test, { beforeEach } from 'node:test'
import assert from 'node:assert/strict'

import {
  IDENTITY_PROMPT_FIELDS, identityPromptFields, PROMPT_SUBJECT_TYPES,
  API_PROMPT_ENGINES,
} from '../src/components/common/promptOverride.js'
import { localEngineIds, apiEngineIds, engineLabel } from '../src/components/dataset/engineSelection.js'
import { CORE_ENGINE_CATALOG } from '../src/engines/catalog.js'
import { setEnabled } from '../src/plugins/registry.js'
import { mountPublicPlugins } from './support/publicPluginFixtures.mjs'

beforeEach(() => mountPublicPlugins())

const sharedPromptEngines = CORE_ENGINE_CATALOG.filter(engine => engine.kind === 'local').map(engine => engine.id)
const kleinIdentity = (fields) => fields.find((f) => f.key === 'klein_identity')

test('the two engine lists agree on which engines are APIs', () => {
  // promptOverride keeps its own copy (it must stay pure and importable), so the
  // two must be pinned together or the badge maps an engine to the wrong box.
  assert.deepEqual([...API_PROMPT_ENGINES].sort(), apiEngineIds().sort())
})

test('disabling the API owner removes its choices and preserves both local engines', () => {
  assert.deepEqual(apiEngineIds().sort(), [...API_PROMPT_ENGINES].sort())
  setEnabled([])
  assert.deepEqual(apiEngineIds(), [])
  assert.deepEqual(localEngineIds(), ['klein', 'krea'])
})

test('klein_identity declares every core local consumer and no unrelated engine', () => {
  for (const subject of PROMPT_SUBJECT_TYPES) {
    const field = kleinIdentity(identityPromptFields(subject))
    assert.ok(field, `no klein_identity field for ${subject}`)
    for (const engine of sharedPromptEngines) {
      assert.ok(field.engines.includes(engine),
        `${subject}: klein_identity does not declare the local engine ${engine}`)
    }
    for (const engine of [...apiEngineIds(), ...localEngineIds().filter(id => !sharedPromptEngines.includes(id))]) {
      assert.ok(!field.engines.includes(engine),
        `${subject}: klein_identity must not claim the unrelated engine ${engine}`)
    }
  }
})

test('its description NAMES every core local consumer, so no engine\'s users skip the box', () => {
  // The failure this prevents, in one sentence: a Krea 2 user reading "Klein"
  // concludes the box is not theirs. Another core consumer would reintroduce it
  // the day it ships, which is why the check is over the core catalog, not a list
  // of two names written here.
  for (const subject of PROMPT_SUBJECT_TYPES) {
    const { desc } = kleinIdentity(identityPromptFields(subject))
    for (const engine of sharedPromptEngines) {
      assert.ok(desc.includes(engineLabel(engine)),
        `${subject}: the description never names ${engineLabel(engine)}`)
    }
  }
})

test('the label belongs to the FAMILY — it never names one local engine alone', () => {
  // "Local engines — …" is the shape that survives a new engine. A label naming
  // one of several is the exact defect this file exists for, so it fails whether
  // the name left behind is Klein's or anyone else's.
  for (const subject of PROMPT_SUBJECT_TYPES) {
    const { label } = kleinIdentity(identityPromptFields(subject))
    const named = sharedPromptEngines.filter((e) => label.includes(engineLabel(e)))
    const missing = sharedPromptEngines.filter((e) => !named.includes(e))
    assert.ok(named.length === 0 || missing.length === 0,
      `${subject}: the label "${label}" names ${named.map(engineLabel).join(', ')}`
      + ` and leaves out ${missing.map(engineLabel).join(', ')}`)
  }
})

test('the human table is the one shipped to human datasets, unchanged in shape', () => {
  // identityPromptFields('human') deliberately returns the ORIGINAL array (the
  // wording users already know); the per-subject variants derive from it. If
  // that shortcut ever diverges, every assertion above would be testing a
  // different object than the one Settings renders.
  assert.equal(identityPromptFields('human'), IDENTITY_PROMPT_FIELDS)
})
