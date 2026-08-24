/**
 * Contract of 📝 the prompt batch — tick several saved prompts, launch once.
 *
 * The rule itself is unit-tested next to it
 * (src/components/dataset/studio/promptBatch.test.js) and the "N prompts → N
 * distinct workflows" proof lives in backend/tests/test_studio_prompt_batch.py.
 * What is asserted HERE is what only the sources can say, because `node --test`
 * cannot parse JSX:
 *
 *   · the feature exists on BOTH generation surfaces by construction — one
 *     history component, mounted through one RunSetupPanel, so the Test Studio
 *     and the board cannot ship one without the other;
 *   · the batch reaches BOTH launch routes: the key travels in the object each
 *     hook spreads into its POST body, and both hooks really spread it;
 *   · ticking nothing changes nothing (no key, no multiplier, same label);
 *   · what the batch costs is announced BEFORE the click, in the counter and on
 *     the button — the failure mode here is a panel that says "1 image" and a
 *     queue that receives nine.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'

import { getHelpTopic } from '../src/help/helpRegistry.js'
import { WHATS_NEW } from '../src/whatsNew.js'
import { WHATS_NEW_ARCHIVE } from '../src/whatsNewArchive.js'

// The entry under test may have moved to the archive since it shipped
// (see whatsNew.js, rule "Keep the list tidy") — search the union.
const ALL_WHATS_NEW = [...WHATS_NEW, ...WHATS_NEW_ARCHIVE]

const read = (rel) => readSource(`src/${rel}`)
const RECENT = read('components/dataset/studio/RecentPrompts.jsx')
const FIELD = read('components/dataset/studio/PromptField.jsx')
const SETUP = read('components/dataset/studio/RunSetupPanel.jsx')
const SEED = read('components/dataset/studio/SeedControls.jsx')
const STUDIO_HOOK = read('hooks/useLoraTestStudio.js')
const CANVAS_HOOK = read('hooks/useCanvasStudio.js')
const CANVAS_PANEL = read('components/canvas/CanvasGenerationPanel.jsx')
const COMPARISON_SETUP = read('components/dataset/studio/StudioRunSetup.jsx')
const STACK = read('components/dataset/studio/loraStack.js')

test('one history component, mounted by both surfaces — parity by construction', () => {
  // The board's panel mounts the Test Studio's own RunSetupPanel, which mounts
  // PromptField, which mounts RecentPrompts. A batch added to that chain exists
  // on both screens or on neither; there is no second list to keep in step.
  assert.match(CANVAS_PANEL, /import RunSetupPanel from '\.\.\/dataset\/studio\/RunSetupPanel'/)
  assert.match(SETUP, /import PromptField from '\.\/PromptField'/)
  assert.match(FIELD, /import RecentPrompts from '\.\/RecentPrompts'/)
  // …and the batch props really run down that chain rather than stopping midway.
  assert.match(SETUP, /batchPrompts=\{pickedPrompts\}/)
  assert.match(SETUP, /onToggleBatchPrompt=\{toggleBatchPrompt\}/)
  assert.match(FIELD, /batch=\{batchPrompts\}/)
  assert.match(FIELD, /onToggleBatch=\{onToggleBatchPrompt\}/)
})

test('each card gets a real checkbox, and the count and a way out are on screen', () => {
  assert.match(RECENT, /role="checkbox"/)
  assert.match(RECENT, /aria-checked=\{inBatch\}/)
  assert.match(RECENT, /\{picked\.length\} selected/)
  assert.match(RECENT, /onClick=\{onClearBatch\}/)
  // Ticking must not write into the prompt field: that is what CLICKING the card
  // does, and conflating the two would clobber a typed prompt on every tick.
  assert.match(RECENT, /onClick=\{\(\) => onToggleBatch\(pr\.prompt\)\}/)
  assert.match(RECENT, /onClick=\{\(\) => onPick\(pr\.prompt\)\}/)
})

test('a caller that does not want the batch gets the component it had before', () => {
  // The Recent-prompts list is mounted elsewhere too; the checkbox column only
  // appears for a host that passes the handler.
  assert.match(RECENT, /const batchable = typeof onToggleBatch === 'function'/)
  assert.match(RECENT, /\{batchable && \(/)
})

test('the batch reaches BOTH launch routes, through the channel both hooks spread', () => {
  // The key rides in the same object as the global generation settings. That is
  // only safe because BOTH hooks spread that object into their body — checking
  // one half would have proved the feature on one screen only.
  // `allPickedPrompts` = the history batch plus the 🎬 scene batch, one channel:
  // the scenes ride the SAME `prompts` key, so both screens get both features.
  assert.match(SETUP, /const settings = launchSettings\(genSettings, allPickedPrompts\)/)
  assert.match(SETUP, /form\.genCount, settings,/)
  assert.match(STUDIO_HOOK, /count, family, \.\.\.genSettings \}/)
  assert.match(CANVAS_HOOK, /count, \.\.\.genSettings,/)
})

test('ticking nothing is not a new code path', () => {
  // launchSettings returns the very object it was given when the batch is empty
  // (asserted for real in promptBatch.test.js); the panel must not add anything
  // of its own around it.
  assert.doesNotMatch(SETUP, /prompts:\s*\[/)
  assert.match(SETUP, /const promptMult = Math\.max\(1, allPickedPrompts\.length\)/)
})

test('what the batch costs is announced before the click, not by the queue', () => {
  assert.match(SETUP, /const total = cells \* promptMult/)
  assert.match(SETUP, /promptMult=\{promptMult\}/)
  assert.match(SEED, /promptMult > 1 &&/)
  assert.match(SEED, /📝 ×\{promptMult\}/)
  // The button says how many prompts it is about to run, on both the inline bar
  // and the Test Studio's sticky action bar.
  assert.match(SETUP, /label=\{launchText\}/)
  assert.match(SETUP, /runLabel=\{launchText \? `🚀 \$\{launchText\}` : undefined\}/)
})

test('nothing about the batch size is refused — it is priced instead', () => {
  // The first version capped the prompt axis at 24 and the first real use ticked
  // 33. The cost is time, so time is what the panel governs: a warning line and
  // ONE confirmation above a duration threshold, never a refusal, and never on a
  // single axis of the six the run multiplies.
  assert.doesNotMatch(SETUP, /at most|too many|untick some/i)
  assert.match(SETUP, /const cost = runCost\(total \* batchMult \* form\.genCount, d\.seconds_per_image\)/)
  assert.match(SETUP, /cost\.heavy && !window\.confirm\(heavyRunConfirm\(cost\)\)/)
  assert.match(SETUP, /data-testid="heavy-run-notice"/)
  // The comparison branch gets the identical guard, not a second policy.
  assert.match(COMPARISON_SETUP, /cost\.heavy && !window\.confirm\(heavyRunConfirm\(cost\)\)/)
  assert.match(COMPARISON_SETUP, /data-testid="heavy-run-notice"/)
})

test('one screen never announces two different durations', () => {
  // The estimate used to be a hardcoded 12 s/image in two places. Every counter
  // now goes through runCost, so the measured pace reaches all of them.
  assert.match(SEED, /const cost = runCost\(total \* genCount, secondsPerImage\)/)
  assert.doesNotMatch(SEED, /\* 12 \/ 60/)
  assert.match(STACK, /const cost = runCost\(cells, secondsPerImage\)/)
  assert.doesNotMatch(STACK, /cells \* 12/)
})

test('the batch is deliberately not persisted', () => {
  // The board persists its 🧬 mode and weights on purpose. A batch is the intent
  // of ONE launch: three boxes still ticked after a reload would triple a run
  // its owner thought was simple.
  assert.match(SETUP, /useState\(\[\]\)/)
  assert.doesNotMatch(SETUP, /localStorage/)
})

test('the batch has a help topic and a What\'s-new entry', () => {
  const topic = getHelpTopic('studio-prompt-batch')
  assert.ok(topic, 'studio-prompt-batch must be a registered help topic')
  assert.ok(topic.keywords.includes('batch'))
  // The words someone types after being turned away by the cap that used to exist.
  assert.ok(topic.keywords.includes('at most 24 prompts'))

  const entry = ALL_WHATS_NEW.find((e) => e.id === '2026-08-03-studio-prompt-batch')
  assert.ok(entry, 'the prompt batch needs a What\'s-new entry')
  const lifted = ALL_WHATS_NEW.find((e) => e.id === '2026-08-03-prompt-batch-no-cap')
  assert.ok(lifted, 'lifting the cap needs its own What\'s-new entry')
})
