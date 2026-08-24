/**
 * Contract of 🧬 Blend on the ◉ LoRA Canvas — « Generate from the board » loading
 * every ticked checkpoint in ONE image instead of one pass each.
 *
 * The pure logic is unit-tested next to it (src/utils/canvasGeneration.test.js).
 * What is asserted HERE is what only the source can say, because `node --test`
 * cannot parse JSX: that the board reuses the Test Studio's stack module rather
 * than growing a second one, that the launch really switches the request, and
 * that the three promises made to the user in the brief are on screen.
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
const BLEND = read('components/canvas/CanvasBlendPanel.jsx')
const PANEL = read('components/canvas/CanvasGenerationPanel.jsx')
const HOOK = read('hooks/useCanvasStudio.js')
const UTIL = read('utils/canvasGeneration.js')
const SETUP = read('components/dataset/studio/RunSetupPanel.jsx')
const ROW = read('components/dataset/studio/BlendWeightRow.jsx')

test('the board imports the Test Studio stack module instead of copying it', () => {
  // A second clamp, a second "one family" rule or a second key shape is a second
  // chance for the two screens to disagree about what a blend is.
  assert.match(UTIL, /from '\.\.\/components\/dataset\/studio\/loraStack\.js'/)
  assert.match(UTIL, /combineBlocker/)
  assert.match(UTIL, /stackWeight/)
  // Le curseur ET les cases d'un LoRA sont UN composant partagé par les deux
  // surfaces (BlendWeightRow) : deux copies du même contrôle, ce serait deux
  // occasions de diverger sur « aucune case cochée = le curseur gouverne ».
  assert.match(BLEND, /import BlendWeightRow from '\.\.\/dataset\/studio\/BlendWeightRow'/)
  assert.match(ROW, /from '\.\/loraStack'/)
  assert.match(ROW, /BLEND_WEIGHT_CHIPS/)
  assert.doesNotMatch(ROW, /const BLEND_WEIGHT_CHIPS\s*=/)
  // …and does not redeclare the bounds it just imported.
  assert.doesNotMatch(BLEND, /const COMBINE_(MIN|MAX)_WEIGHT\s*=/)
})

test('Compare is the default; Blend only appears once there are two picks', () => {
  assert.match(PANEL, /getItem\(MODE_KEY\) === 'blend' \? 'blend' : 'compare'/)
  assert.match(PANEL, /selection\.length > 1 && \(\s*<CanvasBlendPanel/)
})

test('a blocked blend blocks the LAUNCH, it does not just colour a panel', () => {
  // Setting weights for a run the engine will refuse is the failure mode this
  // replaces: the verdict the launch bar reads is overridden, not decorated.
  assert.match(PANEL, /const blendBlocker = \(mode === 'blend' && blendAvailable\)/)
  assert.match(PANEL, /launchBlocked=\{launchVerdict\.blocked\}/)
  assert.match(PANEL, /blocked: true, reason: blendBlocker/)
  // …and a blend rule may only speak while its panel is on screen: un-ticking
  // down to one pick must not leave a dead button explained by a hidden line.
  assert.match(PANEL, /const blendAvailable = selection\.length > 1/)
  assert.match(PANEL, /const blend = mode === 'blend' && blendAvailable && !blendBlocker/)
  // Mixed families keep the launch's own wording — the better one — and kill the
  // toggle rather than letting it be pressed into a refused run.
  assert.match(BLEND, /disabled=\{dead\}/)
  assert.match(BLEND, /familyReason/)
})

test('the launch really switches the request, and the deploy still runs first', () => {
  assert.match(HOOK, /blend && canvasRunSelections\(picks\)\.length > 1/)
  assert.match(HOOK, /\.\.\.\(blending \? \{ combine: true \} : \{ strengths \}\)/)
  assert.match(HOOK, /canvasRunSelections\(picks, \{[\s\S]{0,80}blend: blending, weights/)
  // …et les cases de poids voyagent avec les curseurs, sinon le board annonce
  // neuf images et en lance une.
  assert.match(HOOK, /sets: sets \|\| \{\}/)
  // The deploy gate is upstream of the payload: a blend can never load a subset
  // of the checkpoints it announced.
  const deployAt = HOOK.indexOf('canvasUndeployed(picks).length')
  const payloadAt = HOOK.indexOf("postJson('/api/train/canvas/generate'")
  assert.ok(deployAt > 0 && payloadAt > deployAt)
})

test('a blend announces ONE configuration, not one per pick and strength', () => {
  // The strength sweep has nothing left to sweep, so it leaves the panel AND the
  // counter — a panel promising six images while queueing one is the bug here.
  assert.match(PANEL, /showStrengths=\{!blend\}/)
  // …et depuis le balayage, `configCount` combinaisons plutôt qu'une.
  assert.match(PANEL, /cellTotal=\{blend \? form\.axisTotal \* configCount : null\}/)
  assert.match(PANEL, /const configCount = blend \? canvasBlendConfigCount/)
  // `cells` = ce que la grille rend pour UN prompt ; le lot 📝 le multiplie
  // ensuite (cf. prompt-batch-contract), il ne le remplace pas.
  assert.match(SETUP, /const cells = cellTotal != null \? cellTotal : form\.total/)
  assert.match(SETUP, /const total = cells \* promptMult/)
  assert.match(SETUP, /total=\{total \* batchMult\}/)
})

test('every trigger word is shown before it is injected, and so are the missing ones', () => {
  assert.match(BLEND, /canvasStackTriggers/)
  assert.match(BLEND, /Added to the front of your prompt/)
  assert.match(BLEND, /canvasStackWithoutTrigger/)
  assert.match(BLEND, /no trigger/)
})

test('the honest line about what blending two identities does is on screen', () => {
  assert.match(BLEND, /hybrid person/)
  assert.match(BLEND, /identity \+ style, or identity \+ concept/)
})

test('the toggle has a help topic and the wave has a What\'s-new entry', () => {
  const topic = getHelpTopic('canvas-blend')
  assert.ok(topic, 'canvas-blend must be a registered help topic')
  assert.equal(topic.app.route, '/canvas')
  assert.ok(topic.keywords.includes('blend'))
  assert.match(BLEND, /topic="canvas-blend"/)

  const entry = ALL_WHATS_NEW.find((e) => e.id === '2026-08-03-canvas-blend')
  assert.ok(entry, 'the blend wave needs a What\'s-new entry')
  // Archived → no in-app target, by doctrine (whatsNew.js, "Keep the list tidy").
  assert.equal(entry.to, undefined)
})
