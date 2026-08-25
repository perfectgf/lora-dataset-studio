/**
 * What the user SEES while a bank pass works, and before one starts. Rendered.
 *
 * Two symptoms, both reported from a real 50 397-image bank, both invisible to
 * a source-text test:
 *
 *  · ✨ Score sat on "Scoring pass running — 373 / 373 · resuming — 20847 of
 *    21220 already cached" with a full bar for ~4 minutes while the parent wrote
 *    21 000 rows and the child grouped 25 000 embeddings. The pass was working;
 *    the screen said it had finished. What is pinned here is that a step with no
 *    counter takes the sentence AND drops the stale figure — "373 / 373 ·
 *    grouping styles" would read exactly as badly as before.
 *  · 🎨 Classify medium offered "Classify 2 images" on a scope whose two images
 *    had never been scored, then reported "0 classified". The window now counts
 *    the images it cannot answer for, and refuses the run.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { ProgressBar } = await import('../src/components/bank/BankProgress.jsx')
const { default: PassDialog } = await import('../src/components/bank/PassDialog.jsx')

const text = (markup) => markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()

const bar = (activity) => text(renderToStaticMarkup(
  createElement(ProgressBar, { activity, onCancel: () => {} })))

test('a counted step still shows its counter', () => {
  const seen = bar({ kind: 'score', done: 373, total: 373, finished: false,
    detail: 'resuming — 20847 of 21220 already cached' })
  assert.match(seen, /Scoring pass running — 373 \/ 373/)
})

test('a step with no counter shows the sentence and NO stale figure', () => {
  const seen = bar({ kind: 'score', done: 0, total: 0, finished: false,
    detail: 'grouping styles over 25058 image(s) — the slow tail of this pass; '
      + 'Stop now keeps every score already computed but discards the grouping' })
  assert.match(seen, /grouping styles over 25058/)
  // The consequence of a Stop AT THIS STEP, next to the Stop button itself.
  assert.match(seen, /Stop now keeps every score/)
  // No "373 / 373" left over, and no bare "0" standing in for it either.
  assert.doesNotMatch(seen, /Scoring pass running — 0/)
  assert.doesNotMatch(seen, /\d+ \/ \d+/)
  // Stop is still offered — it has to be, so the sentence above has to hold.
  assert.match(seen, /Stop/)
})

// --- the launch window --------------------------------------------------------

const PAYLOAD = {
  counts: { keep: 24931, pending: 2, reject: 25464 },
  pass_scopes: {
    medium: {
      todo: { keep: 0, pending: 2, reject: 25464 },
      all: { keep: 24931, pending: 2, reject: 25464 },
      blocked: { keep: 0, pending: 2, reject: 0 },
      blocked_all: { keep: 0, pending: 2, reject: 0 },
    },
  },
}

const dialog = (scope) => text(renderToStaticMarkup(createElement(PassDialog, {
  passId: 'medium', payload: PAYLOAD, scope, onClose: () => {}, onLaunch: () => {},
})))

test('the default scope line says the run would classify nothing', () => {
  const seen = dialog('')
  assert.match(seen, /Kept \+ undecided — 2 images in scope, 0 ready/)
  assert.match(seen, /✨ Score has not reached 2 of them/)
  // …and the refusal is printed above the button, with the pass to run instead.
  assert.match(seen, /2 image\(s\) in scope, 0 ready/)
  assert.match(seen, /Run ✨ Score first/)
})

test('the bin line, whose images ARE scored, gains nothing', () => {
  const seen = dialog('reject')
  // Scoped to THAT line: the other scopes in the same window do carry the note,
  // so a whole-markup assertion would prove nothing about this one.
  assert.match(seen, /Unkept only \(the bin\) — 25464 images Images you already rejected/)
  assert.doesNotMatch(seen, /25464 images in scope, /)
})

test('the window states the dependency on ✨ Score where the scope is chosen', () => {
  assert.match(dialog(''), /Score never runs on the bin/)
})
