import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { openCollapsedAncestors } from '../src/help/revealTarget.js'

const source = readFileSync(new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')

// The "More ways out" disclosure hides Import to bank / Backup / Hugging Face,
// which the Import & export sidebar links jump to. Without this the link is
// selected and the button stays invisible inside the collapsed <details>.
test('the workspace landing opens collapsed disclosures around its target', () => {
  assert.match(source, /import \{ openCollapsedAncestors \} from '\.\.\/\.\.\/help\/revealTarget'/)
  const land = source.indexOf('const land = () => {')
  assert.notEqual(land, -1, 'landing effect not found')
  const body = source.slice(land, source.indexOf('target.scrollIntoView', land))
  assert.match(body, /openCollapsedAncestors\(target\.parentElement\)/)
})

// Starting the walk at the parent matters: ds-training-advanced IS a <details>
// whose open state React drives from the navigation panel. Forcing it open from
// the DOM would desync that state.
test('a controlled <details> target is not forced open by the landing', () => {
  const details = { tagName: 'DETAILS', open: false, parentElement: null }
  openCollapsedAncestors(details.parentElement)
  assert.equal(details.open, false)
})

// …while a plain target nested in a collapsed disclosure does get revealed.
test('a target nested in a collapsed disclosure is revealed', () => {
  const details = { tagName: 'DETAILS', open: false, parentElement: null }
  const wrap = { tagName: 'DIV', parentElement: details }
  const target = { tagName: 'DIV', parentElement: wrap }
  openCollapsedAncestors(target.parentElement)
  assert.equal(details.open, true)
})

// The disclosure must stay uncontrolled, otherwise opening it from the DOM is
// undone by React's next render.
test('the "More ways out" disclosure stays uncontrolled', () => {
  const summary = source.indexOf('More ways out')
  assert.notEqual(summary, -1, '"More ways out" disclosure not found')
  const block = source.slice(source.lastIndexOf('<details', summary), summary)
  assert.doesNotMatch(block, /\bopen=\{/)
})
