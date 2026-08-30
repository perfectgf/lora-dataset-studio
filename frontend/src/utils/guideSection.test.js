import test from 'node:test'
import assert from 'node:assert/strict'
import { sliceGuideSection } from './guideSection.js'

const MD = [
  '# Title', 'intro',
  '## First Section', 'first body line', 'second line',
  '```', '## not a heading, it is code', '```',
  '## Second Section', 'other body',
].join('\n')

test('slices one H2 section, heading included, up to the next H2', () => {
  const s = sliceGuideSection(MD, 'first-section')
  assert.ok(s.startsWith('## First Section'))
  assert.ok(s.includes('second line'))
  assert.ok(!s.includes('Second Section'))
})

test('a ## inside a code fence is code, not a section boundary', () => {
  const s = sliceGuideSection(MD, 'first-section')
  assert.ok(s.includes('not a heading'))
})

test('the last section runs to the end, and an unknown anchor is empty', () => {
  assert.ok(sliceGuideSection(MD, 'second-section').includes('other body'))
  assert.equal(sliceGuideSection(MD, 'nope'), '')
  assert.equal(sliceGuideSection('', 'x'), '')
})
