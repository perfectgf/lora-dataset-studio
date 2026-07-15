import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const app = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8')
const guide = readFileSync(new URL('../src/pages/GuidePage.jsx', import.meta.url), 'utf8')

test('Help is a top-level destination beside Settings', () => {
  assert.match(app, /to="\/settings"[\s\S]*>Settings<\/NavLink>[\s\S]*to="\/help"[\s\S]*>Help<\/NavLink>/)
  assert.match(app, /path="\/help" element={<GuidePage helpOnly \/>}/)
})

test('Getting help is no longer a Guide chapter and keeps its existing content', () => {
  assert.match(guide, /const HELP_CHAPTER = .*source: gettingHelp, extra: 'diagnostic'/)
  const chaptersBlock = guide.match(/const CHAPTERS = \[([\s\S]*?)\n\]/)?.[1] || ''
  assert.doesNotMatch(chaptersBlock, /getting-help/)
  assert.match(app, /path="\/guide\/getting-help" element={<Navigate to="\/help" replace \/>}/)
})
