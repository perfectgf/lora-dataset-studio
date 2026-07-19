import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const app = readFileSync(new URL('../src/App.jsx', import.meta.url), 'utf8')
const guide = readFileSync(new URL('../src/pages/GuidePage.jsx', import.meta.url), 'utf8')
const help = readFileSync(new URL('../../docs/guide/getting-help.md', import.meta.url), 'utf8')
const en = JSON.parse(readFileSync(new URL('../src/i18n/locales/en.json', import.meta.url), 'utf8'))

test('Help is a top-level destination beside Settings', () => {
  assert.match(app, /to="\/settings"[\s\S]*t\('nav\.settings'\)[\s\S]*to="\/help"[\s\S]*t\('nav\.help'\)/)
  assert.equal(en.nav.settings, 'Settings')
  assert.equal(en.nav.help, 'Help')
  assert.match(app, /path="\/help" element={<GuidePage helpOnly \/>}/)
})

test('Getting help is no longer a Guide chapter and keeps its existing content', () => {
  assert.match(
    guide,
    /const HELP_CHAPTER = \{[\s\S]*source: gettingHelp,[\s\S]*extra: 'diagnostic'/,
  )
  const chaptersBlock = guide.match(/const CHAPTERS = \[([\s\S]*?)\n\]/)?.[1] || ''
  assert.doesNotMatch(chaptersBlock, /getting-help/)
  assert.match(app, /path="\/guide\/getting-help" element={<Navigate to="\/help" replace \/>}/)
})

test('Help contact links render as links instead of literal Markdown', () => {
  assert.match(help, /\*\*Discord\*\* — \[discord\.gg\/j6hnJBFtXE\]\(https:\/\/discord\.gg\/j6hnJBFtXE\)/)
  assert.match(help, /\*\*GitHub\*\* — \[Issues\]\(https:\/\/github\.com\/perfectgf\/lora-dataset-studio\/issues\)/)
  assert.doesNotMatch(help, /\*\*[^\n]*\[[^\]]+\]\([^)]+\)[^\n]*\*\*/)
})

test('Guide topic iteration does not shadow the translation function', () => {
  assert.match(guide, /for \(const topic of helpTopicsForChapter\(chapter\.id\)\)/)
  assert.doesNotMatch(guide, /for \(const t of helpTopicsForChapter\(chapter\.id\)\)/)
  assert.match(guide, /\{t\('guide\.openScreen'\)\}/)
})
