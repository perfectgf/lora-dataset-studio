import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const readStudio = (name) => readFileSync(
  new URL(`../src/components/dataset/studio/${name}`, import.meta.url), 'utf8')

test('both Test Studio banners split generating work from queued work', () => {
  for (const name of ['RunSetupPanel.jsx', 'ComparisonStudio.jsx']) {
    const source = readStudio(name)
    assert.match(source, /studio\.run\.progress/,
      `${name} must render the localized progress message`)
    assert.match(source, /generating:\s*[^,\n]+/,
      `${name} must pass a distinct generating count`)
    assert.match(source, /queued:\s*[^,\n]+/,
      `${name} must show generating and queued as distinct counts`)
    assert.doesNotMatch(source, /test generation\(s\) queued/,
      `${name} must not label every unfinished cell as queued`)
  }
})

test('pending result tiles expose their real queue state', () => {
  const source = readStudio('ResultTile.jsx')
  assert.match(source, /cell\.queue_status === 'generating'/)
  assert.match(source, /isGenerating \? t\('studio\.tile\.generating'\) : t\('studio\.tile\.queued'\)/)
})
