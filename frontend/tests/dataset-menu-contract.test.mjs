import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const source = readFileSync(new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')

test('dataset More popover uses an opaque overlay surface', () => {
  const summary = source.indexOf("workspace.header.moreTitle")
  assert.notEqual(summary, -1, 'More menu summary not found')
  const panel = source.slice(summary, summary + 1200)
  assert.match(panel, /absolute right-0[^"\n]*bg-surface-overlay/)
})

test('setup never suggests the slow Qwen thinking tag for captioning', () => {
  const setup = readFileSync(new URL('../src/pages/SetupPage.jsx', import.meta.url), 'utf8')
  const tools = readFileSync(new URL('../src/components/settings/LocalToolsSection.jsx', import.meta.url), 'utf8')
  assert.match(setup, /huihui_ai\/qwen3-vl-abliterated:8b-instruct/)
  assert.doesNotMatch(setup, /huihui_ai\/qwen3-vl-abliterated:8b['"]/)
  assert.match(tools, /huihui_ai\/qwen3-vl-abliterated:8b-instruct/)
})
