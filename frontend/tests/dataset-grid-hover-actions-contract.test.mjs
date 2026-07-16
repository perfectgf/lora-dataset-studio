import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const item = fs.readFileSync(path.join(frontend, 'src/components/dataset/DatasetGridItem.jsx'), 'utf8')
const css = fs.readFileSync(path.join(frontend, 'src/index.css'), 'utf8')

test('all image-card control groups share the hover-action contract', () => {
  assert.match(item, /dataset-grid-item rounded-lg/)
  assert.ok((item.match(/dataset-grid-item__actions/g) || []).length >= 4)
})

test('fine pointers hide controls without reflow and hover or focus reveals them', () => {
  assert.match(css, /@media \(hover: hover\) and \(pointer: fine\)/)
  assert.match(css, /\.dataset-grid-item:hover \.dataset-grid-item__actions/)
  assert.match(css, /\.dataset-grid-item:focus-within \.dataset-grid-item__actions/)
  assert.match(css, /visibility: hidden/)
  assert.match(css, /pointer-events: none/)
  assert.doesNotMatch(css, /\.dataset-grid-item \.dataset-grid-item__actions\s*\{[^}]*display:\s*none/s)
})
