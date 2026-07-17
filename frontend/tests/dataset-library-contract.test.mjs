import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const panel = fs.readFileSync(path.join(frontend, 'src/components/dataset/DatasetListPanel.jsx'), 'utf8')
const grid = fs.readFileSync(path.join(frontend, 'src/components/dataset/DatasetGrid.jsx'), 'utf8')

test('library page persists its display preferences under stable keys', () => {
  assert.match(panel, /datasetLibraryTileSize/)
  assert.match(panel, /datasetLibraryCollapsed_v1/)
  // Both go through the storage-hardened normalizers, never raw JSON.parse.
  assert.match(panel, /normalizeTileSize\(localStorage\.getItem/)
  assert.match(panel, /normalizeCollapsedMap\(localStorage\.getItem/)
})

test('library filtering/grouping goes through the tested pure helpers', () => {
  assert.match(panel, /from '\.\.\/\.\.\/utils\/datasetLibrary'/)
  assert.match(panel, /datasetMatches\(d, query, kindFilter\)/)
  assert.match(panel, /groupDatasets\(filtered\)/)
})

test('family sections are collapsible and announce their state', () => {
  assert.match(panel, /aria-expanded=\{open\}/)
  // A fold must never hide search/filter matches — sections force open.
  assert.match(panel, /filterActive \|\| !collapsed\[family\]/)
  assert.match(panel, /disabled=\{filterActive\}/)
})

test('the S/M/L control is the shared segmented component, in both grids', () => {
  assert.match(panel, /import TileSizeControl from '\.\.\/shared\/TileSizeControl'/)
  assert.match(grid, /import TileSizeControl from '\.\.\/shared\/TileSizeControl'/)
  // The library keeps the workspace grid's key untouched (separate prefs).
  assert.match(grid, /datasetGridTileSize/)
})

test('S size renders compact rows, M/L render photo-tile grids', () => {
  assert.match(panel, /tileSize === 'S' \? \(/)
  assert.match(panel, /<DatasetRow /)
  assert.match(panel, /<DatasetTile /)
})
