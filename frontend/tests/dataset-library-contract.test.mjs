import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontend = path.resolve(here, '..')
const panel = fs.readFileSync(path.join(frontend, 'src/components/dataset/DatasetListPanel.jsx'), 'utf8')
const grid = fs.readFileSync(path.join(frontend, 'src/components/dataset/DatasetGrid.jsx'), 'utf8')
const css = fs.readFileSync(path.join(frontend, 'src/index.css'), 'utf8')
const page = fs.readFileSync(path.join(frontend, 'src/pages/DatasetPage.jsx'), 'utf8')

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

test('desktop-first: the library uses the full page width and dense columns', () => {
  // The old max-w-4xl cap must not come back around the list panel (the
  // empty-state hero and creation form re-cap themselves inside the panel).
  assert.doesNotMatch(page, /max-w-4xl/)
  assert.match(panel, /lg:grid-cols-4/)
})

test('library cards follow the fine-pointer hover-action contract', () => {
  assert.match(css, /\.library-card:hover \.library-card__actions/)
  assert.match(css, /\.library-card:focus-within \.library-card__actions/)
  // Hidden without reflow (visibility, not display) and touch keeps controls.
  assert.match(css, /\.library-card \.library-card__actions\s*\{[^}]*visibility: hidden/s)
  assert.doesNotMatch(css, /\.library-card \.library-card__actions\s*\{[^}]*display:\s*none/s)
  // Applied on the photo tile (export bar + delete overlay) AND the S row.
  assert.ok((panel.match(/library-card__actions/g) || []).length >= 3)
  assert.ok((panel.match(/className="library-card /g) || []).length >= 2)
})
