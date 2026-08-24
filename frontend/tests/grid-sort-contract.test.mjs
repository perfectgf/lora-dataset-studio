/**
 * Contract test for the two Sort controls. node --test cannot parse JSX, so the
 * ORDERING LOGIC lives in src/utils/gridSort.js (unit-tested there) and this file
 * greps the JSX for the wiring that logic depends on — the part a rewrite or a
 * "quick tidy" silently drops, leaving a menu that renders but sorts nothing.
 */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

import { BANK_SORTS } from '../src/utils/gridSort.js'

const read = readSource
const BANK = read('src/components/bank/BankWorkspace.jsx')
const BANK_TILE = read('src/components/bank/BankTile.jsx')
// The Sort menu and the text filters live in the RAIL now — beside the grid
// they order, which is the whole point of the Encre restructure.
const BANK_RAIL = read('src/components/bank/BankFilterRail.jsx')
const BANK_PANEL = read('src/components/bank/SelectionTagsPanel.jsx')
const WORKSPACE = read('src/components/dataset/DatasetWorkspace.jsx')
const CSS = read('src/index.css')

test('the bank Sort menu is built from the registry, not hard-coded options', () => {
  assert.match(BANK, /import \{ bankSortGroups, loadBankSort, saveBankSort \} from '\.\.\/\.\.\/utils\/gridSort\.js'/)
  assert.match(BANK, /bankSortGroups\(\s*\n?\s*counts \? \{ \.\.\.counts, faces: payload\?\.faces_scanned \} : counts\)/,
    'the menu must be rendered from bankSortGroups(payload counts + faces_scanned)')
  assert.match(BANK_RAIL, /sortGroups\.map\(\(g\) =>/,
    'the options must come from the grouped registry, one <optgroup> per pass')
  assert.match(BANK_RAIL, /<optgroup key=\{g\.group\} label=\{g\.group\}>/)
  assert.match(BANK_RAIL, /disabled=\{o\.disabled\}/,
    'an option with no data behind it must actually be disabled')
  // No option may be written by hand any more: a literal <option value="res_desc">
  // would silently escape the enabled/disabled rules.
  for (const s of BANK_SORTS) {
    assert.doesNotMatch(BANK, new RegExp(`<option value="${s.id}"`),
      `${s.id} is hard-coded in JSX instead of coming from BANK_SORTS`)
  }
})

test('the id snapshot is fetched lean — one request, ids only', () => {
  // Reported on a 22 940-image bank: ▶ Review took seconds to open. fetchAllIds
  // walked the grid 500 rows at a time and kept only `i.id` — 46 sequential
  // round trips for 16 MB of image payloads, each page re-running the COUNT and
  // the ORDER BY. Measured after: 3771 ms → 44 ms with a measure sort active.
  assert.match(BANK, /ids_only: '1'/,
    'the id snapshot must ask for the lean answer')
  assert.match(BANK, /return d\.ids \|\| \[\]/,
    'and read the ids straight off it')
  // The pagination loop must NOT come back: it is the bug.
  assert.doesNotMatch(BANK, /ids\.push\(\.\.\.d\.images\.map/,
    'walking the paginated grid to collect ids is what made Review slow')
  assert.doesNotMatch(BANK, /limit: '500'/,
    'no 500-row page walk for an id snapshot')
})

test('the bank sort rides to the server (whole filter) and to fetchAllIds', () => {
  // The sort is a query parameter, so SQL orders the WHOLE selection; the same
  // params object feeds fetchAllIds, so "Select all" / Review walk that order.
  assert.match(BANK, /if \(f\.sort && f\.sort !== 'default'\) params\.sort = f\.sort/)
  assert.match(BANK, /fetchAllIds\(bankId, filterParams\(filter\)\)/)
  // …and the page fetch uses the same filterParams(f).
  assert.match(BANK, /\{ \.\.\.filterParams\(f\), offset: String\(off\)/)
})

test('the chosen bank order is remembered, per bank, on both ends', () => {
  // Read at open (and at every bank SWITCH — the workspace is not keyed by id,
  // so without this the second bank inherits the first one's order)…
  assert.match(BANK, /sort: loadBankSort\(bankId\)/)
  assert.match(BANK, /const f = \{ \.\.\.filter, sort: loadBankSort\(bankId\) \}/)
  // …and written at every choice. A read with no write is a preference nobody
  // can set; a write with no read is one nobody gets back.
  assert.match(BANK, /saveBankSort\(bankId, sort\)/)
})

test('the exclude filter is wired like every other facet', () => {
  // The box exists, is debounced into the filter, and clears the page/selection
  // through setF — a text filter that skipped setF would leave page 5 of a
  // narrower result set on screen.
  assert.match(BANK, /setF\(\{ exclude: term \|\| null \}\)/)
  assert.match(BANK_RAIL, /aria-label="Hide images whose caption or file name contains these words"/)
  // It rides to the server on the SAME params object as the search and the sort,
  // so the grid, "Select all in filter", ▶ Review and the curation picks agree
  // on what is hidden.
  assert.match(BANK, /if \(f\.exclude\) params\.exclude = f\.exclude/)
  // Its own clear button — same affordance as the search it mirrors.
  assert.match(BANK_RAIL, /aria-label="Clear the exclude filter"/)
})

test('the 🏷️ tag chips are wired to the gallery filters', () => {
  // The chips come from the pure module (node --test cannot parse JSX, so the
  // extraction logic is unit-tested there and only the WIRING is greppable).
  // The whole tag lane comes from that one module — including the selection
  // aggregation, so there is never a second copy of "what is a tag" in the JSX.
  assert.match(BANK_TILE, /captionChips/)
  /* The tag lane is spread over three files since the Encre redesign, and the
     property that matters is unchanged: every piece of "what is a tag" is
     IMPORTED from bankTags.js, never re-implemented in the JSX. So each file is
     checked against the module it must borrow from. */
  assert.match(BANK, /import \{ tagsParam, selectionTagCounts \} from '\.\/bankTags\.js'/)
  assert.match(BANK_PANEL, /selectionTagsNotes, tagCountLabel, tagFilterSummary \} from '\.\/bankTags\.js'/)
  // …and nowhere does a component roll its own counting or wording.
  for (const [name, src] of [['BankWorkspace', BANK], ['BankTile', BANK_TILE],
    ['BankFilterRail', BANK_RAIL], ['SelectionTagsPanel', BANK_PANEL]]) {
    assert.doesNotMatch(src, /function (captionChips|tagCountLabel|tagFilterSummary)/,
      `${name} must borrow the tag helpers, not redefine them`)
  }
  // Its OWN payload key — never folded into search/exclude/push_down.
  assert.match(BANK, /if \(f\.tags\) params\.tags = f\.tags/)
  assert.match(BANK, /setF\(\{ tags: tagsParam\(next\) \}\)/)
  // The badge is a real button, so the gesture is reachable without a mouse.
  assert.match(BANK_TILE, /aria-label=\{`Use the tags of \$\{img\.name\} as a filter`\}/)
  // AND is written out for the user, not left to be inferred from the chips.
  assert.match(read('src/components/bank/SelectionTagsPanel.jsx'),
    /\{tagFilterSummary\(tagPicked\)\}/)
  // The row stays with the filter/gallery surface (filter zone on phones,
  // right-hand gallery inspector on desktop), never inside the review lightbox:
  // that one walks a frozen snapshot a filter change could not honestly alter.
  const LIGHTBOX = read('src/components/bank/BankReviewLightbox.jsx')
  assert.doesNotMatch(LIGHTBOX, /captionChips|tagsParam/,
    'filtering is a grid gesture; the review lightbox is a frozen decision run')
})

test('the dataset grid renders the sorted+filtered list, and sorts last', () => {
  assert.match(WORKSPACE, /sortDatasetImages, \}? ?from '\.\.\/\.\.\/utils\/gridSort'|from '\.\.\/\.\.\/utils\/gridSort'/)
  // Sort wraps the filters — membership stays the filters' business. The three
  // filters (decision, caption tags, coverage chip) build one list; the sort only
  // ever reorders THAT list, and is the last thing applied.
  assert.match(WORKSPACE, /const filteredGridImages = filterImages\(\n\s*filterImagesByStatus\(/)
  assert.match(WORKSPACE, /const gridImages = sortDatasetImages\(coverageIds/)
  assert.match(WORKSPACE, /: filteredGridImages, gridSort\);/)
  // The coverage chip narrows by ID — never by re-reading the captions in the
  // browser, which would be a second, drifting copy of the lexicon.
  assert.match(WORKSPACE, /filteredGridImages\.filter\(\(i\) => coverageIds\.has\(i\.id\)\)/)
  // The very list that was sorted is what the grid (and thus its select-all) gets.
  assert.match(WORKSPACE, /<DatasetGrid images=\{gridImages\}/)
  assert.match(WORKSPACE, /<GridSortSelect value=\{gridSort\}/)
})

test('the persisted sort id is normalised on read (legacy/hand-edited values)', () => {
  assert.match(WORKSPACE, /const GRID_SORT_KEY = 'datasetGridSort'/)
  assert.match(WORKSPACE, /normalizeDatasetSort\(localStorage\.getItem\(GRID_SORT_KEY\)\)/)
})

test('both selects stay inside a 400 px toolbar', () => {
  // A <select> with long option labels stretches its own box; without a bound it
  // pushes the toolbar into a horizontal scroll on a phone.
  /* The bank's Sort menu is no longer in a flex toolbar it could stretch —
     it is in the rail, a column of fixed width (17rem beside the grid, a
     19rem drawer at 400 px). `min-w-0 flex-1` is the equivalent guarantee
     in that context and a stronger one: a max-width bounds the select at
     one hard-coded size, whereas min-w-0 lets it shrink to whatever the
     rail actually is. min-w-0 is the load-bearing half — without it a
     flex child refuses to go below its content width and overflows. */
  assert.match(BANK_RAIL, /min-w-0 flex-1 rounded-md border border-border bg-surface[^"]*text-xs text-content"\n?\s*>\n?\s*\{sortGroups/)
  assert.match(WORKSPACE, /max-w-\[13rem\]/)
  // The dataset control wraps onto its own line rather than squeezing the chips.
  assert.match(WORKSPACE, /flex flex-wrap items-center gap-x-3 gap-y-1\.5/)
  assert.match(WORKSPACE, /flex shrink-0 items-center gap-1 text-xs/)
})

test('the grouped menu is readable in a dark-only app', () => {
  // Reported from a real 13 299-image bank: the Sort menu's group headers came
  // out as WHITE BANDS with near-invisible text. The page can re-colour an
  // <option> from CSS but NOT an <optgroup> header — the browser owns the popup
  // and, absent `color-scheme`, paints it with the OS light palette.
  assert.match(CSS, /:root,\s*\n\[data-theme="dark"\] \{[\s\S]{0,600}?color-scheme: dark;/,
    'a dark-only app must declare color-scheme: dark — that is what darkens the native popup')
  // The belt-and-braces half, for engines that honour the declarations instead
  // (Firefox). `option` alone was the gap that let this ship.
  assert.match(CSS, /\n\s*optgroup \{[^}]*background-color: rgb\(var\(--surface-overlay\)\);/)
  assert.match(CSS, /\n\s*optgroup \{[^}]*color: rgb\(var\(--content-muted\)\);/)
  assert.match(CSS, /\n\s*option \{[^}]*background-color: rgb\(var\(--surface-overlay\)\);/,
    'the pre-existing option rule must survive — it themes every other menu')
})

test('both selects are labelled for screen readers', () => {
  assert.match(BANK_RAIL, /aria-label="Sort the grid"/)
  assert.match(WORKSPACE, /aria-label="Sort the grid"/)
})
