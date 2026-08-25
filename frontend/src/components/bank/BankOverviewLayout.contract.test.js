// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const app = readFileSync(new URL('../../App.jsx', import.meta.url), 'utf8')
const page = readFileSync(new URL('../../pages/BankPage.jsx', import.meta.url), 'utf8')
const workspace = bankTreeSource()
const facets = readFileSync(new URL('./bankFacets.js', import.meta.url), 'utf8')
const overview = readFileSync(new URL('./BankOverview.jsx', import.meta.url), 'utf8')

test('/bank shares the wide 1800px shell with Canvas', () => {
  assert.match(app, /pathname === '\/canvas' \|\| pathname === '\/bank'/)
  assert.match(app, /max-w-\[1800px\]/)
})

test('bank list grows to three columns only at xl', () => {
  assert.match(page, /grid-cols-1 sm:grid-cols-2 xl:grid-cols-3/)
})

test('the four-megapixel resolution bucket is inclusive in the workspace too', () => {
  // The resolution tiers are a DATA table now (bankFacets.js), shared by the
  // filter rail and the tiles — assert them where they are defined.
  assert.match(facets, /id: 'res_gt_4', label: '≥ 4 MP'/)
  assert.doesNotMatch(facets, /id: 'res_gt_4', label: '> 4 MP'/)
})

test('the rail sits beside the grid, and folds instead of squeezing it', () => {
  /* This used to assert the FOUR-ZONE STACK: two twelve-column xl grids pairing
     Analyze with the overview and Curate with Promote. The Encre redesign
     replaced that stack on purpose — scrolling up to a filter and back down to
     its result was the actual complaint — so the invariant is rewritten, not
     relaxed. What has to stay true is the same thing it always protected: the
     screen is a single column on a phone and uses the width on a desktop.

     Two columns, not twelve: the rail has one job and a fixed measure, so a
     twelve-column grid would only be a more expensive way of writing 17rem.

     The breakpoint is `lg`, not `sm`, and that is the point of the assertion:
     a 17rem rail FITS from 640 px, but it leaves the grid ~350 px — two
     thumbnails — and a triage screen that shows two images is not one. The
     rail stays a drawer until the grid keeps a workable width. Pinning the
     literal here keeps it in step with RAIL_SIDE_BY_SIDE_PX, which the layout
     module tests on the same reasoning. */
  assert.match(workspace, /lg:grid-cols-\[17rem_minmax\(0,1fr\)\]/)
  // …and it really does collapse to one column rather than shrinking the grid.
  assert.match(workspace, /railOpen && railIsColumnNow/)
  assert.match(workspace, /: 'grid-cols-1'/)
  // At 400 px the rail is a drawer OVER the grid, with a backdrop that closes it.
  assert.match(workspace, /isDrawer=\{!railIsColumnNow\}/)
  assert.match(workspace, /railOpen && !railIsColumnNow && \(/)

  // The passes panel keeps the twelve-column pairing: the action list and the
  // read-only overview answer the same question and belong side by side.
  const panel = readFileSync(new URL('./BankPassesPanel.jsx', import.meta.url), 'utf8')
  assert.match(panel, /grid gap-4 xl:grid-cols-12 xl:items-start/)
  assert.match(panel, /xl:col-span-7/)
  assert.match(panel, /xl:col-span-5/)
  assert.match(panel, /<BankOverview payload=\{payload\} \/>/)
})

test('the rail stays on screen while the grid scrolls under it', () => {
  /* The layout's whole justification is that you no longer scroll up to a
     filter and back down to its result. A rail that is merely PLACED beside
     the grid does not deliver that: it is ~500 px tall against a grid that is
     thousands (20 000 images), so it leaves the viewport after one screen and
     the round trip comes straight back. Measured on a 48-image bank at 1440:
     page 2 326 px, rail ending at ~1 345 — the lower 1 000 px of grid had no
     filters beside it at all.

     Three parts, and the assertion names all three because dropping any one
     silently restores the defect:
       · `sticky` — the pin itself;
       · `self-start` — without it the grid item STRETCHES to the row height,
         so the element is already as tall as its container and sticky has no
         room to travel. This is the failure that looks like "sticky doesn't
         work" and gets fixed by deleting sticky;
       · `overflow-y-auto` + a viewport-bounded max height — a rail taller than
         the screen must scroll inside its pin, or its lower half becomes
         permanently unreachable, which is worse than never pinning.
     All four are `lg:`-scoped, matching the breakpoint at which the rail is a
     column at all; below it the rail is a `fixed` drawer, where sticky is
     meaningless. */
  assert.match(workspace, /lg:sticky/)
  assert.match(workspace, /lg:self-start/)
  assert.match(workspace, /lg:overflow-y-auto/)
  assert.match(workspace, /lg:max-h-\[calc\(100vh-var\(--app-header-h\)-1\.5rem\)\]/)
  // …and only as a column: the drawer branch must stay plain.
  assert.match(workspace, /railIsColumnNow\s*\n?\s*\?\s*'min-w-0 lg:sticky/)

  /* The pin clears the app's own sticky top bar, and it does so through a
     token rather than a number repeated in two files.
     ⚠️ This last assertion is the one that matters most, because the failure it
     guards is SILENT: an undefined custom property makes the whole `calc()`
     invalid, `top` falls back to `auto`, and a `sticky` element with no `top`
     simply never pins. Nothing errors, nothing looks broken in review — the
     rail just quietly scrolls away again and the layout loses its reason to
     exist. Deleting the token from index.css must break a test, not a screen. */
  const css = readFileSync(new URL('../../index.css', import.meta.url), 'utf8')
  assert.match(css, /--app-header-h:\s*\d/)
  assert.match(workspace, /lg:top-\[calc\(var\(--app-header-h\)\+0\.75rem\)\]/)
})

test('overview never opens the expensive kept-only coverage endpoint', () => {
  const panel = readFileSync(new URL('./BankPassesPanel.jsx', import.meta.url), 'utf8')
  const overviewMount = panel.slice(panel.indexOf('<BankOverview'),
    panel.indexOf('<BankOverview') + 200)
  assert.doesNotMatch(overviewMount, /coverage/)
})

test('non-zero Bank segments use exact widths and remain physically visible', () => {
  assert.match(overview, /width: `\$\{row\.widthPercent\}%`/)
  assert.match(overview, /minWidth: row\.value > 0 \? '1px'/)
  assert.match(page, /width: `\$\{row\.widthPercent\}%`, minWidth: '1px'/)
  assert.doesNotMatch(page, /width: `\$\{row\.percent\}%`/)
})

test('the overview is open by default and folds without tying its state to live payload refreshes', () => {
  assert.match(overview, /const \[open, setOpen\] = useState\(true\)/)
  assert.match(overview, /onClick=\{\(\) => setOpen\(\(value\) => !value\)\}/)
  assert.match(overview, /aria-expanded=\{open\}/)
  assert.match(overview, /aria-controls=\{contentId\}/)
  assert.match(overview, /<div id=\{contentId\} hidden=\{!open\}/)
  assert.doesNotMatch(overview, /useState\([^)]*payload/)
})

test('the overview header and live total stay visible while its details fold', () => {
  const detailsAt = overview.indexOf('<div id={contentId}')
  assert.ok(detailsAt > 0)
  assert.ok(overview.indexOf('>Bank overview') < detailsAt)
  assert.ok(overview.indexOf('{totalText}') < detailsAt)
})

test('the overview has an explicit unavailable state before bank data arrives', () => {
  assert.match(overview, /!model\.available \? \(/)
  assert.match(overview, /Overview unavailable — waiting for bank data/)
  assert.match(overview, /'Total unavailable'/)
})
