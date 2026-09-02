/* The Dataset clip tab of the start-frame picker is a grid of posters.
 *
 * Found in use (2026-09-02): the tab listed clips as a FLEX COLUMN capped at
 * max-h-72 with `truncate` rows — a flex column shrinks its children to fit
 * the cap rather than scrolling, and truncate's overflow:hidden lets them
 * shrink to nothing — so 21 clips came up as 21 slivers of ~12 px, unreadable
 * and unclickable, while every request behind them answered 200. No test saw
 * it: `node --test` cannot render JSX. This file pins the shape that scrolls.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')
const PICKER = read('./VideoSourcePicker.jsx')

/** The clip tab's JSX, from its guard to the next tab-guarded block. */
function clipTab(src) {
  const start = src.indexOf("{tab === 'clip' && (")
  assert.ok(start > 0, 'the Dataset clip tab is gone')
  const end = src.indexOf('{image && (', start)
  assert.ok(end > start, 'the staged-picture block that follows the tab is gone')
  return src.slice(start, end)
}

test('the clip list is a scrolling grid, not a flex column that shrinks its rows', () => {
  const tab = clipTab(PICKER)
  const scrollBox = tab.match(/className="([^"]*overflow-y-auto[^"]*)"/)
  assert.ok(scrollBox, 'the clip list has no scrolling box')
  const classes = scrollBox[1].split(/\s+/)
  assert.ok(classes.includes('grid'), `the scrolling box is not a grid: ${scrollBox[1]}`)
  assert.ok(!classes.includes('flex-col'), `a flex column shrinks its rows under a max height: ${scrollBox[1]}`)
})

test('one Preview size dial drives all three grids, and remembers itself', () => {
  // Asked for from the picker (2026-09-02): "a slider to enlarge the start
  // frame previews". One dial, not one per tab — a size chosen on the Bank
  // grid holds on the Gallery and the Dataset clip grids.
  const ranges = PICKER.match(/<input type="range"[^]*?\/>/g) || []
  assert.equal(ranges.length, 1, 'the picker has exactly one range input, the preview size')
  assert.match(ranges[0], /aria-label="Preview size"/)
  assert.match(ranges[0], /min=\{TILE_MIN\} max=\{TILE_MAX\} step=\{TILE_STEP\}/,
    'the dial\u2019s range comes from videoPickerTile, not from literals that can drift from it')
  const grids = PICKER.match(/className="grid gap-1 overflow-y-auto" style=\{gridStyle\}/g) || []
  assert.equal(grids.length, 3, 'the Bank, Gallery and Dataset clip grids all take the one gridStyle')
  assert.doesNotMatch(PICKER, /grid-cols-\d|sm:grid-cols-\d/,
    'a fixed column count would ignore the dial')
  assert.match(PICKER, /repeat\(auto-fill, minmax\(\$\{tile\}px, 1fr\)\)/)
  // The other half of gridStyle: with the fixed max-h classes gone, this cap
  // is all that keeps a 640 px box off a landscape phone's fold (70vh = 273).
  assert.match(PICKER, /maxHeight: `min\(\$\{gridBoxHeight\(tile\)\}px, 70vh\)`/,
    'the box must stay capped to the viewport, or a big tile eats a phone\u2019s fold')
  // Where and when the dial sits — three decisions the responsive probe
  // measured, and the probe is not part of the nightly gate, so pin them:
  // only over a grid that has tiles, at the end of the tab strip's row
  // (alone on a row it filled 35 % of a landscape phone's), a row that wraps
  // on a phone while the tab labels stay whole above one.
  assert.match(PICKER, /const gridShown = \(tab === 'bank' && bankId && images\.length > 0\)/)
  assert.match(PICKER, /\{gridShown && \(\n\s*<label className="ml-auto flex/)
  assert.match(PICKER, /<div className="flex w-full flex-wrap gap-1">/)
  assert.match(PICKER, /data-testid=\{`video-source-\$\{id\}`\}\n\s*className=\{`[^`]*\bsm:whitespace-nowrap\b/)
  // The size survives a reload, through the helper that clamps it — the JSX
  // never touches the store by hand.
  // The store is the helper's, never named here: a browser that blocks site
  // data throws on ACCESS to localStorage, and this read happens in a render.
  // And the state follows the VALUE — set from the store after a write, a
  // refused write (quota, private mode) left the dial inert.
  assert.match(PICKER, /useState\(\(\) => readTile\(\)\)/)
  assert.match(PICKER, /const next = clampTile\(value\);\n\s*setTile\(next\);\n\s*writeTile\(next\);/)
  assert.doesNotMatch(PICKER, /localStorage/)
})

test('a clip tile shows the poster the training set shows, and stages it as the preview', () => {
  const tab = clipTab(PICKER)
  assert.match(tab, /datasetClipPoster\(datasetId, c\)/, 'the tile does not resolve its poster')
  assert.match(tab, /preview: poster/, 'the picked clip leaves the staged picture without a preview')
  // The name still travels with the tile — the poster can be a placeholder.
  assert.match(tab, /title=\{c\.filename\}/)
  assert.match(tab, /\{c\.filename\}\s*<\/span>/)
  // The server is asked the way it answers: by dataset and file name.
  assert.match(tab, /\{ dataset_id: datasetId, filename: c\.filename \}/)
})

test('a poster that cannot load becomes its placeholder — in the tile and beside Ready', () => {
  // A bank thumbnail 404s in the ordinary course of things (bank deleted,
  // thumbnails pass never run). Hiding the <img> left a blank tile; a
  // component that swaps in the placeholder, and forgets "broken" when its
  // source changes, keeps every tile a tile.
  const poster = PICKER.slice(PICKER.indexOf('function Poster('), PICKER.indexOf('export default function'))
  assert.ok(poster.length > 0, 'the Poster component is gone')
  assert.match(poster, /onError=\{\(\) => setBroken\(true\)\}/)
  assert.match(poster, /useEffect\(\(\) => \{ setBroken\(false\); \}, \[src\]\)/)
  assert.match(poster, /if \(!src \|\| broken\) return fallback/)
  const tab = clipTab(PICKER)
  assert.match(tab, /<Poster src=\{poster\}[^>]*\n\s*fallback=/, 'the clip tile does not fall back on its placeholder')
  assert.doesNotMatch(tab, /<img src=\{poster\}/)
  const ready = PICKER.slice(PICKER.indexOf('{image && ('))
  assert.match(ready, /<Poster src=\{preview\}/, 'the Ready block does not fall back on its icon')
  assert.doesNotMatch(ready, /<img src=\{preview\}/)
})

test('the clip list empties when the set changes, and a late reply is dropped', () => {
  // Without the reset, the previous set's tiles stayed up under the new name
  // until the new reply arrived; without the flag, a slow first reply could
  // overwrite a fast second one for good. The empty message waits for the
  // reply rather than flashing while the set loads.
  const start = PICKER.indexOf('if (!datasetId) return')
  assert.ok(start > 0, 'the clips effect is gone')
  const effect = PICKER.slice(start, PICKER.indexOf('}, [datasetId]);', start))
  assert.match(effect, /let stale = false;/)
  assert.match(effect, /setClips\(\[\]\);/)
  assert.match(effect, /if \(!stale\) setClips\(datasetClips\(d\)\)/)
  assert.match(effect, /return \(\) => \{ stale = true; \};/)
  assert.match(clipTab(PICKER), /datasetId && !clipsLoading && clips\.length === 0 &&/)
})

test('the picker paints its active tab with a colour the theme defines', () => {
  // `accent` was never a Tailwind colour token in this app — `border-accent`,
  // `bg-accent/10` and friends generated no CSS, so the active tab and the
  // picked mode were never highlighted. The app's idiom is `primary`.
  assert.doesNotMatch(PICKER, /(?<![\w-])[\w:]*-accent(?![\w-])/, 'a class on the undefined `accent` colour')
})
