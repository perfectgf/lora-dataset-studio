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

/** The strip of staged frames follows the last tab: its guard is the end of
 * the clip tab's JSX. */
const STRIP_GUARD = '{frames.length > 0 && ('

/** The clip tab's JSX, from its guard to the strip that follows it. */
function clipTab(src) {
  const start = src.indexOf("{tab === 'clip' && (")
  assert.ok(start > 0, 'the Dataset clip tab is gone')
  const end = src.indexOf(STRIP_GUARD, start)
  assert.ok(end > start, 'the strip of staged frames that follows the tab is gone')
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
  const strip = PICKER.slice(PICKER.indexOf(STRIP_GUARD))
  assert.match(strip, /<Poster src=\{f\.preview\}/, 'the strip does not fall back on its icon')
  assert.doesNotMatch(strip, /<img src=\{f\.preview\}/)
})

test('several start frames: a pick appends to a strip, each frame has its ✕, and the parent gets a batch once', () => {
  // Asked for from the picker (2026-09-02): "batch the image inputs". The
  // strip is the parent's list — what Generate walks — so the picker takes
  // it as a prop and hands additions up; it never keeps a frame of its own.
  assert.match(PICKER, /export default function VideoSourcePicker\(\{ mode, onMode, frames = \[\], onAdd, onRemove, onClear, aspect, onAspect \}\)/)
  assert.doesNotMatch(PICKER, /onPicked|useState\(\{ image: null/)
  // The upload takes several files, and forgets the pick so the same file
  // can be chosen again after a removal (an unchanged value fires no change).
  assert.match(PICKER, /<input type="file" accept="image\/\*" multiple className="hidden"\n\s*onChange=\{\(e\) => \{ onFiles\(e\.target\.files\); e\.target\.value = ''; \}\} \/>/)
  // "Drop images here" is now true: the label listens for the drop.
  assert.match(PICKER, /onDrop=\{\(e\) => \{ e\.preventDefault\(\); if \(!busy\) onFiles\(e\.dataTransfer\.files\); \}\}/)
  // Staging walks the WHOLE list in order — a refused pick is refused alone,
  // inside the loop, and the walk goes on — and hands up ONCE, in `finally`,
  // so five pictures with one bad file are four frames and one message that
  // counts them (refuted 2026-09-02: the first refusal used to end the walk
  // and say nothing of the rest).
  const stage = PICKER.slice(PICKER.indexOf('const stage = useCallback(async (picks)'), PICKER.indexOf('}, [frames, onAdd, toast]);'))
  assert.ok(stage.length > 0, 'the staging walk is gone')
  assert.match(stage, /for \(const pick of fresh\) \{\n\s*try \{\n\s*const r = await pick\.send\(\);/)
  assert.match(stage, /\} catch \(e\) \{\n\s*releasePreview\(pick\);\n\s*if \(!refusal\) refusal = e\?\.message/)
  assert.match(stage, /\} finally \{\n\s*fresh\.forEach\(\(pick\) => inFlight\.current\.delete\(pick\.key\)\);\n\s*if \(staged\.length\) onAdd\(staged\);/)
  assert.match(stage, /`Staged \$\{staged\.length\} of \$\{fresh\.length\} — \$\{refusal\}`/)
  assert.equal((stage.match(/onAdd\(/g) || []).length, 1, 'the parent is handed the batch exactly once')
  // Dedupe is by ORIGIN, before any request: the server stages every pick
  // under a fresh name, so the staged name cannot say "same picture". A pick
  // whose staging is in flight is skipped the same way (a double click on a
  // tile that is not pressed yet), and a pick that never stages lets go of
  // the blob URL the upload minted for it.
  assert.match(stage, /const seen = new Set\(frames\.map\(\(f\) => f\.key\)\);/)
  assert.match(stage, /if \(inFlight\.current\.has\(pick\.key\)\) \{ releasePreview\(pick\); continue; \}/)
  assert.match(stage, /if \(seen\.has\(pick\.key\)\) \{ releasePreview\(pick\); dropped \+= 1; continue; \}/)
  assert.match(stage, /fresh\.forEach\(\(pick\) => inFlight\.current\.add\(pick\.key\)\);/)
  assert.match(PICKER, /const inFlight = useRef\(new Set\(\)\);/)
  // The two ways out of the strip let go of the previews too.
  assert.match(PICKER, /onClick=\{\(\) => \{ releasePreview\(f\); onRemove\(f\.key\); \}\}/)
  assert.match(PICKER, /onClick=\{\(\) => \{ frames\.forEach\(releasePreview\); onClear\(\); \}\}/)
  // The two "already in the batch" sentences, the only sign a re-pick gives.
  assert.match(stage, /'Already in the batch — remove it from the strip to pick it again\.'/)
  assert.match(stage, /`\$\{dropped\} already in the batch — skipped\.`/)
  assert.match(PICKER, /uploadKey\(file\)/)
  for (const key of ['`bank:${bankId}:${im.id}`', '`gallery:${g.id}`', '`clip:${datasetId}:${c.filename}`']) {
    assert.ok(PICKER.includes(key), `a tile is not keyed by its origin: ${key}`)
  }
  // A tile in the strip reads as pressed and clicks OUT again.
  assert.equal((PICKER.match(/aria-pressed=\{held\(key\)\}/g) || []).length, 3, 'the three grids mark a held tile')
  assert.match(PICKER, /const toggle = \(pick\) => \(held\(pick\.key\) \? onRemove\(pick\.key\) : stage\(\[pick\]\)\);/)
  // The strip: one ✕ per frame, named for a screen reader, and Clear all
  // only when there is more than one to clear.
  const strip = PICKER.slice(PICKER.indexOf(STRIP_GUARD))
  assert.match(strip, /\{frames\.map\(\(f, i\) => \(/)
  // ✕ and Clear all let go of the upload preview BEFORE the frame leaves the
  // strip — an object URL nobody revokes lives until the page does.
  assert.match(strip, /onClick=\{\(\) => \{ releasePreview\(f\); onRemove\(f\.key\); \}\}/)
  assert.match(strip, /aria-label=\{`Remove start frame \$\{i \+ 1\}`\}/)
  assert.match(strip, /\{frames\.length > 1 && \(\n\s*<button type="button" onClick=\{\(\) => \{ frames\.forEach\(releasePreview\); onClear\(\); \}\}/)
  // One frame reads as it always did; several say what the click will do.
  assert.match(strip, /Ready — staged into ComfyUI as/)
  assert.match(strip, /\{frames\.length\} start frames — one clip each, on one seed; ✨ reads the first\./)
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
