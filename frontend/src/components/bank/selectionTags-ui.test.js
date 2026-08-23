// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import test from 'node:test';
import assert from 'node:assert/strict';

/* THE WIRING of the 🏷️ row, which node --test cannot mount (JSX). The counting
 * itself is unit-tested in bankTags.test.js; what is pinned here is that the row
 * is driven by the SELECTION and not by a second click, and that the three ways
 * it can be short of the truth are all said out loud.
 *
 * Asked for in these words: "when the captions are already done and you select an
 * image, show the tags in every case. When several images are selected, show the
 * tags in common with the number of times it was cited."
 */
const ws = bankTreeSource()
  .replace(/\r\n/g, '\n');

test('the tag row opens on the SELECTION, with no second click', () => {
  // Not `tagSource &&` any more: a live selection produces the row by itself.
  assert.match(ws, /const selectionTags = useMemo\(/);
  assert.match(ws, /\{tagRow && \(/);
  // Priority: a frozen row, then the live selection, then the tile's 🏷️ button.
  assert.match(ws, /const tagRow = tagFreeze \|\| selectionTags \|\| \(tagSource \?/);
});

test('one selected image reads as one image, several read as several', () => {
  assert.match(ws, /🏷️ Tags of the selected image/);
  assert.match(ws, /🏷️ Tags across \$\{tagRow\.size\} selected images/);
  // The tile button keeps its own wording — it is a different gesture.
  assert.match(ws, /🏷️ Tags of \$\{tagRow\.name\}/);
});

test('the tag inspector is mounted ONCE, in the filter rail, at every width', () => {
  /* It used to be mounted TWICE — a phone copy in the filter zone and a sticky
     desktop inspector — with one of them hidden by `xl:hidden`. Two copies of a
     panel is two places for it to drift, and the reason it existed was that the
     old stack had no single home for a control that had to work at 400 px and
     at 1440. The rail is that home, so the duplicate is gone rather than moved,
     and this asserts the stronger property: exactly one. */
  const mounts = ws.match(/<SelectionTagsPanel /g) || [];
  assert.equal(mounts.length, 1, 'the tag panel is mounted exactly once');
  assert.doesNotMatch(ws, /<div className="xl:hidden">\s*<SelectionTagsPanel/);
  assert.match(ws, /onToggle=\{\(tag\) => toggleTag\(tag, tagRow\)\}/);
});

test('the tag inspector is still a NAMED landmark, and still ahead of the grid', () => {
  // Folding the two mounts into one nearly dropped this label; the surface
  // inventory caught it. A screen reader must still reach the panel by name.
  assert.match(ws, /<section aria-label="Image tags"/);
  assert.ok(ws.indexOf('aria-label="Image tags"')
    < ws.indexOf("{filter.flag === 'dups' ? ("),
  'the tags landmark comes before the grid in reading order');
});

test('every chip carries a FRACTION, never a bare count', () => {
  // "7" alone cannot be read; "7 / 12" is the judgement. tagCountLabel returns ''
  // for a single image, so the one-image row is unchanged.
  assert.match(ws, /const n = tagCountLabel\(count, tagRow\.counted\)/);
  assert.match(ws, /cited by \$\{count\} of the \$\{tagRow\.counted\} captioned images you selected/);
});

test('an empty row SAYS why instead of rendering an empty box', () => {
  assert.match(ws, /has no caption yet — run 🏷️ Caption/);
  assert.match(ws, /No caption here has a word worth filtering on/);
});

test('the shortfalls are all disclosed, none folded into the denominator', () => {
  assert.match(ws, /selectionTagsNotes\(tagRow, tagRow\.unread\)/);
});

test('ticking a chip FREEZES the row it was ticked in', () => {
  // setF clears the selection (it must — the ids no longer match the grid). A row
  // that vanished the moment you used it would read as a crash, so the snapshot
  // is taken before the filter is applied.
  assert.match(ws, /const toggleTag = \(tag, basis = null\) =>/);
  assert.match(ws, /if \(basis && basis\.kind === 'selection' && !basis\.frozen\) \{\s*\n\s*setTagFreeze\(\{ \.\.\.basis, frozen: true \}\)/);
  // …and a NEW selection replaces it, so a stale snapshot never sits over fresh ids.
  assert.match(ws, /useEffect\(\(\) => \{ if \(selected\.size\) setTagFreeze\(null\) \}, \[selected\]\)/);
});

test('off-page captions are fetched once, capped, and the cap is disclosed', () => {
  // `ids=` rides in the query string: a few thousand integers build a request line
  // the server refuses. The cap is a constant with its reason, not a magic number.
  assert.match(ws, /const TAG_CAPTION_FETCH_CAP = 500/);
  assert.match(ws, /missing\.slice\(0, TAG_CAPTION_FETCH_CAP\)/);
  // Only ids the grid never rendered are fetched — clicking tiles costs nothing.
  assert.match(ws, /ids\.filter\(\(id\) => !captionCache\.current\.has\(id\)\)/);
});

test('the tile 🏷️ button clears the selection, so it can never read as dead', () => {
  const fn = ws.slice(ws.indexOf('const openTagPicker'), ws.indexOf('const toggleTag'));
  assert.match(fn, /setSelected\(new Set\(\)\)/);
  assert.match(fn, /setTagFreeze\(null\)/);
});
