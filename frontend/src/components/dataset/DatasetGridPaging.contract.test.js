/**
 * Paging the GRID must never page the CURATION.
 *
 * The whole point of the change is that only the rendering is windowed: a
 * selection made on page 3 has to survive page 7, "select all" has to keep
 * meaning "every image the filters show", and auto-triage/eligibility/bulk
 * actions have to keep reading the full list. Those are one-line regressions
 * (`images` → `view.items`) that no unit test would catch, because `node --test`
 * cannot mount this JSX — so they are pinned on the source.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8');

test('only the tile loop is paged', () => {
  assert.match(source, /const view = pageSlice\(images, page, GRID_PAGE_SIZE\);/);
  assert.match(source, /\{view\.items\.map\(\(img\) => \(/);
  assert.doesNotMatch(source, /\{images\.map\(\(img\) => \(/);
});

test('select all still takes the whole filtered list, not the page', () => {
  // `selectable` is derived from `images` (every filtered row), and both
  // select-all buttons build the selection from it.
  assert.match(source,
    /const selectable = images\.filter\(\(i\) => i\.filename && !isSmallImageRescueRow\(i\)\);/);
  const selectAllCalls = source.match(/setSelected\(new Set\(selectable\.map\(\(i\) => i\.id\)\)\)/g);
  assert.equal(selectAllCalls?.length, 2, 'both "all" buttons select the whole filtered list');
  assert.doesNotMatch(source, /new Set\(view\.items/);
});

test('the page-crossing reach of select all is stated, not implied', () => {
  const titles = source.match(
    /title="Selects every image the current filters show — all pages, not just the tiles on screen"/g);
  assert.equal(titles?.length, 2);
});

test('auto-triage and improve eligibility keep reading the full list', () => {
  assert.match(source, /<AutoTriageBar images=\{images\.filter\(\(image\) => !isSmallImageRescueRow\(image\)\)\}/);
  assert.match(source, /const improveUniverse = Array\.isArray\(eligibilityImages\) \? eligibilityImages : images;/);
  assert.match(source, /improvementStateByParent\(images\), \[images\]\);/);
});

test('a shrinking list cannot strand the user on a page that no longer exists', () => {
  assert.match(source, /setPage\(\(p\) => clampPage\(p, \(images \|\| \[\]\)\.length\)\);/);
  assert.match(source, /setPage\(clampPage\(next, images\.length\)\);/);
});

test('opening another dataset starts at page 1', () => {
  const resetBlock = source.match(/useEffect\(\(\) => \{[^}]*setSelected\(new Set\(\)\);[^}]*\}, \[datasetId\]\);/s);
  assert.ok(resetBlock, 'the dataset-switch reset effect is still there');
  assert.match(resetBlock[0], /setPage\(0\);/);
});

test('a page change lands at the top of the grid', () => {
  assert.match(source, /document\.getElementById\('ds-images-review'\)\?\.scrollIntoView/);
});

test('the pager is drawn above AND below the wall of tiles', () => {
  assert.match(source, /<GridPager view=\{view\} onGo=\{goToPage\} where="top" \/>/);
  assert.match(source, /<GridPager view=\{view\} onGo=\{goToPage\} where="bottom" \/>/);
  // Hidden when everything fits on one page — no dead "1–12 of 12".
  assert.match(source, /if \(!view\.paged\) return null;/);
});

test('pager buttons stay thumb-sized at 400 px', () => {
  // The grid's pager is the shared numbered one; the thumb-sized targets and
  // the wrapping row live there.
  const grid = source.slice(source.indexOf('function GridPager'), source.indexOf('export default function DatasetGrid'));
  assert.match(grid, /<Pager page=\{view\.page \+ 1\} size=\{view\.size\} total=\{view\.total\}/);
  const pager = readFileSync(new URL('../common/Pager.jsx', import.meta.url), 'utf8');
  assert.match(pager, /min-h-11 min-w-11/);
  assert.match(pager, /flex flex-wrap items-center/);
});
