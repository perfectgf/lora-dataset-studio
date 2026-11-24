/* The selection gesture is ONE gesture across two surfaces (Bank grid, dataset
 * grid), and it must never tint the picture being judged — that is the whole
 * premise of the Safelight theme. These read the sources as text, which is all
 * `node --test` can do, so they pin the two things that can silently regress:
 * the shared component being used on both sides, and the tint not coming back. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '../src');
const read = (p) => readFileSync(resolve(SRC, p), 'utf8');

const GRIDS = ['components/bank/BankTile.jsx', 'components/dataset/DatasetGridItem.jsx'];

test('both grids draw the same selection mark, from the shared component', () => {
  for (const f of GRIDS) {
    const src = read(f);
    assert.match(src, /import SelectionMark from '\.\.\/shared\/SelectionMark'/,
      `${f} must use the shared mark, not its own`);
    assert.match(src, /selected && <SelectionMark(?: className="[^"]*")? \/>/,
      `${f} must render it when selected`);
  }
});

test('nothing is laid over the image to say it is selected', () => {
  // The Bank tile used `absolute inset-0 bg-indigo-500/30`: a 30 % accent film
  // over the photo. Judging a warm shot through an orange filter is exactly
  // what the neutral chrome exists to prevent.
  for (const f of GRIDS) {
    const src = read(f);
    assert.doesNotMatch(src, /inset-0[^"]*bg-(indigo|amber|primary)/,
      `${f} tints the picture to mark selection — mark the frame instead`);
  }
});

test('a selected tile shows it, whatever badges it carries', () => {
  // The mark sat top-left, under the state badges: the ✓ of a kept Bank image
  // covered it. Its frame was an OUTER ring inside a tile that clips, so it was
  // cut off. A kept image looked the same selected or not.
  assert.match(read('components/shared/SelectionMark.jsx'),
    /className = 'absolute bottom-1 left-1'/, 'the mark keeps the corner no badge uses');
  const tile = read('components/bank/BankTile.jsx');
  assert.match(tile, /<li [^>]*overflow-hidden/, 'the Bank tile clips what leaves it…');
  assert.match(tile, /absolute inset-0 [^"]*ring-2 ring-inset ring-primary/,
    '…so its frame is drawn inside');
  // One flow for the bottom row: a button pinned at a hand-computed offset runs
  // under its neighbour as soon as a glyph is wider than the offset assumed.
  assert.doesNotMatch(tile, /absolute bottom-1 (?:left|right)-/);
});

test('the mark carries its own contrast, so it survives any picture', () => {
  const mark = read('components/shared/SelectionMark.jsx');
  assert.match(mark, /bg-app\//, 'the stroke needs a disc behind it to read on a light image');
  assert.match(mark, /stroke="rgb\(var\(--accent\)\)"/,
    'the stroke reads the accent token, so it follows the theme');
  assert.match(mark, /aria-hidden="true"/,
    'the mark is decorative — selection is already announced by the control');
});
