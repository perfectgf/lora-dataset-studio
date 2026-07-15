import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('./SmallImageRescueReview.jsx', import.meta.url), 'utf8');

test('Klein review anchor clears the sticky header on mobile and desktop', () => {
  const anchor = source.match(
    /<section id="ds-curation-small-image-rescue"[\s\S]*?aria-labelledby="small-image-rescue-title">/,
  )?.[0] || '';
  assert.match(anchor, /scroll-mt-20/);
  assert.match(anchor, /lg:scroll-mt-24/);
});
