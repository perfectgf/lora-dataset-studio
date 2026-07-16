import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(
  new URL('./ConceptSourcesPanel.jsx', import.meta.url), 'utf8');

test('scan state follows the effective page returned by the backend', () => {
  assert.match(source, /const responsePage = body\.page;/);
  assert.match(source, /const isFreshScan = responsePage === 0;/);
  assert.match(source, /setPage\(responsePage\);/);
  assert.doesNotMatch(source, /setPage\(nextPage\);/);
});

test('appended scan pages are deduplicated by image URL', () => {
  assert.match(source, /const seenUrls = new Set\(prev\.map\(\(it\) => it\.url\)\);/);
  assert.match(source, /seenUrls\.has\(it\.url\)/);
  assert.match(source, /seenUrls\.add\(it\.url\);/);
  assert.match(source, /return \[\.\.\.prev, \.\.\.additions\];/);
});
