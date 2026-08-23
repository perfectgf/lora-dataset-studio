/* The live feed / archive pairing (see whatsNew.js header, rule "Keep the
 * list tidy"). 520 shipped entries moved out of the entry bundle into their
 * own lazy chunk; these assertions keep the split honest:
 *  - an id lives in exactly ONE of the two files (the seen-marker keys on ids,
 *    and a duplicate would render twice in the panel);
 *  - the archive carries no `to:` (months-old in-app targets go stale, and
 *    whatsNew.test.js validates targets against LIVE registries — an archived
 *    target would either break that test or dodge it);
 *  - the archive stays import-free, so its chunk is the entries and nothing
 *    else, and it can never drag the registries into the lazy load;
 *  - the panel loads it lazily — a static import would put all 500+ entries
 *    right back into the entry bundle, which is the regression this split
 *    exists to prevent.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { WHATS_NEW } from './whatsNew.js';
import { WHATS_NEW_ARCHIVE } from './whatsNewArchive.js';

test('an id lives in exactly one of the two files', () => {
  const live = new Set(WHATS_NEW.map((e) => e.id));
  assert.equal(live.size, WHATS_NEW.length, 'duplicate id inside the live feed');
  for (const e of WHATS_NEW_ARCHIVE) {
    assert.ok(!live.has(e.id), `${e.id} is in BOTH the live feed and the archive`);
  }
  const all = new Set([...WHATS_NEW, ...WHATS_NEW_ARCHIVE].map((e) => e.id));
  assert.equal(all.size, WHATS_NEW.length + WHATS_NEW_ARCHIVE.length,
    'duplicate id somewhere across the two files');
});

test('every archived entry is complete, and none carries an in-app target', () => {
  for (const e of WHATS_NEW_ARCHIVE) {
    assert.ok(e.id && e.date && e.title && e.blurb, `incomplete archived entry: ${e.id}`);
    assert.equal(e.to, undefined,
      `${e.id} carries a to: target in the archive — targets are dropped on the way in`);
  }
});

test('the archive is import-free, so its lazy chunk is entries and nothing else', () => {
  const src = readFileSync(new URL('./whatsNewArchive.js', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /^import /m);
});

test('the panel loads the archive lazily — a static import would refill the entry bundle', () => {
  const panel = readFileSync(new URL('./components/common/WhatsNew.jsx', import.meta.url), 'utf8');
  assert.match(panel, /import\('\.\.\/\.\.\/whatsNewArchive\.js'\)/);
  assert.doesNotMatch(panel, /^import .*whatsNewArchive/m);
});
