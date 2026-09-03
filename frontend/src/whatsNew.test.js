import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import {
  WHATS_NEW,
  sortedEntries,
  latestEntryId,
  unseenEntries,
  unseenCount,
  hasUnseen,
  readSeenId,
  markAllSeen,
  isValidTarget,
  parseTarget,
  WHATS_NEW_SEEN_KEY,
} from './whatsNew.js';
import { SETTINGS_SECTIONS } from './components/settings/registry.js';
import { WORKSPACE_SECTIONS } from './components/dataset/workspaceSections.js';

// Minimal localStorage stand-in for the marker helpers.
function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
    _map: map,
  };
}

// ── Content integrity ────────────────────────────────────────────────────────

test('every entry has the required shape and a unique, stable id', () => {
  assert.ok(WHATS_NEW.length > 0, 'feed is not empty');
  const seen = new Set();
  for (const e of WHATS_NEW) {
    assert.equal(typeof e.id, 'string');
    assert.match(e.id, /^\d{4}-\d{2}-\d{2}-[a-z0-9-]+$/, `id shape: ${e.id}`);
    assert.match(e.date, /^\d{4}-\d{2}-\d{2}$/, `date shape: ${e.id}`);
    assert.ok(e.title && typeof e.title === 'string', `title: ${e.id}`);
    assert.ok(e.blurb && typeof e.blurb === 'string', `blurb: ${e.id}`);
    assert.ok(!seen.has(e.id), `duplicate id: ${e.id}`);
    seen.add(e.id);
  }
});

// A screenshot only helps if it is THERE. The release body links it by an
// absolute URL pinned to the tag, so a wrong path does not fail loudly at build
// time — it ships a release page with a broken image on it, which is worse than
// no picture. Catch it here, where a rename is being made.
test('every declared screenshot exists, and is an image', () => {
  const root = new URL('../../', import.meta.url);
  let declared = 0;
  for (const e of WHATS_NEW) {
    if (e.image === undefined) continue;
    declared += 1;
    assert.equal(typeof e.image, 'string', `image must be a path: ${e.id}`);
    assert.ok(!e.image.startsWith('/'), `image path is repo-relative: ${e.id}`);
    assert.match(e.image, /\.(png|jpg|jpeg|gif|webp)$/i, `image extension: ${e.id}`);
    assert.ok(existsSync(new URL(e.image, root)),
      `screenshot missing from the repo: ${e.image} (${e.id})`);
  }
  // Not an assertion on the count — entries without a picture are normal, and
  // the field is optional on purpose.
  assert.ok(declared >= 0);
});

// Rule 2 of the doctrine in whatsNew.js, enforced rather than trusted: the
// maintainer's own images are NSFW and out of bounds for anything public. Only
// the curated, generated showcase set is publishable, and it lives in one place.
test('screenshots come from the tracked showcase folder, never from anywhere else', () => {
  for (const e of WHATS_NEW) {
    if (e.image === undefined) continue;
    assert.match(e.image, /^docs\/screenshots\//,
      `a public screenshot must live under docs/screenshots/: ${e.image} (${e.id})`);
  }
});

test('seed waves are all present', async () => {
  // The founding entries now live in the ARCHIVE (whatsNewArchive.js) — this
  // guard is about deletion, not location: an id must exist in the union.
  const { WHATS_NEW_ARCHIVE } = await import('./whatsNewArchive.js');
  const ids = new Set([...WHATS_NEW, ...WHATS_NEW_ARCHIVE].map((e) => e.id));
  for (const id of [
    '2026-07-17-watermark-engine',
    '2026-07-17-scrape-section',
    '2026-07-17-generation-lora-presets',
    '2026-07-17-prompt-suffixes',
    '2026-07-17-targeted-recaption',
    '2026-07-17-library-taxonomy',
    '2026-07-17-studio-lightbox-nav',
    '2026-07-17-slider-lora-cloud',
    '2026-07-17-pillow-self-heal',
  ]) {
    assert.ok(ids.has(id), `missing seed entry: ${id}`);
  }
});

// ── Ordering ─────────────────────────────────────────────────────────────────

test('sortedEntries is newest-first and stable by (date desc, id desc)', () => {
  const messy = [
    { id: '2026-01-01-a', date: '2026-01-01', title: 't', blurb: 'b' },
    { id: '2026-03-05-z', date: '2026-03-05', title: 't', blurb: 'b' },
    { id: '2026-03-05-a', date: '2026-03-05', title: 't', blurb: 'b' },
    { id: '2026-02-10-m', date: '2026-02-10', title: 't', blurb: 'b' },
  ];
  assert.deepEqual(
    sortedEntries(messy).map((e) => e.id),
    ['2026-03-05-z', '2026-03-05-a', '2026-02-10-m', '2026-01-01-a'],
  );
});

test('the entry at the top of the file is the one the feed shows first', () => {
  // The feed sorts by date, then id: two entries of one day written with ids
  // sorting BELOW that day's others landed in fifth and sixth place, and the
  // badge — which counts entries above the last seen id — lit for nobody
  // (found in verification, 2026-09-03). The newest entry is the first in
  // this file by convention; its id has to agree.
  assert.equal(sortedEntries(WHATS_NEW)[0].id, WHATS_NEW[0].id,
    'the newest entry needs an id that sorts above every other id of its day')
})

test('latestEntryId returns the newest id, null on empty', () => {
  assert.equal(latestEntryId(WHATS_NEW), sortedEntries(WHATS_NEW)[0].id);
  assert.equal(latestEntryId([]), null);
});

// ── Unseen logic (badge) ─────────────────────────────────────────────────────

const SAMPLE = [
  { id: '2026-07-03-c', date: '2026-07-03', title: 't', blurb: 'b' },
  { id: '2026-07-02-b', date: '2026-07-02', title: 't', blurb: 'b' },
  { id: '2026-07-01-a', date: '2026-07-01', title: 't', blurb: 'b' },
];

test('first visit (no marker) treats every entry as unseen', () => {
  assert.equal(unseenCount(null, SAMPLE), 3);
  assert.equal(hasUnseen(null, SAMPLE), true);
  assert.deepEqual(unseenEntries(null, SAMPLE).map((e) => e.id),
    ['2026-07-03-c', '2026-07-02-b', '2026-07-01-a']);
});

test('having seen the latest id clears the badge', () => {
  assert.equal(unseenCount('2026-07-03-c', SAMPLE), 0);
  assert.equal(hasUnseen('2026-07-03-c', SAMPLE), false);
});

test('an older marker leaves only the strictly newer entries unseen', () => {
  assert.deepEqual(unseenEntries('2026-07-01-a', SAMPLE).map((e) => e.id),
    ['2026-07-03-c', '2026-07-02-b']);
  assert.equal(unseenCount('2026-07-02-b', SAMPLE), 1);
});

test('an unknown/pruned marker over-notifies rather than hides new work', () => {
  assert.equal(unseenCount('2019-01-01-gone', SAMPLE), 3);
});

// ── localStorage marker ──────────────────────────────────────────────────────

test('readSeenId round-trips through storage and defaults to null', () => {
  assert.equal(readSeenId(fakeStorage()), null);
  assert.equal(readSeenId(fakeStorage({ [WHATS_NEW_SEEN_KEY]: 'x' })), 'x');
});

test('markAllSeen pins the newest id and then hasUnseen is false', () => {
  const store = fakeStorage();
  const id = markAllSeen(store, WHATS_NEW);
  assert.equal(id, latestEntryId(WHATS_NEW));
  assert.equal(store.getItem(WHATS_NEW_SEEN_KEY), id);
  assert.equal(hasUnseen(readSeenId(store), WHATS_NEW), false);
});

test('markAllSeen degrades gracefully when storage throws', () => {
  const throwing = {
    getItem: () => { throw new Error('denied'); },
    setItem: () => { throw new Error('denied'); },
  };
  assert.doesNotThrow(() => markAllSeen(throwing, WHATS_NEW));
  assert.equal(readSeenId(throwing), null);
});

test('opening the panel after a partial read clears the whole badge', () => {
  // User had seen the middle entry, opens the panel → newest pinned → 0 unseen.
  const store = fakeStorage({ [WHATS_NEW_SEEN_KEY]: '2026-07-02-b' });
  assert.equal(unseenCount(readSeenId(store), SAMPLE), 1);
  markAllSeen(store, SAMPLE);
  assert.equal(unseenCount(readSeenId(store), SAMPLE), 0);
});

// ── Navigation targets ("Try it →") ──────────────────────────────────────────

test('parseTarget splits path and workspace query params', () => {
  assert.deepEqual(parseTarget('/settings/engines'),
    { path: '/settings/engines', section: null, panel: null, step: null });
  assert.deepEqual(parseTarget('/datasets?section=curation&panel=watermarks'),
    { path: '/datasets', section: 'curation', panel: 'watermarks', step: null });
  assert.deepEqual(parseTarget('/setup?step=quality'),
    { path: '/setup', section: null, panel: null, step: 'quality' });
  assert.equal(parseTarget('https://example.com'), null);
  assert.equal(parseTarget(undefined), null);
});

test('every seed entry target is a valid, navigable in-app route', () => {
  for (const e of WHATS_NEW) {
    if (e.to === undefined) continue; // optional — reliability entries omit it
    assert.equal(isValidTarget(e.to), true, `${e.id} → ${e.to}`);
  }
});

test('seed section/panel targets resolve against the LIVE navigation registries', () => {
  const settingsIds = new Set(SETTINGS_SECTIONS.map((s) => s.id));
  for (const e of WHATS_NEW) {
    const t = parseTarget(e.to);
    if (!t) continue;
    if (t.path.startsWith('/settings/')) {
      assert.ok(settingsIds.has(t.path.slice('/settings/'.length)),
        `settings section exists: ${e.id}`);
    }
    if (t.path === '/datasets' && t.section) {
      const ws = WORKSPACE_SECTIONS.find((s) => s.id === t.section);
      assert.ok(ws, `workspace section exists: ${e.id} (${t.section})`);
      if (t.panel) {
        assert.ok(ws.panels.some((p) => p.id === t.panel),
          `workspace panel exists: ${e.id} (${t.section}/${t.panel})`);
      }
    }
  }
});

test('isValidTarget accepts good routes and rejects malformed ones', () => {
  for (const ok of [
    '/datasets', '/studio', '/cloud', '/gallery', '/guide', '/help', '/setup',
    '/settings/engines', '/settings/maintenance', '/guide/using-the-app',
    '/datasets?section=scrape&panel=scan', '/datasets?section=add',
    '/setup?step=quality', '/setup?step=comfyui',
  ]) {
    assert.equal(isValidTarget(ok), true, ok);
  }
  for (const bad of [
    '/setup?step=nope',
    '/settings/engines?step=quality',
    '/settings/does-not-exist',
    '/datasets?section=nope',
    '/datasets?section=curation&panel=nope',
    '/settings/engines?section=x',
    '/datasets?panel=scan',
    'studio',
    'https://example.com',
    '',
    null,
  ]) {
    assert.equal(isValidTarget(bad), false, String(bad));
  }
});

// A merge that folds two entries into ONE object is valid JavaScript: the
// duplicate keys just overwrite each other, JS keeps the last, and the earlier
// entries disappear from WHATS_NEW without a sound. Every other test here still
// passes -- the surviving object has a perfect shape and a unique id. Four
// entries were lost that way while resolving cherry-pick conflicts on this file,
// and nothing caught it. Counting the id: lines in the SOURCE and comparing with
// the parsed array is what makes that class of loss impossible to miss.
test('no entry is swallowed by a merge (source id: lines == parsed entries)', async () => {
  const { readFileSync } = await import('node:fs');
  const { fileURLToPath } = await import('node:url');
  const src = readFileSync(fileURLToPath(new URL('./whatsNew.js', import.meta.url)), 'utf8');
  const body = src.slice(src.indexOf('export const WHATS_NEW = ['));
  const idLines = (body.match(/^ {4}id: '/gm) || []).length;
  assert.equal(idLines, WHATS_NEW.length,
    `${idLines} "id:" lines in the source but ${WHATS_NEW.length} entries parsed `
    + '- an entry was folded into its neighbour (duplicate keys in one object)');
});
