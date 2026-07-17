import test from 'node:test';
import assert from 'node:assert/strict';
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

test('seed waves are all present', () => {
  const ids = new Set(WHATS_NEW.map((e) => e.id));
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
    { path: '/settings/engines', section: null, panel: null });
  assert.deepEqual(parseTarget('/datasets?section=curation&panel=watermarks'),
    { path: '/datasets', section: 'curation', panel: 'watermarks' });
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
    '/datasets', '/studio', '/cloud', '/guide', '/help', '/setup',
    '/settings/engines', '/settings/maintenance', '/guide/using-the-app',
    '/datasets?section=scrape&panel=scan', '/datasets?section=add',
  ]) {
    assert.equal(isValidTarget(ok), true, ok);
  }
  for (const bad of [
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
