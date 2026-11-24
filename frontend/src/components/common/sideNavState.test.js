import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  SIDE_NAV_COLLAPSED_KEY, applySideNavCollapsed, readSideNavCollapsed, writeSideNavCollapsed,
} from './sideNavState.js';

function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (key) => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => { map.set(key, String(value)); },
    removeItem: (key) => { map.delete(key); },
    has: (key) => map.has(key),
  };
}

test('the menu is open until it has been folded, and the choice round-trips', () => {
  const store = fakeStorage();
  assert.equal(readSideNavCollapsed(store), false);
  writeSideNavCollapsed(true, store);
  assert.equal(readSideNavCollapsed(store), true);
  writeSideNavCollapsed(false, store);
  assert.equal(readSideNavCollapsed(store), false);
  assert.equal(store.has(SIDE_NAV_COLLAPSED_KEY), false, 'opening the menu leaves nothing stored');
});

test('storage that throws reads as open and never breaks a toggle', () => {
  const throwing = {
    getItem: () => { throw new Error('denied'); },
    setItem: () => { throw new Error('denied'); },
    removeItem: () => { throw new Error('denied'); },
  };
  assert.equal(readSideNavCollapsed(throwing), false);
  assert.doesNotThrow(() => writeSideNavCollapsed(true, throwing));
  assert.doesNotThrow(() => writeSideNavCollapsed(false, throwing));
});

test('the choice is mirrored on the root element for the stylesheet', () => {
  const attributes = {};
  const root = { setAttribute: (name, value) => { attributes[name] = value; } };
  applySideNavCollapsed(true, root);
  assert.equal(attributes['data-side-nav'], 'collapsed');
  applySideNavCollapsed(false, root);
  assert.equal(attributes['data-side-nav'], 'open');
});

test('the stylesheet sizes the folded menu from that attribute, and the app applies it before mounting', () => {
  // Source text: no layout engine here. What is pinned is the wiring — one
  // variable, set from the attribute, read by the menu and by the shell.
  const css = readFileSync(new URL('../../index.css', import.meta.url), 'utf8');
  assert.match(css, /\[data-side-nav="collapsed"\]\s*\{\s*--app-sidebar-w:\s*3\.5rem;/);
  const app = readFileSync(new URL('../../App.jsx', import.meta.url), 'utf8');
  assert.match(app, /w-\[var\(--app-sidebar-w\)\]/);
  assert.match(app, /xl:pl-\[var\(--app-sidebar-w\)\]/);
  const main = readFileSync(new URL('../../main.jsx', import.meta.url), 'utf8');
  assert.ok(main.indexOf('applySideNavCollapsed(readSideNavCollapsed())') > -1);
  assert.ok(main.indexOf('applySideNavCollapsed(readSideNavCollapsed())') < main.indexOf('loadPlugins().then'),
    'the stored choice is applied before the app mounts');
});

test("a menu opened from the side column is drawn over the generation dock, and under the dialogs", () => {
  // Both float over the page at the bottom left. On one layer the dock, later in the document, hid the last
  // entry of Setup & settings (Plugins) for as long as something was generating.
  const layer = (file, pattern) => {
    const found = readFileSync(new URL(file, import.meta.url), 'utf8').match(pattern);
    assert.ok(found, `${file}: layer not found`);
    return Number(found[1]);
  };
  const menu = layer('../../App.jsx', /app-sidebar fixed inset-y-0 left-0 z-\[?(\d+)\]? /);
  const dock = layer('./GenerationQueueDock.jsx', /left-\[calc\(var\(--app-sidebar-w\)\+0\.75rem\)\] z-\[?(\d+)\]? /);
  const pill = layer('../../pages/SettingsPage.jsx', /fixed left-\[var\(--app-sidebar-w\)\] right-0 bottom-4 z-\[?(\d+)\]? /);
  assert.ok(menu > dock && menu > pill, `menu ${menu}, dock ${dock}, save pill ${pill}`);
  assert.ok(menu < 50, "dialogs and sheets start at z-50");
});
