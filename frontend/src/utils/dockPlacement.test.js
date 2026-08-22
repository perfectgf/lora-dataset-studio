import test from 'node:test';
import assert from 'node:assert/strict';
import { ROUTES_WITH_BOTTOM_BAR, dockBottomClass } from './dockPlacement.js';

test('the dock clears the Test Studio bar, on both of its routes', () => {
  // StudioActionBar is fixed bottom-0 z-[9960] with an opaque background: at
  // bottom-4 the dock is not just overlapped, it is unclickable.
  assert.equal(dockBottomClass('/studio'), 'bottom-20');
  assert.equal(dockBottomClass('/dataset/studio/12'), 'bottom-20');
});

test('every route with a bottom bar is covered, including its children', () => {
  for (const route of ROUTES_WITH_BOTTOM_BAR) {
    assert.equal(dockBottomClass(route), 'bottom-20', route);
    assert.equal(dockBottomClass(`${route}/7`), 'bottom-20', `${route}/7`);
  }
});

test('the dock clears the Settings save bar, on every section', () => {
  // "Unsaved changes / Save changes" is fixed inset-x-0 bottom-4 z-40 — same
  // band, same z, and the dock paints after it. Unconditional on purpose: the
  // bar comes and goes with the form's dirty state, and a dock that jumped
  // when you edited a field would be worse than one sitting a rem higher.
  assert.equal(dockBottomClass('/settings'), 'bottom-20');
  assert.equal(dockBottomClass('/settings/engines'), 'bottom-20');
});

test('an ordinary screen keeps the dock in the corner', () => {
  for (const path of ['/datasets', '/bank', '/canvas', '/', ''])
    assert.equal(dockBottomClass(path), 'bottom-4', path);
});

// A route that merely STARTS with the same letters is not the same screen.
test('a lookalike route is not treated as covered', () => {
  assert.equal(dockBottomClass('/studios'), 'bottom-4');
  assert.equal(dockBottomClass('/studio-export'), 'bottom-4');
});

test('a missing or odd pathname degrades to the default corner', () => {
  for (const path of [null, undefined, 0, {}])
    assert.equal(dockBottomClass(path), 'bottom-4');
});
