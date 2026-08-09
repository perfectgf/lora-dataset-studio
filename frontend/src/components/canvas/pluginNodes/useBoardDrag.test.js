import assert from 'node:assert/strict';
import test from 'node:test';

import { renderToStaticMarkup, createElement } from '../../../../tests/support/mountJsx.mjs';
import { useBoardDrag } from './useBoardDrag.js';

// `useBoardDrag` is a hook — it needs a real render pass to call. A throwaway
// component captures the handlers it returns into a plain object so the test
// can drive them directly, without a DOM or an event system.
function Capture({ box }) {
  box.handlers = useBoardDrag(1, box.onMove, box.onCommit);
  return null;
}

function driveHandlers(onMove, onCommit) {
  const box = { onMove, onCommit };
  renderToStaticMarkup(createElement(Capture, { box }));
  return box.handlers;
}

const noopTarget = { closest: () => null };
const noopCapture = { setPointerCapture: () => {} };

test('a non-primary pointer button never arms a drag', () => {
  let moved = false;
  const { onPointerDown, onPointerMove } = driveHandlers(() => { moved = true; }, () => {});
  onPointerDown({ button: 2, target: noopTarget, currentTarget: noopCapture, pointerId: 1, clientX: 0, clientY: 0 });
  onPointerMove({ clientX: 10, clientY: 10 });
  assert.equal(moved, false, 'a right/middle-click press must never arm a drag');
});

test('a primary-button press on the header arms a drag that onPointerMove then feeds', () => {
  let seen = null;
  const { onPointerDown, onPointerMove } = driveHandlers((dx, dy) => { seen = { dx, dy }; }, () => {});
  onPointerDown({ button: 0, target: noopTarget, currentTarget: noopCapture, pointerId: 1, clientX: 0, clientY: 0 });
  onPointerMove({ clientX: 5, clientY: 7 });
  assert.deepEqual(seen, { dx: 5, dy: 7 });
});

test('a press on a button inside the header still never arms a drag', () => {
  let moved = false;
  const { onPointerDown, onPointerMove } = driveHandlers(() => { moved = true; }, () => {});
  const buttonTarget = { closest: () => ({}) };
  onPointerDown({ button: 0, target: buttonTarget, currentTarget: noopCapture, pointerId: 1, clientX: 0, clientY: 0 });
  onPointerMove({ clientX: 10, clientY: 10 });
  assert.equal(moved, false, 'the × button must close the node, not steal the drag');
});
