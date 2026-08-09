/**
 * The board's three new controls, in the layer where they can be tested without
 * a browser: 💾 layout presets, 📷 PNG export, 🗑 delete a pinned picture.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  boardExportBox, boardExportFilename, boardExportPlan, boardExportRefusal,
  boardExportScale, exportCardLines, EXPORT_MAX_PIXELS, EXPORT_MAX_SIDE, EXPORT_PADDING,
} from './canvasExportPng.js';
import {
  canvasLayoutIsEmpty, canvasLayoutSnapshot, canvasPresetApplied,
  canvasPresetName, canvasPresetSummary, PRESET_NAME_MAX,
} from './canvasLayoutPresets.js';
import {
  canvasDeleteButtonState, canvasImageDeleteTarget,
} from './canvasImageDelete.js';

/* ---------------------------------------------------------------- 📷 export */

test('the export box carries the board’s NEGATIVE origin', () => {
  // A picture dragged above its lane gives the world a negative corner. Cutting
  // the export at zero would drop exactly the picture free placement exists for.
  const box = boardExportBox({ x: -300, y: -120, width: 1000, height: 800 });
  assert.equal(box.x, -300 - EXPORT_PADDING);
  assert.equal(box.y, -120 - EXPORT_PADDING);
  assert.equal(box.width, 1000 + EXPORT_PADDING * 2);
  assert.equal(box.height, 800 + EXPORT_PADDING * 2);
});

test('a small board is exported at full pixel ratio', () => {
  const plan = boardExportPlan({ x: 0, y: 0, width: 1200, height: 900 });
  assert.equal(plan.scale, 2);
  assert.equal(plan.width, Math.round((1200 + EXPORT_PADDING * 2) * 2));
});

test('a huge board is softened, never refused — and stays inside both browser caps', () => {
  const world = { x: 0, y: 0, width: 90000, height: 40000 };
  const plan = boardExportPlan(world);
  assert.ok(plan.scale < 2, 'the ratio must come down');
  assert.ok(plan.width * plan.height <= EXPORT_MAX_PIXELS * 1.001, 'area budget');
  assert.ok(plan.width <= EXPORT_MAX_SIDE && plan.height <= EXPORT_MAX_SIDE, 'side budget');
});

test('a single very wide board is capped by the SIDE, not only by the area', () => {
  // 60 000 × 400 is only 24 MP at ×1 — under the area budget — and still far
  // past the 16 384-px limit a browser silently answers with a blank canvas.
  const plan = boardExportPlan({ x: 0, y: 0, width: 60000, height: 400 });
  assert.ok(plan.width <= EXPORT_MAX_SIDE);
});

test('the scale never goes above what was asked for', () => {
  const box = boardExportBox({ x: 0, y: 0, width: 10, height: 10 });
  assert.equal(boardExportScale(box, { pixelRatio: 1 }), 1);
});

test('only an empty board is refused', () => {
  assert.match(boardExportRefusal({ lanes: [], width: 0, height: 0 }), /nothing on the board/i);
  assert.match(boardExportRefusal({ lanes: [{}], width: 0, height: 0 }), /no size yet/i);
  assert.equal(boardExportRefusal({ lanes: [{}], width: 800, height: 600 }), null);
});

test('the file name says when and how big, so a Downloads folder stays usable', () => {
  const name = boardExportFilename(new Date(2026, 7, 8, 9, 5), 3);
  assert.equal(name, 'lora-canvas-2026-08-08-0905-3-lanes.png');
  assert.match(boardExportFilename(new Date(2026, 7, 8, 9, 5), 1), /-1-lane\.png$/);
});

test('a card in the poster names its run and what it is', () => {
  assert.deepEqual(exportCardLines({ record_id: 42, steps: 3500, train_type: 'krea' }),
    { title: 'Run #42', subtitle: '3,500 steps · krea' });
  // A run with nothing to say gets a title and no second line, not "undefined".
  assert.deepEqual(exportCardLines({}), { title: 'Run', subtitle: '' });
});

/* --------------------------------------------------------------- 💾 presets */

const POSITIONS = { 7: { 100: { x: 10, y: 20 }, 101: { x: 30, y: 40 } } };
const NODES = {
  7: {
    500: { imageId: 500, x: 1, y: 2, w: 260, h: 260, visible: true, groupId: 'g500', groupPos: 0 },
    501: { imageId: 501, x: 5, y: 6, w: 200, h: 200, visible: false, groupId: null, groupPos: null },
  },
};

test('a snapshot is the board, cards and pictures, keyed by dataset', () => {
  const snap = canvasLayoutSnapshot({ positions: POSITIONS, imageNodes: NODES });
  assert.deepEqual(snap.positions['7'],
    [{ record_id: 100, x: 10, y: 20 }, { record_id: 101, x: 30, y: 40 }]);
  assert.equal(snap.images['7'].length, 2);
  assert.equal(snap.images['7'][0].group_id, 'g500');
});

test('a CLOSED picture is part of the layout, not dropped from it', () => {
  // Restoring a board that silently re-opened everything you had closed would
  // be putting a different board back.
  const snap = canvasLayoutSnapshot({ positions: {}, imageNodes: NODES });
  const closed = snap.images['7'].find((n) => n.image_id === 501);
  assert.equal(closed.visible, false);
  assert.equal(closed.w, 200);
});

test('a snapshot can be scoped to the lanes actually on screen', () => {
  const snap = canvasLayoutSnapshot({ positions: POSITIONS, imageNodes: NODES, datasetIds: [9] });
  assert.ok(canvasLayoutIsEmpty(snap));
});

test('unusable rows are dropped rather than sent', () => {
  const snap = canvasLayoutSnapshot({
    positions: { 7: { 100: { x: 'nope', y: 2 } } },
    imageNodes: { 7: { 1: { imageId: NaN, x: 0, y: 0, w: 1, h: 1 } } },
  });
  assert.ok(canvasLayoutIsEmpty(snap));
});

test('the picker says how big a preset is', () => {
  assert.equal(canvasPresetSummary({ lanes: 2, cards: 5, images: 1 }),
    '2 lanes · 5 cards · 1 picture');
  assert.equal(canvasPresetSummary({ lanes: 1, cards: 0, images: 0 }), '1 lane');
});

test('a partial restore SAYS what could not be put back', () => {
  const preset = { name: 'likeness', cards: 6, images: 4 };
  const full = canvasPresetApplied({ applied: { cards: 6, images: 4 } }, preset);
  assert.match(full, /restored/);
  assert.doesNotMatch(full, /no longer exist/);
  const partial = canvasPresetApplied({ applied: { cards: 4, images: 4 } }, preset);
  assert.match(partial, /2 runs no longer exist/);
});

test('a preset name is trimmed, capped, and an empty one is refused', () => {
  assert.equal(canvasPresetName('  likeness review  '), 'likeness review');
  assert.equal(canvasPresetName('   '), null);
  assert.equal(canvasPresetName('x'.repeat(200)).length, PRESET_NAME_MAX);
});

/* ---------------------------------------------------------------- 🗑 delete */

const pinned = (over = {}) => ({
  imageId: 900,
  image: { id: 900, record_id: 12, step: 2500, dataset_id: 7, ...over },
});

test('a pinned picture deletes through its CHECKPOINT’s own gallery route', () => {
  const t = canvasImageDeleteTarget(pinned());
  assert.deepEqual(t, {
    endpoint: '/api/train/checkpoint/12/2500/images/delete',
    imageId: 900,
    scope: 'checkpoint',
  });
});

test('a legacy picture with no step falls back to the RUN scope', () => {
  const t = canvasImageDeleteTarget(pinned({ step: null }));
  assert.equal(t.endpoint, '/api/train/run/12/images/delete');
  assert.equal(t.scope, 'run');
});

test('a picture that cannot be traced back to a run offers no delete at all', () => {
  assert.equal(canvasImageDeleteTarget(pinned({ record_id: null })), null);
  assert.equal(canvasImageDeleteTarget(null), null);
});

test('the bin arms before it deletes, and never claims the file is unrecoverable', () => {
  const idle = canvasDeleteButtonState({ label: 'step 2500' });
  assert.equal(idle.glyph, '🗑');
  assert.match(idle.title, /Press once to arm/);
  // Whether the file is recoverable is an install SETTING; promising either way
  // here would be a lie on half the installs.
  assert.doesNotMatch(idle.title, /permanent/i);
  const armed = canvasDeleteButtonState({ armed: true, label: 'step 2500' });
  assert.equal(armed.glyph, '🗑!');
  assert.match(armed.title, /Press again/);
  assert.equal(canvasDeleteButtonState({ busy: true }).disabled, true);
});
