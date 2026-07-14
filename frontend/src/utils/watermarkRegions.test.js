import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MIN_WATERMARK_REGION_SIZE,
  buildWatermarkReviewState,
  clamp,
  deleteSelectedWatermarkRegion,
  ensureMinimumRegion,
  moveRegion,
  normalizeRegion,
  pointToNormalized,
  regionFromPoints,
  removeRegion,
  replaceRegion,
  resizeRegion,
  serializeWatermarkRegions,
} from './watermarkRegions.js';

function assertRegionClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) < 1e-12,
      `coordinate ${index}: expected ${expected[index]}, got ${value}`,
    );
  });
}

test('clamp and normalizeRegion keep ordered coordinates inside the image', () => {
  assert.equal(clamp(-0.25), 0);
  assert.equal(clamp(0.4), 0.4);
  assert.equal(clamp(1.25), 1);
  assert.deepEqual(normalizeRegion([1.2, 0.9, -0.2, 0.1]), [0, 0.1, 1, 0.9]);
});

test('pointToNormalized converts rendered pixels and clamps outside points', () => {
  const rect = { left: 40, top: 10, width: 200, height: 100 };

  assert.deepEqual(pointToNormalized({ clientX: 140, clientY: 90 }, rect), [0.5, 0.8]);
  assert.deepEqual(pointToNormalized({ clientX: 20, clientY: 130 }, rect), [0, 1]);
});

test('moveRegion preserves size while constraining the box within bounds', () => {
  const region = [0.8, 0.7, 0.95, 0.9];

  assertRegionClose(moveRegion(region, 0.2, 0.3), [0.85, 0.8, 1, 1]);
  assertRegionClose(moveRegion(region, -1, -1), [0, 0, 0.15, 0.2]);
});

test('resizeRegion moves the northwest corner and anchors southeast', () => {
  assertRegionClose(
    resizeRegion([0.2, 0.3, 0.8, 0.9], 'nw', 0.1, 0.1),
    [0.3, 0.4, 0.8, 0.9],
  );
});

test('resizeRegion moves the northeast corner and anchors southwest', () => {
  assertRegionClose(
    resizeRegion([0.2, 0.3, 0.8, 0.9], 'ne', 0.1, 0.1),
    [0.2, 0.4, 0.9, 0.9],
  );
});

test('resizeRegion moves the southwest corner and anchors northeast', () => {
  assertRegionClose(
    resizeRegion([0.2, 0.3, 0.8, 0.9], 'sw', 0.1, 0.1),
    [0.3, 0.3, 0.8, 1],
  );
});

test('resizeRegion moves the southeast corner and anchors northwest', () => {
  assertRegionClose(
    resizeRegion([0.2, 0.3, 0.8, 0.9], 'se', 0.1, 0.1),
    [0.2, 0.3, 0.9, 1],
  );
});

test('minimum-size enforcement works at an image edge and during resize', () => {
  assert.equal(MIN_WATERMARK_REGION_SIZE, 0.005);
  assertRegionClose(regionFromPoints([0.999, 0.999], [1, 1]), [0.995, 0.995, 1, 1]);
  assertRegionClose(
    resizeRegion([0.2, 0.2, 0.8, 0.8], 'nw', 0.7, 0.7),
    [0.795, 0.795, 0.8, 0.8],
  );
  const centered = ensureMinimumRegion([0.5, 0.5, 0.5001, 0.5002]);
  assert.ok(centered[2] - centered[0] >= MIN_WATERMARK_REGION_SIZE);
  assert.ok(centered[3] - centered[1] >= MIN_WATERMARK_REGION_SIZE);
});

test('resizeRegion keeps the binary span at the minimum after clamping a moved edge', () => {
  const region = [
    0.3685284450184554,
    0.046012317296117544,
    0.5553886129055172,
    0.46696452889591455,
  ];

  const resized = resizeRegion(region, 'sw', 1.5540611715987325, -1.3840727284550667);

  assert.ok(resized[2] - resized[0] >= MIN_WATERMARK_REGION_SIZE);
  assert.ok(resized[3] - resized[1] >= MIN_WATERMARK_REGION_SIZE);
});

test('geometry and list helpers never mutate their inputs', () => {
  const region = [0.2, 0.3, 0.8, 0.9];
  const regions = [region, [0.05, 0.05, 0.1, 0.1]];
  const snapshot = structuredClone(regions);

  const moved = moveRegion(region, 0, 0);
  const replaced = replaceRegion(regions, 0, moved);
  const removed = removeRegion(regions, 1);

  assert.deepEqual(regions, snapshot);
  assert.notStrictEqual(moved, region);
  assert.notStrictEqual(replaced, regions);
  assert.notStrictEqual(replaced[0], moved);
  assert.notStrictEqual(replaced[1], regions[1]);
  assert.notStrictEqual(removed, regions);
  assert.notStrictEqual(removed[0], region);
});

test('serializeWatermarkRegions rounds only a cloned API payload and preserves null', () => {
  const regions = [[0.123456, 0.234567, 0.765432, 0.876543]];
  const snapshot = structuredClone(regions);

  const payload = serializeWatermarkRegions(regions);

  assert.deepEqual(payload, [[0.1235, 0.2346, 0.7654, 0.8765]]);
  assert.deepEqual(regions, snapshot);
  assert.notStrictEqual(payload, regions);
  assert.notStrictEqual(payload[0], regions[0]);
  assert.equal(serializeWatermarkRegions(null), null);
});

test('buildWatermarkReviewState clones effective regions without mutating the frozen queue', () => {
  const queue = [
    {
      id: 7,
      watermark_bbox: [0.1, 0.1, 0.2, 0.2],
      watermark_regions: null,
      effective_watermark_regions: [[0.1, 0.1, 0.2, 0.2]],
    },
    {
      id: 8,
      watermark_bbox: [0.3, 0.3, 0.4, 0.4],
      watermark_regions: [],
      effective_watermark_regions: [],
    },
  ];
  const snapshot = structuredClone(queue);

  const state = buildWatermarkReviewState(queue);

  assert.deepEqual(queue, snapshot);
  assert.deepEqual(state.regionsById, {
    7: [[0.1, 0.1, 0.2, 0.2]],
    8: [],
  });
  assert.deepEqual(state.detectionRegionsById, {
    7: [[0.1, 0.1, 0.2, 0.2]],
    8: [[0.3, 0.3, 0.4, 0.4]],
  });
  assert.deepEqual(state.manualById, { 7: false, 8: true });
  assert.deepEqual(state.selectedById, { 7: 0, 8: null });
  assert.deepEqual(state.addModeById, { 7: false, 8: false });
  assert.deepEqual(state.saveStateById, {
    7: { status: 'saved', error: null },
    8: { status: 'saved', error: null },
  });

  state.regionsById[7][0][0] = 0.9;
  assert.deepEqual(queue, snapshot);
});

test('deleteSelectedWatermarkRegion removes only the selected zone and reclamps selection', () => {
  const regions = [
    [0.1, 0.1, 0.2, 0.2],
    [0.3, 0.3, 0.4, 0.4],
    [0.5, 0.5, 0.6, 0.6],
  ];
  const snapshot = structuredClone(regions);

  assert.deepEqual(deleteSelectedWatermarkRegion(regions, 1), {
    regions: [regions[0], regions[2]],
    selectedIndex: 1,
  });
  assert.deepEqual(deleteSelectedWatermarkRegion(regions, 2), {
    regions: [regions[0], regions[1]],
    selectedIndex: 1,
  });
  assert.deepEqual(deleteSelectedWatermarkRegion([regions[0]], 0), {
    regions: [],
    selectedIndex: null,
  });
  assert.deepEqual(deleteSelectedWatermarkRegion(regions, null), {
    regions,
    selectedIndex: null,
  });
  assert.deepEqual(regions, snapshot);
});
