import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LANE_MAX_H, LANE_MIN_H, LANE_REACH,
  clampLanePlacement, laneOverflows, mergeLanePlacement, moveLaneTo,
  resizeLaneHeight, toLanePlacementMap,
} from './canvasLanePlacement.js';
import { LANE_GAP, LANE_HEADER_H, stackLanes } from './canvasLayout.js';

// ---- clampLanePlacement ----------------------------------------------------

test('nothing usable is no placement at all, not a placement of zeros', () => {
  for (const value of [null, undefined, {}, 'x', 42, { x: NaN, y: NaN, h: NaN },
    { x: null, y: null, h: null }, { h: Infinity }]) {
    assert.equal(clampLanePlacement(value), null, `${JSON.stringify(value)} should be no placement`);
  }
});

test('a reserved height is clamped into a usable range', () => {
  assert.equal(clampLanePlacement({ h: 5 }).h, LANE_MIN_H);
  assert.equal(clampLanePlacement({ h: 1e9 }).h, LANE_MAX_H);
  assert.equal(clampLanePlacement({ h: 640 }).h, 640);
});

test('a position is clamped to the board reach, on both sides of zero', () => {
  assert.deepEqual(clampLanePlacement({ x: -1e9, y: 1e9 }), { x: -LANE_REACH, y: LANE_REACH });
  // Negative is LEGAL — a lane may be parked above and left of the origin, like
  // a pinned picture. Only the rail is enforced.
  assert.deepEqual(clampLanePlacement({ x: -200, y: -80 }), { x: -200, y: -80 });
});

test('half a position is no position', () => {
  // A lane with a y but no x would sit at the board's left edge for reasons
  // nobody could read off the row.
  assert.equal(clampLanePlacement({ y: 300 }), null);
  assert.equal(clampLanePlacement({ x: 300 }), null);
  assert.deepEqual(clampLanePlacement({ y: 300, h: 400 }), { h: 400 });
});

test('a height alone, and a position alone, are both complete placements', () => {
  assert.deepEqual(clampLanePlacement({ h: 400 }), { h: 400 });
  assert.deepEqual(clampLanePlacement({ x: 10, y: 20 }), { x: 10, y: 20 });
});

// ---- toLanePlacementMap ----------------------------------------------------

test('rows become a map keyed by dataset id, and empty rows are dropped', () => {
  const map = toLanePlacementMap([
    { dataset_id: 7, x: 100, y: 200, h: 500 },
    { dataset_id: 8, x: null, y: null, h: null },   // an "auto" row is no row
    { datasetId: 9, h: 300 },
    { x: 1, y: 2 },                                  // no id
  ]);
  assert.deepEqual(map, { 7: { x: 100, y: 200, h: 500 }, 9: { h: 300 } });
});

// ---- the gestures ----------------------------------------------------------

test('resizing starts from the height the lane HAD, and cannot trap the handle', () => {
  assert.equal(resizeLaneHeight(400, 120), 520);
  assert.equal(resizeLaneHeight(400, -1000), LANE_MIN_H);
  assert.equal(resizeLaneHeight(400, 1e9), LANE_MAX_H);
  // A drag on a lane whose height was never set still starts somewhere usable.
  assert.equal(resizeLaneHeight(undefined, 0), LANE_MIN_H);
});

test('moving is a delta from where the lane currently sits', () => {
  assert.deepEqual(moveLaneTo({ x: 0, y: 90 }, 40, -30), { x: 40, y: 60 });
  assert.deepEqual(moveLaneTo(null, 10, 10), { x: 10, y: 10 });
});

test('a gesture never forgets what the other one said', () => {
  // Move a lane, then resize it: the position must survive the resize.
  const moved = mergeLanePlacement(null, moveLaneTo({ x: 0, y: 0 }, 300, 120));
  const resized = mergeLanePlacement(moved, { h: resizeLaneHeight(200, 400) });
  assert.deepEqual(resized, { x: 300, y: 120, h: 600 });
  // …and the other way round.
  const sized = mergeLanePlacement(null, { h: 600 });
  assert.deepEqual(mergeLanePlacement(sized, { x: 5, y: 6 }), { x: 5, y: 6, h: 600 });
});

// ---- what the whole thing exists for --------------------------------------

test('a lane reserves what its content needs until somebody says otherwise', () => {
  const [lane] = stackLanes([{ datasetId: 1, width: 500, height: 200 }]).lanes;
  assert.equal(lane.reserved, 200);
  assert.equal(lane.placement, null);
});

test('the measured collision: a contact-sheet band lands on the next dataset', () => {
  // The lane draws a 📌 Pin all band down to y = 1100 (maxY) while its TREE is
  // 150 tall — which is all the stack used to advance by.
  const { lanes } = stackLanes([
    { datasetId: 48, width: 900, height: 150, maxY: 1100 },
    { datasetId: 18, width: 900, height: 150, maxY: 150 },
  ]);
  const [a, b] = lanes;
  const aReaches = a.graphY + a.contentH;
  assert.ok(aReaches > b.y, 'the band must be the thing that overlaps');
  assert.equal(aReaches - b.y, 894);   // the number this feature was opened on
});

test('a reserved height ends it, and moves the lanes below by exactly that much', () => {
  const { lanes } = stackLanes([
    { datasetId: 48, width: 900, height: 150, maxY: 1100, placement: { h: 1100 } },
    { datasetId: 18, width: 900, height: 150, maxY: 150 },
  ]);
  const [a, b] = lanes;
  assert.equal(a.reserved, 1100);
  // The band now ends ABOVE the next lane, with the board's usual air between
  // them — the collision the test above measures at 894 units is gone.
  assert.equal(b.y - (a.graphY + a.contentH), LANE_GAP);
  assert.equal(b.y, LANE_HEADER_H + 1100 + LANE_GAP);
  assert.equal(laneOverflows(a), false);
});

test('a lane whose room is too small still overflows — and is still framed', () => {
  const { lanes, height } = stackLanes([
    { datasetId: 48, width: 900, height: 150, maxY: 1100, placement: { h: 300 } },
  ]);
  const [a] = lanes;
  assert.equal(laneOverflows(a), true);
  // ✦ Fit frames the REACH, not the reservation: shrinking a lane must never
  // put its own pictures out of reach.
  assert.equal(height, LANE_HEADER_H + 1100);
});

test('a moved lane goes where it was put and moves NOBODY else', () => {
  const auto = stackLanes([
    { datasetId: 1, width: 400, height: 100 },
    { datasetId: 2, width: 400, height: 100 },
    { datasetId: 3, width: 400, height: 100 },
  ]).lanes;
  const moved = stackLanes([
    { datasetId: 1, width: 400, height: 100 },
    { datasetId: 2, width: 400, height: 100, placement: { x: 2000, y: -500 } },
    { datasetId: 3, width: 400, height: 100 },
  ]).lanes;
  assert.deepEqual([moved[1].x, moved[1].y], [2000, -500]);
  assert.equal(moved[1].graphY, -500 + LANE_HEADER_H);
  // Lanes 1 and 3 sit exactly where they sat before lane 2 was touched.
  assert.deepEqual([moved[0].x, moved[0].y], [auto[0].x, auto[0].y]);
  assert.deepEqual([moved[2].x, moved[2].y], [auto[2].x, auto[2].y]);
});

test('a lane parked above and left of the origin grows the board box', () => {
  const world = stackLanes([
    { datasetId: 1, width: 400, height: 100, placement: { x: -600, y: -400 } },
    { datasetId: 2, width: 400, height: 100 },
  ]);
  assert.equal(world.x, -600);
  assert.equal(world.y, -400 + LANE_HEADER_H);
});

test('a board that has never been arranged lays out exactly as it always did', () => {
  const entries = [
    { datasetId: 1, width: 400, height: 100 },
    { datasetId: 2, width: 900, height: 250, maxY: 250 },
  ];
  const { lanes, width, height } = stackLanes(entries);
  assert.deepEqual(lanes.map((l) => [l.x, l.y, l.graphY]), [
    [0, 0], [0, LANE_HEADER_H + 100 + LANE_GAP],
  ].map(([x, y]) => [x, y, y + LANE_HEADER_H]));
  assert.equal(width, 900);
  assert.equal(height, lanes[1].graphY + 250);
});
