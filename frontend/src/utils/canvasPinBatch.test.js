import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PIN_BATCH_MAX, batchTileSize, boardObstacles, pinBatchAnnouncement,
  groupPinnedBatchBySource, groupPinnedBatchTogether, pinBatchLabel, pinBatchPending,
  pinBatchPendingAcrossLanes, placeImageBatch, tidyGroupRows, tidyLaneReach,
  tidyLaneRows, laneStackEntries,
} from './canvasPinBatch.js';
import { layoutImageNodes, occupiedBox } from './canvasImageGroups.js';
import { CARD_W } from './lineageGraph.js';
import { stackLanes } from './canvasLayout.js';

/* 📌 Pin all — the one assertion that decides whether this feature is worth
   having: NOTHING may end up on top of anything else.

   Everything here works in BOARD units, never in screen pixels. The user's own
   board was at 24 % zoom when the feature was asked for, and a rule expressed in
   pixels would have passed there and failed at 100 %. */

// ---- a realistic board ----------------------------------------------------
// Four runs, two generations, several checkpoint pills each — the shape the
// automatic tree produces, with the fields placeImageBatch actually reads.

const pill = (step, x, y) => ({ step, x, y, w: 60, h: 20 });

const card = (recordId, x, y, steps) => ({
  node: { record_id: recordId },
  x,
  y,
  cellH: 64 + 8 + 20,
  checkpoints: steps.map((s, i) => pill(s, x + i * 66, y + 72)),
});

const GRAPH = {
  nodes: [
    card(106, 22, 22, [2500, 3000]),
    card(107, 22, 160, [2000]),
    card(114, 348, 22, [1500, 2000]),
    card(117, 348, 160, [4000]),
  ],
  edges: [],
  width: 700,
  height: 300,
};

const img = (id, recordId, step) => ({
  id, dataset_id: 3, record_id: recordId, step, url: `/img/${id}.png`,
});

const placed = (id, recordId, step) => ({
  imageId: id, x: id * 10, y: 500, w: 320, h: 320, image: img(id, recordId, step),
});

test('successive pins from the same checkpoint automatically form one strip', () => {
  const first = { imageId: 1, x: 10, y: 10, w: 320, h: 320, visible: true,
    groupId: null, groupPos: null, image: img(1, 106, 2500) };
  const result = groupPinnedBatchBySource({ nodes: [first], placed: [placed(2, 106, 2500)] });
  assert.equal(result.rows[0].groupId, result.rows[1].groupId);
  assert.deepEqual(result.rows.map((r) => r.groupPos), [0, 1]);
});

test('a later generation joins the homogeneous group for that checkpoint', () => {
  const nodes = [1, 2].map((id, pos) => ({ imageId: id, x: id * 10, y: 10,
    w: 320, h: 320, visible: true, groupId: 'g1', groupPos: pos,
    image: img(id, 106, 2500) }));
  const result = groupPinnedBatchBySource({ nodes, placed: [placed(3, 106, 2500)] });
  assert.ok(result.rows.every((r) => r.groupId === 'g1'));
  assert.deepEqual(result.rows.map((r) => r.groupPos), [0, 1, 2]);
});

test('different checkpoints and unknown sources never auto-group', () => {
  const result = groupPinnedBatchBySource({ placed: [
    placed(1, 106, 2500), placed(2, 106, 3000),
    { ...placed(3, null, null), image: { id: 3, url: '/img/3.png' } },
  ] });
  assert.ok(result.rows.every((r) => r.groupId == null));
});

test('legacy Pin all with no run id keeps one whole-gesture grid across checkpoints', () => {
  const result = groupPinnedBatchTogether({ placed: [
    placed(1, 82, 500), placed(2, 82, 1000), placed(3, 86, 6000),
  ] });
  assert.ok(result.rows[0].groupId);
  assert.ok(result.rows.every((r) => r.groupId === result.rows[0].groupId));
  assert.deepEqual(result.rows.map((r) => r.groupPos), [0, 1, 2]);
  assert.ok(result.undoRows.every((r) => r.visible === false && r.groupId == null));
});

test('separate Pin all gestures get separate groups', () => {
  const first = groupPinnedBatchTogether({ placed: [placed(1, 82, 500), placed(2, 82, 1000)] });
  const second = groupPinnedBatchTogether({
    nodes: first.rows,
    placed: [placed(3, 82, 1500), placed(4, 82, 2000)],
  });
  assert.notEqual(first.rows[0].groupId, second.rows[0].groupId);
});

test('a manual mixed-checkpoint group is never used as an automatic target', () => {
  const nodes = [
    { imageId: 1, x: 10, y: 10, w: 320, h: 320, visible: true,
      groupId: 'mixed', groupPos: 0, image: img(1, 106, 2500) },
    { imageId: 2, x: 20, y: 10, w: 320, h: 320, visible: true,
      groupId: 'mixed', groupPos: 1, image: img(2, 107, 2500) },
  ];
  const result = groupPinnedBatchBySource({ nodes,
    placed: [placed(3, 106, 2500), placed(4, 106, 2500)] });
  assert.equal(result.rows.find((r) => r.imageId === 1), undefined);
  assert.notEqual(result.rows.find((r) => r.imageId === 3).groupId, 'mixed');
});

test('undo restores existing membership and closes newly pinned images', () => {
  const nodes = [1, 2].map((id, pos) => ({ imageId: id, x: id * 10, y: 10,
    w: 320, h: 320, visible: true, groupId: 'g1', groupPos: pos,
    image: img(id, 106, 2500) }));
  const result = groupPinnedBatchBySource({ nodes, placed: [placed(3, 106, 2500)] });
  assert.deepEqual(result.undoRows.map((r) => [r.imageId, r.visible, r.groupId, r.groupPos]),
    [[1, true, 'g1', 0], [2, true, 'g1', 1], [3, false, null, null]]);
});

// The lot from the report: five images, four different runs.
const LOT = [
  img(501, 106, 2500),
  img(502, 107, 2000),
  img(503, 114, 2000),
  img(504, 114, 2000),
  img(505, 117, 4000),
];

const rectsOverlap = (a, b) => (
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
);

/** Every pair of rectangles on the board, checked. Not three chosen cases: the
 *  full cross-product, so a placement that happens to be lucky cannot pass. */
function assertNoOverlap(rects) {
  for (let i = 0; i < rects.length; i += 1) {
    for (let j = i + 1; j < rects.length; j += 1) {
      assert.ok(!rectsOverlap(rects[i], rects[j]),
        `${JSON.stringify(rects[i])} overlaps ${JSON.stringify(rects[j])}`);
    }
  }
}

// ---- THE assertion --------------------------------------------------------

test('THE assertion: a lot placed on a loaded board overlaps NOTHING', () => {
  // A board that is already busy: five images pinned by hand, all over it.
  const existing = [
    { imageId: 900, x: 700, y: 20, w: 320, h: 320, visible: true },
    { imageId: 901, x: 700, y: 400, w: 320, h: 320, visible: true },
    { imageId: 902, x: 60, y: 420, w: 260, h: 260, visible: true },
    { imageId: 903, x: 1100, y: 120, w: 400, h: 400, visible: true },
    { imageId: 904, x: 380, y: 500, w: 180, h: 180, visible: true },
  ];
  const res = placeImageBatch({ graph: GRAPH, existing, images: LOT });

  assert.equal(res.placed.length, 5, 'all five got a spot');
  assert.equal(res.skipped.length, 0);

  // Cards (card + its pill block), pinned images, and the new tiles: one flat
  // list, every pair checked.
  assertNoOverlap([
    ...boardObstacles(GRAPH, existing),
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

test('no overlap holds for a lot of thirty from a single checkpoint', () => {
  const many = Array.from({ length: 30 }, (_, i) => img(1000 + i, 114, 2000));
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: many });
  assert.equal(res.placed.length, 30);
  assertNoOverlap([
    ...boardObstacles(GRAPH, []),
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

test('no overlap holds when the batch is bigger than the board is wide', () => {
  // Every image from a different run, including runs that are NOT on the board.
  const wide = Array.from({ length: 18 }, (_, i) => img(2000 + i, 106 + i, 2500));
  const existing = [{ imageId: 1, x: 0, y: 900, w: 1400, h: 300, visible: true }];
  const res = placeImageBatch({ graph: GRAPH, existing, images: wide });
  assertNoOverlap([
    ...boardObstacles(GRAPH, existing),
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

// ---- near its own source --------------------------------------------------

test('each image lands under its OWN checkpoint, not all in one heap', () => {
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: LOT });
  const by = new Map(res.placed.map((p) => [p.imageId, p]));

  // Two images from the same checkpoint share a column…
  assert.equal(by.get(503).x, by.get(504).x, 'same source ⇒ same column');
  assert.notEqual(by.get(503).y, by.get(504).y, 'stacked, not superimposed');

  // …and four different sources produce four different columns.
  const columns = new Set(res.placed.map((p) => p.x));
  assert.equal(columns.size, 4, 'one column per source checkpoint');

  // A source on the left of the tree keeps its images on the left.
  assert.ok(by.get(501).x < by.get(503).x,
    'the run at depth 0 is left of the run at depth 1');
});

test('an image whose run is not on the board still gets a spot', () => {
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: [img(700, 999, 1)] });
  assert.equal(res.placed.length, 1);
  assert.ok(res.placed[0].x >= 0 && res.placed[0].y >= 0);
});

// ---- determinism ----------------------------------------------------------

test('the same board and the same lot give exactly the same positions', () => {
  const existing = [{ imageId: 900, x: 700, y: 20, w: 320, h: 320, visible: true }];
  const a = placeImageBatch({ graph: GRAPH, existing, images: LOT });
  // Same lot, listed in another order: the answer must not depend on it.
  const b = placeImageBatch({ graph: GRAPH, existing, images: [...LOT].reverse() });
  const key = (r) => r.placed.map((p) => `${p.imageId}@${p.x},${p.y},${p.w}`).sort().join('|');
  assert.equal(key(a), key(b));
});

// ---- already pinned -------------------------------------------------------

test('an image already ON the board is neither moved nor duplicated', () => {
  const laneMap = {
    503: { imageId: 503, x: 999, y: 999, w: 320, h: 320, visible: true },
  };
  const pending = pinBatchPending(LOT, laneMap);
  assert.deepEqual(pending.pending.map((i) => i.id), [501, 502, 504, 505]);
  assert.equal(pending.already, 1);

  const res = placeImageBatch({ graph: GRAPH, existing: [laneMap[503]], images: pending.pending });
  assert.ok(!res.placed.some((p) => p.imageId === 503), 'not re-placed');
  // And its own rectangle is still where the user left it.
  assertNoOverlap([
    { x: 999, y: 999, w: 320, h: 320 },
    ...res.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
  ]);
});

test('a CLOSED image comes back at its remembered spot when that spot is free', () => {
  const laneMap = {
    501: { imageId: 501, x: 1600, y: 40, w: 300, h: 300, visible: false },
  };
  const pending = pinBatchPending(LOT, laneMap);
  assert.equal(pending.pending.length, 5, 'a closed image is part of the lot again');
  const res = placeImageBatch({
    graph: GRAPH, existing: [], images: pending.pending, remembered: laneMap });
  const back = res.placed.find((p) => p.imageId === 501);
  assert.deepEqual({ x: back.x, y: back.y, w: back.w, h: back.h },
    { x: 1600, y: 40, w: 300, h: 300 });
});

test('a remembered spot that would collide is overruled — no overlap wins', () => {
  const laneMap = {
    501: { imageId: 501, x: 22, y: 22, w: 300, h: 300, visible: false },  // on top of card 106
  };
  const res = placeImageBatch({
    graph: GRAPH, existing: [], images: [img(501, 106, 2500)], remembered: laneMap });
  const back = res.placed[0];
  assert.notDeepEqual({ x: back.x, y: back.y }, { x: 22, y: 22 });
  assertNoOverlap([
    ...boardObstacles(GRAPH, []),
    { x: back.x, y: back.y, w: back.w, h: back.h },
  ]);
});

// ---- when there is no more room -------------------------------------------

test('a lot over the cap places the cap and SAYS what it refused', () => {
  const huge = Array.from({ length: PIN_BATCH_MAX + 7 }, (_, i) => img(3000 + i, 106, 2500));
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: huge });
  assert.equal(res.placed.length, PIN_BATCH_MAX);
  assert.equal(res.skipped.length, 7);
  assert.ok(res.skipped.every((s) => s.reason === 'over-cap'));

  const said = pinBatchAnnouncement(res);
  assert.match(said, new RegExp(`${PIN_BATCH_MAX} images pinned`));
  assert.match(said, /7 left out/);
  assert.match(said, /gallery/i, 'the refusal names the way to get the rest');
});

test('nothing pinned is announced as nothing, never as a success', () => {
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: [] });
  assert.deepEqual(res.placed, []);
  assert.match(pinBatchAnnouncement(res), /Nothing/i);
});

// ---- the tile size --------------------------------------------------------

test('a big lot lands as a contact sheet, a small one at full size', () => {
  assert.equal(batchTileSize(1), 320);
  assert.ok(batchTileSize(30) < batchTileSize(3), 'thirty images are thumbnails');
  assert.ok(batchTileSize(40) >= 96, 'never below the grabbable floor');
  // The chosen size is what the placed tiles actually carry.
  const many = Array.from({ length: 20 }, (_, i) => img(4000 + i, 106, 2500));
  const res = placeImageBatch({ graph: GRAPH, existing: [], images: many });
  assert.equal(res.size, batchTileSize(20));
  assert.ok(res.placed.every((p) => p.w === res.size && p.h === res.size));
});

// ---- the button's own words -----------------------------------------------

test('the button says how many it will put down', () => {
  assert.equal(pinBatchLabel(5), '📌 Pin all 5 to the board');
  assert.equal(pinBatchLabel(1), '📌 Pin this image to the board');
  assert.equal(pinBatchLabel(0), '');
});

// ---- the obstacles are the whole board ------------------------------------

// ---- the button's count, across the whole board ---------------------------

test('the count is what is NOT on the board yet, lane by lane', () => {
  const candidates = [
    { id: 501, datasetId: 3 }, { id: 502, datasetId: 3 },
    { id: 601, datasetId: 8 },
  ];
  const lanes = {
    3: { 501: { imageId: 501, visible: true }, 502: { imageId: 502, visible: false } },
    8: {},
  };
  const { pending, already } = pinBatchPendingAcrossLanes(candidates, lanes);
  assert.deepEqual(pending.map((p) => p.id), [502, 601], 'a CLOSED one still counts');
  assert.equal(already, 1);
  assert.equal(pinBatchLabel(pending.length), '📌 Pin all 2 to the board');
});

test('a lot entirely on the board already offers nothing', () => {
  const lanes = { 3: { 501: { visible: true } } };
  const { pending } = pinBatchPendingAcrossLanes([{ id: 501, datasetId: 3 }], lanes);
  assert.equal(pending.length, 0);
  assert.equal(pinBatchLabel(pending.length), '', 'no label ⇒ no button');
});

test('a run card and its pill block are ONE obstacle, pinned images are the rest', () => {
  const obstacles = boardObstacles(GRAPH, [
    { imageId: 900, x: 700, y: 20, w: 320, h: 320, visible: true },
    { imageId: 901, x: 0, y: 0, w: 320, h: 320, visible: false },
  ]);
  assert.equal(obstacles.length, 5, '4 cards + 1 VISIBLE image');
  const cardRect = obstacles.find((o) => o.x === 22 && o.y === 22);
  assert.equal(cardRect.w, CARD_W);
  assert.ok(cardRect.h >= 64 + 20, 'the pills under the card are inside the rect');
});

/* ── Two runs are two lots, and a grid never mixes LoRAs ────────────────────

   The boundaries a board built to COMPARE has to get right — each one added by
   a report of them being wrong:

     • a second generation fired at the same checkpoint had its pictures
       APPENDED to the first generation's strip, so two runs read as one lot.
       The only thing a strip could be keyed on was "same checkpoint", which is
       true of both runs. `run_id` — already in the database, now published by
       the gallery — is the thing that actually differs;
     • the order inside a strip came from `${record_id}:${step}` compared as
       TEXT, which puts step 1000 before step 500. An epoch axis sorted
       alphabetically is not an epoch axis;
     • a batch launched at SEVERAL LoRAs was pinned as one fused grid in which
       the per-LoRA batches were indistinguishable. The LoRA (`record_id`)
       joined the key: one grid per LoRA, per prompt, per launch — while the
       EPOCHS of one LoRA still share their comparison strip (one grid per
       epoch was tried and rejected: it dismantled that strip). */

const runImg = (id, recordId, step, runId, prompt = '') => ({
  id, dataset_id: 3, record_id: recordId, step, run_id: runId,
  prompt,
  url: `/img/${id}.png`,
});
const runPlaced = (id, recordId, step, runId, prompt = '') => ({
  imageId: id, x: id * 10, y: 500, w: 320, h: 320,
  image: runImg(id, recordId, step, runId, prompt),
});
// Pin one picture the way the gallery does, and fold the result back into the
// board — the sequence that produced the reported bug.
const pinOneByOne = (laneMap, image,
  box = { x: 0, y: 0, w: 320, h: 320 }, graph = null) => {
  const grouped = groupPinnedBatchBySource({
    nodes: Object.values(laneMap),
    placed: [{ imageId: image.id, ...box, image }],
    graph,
  });
  for (const r of grouped.rows) laneMap[r.imageId] = r;
  return laneMap;
};

test('two generations of the SAME checkpoint stay two strips', () => {
  const lane = {};
  // Run A, pinned one by one from the checkpoint gallery.
  pinOneByOne(lane, runImg(11, 7, 500, 'runA'));
  pinOneByOne(lane, runImg(12, 7, 500, 'runA'));
  // Run B — a LATER generation, same checkpoint, same dataset.
  pinOneByOne(lane, runImg(21, 7, 500, 'runB'));
  pinOneByOne(lane, runImg(22, 7, 500, 'runB'));

  const groupOf = (id) => lane[id].groupId;
  assert.ok(groupOf(11), 'run A formed a strip');
  assert.equal(groupOf(11), groupOf(12));
  assert.ok(groupOf(21), 'run B formed its own strip');
  assert.equal(groupOf(21), groupOf(22));
  assert.notEqual(groupOf(11), groupOf(21),
    'run B must NOT be concatenated onto run A');
});

test('Pin all makes one separate grid per prompt inside the SAME run', () => {
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(101, 82, 500, 'prompt-run', 'on a rooftop'),
    runPlaced(102, 82, 1000, 'prompt-run', 'on a rooftop'),
    runPlaced(103, 82, 1500, 'prompt-run', 'on a rooftop'),
    runPlaced(201, 82, 500, 'prompt-run', 'in a cafe'),
    runPlaced(202, 82, 1000, 'prompt-run', 'in a cafe'),
    runPlaced(203, 82, 1500, 'prompt-run', 'in a cafe'),
  ] });
  const groupOf = (id) => result.rows.find((r) => r.imageId === id).groupId;

  assert.ok(groupOf(101), 'the first prompt formed a real grid');
  assert.equal(groupOf(101), groupOf(102));
  assert.equal(groupOf(102), groupOf(103));
  assert.ok(groupOf(201), 'the second prompt formed a real grid');
  assert.equal(groupOf(201), groupOf(202));
  assert.equal(groupOf(202), groupOf(203));
  assert.notEqual(groupOf(101), groupOf(201),
    'two prompts of one run must never be fused into the same Canvas grid');
});

test('prompt whitespace is normalised without merging different prompt text', () => {
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(301, 82, 500, 'normalised-run', '  on   a\nrooftop  '),
    runPlaced(302, 82, 1000, 'normalised-run', 'on a rooftop'),
    runPlaced(401, 82, 500, 'normalised-run', 'on a rooftop at night'),
    runPlaced(402, 82, 1000, 'normalised-run', 'on a rooftop at night'),
  ] });
  const byId = Object.fromEntries(result.rows.map((row) => [row.imageId, row.groupId]));

  assert.ok(byId[301]);
  assert.equal(byId[301], byId[302], 'equivalent whitespace is one prompt grid');
  assert.equal(byId[401], byId[402]);
  assert.notEqual(byId[301], byId[401], 'meaningfully different prompts stay apart');
});

test('a batch fired at several LoRAs lands as one grid PER LoRA', () => {
  // The report this boundary comes from: a Canvas Compare batch over several
  // LoRAs, pinned, arrived as ONE grid where the batches were
  // indistinguishable. Same run, same prompt — the LoRA is the split, and the
  // epochs INSIDE each LoRA still share that LoRA's grid.
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(501, 82, 500, 'one-prompt', 'portrait in soft light'),
    runPlaced(502, 82, 1000, 'one-prompt', 'portrait in soft light'),
    runPlaced(503, 86, 1000, 'one-prompt', 'portrait in soft light'),
    runPlaced(504, 86, 2000, 'one-prompt', 'portrait in soft light'),
  ] });
  const byId = Object.fromEntries(result.rows.map((row) => [row.imageId, row.groupId]));
  assert.ok(byId[501], 'the first LoRA formed its own grid');
  assert.equal(byId[501], byId[502], 'two epochs of one LoRA share its grid');
  assert.ok(byId[503], 'the second LoRA formed its own grid');
  assert.equal(byId[503], byId[504]);
  assert.notEqual(byId[501], byId[503],
    'two LoRAs of one launch must never be fused into the same grid');
});

test('the epochs of ONE LoRA in a mono-prompt run are still one grid', () => {
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(521, 82, 500, 'one-lora', 'portrait in soft light'),
    runPlaced(522, 82, 1000, 'one-lora', 'portrait in soft light'),
    runPlaced(523, 82, 1500, 'one-lora', 'portrait in soft light'),
  ] });
  assert.ok(result.rows[0].groupId);
  assert.equal(new Set(result.rows.map((row) => row.groupId)).size, 1,
    'the epoch-comparison strip survives the per-LoRA boundary');
});

test('one image per LoRA forms no grid — the singles stay loose in their columns', () => {
  // A grid of one is not a grid: each picture stays a loose tile, already
  // parked in its own checkpoint's band column by placeImageBatch.
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(511, 82, 500, 'one-each', 'portrait in soft light'),
    runPlaced(512, 86, 1000, 'one-each', 'portrait in soft light'),
    runPlaced(513, 91, 1500, 'one-each', 'portrait in soft light'),
  ] });
  assert.ok(result.rows.every((row) => row.groupId == null));
});

test('gallery pins use the same run-prompt-LoRA boundary as Pin all', () => {
  const lane = {};
  pinOneByOne(lane, runImg(601, 7, 500, 'gallery-run', 'on a rooftop'));
  pinOneByOne(lane, runImg(701, 7, 500, 'gallery-run', 'in a cafe'));
  pinOneByOne(lane, runImg(602, 7, 1000, 'gallery-run', 'on a rooftop'));
  pinOneByOne(lane, runImg(702, 7, 1000, 'gallery-run', 'in a cafe'));

  assert.ok(lane[601].groupId);
  assert.equal(lane[601].groupId, lane[602].groupId);
  assert.ok(lane[701].groupId);
  assert.equal(lane[701].groupId, lane[702].groupId);
  assert.notEqual(lane[601].groupId, lane[701].groupId,
    'pinning prompts one by one must not recombine them');
});

test('interleaved gallery pins reflow their automatic prompt grids without overlap', () => {
  const lane = {};
  const a = 'on a rooftop';
  const b = 'in a cafe';
  // Each first picture is individually valid. Once A gets its second member,
  // its derived 640-wide strip reaches across B's node; the gallery path must
  // reflow that new automatic group just like Pin all does.
  pinOneByOne(lane, runImg(711, 7, 500, 'interleaved-run', a),
    { x: 0, y: 500, w: 320, h: 320 });
  pinOneByOne(lane, runImg(721, 7, 500, 'interleaved-run', b),
    { x: 344, y: 500, w: 320, h: 320 });
  pinOneByOne(lane, runImg(712, 7, 1000, 'interleaved-run', a),
    { x: 0, y: 900, w: 320, h: 320 });
  pinOneByOne(lane, runImg(722, 7, 1000, 'interleaved-run', b),
    { x: 344, y: 900, w: 320, h: 320 });

  const layout = layoutImageNodes(Object.values(lane));
  const groups = layout.filter((row) => row.kind === 'group');
  assert.equal(groups.length, 2);
  assert.deepEqual(new Set(groups.map((group) => group.members[0].node.image.prompt)),
    new Set([a, b]));
  assertNoOverlap(layout.map(occupiedBox));
});

test('gallery reflow moves an automatic grid around a manual group, never the manual group', () => {
  const manual = [
    { ...runPlaced(731, 7, 500, 'manual-a', 'manual left'), x: 0, y: 500,
      visible: true, groupId: 'hand-made', groupPos: 0 },
    { ...runPlaced(732, 7, 1000, 'manual-b', 'manual right'), x: 900, y: 900,
      visible: true, groupId: 'hand-made', groupPos: 1 },
  ];
  const autoAnchor = { ...runPlaced(741, 7, 500, 'automatic', 'new prompt'),
    x: 344, y: 500, visible: true, groupId: null, groupPos: null };
  const nodes = [...manual, autoAnchor];
  const result = groupPinnedBatchBySource({
    nodes,
    placed: [{ ...runPlaced(742, 7, 1000, 'automatic', 'new prompt'),
      x: 344, y: 900 }],
  });
  const updates = new Map(result.rows.map((row) => [row.imageId, row]));
  const after = nodes.map((node) => updates.get(node.imageId) || node);
  after.push(...result.rows.filter((row) => !nodes.some((node) => node.imageId === row.imageId)));

  assert.deepEqual(manual.map((node) => [node.x, node.y, node.groupId, node.groupPos]),
    [[0, 500, 'hand-made', 0], [900, 900, 'hand-made', 1]]);
  assertNoOverlap(layoutImageNodes(after).map(occupiedBox));
});

const promptGridLayout = (promptCount) => {
  // Two LoRAs, two epochs each, per prompt — so every prompt yields TWO grids
  // (one per LoRA, each spanning its epochs) and the wrap/overlap contracts
  // below are exercised on the real boundary.
  const images = Array.from({ length: promptCount }, (_, i) => `prompt ${i + 1}`)
    .flatMap((prompt, i) => [
      runImg(800 + i * 4, 106, 2500, 'multi-prompt-layout', prompt),
      runImg(801 + i * 4, 106, 3000, 'multi-prompt-layout', prompt),
      runImg(802 + i * 4, 114, 1500, 'multi-prompt-layout', prompt),
      runImg(803 + i * 4, 114, 2000, 'multi-prompt-layout', prompt),
    ]);
  const placedBatch = placeImageBatch({ graph: GRAPH, existing: [], images });
  const grouped = groupPinnedBatchTogether({ placed: placedBatch.placed, graph: GRAPH });
  return layoutImageNodes(grouped.rows);
};

const assertSeparatePromptGrids = (layout, promptCount) => {
  const groups = layout.filter((row) => row.kind === 'group');
  assert.equal(groups.length, promptCount * 2,
    'one Canvas renderable per prompt per LoRA');
  for (const group of groups) {
    assert.equal(new Set(group.members.map((m) => m.node.image.prompt)).size, 1,
      'a rendered grid never mixes prompt text');
    assert.equal(new Set(group.members.map((m) => m.node.image.record_id)).size, 1,
      'a rendered grid never mixes LoRAs');
    const steps = group.members.map((m) => m.node.image.step);
    assert.deepEqual(steps, [...steps].sort((a, b) => a - b),
      'each LoRA grid keeps epoch training order');
  }
  const gridBoxes = layout.map(occupiedBox);
  // Keep the two contracts separate so a failure says whether prompt grids hit
  // EACH OTHER (the COLUMN_ROWS wrap regression) or climb back into the tree
  // through their title bar (the BAND_GAP regression).
  assertNoOverlap(gridBoxes);
  assertNoOverlap([
    ...boardObstacles(GRAPH, []),
    ...gridBoxes,
  ]);
};

test('two prompts over two LoRAs become four non-overlapping rendered Canvas grids', () => {
  assertSeparatePromptGrids(promptGridLayout(2), 2);
});

test('seven prompts of LoRA grids stay separate and non-overlapping after the placement column wraps', () => {
  // COLUMN_ROWS is six: the seventh image of each source starts another
  // placement column. This is the boundary where two independently derived
  // strips can otherwise acquire the same visible footprint.
  assertSeparatePromptGrids(promptGridLayout(7), 7);
});

test('remembered free pictures reflow when their future wide grid reaches across a card', () => {
  const graph = { nodes: [card(999, 400, 0, [1000])], edges: [], width: 700, height: 120 };
  const images = [
    runImg(751, 999, 1000, 'remembered-run', 'remember this prompt'),
    runImg(752, 999, 2000, 'remembered-run', 'remember this prompt'),
  ];
  const remembered = Object.fromEntries(images.map((image, i) => [image.id, {
    imageId: image.id,
    x: i ? 800 : 100,
    y: 100,
    w: 320,
    h: 320,
    visible: false,
    groupId: null,
    groupPos: null,
    image,
  }]));
  const placedBatch = placeImageBatch({
    graph, existing: [], images, remembered,
  });

  // Both remembered squares are individually clear of the card. The overlap
  // only appears after the first one anchors the future 640-wide strip and its
  // 112-unit title bar reaches upward.
  assert.deepEqual(placedBatch.placed.map((row) => [row.imageId, row.x, row.y]),
    [[751, 100, 100], [752, 800, 100]]);
  assertNoOverlap([
    ...boardObstacles(graph, []),
    ...placedBatch.placed.map((row) => ({ x: row.x, y: row.y, w: row.w, h: row.h })),
  ]);

  const grouped = groupPinnedBatchTogether({
    nodes: Object.values(remembered), placed: placedBatch.placed, graph,
  });
  const layout = layoutImageNodes(grouped.rows);
  assert.equal(layout.length, 1);
  assert.equal(layout[0].kind, 'group');
  assertNoOverlap([...boardObstacles(graph, []), ...layout.map(occupiedBox)]);
  assert.ok(grouped.rows.find((row) => row.imageId === 751).y > 100,
    'the automatic grid anchor moved below the real card obstacle');
  assert.deepEqual(grouped.undoRows.map((row) => [row.imageId, row.x, row.y, row.visible]),
    [[751, 100, 100, false], [752, 800, 100, false]],
    'undo restores every remembered position exactly');

  const again = groupPinnedBatchTogether({
    nodes: Object.values(remembered), placed: placedBatch.placed, graph,
  });
  assert.deepEqual(again.rows.map((row) => [row.imageId, row.x, row.y]),
    grouped.rows.map((row) => [row.imageId, row.x, row.y]),
    'the graph-aware reflow is deterministic');
});

test('images with no run id keep the old per-checkpoint behaviour', () => {
  // A board that predates the published run id must draw what it always drew:
  // successive pins of one checkpoint still gather into one strip.
  const lane = {};
  pinOneByOne(lane, { id: 31, dataset_id: 3, record_id: 7, step: 500 });
  pinOneByOne(lane, { id: 32, dataset_id: 3, record_id: 7, step: 500 });
  assert.ok(lane[31].groupId);
  assert.equal(lane[31].groupId, lane[32].groupId);
});

test('a strip reads in TRAINING order, not alphabetical order', () => {
  // 1000 < 500 as text; as epochs it is the other way round.
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(1, 82, 1000, 'r1'), runPlaced(2, 82, 500, 'r1'),
    runPlaced(3, 82, 2000, 'r1'), runPlaced(4, 82, 1500, 'r1'),
  ] });
  const strip = [...result.rows].sort((a, b) => a.groupPos - b.groupPos);
  assert.deepEqual(strip.map((r) => r.image.step), [500, 1000, 1500, 2000]);
  assert.deepEqual(strip.map((r) => r.groupPos), [0, 1, 2, 3]);
});

test('a picture pinned later at an EARLIER epoch lands at the head of the strip', () => {
  const lane = {};
  pinOneByOne(lane, runImg(41, 7, 2000, 'runC'));
  pinOneByOne(lane, runImg(42, 7, 3000, 'runC'));
  pinOneByOne(lane, runImg(43, 7, 1000, 'runC'));
  const strip = Object.values(lane).sort((a, b) => a.groupPos - b.groupPos);
  assert.deepEqual(strip.map((r) => r.image.step), [1000, 2000, 3000]);
});

test('a lot that carries two runs is SPLIT, never concatenated', () => {
  const result = groupPinnedBatchTogether({ placed: [
    runPlaced(1, 82, 500, 'r1'), runPlaced(2, 82, 1000, 'r1'),
    runPlaced(3, 82, 500, 'r2'), runPlaced(4, 82, 1000, 'r2'),
  ] });
  const by = Object.fromEntries(result.rows.map((r) => [r.imageId, r.groupId]));
  assert.equal(by[1], by[2]);
  assert.equal(by[3], by[4]);
  assert.notEqual(by[1], by[3]);
});

test('the band lays its columns out in epoch order', () => {
  // One card, so every checkpoint anchors at the same x and the tie-break is
  // the whole answer. Left to right must be 500, 1000, 1500 — it was
  // 1000, 1500, 500 while the tie was broken by the alphabet.
  const graph = { nodes: [{ x: 0, y: 0, cellH: 100, node: { record_id: 9 } }] };
  const { placed: out } = placeImageBatch({ graph, existing: [], images: [
    img(51, 9, 1000), img(52, 9, 500), img(53, 9, 1500),
  ] });
  const xOf = (id) => out.find((p) => p.imageId === id).x;
  assert.ok(xOf(52) < xOf(51), 'step 500 sits left of step 1000');
  assert.ok(xOf(51) < xOf(53), 'step 1000 sits left of step 1500');
});

test('an over-cap lot keeps the EARLY epochs and says what it dropped', () => {
  const graph = { nodes: [{ x: 0, y: 0, cellH: 100, node: { record_id: 9 } }] };
  const images = Array.from({ length: PIN_BATCH_MAX + 3 },
    (_, i) => img(600 + i, 9, (i + 1) * 100));
  const res = placeImageBatch({ graph, existing: [], images });
  assert.equal(res.placed.length, PIN_BATCH_MAX);
  assert.equal(res.skipped.length, 3);
  const droppedSteps = res.skipped.map((s) => s.image.step);
  const keptSteps = res.placed.map((p) => p.image.step);
  assert.ok(Math.min(...droppedSteps) > Math.max(...keptSteps),
    'the tail of the training is what gets refused, not an arbitrary slice');
});

// ---- ✦ Tidy up: bringing a strayed STRIP home -----------------------------
/* Free placement lets a whole side-by-side strip be parked thousands of units
   above or left of everything. ✦ Tidy up is the way back — and it has to bring
   the strip back as ONE object. A tidy that re-flowed its members one by one
   would not tidy the strip, it would dismantle it. */

const stripRow = (groupId, anchor, members) => ({
  kind: 'group',
  key: `grp:${groupId}`,
  groupId,
  x: anchor.x,
  y: anchor.y,
  w: members.length * anchor.h,
  h: anchor.h,
  members: members.map((m, i) => ({
    key: `img:${m.imageId}`,
    node: m,
    x: anchor.x + i * anchor.h,
    y: anchor.y,
    w: anchor.h,
    h: anchor.h,
  })),
});

const member = (imageId, recordId, over = {}) => ({
  imageId,
  x: 0,
  y: 0,
  w: 200,
  h: 200,
  visible: true,
  groupId: 'g1',
  groupPos: 0,
  image: { id: imageId, url: `/img/${imageId}.png`, record_id: recordId, step: 2500 },
  ...over,
});

const overlaps = (a, b) => (a.x < b.x + b.w && b.x < a.x + a.w
  && a.y < b.y + b.h && b.y < a.y + a.h);

test('a strip parked far off the board comes back beside the run that made it', () => {
  const anchor = member(80, 114, { x: -9000, y: -7000, groupPos: 0 });
  const layout = [stripRow('g1', anchor, [anchor, member(81, 114, { groupPos: 1 })])];
  const { rows } = tidyGroupRows({ graph: GRAPH, layout });
  assert.equal(rows.length, 1, 'only the anchor is written — the strip derives from it');
  assert.equal(rows[0].imageId, 80);
  const sourceCard = GRAPH.nodes.find((n) => n.node.record_id === 114);
  assert.ok(rows[0].x >= sourceCard.x + CARD_W, 'to the right of its own card');
  assert.ok(rows[0].x >= 0 && rows[0].y >= 0, 'back inside the lane');
});

test('tidying a strip never touches its membership, so it cannot dissolve', () => {
  const anchor = member(80, 114, { x: -500, y: -500 });
  const layout = [stripRow('g1', anchor, [anchor, member(81, 114, { groupPos: 1 })])];
  const { rows } = tidyGroupRows({ graph: GRAPH, layout });
  assert.ok(!('groupId' in rows[0]) && !('groupPos' in rows[0]),
    'a write that only mentions geometry can never dissolve a group');
  assert.deepEqual([rows[0].w, rows[0].h], [anchor.w, anchor.h],
    'the strip keeps the size it had — only where it sits changes');
});

test('a repatriated strip lands on no card and on nothing already placed', () => {
  const a1 = member(80, 114, { x: -900, y: -900, groupPos: 0 });
  const a2 = member(90, 114, { x: -400, y: -400, groupId: 'g2', groupPos: 0 });
  const layout = [
    stripRow('g1', a1, [a1, member(81, 114, { groupPos: 1 })]),
    stripRow('g2', a2, [a2, member(91, 114, { groupId: 'g2', groupPos: 1 })]),
  ];
  const { rows, boxes } = tidyGroupRows({ graph: GRAPH, layout });
  assert.equal(rows.length, 2);
  for (const n of GRAPH.nodes) {
    const cardBox = { x: n.x, y: n.y, w: CARD_W, h: n.cellH };
    for (const row of rows) {
      assert.ok(!overlaps(row, cardBox),
        `strip ${row.imageId} landed on run ${n.node.record_id}`);
    }
  }
  const [b1, b2] = boxes.slice(-2);
  assert.ok(!overlaps(b1, b2), 'the two strips do not land on each other');
  // The footprints handed back are what the contact-sheet band must avoid, and
  // they reserve the group's drag BAR as well as its pictures.
  assert.ok(b1.h > rows[0].h, 'the bar above the strip is reserved too');
});

test('two strips are repatriated in the same order whatever order they arrive in', () => {
  const build = (straight) => {
    const a1 = member(80, 106, { x: -900, y: -900 });
    const a2 = member(90, 117, { x: -400, y: -400, groupId: 'g2' });
    const s1 = stripRow('g1', a1, [a1, member(81, 106, { groupPos: 1 })]);
    const s2 = stripRow('g2', a2, [a2, member(91, 117, { groupId: 'g2', groupPos: 1 })]);
    return tidyGroupRows({ graph: GRAPH, layout: straight ? [s1, s2] : [s2, s1] }).rows;
  };
  const byId = (list) => [...list].sort((a, b) => a.imageId - b.imageId)
    .map((r) => [r.imageId, r.x, r.y]);
  assert.deepEqual(byId(build(true)), byId(build(false)));
});

test('a lane with no strip on it asks for no writes at all', () => {
  const lone = member(80, 114, { groupId: null, groupPos: null });
  const layout = [{ kind: 'single', key: 'img:80', node: lone, x: 0, y: 0, w: 200, h: 200 }];
  const { rows, boxes } = tidyGroupRows({ graph: GRAPH, layout });
  assert.deepEqual(rows, []);
  assert.deepEqual(boxes, [], 'and it does not pretend the cards are taken either');
});

/* ── ✦ Tidy up across LANES ──────────────────────────────────────────────────
   The board stacks datasets vertically, and a lane's stacking height decides
   where the next one starts. Tidy up lays a lane's strips and its contact-sheet
   band BELOW its tree, so a stack measured on the tree alone starts the next
   dataset straight through them — strips piled on strips and on other lanes'
   run cards, with nobody having dragged anything.

   These build a DENSE lane the way a real board is dense (strips of 2 to 6
   pictures at four different sizes, resized well past the 320 default) and
   assert what the user sees, in WORLD units. */

const denseLane = (base) => {
  // [groupId, recordId, memberCount, tile size]
  const specs = [
    [`${base}a`, 106, 6, 320],
    [`${base}b`, 107, 4, 620],
    [`${base}c`, 114, 3, 900],
    [`${base}d`, 117, 2, 180],
  ];
  const nodes = [];
  let id = base;
  for (const [groupId, recordId, count, size] of specs) {
    for (let i = 0; i < count; i += 1) {
      nodes.push(member(id, recordId, {
        // Parked all over the board, including far above and left of the lane.
        x: -900 + i * size, y: -700 + i * 40, w: size, h: size,
        groupId, groupPos: i,
      }));
      id += 1;
    }
  }
  // …and two loose pictures, which land in the band under the strips.
  for (const recordId of [106, 114]) {
    nodes.push(member(id, recordId, { x: -300, y: -300, w: 320, h: 320,
      groupId: null, groupPos: null }));
    id += 1;
  }
  return nodes;
};

/** The lane, tidied: every node at the geometry ✦ Tidy up writes for it. */
const tidiedLane = (nodes) => {
  const map = new Map(nodes.map((n) => [n.imageId, { ...n }]));
  for (const row of tidyLaneRows({ graph: GRAPH, nodes }).rows) {
    map.set(row.imageId, { ...map.get(row.imageId), ...row });
  }
  return [...map.values()];
};

/** Every footprint a tidied board really draws, in WORLD units. */
const worldBoxes = (lanes) => {
  const world = stackLanes(lanes.map((nodes) => ({
    width: GRAPH.width,
    height: Math.max(GRAPH.height, tidyLaneReach({ graph: GRAPH, nodes })),
  })));
  const out = [];
  world.lanes.forEach((lane, i) => {
    for (const c of GRAPH.nodes) {
      out.push({ label: `L${i} card ${c.node.record_id}`,
        x: lane.x + c.x, y: lane.graphY + c.y, w: CARD_W, h: c.cellH });
    }
    for (const row of layoutImageNodes(lanes[i])) {
      const box = occupiedBox(row);
      out.push({
        label: `L${i} ${row.kind === 'group' ? `strip ${row.groupId}` : `img ${row.node.imageId}`}`,
        x: lane.x + box.x, y: lane.graphY + box.y, w: box.w, h: box.h });
    }
  });
  return out;
};

const collisions = (boxes) => {
  const out = [];
  for (let i = 0; i < boxes.length; i += 1) {
    for (let j = i + 1; j < boxes.length; j += 1) {
      if (overlaps(boxes[i], boxes[j])) out.push(`${boxes[i].label} / ${boxes[j].label}`);
    }
  }
  return out;
};

test('a dense lane tidies onto itself without a single overlap', () => {
  assert.deepEqual(collisions(worldBoxes([tidiedLane(denseLane(200))])), [],
    'strips, band and run cards all clear of each other');
});

test('✦ Tidy up on three dense lanes puts nothing on another lane', () => {
  const lanes = [200, 400, 600].map((base) => tidiedLane(denseLane(base)));
  const boxes = worldBoxes(lanes);
  // Without the tidy reach in the stacking height, every one of these strips
  // lands on the lane below — 18 overlaps on this very board.
  assert.deepEqual(collisions(boxes), []);
  const laneTops = boxes.filter((b) => b.label.endsWith('card 106'))
    .map((b) => b.y).sort((a, b) => a - b);
  assert.equal(laneTops.length, 3);
  assert.ok(laneTops[1] - laneTops[0] > GRAPH.height,
    "the second lane starts below the first one's pictures, not below its tree");
});

test('the tidy reach is measured on SIZES, so dragging a picture moves no lane', () => {
  const nodes = tidiedLane(denseLane(200));
  const before = tidyLaneReach({ graph: GRAPH, nodes });
  // The gesture the free-placement fix exists for: one picture dragged far
  // below its own lane. The lane underneath must not budge.
  const dragged = nodes.map((n, i) => (i === 0 ? { ...n, x: n.x + 2500, y: n.y + 9000 } : n));
  assert.equal(tidyLaneReach({ graph: GRAPH, nodes: dragged }), before);
  // Resizing one, on the other hand, genuinely needs more room and says so.
  const bigger = nodes.map((n, i) => (i === 0 ? { ...n, w: n.w + 600, h: n.h + 600 } : n));
  assert.ok(tidyLaneReach({ graph: GRAPH, nodes: bigger }) > before,
    'a strip made bigger reserves more room');
});

test('the tidy reach of a lane with no pinned picture is nothing at all', () => {
  assert.equal(tidyLaneReach({ graph: GRAPH, nodes: [] }), 0,
    'a board that never pinned anything stacks exactly as it always did');
  assert.equal(tidyLaneReach({ graph: GRAPH,
    nodes: denseLane(200).map((n) => ({ ...n, visible: false })) }), 0,
  'and a lane whose pictures are all CLOSED reserves nothing either');
});

test('tidyLaneRows writes every visible picture exactly once', () => {
  const nodes = denseLane(200);
  const { rows } = tidyLaneRows({ graph: GRAPH, nodes });
  const ids = rows.map((r) => r.imageId);
  assert.equal(new Set(ids).size, ids.length, 'no picture is written twice');
  // Strips are moved by their ANCHOR alone; the loose ones are each placed.
  const anchors = [200, 206, 210, 213];
  const loose = nodes.filter((n) => !n.groupId).map((n) => n.imageId);
  for (const id of [...anchors, ...loose]) assert.ok(ids.includes(id), `${id} was placed`);
  assert.equal(rows.length, anchors.length + loose.length);
});

test('tidyLaneRows is deterministic — the same lane twice gives the same board', () => {
  const nodes = denseLane(200);
  const key = (list) => [...list].sort((a, b) => a.imageId - b.imageId)
    .map((r) => [r.imageId, r.x, r.y, r.w, r.h]);
  assert.deepEqual(
    key(tidyLaneRows({ graph: GRAPH, nodes: [...nodes].reverse() }).rows),
    key(tidyLaneRows({ graph: GRAPH, nodes }).rows));
});

/* ── The lane stack, during a drag ───────────────────────────────────────────
   Reported as "I cannot bring the top node down without bringing the bottom one
   down too": a pinned picture dragged out of a strip in the FIRST lane, and the
   whole floor below spreads apart while the hand is still moving.

   `tidyLaneReach` is position-independent, and these tests do not doubt it —
   the first one pins exactly that. But it is not membership-independent, and a
   picture on its way out of a strip changes the membership on every frame: the
   strip loses a member and the contact-sheet band gains one, so the room the
   stack reserves moves under the hand. Hence the split the rest of these drive:
   the REACH follows the gesture, the STACK is measured on the resting rows. */

const dragLane = () => {
  const mk = (id, x, y, w, h, extra = {}) => ({
    imageId: id, x, y, w, h, visible: true,
    image: { record_id: 'r1', step: id * 100, aspect: '1:1' }, ...extra,
  });
  return [
    mk(11, 40, 20, 200, 200, { groupId: 'g11', groupPos: 0 }),
    mk(12, 40, 20, 200, 200, { groupId: 'g11', groupPos: 1 }),
    mk(13, 40, 20, 200, 200, { groupId: 'g11', groupPos: 2 }),
  ];
};

// Mid-gesture, `imagesByLane` gives the dragged member the drag box — and once
// it clears the strip that box is its OWN remembered size, three times the slot
// it had in the band (LineageCanvas: the "leaving" preview).
const leavingList = (nodes) => nodes.map((n) => (n.imageId === 11
  ? { ...n, x: 40, y: 720, w: 620, h: 620 } : n));

test('tidyLaneReach ignores where a picture SITS', () => {
  const nodes = dragLane();
  const moved = nodes.map((n) => (n.imageId === 11 ? { ...n, x: 900, y: 4000 } : n));
  assert.equal(tidyLaneReach({ graph: GRAPH, nodes: moved }),
    tidyLaneReach({ graph: GRAPH, nodes }));
});

test('tidyLaneReach is NOT membership-independent — which is why the stack '
  + 'must not be fed the gesture', () => {
  const nodes = dragLane();
  // Not a bug in this function: a lane really does need different room once a
  // picture has left its strip. Asserted so the reason the fix lives in
  // `laneStackEntries` rather than here stays on the record.
  assert.notEqual(tidyLaneReach({ graph: GRAPH, nodes: leavingList(nodes) }),
    tidyLaneReach({ graph: GRAPH, nodes }));
});

test('a drag moves NO lane below it, however far it goes', () => {
  const resting = { 1: dragLane(), 2: [] };
  const placed = [
    { datasetId: 1, graph: GRAPH },
    { datasetId: 2, graph: GRAPH },
  ];
  const laneTops = (restingByLane, layoutByLane) => stackLanes(
    laneStackEntries({ placed, layoutByLane, restingByLane }),
  ).lanes.map((l) => l.graphY);

  const atRest = laneTops(resting, {
    1: layoutImageNodes(resting[1]), 2: [],
  });
  // The frame the report is about: member 11 dragged 700 units down and out of
  // its strip. The layout follows it (detached, at its own size) — the stack
  // must not.
  const live = leavingList(resting[1]).map((n) => (n.imageId === 11
    ? { ...n, groupId: null, groupPos: null } : n));
  const mid = laneTops(resting, { 1: layoutImageNodes(live), 2: [] });

  assert.deepEqual(mid, atRest, 'no lane moved while the picture was in flight');
});

test('the dragged picture still grows the BOX that ✦ Fit and 📷 Export frame', () => {
  const resting = { 1: dragLane(), 2: [] };
  const placed = [{ datasetId: 1, graph: GRAPH }, { datasetId: 2, graph: GRAPH }];
  const live = leavingList(resting[1]).map((n) => (n.imageId === 11
    ? { ...n, groupId: null, groupPos: null } : n));
  const world = stackLanes(laneStackEntries({
    placed, restingByLane: resting, layoutByLane: { 1: layoutImageNodes(live), 2: [] },
  }));
  const still = stackLanes(laneStackEntries({
    placed, restingByLane: resting, layoutByLane: { 1: layoutImageNodes(resting[1]), 2: [] },
  }));
  assert.ok(world.height > still.height,
    'a picture dragged below its lane is still reachable, framed and exported');
});

test('moving a RUN CARD down moves no dataset below it — during OR after', () => {
  /* Two reports, one cause. First: "if I take a dataset node and drag it down,
     all the dataset nodes below come down with it". Then, after freezing it for
     the gesture only: "…which creates space for nothing" — a card DROPPED low
     left its lane permanently taller, so the next dataset stayed pushed down
     with dead board between them. The stack must therefore ignore the
     arrangement entirely, not merely ignore it while a finger is down. */
  const auto = [
    { datasetId: 1, graph: GRAPH },
    { datasetId: 2, graph: GRAPH },
    { datasetId: 3, graph: GRAPH },
  ];
  const arranged = (h) => [
    { datasetId: 1, graph: { ...GRAPH, height: h } },
    { datasetId: 2, graph: GRAPH },
    { datasetId: 3, graph: GRAPH },
  ];
  const laneTops = (live) => stackLanes(laneStackEntries({
    placed: live, stackPlaced: auto, layoutByLane: {}, restingByLane: {},
  })).lanes.map((l) => l.graphY);

  const atRest = laneTops(auto);
  // 700 units down: mid-gesture and, identically, once dropped and stored.
  assert.deepEqual(laneTops(arranged(GRAPH.height + 700)), atRest,
    'a dataset below moved while the card above was still under the hand');
  assert.deepEqual(laneTops(arranged(GRAPH.height + 700)), atRest,
    'the gap survived the drop — the lane below stayed pushed away');
});

test('the moved run card still grows the BOX that ✦ Fit and 📷 Export frame', () => {
  const auto = [{ datasetId: 1, graph: GRAPH }, { datasetId: 2, graph: GRAPH }];
  const arranged = [
    { datasetId: 1, graph: { ...GRAPH, height: GRAPH.height + 700 } },
    { datasetId: 2, graph: GRAPH },
  ];
  const box = (p) => stackLanes(laneStackEntries({
    placed: p, stackPlaced: auto, layoutByLane: {}, restingByLane: {},
  })).height;
  assert.ok(box(arranged) > box(auto),
    'a card parked below its lane must still be framed and exported');
});

test('without stackPlaced the entries are what they always were', () => {
  // A caller that never arranges anything cannot tell this change happened.
  const placed = [{ datasetId: 1, graph: GRAPH }];
  const withOut = laneStackEntries({ placed, layoutByLane: {}, restingByLane: {} });
  const withSame = laneStackEntries({
    placed, stackPlaced: placed, layoutByLane: {}, restingByLane: {},
  });
  assert.deepEqual(withOut, withSame);
  assert.equal(withOut[0].height, GRAPH.height);
});

test('✦ Tidy up spends WIDTH, so a lane keeps its own height', () => {
  /* The room a tidied lane needs under its tree is room the dataset below it
     gets pushed away by — permanently, and whether or not the button is ever
     pressed. Measured before this changed, on a 150-unit tree: one pinned
     picture reserved 518 units and two reserved 862. Tidy up therefore lays its
     strips and its contact sheet to the RIGHT, wrapping into a new column
     rather than a seventh row. */
  const graph = { nodes: [card(1, 22, 22, [1000])], edges: [], width: 700, height: 150 };
  const pic = (id, i) => ({ imageId: id, x: 40 + i * 340, y: 900, w: 320, h: 320,
    visible: true, image: { record_id: 1, step: 1000 * (i + 1) } });

  const reachOf = (n) => tidyLaneReach({
    graph, nodes: Array.from({ length: n }, (_, i) => pic(i + 1, i)),
  });
  assert.ok(reachOf(8) <= reachOf(1) + 1,
    'the eighth picture made the lane taller — the band is still stacking down');
  assert.ok(reachOf(4) < 500, `four pictures still reserve ${reachOf(4)} units`);

  // …and they really are out to the RIGHT of the cards, not under them.
  const rightOfCards = Math.max(...graph.nodes.map((n) => n.x + CARD_W));
  const { rows } = tidyLaneRows({ graph, nodes: [pic(1, 0), pic(2, 1), pic(3, 2)] });
  assert.ok(rows.every((r) => r.x >= rightOfCards),
    'a tidied picture landed under the tree instead of beside it');
});

test('a fresh 📌 batch still hangs UNDER the lineage that made it', () => {
  // The other half of the same decision: a contact sheet of a run's outputs
  // reads under that run. Only ✦ Tidy up asks for the sideways layout.
  const graph = { nodes: [card(1, 22, 22, [1000])], edges: [], width: 700, height: 150 };
  const images = [1, 2, 3].map((i) => ({ id: i, record_id: 1, step: 1000 * i }));
  const under = placeImageBatch({ graph, images, max: 3 });
  assert.ok(under.placed.every((p) => p.y >= graph.height),
    'the pin batch moved sideways — that is the tidy layout, not this one');
});
