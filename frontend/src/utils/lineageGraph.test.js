import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildLineageGraph, graphMetrics, CARD_W, CARD_H, PAD, PILL_W, PILL_H,
  PILL_W_BIG, PILL_H_BIG, PILLS_PER_ROW,
} from './lineageGraph.js';

const ck = (step, extra = {}) => ({ step, present: true,
  download_url: `/dl?step=${step}`, ...extra });

const chain = {
  root_id: 1,
  current_id: 3,
  nodes: [
    { record_id: 1, parent_record_id: null, created_at: '2026-07-01T00:00:00' },
    { record_id: 2, parent_record_id: 1, resumed_from: 1000, created_at: '2026-07-02T00:00:00' },
    { record_id: 3, parent_record_id: 2, resumed_from: 1500, created_at: '2026-07-03T00:00:00', is_current: true },
  ],
  edges: [
    { parent: 1, child: 2, resumed_from: 1000 },
    { parent: 2, child: 3, resumed_from: 1500, superseded: false },
  ],
};

const byId = (g) => new Map(g.nodes.map((n) => [n.node.record_id, n]));

test('a chain lays out left-to-right: depth increases, x increases, one row', () => {
  const g = buildLineageGraph(chain);
  const m = byId(g);
  assert.deepEqual([1, 2, 3].map((i) => m.get(i).depth), [0, 1, 2]);
  // strictly increasing x, one generation apart
  assert.ok(m.get(1).x < m.get(2).x && m.get(2).x < m.get(3).x);
  // a straight chain shares a single row → identical y
  assert.equal(m.get(1).y, m.get(2).y);
  assert.equal(m.get(2).y, m.get(3).y);
  // first card sits at the padding origin
  assert.equal(m.get(1).x, PAD);
  assert.equal(m.get(1).y, PAD);
});

test('a parent with a tall pill block is not overlapped by the next root (#76-over-#81 bug)', () => {
  // #76 had 12 checkpoints (3 pill rows) but its child continued far to the right;
  // the next root (#81) was placed in the same column and landed INSIDE #76's pills.
  const many = Array.from({ length: 12 }, (_, i) => ck((i + 1) * 250));
  const tree = {
    root_id: 1,
    nodes: [
      { record_id: 1, parent_record_id: null, created_at: '2026-07-01T00:00:00', checkpoints: many },
      { record_id: 2, parent_record_id: 1, resumed_from: 3000, created_at: '2026-07-02T00:00:00',
        checkpoints: [ck(500), ck(1000)] },
      { record_id: 3, parent_record_id: null, created_at: '2026-07-03T00:00:00',
        checkpoints: [ck(500), ck(1000)] },
    ],
    edges: [{ parent: 1, child: 2, resumed_from: 3000 }],
  };
  const g = buildLineageGraph(tree);
  const m = byId(g);
  // #1 (parent, 12 pills) and #3 (second root) share the left column
  assert.equal(m.get(1).x, m.get(3).x);
  // #3's card must start below #1's WHOLE cell (card + tall pill block)
  assert.ok(m.get(3).y >= m.get(1).y + m.get(1).cellH,
    `#3 (y=${m.get(3).y}) overlaps #1's cell (y=${m.get(1).y}, cellH=${m.get(1).cellH})`);
  // concretely: #1's lowest pill sits above #3's card top — no visual collision
  const lowestPill = Math.max(...m.get(1).checkpoints.map((p) => p.y + p.h));
  assert.ok(lowestPill <= m.get(3).y,
    `#1 pills (bottom ${lowestPill}) overlap #3 card (top ${m.get(3).y})`);
});

test('every edge carries a bezier path and the whole chain is on the spine', () => {
  const g = buildLineageGraph(chain);
  assert.equal(g.edges.length, 2);
  for (const e of g.edges) {
    assert.match(e.d, /^M[\d.]+,[\d.]+ C/); // move-then-cubic
    assert.equal(e.onSpine, true);
  }
  assert.deepEqual([...g.spine].sort((a, b) => a - b), [1, 2, 3]);
});

test('a fork centres the parent between its two children and never overlaps', () => {
  const forked = {
    root_id: 1,
    current_id: 4,
    nodes: [
      { record_id: 1, parent_record_id: null, created_at: '2026-07-01T00:00:00' },
      { record_id: 2, parent_record_id: 1, created_at: '2026-07-02T00:00:00' },
      { record_id: 3, parent_record_id: 1, created_at: '2026-07-03T00:00:00' },
      { record_id: 4, parent_record_id: 2, created_at: '2026-07-04T00:00:00', is_current: true },
    ],
    edges: [
      { parent: 1, child: 2 }, { parent: 1, child: 3 }, { parent: 2, child: 4 },
    ],
  };
  const g = buildLineageGraph(forked);
  const m = byId(g);
  // root centred vertically between its branches
  const mid = (m.get(2).y + m.get(3).y) / 2;
  assert.ok(Math.abs(m.get(1).y - mid) < 0.01);
  // no two cards share the same rectangle
  const seen = new Set();
  for (const n of g.nodes) {
    const key = `${n.x},${n.y}`;
    assert.ok(!seen.has(key), 'cards must not overlap');
    seen.add(key);
  }
  // spine is 1 → 2 → 4 (the current run), not the 3 fork
  assert.deepEqual([...g.spine].sort((a, b) => a - b), [1, 2, 4]);
  assert.equal(g.edges.find((e) => e.childId === 3).onSpine, false);
});

test('superseded flag rides through onto the edge', () => {
  const g = buildLineageGraph({
    root_id: 1, current_id: 2,
    nodes: [
      { record_id: 1, parent_record_id: null },
      { record_id: 2, parent_record_id: 1, is_current: true },
    ],
    edges: [{ parent: 1, child: 2, superseded: true }],
  });
  assert.equal(g.edges[0].superseded, true);
});

test('ancestorsOf gives each node its path back to the root', () => {
  const g = buildLineageGraph(chain);
  assert.deepEqual([...g.ancestorsOf.get(3)].sort((a, b) => a - b), [1, 2]);
  assert.deepEqual([...g.ancestorsOf.get(1)], []);
});

test('content bounds cover every card', () => {
  const g = buildLineageGraph(chain);
  for (const n of g.nodes) {
    assert.ok(n.x + CARD_W <= g.width);
    assert.ok(n.y + CARD_H <= g.height);
    assert.ok(n.x >= 0 && n.y >= 0);
  }
});

test('an orphaned node (broken edge) is still placed as its own root', () => {
  const g = buildLineageGraph({
    root_id: 1, current_id: 1,
    nodes: [
      { record_id: 1, parent_record_id: null },
      { record_id: 9, parent_record_id: 7 }, // parent 7 does not exist
    ],
    edges: [{ parent: 7, child: 9 }],
  });
  assert.equal(g.nodes.length, 2);
  assert.ok(g.nodes.some((n) => n.node.record_id === 9));
});

test('a cyclic edge never loops and places each node once', () => {
  const g = buildLineageGraph({
    root_id: 1,
    nodes: [
      { record_id: 1, parent_record_id: 2 },
      { record_id: 2, parent_record_id: 1 },
    ],
    edges: [{ parent: 1, child: 2 }, { parent: 2, child: 1 }],
  });
  assert.equal(g.nodes.length, 2);
});

// --- checkpoints as pills ----------------------------------------------------

test('a run carries its checkpoints as wrapped pills within the card width', () => {
  const many = Array.from({ length: 13 }, (_, i) => ck((i + 1) * 500));
  const g = buildLineageGraph({
    root_id: 1, current_id: 1, single: true,
    nodes: [{ record_id: 1, parent_record_id: null, is_current: true, checkpoints: many }],
    edges: [],
  });
  const n = g.nodes[0];
  assert.equal(n.checkpoints.length, 13);
  // pills never spill past the card's width, and wrap into several rows
  for (const p of n.checkpoints) {
    assert.ok(p.x >= n.x && p.x + p.w <= n.x + CARD_W + 0.01, 'pill within card width');
    assert.ok(p.y >= n.y + CARD_H, 'pill sits below the card');
  }
  const rows = new Set(n.checkpoints.map((p) => p.y)).size;
  assert.equal(rows, Math.ceil(13 / PILLS_PER_ROW));
  // the cell (and the content bounds) grow to include the pills block
  assert.ok(n.cellH > CARD_H);
  assert.ok(n.y + n.cellH <= g.height);
});

test('a lone run with checkpoints still produces a graph (no lineage needed)', () => {
  const g = buildLineageGraph({
    root_id: 5, current_id: 5, single: true,
    nodes: [{ record_id: 5, parent_record_id: null, is_current: true,
      checkpoints: [ck(500), ck(1000, { final: true })] }],
    edges: [],
  });
  assert.equal(g.nodes.length, 1);
  assert.equal(g.nodes[0].checkpoints.length, 2);
  assert.equal(g.nodes[0].checkpoints[1].final, true);
});

test('a continuation edge anchors on the parent pill it resumed from', () => {
  const g = buildLineageGraph({
    root_id: 1, current_id: 2,
    nodes: [
      { record_id: 1, parent_record_id: null,
        checkpoints: [ck(500), ck(1000), ck(1500)] },
      { record_id: 2, parent_record_id: 1, resumed_from: 1000, is_current: true },
    ],
    edges: [{ parent: 1, child: 2, resumed_from: 1000, superseded: false }],
  });
  const parent = g.nodes.find((n) => n.node.record_id === 1);
  const anchor = parent.checkpoints.find((p) => p.step === 1000);
  const edge = g.edges[0];
  assert.equal(edge.anchoredStep, 1000);
  assert.equal(anchor.isResumeSource, true);
  // the path starts at the anchor pill's right-centre, not the card edge
  const start = edge.d.match(/^M([\d.]+),([\d.]+)/);
  assert.ok(Math.abs(Number(start[1]) - (anchor.x + anchor.w)) < 0.01);
  assert.ok(Math.abs(Number(start[2]) - (anchor.y + PILL_H / 2)) < 0.01);
  // only the resumed pill is flagged
  assert.equal(parent.checkpoints.filter((p) => p.isResumeSource).length, 1);
});

test('edge falls back to the card edge when no pill matches the resume step', () => {
  const g = buildLineageGraph({
    root_id: 1, current_id: 2,
    nodes: [
      { record_id: 1, parent_record_id: null, checkpoints: [ck(500)] },
      { record_id: 2, parent_record_id: 1, resumed_from: 9999, is_current: true },
    ],
    edges: [{ parent: 1, child: 2, resumed_from: 9999 }],
  });
  const parent = g.nodes.find((n) => n.node.record_id === 1);
  assert.equal(g.edges[0].anchoredStep, null);
  assert.equal(parent.checkpoints.every((p) => !p.isResumeSource), true);
  const start = g.edges[0].d.match(/^M([\d.]+),/);
  assert.ok(Math.abs(Number(start[1]) - (parent.x + CARD_W)) < 0.01); // card right edge
});

// --- 🔍 big-preview mode (adaptive geometry) --------------------------------

test('graphMetrics: compact is the default, big enlarges the tiles', () => {
  const compact = graphMetrics(false);
  assert.equal(compact.pillW, PILL_W);
  assert.equal(compact.pillH, PILL_H);
  assert.equal(compact.perRow, PILLS_PER_ROW);
  const big = graphMetrics(true);
  assert.equal(big.pillW, PILL_W_BIG);
  assert.equal(big.pillH, PILL_H_BIG);
  assert.ok(big.pillW > compact.pillW && big.pillH > compact.pillH);
  // fewer, bigger tiles per row — but always at least one
  assert.ok(big.perRow >= 1 && big.perRow < compact.perRow);
});

test('big-preview mode sizes the pills up and grows the cell height', () => {
  const nodes = [{ record_id: 1, parent_record_id: null, is_current: true,
    checkpoints: [ck(500), ck(1000, { final: true }), ck(1500)] }];
  const tree = { root_id: 1, current_id: 1, single: true, nodes, edges: [] };
  const compact = buildLineageGraph(tree);
  const big = buildLineageGraph(tree, { bigPreviews: true });
  // Each pill carries the big geometry...
  for (const p of big.nodes[0].checkpoints) {
    assert.equal(p.w, PILL_W_BIG);
    assert.equal(p.h, PILL_H_BIG);
    // ...and still never spills past the card width.
    assert.ok(p.x + p.w <= big.nodes[0].x + CARD_W + 0.01);
  }
  // The taller tiles make the cell (and the whole graph) taller than compact.
  assert.ok(big.nodes[0].cellH > compact.nodes[0].cellH);
  assert.ok(big.height > compact.height);
});

test('empty / missing payloads return a safe empty shape', () => {
  for (const bad of [null, undefined, {}, { nodes: [] }]) {
    const g = buildLineageGraph(bad);
    assert.deepEqual(g.nodes, []);
    assert.deepEqual(g.edges, []);
    assert.equal(g.width, 0);
    assert.equal(g.spine.size, 0);
  }
});
