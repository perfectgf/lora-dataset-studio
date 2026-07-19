import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildLineageGraph, CARD_W, CARD_H, PAD,
} from './lineageGraph.js';

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

test('empty / missing payloads return a safe empty shape', () => {
  for (const bad of [null, undefined, {}, { nodes: [] }]) {
    const g = buildLineageGraph(bad);
    assert.deepEqual(g.nodes, []);
    assert.deepEqual(g.edges, []);
    assert.equal(g.width, 0);
    assert.equal(g.spine.size, 0);
  }
});
