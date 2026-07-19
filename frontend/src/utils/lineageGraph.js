/* Pure layout for the ◉ Graph view of a run's lineage — the second, showcase
   rendering of the same {nodes, edges} genealogy the ☰ List view draws
   (utils/lineageTree.js). Framework-free (no JSX, no d3) so the geometry is
   unit-testable with node:test; the SVG renderer lives in
   components/dataset/RunLineageGraph.jsx.

   Layout is a tidy left-to-right tree: the root sits on the left and each
   continuation flows one generation to the right, so a lineage reads as a
   timeline (run → continue → re-continue) with forks stacking vertically.
   Positions use the leaf-slot method — every leaf gets its own row, each parent
   is centred over the span of its children — which never overlaps for the small
   trees a lineage is (3-10 runs) without pulling in a graph library. Defensive
   against a missing root, a cycle, or an orphaned edge, exactly like the list:
   every node is placed once, unreachable nodes become their own roots so
   nothing vanishes. */

import { siblingSort } from './lineageTree.js';

// Card + spacing geometry, in SVG user units (1 unit ≈ 1px at scale 1). The
// renderer draws each run as a fixed-size card so the layout is deterministic.
export const CARD_W = 264;
export const CARD_H = 64;
export const H_GAP = 62;   // gap between one generation and the next (→)
export const V_GAP = 26;   // gap between sibling rows (↓)
export const PAD = 22;     // breathing room around the whole tree

const COL = CARD_W + H_GAP;   // centre-to-centre horizontal step per depth
const ROW = CARD_H + V_GAP;   // centre-to-centre vertical step per leaf row

/** Children-by-parent map (sorted oldest-first, same order as the list), plus a
 *  by-id index — the shared spine of both layouts. */
function indexTree(tree) {
  const nodes = Array.isArray(tree?.nodes) ? tree.nodes : [];
  const byId = new Map(nodes.map((n) => [n.record_id, n]));
  const childrenOf = new Map();
  for (const e of (tree?.edges || [])) {
    if (!byId.has(e.parent) || !byId.has(e.child)) continue;
    if (!childrenOf.has(e.parent)) childrenOf.set(e.parent, []);
    childrenOf.get(e.parent).push(e.child);
  }
  for (const [pid, kids] of childrenOf) {
    childrenOf.set(pid, kids
      .map((cid) => byId.get(cid)).filter(Boolean).sort(siblingSort)
      .map((n) => n.record_id));
  }
  return { nodes, byId, childrenOf };
}

/** Cubic-bezier path from a parent card's right edge to a child card's left
 *  edge, with horizontal tangents so the curve leaves and arrives flat — the
 *  smooth "flowing" connector, not a kinked polyline. */
function edgePath(parent, child) {
  const x1 = parent.x + CARD_W, y1 = parent.y + CARD_H / 2;
  const x2 = child.x, y2 = child.y + CARD_H / 2;
  const mx = x1 + (x2 - x1) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

/**
 * Lay out a {root_id, current_id, nodes, edges} lineage for the graph view.
 * Returns:
 *   nodes: [{ node, x, y, depth, isCurrent, onSpine }]  — top-left of each card
 *   edges: [{ parentId, childId, d, superseded, onSpine }]  — `d` is an SVG path
 *   width, height     — content bounds (PAD included) for the viewBox
 *   spine             — Set of node ids on the root→current path (the trunk)
 *   ancestorsOf       — Map id → Set of that node's ancestor ids (self excluded),
 *                       so hover can light a node's whole path back to the root
 * Empty / malformed payloads return an empty, safe shape.
 */
export function buildLineageGraph(tree) {
  const empty = { nodes: [], edges: [], width: 0, height: 0,
    spine: new Set(), ancestorsOf: new Map() };
  const { nodes, byId, childrenOf } = indexTree(tree);
  if (!nodes.length) return empty;

  // Roots: the declared root if present, else every parentless node, else the
  // first node — then any node an edge never reached is appended as its own root
  // (a broken/foreign edge must not make a run disappear).
  const roots = [];
  const declared = tree?.root_id != null && byId.has(tree.root_id) ? tree.root_id : null;
  if (declared != null) roots.push(declared);
  for (const n of nodes) {
    if (n.record_id === declared) continue;
    if (n.parent_record_id == null || !byId.has(n.parent_record_id)) roots.push(n.record_id);
  }
  if (!roots.length) roots.push(nodes[0].record_id);

  const placed = new Map();   // id -> { node, x, y, depth }
  const parentOf = new Map(); // id -> parent id (as laid out)
  let nextRow = 0;

  // Post-order walk: leaves take the next free row, a parent centres on the span
  // of its children. A seen-set breaks cycles and shared-child anomalies so each
  // node is placed exactly once.
  const place = (id, depth, parentId) => {
    if (placed.has(id)) return placed.get(id).row;
    const node = byId.get(id);
    if (!node) return null;
    placed.set(id, { node, depth, row: 0, x: 0, y: 0 }); // reserve (cycle guard)
    if (parentId != null) parentOf.set(id, parentId);
    const kids = (childrenOf.get(id) || []).filter((cid) => !placed.has(cid));
    let row;
    if (!kids.length) {
      row = nextRow++;
    } else {
      const childRows = kids.map((cid) => place(cid, depth + 1, id)).filter((r) => r != null);
      row = childRows.length
        ? (Math.min(...childRows) + Math.max(...childRows)) / 2
        : nextRow++;
    }
    const slot = placed.get(id);
    slot.row = row;
    slot.depth = depth;
    slot.x = PAD + depth * COL;
    slot.y = PAD + row * ROW;
    return row;
  };
  for (const rid of roots) place(rid, 0, null);
  // Any node still unplaced (unreachable) becomes its own root band.
  for (const n of nodes) if (!placed.has(n.record_id)) place(n.record_id, 0, null);

  const outNodes = [];
  let maxX = 0, maxY = 0;
  for (const { node, x, y, depth } of placed.values()) {
    outNodes.push({ node, x, y, depth, isCurrent: !!node.is_current, onSpine: false });
    maxX = Math.max(maxX, x + CARD_W);
    maxY = Math.max(maxY, y + CARD_H);
  }

  // Ancestor chains, from the laid-out parent links (each node has ≤ 1 parent).
  const ancestorsOf = new Map();
  for (const { node } of placed.values()) {
    const chain = new Set();
    let cur = parentOf.get(node.record_id);
    while (cur != null && !chain.has(cur)) { chain.add(cur); cur = parentOf.get(cur); }
    ancestorsOf.set(node.record_id, chain);
  }

  // The trunk: the path from the current run up to its root. Highlighted brighter
  // so the eye follows "how did I get to this run" at a glance.
  const spine = new Set();
  const currentId = tree?.current_id != null && placed.has(tree.current_id)
    ? tree.current_id
    : (outNodes.find((n) => n.isCurrent)?.node.record_id ?? null);
  if (currentId != null) {
    spine.add(currentId);
    for (const a of (ancestorsOf.get(currentId) || [])) spine.add(a);
  }
  for (const on of outNodes) on.onSpine = spine.has(on.node.record_id);

  // Edges from the persisted graph, carrying the superseded flag through. An
  // edge is on the spine only when BOTH ends are (so the trunk lights, forks
  // off it stay quiet).
  const posOf = new Map([...placed.values()].map((p) => [p.node.record_id, p]));
  const outEdges = [];
  for (const e of (tree?.edges || [])) {
    const p = posOf.get(e.parent), c = posOf.get(e.child);
    if (!p || !c) continue;
    outEdges.push({
      parentId: e.parent, childId: e.child,
      superseded: !!e.superseded,
      onSpine: spine.has(e.parent) && spine.has(e.child),
      d: edgePath(p, c),
    });
  }

  return {
    nodes: outNodes, edges: outEdges,
    width: maxX + PAD, height: maxY + PAD,
    spine, ancestorsOf,
  };
}
