/* Pure layout for the ◉ Graph view of a run's lineage — the second, showcase
   rendering of the same {nodes, edges} genealogy the ☰ List view draws
   (utils/lineageTree.js). Framework-free (no JSX, no d3) so the geometry is
   unit-testable with node:test; the SVG renderer lives in
   components/dataset/RunLineageGraph.jsx.

   Layout is a tidy left-to-right tree: the root sits on the left and each
   continuation flows one generation to the right, so a lineage reads as a
   timeline (run → continue → re-continue) with forks stacking vertically.
   Positions use the leaf-slot method — every leaf gets its own vertical band,
   each parent is centred over the span of its children — which never overlaps
   for the small trees a lineage is (3-10 runs) without pulling in a graph
   library. Defensive against a missing root, a cycle, or an orphaned edge,
   exactly like the list: every node is placed once, unreachable nodes become
   their own roots so nothing vanishes.

   Each run also carries its CHECKPOINTS as compact pills laid out in a wrapped
   row beneath the card. A continuation's run→run edge is anchored on the exact
   pill it resumed from (step === resumed_from), so the graph reads "this run
   started from THIS checkpoint"; when no pill matches (final save, superseded
   and gone, or a legacy run) the edge falls back to the parent card's edge. */

import { siblingSort } from './lineageTree.js';

// Card + spacing geometry, in SVG user units (1 unit ≈ 1px at scale 1). The
// renderer draws each run as a fixed-size card so the layout is deterministic.
export const CARD_W = 264;
export const CARD_H = 64;
export const H_GAP = 62;   // gap between one generation and the next (→)
export const V_GAP = 26;   // gap between sibling bands (↓)
export const PAD = 22;     // breathing room around the whole tree

// Checkpoint pills: a compact wrapped grid under the card. Sized so a run with
// ~13 saves stays a tidy few rows (screenshot-able) instead of an exploding
// tree — the saves are satellites of the run, never their own generations.
export const PILL_W = 60;
export const PILL_H = 20;
// 🔍 Big-preview mode: the pill becomes a ComfyUI-style preview tile so several
// checkpoints' generated images can be compared at a glance without opening each.
// Square-ish, ~2 tiles per card row — the layout below adapts to these.
export const PILL_W_BIG = 128;
export const PILL_H_BIG = 132;
export const PILL_GAP = 6;         // between pills, both axes
export const PILL_TOP_GAP = 8;     // card bottom → first pill row
export const PILLS_PER_ROW = Math.max(1, Math.floor((CARD_W + PILL_GAP) / (PILL_W + PILL_GAP)));

const COL = CARD_W + H_GAP;   // centre-to-centre horizontal step per depth

/** Per-mode pill geometry. Compact (default) keeps the satellite pills; big turns
 *  each into a large preview tile. `perRow` re-derives from the tile width so the
 *  wrapped grid still fits the card. Threaded through the layout so cells never
 *  overlap in either mode. */
export function graphMetrics(bigPreviews = false) {
  const pillW = bigPreviews ? PILL_W_BIG : PILL_W;
  const pillH = bigPreviews ? PILL_H_BIG : PILL_H;
  const perRow = Math.max(1, Math.floor((CARD_W + PILL_GAP) / (pillW + PILL_GAP)));
  return { pillW, pillH, perRow };
}

/** Rows a run's pills wrap into, and the pixel height of that block (0 saves =
 *  no block). */
function pillRows(n, m) { return n > 0 ? Math.ceil(n / m.perRow) : 0; }
function pillsBlockH(n, m) {
  const rows = pillRows(n, m);
  return rows ? rows * m.pillH + (rows - 1) * PILL_GAP : 0;
}
/** A run cell's full height: the card, plus its wrapped pills when it has any. */
function cellHeight(nCk, m) {
  const block = pillsBlockH(nCk, m);
  return CARD_H + (block ? PILL_TOP_GAP + block : 0);
}
/** Pill position (top-left) relative to the run card's top-left, by index. */
function pillOffset(i, m) {
  const col = i % m.perRow, row = Math.floor(i / m.perRow);
  return {
    dx: col * (m.pillW + PILL_GAP),
    dy: CARD_H + PILL_TOP_GAP + row * (m.pillH + PILL_GAP),
  };
}

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

/** Cubic-bezier path from a point on the parent (card edge OR a checkpoint pill)
 *  to a child card's left edge, with horizontal tangents so the curve leaves and
 *  arrives flat — the smooth "flowing" connector, not a kinked polyline. */
function edgePath(x1, y1, x2, y2) {
  const mx = x1 + (x2 - x1) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

/**
 * Lay out a {root_id, current_id, nodes, edges} lineage for the graph view.
 * Returns:
 *   nodes: [{ node, x, y, depth, isCurrent, onSpine, cellH, checkpoints }]
 *          — top-left of each card; `checkpoints` are the pills in ABSOLUTE
 *            coords: [{ step, final, present, download_url, x, y, w, h,
 *            isResumeSource }]
 *   edges: [{ parentId, childId, d, superseded, onSpine, anchoredStep }]
 *          — `d` is an SVG path; `anchoredStep` is the pill step it starts from
 *            (null when it leaves the parent card edge)
 *   width, height     — content bounds (PAD included) for the viewBox
 *   spine             — Set of node ids on the root→current path (the trunk)
 *   ancestorsOf       — Map id → Set of that node's ancestor ids (self excluded),
 *                       so hover can light a node's whole path back to the root
 * Empty / malformed payloads return an empty, safe shape.
 */
export function buildLineageGraph(tree, { bigPreviews = false } = {}) {
  const m = graphMetrics(bigPreviews);
  const empty = { nodes: [], edges: [], width: 0, height: 0,
    spine: new Set(), ancestorsOf: new Map() };
  const { nodes, byId, childrenOf } = indexTree(tree);
  if (!nodes.length) return empty;

  const ckOf = (id) => (Array.isArray(byId.get(id)?.checkpoints) ? byId.get(id).checkpoints : []);

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

  const placed = new Map();   // id -> { node, x, y, depth, cellH }
  const parentOf = new Map(); // id -> parent id (as laid out)
  let nextY = PAD;            // running top for the next leaf band

  // Post-order walk: a leaf takes the next free vertical band (its own cell
  // height), a parent centres its CARD on the span of its children's cards. A
  // seen-set breaks cycles and shared-child anomalies so each node is placed
  // exactly once. A parent's card centres on its children (a full generation to
  // the RIGHT), but the parent still owns a tall PILL BLOCK in its own column —
  // and other roots/siblings share that column. So after centring a parent we
  // push nextY below the BOTTOM of the parent's whole cell (card + pill block),
  // otherwise a parent with more checkpoints than its child gets overlapped by
  // the next node placed in the same column (the #76-over-#81 bug).
  const place = (id, depth, parentId) => {
    if (placed.has(id)) return placed.get(id).cy;
    const node = byId.get(id);
    if (!node) return null;
    const cellH = cellHeight(ckOf(id).length, m);
    const slot = { node, depth, x: PAD + depth * COL, y: 0, cellH, cy: 0 };
    placed.set(id, slot);                       // reserve (cycle guard)
    if (parentId != null) parentOf.set(id, parentId);
    const kids = (childrenOf.get(id) || []).filter((cid) => !placed.has(cid));
    let cardCenter;
    if (!kids.length) {
      slot.y = nextY;
      cardCenter = slot.y + CARD_H / 2;
      nextY += cellH + V_GAP;                   // advance past this whole cell
    } else {
      const centers = kids.map((cid) => place(cid, depth + 1, id)).filter((c) => c != null);
      cardCenter = centers.length
        ? (Math.min(...centers) + Math.max(...centers)) / 2
        : (nextY += 0, nextY);
      slot.y = cardCenter - CARD_H / 2;
      // Reserve the parent's OWN cell height too: its pill block extends below
      // the card and must not be overlapped by the next node in this column.
      nextY = Math.max(nextY, slot.y + cellH + V_GAP);
    }
    slot.cy = cardCenter;
    return cardCenter;
  };
  for (const rid of roots) place(rid, 0, null);
  // Any node still unplaced (unreachable) becomes its own root band.
  for (const n of nodes) if (!placed.has(n.record_id)) place(n.record_id, 0, null);

  // Absolute pill boxes per node, and a step → pill index lookup for edge anchoring.
  const pillsByNode = new Map();   // id -> [{ step, final, present, download_url, x, y, w, h }]
  for (const { node, x, y } of placed.values()) {
    const cks = ckOf(node.record_id).map((c, i) => {
      const { dx, dy } = pillOffset(i, m);
      return { step: c.step, final: !!c.final, present: c.present !== false,
        download_url: c.download_url || null, filename: c.filename,
        // Lab inline generation: whether this checkpoint has a deployed LoRA the
        // engine can preview, and its generated preview (url + async status).
        testable: c.testable === true,
        preview_url: c.preview_url || null, preview_status: c.preview_status || null,
        x: x + dx, y: y + dy, w: m.pillW, h: m.pillH, isResumeSource: false };
    });
    pillsByNode.set(node.record_id, cks);
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
    : ([...placed.values()].find((p) => p.node.is_current)?.node.record_id ?? null);
  if (currentId != null) {
    spine.add(currentId);
    for (const a of (ancestorsOf.get(currentId) || [])) spine.add(a);
  }

  // Edges from the persisted graph, carrying the superseded flag through and
  // anchored on the parent pill the child resumed from when one matches.
  const outEdges = [];
  for (const e of (tree?.edges || [])) {
    const p = placed.get(e.parent), c = placed.get(e.child);
    if (!p || !c) continue;
    const x2 = c.x, y2 = c.y + CARD_H / 2;
    // The child's resume step: prefer the edge's, fall back to the child node's.
    const step = e.resumed_from != null ? e.resumed_from : c.node.resumed_from;
    const pills = pillsByNode.get(e.parent) || [];
    const anchor = step != null ? pills.find((pl) => pl.step === step) : undefined;
    let x1, y1, anchoredStep = null;
    if (anchor) {
      anchor.isResumeSource = true;
      x1 = anchor.x + anchor.w; y1 = anchor.y + anchor.h / 2;
      anchoredStep = anchor.step;
    } else {
      x1 = p.x + CARD_W; y1 = p.y + CARD_H / 2;
    }
    outEdges.push({
      parentId: e.parent, childId: e.child,
      superseded: !!e.superseded,
      onSpine: spine.has(e.parent) && spine.has(e.child),
      anchoredStep,
      d: edgePath(x1, y1, x2, y2),
    });
  }

  const outNodes = [];
  let maxX = 0, maxY = 0;
  for (const { node, x, y, depth, cellH } of placed.values()) {
    outNodes.push({ node, x, y, depth, cellH,
      isCurrent: !!node.is_current, onSpine: spine.has(node.record_id),
      checkpoints: pillsByNode.get(node.record_id) || [] });
    maxX = Math.max(maxX, x + CARD_W);
    maxY = Math.max(maxY, y + cellH);
  }

  return {
    nodes: outNodes, edges: outEdges,
    width: maxX + PAD, height: maxY + PAD,
    spine, ancestorsOf,
  };
}
