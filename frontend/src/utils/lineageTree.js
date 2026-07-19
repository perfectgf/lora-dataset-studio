/* Pure layout helpers behind the 🌳 Runs genealogy tree. Framework-free (no
   JSX) so the shape logic is unit-testable with node:test; the presentational
   tree lives in components/dataset/RunLineageTree.jsx.

   The backend (/dataset/train/runs/<id>/lineage) returns a flat {nodes, edges}
   graph linked by parent_record_id. `buildLineageRows` turns it into an ordered,
   depth-tagged PRE-ORDER list — an indented tree (like a file tree), the lightest
   rendering that still reads as a genealogy: no graph library, one row per run,
   children indented under their parent. */

// Order siblings oldest-first so a lineage reads top-to-bottom in the order the
// runs actually happened (created_at, record_id as the stable tiebreaker).
function siblingSort(a, b) {
  const ta = a.created_at || '';
  const tb = b.created_at || '';
  if (ta !== tb) return ta < tb ? -1 : 1;
  return (a.record_id || 0) - (b.record_id || 0);
}

/**
 * Flatten a {root_id, nodes, edges} lineage into pre-order rows for an indented
 * tree. Each row: { node, depth, isLast, hasChildren }. `depth` = distance from
 * the root (0 = root). Defensive against a missing root or a cycle: every node
 * is emitted at most once, and an unreachable node (orphaned by a broken edge)
 * is appended at depth 0 so nothing silently vanishes.
 */
export function buildLineageRows(tree) {
  const nodes = Array.isArray(tree?.nodes) ? tree.nodes : [];
  if (!nodes.length) return [];
  const byId = new Map(nodes.map((n) => [n.record_id, n]));
  const childrenOf = new Map();
  for (const e of (tree.edges || [])) {
    if (!childrenOf.has(e.parent)) childrenOf.set(e.parent, []);
    childrenOf.get(e.parent).push(e.child);
  }
  const rows = [];
  const seen = new Set();
  const walk = (id, depth, isLast) => {
    const node = byId.get(id);
    if (!node || seen.has(id)) return;
    seen.add(id);
    const kids = (childrenOf.get(id) || [])
      .map((cid) => byId.get(cid)).filter(Boolean).sort(siblingSort)
      .map((n) => n.record_id);
    rows.push({ node, depth, isLast, hasChildren: kids.length > 0 });
    kids.forEach((cid, i) => walk(cid, depth + 1, i === kids.length - 1));
  };
  const rootId = tree.root_id != null && byId.has(tree.root_id)
    ? tree.root_id
    : (nodes.find((n) => n.parent_record_id == null) || nodes[0]).record_id;
  walk(rootId, 0, true);
  // Any node not reached (broken/foreign edge) still gets shown, as its own root.
  for (const n of nodes) {
    if (!seen.has(n.record_id)) walk(n.record_id, 0, true);
  }
  return rows;
}

/** Short "resumed from step N" caption for a node, or '' for a root / a node
 *  whose resume step is unknown (legacy). */
export function resumeCaption(node) {
  if (node?.resumed_from == null) return '';
  return `resumed from step ${node.resumed_from}`;
}

/** Whether the whole payload is a lone run (no tree worth drawing). */
export function isSingleRun(tree) {
  return !!tree?.single || (Array.isArray(tree?.nodes) && tree.nodes.length < 2);
}
