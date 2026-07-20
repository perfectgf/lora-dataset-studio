/* Can a lineage run be removed from the graph? Pure predicate, no React, so the
   rule is unit-testable with node:test.

   Only a GONE run qualifies — one the graph already badges as having no
   checkpoints on disk (checkpoint_ready === false, the exact condition
   lineageChrome's SavesChip draws "gone" for). A run with checkpoints on disk
   (checkpoint_ready === true) is a recoverable run and is never offered for
   deletion; a run whose availability we couldn't determine (null/undefined — an
   active run or a scan that failed) is left alone rather than guessed removable. */
export function isRunDeletable(node) {
  return !!node && node.checkpoint_ready === false;
}

/* Drop a run from a {nodes, edges} lineage tree WITHOUT a refetch: remove the
   node, and every edge touching it. A child that resumed from the removed run
   loses its parent edge and re-roots on its own — mirroring the backend, which
   detaches (never deletes) a living child. Returns a new tree; the input is
   untouched. Robust to a missing/empty tree. */
export function removeRunFromTree(tree, recordId) {
  const nodes = Array.isArray(tree?.nodes) ? tree.nodes : [];
  const edges = Array.isArray(tree?.edges) ? tree.edges : [];
  const gone = (id) => id === recordId;
  return {
    ...tree,
    nodes: nodes.filter((n) => !gone(n.record_id)),
    edges: edges.filter((e) => !gone(e.parent) && !gone(e.child)),
  };
}
