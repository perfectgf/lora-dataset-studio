import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isRunDeletable, removeRunFromTree } from './runDeletable.js';

test('isRunDeletable: only a gone run (checkpoint_ready === false) qualifies', () => {
  assert.equal(isRunDeletable({ checkpoint_ready: false }), true);   // gone
  assert.equal(isRunDeletable({ checkpoint_ready: true }), false);   // on disk
  assert.equal(isRunDeletable({ checkpoint_ready: null }), false);   // scan unknown
  assert.equal(isRunDeletable({ saves: 0 }), false);                 // undefined readiness
  assert.equal(isRunDeletable(null), false);
});

test('removeRunFromTree: drops the node and every edge touching it', () => {
  const tree = {
    root_id: 1, current_id: 2,
    nodes: [{ record_id: 1 }, { record_id: 2 }, { record_id: 3 }],
    edges: [
      { parent: 1, child: 2 },
      { parent: 2, child: 3 },
    ],
  };
  const out = removeRunFromTree(tree, 2);
  assert.deepEqual(out.nodes.map((n) => n.record_id), [1, 3]);
  assert.deepEqual(out.edges, []);            // both edges touched #2
  assert.equal(out.root_id, 1);               // rest of the tree preserved
  // input untouched
  assert.equal(tree.nodes.length, 3);
});

test('removeRunFromTree: a child re-roots when its parent is removed', () => {
  const tree = {
    nodes: [{ record_id: 1 }, { record_id: 2 }],
    edges: [{ parent: 1, child: 2 }],
  };
  const out = removeRunFromTree(tree, 1);
  assert.deepEqual(out.nodes.map((n) => n.record_id), [2]);
  assert.deepEqual(out.edges, []);            // #2 keeps existing, edge gone
});

test('removeRunFromTree: tolerates a missing/empty tree', () => {
  assert.deepEqual(removeRunFromTree(null, 1), { nodes: [], edges: [] });
  assert.deepEqual(removeRunFromTree({}, 1), { nodes: [], edges: [] });
});
