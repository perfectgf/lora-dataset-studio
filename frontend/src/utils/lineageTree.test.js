import test from 'node:test';
import assert from 'node:assert/strict';
import { buildLineageRows, resumeCaption, isSingleRun } from './lineageTree.js';

const chain = {
  root_id: 1,
  nodes: [
    { record_id: 1, parent_record_id: null, created_at: '2026-07-01T00:00:00' },
    { record_id: 2, parent_record_id: 1, resumed_from: 1000, created_at: '2026-07-02T00:00:00' },
    { record_id: 3, parent_record_id: 2, resumed_from: 1500, created_at: '2026-07-03T00:00:00', is_current: true },
  ],
  edges: [
    { parent: 1, child: 2, resumed_from: 1000 },
    { parent: 2, child: 3, resumed_from: 1500 },
  ],
};

test('buildLineageRows flattens a chain root-first with increasing depth', () => {
  const rows = buildLineageRows(chain);
  assert.deepEqual(rows.map((r) => r.node.record_id), [1, 2, 3]);
  assert.deepEqual(rows.map((r) => r.depth), [0, 1, 2]);
  assert.equal(rows[0].hasChildren, true);
  assert.equal(rows[2].hasChildren, false);
});

test('buildLineageRows orders sibling branches oldest-first under the shared parent', () => {
  const forked = {
    root_id: 1,
    nodes: [
      { record_id: 1, parent_record_id: null, created_at: '2026-07-01T00:00:00' },
      { record_id: 3, parent_record_id: 1, resumed_from: 500, created_at: '2026-07-03T00:00:00' },
      { record_id: 2, parent_record_id: 1, resumed_from: 1000, created_at: '2026-07-02T00:00:00' },
    ],
    edges: [
      { parent: 1, child: 3, resumed_from: 500 },
      { parent: 1, child: 2, resumed_from: 1000 },
    ],
  };
  const rows = buildLineageRows(forked);
  // root, then the older branch (id 2, created 07-02) before the newer (id 3)
  assert.deepEqual(rows.map((r) => r.node.record_id), [1, 2, 3]);
  assert.deepEqual(rows.map((r) => r.depth), [0, 1, 1]);
});

test('buildLineageRows never loops on a cyclic edge and shows every node once', () => {
  const cyclic = {
    root_id: 1,
    nodes: [
      { record_id: 1, parent_record_id: 2 },
      { record_id: 2, parent_record_id: 1 },
    ],
    edges: [{ parent: 1, child: 2 }, { parent: 2, child: 1 }],
  };
  const rows = buildLineageRows(cyclic);
  assert.equal(rows.length, 2);
  assert.deepEqual([...new Set(rows.map((r) => r.node.record_id))].sort(), [1, 2]);
});

test('buildLineageRows tolerates empty / missing payloads', () => {
  assert.deepEqual(buildLineageRows(null), []);
  assert.deepEqual(buildLineageRows({ nodes: [] }), []);
});

test('resumeCaption reads the resume step, empty for a root', () => {
  assert.equal(resumeCaption({ resumed_from: 1500 }), 'resumed from step 1500');
  assert.equal(resumeCaption({ resumed_from: null }), '');
  assert.equal(resumeCaption({}), '');
});

test('isSingleRun flags a lone run', () => {
  assert.equal(isSingleRun({ single: true, nodes: [{ record_id: 1 }] }), true);
  assert.equal(isSingleRun({ nodes: [{ record_id: 1 }] }), true);
  assert.equal(isSingleRun(chain), false);
});
