import test from 'node:test';
import assert from 'node:assert/strict';
import { externalMembersOf, externalEdgesFor } from './canvasExternalEdges.js';

const IMG = (extra) => ({ imageId: 7, x: 10, y: 20, w: 100, h: 80, datasetId: 1,
  image: { extra_loras: extra } });
const LANES = [{ datasetId: 1, x: 1000, graphY: 500 }];
const NODE = { filename: 'Krea\\Detail.safetensors', strength: 0.7, x: 40, y: 60 };
const KEY = 'ext:krea/detail.safetensors';
const BOXES = new Map([[KEY, { x: 40, y: 60, w: 172, h: 56 }]]);

test('externalMembersOf keeps only external entries, never throws', () => {
  const raw = JSON.stringify([
    { filename: 'a.safetensors', strength: 1, external: true },
    { filename: 'b.safetensors', strength: 1, combined: true },
    { filename: 'c.safetensors', strength: 1 },
  ]);
  assert.deepEqual(externalMembersOf({ extra_loras: raw }).map((e) => e.filename),
    ['a.safetensors']);
  assert.deepEqual(externalMembersOf({ extra_loras: '{broken' }), []);
  assert.deepEqual(externalMembersOf({}), []);
  assert.deepEqual(externalMembersOf(null), []);
});

test('an edge joins the node box to the image, world coords, external flag', () => {
  const raw = JSON.stringify([{ filename: 'krea/detail.safetensors', strength: 0.7,
    external: true }]);
  const edges = externalEdgesFor([IMG(raw)], LANES, [NODE], BOXES);
  assert.equal(edges.length, 1);
  const e = edges[0];
  assert.equal(e.external, true);
  assert.equal(e.childId, 'img:7');
  assert.equal(e.parentId, KEY);
  assert.equal(e.x1, 40 + 172);          // node right edge
  assert.equal(e.y1, 60 + 56 / 2);       // node vertical centre
  assert.equal(e.x2, 1000 + 10);         // lane.x + image.x
  assert.equal(e.y2, 500 + 20 + 80 / 2); // lane.graphY + image.y + h/2
  assert.ok(typeof e.d === 'string' && e.d.length > 0);
});

test('matching is separator- and case-insensitive both ways', () => {
  const raw = JSON.stringify([{ filename: 'KREA\\Detail.SAFETENSORS', external: true }]);
  assert.equal(externalEdgesFor([IMG(raw)], LANES, [NODE], BOXES).length, 1);
});

test('no node on the board, no measured box, or no external entry → no edge', () => {
  const raw = JSON.stringify([{ filename: 'krea/detail.safetensors', external: true }]);
  assert.equal(externalEdgesFor([IMG(raw)], LANES, [], BOXES).length, 0);
  assert.equal(externalEdgesFor([IMG(raw)], LANES, [NODE], new Map()).length, 0);
  assert.equal(externalEdgesFor([IMG(null)], LANES, [NODE], BOXES).length, 0);
});
