import assert from 'node:assert/strict';
import test from 'node:test';

import { refreshDatasetIfActive } from './datasetRefresh.js';

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test('pending refresh A is ignored after opening dataset B', async () => {
  const response = deferred();
  let activeId = 1;
  let data = { id: 1, name: 'A before refresh' };
  const open = (id, payload) => {
    activeId = id;
    data = payload;
  };

  const pendingRefresh = refreshDatasetIfActive({
    datasetId: 1,
    getActiveDatasetId: () => activeId,
    request: () => response.promise,
    commitData: (next) => { data = next; },
    clearActiveDataset: () => { activeId = null; },
  });

  open(2, { id: 2, name: 'B' });
  response.resolve({ ok: true, status: 200, json: async () => ({ id: 1, name: 'late A' }) });

  assert.deepEqual(await pendingRefresh, { status: 'stale' });
  assert.deepEqual(data, { id: 2, name: 'B' });
  assert.equal(activeId, 2);
});

test('a stale 404 cannot close the newly active dataset', async () => {
  const response = deferred();
  let activeId = 1;
  let data = { id: 1 };
  let clears = 0;

  const pendingRefresh = refreshDatasetIfActive({
    datasetId: 1,
    getActiveDatasetId: () => activeId,
    request: () => response.promise,
    commitData: (next) => { data = next; },
    clearActiveDataset: () => { clears += 1; activeId = null; },
  });

  activeId = 2;
  data = { id: 2 };
  response.resolve({ ok: false, status: 404 });

  assert.deepEqual(await pendingRefresh, { status: 'stale' });
  assert.deepEqual(data, { id: 2 });
  assert.equal(activeId, 2);
  assert.equal(clears, 0);
});

test('the active dataset still commits normally', async () => {
  let data = null;
  const result = await refreshDatasetIfActive({
    datasetId: 7,
    getActiveDatasetId: () => 7,
    request: async () => ({ ok: true, status: 200, json: async () => ({ id: 7 }) }),
    commitData: (next) => { data = next; },
    clearActiveDataset: () => {},
  });

  assert.equal(result.status, 'applied');
  assert.deepEqual(data, { id: 7 });
});
