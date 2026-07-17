import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cloudGroupsFrom,
  localRunIdentity,
  runIdentityOf,
  runRowDomId,
} from './runIdentity.js';

test('runRowDomId keys cloud vs local rows distinctly, null when unaddressable', () => {
  assert.equal(runRowDomId('cloud', 49), 'run-cloud-49');
  assert.equal(runRowDomId('local', 12), 'run-local-12');
  assert.equal(runRowDomId('cloud', null), null);
  assert.equal(runRowDomId('local', undefined), null);
});

test('runIdentityOf resolves cloud run_id and local record_id, null otherwise', () => {
  assert.deepEqual(runIdentityOf({ source: 'cloud', run_id: 49 }), { source: 'cloud', id: 49 });
  // a cloud row is recognised by run_id even without an explicit source
  assert.deepEqual(runIdentityOf({ run_id: 7 }), { source: 'cloud', id: 7 });
  assert.deepEqual(runIdentityOf({ source: 'local', record_id: 12 }), { source: 'local', id: 12 });
  assert.equal(runIdentityOf({ source: 'local' }), null);   // legacy row, no id
  assert.equal(runIdentityOf(null), null);
});

test('localRunIdentity picks the newest checkpoint that carries a run id', () => {
  const cks = [
    { step: 500, run_id: 12, run_source: 'local' },
    { step: 2500, run_id: 12, run_source: 'local' },
    { step: 1000 },   // untagged legacy file — ignored
  ];
  assert.deepEqual(localRunIdentity(cks), { source: 'local', id: 12 });
  assert.equal(localRunIdentity([{ step: 500 }]), null);   // nothing tagged
  assert.equal(localRunIdentity([]), null);
  // a cloud-sourced local record maps to the cloud family
  assert.deepEqual(
    localRunIdentity([{ step: 100, run_id: 4, run_source: 'cloud' }]),
    { source: 'cloud', id: 4 });
});

test('cloudGroupsFrom prefers the server groups payload verbatim', () => {
  const groups = [{ run_id: 49, checkpoints: [{ step: 500 }] }];
  assert.equal(cloudGroupsFrom({ cloud_checkpoint_groups: groups }), groups);
});

test('cloudGroupsFrom rebuilds per-run groups from a legacy flat list', () => {
  const flat = [
    { run_id: 50, step: 500, version: 2, active: false, train_type: 'krea' },
    { run_id: 49, step: 2500, version: 1, active: false, train_type: 'krea' },
    { run_id: 50, step: 2500, version: 2, active: false, train_type: 'krea' },
  ];
  const groups = cloudGroupsFrom({ cloud_checkpoints: flat });
  assert.equal(groups.length, 2);                       // one group per run
  const g50 = groups.find((g) => g.run_id === 50);
  const g49 = groups.find((g) => g.run_id === 49);
  assert.equal(g50.checkpoints.length, 2);
  assert.equal(g49.checkpoints.length, 1);
  assert.equal(g50.version, 2);
  assert.equal(g50.train_type, 'krea');
  // every checkpoint in a group belongs to that run — no cross-mixing
  assert.ok(g50.checkpoints.every((c) => c.run_id === 50));
});

test('cloudGroupsFrom returns [] with no cloud data', () => {
  assert.deepEqual(cloudGroupsFrom({}), []);
  assert.deepEqual(cloudGroupsFrom(null), []);
});
