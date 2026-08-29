import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cloudGroupsFrom,
  localRunIdentity,
  runIdentityOf,
  runRowDomId,
  runNumber,
  cloudNumber,
  runIdentityLabel,
} from './runIdentity.js';

/* ── One run number across the lineage surfaces ───────────────────────────────
   The graph card printed the CLOUD id and the inspector the RECORD id, so
   "#103" opened "Run #107" and nothing said they were the same run. */

test('runNumber is always the record id — a local run has no cloud id at all', () => {
  assert.equal(runNumber({ record_id: 107, source: 'cloud', run_id: 103 }), '#107');
  assert.equal(runNumber({ record_id: 12, source: 'local' }), '#12');
  assert.equal(runNumber(null), '#?');
  assert.equal(runNumber({}), '#?');
});

test('cloudNumber only answers for a cloud run that has one', () => {
  assert.equal(cloudNumber({ record_id: 107, source: 'cloud', run_id: 103 }), 103);
  assert.equal(cloudNumber({ record_id: 12, source: 'local', run_id: 103 }), null);
  assert.equal(cloudNumber({ record_id: 9, source: 'cloud' }), null);
  assert.equal(cloudNumber(null), null);
});

test('runIdentityLabel names the run once and the cloud run explicitly', () => {
  assert.equal(runIdentityLabel({ record_id: 107, source: 'cloud', run_id: 103 }),
    'Run #107 · cloud #103');
  assert.equal(runIdentityLabel({ record_id: 12, source: 'local' }), 'Run #12');
});

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
  // `id` stays the ADDRESS (Runs-row anchor); recordId is what the chip
  // prints — for a local tag they are the same number by construction.
  assert.deepEqual(localRunIdentity(cks),
    { source: 'local', id: 12, recordId: 12, cloudId: null });
  assert.equal(localRunIdentity([{ step: 500 }]), null);   // nothing tagged
  assert.equal(localRunIdentity([]), null);
  // A cloud-sourced set keeps the cloud id as its ADDRESS but displays the
  // record id the save was stamped with — ONE run number on every surface.
  assert.deepEqual(
    localRunIdentity([{ step: 100, run_id: 4, run_source: 'cloud', record_id: 9 }]),
    { source: 'cloud', id: 4, recordId: 9, cloudId: 4 });
  // A pre-registry cloud set has no record: recordId stays null and the chip
  // falls back to the cloud id, saying which kind of number it is.
  assert.deepEqual(
    localRunIdentity([{ step: 100, run_id: 4, run_source: 'cloud' }]),
    { source: 'cloud', id: 4, recordId: null, cloudId: 4 });
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
