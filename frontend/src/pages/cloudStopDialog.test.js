import test from 'node:test';
import assert from 'node:assert/strict';
import {
  banHostLabel, canBanHost, machineLabel, stopConsequence, stopTitle,
} from './cloudStopDialog.js';

test('the consequence is not the same sentence for the two run kinds', () => {
  // A full-model run can lose its latest checkpoint outright; a LoRA run cannot.
  assert.match(stopConsequence(true), /permanently lost/);
  assert.match(stopConsequence(false), /still downloaded/);
  assert.doesNotMatch(stopConsequence(false), /permanently lost/);
});

test('the run is named the way the row names it', () => {
  assert.equal(stopTitle({ dataset_name: 'Elsa' }), 'Stop the cloud run for “Elsa”?');
  assert.equal(stopTitle({ run_name: 'r2' }), 'Stop the cloud run for “r2”?');
  assert.equal(stopTitle({ run_id: 9 }), 'Stop the cloud run for “run #9”?');
});

test('the machine is named from what the row actually shows', () => {
  assert.equal(machineLabel({ gpu: 'RTX 4090', vast_instance_id: '90001' }),
    'RTX 4090 · instance 90001');
  assert.equal(machineLabel({ gpu: 'RTX 4090' }), 'RTX 4090');
});

test('an unnameable machine is never offered for banning', () => {
  // Promising to exile "this machine" while being unable to say which one is a
  // promise the screen cannot keep.
  assert.equal(canBanHost({}), false);
  assert.equal(canBanHost(null), false);
  assert.equal(canBanHost({ gpu: 'RTX 4090' }), true);
});

test('a pod stuck booting is still a named machine you can ban', () => {
  // Status `provisioning` is set in the same write as gpu_name + instance id,
  // so the launch step "Renting the machine and booting the pod" already knows
  // which box it rented. Hiding the tick here would make the option vanish
  // exactly when a stuck boot is why you are stopping.
  assert.equal(canBanHost({
    status: 'provisioning', gpu: 'RTX 4090', vast_instance_id: '90001',
  }), true);
  assert.equal(
    machineLabel({ gpu: 'RTX 4090', vast_instance_id: '90001' }),
    'RTX 4090 · instance 90001');
});

test('the tick box says when NOT to tick it', () => {
  const { label, detail } = banHostLabel({ gpu: 'RTX 4090', vast_instance_id: '7' });
  assert.match(label, /not rent this machine again/i);
  assert.match(detail, /RTX 4090 · instance 7/);
  // A ban costs the user their next launch on that box: the sentence has to
  // keep a good machine out of it.
  assert.match(detail, /only if the machine itself was the problem/i);
  assert.match(detail, /says nothing about it/i);
  // And it names where the duration is decided rather than inventing one.
  assert.match(detail, /Settings ▸ Cloud/);
});
