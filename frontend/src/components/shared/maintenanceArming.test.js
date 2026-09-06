import assert from 'node:assert/strict';
import test, { beforeEach } from 'node:test';
import { ARM_MS, ARM_SETTLE_MS, armFrom, armedUntil, consume, disarm, isArmed, resetArmingForTests } from './maintenanceArming.js';

beforeEach(() => resetArmingForTests());
const T0 = 1_000_000;

test('only a refusal carrying the matching flag arms, whatever else the body says', () => {
  assert.equal(armFrom('free', { error: 'a LoRA training is running', can_interrupt: false }, T0), false);
  assert.equal(armFrom('free', { error: 'network', status: 502 }, T0), false);
  assert.equal(armFrom('free', null, T0), false);
  assert.equal(armFrom('free', { can_force: true, job_id: 'j' }, T0), false, 'the other button\'s flag');
  assert.equal(armFrom('restart', { can_interrupt: true, job_id: 'j' }, T0), false);
  assert.equal(armFrom('nothing', { can_interrupt: true }, T0), false);
  assert.equal(isArmed('free', T0 + ARM_SETTLE_MS), false);
  assert.equal(consume('free', T0 + ARM_SETTLE_MS), null);
  assert.equal(armFrom('free', { can_interrupt: true, job_id: 'job-own' }, T0), true);
  assert.equal(armFrom('restart', { can_force: true }, T0), true);
});

test('a press within the settle window is a double-click, not the second press — and spends the arming', () => {
  armFrom('free', { can_interrupt: true, job_id: 'job-own' }, T0);
  assert.equal(isArmed('free', T0 + 200), false);
  assert.equal(consume('free', T0 + 200), null);
  assert.equal(isArmed('free', T0 + ARM_SETTLE_MS + 1), false, 'spent by the double-click; the next refusal re-arms');
});

test('the second press within the minute carries the job the offer named, once', () => {
  armFrom('free', { can_interrupt: true, job_id: 'job-own' }, T0);
  assert.equal(isArmed('free', T0 + ARM_SETTLE_MS), true);
  assert.equal(armedUntil('free'), T0 + ARM_MS);
  assert.deepEqual(consume('free', T0 + 5000), { jobId: 'job-own' });
  assert.equal(consume('free', T0 + 5001), null, 'consumed');
  armFrom('restart', { can_force: true }, T0);
  assert.deepEqual(consume('restart', T0 + 5000), { jobId: null }, 'an offer that named no job');
});

test('an offer expires with the minute', () => {
  armFrom('free', { can_interrupt: true, job_id: 'job-own' }, T0);
  assert.equal(isArmed('free', T0 + ARM_MS - 1), true);
  assert.equal(isArmed('free', T0 + ARM_MS), false);
  assert.equal(consume('free', T0 + ARM_MS), null);
});

test('the two buttons never arm each other, and one store serves every mount', () => {
  armFrom('free', { can_interrupt: true, job_id: 'job-own' }, T0);
  assert.equal(isArmed('restart', T0 + 5000), false);
  assert.equal(consume('restart', T0 + 5000), null, 'the other button\'s press');
  assert.equal(isArmed('free', T0 + 5000), true, 'left in place by the other button');
  armFrom('restart', { can_force: true, job_id: 'job-own' }, T0);
  assert.deepEqual(consume('free', T0 + 5000), { jobId: 'job-own' });
  assert.deepEqual(consume('restart', T0 + 6000), { jobId: 'job-own' });
  armFrom('free', { can_interrupt: true, job_id: 'job-2' }, T0);
  disarm('free');
  assert.equal(isArmed('free', T0 + 5000), false);
});

test('a re-arm restarts the window and the settle guard', () => {
  armFrom('free', { can_interrupt: true, job_id: 'job-a' }, T0);
  armFrom('free', { can_interrupt: true, job_id: 'job-b' }, T0 + 30_000);
  assert.equal(isArmed('free', T0 + 30_000 + 100), false);
  assert.deepEqual(consume('free', T0 + 30_000 + ARM_SETTLE_MS), { jobId: 'job-b' });
});
