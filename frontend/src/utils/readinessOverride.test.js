import assert from 'node:assert/strict';
import test from 'node:test';
import { readinessSignature, overrideAck } from './readinessOverride.js';

const withFails = (verdict, canOverride, fails) => ({
  verdict, can_override: canOverride,
  checks: [{ id: 'triage', status: 'warn' }, ...fails],
});

test('overrideAck is true only when the override is offered AND the box is ticked', () => {
  const data = withFails('blocked', true, [{ id: 'images', status: 'fail', bypassable: true }]);
  assert.equal(overrideAck(data, true), true);
  assert.equal(overrideAck(data, false), false);
});

test('overrideAck can never be true for a physical impossibility (can_override false)', () => {
  const data = withFails('blocked', false, [{ id: 'images', status: 'fail', bypassable: false }]);
  assert.equal(overrideAck(data, true), false);   // ticked but not offerable
  assert.equal(overrideAck(null, true), false);
});

test('signature changes when the blocking set changes → the ack resets', () => {
  const few = withFails('blocked', true, [{ id: 'images', status: 'fail', bypassable: true }]);
  const prompt = withFails('blocked', false,
    [{ id: 'slider_prompts', status: 'fail', bypassable: false }]);
  assert.notEqual(readinessSignature(few), readinessSignature(prompt));
});

test('signature is stable when only background warnings move (no phantom reset)', () => {
  const base = { verdict: 'blocked', can_override: true,
    checks: [{ id: 'images', status: 'fail', bypassable: true }, { id: 'triage', status: 'warn' }] };
  const moreWarnings = { verdict: 'blocked', can_override: true,
    checks: [{ id: 'images', status: 'fail', bypassable: true },
      { id: 'triage', status: 'warn' }, { id: 'duplicates', status: 'warn' }] };
  assert.equal(readinessSignature(base), readinessSignature(moreWarnings));
});

test('a ready dataset offers no override and has an empty signature-ish state', () => {
  const ready = { verdict: 'ready', can_override: false, checks: [{ id: 'images', status: 'ok' }] };
  assert.equal(overrideAck(ready, true), false);
  assert.equal(readinessSignature(null), '');
});
