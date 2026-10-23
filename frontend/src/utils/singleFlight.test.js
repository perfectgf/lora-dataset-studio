import { test } from 'node:test';
import assert from 'node:assert/strict';
import { singleFlight } from './singleFlight.js';

test('a second click shares the pending action, including during preparation', async () => {
  const run = singleFlight();
  let release;
  let calls = 0;
  const action = async () => {
    calls++;
    await new Promise((resolve) => { release = resolve; });
    return 'accepted';
  };
  const first = run(action);
  assert.equal(run(action), first);
  await Promise.resolve();
  assert.equal(run(action), first);
  assert.equal(calls, 1);
  release();
  assert.equal(await first, 'accepted');
  assert.equal(await run(() => 'next'), 'next');
});

test('a rejected or cancelled preparation permits another attempt', async () => {
  const run = singleFlight();
  await assert.rejects(run(() => { throw new Error('offline'); }), /offline/);
  assert.equal(await run(() => null), null);
  assert.equal(await run(() => 'retry'), 'retry');
});
