/* What a `background: true` request is allowed to say out loud.
 *
 * connectionStatus.test.js already covers the store's arithmetic; this file
 * drives apiFetch itself, because the half that used to leak was NOT the store:
 * a poll whose request landed on a 503 skipped the store entirely and went
 * straight to toastRef.error, once every three seconds, for as long as the
 * server stayed unhappy.
 */
import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';

import { apiFetch, setToastRef, CONNECTION_BACK_MESSAGE } from './fetchClient.js';
import { getConnectionState, reportRequestFailure, resetConnectionStatus } from '../utils/connectionStatus.js';

const originalFetch = globalThis.fetch;

function response(status, body = {}) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: () => 'application/json' },
    json: async () => body,
  };
}

function recorder() {
  const seen = [];
  setToastRef({
    error: (m) => seen.push(['error', m]),
    warning: (m) => seen.push(['warning', m]),
    success: (m) => seen.push(['success', m]),
  });
  return seen;
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  setToastRef(null);
  resetConnectionStatus();
});

test('a background poll answered with 401/429/5xx stays silent', async () => {
  const seen = recorder();

  for (const status of [401, 429, 500, 503]) {
    globalThis.fetch = async () => response(status, { error: `failure-${status}` });
    await assert.rejects(
      apiFetch('/api/setup/runtime-readiness', { background: true }),
      new RegExp(`failure-${status}`),
      'the caller must still get the rejection — silence is about the toast, not the error',
    );
  }

  assert.deepEqual(seen, []);
});

test('a background poll that never reaches the server stays silent too', async () => {
  const seen = recorder();
  globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };

  for (let i = 0; i < 5; i += 1) {
    await assert.rejects(apiFetch('/api/train/activity', { background: true }), /Network error/);
  }

  assert.deepEqual(seen, []);
});

test('a foreground call keeps its actionable toast for every status', async () => {
  const cases = [
    [401, ['error', 'Session expired. Please log in again.']],
    [429, ['warning', 'Too many requests. Please wait a moment.']],
    [503, ['error', 'Server error. Please try again later.']],
  ];
  for (const [status, expected] of cases) {
    const seen = recorder();
    globalThis.fetch = async () => response(status, { error: 'nope' });
    await assert.rejects(apiFetch('/api/user-action'), /nope/);
    assert.deepEqual(seen, [expected], `status ${status}`);
    resetConnectionStatus();
  }
});

test('the poll that notices the server came back still announces it once', async () => {
  const seen = recorder();
  // The outage opened on a user action, so it was announced; nobody clicks
  // during an outage, so the poll is what closes it.
  reportRequestFailure({ background: false });
  globalThis.fetch = async () => response(200, { ok: true });

  assert.deepEqual(await apiFetch('/api/train/activity', { background: true }), { ok: true });
  assert.deepEqual(await apiFetch('/api/train/activity', { background: true }), { ok: true });

  assert.deepEqual(seen, [['success', CONNECTION_BACK_MESSAGE]]);
});

test('a background 503 closes the outage without becoming a toast', async () => {
  const seen = recorder();
  reportRequestFailure({ background: false });
  globalThis.fetch = async () => response(503, { error: 'still booting' });

  await assert.rejects(apiFetch('/api/capabilities', { background: true }), /still booting/);

  // Reachable again — that is worth the one recovery line, and nothing else.
  assert.deepEqual(seen, [['success', CONNECTION_BACK_MESSAGE]]);
});

test('cancelling a replaced library read preserves AbortError and does not invent an outage or recovery', async () => {
  const seen = recorder();
  const before = getConnectionState();
  const controller = new AbortController();
  const aborted = new DOMException('The request was replaced', 'AbortError');
  globalThis.fetch = async () => { controller.abort(); throw aborted; };
  await assert.rejects(apiFetch('/api/video-studio/reference-library', { signal: controller.signal, background: true }),
    (error) => error === aborted, 'the caller must retain the cancellation identity');
  assert.equal(getConnectionState(), before);
  globalThis.fetch = async () => response(200, { items: [] });
  await apiFetch('/api/video-studio/reference-library', { background: true });
  assert.deepEqual(seen, [], 'a successful replacement must not announce Back online');
});

test('an explicit abort reason stays a cancellation even when it is not named AbortError', async () => {
  const seen = recorder();
  const before = getConnectionState();
  const controller = new AbortController();
  const reason = new Error('Selection removed');
  controller.abort(reason);
  globalThis.fetch = async () => { throw reason; };
  await assert.rejects(apiFetch('/api/video-studio/reference-library/info', { signal: controller.signal }), (error) => error === reason);
  assert.equal(getConnectionState(), before); assert.deepEqual(seen, []);
});

test('a cancelled read neither grows nor closes an existing network outage', async () => {
  const seen = recorder();
  reportRequestFailure({ background: true });
  const before = getConnectionState();
  const aborted = new DOMException('Cancelled', 'AbortError');
  globalThis.fetch = async () => { throw aborted; };
  await assert.rejects(apiFetch('/api/read', { background: true }), (error) => error === aborted);
  assert.equal(getConnectionState(), before); assert.deepEqual(seen, []);
});
