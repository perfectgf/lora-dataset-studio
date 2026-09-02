import test from 'node:test'
import assert from 'node:assert/strict'

import { shouldEjectOnLoadError, staleNote } from './videoDatasetLoad.js'

// The rule the page owes, as values. It had none of these before: the
// condition lived inline in a callback, unreachable by node --test, and the
// whole "a failed refresh must not take the workspace away" fix rested on
// re-reading it.

test('a transient failure AFTER a successful load keeps the workspace', () => {
  for (const err of [
    { status: 500, message: 'Server error' },
    { status: 503, message: 'Service unavailable' },
    { status: 429, message: 'Too many requests' },
    new Error('Network error'),
    null, undefined,
  ]) {
    assert.equal(shouldEjectOnLoadError(err, true), false,
      `${JSON.stringify(err?.message)} must not eject a page that has a payload`)
  }
})

test('a definitive 404 ejects even with a payload on screen', () => {
  assert.equal(shouldEjectOnLoadError({ status: 404, message: 'video dataset 9 not found' }, true), true)
  assert.equal(shouldEjectOnLoadError({ status: '404' }, true), true)
})

test('the very first load has nothing to keep, so any failure ejects', () => {
  assert.equal(shouldEjectOnLoadError({ status: 500 }, false), true)
  assert.equal(shouldEjectOnLoadError(new Error('Network error'), false), true)
  assert.equal(shouldEjectOnLoadError(undefined, false), true)
})

test('the message is never consulted — a proxy page saying "not found" is not a 404', () => {
  // The old predicate matched /not found/i on the text and would have ejected a
  // user holding caption drafts over an HTML error page from a proxy.
  assert.equal(shouldEjectOnLoadError({ status: 502, message: 'upstream not found' }, true), false)
  assert.equal(shouldEjectOnLoadError({ message: 'not found' }, true), false)
})

test('the stale line names the status when there is one, and says so when there is none', () => {
  assert.match(staleNote({ status: 500 }), /answered 500/)
  assert.match(staleNote(new Error('Network error')), /did not reach the server/)
  assert.match(staleNote(undefined), /did not reach the server/)
  for (const note of [staleNote({ status: 503 }), staleNote(null)]) {
    assert.match(note, /Showing the last loaded state/)
  }
})
