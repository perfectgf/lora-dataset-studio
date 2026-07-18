import test from 'node:test'
import assert from 'node:assert/strict'

import { openCollapsedAncestors, resolveFocusTarget } from '../src/help/revealTarget.js'

// Minimal fake DOM: nodes carry only what the helper reads, and the document is
// injectable — no jsdom needed.
const node = (props = {}) => ({ tagName: 'DIV', open: undefined, parentElement: null, ...props })
const doc = ({ byId = {}, gates = [] } = {}) => ({
  getElementById: (id) => byId[id] || null,
  querySelectorAll: () => gates,
})

test('openCollapsedAncestors opens every collapsed <details> on the path up', () => {
  const details = node({ tagName: 'DETAILS', open: false })
  const wrap = node({ parentElement: details })
  const field = node({ tagName: 'INPUT', parentElement: wrap })
  openCollapsedAncestors(field)
  assert.equal(details.open, true)
})

test('openCollapsedAncestors leaves an already-open <details> and non-details alone', () => {
  const already = node({ tagName: 'DETAILS', open: true })
  const plain = node({ tagName: 'SECTION', parentElement: already })
  const field = node({ tagName: 'INPUT', parentElement: plain })
  openCollapsedAncestors(field)
  assert.equal(already.open, true)       // untouched
  assert.equal(plain.open, undefined)    // never given an .open
})

test('openCollapsedAncestors terminates even on a parent cycle', () => {
  const a = node()
  const b = node({ parentElement: a })
  a.parentElement = b                    // cycle
  // The guard bounds the walk; this must return rather than hang.
  openCollapsedAncestors(b)
  assert.ok(true)
})

test('resolveFocusTarget returns the field itself when it is in the DOM', () => {
  const el = node({ tagName: 'INPUT' })
  const r = resolveFocusTarget('server-token', doc({ byId: { 'server-token': el } }))
  assert.deepEqual(r, { el, gated: false })
})

test('resolveFocusTarget falls back to the deepest (last) gate when the field is absent', () => {
  const outer = node({ tagName: 'BUTTON' })   // e.g. the LAN switch
  const inner = node({ tagName: 'BUTTON' })   // e.g. the require-token switch
  const r = resolveFocusTarget('server-token', doc({ byId: {}, gates: [outer, inner] }))
  assert.equal(r.el, inner)
  assert.equal(r.gated, true)
})

test('resolveFocusTarget returns null when neither the field nor a gate exists', () => {
  assert.equal(resolveFocusTarget('server-token', doc({ byId: {}, gates: [] })), null)
})

test('resolveFocusTarget is defensive about empty ids and a missing document', () => {
  assert.equal(resolveFocusTarget('', doc()), null)
  assert.equal(resolveFocusTarget('server-token', null), null)
})
