import test from 'node:test'
import assert from 'node:assert/strict'
import { contributions, plugins, registerDescriptor, resetRegistry, setEnabled, setMembership, subscribePlugins } from './registry.js'
import { refreshMembership, notifyMembershipChanged, watchMembership, MEMBERSHIP_CHANGED } from './membership.js'
import { pluginActive } from './lifecycle.js'

const member = 'patreon.example'
const access = (allowed) => ({ plugin_ids: [member], allowed })
function boot() {
  resetRegistry()
  assert.equal(registerDescriptor({ id: member, slots: { 'sources.panel': [{ id: 'member' }] } }), true)
  registerDescriptor({ id: 'video' })
  setEnabled([member, 'video'])
  setMembership(access(true))
}

test('revocation removes installed routes and contributions, reconnect restores without reloading', () => {
  boot()
  let changes = 0
  const unsubscribe = subscribePlugins(() => changes++)
  assert.equal(contributions('sources.panel').length, 1)
  notifyMembershipChanged(true, null)
  assert.deepEqual(plugins().map(p => p.id), ['video'])
  assert.equal(contributions('sources.panel').length, 0)
  assert.equal(pluginActive({ id: member, active: true }), false)
  setMembership(access(true))
  assert.equal(contributions('sources.panel').length, 1)
  assert.equal(pluginActive({ id: member, active: true, membership: { allowed: false } }), true)
  assert.equal(changes, 2)
  unsubscribe()
})

test('a pending active response cannot restore access after local disconnection', async () => {
  boot()
  let release
  const pending = refreshMembership(() => new Promise(resolve => { release = resolve }))
  notifyMembershipChanged(true, null)
  release(access(true))
  await pending
  assert.deepEqual(plugins().map(p => p.id), ['video'])
})

test('unavailable verification hides member screens and keeps public plugins', async () => {
  boot()
  await refreshMembership(async () => { throw new Error('offline') })
  assert.deepEqual(plugins().map(p => p.id), ['video'])
})

test('another tab disconnecting removes the current contributions before refreshing', () => {
  boot()
  const listeners = new Map()
  const stop = watchMembership({ setInterval: () => 1, clearInterval() {},
    addEventListener: (name, fn) => listeners.set(name, fn), removeEventListener: name => listeners.delete(name) })
  listeners.get('storage')({ key: MEMBERSHIP_CHANGED, newValue: JSON.stringify({ revoked: true }) })
  assert.deepEqual(plugins().map(p => p.id), ['video'])
  stop()
  assert.equal(listeners.size, 0)
})
