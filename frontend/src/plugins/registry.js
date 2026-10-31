// The plugin registry the core reads its contributions from — pure logic, no
// Vite, no DOM, so node --test can exercise it. Bundled descriptors arrive
// from bundled.js (a build-time glob); external ones through
// window.lds.registerPlugin, BEFORE React mounts (loadPlugins.js). Which
// plugins are enabled comes from /api/plugins/; the descriptor of a disabled
// plugin stays registered but contributes nothing.
import { KNOWN_SLOTS, SETTINGS_GROUP_SECTIONS, surfacesOf } from './parity.js'
import { validateGuide, composeGuide } from './guideContent.js'

const ID = /^[a-z][a-z0-9_]{1,31}(\.[a-z][a-z0-9_]{1,31})?$/

const state = {
  descriptors: new Map(),   // id -> { descriptor, external }
  enabled: null,            // Set of ids, or null = everything registered counts (before the manifest)
  membership: { plugin_ids: [], allowed: false },
  problems: [],             // { plugin, message } — shown by the Plugins page
}

let revision = 0
const listeners = new Set()
export const pluginRevision = () => revision
export const subscribePlugins = (listener) => { listeners.add(listener); return () => listeners.delete(listener) }
function changed() { revision += 1; for (const listener of listeners) listener() }

export function setMembership(value) {
  const next = { ...value, plugin_ids: Array.isArray(value?.plugin_ids) ? [...value.plugin_ids].sort() : state.membership.plugin_ids,
    allowed: value?.allowed === true }
  if (JSON.stringify(next) === JSON.stringify(state.membership)) return
  state.membership = next
  changed()
}

export function membershipBlocked(id) {
  return state.membership.plugin_ids.includes(id) && !state.membership.allowed
}

export function pluginMembership(id, fallback = null) {
  return state.membership.plugin_ids.includes(id) ? state.membership : fallback
}

export function resetRegistry() {
  state.descriptors.clear()
  state.enabled = null
  state.membership = { plugin_ids: [], allowed: false }
  state.problems.length = 0
  changed()
}

export function validateDescriptor(descriptor, guideOwnership) {
  if (!descriptor || typeof descriptor !== 'object') return 'a descriptor must be an object'
  if (typeof descriptor.id !== 'string' || !ID.test(descriptor.id)) return `"id" ${JSON.stringify(descriptor.id)} is not a plugin id`
  for (const key of ['nav', 'routes', 'help', 'whatsNew', 'hosts', 'paritySkip']) {
    if (descriptor[key] !== undefined && !Array.isArray(descriptor[key])) return `"${key}" must be a list`
  }
  for (const topic of descriptor.help || []) {
    if (topic?.requires !== undefined && (!Array.isArray(topic.requires) || topic.requires.some(id => typeof id !== 'string' || !ID.test(id)))) return 'Help requires must list plugin ids.'
  }
  if (descriptor.slots !== undefined) {
    if (typeof descriptor.slots !== 'object' || Array.isArray(descriptor.slots)) return '"slots" must be an object'
    for (const [slot, items] of Object.entries(descriptor.slots)) {
      if (!Array.isArray(items)) return `slot "${slot}" must be a list`
      const hosted = [...state.descriptors.values()].some((e) => (e.descriptor.hosts || []).includes(slot))
      if (!KNOWN_SLOTS.includes(slot) && !hosted) return `slot "${slot}" is not a known slot`
      if (slot === 'settings.group') {
        for (const item of items) {
          if (!item || !SETTINGS_GROUP_SECTIONS.includes(item.section)) {
            return `settings.group ${JSON.stringify(item && item.id)}: section ${JSON.stringify(item && item.section)} renders no plugin groups (one of ${SETTINGS_GROUP_SECTIONS.join(', ')})`
          }
        }
      }
    }
  }
  return validateGuide(descriptor, [...state.descriptors.values()].map(e => e.descriptor), guideOwnership)
}

export function registerDescriptor(descriptor, { external = false, official = false, guideOwnership, pluginName } = {}) {
  if (external && !official && typeof descriptor?.id === 'string' && !descriptor.id.includes('.')) {
    state.problems.push({ plugin: descriptor.id, message: 'A short plugin id requires a verified official package.' })
    return false
  }
  const problem = validateDescriptor(descriptor, guideOwnership)
  if (problem) {
    state.problems.push({ plugin: descriptor && descriptor.id, message: problem })
    return false
  }
  if (state.descriptors.has(descriptor.id)) {
    state.problems.push({ plugin: descriptor.id, message: 'registered twice' })
    return false
  }
  state.descriptors.set(descriptor.id, { descriptor, external, name: typeof pluginName === 'string' ? pluginName : descriptor.id })
  return true
}

export function setEnabled(ids) {
  state.enabled = new Set(ids)
  changed()
}

/** A module that failed after registration must contribute nothing this boot. */
export function discardExternalDescriptor(id) {
  if (state.descriptors.get(id)?.external) state.descriptors.delete(id)
}

export function problems() {
  return state.problems.slice()
}

function active() {
  const out = []
  for (const { descriptor } of state.descriptors.values()) {
    if ((state.enabled === null || state.enabled.has(descriptor.id)) && !membershipBlocked(descriptor.id)) out.push(descriptor)
  }
  return out
}

/** Every registered descriptor with its origin, enabled or not — what the
 *  loader falls back on when the plugin list never arrives. */
export function registeredDescriptors() {
  return [...state.descriptors.values()].map((e) => ({ descriptor: e.descriptor, external: e.external }))
}

/** Enabled descriptors, in registration order (bundled first, then external). */
export function plugins() {
  return active()
}

/** What enabled plugins contribute to `slot` on `surface`
 *  (`surface` omitted = every contribution of the slot). */
export function contributions(slot, surface) {
  const out = []
  for (const descriptor of active()) {
    const items = (descriptor.slots || {})[slot]
    if (!Array.isArray(items)) continue
    for (const item of items) {
      if (surface && !surfacesOf(slot, item).includes(surface)) continue
      out.push({ ...item, plugin: descriptor.id })
    }
  }
  return out
}

export function navItems(caps = {}) {
  return active().flatMap((d) => (d.nav || [])
    .filter((n) => typeof n.when !== 'function' || n.when(caps))
    .map((n) => ({ ...n, plugin: d.id })))
}

export function routes() {
  return active().flatMap((d) => (d.routes || []).map((r) => ({ ...r, plugin: d.id })))
}

export function helpTopics() {
  const descriptors = active()
  const ids = new Set(descriptors.map(d => d.id))
  return descriptors.flatMap((d) => (d.help || []).filter(t => (t.requires || []).every(id => ids.has(id))).map((t) => ({ ...t, plugin: d.id, pluginName: state.descriptors.get(d.id).name })))
}

export function whatsNewEntries() {
  return active().flatMap((d) => (d.whatsNew || []).map((e) => ({ ...e, plugin: d.id })))
}

/** Read a registered descriptor's history independently of its enabled state.
 * Bundled descriptors register eagerly; disabled external scripts do not load. */
export function pluginWhatsNew(id) {
  return [...(state.descriptors.get(id)?.descriptor.whatsNew || [])]
}

/** Content comes from the installed version and only the effective boot set. */
export function guideChapters(coreChapters) {
  return composeGuide(coreChapters, active())
}
