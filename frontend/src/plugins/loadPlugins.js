// Boot-time plugin loading — BEFORE React mounts, so a plugin's nav entry,
// route, panel or settings group is there on the first render exactly like a
// bundled one (the legacy extension loader ran without await and its scripts
// arrived after the nav had rendered).
//
// 1. window.lds is published: what an external plugin's script may use.
// 2. /api/plugins/ says which plugins are enabled and where each external
//    plugin's UI entry is served.
// 3. every enabled external entry is loaded as a <script type="module">; the
//    script calls window.lds.registerPlugin(descriptor) while it evaluates.
//    Each load is bounded by a timeout; a failure is recorded, never fatal.
//
// The active set comes only from this boot's backend. Neither a cached list
// nor the presence of development sources proves that a plugin is active.
// Requests are retried within a timeout; failures leave contributions off.
import React from 'react'
import ReactDOM from 'react-dom/client'
import * as ReactDOMPortal from 'react-dom'
import * as ReactJSXRuntime from 'react/jsx-runtime'
import { Link, NavLink, useLocation, useNavigate, useParams, useSearchParams } from 'react-router'
import { discardExternalDescriptor, plugins, problems, registerDescriptor, setEnabled, setMembership } from './registry.js'
import { apiFetch, del, patchJson, postForm, postJson, putJson } from '../api/fetchClient.js'
import { loadModuleScript } from './moduleScripts.js'
import { pluginLoaded } from './lifecycle.js'
import { loadPluginStyles } from './styles.js'

export const LDS_PLUGIN_API_MAJOR = 1
export const ENABLED_CACHE_KEY = 'lds:plugins:enabled:v1'
let runtimeServices = {}
let expectedRegistration = null
let loadProblems = []

export function getLoadProblems() { return loadProblems.map((problem) => ({ ...problem })) }

function finishLoading(result) {
  loadProblems = result.failed.map(({ plugin, why }) => ({
    plugin: typeof plugin === 'string' && /^[a-z][a-z0-9_.]{1,65}$/.test(plugin) ? plugin : null,
    reason: String(why).replace(/(?:https?:\/\/|file:\/\/|[a-z]:[\\/])\S+/gi, '[location]').slice(0, 400),
  }))
  if (globalThis.window?.lds) globalThis.window.lds.loadProblems = getLoadProblems()
  return result
}

export function configureRuntimeServices(services) {
  runtimeServices = { ...services }
}

function registerLoadedPlugin(descriptor) {
  const expected = expectedRegistration
  if (!expected || descriptor?.id !== expected.id) {
    if (expected) expected.error = 'The UI descriptor does not match the installed plugin id.'
    return false
  }
  const ok = registerDescriptor(descriptor, { external: true, official: expected.official, guideOwnership: expected.guideOwnership, pluginName: expected.name })
  if (ok) expected.registered = true
  else expected.error = 'The installed plugin UI descriptor was rejected.'
  return ok
}

export function publishRuntime(win = globalThis.window) {
  if (!win) return null
  const runtime = win.lds || (win.lds = {})
  Object.assign(runtime, {
    api: LDS_PLUGIN_API_MAJOR,
    sdkVersion: '1.15.0',
    loadProblems: getLoadProblems(),
    React,
    ReactDOM,
    ReactDOMPortal,
    ReactJSXRuntime,
    router: { Link, NavLink, useLocation, useNavigate, useParams, useSearchParams },
    // Reads AND writes: the mutating helpers carry the CSRF token the way the
    // core's own screens do. `apiFetch` alone cannot POST (measured: every
    // write through it was refused as a stale session).
    apiFetch, postJson, putJson, patchJson, del, postForm,
    registerPlugin: registerLoadedPlugin,
    plugins,
    ...runtimeServices,
  })
  return runtime
}

function defaultStorage() {
  try { return globalThis.localStorage || null } catch { return null }
}

function writeCache(storage, ids) {
  try { storage && storage.setItem(ENABLED_CACHE_KEY, JSON.stringify(ids)) } catch { /* private mode */ }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function validPluginList(payload) {
  if (!payload || !Array.isArray(payload.plugins)) return false
  const ids = new Set()
  return payload.plugins.every(plugin => {
    if (!plugin || typeof plugin !== 'object' || typeof plugin.id !== 'string'
      || !/^[a-z][a-z0-9_]{1,31}(\.[a-z][a-z0-9_]{1,31})?$/.test(plugin.id)
      || ids.has(plugin.id)) return false
    ids.add(plugin.id)
    return true
  })
}

async function fetchPluginList(fetchImpl, { attempts, delayMs, timeoutMs }) {
  let lastError = null
  for (let i = 0; i < attempts; i += 1) {
    const controller = new AbortController()
    let timer
    try {
      const request = (async () => {
        const res = await fetchImpl('/api/plugins/', { cache: 'no-store', signal: controller.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })()
      const payload = await Promise.race([request, new Promise((_, reject) => {
        timer = setTimeout(() => {
          controller.abort()
          reject(new Error('Plugin list request timed out.'))
        }, timeoutMs)
      })])
      if (!validPluginList(payload)) throw new Error('Invalid plugin list.')
      return { payload, error: null }
    } catch (err) {
      lastError = err
    } finally {
      clearTimeout(timer)
    }
    if (i + 1 < attempts) await sleep(delayMs * (2 ** i))
  }
  return { payload: null, error: lastError }
}

/** Resolves to { enabled, loaded, failed } and never rejects: the app mounts
 *  either way, and the Plugins page shows what did not load. */
export async function loadPlugins({
  fetchImpl = globalThis.fetch, timeoutMs = 5000, doc = globalThis.document,
  storage = defaultStorage(), attempts = 3, delayMs = 300,
  moduleLoader = loadModuleScript, styleLoader = loadPluginStyles,
} = {}) {
  setEnabled([])
  publishRuntime()
  const result = { enabled: [], loaded: [], failed: [] }
  const boundedTimeout = Number.isFinite(timeoutMs) && timeoutMs > 0 ? Math.min(timeoutMs, 30000) : 5000
  const { payload, error } = await fetchPluginList(fetchImpl, {
    attempts: Number.isInteger(attempts) && attempts > 0 ? Math.min(attempts, 5) : 3,
    delayMs: Number.isFinite(delayMs) && delayMs >= 0 ? Math.min(delayMs, 1000) : 300,
    timeoutMs: boundedTimeout,
  })
  if (!payload) {
    const why = `plugin list unavailable: ${error && error.message ? error.message : error}`
    result.failed.push({ plugin: null, why: `${why} — plugins remain unavailable until LDS confirms the active set` })
    for (const problem of problems()) result.failed.push({ plugin: problem.plugin, why: problem.message })
    return finishLoading(result)
  }
  const list = payload.plugins
  setMembership(payload.membership || { plugin_ids: [], allowed: false })
  if (globalThis.window?.lds) globalThis.window.lds.bootId = payload.boot_id
  result.enabled = list.filter(pluginLoaded).map((p) => p.id)
  setEnabled(result.enabled)
  writeCache(storage, result.enabled)
  for (const p of list) {
    // Legacy bundled sources have no frontend URL. Packaged plugins use the
    // same loader regardless of how they were delivered to the installation.
    if (!pluginLoaded(p) || !p.frontend || !doc) continue
    let css
    const expected = { id: p.id, name: p.name, official: p.official === true, guideOwnership: p.guide_ownership, registered: false, error: null }
    try {
      css = await styleLoader(p.styles || [], boundedTimeout, doc)
      if (!css?.ok) {
        result.failed.push({ plugin: p.id, why: `Stylesheet ${css?.why || 'failed to load'}` })
        continue
      }
      expectedRegistration = expected
      const { ok, why } = await moduleLoader(p.frontend, boundedTimeout, doc)
      // Existing extension modules can publish legacy capabilities instead of
      // a descriptor. SDK packages explicitly require descriptor registration.
      const needsDescriptor = p.official === true || p.schema_version >= 2 || (p.styles || []).length > 0
      if (ok && !expected.error && (!needsDescriptor || expected.registered)) result.loaded.push(p.id)
      else {
        css?.remove?.()
        if (expected.registered) discardExternalDescriptor(p.id)
        result.failed.push({ plugin: p.id, why: expected.error || why || 'The plugin UI did not register its descriptor.' })
      }
    } catch (error) {
      css?.remove?.()
      if (expected.registered) discardExternalDescriptor(p.id)
      result.failed.push({ plugin: p.id, why: 'The plugin UI failed to load.' })
    } finally {
      expectedRegistration = null
    }
  }
  for (const problem of problems()) result.failed.push({ plugin: problem.plugin, why: problem.message })
  return finishLoading(result)
}
