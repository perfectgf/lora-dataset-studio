import { Suspense, lazy, useMemo, useSyncExternalStore } from 'react'
import ErrorBoundary from '../components/common/ErrorBoundary'
import { contributions, subscribePlugins, pluginRevision } from './registry.js'

/* A place in the core where plugins contribute — rendered as the list of
   contributions enabled plugins made to `slot` on `surface`. Each contribution
   names a component per surface (`panels[surface]`) or one for all (`panel`),
   as a lazy `() => import(...)`, so a plugin's screen is its own chunk and only
   loads where it mounts. The host passes its own props through; the
   contribution also receives `surface`. Props that must differ PER
   contribution — a callback the host has to attribute to one of them, like a
   layer report — come from `itemProps(item)`, merged over the shared ones.

   Rendering is fenced: a plugin panel that throws shows a small notice in its
   place instead of taking the host page down. */

const cache = new Map()

function componentFor(item, surface) {
  const importer = (item.panels && item.panels[surface]) || item.panel
  if (typeof importer !== 'function') return null
  const key = `${item.plugin}:${item.id}:${surface}`
  if (!cache.has(key)) cache.set(key, lazy(importer))
  return cache.get(key)
}

export function hasContributions(slot, surface) {
  return contributions(slot, surface).length > 0
}

/** The lazy component for one importer, cached under `key` so a re-render
 *  never re-creates it (React would remount and lose the panel's state). */
export function lazyPanel(key, importer) {
  if (!cache.has(key)) cache.set(key, lazy(importer))
  return cache.get(key)
}

/** One plugin panel outside a slot — a component a plugin names on a piece of
 *  DATA it contributed (an engine spec's `card`, a settings group's `panel`),
 *  rendered with the same fence as a slot contribution. */
export function PluginPanel({ panelKey, importer, ...props }) {
  const Component = lazyPanel(panelKey, importer)
  return (
    <ErrorBoundary>
      <Suspense fallback={null}>
        <Component {...props} />
      </Suspense>
    </ErrorBoundary>
  )
}

export default function PluginSlot({ slot, surface, fallback = null, itemProps = null, ...hostProps }) {
  useSyncExternalStore(subscribePlugins, pluginRevision, pluginRevision)
  const items = contributions(slot, surface)
  // Keyed on WHICH contributions are here, not how many: two plugins on one
  // slot swapped for each other keep the count and must not keep the memo
  // (a refutation finding, 2026-09-04). Membership changes can remove contributions.
  const key = items.map((item) => `${item.plugin}:${item.id}`).join('|')
  const rendered = useMemo(() => items.map((item) => ({ item, Component: componentFor(item, surface) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [slot, surface, key])
  if (!rendered.length) return fallback
  return rendered.map(({ item, Component }) => Component && (
    <ErrorBoundary key={`${item.plugin}:${item.id}`}>
      <Suspense fallback={null}>
        <Component surface={surface} {...hostProps} {...(typeof itemProps === 'function' ? itemProps(item) : null)} />
      </Suspense>
    </ErrorBoundary>
  ))
}
