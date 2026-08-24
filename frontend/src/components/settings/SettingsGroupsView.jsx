/* The summary-and-groups shell a long settings section renders into.
 * Decisions and storage live in settingsGroups.js (pure, tested); this file
 * is only the markup. Built for Image engines first, shaped to be reused by
 * the other long sections when they adopt the same layout.
 */
import { useState } from 'react'
import { groupDomId, readOpenGroups, storeGroupToggle } from './settingsGroups'

/** The three lines every grouped section repeats, once: freeze the open set at
 *  mount and hand back the props one SettingsGroup takes. Frozen on purpose —
 *  the uncontrolled <details> contract (below) hangs on `defaultOpen` never
 *  changing across renders. */
export function useSettingsGroupProps(sectionId) {
  const [initiallyOpen] = useState(() => readOpenGroups(
    typeof localStorage === 'undefined' ? null : localStorage, sectionId))
  return (group) => ({ sectionId, group, defaultOpen: initiallyOpen.has(group.id) })
}

/** The clickable map at the top of the section: one row per group. Clicking
 *  opens the group (DOM-driven, same channel revealTarget uses) and scrolls
 *  to it — on a phone the map IS the navigation, so rows are finger-sized. */
export function SettingsGroupsToc({ sectionId, groups }) {
  const jump = (groupId) => {
    const el = document.getElementById(groupDomId(sectionId, groupId))
    if (!el) return
    if (!el.open) {
      el.open = true
      // The manual channel and the user's own clicks persist the same way.
      storeGroupToggle(localStorage, sectionId, groupId, true)
    }
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  return (
    <nav aria-label="Section contents"
      className="rounded-xl border border-border bg-surface p-2">
      <ul className="m-0 grid list-none grid-cols-1 gap-1 p-0 sm:grid-cols-2">
        {groups.map((g) => (
          <li key={g.id} className="m-0">
            <button type="button" onClick={() => jump(g.id)}
              data-testid={`settings-toc-${g.id}`}
              className="min-h-10 lg:min-h-0 flex w-full items-baseline gap-2 rounded-lg px-2.5 py-1.5 text-left hover:bg-surface-raised">
              <span aria-hidden className="shrink-0 text-sm">{g.icon}</span>
              <span className="min-w-0">
                <span className="block text-sm font-medium text-content">{g.title}</span>
                <span className="block text-xs text-content-muted">{g.blurb}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  )
}

/** One collapsible group. A NATIVE, UNCONTROLLED <details>: the ?focus= reveal
 *  opens collapsed ancestors on the DOM directly, and a controlled `open`
 *  prop would close it back on the next render. `defaultOpen` is frozen at
 *  mount, so React never rewrites the attribute over a user's toggle; the
 *  children stay MOUNTED while collapsed, so nothing loses its edit state. */
export function SettingsGroup({ sectionId, group, defaultOpen = false, children }) {
  // Frozen: the whole uncontrolled contract hangs on this never changing.
  const [initiallyOpen] = useState(defaultOpen)
  return (
    <details id={groupDomId(sectionId, group.id)}
      open={initiallyOpen || undefined}
      onToggle={(e) => storeGroupToggle(
        localStorage, sectionId, group.id, e.currentTarget.open)}
      className="group scroll-mt-24 rounded-xl border border-border bg-surface">
      <summary
        className="flex min-h-10 lg:min-h-0 cursor-pointer list-none items-baseline gap-2 rounded-xl px-4 py-3 hover:bg-surface-raised [&::-webkit-details-marker]:hidden">
        <span aria-hidden
          className="shrink-0 text-xs text-content-subtle transition-transform group-open:rotate-90">▸</span>
        <span aria-hidden className="shrink-0 text-sm">{group.icon}</span>
        <span className="min-w-0">
          <span className="block text-sm font-semibold text-content">{group.title}</span>
          {/* The blurb hides once the group is open — it described what you
              would find, and you are now looking at it. */}
          <span className="block text-xs text-content-muted group-open:hidden">{group.blurb}</span>
        </span>
      </summary>
      <div className="space-y-6 px-2 pb-2 sm:px-3 sm:pb-3">
        {children}
      </div>
    </details>
  )
}
