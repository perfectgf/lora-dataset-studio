import PluginSlot from './plugins/PluginSlot.jsx'
import { Suspense, useEffect, useState, useSyncExternalStore } from 'react'
import { HashRouter, Routes, Route, Navigate, Outlet, NavLink, useLocation } from 'react-router'
import { Archive, ArrowUp, CircleHelp, Dumbbell, FlaskConical, FolderOpen, Images, Loader2, Menu, PanelLeftClose, PanelLeftOpen, Puzzle, Settings, X } from 'lucide-react'
import { apiFetch, postJson } from './api/fetchClient'
import { JobsProvider } from './context/JobsContext'
import { ToastProvider, useToast } from './components/common/Toast'
import { CapabilitiesProvider, useCapabilities } from './context/CapabilitiesContext'
import { setToastRef } from './api/fetchClient'
import ErrorBoundary from './components/common/ErrorBoundary'
import { WhatsNewButton, WhatsNewModal } from './components/common/WhatsNew'
import ConnectionBanner from './components/common/ConnectionBanner'
import SetupHealthNotice from './components/setup/SetupHealthNotice'
import SetupJourneyNotice from './components/setup/SetupJourneyNotice'
import ComfyRecoveryBanner from './components/common/ComfyRecoveryBanner'
import GenerationQueueDock from './components/common/GenerationQueueDock'
import DockerUpdateInstructions from './components/common/DockerUpdateInstructions'
import PinokioUpdateInstructions from './components/common/PinokioUpdateInstructions'
import { lazyPage } from './utils/lazyPage'

// Each page is its own chunk, fetched on first navigation — the entry bundle
// stops carrying all eighteen routes to paint one. `lazyPage` also owns the
// stale-chunk reload after an Update & restart (see utils/lazyPage.js).
const DatasetPage = lazyPage(() => import('./pages/DatasetPage'))
const BankPage = lazyPage(() => import('./pages/BankPage'))
const StudioPage = lazyPage(() => import('./pages/StudioPage'))
const SettingsPage = lazyPage(() => import('./pages/SettingsPage'))
const SetupPage = lazyPage(() => import('./pages/SetupPage'))
const GuidePage = lazyPage(() => import('./pages/GuidePage'))
const CloudRunsPage = lazyPage(() => import('./pages/CloudRunsPage'))
const GalleryPage = lazyPage(() => import('./pages/GalleryPage'))
const PluginsPage = lazyPage(() => import('./pages/PluginsPage'))
const PluginSettingsPage = lazyPage(() => import('./pages/PluginSettingsPage'))
import { navItems as pluginNavItems, routes as pluginRoutes, subscribePlugins, pluginRevision } from './plugins/registry'
import { plainNavLabel } from './plugins/navLabel.js'
import { watchMembership } from './plugins/membership.js'
import { lazyRoute } from './plugins/lazyRoute'
import UnavailablePluginPage from './pages/UnavailablePluginPage.jsx'

// A page a plugin contributes: `page` is its `() => import(...)`, wrapped in
// the same lazyPage as the core's routes so it gets the stale-chunk reload —
// through a cache keyed by the route, NOT a useMemo: a render that suspends
// never commits, so a memo made in it is remade on the retry, with a fresh
// import each time, and the page spins forever (plugins/lazyRoute.js).
function PluginRoute({ routeKey, page }) {
  const Page = lazyRoute(routeKey, page)
  return <Page />
}
import { HelpModeProvider, useHelpMode, TipHost } from './help/HelpMode'
import HeaderMenu from './components/common/HeaderMenu'
import { SIDE_ITEM_BASE, SIDE_ITEM_REST, SIDE_NAV_QUERY, railItemClass, sideItemClass } from './components/common/navItemStyles.js'
import { applySideNavCollapsed, readSideNavCollapsed, writeSideNavCollapsed } from './components/common/sideNavState.js'
import { useMediaQuery } from './hooks/useMediaQuery'
import { versionLabel } from './utils/versionLabel'
import { useTrainingActivity } from './hooks/useTrainingActivity'
import { activityLabel } from './utils/trainingActivity'
import { installMode, membersUpdateAvailable, patreonCheckFailure, updateCheckFeedback } from './components/settings/updateStatus'
import { settingsLinkHref } from './components/common/settingsLinkHref'
import { waitForRelaunchedServer } from './utils/serverRestart'
import { UsageStatisticsProvider } from './usage/UsageStatisticsContext'
import UsageStatisticsBanner from './usage/UsageStatisticsBanner'

// The compact buttons of the top bar (below `xl`) and the rows of the panel its
// hamburger opens.
const NAV_ITEM_BASE =
  'px-2 lg:px-3 py-1.5 rounded-md text-sm font-medium whitespace-nowrap no-underline transition-colors'
const navItemClass = ({ isActive }) =>
  `${NAV_ITEM_BASE} ${
    isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:text-content hover:bg-surface-raised'
  }`


// Full-width variant for links that live inside a HeaderMenu dropdown.
const MENU_ITEM_BASE =
  'block w-full text-left px-3 py-1.5 rounded-md text-sm font-medium no-underline transition-colors'
const menuItemClass = ({ isActive }) =>
  `${MENU_ITEM_BASE} ${
    isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:text-content hover:bg-surface-raised'
  }`

/** Asks the update notices to open. An update is announced by the menu entry
 * (lit, and worded "Update available" where it has a label); the notice that
 * carries the action is opened by pressing that entry, never by itself. */
const SHOW_UPDATE_NOTICE_EVENT = 'lds:show-update-notice'

/** Nav action: force an update check and give immediate feedback — a toast when
 * up to date, and the actionable UpdateBanner (with the one-click "Update &
 * restart") when there is an update.
 * AUTO-DETECTION: on mount (and every hour while the tab stays open) it runs
 * the git-aware check — a server-side TTL cache keeps the network cost down
 * across page loads. An available update LIGHTS THIS ENTRY and nothing else:
 * the banner used to open on every page until dismissed, which put a block
 * above the work for something the entry already says. `withLabel` is the row
 * of the side menu; without it, the icon button of the top bar. */
function CheckUpdatesButton({ withLabel = false }) {
  const toast = useToast()
  const [busy, setBusy] = useState(false)
  const [available, setAvailable] = useState(false)
  const [members, setMembers] = useState(null)
  useEffect(() => {
    let alive = true
    const onPublicStatus = (event) => setAvailable(!!event.detail?.update_available)
    const onPatreonStatus = (event) => {
      const status = event.detail
      // A failed refresh must not erase a previously confirmed update.
      if (!status || !['ready', 'not_configured'].includes(status.status)) return
      const next = membersUpdateAvailable(status) ? status : null
      setMembers(next)
      window.dispatchEvent(new CustomEvent('lds:members-update', { detail: next }))
    }
    window.addEventListener('lds:update-available', onPublicStatus)
    window.addEventListener('lds:patreon-status', onPatreonStatus)
    const autoCheck = async () => {
      try {
        const d = await apiFetch('/api/update/check?auto=1', { background: true })
        if (!alive) return
        setAvailable(!!d?.update_available)
        // Shares the status (this entry, the banner's content, Settings ›
        // Updates). It does not open the banner: only a press does.
        if (d?.ok) window.dispatchEvent(new CustomEvent('lds:update-available', { detail: d }))
      } catch { /* offline — the manual button stays available */ }
      const m = await membersUpdate()
      if (!alive) return
      window.dispatchEvent(new CustomEvent('lds:patreon-status', { detail: m }))
    }
    autoCheck()
    // 1 h: the project ships several times a day right now — 6 h let a tab
    // sit stale most of a working day. Server-side TTL matches.
    const t = setInterval(autoCheck, 3600 * 1000)
    return () => {
      alive = false; clearInterval(t)
      window.removeEventListener('lds:update-available', onPublicStatus)
      window.removeEventListener('lds:patreon-status', onPatreonStatus)
    }
  }, [])
  const check = async () => {
    if (busy) return
    setBusy(true)
    try {
      const d = await apiFetch('/api/update/check?force=1')
      const m = await membersUpdate(true)
      setAvailable(!!d?.update_available)
      window.dispatchEvent(new CustomEvent('lds:update-available', { detail: d }))
      window.dispatchEvent(new CustomEvent('lds:patreon-status', { detail: m }))
      // After the statuses, so the notices open on what was just found.
      window.dispatchEvent(new CustomEvent(SHOW_UPDATE_NOTICE_EVENT))
      const feedback = updateCheckFeedback(d, m)
      toast[feedback.type](feedback.message)
    } catch (e) {
      toast.error(e?.message || 'Update check failed.')
    } finally {
      setBusy(false)
    }
  }
  const lit = available || !!members
  const icon = busy
    ? <Loader2 aria-hidden="true" className="h-4 w-4 shrink-0 animate-spin" />
    : <ArrowUp aria-hidden="true" className="h-4 w-4 shrink-0" />
  // Lit = the app's one accent, not a second hue: the words (or the sr-only
  // label) carry the meaning, the colour only draws the eye to them.
  if (withLabel) {
    return (
      <button type="button" onClick={check} disabled={busy}
        title={lit ? 'Update available — click to review' : 'Check for updates'}
        className={`${SIDE_ITEM_BASE} ${lit ? 'text-primary hover:bg-surface-raised' : SIDE_ITEM_REST} disabled:opacity-50`}>
        {icon}
        <span>{lit ? 'Update available' : 'Check for updates'}</span>
        {lit && <span aria-hidden className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />}
      </button>
    )
  }
  return (
    <button type="button" onClick={check} disabled={busy}
      title={lit ? 'Update available — click to review' : 'Check for updates'}
      className={`${NAV_ITEM_BASE} relative ${lit
        ? 'text-primary'
        : 'text-content-muted hover:text-content'} hover:bg-surface-raised disabled:opacity-50`}>
      {icon}
      {lit && (
        <span aria-hidden className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-primary" />
      )}
      <span className="sr-only">{lit ? 'Update available' : 'Check for updates'}</span>
    </button>
  )
}

/** The members version comes through the Patreon channel, which
 * /api/update/check cannot see: a member who had just linked Patreon on a
 * public install was never told it existed. Preserve the full result so an
 * unreachable channel cannot be mistaken for "no update". */
async function membersUpdate(force = false) {
  try {
    return await apiFetch(`/api/update/patreon${force ? '?force=1' : ''}`, { background: true })
  } catch (error) { return patreonCheckFailure(error) }
}

/** Whether the update notices are open: true from a press on the update entry
 * (SHOW_UPDATE_NOTICE_EVENT) until their ✕. Each notice keeps its own copy, so
 * closing one leaves the other. */
function useUpdateNoticeShown() {
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const onShow = () => setShown(true)
    window.addEventListener(SHOW_UPDATE_NOTICE_EVENT, onShow)
    return () => window.removeEventListener(SHOW_UPDATE_NOTICE_EVENT, onShow)
  }, [])
  return [shown, setShown]
}

/** Says that the members version exists. Settings › Updates installs it: its
 * card owns the download, the progress and the restart. */
function MembersUpdateBanner() {
  const [info, setInfo] = useState(null)
  const [shown, setShown] = useUpdateNoticeShown()
  useEffect(() => {
    const onFound = (e) => setInfo(e.detail || null)
    window.addEventListener('lds:members-update', onFound)
    return () => window.removeEventListener('lds:members-update', onFound)
  }, [])
  const version = info?.release?.version || ''
  if (!info || !shown) return null
  return (
    <div className="mx-auto max-w-5xl px-4 pt-3">
      <div role="status"
        className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm">
        <ArrowUp aria-hidden="true" className="h-4 w-4 shrink-0 text-primary" />
        <span className="text-content">
          LDS members version <span className="font-semibold">{version || 'update'}</span> is available.
        </span>
        <a href={settingsLinkHref('maintenance', 'settings-updates')}
          className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 transition-transform hover:-translate-y-px">
          Install it from Updates
        </a>
        <button type="button"
          onClick={() => setShown(false)}
          aria-label="Dismiss members update notice"
          className="ml-auto px-1.5 text-content-subtle hover:text-content">✕</button>
      </div>
    </div>
  )
}

/** Header toggle for Help mode. When on, every instrumented heading/action
 * shows a "?" badge that jumps to the matching spot in the guide. aria-pressed
 * spells the state out; the indigo ring makes "on" visually unmistakable. */
function HelpModeToggle({ onToggle }) {
  const { enabled, toggle } = useHelpMode()
  return (
    <button type="button" onClick={() => { toggle(); onToggle?.() }}
      aria-pressed={enabled}
      title={enabled
        ? 'Help mode is on — click any ? badge to jump to the guide'
        : 'Turn on Help mode to reveal ? badges that link to the guide'}
      className={`${NAV_ITEM_BASE} inline-flex items-center gap-1.5 ${enabled
        ? 'bg-indigo-500/20 text-indigo-200 ring-1 ring-inset ring-indigo-400/50'
        : 'text-content-muted hover:text-content hover:bg-surface-raised'}`}>
      <span aria-hidden className="grid h-4 w-4 place-items-center rounded-full border border-current text-[10px] font-bold leading-none">?</span>
      <span>Help mode</span>
    </button>
  )
}

function NavBar() {
  const { caps } = useCapabilities()
  // Live indicator on Runs: a training can hold the GPU for hours (local) or
  // bill by the minute (cloud), and from any other page nothing said so.
  const activity = useTrainingActivity()
  const activityTitle = activityLabel(activity)
  // ONE MENU, TWO PLACEMENTS, chosen by width and never merely hidden: a column
  // down the left edge from `xl`, and below it a one-row bar whose hamburger
  // opens the same destinations as a panel. The bar used to carry every page in
  // a wrapping row — five workspaces plus one entry per plugin page — and
  // measured 142 px over two rows and a tools shelf on an install with ten
  // plugin pages; a column takes no height at all, however many pages there are.
  // Only one placement is mounted at a time, so the update check, the What's-new
  // badge and the machine-load poll exist once (a CSS-hidden copy still polls).
  const sideNav = useMediaQuery(SIDE_NAV_QUERY)
  const [open, setOpen] = useState(false)
  const close = () => setOpen(false)
  // The column FOLDS to a strip of icons (56 px instead of 224) and stays that
  // way across visits. The strip is the same menu with the words taken away:
  // every workspace keeps its icon, and what has no icon of its own — the
  // plugin pages, the two grouped menus — opens as a list beside the strip.
  // The width itself is the stylesheet's (see sideNavState.js): this only
  // records the choice and tells the root element.
  const [collapsed, setCollapsed] = useState(() => readSideNavCollapsed())
  const toggleCollapsed = () => {
    const next = !collapsed
    setCollapsed(next)
    writeSideNavCollapsed(next)
    applySideNavCollapsed(next)
  }
  const rail = sideNav && collapsed
  const goHome = () => {
    // Home = the datasets LIST: clear the persisted open dataset and tell
    // the mounted page (same-route clicks don't remount) to close it.
    try { localStorage.removeItem('datasetCurrentId'); } catch { /* ignore */ }
    window.dispatchEvent(new CustomEvent('lds:home'))
    setOpen(false)
  }
  // Which grouped menu owns the current route — so the Help and Settings
  // triggers can reflect the active-nav style when you're on one of their screens.
  const path = useLocation().pathname
  const helpMenuActive = path === '/guide' || path === '/help'
  const settingsMenuActive = path === '/setup' || path.startsWith('/settings') || path.startsWith('/plugins')
  const machineLoad = (
    <PluginSlot slot="resource_monitor.readout" surface="header" />
  )
  const itemClass = rail ? railItemClass : sideNav ? sideItemClass : navItemClass
  // A destination with an icon, in the shape of whichever placement is mounted.
  // On the folded strip the icon stands alone and the name is the link's title.
  const entry = (to, Icon, label, extra = null) => (
    <NavLink to={to} className={itemClass} onClick={close}
      title={rail ? label : undefined} aria-label={rail ? label : undefined}>
      {rail
        ? <><Icon aria-hidden="true" className="h-[1.125rem] w-[1.125rem]" />{extra}</>
        : sideNav
          ? <><Icon aria-hidden="true" className="h-4 w-4 shrink-0" /><span>{label}</span>{extra}</>
          : <span className="inline-flex items-center gap-1"><Icon aria-hidden="true" className="h-3.5 w-3.5" /> {label}{extra}</span>}
    </NavLink>
  )

  // The workspaces: the top of the side menu, and the first rows of the panel.
  // Same caps gates in both places.
  const workspaceLinks = (
    <>
      {entry('/datasets', FolderOpen, 'Datasets')}
      {/* Bank sits right after Datasets: it FEEDS them (triage a big unsorted
          folder, then promote the keepers into a dataset). */}
      {entry('/bank', Archive, 'Bank')}
      {/* Unified runs hub (cloud + local history) — useful as soon as ANY
          training path exists, not just the cloud one. */}
      {(caps.cloud_training || caps.training_visible) && entry('/cloud', Dumbbell, 'Runs', activity.running && (
        /* Presence IS the message, so it must not be colour-only: the
           label is read out and shown on hover/long-press. */
        <span title={activityTitle} aria-label={activityTitle} role="status"
          className={`inline-flex h-2 w-2 shrink-0 ${rail ? 'absolute right-1.5 top-1.5' : sideNav ? 'relative ml-auto' : 'relative ml-1'}`}>
          <span aria-hidden className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/70" />
          <span aria-hidden className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
        </span>
      ))}
      {entry('/studio', FlaskConical, 'Test Studio')}
      {/* Gallery — every generated image, one feed. Last: it is where the
          OUTPUT of the other workspaces accumulates, so it reads as the shelf
          at the end of the list. Visible whenever any surface that generates
          is: renders can outlive a broken ComfyUI, so the training gates keep
          it reachable even while the studio gate is down. */}
      {(caps.studio_visible || caps.cloud_training || caps.training_visible) && entry('/gallery', Images, 'Gallery')}
    </>
  )
  // Nav entries an enabled plugin contributes ({ to, label, when(caps) }), in
  // both placements (measured missing on the desktop bar, 2026-09-04: a plugin's
  // page was reachable from a phone and from nowhere on a desktop). Words only:
  // see plugins/navLabel.js.
  const pluginItems = pluginNavItems(caps)
  const pluginLinks = pluginItems.map((item) => (
    <NavLink key={`${item.plugin}:${item.to}`} to={item.to} className={sideNav ? sideItemClass : navItemClass} onClick={close}>
      {/* The side menu aligns these words with the workspaces' above, which
          carry an icon: an empty slot of the same width does it. */}
      {sideNav && <span aria-hidden className="h-4 w-4 shrink-0" />}
      <span className={sideNav ? 'min-w-0 truncate' : undefined}>{plainNavLabel(item.label)}</span>
    </NavLink>
  ))
  const onPluginPage = pluginItems.some((item) => path === item.to || path.startsWith(`${item.to}/`))

  if (sideNav) {
    const foldLabel = rail ? 'Expand the menu' : 'Fold the menu'
    return (
      <aside className="app-sidebar fixed inset-y-0 left-0 z-40 flex w-[var(--app-sidebar-w)] flex-col border-r border-border bg-surface-overlay"
        data-collapsed={rail || undefined}>
        <div className={rail ? 'flex justify-center px-2 pb-2 pt-4' : 'flex items-center gap-1 pb-3 pl-5 pr-2 pt-4'}>
          {!rail && (
            <NavLink to="/datasets" title="Back to the datasets page" onClick={goHome}
              className="min-w-0 truncate bg-gradient-primary bg-clip-text text-base font-bold tracking-tight text-transparent no-underline">
              LoRA Dataset Studio
            </NavLink>
          )}
          <button type="button" onClick={toggleCollapsed} title={foldLabel}
            aria-expanded={!rail} aria-controls="app-side-menu-pages"
            className={`grid h-8 w-8 shrink-0 place-items-center rounded-md text-content-subtle hover:bg-surface-raised hover:text-content ${rail ? '' : 'ml-auto'}`}>
            {rail
              ? <PanelLeftOpen aria-hidden="true" className="h-4 w-4" />
              : <PanelLeftClose aria-hidden="true" className="h-4 w-4" />}
            <span className="sr-only">{foldLabel}</span>
          </button>
        </div>
        {/* Open: the one part that scrolls — a long plugin list moves under the
            brand and above the foot, both of which stay where they are.
            Folded: a handful of icons that never scroll, and the box must not
            clip the plugin list that opens beside it. */}
        <nav id="app-side-menu-pages" aria-label="Main navigation"
          className={`app-nav flex min-h-0 flex-1 flex-col px-2 pb-3 ${rail ? 'items-center gap-1' : 'gap-4 overflow-y-auto'}`}>
          <div className={`flex flex-col ${rail ? 'items-center gap-1' : 'gap-0.5'}`}>{workspaceLinks}</div>
          {pluginItems.length > 0 && (rail ? (
            /* Plugin pages declare no icon, so the strip gives them one door:
               their list opens beside it, top aligned, and scrolls if it is long. */
            <HeaderMenu placement="side-down" triggerClassName={railItemClass}
              triggerLabel={<Puzzle aria-hidden="true" className="h-[1.125rem] w-[1.125rem]" />}
              triggerTitle="Plugin pages" active={onPluginPage}>
              {(closeMenu) => pluginItems.map((item) => (
                <NavLink key={`${item.plugin}:${item.to}`} to={item.to} role="menuitem"
                  className={menuItemClass} onClick={closeMenu}>
                  {plainNavLabel(item.label)}
                </NavLink>
              ))}
            </HeaderMenu>
          ) : (
            <div className="flex flex-col gap-0.5">
              <p className="m-0 px-3 pb-1 font-mono text-xs uppercase tracking-[0.18em] text-content-subtle">Plugins</p>
              {pluginLinks}
            </div>
          ))}
        </nav>
        {/* The foot: what is about the app rather than about the work. The two
            grouped menus open beside the column, not below their trigger —
            there is no "below" at the bottom of the window. */}
        <div className={`app-sidebar-foot flex flex-col border-t border-border p-2 ${rail ? 'items-center gap-1' : 'gap-0.5'}`}>
          {/* The machine-load readout is a line of figures: it has no folded
              form, so the strip does not mount it (and its poll stops). */}
          {!rail && <div className="empty:hidden px-3 py-1">{machineLoad}</div>}
          <HeaderMenu placement="side" labelled={!rail} triggerClassName={rail ? railItemClass : sideItemClass}
            triggerLabel={rail
              ? <CircleHelp aria-hidden="true" className="h-[1.125rem] w-[1.125rem]" />
              : <><CircleHelp aria-hidden="true" className="h-4 w-4 shrink-0" /><span>Help &amp; guide</span></>}
            triggerTitle="Help & guide" active={helpMenuActive}>
            {(closeMenu) => (
              <>
                <NavLink to="/guide" role="menuitem" className={menuItemClass} onClick={closeMenu}>Guide</NavLink>
                <NavLink to="/help" role="menuitem" className={menuItemClass} onClick={closeMenu}>Help</NavLink>
                <NavLink to="/help?h=support-the-project" role="menuitem" className={menuItemClass} onClick={closeMenu}>Support LDS</NavLink>
                <HelpModeToggle onToggle={closeMenu} />
              </>
            )}
          </HeaderMenu>
          <HeaderMenu placement="side" labelled={!rail} triggerClassName={rail ? railItemClass : sideItemClass}
            triggerLabel={rail
              ? <Settings aria-hidden="true" className="h-[1.125rem] w-[1.125rem]" />
              : <><Settings aria-hidden="true" className="h-4 w-4 shrink-0" /><span>Setup &amp; settings</span></>}
            triggerTitle="Setup & settings" active={settingsMenuActive}>
            {(closeMenu) => (
              <>
                <NavLink to="/setup" role="menuitem" className={menuItemClass} onClick={closeMenu}>Setup</NavLink>
                <NavLink to="/settings" role="menuitem" className={menuItemClass} onClick={closeMenu}>Settings</NavLink>
                <NavLink to="/plugins" role="menuitem" className={menuItemClass} onClick={closeMenu}>Plugins</NavLink>
              </>
            )}
          </HeaderMenu>
          <WhatsNewButton withLabel={!rail} />
          <CheckUpdatesButton withLabel={!rail} />
        </div>
      </aside>
    )
  }

  // Below `xl`: every destination as a flat stack — no nested dropdowns on
  // touch. Order mirrors the side menu, top to bottom.
  const panelLinks = (
    <>
      {workspaceLinks}
      {pluginLinks}
      <NavLink to="/guide" className={navItemClass} onClick={close}>Guide</NavLink>
      <NavLink to="/setup" className={navItemClass} onClick={close}>Setup</NavLink>
      <NavLink to="/settings" className={navItemClass} onClick={close}>Settings</NavLink>
      <NavLink to="/plugins" className={navItemClass} onClick={close}>Plugins</NavLink>
      <NavLink to="/help" className={navItemClass} onClick={close}>Help</NavLink>
      <NavLink to="/help?h=support-the-project" className={navItemClass} onClick={close}>Support LDS</NavLink>
      <HelpModeToggle onToggle={close} />
    </>
  )
  return (
    <header className="app-header border-b border-border bg-surface-overlay/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="app-header-shell flex items-center gap-3 px-4 py-3 sm:gap-6">
        <NavLink to="/datasets" title="Back to the datasets page" onClick={goHome}
          className="shrink-0 whitespace-nowrap bg-gradient-primary bg-clip-text text-base font-bold text-transparent no-underline">
          LoRA Dataset Studio
        </NavLink>
        <div className="ml-auto flex items-center gap-1">
          <WhatsNewButton />
          <CheckUpdatesButton />
          <button type="button" onClick={() => setOpen((v) => !v)}
            aria-expanded={open} aria-label={open ? 'Close navigation menu' : 'Open navigation menu'}
            aria-controls="mobile-navigation"
            className="rounded-md p-2 text-content-muted hover:text-content hover:bg-surface-raised">
            {open
            ? <X aria-hidden="true" className="h-5 w-5" />
            : <Menu aria-hidden="true" className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-navigation" aria-label="Main navigation"
          className="app-mobile-nav absolute inset-x-0 top-full flex flex-col gap-1 overflow-y-auto overscroll-contain border-t border-border bg-surface-overlay px-4 py-2 shadow-lg">
          {panelLinks}
          {/* Mounted only while the panel is open, so a phone pays for the
              poll exactly while someone is looking at the answer. */}
          <div className="px-2 py-1.5">{machineLoad}</div>
        </nav>
      )}
    </header>
  )
}

/** Cached installation-aware check, shared with the nav badge and Settings:
 * Git compares its branch; packaged installs compare releases.
 * The status reaches it through 'lds:update-available' (the update entry's own
 * check, on mount and hourly, and Settings › Updates) — it asks the server
 * nothing itself. It OPENS only when that entry is pressed. */
function UpdateBanner() {
  const [info, setInfo] = useState(null)
  const [shown, setShown] = useUpdateNoticeShown()
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'
  const [error, setError] = useState(null)
  useEffect(() => {
    const onFound = (e) => {
      if (e.detail?.update_available) setInfo(e.detail)
      else if (e.detail?.ok && !e.detail.reason) setInfo(null)
    }
    window.addEventListener('lds:update-available', onFound)
    return () => window.removeEventListener('lds:update-available', onFound)
  }, [])

  // Wait for the re-execed server (not the old process still answering), then
  // hard-reload so the new frontend/dist loads. Mirrors the Settings "Updates" card.
  const waitForHealthAndReload = async () => {
    if (await waitForRelaunchedServer()) { window.location.reload(); return }
    setApplying(false); setPhase('')          // gave up after 15 min
  }

  // One-click pull + restart, same backend action as the Settings card. A packaged
  // build (no git) comes back {manual:true} → fall back to the download page.
  const apply = async () => {
    // Docker owns /app as image content. The action is hidden below, but keep a
    // hard guard so an already-bound/stale callback cannot call the endpoint.
    if (installMode(info) === 'docker') return
    // Pinokio owns start/stop: an in-app restart would orphan the server.
    if (installMode(info) === 'pinokio') return
    setApplying(true); setPhase('pulling'); setError(null)
    try {
      const res = await postJson('/api/update/apply', {})
      if (res.restarting) {
        setPhase('restarting')
        waitForHealthAndReload()              // not awaited: the banner shows "restarting…"
      } else if (res.manual) {
        window.open(res.url || info.url, '_blank', 'noreferrer')
        setApplying(false); setPhase('')
      } else {
        setApplying(false); setPhase('')
        setError(res.reason || (res.ok ? null : 'Update failed'))
      }
    } catch (e) {
      setApplying(false); setPhase('')
      setError(e.message || 'Update failed')
    }
  }

  if (!info || !shown) return null
  const dockerMode = installMode(info) === 'docker'
  const pinokioMode = installMode(info) === 'pinokio'
  return (
    <div className="mx-auto max-w-5xl px-4 pt-3">
      <div role="status"
        className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm">
        <ArrowUp aria-hidden="true" className="h-4 w-4 shrink-0 text-primary" />
        {applying ? (
          <span className="text-content">
            {phase === 'restarting'
              ? '↻ Updated — the app is restarting. This page reloads automatically when it’s back…'
              : '⬇ Pulling the latest version…'}
          </span>
        ) : (
          <>
            <span className="text-content">
              Update available — <span className="font-semibold">
                {info.latest
                  ? `v${info.latest}`
                  : info.behind
                    ? `${info.behind} new commit${info.behind === 1 ? '' : 's'}`
                    : 'a new version'}
              </span> (you run {versionLabel(info)}).
            </span>
            {dockerMode ? (
              <DockerUpdateInstructions status={info} />
            ) : pinokioMode ? (
              <PinokioUpdateInstructions />
            ) : (
              <>
                <button type="button" onClick={apply}
                  className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 transition-transform hover:-translate-y-px">
                  Update &amp; restart
                </button>
                {/* Download link only for packaged builds (a git checkout updates in
                    place via the button — a release ZIP would be the wrong artifact). */}
                {!info.is_git && (
                  <a href={info.url} target="_blank" rel="noreferrer"
                    className="text-primary underline">
                    Download
                  </a>
                )}
              </>
            )}
            {error && <span className="text-rose-300">{error}</span>}
            <button type="button"
              onClick={() => setShown(false)}
              aria-label="Dismiss update notice"
              className="ml-auto px-1.5 text-content-subtle hover:text-content">✕</button>
          </>
        )}
      </div>
    </div>
  )
}

/* Onboarding + the background setup re-check both live in SetupHealthNotice:
 * they are the same decision seen from two sides ("has this install ever been
 * seen working?"), and splitting them meant asking the server twice and letting
 * the two answers disagree. A never-verified backend still gets the classic
 * once-per-session redirect to the wizard; a verified one is never interrupted
 * again and re-verifies quietly in the background. */

/* What the content area shows for the instant a page's chunk is in flight —
   first navigation to a route, or the one reload after an update. Quiet on
   purpose: the shell around it is already painted, so a big spinner would
   shout about a wait that is usually under a second. */
function PageLoading() {
  return (
    <div className="flex items-center justify-center py-24" role="status" aria-label="Loading this page">
      <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-border border-t-content-muted" aria-hidden />
    </div>
  )
}

function Shell() {
  const { pathname } = useLocation();
  const wideWorkspaceRoute = pathname === '/canvas' || pathname === '/bank'
    || pathname === '/gallery' || pathname === '/plugins';
  /* 🖼 THE BOARD IS THE WHOLE SCREEN. The canvas is not a document with a
     picture in it — it is a surface you pan and zoom, and every pixel the page
     keeps for itself is a pixel of board you have to pan to reach. So `/canvas`
     alone (the Bank is a scrolling GRID and needs its reading measure) drops
     the 1800-px measure and the generous gutters for a near edge-to-edge 8/12-px
     one, and turns the shell into a column exactly one viewport tall so the
     frame inside it can claim everything left under the header.

     `h-svh`, not `h-screen`: on a phone `100vh` is the LARGEST viewport, the one
     you only get once the URL bar has scrolled away — so `h-screen` here would
     put the bottom of the board under the browser chrome on every load, which
     is the exact fold bug the frame's old `vh` heights were fighting.

     …and below `sm` the gutter goes to ZERO. 8 px a side reads as a considered
     margin on a desktop; on a 360-px phone it is 16 px of the 328 the toolbar
     has to fit in, and it was the difference between that toolbar being one row
     and being two. The frame keeps its own border, so the board still ends
     somewhere visible — it just ends at the edge of the screen, which is where
     a surface you pan and zoom should end. */
  const boardRoute = pathname === '/canvas';
  /* The dataset library and the workspace of an open dataset: wider than a
     reading page, narrower than the walls of images above. Inside the 1024-px
     measure the workspace's own rail left the work 704 px — 37 % of a 1920-px
     screen — which is what made every section read as a crowded strip. 1480 px
     leaves it about 1160. Reading pages (Settings, Setup, the Guide) keep the
     1024-px measure: their lines must not get longer. */
  const datasetRoute = pathname === '/datasets';
  return (
    /* The left padding is the side menu's width (NavBar, fixed at the left edge
       from `xl`): the shell steps aside, the menu does not cover the page. It
       reads the same variable as the menu, so it follows when the menu folds. */
    <div className={`app-shell xl:pl-[var(--app-sidebar-w)] ${boardRoute ? 'flex h-svh flex-col' : ''}`}>
      <NavBar />
      <WhatsNewModal />
      {/* Above the update banner: "can I reach the server at all" outranks
          "there is a newer version". Same ladder for the setup notice: a part of
          the install that stopped working matters more than a new version being
          out, and less than not reaching the server at all. */}
      <ConnectionBanner />
      {/* Below the setup notice, above the update banner: a paused ComfyUI job
          blocks work right now, which outranks "a newer version exists" and is
          outranked by a broken install. */}
      <SetupHealthNotice />
      <ComfyRecoveryBanner />
      <UpdateBanner />
      <MembersUpdateBanner />
      <UsageStatisticsBanner />
      <main id="main-content" tabIndex={-1}
        className={boardRoute
          ? 'flex min-h-0 w-full flex-1 flex-col p-0 sm:px-3 sm:py-3'
          : wideWorkspaceRoute
            ? 'mx-auto w-full max-w-[1800px] px-3 py-4 sm:px-4 sm:py-6'
            : datasetRoute
              ? 'mx-auto w-full max-w-[1480px] px-4 py-4'
              : 'mx-auto max-w-5xl px-4 py-6'}>
        {/* The Suspense sits INSIDE the shell on purpose: a page chunk loading
            on first navigation swaps only the content area, while the nav, the
            banners and the queue dock stay put — wrapping <Routes> instead
            made the whole chrome blink away on every first visit. */}
        <Suspense fallback={<PageLoading />}>
          <SetupJourneyNotice />
          <Outlet />
        </Suspense>
      </main>
      <TipHost />
      {/* One ComfyUI, one queue, fed by every surface — so the dock that shows
          it is mounted once here rather than on the screen that happens to have
          queued the work. Silent while the queue is empty (GitHub #44). */}
      <GenerationQueueDock />
    </div>
  )
}

function AppInner() {
  useSyncExternalStore(subscribePlugins, pluginRevision, pluginRevision)
  useEffect(() => watchMembership(), [])
  const toast = useToast()
  useEffect(() => { setToastRef(toast) }, [toast])
  return (
    <>
      <a
        href="#main-content"
        className="skip-link"
        onClick={(e) => {
          e.preventDefault();
          const el = document.getElementById('main-content');
          if (el) { el.focus(); el.scrollIntoView(); }
        }}
      >
        Skip to main content
      </a>
      <HashRouter>
        <UsageStatisticsProvider>
        <HelpModeProvider>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<Navigate to="/datasets" replace />} />
            <Route path="/datasets" element={<DatasetPage />} />
            <Route path="/bank" element={<BankPage />} />
            {/* /video-bank and /video-dataset/:id are the video plugin's routes
                (bundled/video/frontend/index.js) — rendered with the plugins' below. */}
            <Route path="/guide" element={<GuidePage />} />
            <Route path="/guide/getting-help" element={<Navigate to="/help" replace />} />
            <Route path="/guide/:section" element={<GuidePage />} />
            <Route path="/help" element={<GuidePage helpOnly />} />
            <Route path="/studio" element={<StudioPage />} />
            <Route path="/dataset/studio/:id" element={<StudioPage />} />
            <Route path="/cloud" element={<CloudRunsPage />} />
            <Route path="/gallery" element={<GalleryPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/settings/:section" element={<SettingsPage />} />
            <Route path="/setup" element={<SetupPage />} />
            <Route path="/plugins" element={<PluginsPage />} />
            <Route path="/plugins/:pluginId/settings" element={<PluginSettingsPage />} />
            {/* Pages an enabled plugin contributes: each one its own lazy chunk,
                registered before React mounted (plugins/loadPlugins.js). */}
            {pluginRoutes().map((r) => (
              <Route key={`${r.plugin}:${r.path}`} path={r.path} element={<PluginRoute routeKey={`${r.plugin}:${r.path}`} page={r.page} />} />
            ))}
            {['/canvas', '/video-bank', '/video-dataset/:id'].filter(path => !pluginRoutes().some(route => route.path === path)).map(path => (
              <Route key={path} path={path} element={<UnavailablePluginPage />} />
            ))}
            <Route path="/p/:pluginId/*" element={<UnavailablePluginPage />} />
            <Route path="*" element={<Navigate to="/datasets" replace />} />
          </Route>
        </Routes>
        </HelpModeProvider>
        </UsageStatisticsProvider>
      </HashRouter>
    </>
  )
}

export default function App() {
  return (
    // Root error boundary — outermost so it also catches crashes thrown from
    // the providers themselves.
    <ErrorBoundary showReload>
      <JobsProvider>
        <ToastProvider>
          <CapabilitiesProvider>
            <AppInner />
          </CapabilitiesProvider>
        </ToastProvider>
      </JobsProvider>
    </ErrorBoundary>
  )
}
