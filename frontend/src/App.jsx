import PluginSlot from './plugins/PluginSlot.jsx'
import { Suspense, useEffect, useState, useSyncExternalStore } from 'react'
import { HashRouter, Routes, Route, Navigate, Outlet, NavLink, useLocation } from 'react-router'
import { Archive, ArrowUp, Dumbbell, FlaskConical, FolderOpen, Images, Loader2, Menu, Settings, X } from 'lucide-react'
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
import { useMediaQuery } from './hooks/useMediaQuery'
import { versionLabel } from './utils/versionLabel'
import { useTrainingActivity } from './hooks/useTrainingActivity'
import { activityLabel } from './utils/trainingActivity'
import { installMode, membersUpdateAvailable, patreonCheckFailure, updateCheckFeedback } from './components/settings/updateStatus'
import { settingsLinkHref } from './components/common/settingsLinkHref'
import { waitForRelaunchedServer } from './utils/serverRestart'
import { UsageStatisticsProvider } from './usage/UsageStatisticsContext'
import UsageStatisticsBanner from './usage/UsageStatisticsBanner'

// px-2 up to `lg`: the desktop bar starts at `md` (768 px) and now carries five
// workspaces (Datasets · Bank · Runs · Canvas · Test Studio) plus the utility
// icons. At the old px-3 that row overflowed the viewport at exactly 768 and
// clipped the What's-new button off the right edge. Nothing is hidden — the
// items simply breathe less until there is room for it.
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

/** Nav action (right of Settings): force an update check and give immediate
 * feedback — a toast when up to date, and the actionable UpdateBanner (with the
 * one-click "Update & restart") when there is an update.
 * AUTO-DETECTION: on mount (and every 6 h while the tab stays open) it runs the
 * git-aware check — server-side TTL cache keeps the network cost to one fetch
 * per 6 h across all page loads. An available update lights a dot on the button
 * and surfaces the UpdateBanner without any click. */
function CheckUpdatesButton() {
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
        // The dot always lights up; the banner only surfaces if the user
        // hasn't dismissed it this session (manual checks clear the flag).
        if (d?.ok && (!d.update_available
            || sessionStorage.getItem('updateBannerDismissed') !== '1')) {
          window.dispatchEvent(new CustomEvent('lds:update-available', { detail: d }))
        }
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
      if (membersUpdateAvailable(m)) sessionStorage.removeItem('membersUpdateBannerDismissed')
      window.dispatchEvent(new CustomEvent('lds:patreon-status', { detail: m }))
      if (d?.update_available) {
        sessionStorage.removeItem('updateBannerDismissed')     // re-show even if dismissed
      }
      const feedback = updateCheckFeedback(d, m)
      toast[feedback.type](feedback.message)
    } catch (e) {
      toast.error(e?.message || 'Update check failed.')
    } finally {
      setBusy(false)
    }
  }
  const lit = available || !!members
  return (
    <button type="button" onClick={check} disabled={busy}
      title={lit ? 'Update available — click to review' : 'Check for updates'}
      className={`${NAV_ITEM_BASE} relative ${lit
        ? 'text-emerald-300 hover:text-emerald-200'
        : 'text-content-muted hover:text-content'} hover:bg-surface-raised disabled:opacity-50`}>
      {busy
        ? <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
        : <ArrowUp aria-hidden="true" className="h-4 w-4" />}
      {lit && (
        <span aria-hidden className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-emerald-400" />
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

/** Says that the members version exists. Settings › Updates installs it: its
 * card owns the download, the progress and the restart. */
function MembersUpdateBanner() {
  const [info, setInfo] = useState(null)
  useEffect(() => {
    const onFound = (e) => setInfo(e.detail || null)
    window.addEventListener('lds:members-update', onFound)
    return () => window.removeEventListener('lds:members-update', onFound)
  }, [])
  const version = info?.release?.version || ''
  if (!info || sessionStorage.getItem('membersUpdateBannerDismissed') === version) return null
  return (
    <div className="mx-auto max-w-5xl px-4 pt-3">
      <div role="status"
        className="flex flex-wrap items-center gap-2 rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm">
        <span aria-hidden>⬆</span>
        <span className="text-content">
          LDS members version <span className="font-semibold">{version || 'update'}</span> is available.
        </span>
        <a href={settingsLinkHref('maintenance', 'settings-updates')}
          className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 transition-transform hover:-translate-y-px">
          Install it from Updates
        </a>
        <button type="button"
          onClick={() => { sessionStorage.setItem('membersUpdateBannerDismissed', version); setInfo(null) }}
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
  // 🏋️ Live indicator on Runs: a training can hold the GPU for hours (local) or
  // bill by the minute (cloud), and from any other page nothing said so.
  const activity = useTrainingActivity()
  const activityTitle = activityLabel(activity)
  // Below `md` the horizontal link row has nowhere to go (it used to just wrap
  // mid-word, brand included) -- collapse it into a hamburger-triggered panel
  // instead. navLinks is shared markup: `hidden md:flex` on desktop, only
  // mounted (not just hidden) inside the mobile panel so a closed menu costs
  // nothing extra in the DOM.
  const [open, setOpen] = useState(false)
  const goHome = () => {
    // Home = the datasets LIST: clear the persisted open dataset and tell
    // the mounted page (same-route clicks don't remount) to close it.
    try { localStorage.removeItem('datasetCurrentId'); } catch { /* ignore */ }
    window.dispatchEvent(new CustomEvent('lds:home'))
    setOpen(false)
  }
  // Which grouped menu owns the current route — so the ? / ⚙ triggers can
  // reflect the active-nav style when you're on one of their screens.
  const path = useLocation().pathname
  const helpMenuActive = path === '/guide' || path === '/help'
  const settingsMenuActive = path === '/setup' || path.startsWith('/settings') || path.startsWith('/plugins')
  // 📊 The machine-load readout is mounted ONCE and PLACED (useMediaQuery, the
  // board's own rule): in the desktop bar's utility cluster, or in the mobile
  // panel. Tailwind's `hidden` cannot serve here — a CSS-hidden mount still
  // POLLS, and this is the only thing in the header that would.
  const desktopNav = useMediaQuery('(min-width: 768px)')
  const machineLoad = (
    <PluginSlot slot="resource_monitor.readout" surface="header" />
  )

  // The four workspaces, left-aligned on desktop AND reused (flat) in the
  // mobile panel. Same caps gates in both places.
  const workspaceLinks = (
    <>
      <NavLink to="/datasets" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1"><FolderOpen aria-hidden="true" className="h-3.5 w-3.5" /> Datasets</span>
      </NavLink>
      {/* Bank sits right after Datasets: it FEEDS them (triage a big unsorted
          folder, then promote the keepers into a dataset). */}
      <NavLink to="/bank" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1"><Archive aria-hidden="true" className="h-3.5 w-3.5" /> Bank</span>
      </NavLink>
      {/* Unified runs hub (cloud + local history) — useful as soon as ANY
          training path exists, not just the cloud one. */}
      {(caps.cloud_training || caps.training_visible) && (
        <NavLink to="/cloud" className={navItemClass} onClick={() => setOpen(false)}>
          <span className="inline-flex items-center gap-1"><Dumbbell aria-hidden="true" className="h-3.5 w-3.5" /> Runs
            {activity.running && (
              /* Presence IS the message, so it must not be colour-only: the
                 label is read out and shown on hover/long-press. */
              <span title={activityTitle} aria-label={activityTitle} role="status"
                className="relative inline-flex h-2 w-2 shrink-0">
                <span aria-hidden className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/70" />
                <span aria-hidden className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>
            )}
          </span>
        </NavLink>
      )}
      <NavLink to="/studio" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1"><FlaskConical aria-hidden="true" className="h-3.5 w-3.5" /> Test Studio</span>
      </NavLink>
      {/* 🖼 Gallery — every generated image, one feed. Last: it is where the
          OUTPUT of the other workspaces accumulates, so it reads as the shelf
          at the end of the row. Visible whenever any surface that generates
          is: renders can outlive a broken ComfyUI, so the training gates keep
          it reachable even while the studio gate is down. */}
      {(caps.studio_visible || caps.cloud_training || caps.training_visible) && (
        <NavLink to="/gallery" className={navItemClass} onClick={() => setOpen(false)}>
          <span className="inline-flex items-center gap-1"><Images aria-hidden="true" className="h-3.5 w-3.5" /> Gallery
          </span>
        </NavLink>
      )}
    </>
  )

  // Mobile keeps every destination reachable as a flat stack — no nested
  // dropdowns on touch. Order mirrors the old top bar.
  const mobileLinks = (
    <>
      {workspaceLinks}
      {/* Nav entries an enabled plugin contributes ({ to, label, when(caps) }). */}
      {pluginNavItems(caps).map((item) => (
        <NavLink key={`${item.plugin}:${item.to}`} to={item.to} className={navItemClass} onClick={() => setOpen(false)}>
          {item.label}
        </NavLink>
      ))}
      <NavLink to="/guide" className={navItemClass} onClick={() => setOpen(false)}>Guide</NavLink>
      <NavLink to="/setup" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1">
          Setup
        </span>
      </NavLink>
      <NavLink to="/settings" className={navItemClass} onClick={() => setOpen(false)}>Settings</NavLink>
      <NavLink to="/plugins" className={navItemClass} onClick={() => setOpen(false)}>Plugins</NavLink>
      <NavLink to="/help" className={navItemClass} onClick={() => setOpen(false)}>Help</NavLink>
      <NavLink to="/help?h=support-the-project" className={navItemClass} onClick={() => setOpen(false)}>Support LDS</NavLink>
      <HelpModeToggle onToggle={() => setOpen(false)} />
    </>
  )
  return (
    <header className="app-header border-b border-border bg-surface-overlay/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="app-header-shell mx-auto flex max-w-5xl items-center gap-3 px-4 py-3 sm:gap-6">
        <NavLink to="/datasets" title="Back to the datasets page" onClick={goHome}
          className="shrink-0 whitespace-nowrap bg-gradient-primary bg-clip-text text-base font-bold text-transparent no-underline">
          LoRA Dataset Studio
        </NavLink>
        {/* Desktop: workspaces on the left, utilities grouped into icon menus
            on the right (Guide/Help under ?, Setup/Settings under ⚙). */}
        {/* Desktop CSS places workspaces beside the brand, with machine load
            and utility menus on a separate full-width row. Links and readings
            can wrap at narrow widths without clipping the menu popovers. */}
        <nav className="hidden md:flex flex-1 min-w-0 flex-wrap items-center gap-x-1 gap-y-1.5" aria-label="Main navigation">
          <div className="app-header-workspaces flex min-w-0 shrink grow basis-auto flex-wrap items-center gap-1">
            {workspaceLinks}
            {/* Nav entries an enabled plugin contributes — on the desktop bar as
                on the phone panel below (measured missing here, 2026-09-04: a
                plugin's page was reachable from a phone and from nowhere on a
                desktop). */}
            {pluginNavItems(caps).map((item) => (
              <NavLink key={`${item.plugin}:${item.to}`} to={item.to} className={navItemClass}>
                {item.label}
              </NavLink>
            ))}
          </div>
          <div className="app-header-tools ml-auto flex shrink-0 items-center gap-1">
            {desktopNav && machineLoad}
            <HeaderMenu triggerLabel={<span aria-hidden>?</span>}
              triggerTitle="Help & guide" active={helpMenuActive}>
              {(close) => (
                <>
                  <NavLink to="/guide" role="menuitem" className={menuItemClass} onClick={close}>Guide</NavLink>
                  <NavLink to="/help" role="menuitem" className={menuItemClass} onClick={close}>Help</NavLink>
                  <NavLink to="/help?h=support-the-project" role="menuitem" className={menuItemClass} onClick={close}>Support LDS</NavLink>
                  <HelpModeToggle onToggle={close} />
                </>
              )}
            </HeaderMenu>
            <HeaderMenu triggerLabel={<Settings aria-hidden="true" className="h-4 w-4" />}
              triggerTitle="Setup & settings" active={settingsMenuActive}>
              {(close) => (
                <>
                  <NavLink to="/setup" role="menuitem" className={menuItemClass} onClick={close}>
                    <span className="inline-flex items-center gap-1">
                      Setup
                    </span>
                  </NavLink>
                  <NavLink to="/settings" role="menuitem" className={menuItemClass} onClick={close}>Settings</NavLink>
                  <NavLink to="/plugins" role="menuitem" className={menuItemClass} onClick={close}>Plugins</NavLink>
                </>
              )}
            </HeaderMenu>
            <WhatsNewButton />
            <CheckUpdatesButton />
          </div>
        </nav>
        <div className="ml-auto flex items-center gap-1 md:hidden">
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
        <nav id="mobile-navigation" aria-label="Main navigation (mobile)"
          className="app-mobile-nav absolute inset-x-0 top-full flex flex-col gap-1 overflow-y-auto overscroll-contain border-t border-border bg-surface-overlay px-4 py-2 shadow-lg md:hidden">
          {mobileLinks}
          {/* Mounted only while the panel is open, so a phone pays for the
              poll exactly while someone is looking at the answer. */}
          {!desktopNav && <div className="px-2 py-1.5">{machineLoad}</div>}
        </nav>
      )}
    </header>
  )
}

/** Cached installation-aware check, shared with the nav badge and Settings:
 * Git compares its branch; packaged installs compare releases. */
function UpdateBanner() {
  const [info, setInfo] = useState(null)
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'
  const [error, setError] = useState(null)
  useEffect(() => {
    if (sessionStorage.getItem('updateBannerDismissed') === '1') return
    apiFetch('/api/update/check?auto=1', { background: true })
      .then((d) => { if (d && d.update_available) setInfo(d) })
      .catch(() => { /* best-effort */ })
  }, [])
  // A manual "Check for updates" (nav button) surfaces the banner even after it
  // was dismissed this session, or when the passive mount check found nothing yet.
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

  if (!info) return null
  const dockerMode = installMode(info) === 'docker'
  const pinokioMode = installMode(info) === 'pinokio'
  return (
    <div className="mx-auto max-w-5xl px-4 pt-3">
      <div role="status"
        className="flex flex-wrap items-center gap-2 rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm">
        <span aria-hidden>⬆</span>
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
                    className="text-emerald-300 underline">
                    Download
                  </a>
                )}
              </>
            )}
            {error && <span className="text-rose-300">{error}</span>}
            <button type="button"
              onClick={() => { setInfo(null); sessionStorage.setItem('updateBannerDismissed', '1') }}
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
  return (
    <div className={boardRoute ? 'flex h-svh flex-col' : undefined}>
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
