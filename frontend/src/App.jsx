import { useEffect, useState } from 'react'
import { HashRouter, Routes, Route, Navigate, Outlet, NavLink, useLocation } from 'react-router'
import { apiFetch, postJson } from './api/fetchClient'
import { JobsProvider } from './context/JobsContext'
import { ToastProvider, useToast } from './components/common/Toast'
import { CapabilitiesProvider, useCapabilities } from './context/CapabilitiesContext'
import { setToastRef } from './api/fetchClient'
import ErrorBoundary from './components/common/ErrorBoundary'
import { WhatsNewButton, WhatsNewModal } from './components/common/WhatsNew'
import ConnectionBanner from './components/common/ConnectionBanner'
import SetupHealthNotice from './components/setup/SetupHealthNotice'
import ComfyRecoveryBanner from './components/common/ComfyRecoveryBanner'
import GenerationQueueDock from './components/common/GenerationQueueDock'
import DockerUpdateInstructions from './components/common/DockerUpdateInstructions'
import PinokioUpdateInstructions from './components/common/PinokioUpdateInstructions'
import DatasetPage from './pages/DatasetPage'
import BankPage from './pages/BankPage'
import VideoBankPage from './pages/VideoBankPage'
import StudioPage from './pages/StudioPage'
import SettingsPage from './pages/SettingsPage'
import SetupPage from './pages/SetupPage'
import GuidePage from './pages/GuidePage'
import CloudRunsPage from './pages/CloudRunsPage'
import CanvasPage from './pages/CanvasPage'
import { recommendedMet } from './hooks/useSetupSteps'
import { HelpModeProvider, useHelpMode, TipHost } from './help/HelpMode'
import HeaderMenu from './components/common/HeaderMenu'
import { versionLabel } from './utils/versionLabel'
import { useTrainingActivity } from './hooks/useTrainingActivity'
import { activityLabel } from './utils/trainingActivity'
import { installMode } from './components/settings/updateStatus'

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
  useEffect(() => {
    let alive = true
    const autoCheck = async () => {
      try {
        const d = await apiFetch('/api/update/check?auto=1', { background: true })
        if (!alive) return
        setAvailable(!!d?.update_available)
        // The dot always lights up; the banner only surfaces if the user
        // hasn't dismissed it this session (manual checks clear the flag).
        if (d?.update_available
            && sessionStorage.getItem('updateBannerDismissed') !== '1') {
          window.dispatchEvent(new CustomEvent('lds:update-available', { detail: d }))
        }
      } catch { /* offline — the manual button stays available */ }
    }
    autoCheck()
    // 1 h: the project ships several times a day right now — 6 h let a tab
    // sit stale most of a working day. Server-side TTL matches.
    const t = setInterval(autoCheck, 3600 * 1000)
    return () => { alive = false; clearInterval(t) }
  }, [])
  const check = async () => {
    if (busy) return
    setBusy(true)
    try {
      const d = await apiFetch('/api/update/check?force=1')
      setAvailable(!!d?.update_available)
      if (d?.update_available) {
        sessionStorage.removeItem('updateBannerDismissed')     // re-show even if dismissed
        window.dispatchEvent(new CustomEvent('lds:update-available', { detail: d }))
        toast.success(`Update available — v${d.latest || d.remote_sha || 'new'}`)
      } else if (d?.ok) {
        // On a git checkout the release number alone is misleading — see versionLabel.
        toast.info(`You're up to date — ${versionLabel(d)}`)
      } else {
        toast.error(d?.reason || 'Could not check for updates.')
      }
    } catch (e) {
      toast.error(e?.message || 'Update check failed.')
    } finally {
      setBusy(false)
    }
  }
  return (
    <button type="button" onClick={check} disabled={busy}
      title={available ? 'Update available — click to review' : 'Check for updates'}
      className={`${NAV_ITEM_BASE} relative ${available
        ? 'text-emerald-300 hover:text-emerald-200'
        : 'text-content-muted hover:text-content'} hover:bg-surface-raised disabled:opacity-50`}>
      <span aria-hidden>{busy ? '⏳' : '⬆'}</span>
      {available && (
        <span aria-hidden className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-emerald-400" />
      )}
      <span className="sr-only">{available ? 'Update available' : 'Check for updates'}</span>
    </button>
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
  const settingsMenuActive = path === '/setup' || path.startsWith('/settings')
  const setupNeedsAttention = !recommendedMet(caps)

  // The four workspaces, left-aligned on desktop AND reused (flat) in the
  // mobile panel. Same caps gates in both places.
  const workspaceLinks = (
    <>
      <NavLink to="/datasets" className={navItemClass} onClick={() => setOpen(false)}>Datasets</NavLink>
      {/* Bank sits right after Datasets: it FEEDS them (triage a big unsorted
          folder, then promote the keepers into a dataset). */}
      <NavLink to="/bank" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1"><span aria-hidden>🗃️</span> Bank</span>
      </NavLink>
      {/* Unified runs hub (cloud + local history) — useful as soon as ANY
          training path exists, not just the cloud one. */}
      {(caps.cloud_training || caps.training_visible) && (
        <NavLink to="/cloud" className={navItemClass} onClick={() => setOpen(false)}>
          <span className="inline-flex items-center gap-1"><span aria-hidden>🏋️</span> Runs
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
      {/* ◉ Canvas — the whole training history on one board. It lives next to
          Runs because it answers the same question from the other end: Runs
          lists what happened, the canvas shows how the runs descend from each
          other, across every dataset at once. */}
      {(caps.cloud_training || caps.training_visible) && (
        <NavLink to="/canvas" className={navItemClass} onClick={() => setOpen(false)}>
          <span className="inline-flex items-center gap-1"><span aria-hidden>◉</span> Canvas
            {/* The Beta chip marks the newest surface, not the oldest: the Bank
                has been in daily use for weeks, the canvas ships today.

                It hides ONLY on the tight desktop bar (md→lg), where a fifth
                workspace already overflows the row. It stays visible in the
                mobile panel — a vertical list with room to spare — because that
                is where this app is actually browsed, and a "beta" warning that
                disappears on the reader's own screen warns nobody. */}
            <span className="px-1 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.5625rem] font-semibold uppercase tracking-wide leading-none md:hidden lg:inline">Beta</span>
          </span>
        </NavLink>
      )}
      {caps.studio_visible && (
        <NavLink to="/studio" className={navItemClass} onClick={() => setOpen(false)}>Test Studio</NavLink>
      )}
    </>
  )

  // Mobile keeps every destination reachable as a flat stack — no nested
  // dropdowns on touch. Order mirrors the old top bar.
  const mobileLinks = (
    <>
      {workspaceLinks}
      <NavLink to="/guide" className={navItemClass} onClick={() => setOpen(false)}>Guide</NavLink>
      <NavLink to="/setup" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1">
          Setup
          {setupNeedsAttention && <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-primary" />}
        </span>
      </NavLink>
      <NavLink to="/settings" className={navItemClass} onClick={() => setOpen(false)}>Settings</NavLink>
      <NavLink to="/help" className={navItemClass} onClick={() => setOpen(false)}>Help</NavLink>
      <HelpModeToggle onToggle={() => setOpen(false)} />
    </>
  )
  return (
    <header className="border-b border-border bg-surface-overlay/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3 sm:gap-6">
        <NavLink to="/datasets" title="Back to the datasets page" onClick={goHome}
          className="shrink-0 whitespace-nowrap bg-gradient-primary bg-clip-text text-base font-bold text-transparent no-underline">
          LoRA Dataset Studio
        </NavLink>
        {/* Desktop: workspaces on the left, utilities grouped into icon menus
            on the right (Guide/Help under ?, Setup/Settings under ⚙). */}
        {/* The workspace row is its own flex-wrap box, not a run of direct nav
            children. Measured at 768 px, the row is saturated to the PIXEL with
            the five workspaces it has today: as direct children they cannot
            wrap, so the smallest growth — one more link, one longer label —
            widened the header past the viewport and the whole PAGE scrolled
            sideways, which is the one thing a layout must never do. Wrapping
            keeps any overflow inside this box (a taller header) instead of
            widening the document, and unlike `overflow-x-auto` here it cannot
            clip the ? / ⚙ popovers, which live in the sibling below. Above lg
            everything fits on one line, so a normal desktop is unchanged. */}
        <nav className="hidden md:flex flex-1 min-w-0 items-center gap-1" aria-label="Main navigation">
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
            {workspaceLinks}
          </div>
          <div className="ml-auto flex shrink-0 items-center gap-1">
            <HeaderMenu triggerLabel={<span aria-hidden>?</span>}
              triggerTitle="Help & guide" active={helpMenuActive}>
              {(close) => (
                <>
                  <NavLink to="/guide" role="menuitem" className={menuItemClass} onClick={close}>Guide</NavLink>
                  <NavLink to="/help" role="menuitem" className={menuItemClass} onClick={close}>Help</NavLink>
                  <HelpModeToggle onToggle={close} />
                </>
              )}
            </HeaderMenu>
            <HeaderMenu triggerLabel={<span aria-hidden>⚙</span>}
              triggerTitle="Setup & settings" active={settingsMenuActive} dot={setupNeedsAttention}>
              {(close) => (
                <>
                  <NavLink to="/setup" role="menuitem" className={menuItemClass} onClick={close}>
                    <span className="inline-flex items-center gap-1">
                      Setup
                      {setupNeedsAttention && <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-primary" />}
                    </span>
                  </NavLink>
                  <NavLink to="/settings" role="menuitem" className={menuItemClass} onClick={close}>Settings</NavLink>
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
            className="rounded-md p-2 text-content-muted hover:text-content hover:bg-surface-raised">
            <span aria-hidden className="block text-lg leading-none">{open ? '✕' : '☰'}</span>
          </button>
        </div>
      </div>
      {open && (
        <nav aria-label="Main navigation (mobile)"
          className="flex flex-col gap-1 border-t border-border px-4 py-2 md:hidden">
          {mobileLinks}
        </nav>
      )}
    </header>
  )
}

/** One-shot update banner: the server caches the GitHub release check 6 h, the
 * banner shows once per browser session and is dismissible. Silent when the
 * feed is unreachable (offline / no public release yet). */
function UpdateBanner() {
  const [info, setInfo] = useState(null)
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'
  const [error, setError] = useState(null)
  useEffect(() => {
    if (sessionStorage.getItem('updateBannerDismissed') === '1') return
    apiFetch('/api/update/check')
      .then((d) => { if (d && d.update_available) setInfo(d) })
      .catch(() => { /* best-effort */ })
  }, [])
  // A manual "Check for updates" (nav button) surfaces the banner even after it
  // was dismissed this session, or when the passive mount check found nothing yet.
  useEffect(() => {
    const onFound = (e) => { if (e.detail) setInfo(e.detail) }
    window.addEventListener('lds:update-available', onFound)
    return () => window.removeEventListener('lds:update-available', onFound)
  }, [])

  // Poll /api/health until the re-execed server answers, then hard-reload so the
  // new frontend/dist loads. Mirrors the Settings "Updates" card.
  const waitForHealthAndReload = async () => {
    for (let i = 0; i < 120; i += 1) {
      await new Promise((r) => setTimeout(r, 1000))
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (res.ok) { window.location.reload(); return }
      } catch { /* still restarting — keep waiting */ }
    }
    setApplying(false); setPhase('')          // gave up after ~2 min
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
              </span> (you run v{info.current}).
            </span>
            {dockerMode ? (
              <DockerUpdateInstructions />
            ) : pinokioMode ? (
              <PinokioUpdateInstructions />
            ) : (
              <>
                <button type="button" onClick={apply}
                  className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white transition-transform hover:-translate-y-px">
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

function Shell() {
  const { pathname } = useLocation();
  const wideWorkspaceRoute = pathname === '/canvas' || pathname === '/bank';
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
      <main id="main-content" tabIndex={-1}
        className={boardRoute
          ? 'flex min-h-0 w-full flex-1 flex-col p-0 sm:px-3 sm:py-3'
          : wideWorkspaceRoute
            ? 'mx-auto w-full max-w-[1800px] px-3 py-4 sm:px-4 sm:py-6'
            : 'mx-auto max-w-5xl px-4 py-6'}>
        <Outlet />
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
        <HelpModeProvider>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<Navigate to="/datasets" replace />} />
            <Route path="/datasets" element={<DatasetPage />} />
            <Route path="/bank" element={<BankPage />} />
            {/* Its OWN route rather than a sixth nav item: the desktop bar
                already overflows at 768 px with five workspaces, and these are
                two kinds of material for one job — the lane switch lives on both
                bank pages (components/videobank/BankLaneTabs). */}
            <Route path="/video-bank" element={<VideoBankPage />} />
            <Route path="/guide" element={<GuidePage />} />
            <Route path="/guide/getting-help" element={<Navigate to="/help" replace />} />
            <Route path="/guide/:section" element={<GuidePage />} />
            <Route path="/help" element={<GuidePage helpOnly />} />
            <Route path="/studio" element={<StudioPage />} />
            <Route path="/dataset/studio/:id" element={<StudioPage />} />
            <Route path="/cloud" element={<CloudRunsPage />} />
            <Route path="/canvas" element={<CanvasPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/settings/:section" element={<SettingsPage />} />
            <Route path="/setup" element={<SetupPage />} />
            <Route path="*" element={<Navigate to="/datasets" replace />} />
          </Route>
        </Routes>
        </HelpModeProvider>
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
