import { useEffect, useState } from 'react'
import { HashRouter, Routes, Route, Navigate, Outlet, NavLink, useNavigate } from 'react-router-dom'
import { apiFetch, postJson } from './api/fetchClient'
import { JobsProvider } from './context/JobsContext'
import { ToastProvider, useToast } from './components/common/Toast'
import { CapabilitiesProvider, useCapabilities } from './context/CapabilitiesContext'
import { setToastRef } from './api/fetchClient'
import ErrorBoundary from './components/common/ErrorBoundary'
import DatasetPage from './pages/DatasetPage'
import StudioPage from './pages/StudioPage'
import SettingsPage from './pages/SettingsPage'
import SetupPage from './pages/SetupPage'
import GuidePage from './pages/GuidePage'
import CloudRunsPage from './pages/CloudRunsPage'
import { recommendedMet } from './hooks/useSetupSteps'

const NAV_ITEM_BASE =
  'px-3 py-1.5 rounded-md text-sm font-medium no-underline transition-colors'
const navItemClass = ({ isActive }) =>
  `${NAV_ITEM_BASE} ${
    isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:text-content hover:bg-surface-raised'
  }`

/** Nav action (right of Settings): force an update check and give immediate
 * feedback — a toast when up to date, and the actionable UpdateBanner (with the
 * one-click "Update & restart") when there is an update. */
function CheckUpdatesButton() {
  const toast = useToast()
  const [busy, setBusy] = useState(false)
  const check = async () => {
    if (busy) return
    setBusy(true)
    try {
      const d = await apiFetch('/api/update/check?force=1')
      if (d?.update_available) {
        sessionStorage.removeItem('updateBannerDismissed')     // re-show even if dismissed
        window.dispatchEvent(new CustomEvent('lds:update-available', { detail: d }))
        toast.success(`Update available — v${d.latest || d.remote_sha || 'new'}`)
      } else if (d?.ok) {
        toast.info(`You're up to date — v${d.current}`)
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
    <button type="button" onClick={check} disabled={busy} title="Check for updates"
      className={`${NAV_ITEM_BASE} text-content-muted hover:text-content hover:bg-surface-raised disabled:opacity-50`}>
      <span aria-hidden>{busy ? '⏳' : '⬆'}</span>
      <span className="sr-only">Check for updates</span>
    </button>
  )
}

function NavBar() {
  const { caps } = useCapabilities()
  return (
    <header className="border-b border-border bg-surface-overlay/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
        <NavLink to="/datasets" title="Back to the datasets page"
          onClick={() => {
            // Home = the datasets LIST: clear the persisted open dataset and tell
            // the mounted page (same-route clicks don't remount) to close it.
            try { localStorage.removeItem('datasetCurrentId'); } catch { /* ignore */ }
            window.dispatchEvent(new CustomEvent('lds:home'));
          }}
          className="bg-gradient-primary bg-clip-text text-base font-bold text-transparent no-underline">
          LoRA Dataset Studio
        </NavLink>
        {/* Workflow first (make → train in cloud → test), docs/config last. */}
        <nav className="flex gap-1" aria-label="Main navigation">
          <NavLink to="/datasets" className={navItemClass}>Datasets</NavLink>
          {caps.cloud_training && (
            <NavLink to="/cloud" className={navItemClass}>
              <span className="inline-flex items-center gap-1"><span aria-hidden>☁️</span> Cloud</span>
            </NavLink>
          )}
          {caps.studio_visible && (
            <NavLink to="/studio" className={navItemClass}>Test Studio</NavLink>
          )}
          <NavLink to="/guide" className={navItemClass}>Guide</NavLink>
          <NavLink to="/setup" className={navItemClass}>
            <span className="inline-flex items-center gap-1">
              Setup
              {!recommendedMet(caps) && <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-primary" />}
            </span>
          </NavLink>
          <NavLink to="/settings" className={navItemClass}>Settings</NavLink>
          <CheckUpdatesButton />
        </nav>
      </div>
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
              Update available — <span className="font-semibold">v{info.latest}</span> (you run v{info.current}).
            </span>
            <button type="button" onClick={apply}
              className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white transition-transform hover:-translate-y-px">
              Update &amp; restart
            </button>
            <a href={info.url} target="_blank" rel="noreferrer"
              className="text-emerald-300 underline">
              Download
            </a>
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

/** Onboarding: a never-configured backend (no config.json yet) sends the
 * user straight to Settings instead of a workspace with nothing wired up. */
function OnboardingRedirect() {
  const { caps, loading } = useCapabilities()
  const navigate = useNavigate()
  useEffect(() => {
    if (!loading && !caps.configured) navigate('/setup', { replace: true })
  }, [loading, caps.configured, navigate])
  return null
}

function Shell() {
  return (
    <>
      <NavBar />
      <OnboardingRedirect />
      <UpdateBanner />
      <main id="main-content" tabIndex={-1} className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </>
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
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<Navigate to="/datasets" replace />} />
            <Route path="/datasets" element={<DatasetPage />} />
            <Route path="/guide" element={<GuidePage />} />
            <Route path="/studio" element={<StudioPage />} />
            <Route path="/dataset/studio/:id" element={<StudioPage />} />
            <Route path="/cloud" element={<CloudRunsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/setup" element={<SetupPage />} />
            <Route path="*" element={<Navigate to="/datasets" replace />} />
          </Route>
        </Routes>
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
