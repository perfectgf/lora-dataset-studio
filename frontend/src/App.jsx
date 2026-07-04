import { useEffect } from 'react'
import { HashRouter, Routes, Route, Navigate, Outlet, NavLink, useNavigate } from 'react-router-dom'
import { JobsProvider } from './context/JobsContext'
import { ToastProvider, useToast } from './components/common/Toast'
import { CapabilitiesProvider, useCapabilities } from './context/CapabilitiesContext'
import { setToastRef } from './api/fetchClient'
import ErrorBoundary from './components/common/ErrorBoundary'
import DatasetsPage from './pages/DatasetsPage'
import StudioPage from './pages/StudioPage'
import SettingsPage from './pages/SettingsPage'

const NAV_ITEM_BASE =
  'px-3 py-1.5 rounded-md text-sm font-medium no-underline transition-colors'
const navItemClass = ({ isActive }) =>
  `${NAV_ITEM_BASE} ${
    isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:text-content hover:bg-surface-raised'
  }`

function NavBar() {
  const { caps } = useCapabilities()
  return (
    <header className="border-b border-border bg-surface-overlay/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
        <span className="bg-gradient-primary bg-clip-text text-base font-bold text-transparent">
          LoRA Dataset Studio
        </span>
        <nav className="flex gap-1" aria-label="Main navigation">
          <NavLink to="/datasets" className={navItemClass}>Datasets</NavLink>
          {caps.studio_visible && (
            <NavLink to="/studio" className={navItemClass}>Test Studio</NavLink>
          )}
          <NavLink to="/settings" className={navItemClass}>Settings</NavLink>
        </nav>
      </div>
    </header>
  )
}

/** Onboarding: a never-configured backend (no config.json yet) sends the
 * user straight to Settings instead of a workspace with nothing wired up. */
function OnboardingRedirect() {
  const { caps, loading } = useCapabilities()
  const navigate = useNavigate()
  useEffect(() => {
    if (!loading && !caps.configured) navigate('/settings', { replace: true })
  }, [loading, caps.configured, navigate])
  return null
}

function Shell() {
  return (
    <>
      <NavBar />
      <OnboardingRedirect />
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
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/studio" element={<StudioPage />} />
            <Route path="/settings" element={<SettingsPage />} />
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
