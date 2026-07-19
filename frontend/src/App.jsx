import { useEffect, useState } from 'react'
import { HashRouter, Routes, Route, Navigate, Outlet, NavLink, useNavigate } from 'react-router-dom'
import { apiFetch, postJson } from './api/fetchClient'
import { JobsProvider } from './context/JobsContext'
import { ToastProvider, useToast } from './components/common/Toast'
import { CapabilitiesProvider, useCapabilities } from './context/CapabilitiesContext'
import { setToastRef } from './api/fetchClient'
import ErrorBoundary from './components/common/ErrorBoundary'
import { WhatsNewButton, WhatsNewModal } from './components/common/WhatsNew'
import DatasetPage from './pages/DatasetPage'
import BankPage from './pages/BankPage'
import StudioPage from './pages/StudioPage'
import SettingsPage from './pages/SettingsPage'
import SetupPage from './pages/SetupPage'
import GuidePage from './pages/GuidePage'
import CloudRunsPage from './pages/CloudRunsPage'
import { recommendedMet } from './hooks/useSetupSteps'
import { HelpModeProvider, useHelpMode, TipHost } from './help/HelpMode'
import { useI18n } from './i18n/I18nContext'

const NAV_ITEM_BASE =
  'px-3 py-1.5 rounded-md text-sm font-medium no-underline transition-colors'
const navItemClass = ({ isActive }) =>
  `${NAV_ITEM_BASE} ${
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
  const { t } = useI18n()
  const [busy, setBusy] = useState(false)
  const [available, setAvailable] = useState(false)
  useEffect(() => {
    let alive = true
    const autoCheck = async () => {
      try {
        const d = await apiFetch('/api/update/check?auto=1')
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
        toast.success(`${t('updates.available')} — v${d.latest || d.remote_sha || 'new'}`)
      } else if (d?.ok) {
        toast.info(t('updates.upToDate', { version: d.current }))
      } else {
        toast.error(d?.reason || t('updates.couldNotCheck'))
      }
    } catch (e) {
      toast.error(e?.message || t('updates.checkFailed'))
    } finally {
      setBusy(false)
    }
  }
  return (
    <button type="button" onClick={check} disabled={busy}
      title={available ? t('updates.availableReview') : t('updates.check')}
      className={`${NAV_ITEM_BASE} relative ${available
        ? 'text-emerald-300 hover:text-emerald-200'
        : 'text-content-muted hover:text-content'} hover:bg-surface-raised disabled:opacity-50`}>
      <span aria-hidden>{busy ? '⏳' : '⬆'}</span>
      {available && (
        <span aria-hidden className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-emerald-400" />
      )}
      <span className="sr-only">{available ? t('updates.available') : t('updates.check')}</span>
    </button>
  )
}

/** Header toggle for Help mode. When on, every instrumented heading/action
 * shows a "?" badge that jumps to the matching spot in the guide. aria-pressed
 * spells the state out; the indigo ring makes "on" visually unmistakable. */
function HelpModeToggle({ onToggle }) {
  const { enabled, toggle } = useHelpMode()
  const { t } = useI18n()
  return (
    <button type="button" onClick={() => { toggle(); onToggle?.() }}
      aria-pressed={enabled}
      title={enabled ? t('helpMode.on') : t('helpMode.off')}
      className={`${NAV_ITEM_BASE} inline-flex items-center gap-1.5 ${enabled
        ? 'bg-indigo-500/20 text-indigo-200 ring-1 ring-inset ring-indigo-400/50'
        : 'text-content-muted hover:text-content hover:bg-surface-raised'}`}>
      <span aria-hidden className="grid h-4 w-4 place-items-center rounded-full border border-current text-[10px] font-bold leading-none">?</span>
      <span>{t('nav.helpMode')}</span>
    </button>
  )
}

function LanguageSwitcher({ onChange }) {
  const { locale, locales, setLocale, t } = useI18n()
  return (
    <label
      title={t('language.label')}
      className={`${NAV_ITEM_BASE} inline-flex items-center gap-1.5 text-content-muted hover:bg-surface-raised hover:text-content`}>
      <span aria-hidden>🌐</span>
      <span className="sr-only">{t('language.label')}</span>
      <select value={locale}
        onChange={(event) => { setLocale(event.target.value); onChange?.() }}
        aria-label={t('language.label')}
        className="max-w-28 cursor-pointer border-0 bg-transparent p-0 text-sm font-medium text-inherit outline-none">
        {locales.map((option) => (
          <option key={option.code} value={option.code} className="bg-surface text-content">
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function NavBar() {
  const { caps } = useCapabilities()
  const { t } = useI18n()
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
  const navLinks = (
    <>
      <NavLink to="/datasets" className={navItemClass} onClick={() => setOpen(false)}>{t('nav.datasets')}</NavLink>
      {/* Bank sits right after Datasets: it FEEDS them (triage a big unsorted
          folder, then promote the keepers into a dataset). */}
      <NavLink to="/bank" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1"><span aria-hidden>🗃️</span> {t('nav.bank')}
          <span className="px-1 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.5625rem] font-semibold uppercase tracking-wide leading-none">{t('common.beta')}</span>
        </span>
      </NavLink>
      {/* Unified runs hub (cloud + local history) — useful as soon as ANY
          training path exists, not just the cloud one. */}
      {(caps.cloud_training || caps.training_visible) && (
        <NavLink to="/cloud" className={navItemClass} onClick={() => setOpen(false)}>
          <span className="inline-flex items-center gap-1"><span aria-hidden>🏋️</span> {t('nav.runs')}</span>
        </NavLink>
      )}
      {caps.studio_visible && (
        <NavLink to="/studio" className={navItemClass} onClick={() => setOpen(false)}>{t('nav.studio')}</NavLink>
      )}
      <NavLink to="/guide" className={navItemClass} onClick={() => setOpen(false)}>{t('nav.guide')}</NavLink>
      <NavLink to="/setup" className={navItemClass} onClick={() => setOpen(false)}>
        <span className="inline-flex items-center gap-1">
          {t('nav.setup')}
          {!recommendedMet(caps) && <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-primary" />}
        </span>
      </NavLink>
      <NavLink to="/settings" className={navItemClass} onClick={() => setOpen(false)}>{t('nav.settings')}</NavLink>
      <NavLink to="/help" className={navItemClass} onClick={() => setOpen(false)}>{t('nav.help')}</NavLink>
      <HelpModeToggle onToggle={() => setOpen(false)} />
      <LanguageSwitcher onChange={() => setOpen(false)} />
    </>
  )
  return (
    <header className="border-b border-border bg-surface-overlay/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3 sm:gap-6">
        <NavLink to="/datasets" title={t('nav.backToDatasets')} onClick={goHome}
          className="shrink-0 whitespace-nowrap bg-gradient-primary bg-clip-text text-base font-bold text-transparent no-underline">
          LoRA Dataset Studio
        </NavLink>
        {/* Workflow first (make → train in cloud → test), docs/config last. */}
        <nav className="hidden md:flex gap-1" aria-label={t('nav.main')}>
          {navLinks}
          <WhatsNewButton />
          <CheckUpdatesButton />
        </nav>
        <div className="ml-auto flex items-center gap-1 md:hidden">
          <WhatsNewButton />
          <CheckUpdatesButton />
          <button type="button" onClick={() => setOpen((v) => !v)}
            aria-expanded={open} aria-label={open ? t('nav.closeMenu') : t('nav.openMenu')}
            className="rounded-md p-2 text-content-muted hover:text-content hover:bg-surface-raised">
            <span aria-hidden className="block text-lg leading-none">{open ? '✕' : '☰'}</span>
          </button>
        </div>
      </div>
      {open && (
        <nav aria-label={t('nav.mainMobile')}
          className="flex flex-col gap-1 border-t border-border px-4 py-2 md:hidden">
          {navLinks}
        </nav>
      )}
    </header>
  )
}

/** One-shot update banner: the server caches the GitHub release check 6 h, the
 * banner shows once per browser session and is dismissible. Silent when the
 * feed is unreachable (offline / no public release yet). */
function UpdateBanner() {
  const { t } = useI18n()
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
        setError(res.reason || (res.ok ? null : t('updates.failed')))
      }
    } catch (e) {
      setApplying(false); setPhase('')
      setError(e.message || t('updates.failed'))
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
              ? t('updates.restarting')
              : t('updates.pulling')}
          </span>
        ) : (
          <>
            <span className="text-content">
              {t('updates.available')} — <span className="font-semibold">
                {info.latest
                  ? `v${info.latest}`
                  : info.behind
                    ? t('updates.newCommits', { count: info.behind })
                    : t('updates.newVersion')}
              </span> ({t('updates.currentVersion', { version: info.current })}).
            </span>
            <button type="button" onClick={apply}
              className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white transition-transform hover:-translate-y-px">
              {t('updates.apply')}
            </button>
            {/* Download link only for packaged builds (a git checkout updates in
                place via the button — a release ZIP would be the wrong artifact). */}
            {!info.is_git && (
              <a href={info.url} target="_blank" rel="noreferrer"
                className="text-emerald-300 underline">
                {t('updates.download')}
              </a>
            )}
            {error && <span className="text-rose-300">{error}</span>}
            <button type="button"
              onClick={() => { setInfo(null); sessionStorage.setItem('updateBannerDismissed', '1') }}
              aria-label={t('updates.dismiss')}
              className="ml-auto px-1.5 text-content-subtle hover:text-content">✕</button>
          </>
        )}
      </div>
    </div>
  )
}

// sessionStorage key shared with SetupPage's "Skip setup" link (defense in depth).
const SETUP_REDIRECT_KEY = 'lds_setup_redirected'

/** Onboarding: a never-configured backend (no config.json yet) sends the
 * user straight to Settings instead of a workspace with nothing wired up.
 * Fires AT MOST ONCE per browser session: `caps.configured` stays false for
 * the whole session once the user skips setup (or just navigates away without
 * finishing it), so re-running this on every render would bounce every later
 * navigation — including "Skip setup" and a manual click on Settings — straight
 * back to #/setup, trapping the user. The sessionStorage flag remembers that
 * the redirect already happened; it dies with the tab, so a NEW tab (or next
 * browser session) re-offers Setup once — fine; an in-session trap is not. */
function OnboardingRedirect() {
  const { caps, loading } = useCapabilities()
  const navigate = useNavigate()
  useEffect(() => {
    if (loading || caps.configured) return
    if (sessionStorage.getItem(SETUP_REDIRECT_KEY)) return
    sessionStorage.setItem(SETUP_REDIRECT_KEY, '1')
    navigate('/setup', { replace: true })
  }, [loading, caps.configured, navigate])
  return null
}

function Shell() {
  return (
    <>
      <NavBar />
      <OnboardingRedirect />
      <WhatsNewModal />
      <UpdateBanner />
      <main id="main-content" tabIndex={-1} className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
      <TipHost />
    </>
  )
}

function AppInner() {
  const toast = useToast()
  const { t } = useI18n()
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
        {t('nav.skipToContent')}
      </a>
      <HashRouter>
        <HelpModeProvider>
        <Routes>
          <Route element={<Shell />}>
            <Route path="/" element={<Navigate to="/datasets" replace />} />
            <Route path="/datasets" element={<DatasetPage />} />
            <Route path="/bank" element={<BankPage />} />
            <Route path="/guide" element={<GuidePage />} />
            <Route path="/guide/getting-help" element={<Navigate to="/help" replace />} />
            <Route path="/guide/:section" element={<GuidePage />} />
            <Route path="/help" element={<GuidePage helpOnly />} />
            <Route path="/studio" element={<StudioPage />} />
            <Route path="/dataset/studio/:id" element={<StudioPage />} />
            <Route path="/cloud" element={<CloudRunsPage />} />
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
