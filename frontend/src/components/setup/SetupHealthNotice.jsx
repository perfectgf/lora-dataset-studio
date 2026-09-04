/* "Don't make me run Setup again."
 *
 * An install the server has already seen working goes straight to the app. The
 * re-verification — the SAME full capability probe the wizard runs — happens in
 * the background while the user works, and says so in one discreet line that
 * fades on its own. The screen is only taken over when something that used to
 * work has stopped: that is a real failure, and it names WHAT broke, because
 * "a check failed" is not something anyone can act on.
 *
 * A machine the server has never seen working keeps the classic first-run
 * behaviour, redirect included — nothing about a genuine first launch changes.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { setupHealthPhase, shouldRedirectToSetup, regressionNotice, statusMessage,
  needsDockerDeploymentChoice, shouldProbeDockerChoice,
  SETUP_OK_VISIBLE_MS } from '../../hooks/setupHealth'

// sessionStorage key shared with SetupPage's "Skip setup" link (defense in depth).
// Only reached now by an install the server has NEVER seen working; a verified
// one is never redirected, so this flag no longer decides anything for it.
const SETUP_REDIRECT_KEY = 'lds_setup_redirected'

export default function SetupHealthNotice() {
  const { caps, loading, known, refresh } = useCapabilities()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const [state, setState] = useState(null)      // GET /api/setup-state
  const [checking, setChecking] = useState(false)
  const [result, setResult] = useState(null)    // POST /api/setup-state/recheck
  const [hidden, setHidden] = useState(false)   // the "all good" line has faded
  const startedRef = useRef(false)

  // One pass per page load. Runs only once capabilities have ANSWERED (not
  // merely settled), so the cached probe the state endpoint reads is already
  // warm and this costs a round-trip, not a second machine scan.
  useEffect(() => {
    if (loading || !known || startedRef.current) return
    startedRef.current = true
    let alive = true
    ;(async () => {
      // The server's answer, or nothing. A request that failed used to fall
      // back to "never verified" — and a phone coming back to the app after
      // a render, its link reconnecting, had its first requests dropped and
      // was sent through Setup on a machine set up for weeks (2026-09-04).
      // One quiet retry, then silence: the next page load asks again, and
      // a genuine first run is still redirected the moment the server says so.
      let s = null
      for (let attempt = 0; attempt < 2 && alive; attempt += 1) {
        try {
          s = await apiFetch('/api/setup-state', { background: true })
          break
        } catch {
          await new Promise((r) => setTimeout(r, 1500))
        }
      }
      if (!alive || !s) return
      setState(s)
      if (!s.verified) {
        // Only asked for when the answer can change the decision — a Docker
        // install looks `configured` on its first boot because ComfyUI is
        // bundled, and would otherwise never be shown the Ollama choice its
        // own launcher is blocking on.
        let pendingDockerChoice = false
        if (shouldProbeDockerChoice({ state: s, caps })) {
          try {
            pendingDockerChoice = needsDockerDeploymentChoice(
              await apiFetch('/api/setup/runtime-readiness', { background: true }))
          } catch {
            // Older backend without the route: keep the previous behaviour.
          }
          if (!alive) return
        }
        if (shouldRedirectToSetup({
          loading: false, caps, capsKnown: known, state: s, pendingDockerChoice,
          alreadyRedirected: !!sessionStorage.getItem(SETUP_REDIRECT_KEY),
        })) {
          sessionStorage.setItem(SETUP_REDIRECT_KEY, '1')
          navigate('/setup', { replace: true })
        }
        return
      }
      // Already on the wizard: the user is re-checking by hand, on a page that
      // shows far more than this line ever could. Don't race it.
      if (pathname === '/setup') { setResult({ regressions: [], skipped: true }); return }
      setChecking(true)
      try {
        const r = await postJson('/api/setup-state/recheck', {}, { background: true })
        if (!alive) return
        setResult(r)
        // The forced probe just refreshed the server-side cache; this reads it
        // back into the app (no second probe) so every gated screen sees the
        // freshly verified truth.
        refresh()
      } catch {
        if (alive) setResult({ regressions: [] })   // silent: a failed re-check is not a regression
      } finally {
        if (alive) setChecking(false)
      }
    })()
    return () => { alive = false }
  }, [loading, known, caps, navigate, pathname, refresh])

  const phase = setupHealthPhase({ state, checking, result })

  // The reassuring line is not furniture: it fades once it has been read.
  useEffect(() => {
    if (phase !== 'ok') return undefined
    const t = setTimeout(() => setHidden(true), SETUP_OK_VISIBLE_MS)
    return () => clearTimeout(t)
  }, [phase])

  const notice = phase === 'regressed' ? regressionNotice(result.regressions) : null

  const dismiss = useCallback(async () => {
    const keys = notice ? notice.keys : []
    setResult({ regressions: [] })
    try { await postJson('/api/setup-state/dismiss', { keys }) } catch { /* best-effort */ }
  }, [notice])

  if (notice) {
    return (
      <div className="mx-auto max-w-5xl px-4 pt-3">
        <div role="alert"
          className="flex flex-wrap items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm">
          <span aria-hidden="true">⚠</span>
          <span className="text-content">
            <span className="font-semibold">{notice.title}</span> — {notice.body}
          </span>
          <Link to="/setup"
            className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 no-underline transition-transform hover:-translate-y-px">
            Open Setup
          </Link>
          {/* Not a "close": an uninstall can be deliberate, and without a way to
              say so the warning would come back on every single page load. */}
          <button type="button" onClick={dismiss}
            className="text-xs text-content-subtle underline hover:text-content">
            That was on purpose
          </button>
        </div>
      </div>
    )
  }

  const message = hidden ? null : statusMessage(phase)
  if (!message) return null
  return (
    // Bottom-right on desktop, but pinned to both edges on a narrow screen so it
    // never squeezes into a two-word column at 400 px.
    <div className="pointer-events-none fixed bottom-3 left-3 right-3 z-30 flex justify-center sm:left-auto sm:right-4 sm:justify-end">
      <div role="status"
        className="max-w-full rounded-full border border-border bg-surface-raised px-3 py-1.5 text-xs text-content-muted shadow-lg">
        {phase === 'checking' && (
          <span aria-hidden="true"
            className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-border-strong border-t-primary align-[-1px]" />
        )}
        {message}
      </div>
    </div>
  )
}
