/* Background setup verification — the decisions, kept pure so node --test can
   drive them without a DOM.

   The problem: coming back to the app meant going through Setup again. The
   onboarding redirect keyed off a per-TAB sessionStorage flag, so a new tab, a
   new browser session or a second device re-offered the wizard on a machine
   that had been set up for weeks — and the wizard's own machine scan ran from
   scratch every time.

   The rule now: an install the server has already seen working is never
   interrupted. It re-verifies in the BACKGROUND while the user works, says so
   discreetly, and only takes over the screen when something that used to work
   has stopped. "Not everything is installed" is the normal state of nearly
   every install and must never nag — see backend/app/setup_state.py, which is
   the authority for both the verified flag and the regression list. */

// How long the discreet "all good" chip stays before it fades. Long enough to
// be read, short enough that it is not a permanent piece of furniture.
export const SETUP_OK_VISIBLE_MS = 4000

/** Phase of the background check, from what we know so far.
 *  - 'waiting'   capabilities/state not loaded yet — show nothing at all
 *  - 'first-run' the server has never seen this install working → the classic
 *                wizard owns the screen; no chip, no background check
 *  - 'checking'  re-verifying in the background
 *  - 'ok'        re-verified, nothing broken
 *  - 'regressed' something that used to work no longer does → interrupt
 *  - 'skipped'   deliberately not checked (the user is ON the wizard, which
 *                shows all of this in far more detail) — claiming "checked,
 *                everything works" there would be a lie about work not done */
export function setupHealthPhase({ state, checking, result }) {
  if (!state) return 'waiting'
  if (!state.verified) return 'first-run'
  if (result && result.skipped) return 'skipped'
  if (checking || !result) return 'checking'
  return (result.regressions || []).length ? 'regressed' : 'ok'
}

/** Is this install still owing the one Docker choice only Setup can take?
 *
 *  `unconfigured` is emitted for a Docker runtime and nothing else — a native
 *  install reports `local` — so this doubles as the "am I in Docker" test. It
 *  reads a readiness payload, never a URL or a path. */
export function needsDockerDeploymentChoice(readiness) {
  return readiness?.ollama?.mode === 'unconfigured'
}

/** Should we spend a request finding that out?
 *
 *  Only in the one case that would otherwise be waved through: a machine the
 *  server has never seen working, which nevertheless looks configured. The
 *  nominal path — an install already verified — returns before this and keeps
 *  costing exactly one round-trip. */
export function shouldProbeDockerChoice({ state, caps }) {
  if (!state || state.verified) return false
  return !!(caps && caps.configured)
}

/** Does the onboarding redirect still apply?
 *  A verified install is NEVER bounced to the wizard, whatever the tab-local
 *  flag says — that flag is what made this repeat once per browser session.
 *  Everything else keeps the previous behaviour exactly: an unconfigured
 *  backend is offered Setup once, and the flag stops it becoming a trap.
 *
 *  One addition, for the Docker GPU lane: ComfyUI is bundled there, so `caps`
 *  reports `configured` from the very first boot and the wizard was skipped on
 *  a brand-new install — while the launcher stood waiting up to 15 minutes for
 *  an Ollama deployment choice that only Setup can offer. A pending choice
 *  therefore beats `configured`; the once-per-session flag still applies, so
 *  this cannot become a loop. */
export function shouldRedirectToSetup({
  loading, caps, capsKnown = true, state, alreadyRedirected, pendingDockerChoice,
}) {
  if (loading || !state) return false          // never redirect on a guess
  // Capabilities that could not be read look exactly like an unconfigured
  // machine (EMPTY_CAPS). "I could not ask" is not a verdict.
  if (capsKnown === false) return false
  if (state.verified) return false
  if (caps && caps.configured && !pendingDockerChoice) return false
  return !alreadyRedirected
}

/** Human list: "A", "A and B", "A, B and C". */
export function joinLabels(labels) {
  const l = (labels || []).filter(Boolean)
  if (l.length <= 1) return l[0] || ''
  return `${l.slice(0, -1).join(', ')} and ${l[l.length - 1]}`
}

/** The interruption's wording. States WHAT stopped working (the user cannot act
 *  on "a check failed"), and stays honest about the two ways out: fix it in
 *  Setup, or say it was intentional. */
export function regressionNotice(regressions) {
  const list = regressions || []
  if (!list.length) return null
  const labels = joinLabels(list.map((r) => r.label))
  return {
    title: list.length === 1 ? 'A part of your setup stopped working'
      : 'Parts of your setup stopped working',
    body: `${labels} ${list.length === 1 ? 'was' : 'were'} working before and `
      + `${list.length === 1 ? 'is' : 'are'} not responding now.`,
    keys: list.map((r) => r.key),
  }
}

/** The discreet line shown while/after the background check. Null when there is
 *  nothing to say — a first run and a not-yet-loaded app both stay silent. */
export function statusMessage(phase) {
  if (phase === 'checking') return 'Checking your setup in the background…'
  if (phase === 'ok') return 'Setup checked — everything still works.'
  return null
}
