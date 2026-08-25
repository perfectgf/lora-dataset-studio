import { useEffect, useMemo, useRef, useState } from 'react'
import { attemptModalSubmit } from '../../utils/submitOutcome.js'
import { lastPassNote, lastPassSentence } from './lastPassNote.js'
import {
  bankPass, passScopeRows, passSelectionAvailability, watermarkSettings,
} from './bankPasses.js'
import {
  DEFAULT_PASS_SCOPE, passBinWarning, passLaunchDisabledReason, passLaunchLabel,
  passScopeCount, passScopeLineLabel, passScopeStatuses,
} from './bankPassScope.js'

/** THE LAUNCH WINDOW every bank pass now opens instead of firing on click.
 *
 * Asked for in these words: "all these passes should be settable to kept,
 * undecided, unkept or all, to choose where they apply — or individually. Each
 * button, once pressed, opens a window showing the available options, with the
 * launch button at the end. This way we can gather all the caption options, for
 * example."
 *
 * ONE component for every pass, because the eight dialogs that already exist in
 * this folder each re-implement the same markup and had already drifted. The
 * pattern kept from LaunchAllDialog, which is the one that got it right:
 *   · backdrop closes on mousedown only when the click STARTED on the backdrop;
 *   · Escape closes, and neither works while a launch is in flight;
 *   · the submission goes through attemptModalSubmit, so a refusal KEEPS THE
 *     WINDOW OPEN WITH ITS INPUTS (ModalRefusalKeepsInput.contract.test.js);
 *   · an unavailable choice stays VISIBLE and disabled WITH ITS REASON — never
 *     removed, never offered-then-ignored.
 *
 * WHAT IS NEW HERE, and it is the point: the card is a flex column, so the launch
 * button lives OUTSIDE the scrolling area. At a 400 px viewport the three blocks
 * are far taller than the screen, and a button that has scrolled off the bottom
 * of a modal is a button that is not there.
 *
 * The window carries THREE blocks, and their separation is the whole design — see
 * the header of bankPasses.js for why a single "settings for this pass" panel
 * would have been a measured lie.
 */

function Block({ title, subtitle, children }) {
  return (
    <section className="rounded-lg border border-border bg-surface p-3 space-y-2">
      <div>
        <h3 className="text-xs font-bold uppercase tracking-wide text-content-muted">{title}</h3>
        {subtitle && <p className="mt-0.5 text-[11px] leading-snug text-content-subtle">{subtitle}</p>}
      </div>
      {children}
    </section>
  )
}

/** A choice that cannot be taken. Rendered, greyed, and carrying its objection —
 *  the alternative (leaving it out) is what made a shipped feature read as an
 *  absent one twice on this screen. */
function DisabledNote({ children }) {
  return (
    <span className="mt-0.5 block text-[11px] leading-snug text-amber-300/90">{children}</span>
  )
}

export default function PassDialog({
  passId, payload, selectionSize = 0, live = false, detectorReady = false,
  semanticEngine = 'clip',
  scope = '', onScope, redo = false, onRedo,
  onClose, onLaunch, children, secondary, extraSettings,
}) {
  const spec = bankPass(passId, { semanticEngine })
  const selection = passSelectionAvailability(passId)
  const rows = useMemo(() => passScopeRows(passId), [passId])
  const lastRun = useMemo(() => lastPassNote(payload, passId), [payload, passId])

  /* A live selection is the FIRST line of THIS RUN, and it wins by default when
     the pass can honour it — the user just picked those images. Turning it off
     falls back to the status scope; the server intersects the two, so the two
     modes are never combined into a contradiction the user cannot see. */
  const canUseSelection = selectionSize > 0 && selection.ok
  const [useSelection, setUseSelection] = useState(canUseSelection)
  /* Scope and redo are OWNED BY THE WORKSPACE, not by this window. Closing a
     dialog must not silently undo a choice the user made — the same reason the
     caption engine/model/register pickers were never allowed to reset either. */
  const setScope = (v) => onScope?.(v)
  const setRedo = (v) => onRedo?.(v)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const dismiss = () => { if (!busy) onClose() }
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') dismiss() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])   // eslint-disable-line react-hooks/exhaustive-deps

  /* The refusal is rendered above the launch button that produced it, and the
     scroll area is brought to its end so it cannot land off-screen. */
  const bodyRef = useRef(null)
  useEffect(() => {
    const el = bodyRef.current
    if (error && el) el.scrollTop = el.scrollHeight
  }, [error])

  if (!spec) return null

  const effectiveSelection = useSelection && canUseSelection ? selectionSize : 0
  const countable = spec.countable !== false
  const runLabel = passLaunchLabel({
    verb: spec.verb, payload, passId, scopeId: scope, redo,
    selectionSize: effectiveSelection, countable,
  })
  const blocked = passLaunchDisabledReason({
    payload, passId, scopeId: scope, redo, selectionSize: effectiveSelection,
    countable, live,
  })
  const binWarning = effectiveSelection ? '' : passBinWarning(scope, {
    free: !!spec.binFree, cost: spec.binCost || '',
  })
  const wmRoute = passId === 'watermark' ? watermarkSettings(detectorReady) : null
  const settings = extraSettings || wmRoute?.settings || spec.settings

  const launch = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    let outcome
    try {
      outcome = await attemptModalSubmit(() => onLaunch({
        scope,
        statuses: effectiveSelection ? null : passScopeStatuses(scope),
        imageIds: effectiveSelection ? 'selection' : null,
        redo,
      }), { fallback: `Could not start ${spec.label}` })
    } finally { setBusy(false) }
    if (outcome.close) onClose()
    else setError(outcome.error)
  }

  return (
    <div role="dialog" aria-modal="true" aria-label={spec.label}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-2 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget) dismiss() }}>
      <div className="flex w-full max-w-lg max-h-[92vh] flex-col overflow-hidden rounded-xl border border-border bg-surface-overlay shadow-2xl">
        <header className="shrink-0 border-b border-border p-4">
          <h2 className="text-base font-bold text-content">{spec.label}</h2>
          <p className="mt-1 text-sm text-content-muted">{spec.what}</p>
          {/* "Did I already run this?" used to be answerable only by running it
              again and watching. See lastPassNote.js for why this line stops at
              what the last run FOUND and leaves "is it still current" to the run
              itself. */}
          {lastRun && (
            <p className="mt-1 text-xs text-content-subtle">↻ {lastPassSentence(lastRun)}</p>
          )}
        </header>

        <div ref={bodyRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3 sm:p-4">
          <Block title="This run"
            subtitle="Where this pass applies, and how many images that is.">
            {/* ① the selection, when there is one. */}
            {selectionSize > 0 && (
              <label className={`flex items-start gap-2 rounded-md border border-border p-2 text-sm ${selection.ok ? 'bg-surface-raised' : 'opacity-70'}`}>
                <input type="radio" name={`scope-${passId}`} className="mt-1"
                  checked={selection.ok && useSelection} disabled={!selection.ok}
                  onChange={() => setUseSelection(true)} />
                <span className="min-w-0">
                  <span className="font-medium text-content">
                    Your selection — up to {selectionSize} image(s)
                  </span>
                  {selection.ok ? (
                    <span className="block text-[11px] leading-snug text-content-subtle">
                      “Up to”, because the selection is narrowed by what this pass still
                      has to do — never widened.
                    </span>
                  ) : <DisabledNote>{selection.reason}</DisabledNote>}
                </span>
              </label>
            )}

            {/* ② the four piles + the default. A pass with a global result shows
                ONE immutable line instead, and the objection under it. */}
            {spec.scopes === true ? rows.map((r) => (
              <label key={r.id || 'default'}
                className={`flex items-start gap-2 rounded-md border border-border p-2 text-sm ${
                  r.ok ? 'bg-surface-raised' : 'opacity-70'}`}>
                <input type="radio" name={`scope-${passId}`} className="mt-1"
                  checked={!(useSelection && canUseSelection) && scope === r.id}
                  disabled={!r.ok}
                  onChange={() => { setUseSelection(false); setScope(r.id) }} />
                <span className="min-w-0">
                  <span className="font-medium text-content">
                    {passScopeLineLabel(payload, passId, r.id, redo)}
                  </span>
                  {r.ok ? (
                    <span className="block text-[11px] leading-snug text-content-subtle">{r.hint}</span>
                  ) : <DisabledNote>{r.reason}</DisabledNote>}
                </span>
              </label>
            )) : (
              <div className="rounded-md border border-border bg-surface-raised p-2 text-sm">
                <p className="m-0 font-medium text-content">
                  {spec.fixedScopeLine}
                  {countable && passScopeCount(payload, passId, DEFAULT_PASS_SCOPE, false) !== null
                    && ` — ${passScopeCount(payload, passId, DEFAULT_PASS_SCOPE, false)} image(s)`}
                </p>
                <DisabledNote>{spec.scopes}</DisabledNote>
              </div>
            )}

            {/* ③ the "do it again" line — where “Rescan all” went. */}
            {spec.redo && (
              <label className="flex items-start gap-2 rounded-md border border-border bg-surface-raised p-2 text-sm">
                <input type="checkbox" className="mt-1" checked={redo}
                  onChange={(e) => setRedo(e.target.checked)} />
                <span className="min-w-0">
                  <span className="font-medium text-content">{spec.redo.label}</span>
                  {spec.redo.note && (
                    <span className="block text-[11px] leading-snug text-content-subtle">
                      {spec.redo.note}
                    </span>
                  )}
                </span>
              </label>
            )}

            {binWarning && (
              <p role="note"
                className="m-0 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-[11px] leading-snug text-amber-200">
                ⚠ {binWarning}
              </p>
            )}

            {/* Per-pass options that belong to THIS RUN — the caption engine,
                model, register and length live here now, not spread under the
                panel where they read as bank-wide settings. */}
            {typeof children === 'function'
              ? children({ scope, redo, selectionSize: effectiveSelection, busy })
              : children}
          </Block>

          <Block title="Settings this pass reads"
            subtitle="Only what the CALCULATION uses. These are global — the same for every bank — unless a line says otherwise.">
            {wmRoute && (
              <p className="m-0 rounded-md border border-indigo-400/40 bg-indigo-500/10 px-2 py-1.5 text-[11px] leading-snug text-content-muted">
                {wmRoute.route}
              </p>
            )}
            {settings.length === 0 ? (
              <p className="m-0 text-sm text-content-muted">
                None — this pass reads no stored setting.
              </p>
            ) : (
              <ul className="m-0 list-disc space-y-1 pl-5 text-sm text-content-muted">
                {settings.map((s) => (
                  <li key={s.name}>
                    <span className="text-content">{s.name}</span>
                    {s.note && (
                      <span className="block text-[11px] leading-snug text-content-subtle">{s.note}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Block>

          <Block title="Not decided here"
            subtitle="Re-read when the grid sorts and flags — retune them any time, no pass needed.">
            <ul className="m-0 list-disc space-y-1 pl-5 text-sm text-content-muted">
              {spec.notHere.map((n) => <li key={n}>{n}</li>)}
            </ul>
            {!!(spec.caveats || []).length && (
              <ul className="m-0 list-disc space-y-1 pl-5 text-sm text-amber-200/90">
                {spec.caveats.map((c) => <li key={c}>{c}</li>)}
              </ul>
            )}
          </Block>
        </div>

        <div className="shrink-0 space-y-2 border-t border-border p-3">
          {error && (
            <div role="alert"
              className="max-h-24 overflow-y-auto rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2">
              <span className="block whitespace-pre-wrap break-words text-xs leading-relaxed text-red-200">
                {error}
              </span>
              <span className="mt-1 block text-[0.625rem] text-content-subtle">
                Your choices are kept — adjust and try again.
              </span>
            </div>
          )}
          {blocked && !error && (
            <p className="m-0 text-[11px] leading-snug text-amber-300/90">{blocked}</p>
          )}
          {typeof secondary === 'function'
            ? secondary({ scope, redo, busy, close: onClose, fail: setError })
            : secondary}
          <div className="flex flex-wrap justify-end gap-2">
            <button type="button" onClick={dismiss} disabled={busy}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50">
              Cancel
            </button>
            <button type="button" onClick={launch} disabled={busy || !!blocked}
              className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50">
              {busy ? 'Starting…' : runLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
