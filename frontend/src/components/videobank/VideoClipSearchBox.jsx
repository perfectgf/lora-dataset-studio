import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { HelpBadge } from '../../help/HelpMode'
import { videoSearchUrl } from './videoBankApi'
import {
  searchUnavailableReason, summarize, readinessHint, pendingLabel,
  suggestPushDown, limitsSentence, VIDEO_CLIP_LIMITS, searchBasisNote,
  captionModelNote,
} from './videoClipSearch'

/** 🔎 Find scenes — type a word, get the shots that look like it.
 *
 * A THIN component on purpose: every sentence it renders is computed in
 * videoClipSearch.js, where `node --test` can hold it to account. What lives
 * here is the shape of the box and when it talks to the server.
 *
 * IT NEVER SEARCHES AS YOU TYPE. The first search of a session loads CLIP (about
 * ten seconds, measured), and after that each phrase is ~20 ms — but a debounce
 * would still fire on "a", "a r", "a re" and spend the load on a prefix nobody
 * meant. Submit is explicit, which also makes the cost attributable.
 *
 * The readiness line is fetched BEFORE the click, from the image lane's own
 * status route (one encoder, one query cache, app-wide): a first search that
 * takes ten seconds with no warning reads as a freeze, and the same route is
 * what tells us this install cannot run CLIP at all — in which case the panel
 * says so instead of offering a pass that would 503 for the same reason.
 */
export default function VideoClipSearchBox({
  bankId, counts, busy, onResult, onClear, onRunPass, result, searching,
  captionModel,
}) {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')
  const [showLimits, setShowLimits] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    let alive = true
    apiFetch('/api/bank/text-search/status', { background: true })
      .then((d) => { if (alive) setStatus(d) })
      .catch(() => {})
    return () => { alive = false }
  }, [])

  const blocked = searchUnavailableReason(counts, status)
  const negated = suggestPushDown(query)

  const submit = async (e) => {
    e?.preventDefault?.()
    const q = query.trim()
    if (!q || blocked) return
    setError('')
    onResult?.(null, true)
    try {
      const d = await apiFetch(videoSearchUrl(bankId, { q }),
        { background: true })
      onResult?.(d, false)
    } catch (err) {
      onResult?.(null, false)
      // The server distinguishes "this bank cannot answer yet" (400) from "this
      // install cannot answer at all" (503) — keep them distinct here too, since
      // they send the user to two different places.
      setError(err?.message || 'The search could not be run.')
    }
  }

  const clear = () => {
    setQuery('')
    setError('')
    onClear?.()
    inputRef.current?.focus()
  }

  return (
    <section className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-content">🔎 Find scenes</h2>
        <HelpBadge topic="video-bank-search" />
        <span className="text-xs text-content-subtle">
          {Number(counts?.embedded) || 0} of {Number(counts?.clips) || 0} shots searchable
        </span>
      </div>

      {blocked ? (
        <div className="mt-2 space-y-2">
          <p className="text-xs text-amber-300">⚠ {blocked}</p>
          {/* Only offered when a pass would actually help — an install that
              cannot run CLIP is not one click away from being able to. */}
          {status?.available !== false && (Number(counts?.clips) || 0) > 0 && (
            <button type="button" onClick={onRunPass} disabled={busy}
              className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-40">
              🔎 Find scenes
            </button>
          )}
        </div>
      ) : (
        <form onSubmit={submit} className="mt-2 space-y-2">
          {/* flex-wrap + a min-width on the field: at 400 px the button drops to
              its own line instead of squeezing the input to nothing. */}
          <div className="flex flex-wrap items-center gap-2">
            <input ref={inputRef} type="search" value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="a woman walking on a beach"
              aria-label="Describe the scene to look for"
              className="min-w-[12rem] flex-1 rounded-md border border-border bg-app px-2.5 py-1.5 text-sm text-content placeholder:text-content-subtle" />
            <button type="submit" disabled={searching || !query.trim()}
              className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-40">
              {searching ? pendingLabel(status) : 'Search'}
            </button>
            {result && (
              <button type="button" onClick={clear}
                className="rounded-md border border-border bg-surface-raised px-2.5 py-1.5 text-xs font-semibold text-content hover:bg-surface">
                Clear
              </button>
            )}
          </div>

          {/* The negation trap, caught where it happens. CLIP does not weigh
              "without" — it ignores it — so the results come back full and
              confident carrying exactly what was asked to be gone. */}
          {negated && (
            <p className="text-xs text-amber-300">
              ⚠ “without/no” is ignored by the search. Type
              {' '}<code className="font-mono">-{negated.split(/\s+/).slice(-1)[0]}</code>{' '}
              instead to push it down the ranking.
            </p>
          )}

          {error && <p className="text-xs text-rose-300">✕ {error}</p>}

          {/* WHAT this bank's search can reach. CLIP finds what is visible;
              captions find what HAPPENS. Someone who does not know which halves
              are running cannot read an empty result correctly. */}
          {searchBasisNote(counts) && (
            <p className="text-xs text-content-muted">{searchBasisNote(counts)}</p>
          )}
          {/* WHICH checkpoint wrote them. A caption that talks around what it
              shows is a dataset defect, and the remedy is a different model —
              so the model has to be visible, not implied. */}
          {captionModelNote(captionModel) && (
            <p className={`text-xs ${captionModel?.cached === false
              ? 'text-amber-300' : 'text-content-subtle'}`}>
              {captionModelNote(captionModel)}
            </p>
          )}
          <p className="text-xs text-content-subtle">{readinessHint(status)}</p>

          {result && (
            <p role="status" className="text-xs text-content-muted">{summarize(result)}</p>
          )}

          <button type="button" onClick={() => setShowLimits((v) => !v)}
            aria-expanded={showLimits}
            className="text-left text-xs text-content-subtle underline decoration-dotted">
            {showLimits ? 'Hide what it cannot do' : 'What it cannot do'}
          </button>
          {showLimits && (
            <div className="space-y-1 rounded-md border border-border bg-app/40 p-2">
              <p className="text-xs text-content-muted">{limitsSentence()}</p>
              <ul className="list-disc space-y-0.5 pl-4 text-xs text-content-subtle">
                {VIDEO_CLIP_LIMITS.map((l) => <li key={l}>{l}</li>)}
              </ul>
            </div>
          )}
        </form>
      )}
    </section>
  )
}
