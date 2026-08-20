/**
 * 📊 Coverage advice (idea by @antonp) — a read-only panel. Reads what the
 * passes already computed (framing, person/style clusters, resolution) and says,
 * in plain sentences, what the kept set leans on and what's thin for a good
 * LoRA. Never selects or rejects; warnings first, then gentler notes.
 *
 * Moved out of BankWorkspace.jsx by the Encre redesign, unchanged.
 */
import { FRAMING_BUCKETS } from './bankFacets.js'
import { spreadCoverageNote, spreadReadout } from './coverageVisual.js'
// Shared with the dataset coverage panel on purpose: both render the SAME
// caption-lexicon payload, so the row/summary logic lives in one place rather
// than being copied and left to drift.
import { axisRows, axisSummary } from '../dataset/datasetCoverage.js'

function FramingBar({ framing }) {
  const total = FRAMING_BUCKETS.reduce((a, b) => a + (framing[b.id] || 0), 0)
  if (!total) return null
  const tone = { face: 'bg-teal-400', bust: 'bg-sky-400', body: 'bg-indigo-400',
    back: 'bg-fuchsia-400', unknown: 'bg-content-subtle' }
  return (
    <div className="space-y-1">
      <div className="flex h-2 overflow-hidden rounded bg-surface-raised" role="img"
        aria-label="Framing distribution of the set">
        {FRAMING_BUCKETS.map((b) => (framing[b.id] || 0) > 0 && (
          <div key={b.id} className={tone[b.id]} style={{ width: `${(100 * framing[b.id]) / total}%` }}
            title={`${b.label}: ${framing[b.id]}`} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-content-subtle">
        {FRAMING_BUCKETS.map((b) => (framing[b.id] || 0) > 0 && (
          <span key={b.id}>{b.label} {framing[b.id]}</span>
        ))}
      </div>
    </div>
  )
}

// 👁 What the labels cannot see: how alike the pool actually LOOKS, measured on
// the currently selected semantic cache. A Bank with no usable index shows "Not
// measured" rather than a reassuring colour — the whole point is that silence
// must never read as variety.
function VisualSpread({ visual, total, semanticEngine }) {
  const r = spreadReadout(visual, semanticEngine)
  if (!r) return null
  const tone = r.tone === 'warn' ? 'border-amber-400/50 bg-amber-400/10 text-amber-200'
    : r.tone === 'ok' ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
      : 'border-border bg-surface-raised text-content-subtle'
  const note = spreadCoverageNote(visual, total, semanticEngine)
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-wide text-content-muted">Visual spread</span>
        <span className={`rounded-full border px-2 py-0.5 text-[11px] ${tone}`}>{r.label}</span>
      </div>
      <p className="m-0 text-[11px] text-content-subtle">{r.detail}</p>
      {note && <p className="m-0 text-[11px] text-content-subtle">{note}</p>}
    </div>
  )
}

// The caption-derived axes, rendered from the SAME pure helpers the dataset panel
// uses (imported, not copied) so the two surfaces cannot drift apart.
function VarietyAxes({ variety }) {
  if (!variety || !variety.captioned || !(variety.axes || []).length) return null
  const chip = { ok: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
    thin: 'border-amber-400/50 bg-amber-400/10 text-amber-300',
    gap: 'border-rose-400/50 bg-rose-400/10 text-rose-300',
    none: 'border-border bg-surface-raised text-content-subtle' }
  return (
    <div className="flex flex-col gap-2 border-t border-border pt-2">
      {variety.axes.map((axis) => (
        <div key={axis.id} className="flex flex-col gap-1">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-[11px] uppercase tracking-wide text-content-muted">{axis.label}</span>
            {axis.hint && <span className="text-[11px] text-content-subtle">{axis.hint}</span>}
          </div>
          {/* The chips decorate a sentence a screen reader can read out; the
              sentence is the carrier, never the colour.
              DELIBERATE DIVERGENCE from the dataset panel, where a chip is a
              button that narrows the grid to the images it counted. That works
              there because a dataset holds every row in memory, so the filter is
              an id list. A bank pages over SQL: the equivalent would have to be a
              server-side facet running the caption lexicon in the query, next to
              the flag/framing/medium facets the rail already offers — a bigger
              piece than the panel, and not one to fake by shipping a chip that
              can only filter the page you happen to be looking at. */}
          <span className="sr-only">{axisSummary(axis)}</span>
          <div aria-hidden className="flex flex-wrap gap-1">
            {axisRows(axis).map((r) => (
              <span key={r.id}
                title={r.count ? `${r.count} caption${r.count === 1 ? '' : 's'} mention this`
                  : (r.state === 'gap' ? 'No caption mentions this' : 'Not mentioned (optional)')}
                className={`rounded-full border px-2 py-0.5 text-[11px] ${chip[r.state]}`}>
                {r.label}<span className="opacity-60"> {r.count}</span>
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export default function CoveragePanel({ coverage, semanticEngine, semanticLabel,
  onClose, onBalance = null, balanceReason = '' }) {
  if (!coverage) {
    return <p className="text-sm text-content-subtle">Reading coverage…</p>
  }
  const poolWord = coverage.pool === 'kept' ? 'kept' : 'candidate (nothing kept yet)'
  return (
    <div className="space-y-3 rounded-lg border border-indigo-400/40 bg-indigo-500/5 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-content">📊 Coverage advice</span>
        <span className="text-xs text-content-subtle">
          {coverage.total.toLocaleString()} {poolWord} image{coverage.total === 1 ? '' : 's'}
        </span>
        <span className="ml-auto rounded border border-indigo-400/40 px-1.5 py-px text-[10px] uppercase tracking-wide text-indigo-300"
          title="Community idea by @antonp">idea by @antonp</span>
        <button type="button" onClick={onClose} aria-label="Hide coverage advice"
          className="rounded-md border border-border px-1.5 py-0.5 text-xs text-content-subtle hover:text-content">✕</button>
      </div>
      {coverage.framing_available && <FramingBar framing={coverage.framing} />}
      <VisualSpread visual={coverage.visual} total={coverage.total}
        semanticEngine={semanticEngine} />
      <ul className="space-y-1 text-sm">
        {coverage.advice.map((a, i) => (
          <li key={i} className="flex items-start gap-2">
            <span aria-hidden>{a.tone === 'warn' ? '⚠️' : '💡'}</span>
            <span className={a.tone === 'warn' ? 'text-amber-200' : 'text-content-muted'}>{a.text}</span>
          </li>
        ))}
      </ul>
      {/* The advice said what leans; this is the gesture that acts on it. Still
          only a SELECTION — the panel itself never keeps or rejects anything. */}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={onBalance || undefined} disabled={!onBalance}
          title={onBalance
            ? 'Select a set spread evenly over the framings, instead of the top of one ranking'
            : balanceReason}
          className="rounded-md border border-emerald-400/40 bg-emerald-500/10 px-2 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-emerald-500/20">
          ⚖️ Pick a balanced set…
        </button>
        {!onBalance && balanceReason && (
          <span className="text-[11px] text-content-subtle">{balanceReason}</span>
        )}
      </div>
      <VarietyAxes variety={coverage.variety} />
      <p className="text-[11px] text-content-subtle">
        Advice only — nothing is kept or rejected. Based on what the passes already computed:
        the labels, your captions (words, not pixels — a shot the captioner never described is
        invisible here, and “not smiling” still counts as a smile) and the {semanticLabel} semantic index.
        Judged as a character source, like the framing target above.
      </p>
    </div>
  )
}
