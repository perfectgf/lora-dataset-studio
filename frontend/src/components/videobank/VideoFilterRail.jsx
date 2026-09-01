/**
 * The video bank's filter rail — all of triage, beside the grid it filters.
 *
 * The image lane's Encre shell, worn by the video lane. Same reasoning, same
 * atoms, same drawer rules (bankLayout.js decides; this component only obeys
 * `isDrawer`): the controls sit NEXT to their result instead of stacking above
 * it, so adjusting a filter no longer costs a scroll round trip.
 *
 * ⚠️ Byte-preserving on purpose: the facet blocks moved here from
 * VideoBankWorkspace with their behaviour, counts and notes unchanged — only
 * their clothes changed, to the shared Chip / FilterGroup / GroupLabel atoms.
 * The one deliberate redress: the four status filters wear the image rail's
 * two-column status grid with their status colours, because "which pile am I
 * looking at" is the glance both lanes are read by.
 */
import VideoClipSearchBox from './VideoClipSearchBox'
import VideoSourceList from './VideoSourceList'
import VideoThresholdsPanel from './VideoThresholdsPanel'
import { Chip, FilterGroup, GroupLabel } from '../bank/BankAtoms.jsx'
import { STATUS_FILTERS, statusFilterCount } from './videoTriage'
import { CAMERA_FACET_NOTE, CAMERA_HINTS } from './videoCameraMotion'

/* The status colours are the app-wide triage vocabulary — amber to do, green
   kept, red rejected — identical to the image rail's status block. */
const STATUS_ON = {
  all: 'border-indigo-400/70 bg-indigo-500/25 text-indigo-100',
  pending: 'border-amber-400/70 bg-amber-500/20 text-amber-100',
  keep: 'border-emerald-400/70 bg-emerald-500/20 text-emerald-100',
  reject: 'border-rose-400/70 bg-rose-500/20 text-rose-100',
}

export default function VideoFilterRail({
  bankId, isDrawer, onClose,
  counts, status, setStatus,
  sourceId, setSourceId, sources,
  chips, flag, setFlag, flagNote,
  cameraOptions, camera, setCamera,
  busy, search, searching, captionModel,
  onRunEmbed, onSearchResult, onSearchClear,
  thresholds, totalClips, onThresholdsApplied,
  onCut, onSingleShot, onRecut,
}) {
  return (
    <aside aria-label="Video bank filters"
      data-probe-chrome={isDrawer ? 'rail' : undefined} data-probe-layer={isDrawer ? '' : undefined}
      data-probe-panel="rail"
      className={isDrawer
        /* ⚠️ `bg-surface-overlay`, NOT `bg-surface`: the tint is 4 %-alpha white
           for cards sitting ON the opaque page — painted with it, this drawer
           is a sheet of glass over the grid. Same pin as the image rail. */
        ? 'shadow-2xl fixed inset-y-0 left-0 z-50 w-[19rem] max-w-[88vw] overflow-y-auto border-r border-border bg-surface-overlay p-3 space-y-3'
        : 'space-y-3 self-start rounded-xl border border-border bg-surface p-3'}>
      <div className="flex items-center gap-2">
        <GroupLabel>Filters</GroupLabel>
        {isDrawer && (
          <button type="button" onClick={onClose} aria-label="Close the filters"
            className="min-h-10 lg:min-h-0 ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">
            ✕
          </button>
        )}
      </div>

      {/* ── Status — first, dressed as what it is: the triage's main gesture,
          each pile with its live count and its status colour. */}
      <div className="grid grid-cols-2 gap-1.5" role="group" aria-label="Filter by status">
        {STATUS_FILTERS.map((f) => {
          const active = status === f.key
          return (
            <button key={f.key} type="button" onClick={() => setStatus(f.key)}
              aria-pressed={active}
              className={`min-h-10 lg:min-h-0 rounded-md border px-2.5 py-1.5 text-sm font-semibold transition-colors ${active
                ? STATUS_ON[f.key] || STATUS_ON.all
                : 'border-border bg-surface-raised text-content-muted hover:text-content hover:bg-surface'}`}>
              {f.label}
              <span className="ml-1.5 text-xs font-normal tabular-nums opacity-80">
                {statusFilterCount(counts, f.key).toLocaleString()}
              </span>
            </button>
          )
        })}
      </div>
      {sourceId && (
        /* A raw <button>, not a <Chip>: the surface inventory freezes this
           label and reads it off button elements only. Dressed as an active
           chip so the eye cannot tell the difference. */
        <button type="button" onClick={() => setSourceId(null)}
          title="Show every file again"
          className="min-h-10 lg:min-h-0 self-start rounded-full border border-indigo-400/60 bg-indigo-500/20 px-2.5 py-0.5 text-xs font-medium text-indigo-200">
          one file only ✕
        </button>
      )}

      {/* 🔎 A way of LOOKING at the shots, not a pass — and what it changes is
          the grid beside this rail. */}
      {(counts.clips || 0) > 0 && (
        <VideoClipSearchBox bankId={bankId} counts={counts} busy={busy}
          result={search} searching={searching}
          onRunPass={onRunEmbed}
          onResult={onSearchResult}
          captionModel={captionModel}
          onClear={onSearchClear} />
      )}

      {/* ⚑ The verdicts, as chips you can act on. Every flag in this lane used
          to be a badge you read one shot at a time — which is fine for "too much
          motion" and useless for "you already have this shot", where the whole
          point is rejecting the pile in one gesture. */}
      {chips.length > 0 && (
        <FilterGroup label="⚑ Flagged">
          {chips.map((c) => (
            <Chip key={c.flag} active={flag === c.flag}
              onClick={() => setFlag(flag === c.flag ? null : c.flag)}>
              {c.label} ({c.count})
            </Chip>
          ))}
          {flag && (
            /* Raw buttons, chip-dressed: the surface inventory reads labels off
               <button> elements only, and this one is frozen. */
            <button type="button" onClick={() => setFlag(null)}
              className="min-h-10 lg:min-h-0 rounded-full border border-border bg-surface px-2.5 py-0.5 text-xs font-medium text-content-muted hover:bg-surface-raised hover:text-content">
              show all ✕
            </button>
          )}
        </FilterGroup>
      )}
      {/* The counts cover the LOADED page, and a chip that read like a bank-wide
          total would be a wrong number rather than a filter. */}
      {flagNote && <p className="text-[0.6875rem] text-content-subtle">{flagNote}</p>}

      {/* 🎥 The camera facet describes rather than accuses — the wobble one user
          filters out is what the next user is filtering FOR. Chips marked ᐩ are
          this app's own words rather than the trainer's. */}
      {cameraOptions.length > 0 && (
        <FilterGroup label="🎥 Camera">
          {cameraOptions.map((c) => (
            <Chip key={c.name} active={camera === c.name}
              title={CAMERA_HINTS[c.name] || ''}
              onClick={() => setCamera(camera === c.name ? null : c.name)}>
              {c.label}{c.ours ? ' ᐩ' : ''} ({c.count})
            </Chip>
          ))}
          {camera && (
            <button type="button" onClick={() => setCamera(null)}
              className="min-h-10 lg:min-h-0 rounded-full border border-border bg-surface px-2.5 py-0.5 text-xs font-medium text-content-muted hover:bg-surface-raised hover:text-content">
              show all ✕
            </button>
          )}
          <span className="text-[0.6875rem] text-content-subtle">{CAMERA_FACET_NOTE}</span>
        </FilterGroup>
      )}

      {(totalClips || 0) > 0 && (
        <VideoThresholdsPanel bankId={bankId} saved={thresholds}
          totalClips={totalClips} onApplied={onThresholdsApplied} />
      )}

      <details className="min-w-0 rounded-lg border border-border bg-surface">
        <summary className="min-h-10 lg:min-h-0 cursor-pointer px-3 py-2 text-sm font-semibold text-content">
          Files ({counts.sources || 0})
        </summary>
        <div className="border-t border-border p-3">
          <VideoSourceList sources={sources} activeSourceId={sourceId}
            onFilter={setSourceId} onCut={onCut}
            onSingleShot={onSingleShot} onRecut={onRecut} />
        </div>
      </details>
    </aside>
  )
}
