import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { useCapabilities } from '../../context/CapabilitiesContext'
import DupGroupsPanel from './DupGroupsPanel'
import PromoteDialog from './PromoteDialog'
import LaunchAllDialog from './LaunchAllDialog'
import PipelineReport from './PipelineReport'

const PAGE_SIZE = 120

const FLAG_LABEL = {
  blur: '🌫 Blurry', noise: '📺 Noisy', uniform: '⬜ Flat',
  small: '📐 Small', unreadable: '❌ Unreadable',
  // V2 scoring flags (aesthetic · NSFW · watermark passes).
  low_aesthetic: '💔 Low aesthetic', nsfw: '🔞 NSFW', watermark: '🚩 Watermark',
}
// Quality flags the CPU scan produces vs the ones the ML scoring/watermark
// passes add — auto-reject only offers a flag whose pass has actually run.
const QUALITY_REJECT_FLAGS = ['blur', 'noise', 'uniform', 'small']
const SCORE_REJECT_FLAGS = ['low_aesthetic', 'nsfw', 'watermark']
const STATUS_RING = {
  keep: 'ring-2 ring-emerald-400',
  reject: 'ring-2 ring-rose-400 opacity-60',
  pending: '',
}

/** Fetch EVERY image id matching a filter, page by page (used by the
 * cluster/flag "select all" actions — a cluster can exceed one grid page). */
async function fetchAllIds(bankId, params) {
  const ids = []
  let offset = 0
  for (;;) {
    const qs = new URLSearchParams({ ...params, offset: String(offset), limit: '500' })
    const d = await apiFetch(`/api/bank/${bankId}/images?${qs}`)
    ids.push(...d.images.map((i) => i.id))
    offset += d.images.length
    if (offset >= d.total || d.images.length === 0) break
  }
  return ids
}

const STEP_SHORT = {
  scan: '🔎 Scan', auto_reject: '🧹 Auto-reject', score: '✨ Score',
  semantic_dedup: '✂ Crops', watermark: '🚩 Watermarks', faces: '👥 Person',
  caption: '🏷️ Caption',
}

function ProgressBar({ activity, onCancel }) {
  if (!activity || activity.finished) return null
  const { kind, done, total, detail } = activity
  const pct = total > 0 ? Math.round((100 * done) / total) : null
  const pipe = kind === 'pipeline' ? activity.pipeline : null
  return (
    <div className="space-y-2 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm">
      <div className="flex items-center gap-3">
        <span aria-hidden>⏳</span>
        <span className="text-content">
          {pipe
            ? `🚀 Launch all — step ${(pipe.index ?? 0) + 1}/${pipe.total_steps} · ${STEP_SHORT[pipe.current] || pipe.current}`
            : ({ scan: 'Quality scan', faces: 'Face pass', score: 'Scoring pass',
              semantic_dedup: 'Crops & variants', watermark: 'Watermark scan',
              caption: 'Captioning', promote: 'Promotion' }[kind] || 'Job') + ' running'}
          {' — '}{done}{total ? ` / ${total}` : ''}{detail ? ` · ${detail}` : ''}
        </span>
        {pct != null && (
          <div className="h-1.5 w-40 overflow-hidden rounded bg-surface-raised" role="progressbar"
            aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-amber-400" style={{ width: `${pct}%` }} />
          </div>
        )}
        <button type="button" onClick={onCancel}
          className="ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
          Stop
        </button>
      </div>
      {pipe && Array.isArray(pipe.results) && pipe.results.length > 0 && (
        <ul className="flex flex-wrap gap-1.5 pl-6 text-xs">
          {pipe.results.map((r, i) => (
            <li key={`${r.step}-${i}`}
              className={`rounded px-1.5 py-px ${r.status === 'done' ? 'bg-emerald-500/15 text-emerald-300'
                : r.status === 'error' ? 'bg-rose-500/15 text-rose-300'
                : 'bg-black/20 text-content-subtle'}`}
              title={r.reason || r.detail || ''}>
              {r.status === 'done' ? '✅' : r.status === 'error' ? '⚠️' : '⏭️'} {STEP_SHORT[r.step] || r.step}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Chip({ active, onClick, children, title }) {
  return (
    <button type="button" onClick={onClick} title={title}
      className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${active
        ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
        : 'border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised'}`}>
      {children}
    </button>
  )
}

function Tile({ img, bankId, selected, onToggle, size }) {
  const badge = (txt, cls) => (
    <span className={`rounded px-1 py-px text-[10px] font-semibold leading-none ${cls}`}>{txt}</span>
  )
  return (
    <li className={`relative overflow-hidden rounded-lg border border-border bg-surface ${STATUS_RING[img.status] || ''}`}>
      <button type="button" onClick={onToggle}
        title={`${img.name} — ${img.width || '?'}×${img.height || '?'}`
          + (img.blur_score != null ? ` · sharpness ${Math.round(img.blur_score)}` : '')
          + (img.aesthetic_score != null ? ` · aesthetic ${img.aesthetic_score.toFixed(1)}` : '')
          + (img.nsfw_score != null ? ` · NSFW ${Math.round(img.nsfw_score * 100)}%` : '')
          + (img.face_cluster ? ` · person #${img.face_cluster}` : '')
          + (img.style_cluster ? ` · style #${img.style_cluster}` : '')
          + (img.semantic_dup_group ? ` · same shot #${img.semantic_dup_group}` : '')
          + (img.caption ? `\n${img.caption}` : '')}
        className="block w-full">
        <img src={`/api/bank/${bankId}/thumb/${img.id}`} alt={img.name} loading="lazy"
          className={`w-full object-cover ${size === 'S' ? 'h-24' : 'h-36'}`} />
      </button>
      {selected && (
        <span aria-hidden className="absolute inset-0 bg-indigo-500/30 ring-2 ring-indigo-400 rounded-lg pointer-events-none" />
      )}
      <span className="absolute left-1 top-1 flex flex-wrap gap-0.5 max-w-[85%]">
        {img.status === 'keep' && badge('✓', 'bg-emerald-500/80 text-white')}
        {img.status === 'reject' && badge(`✕ ${img.reject_reason || ''}`.trim(), 'bg-rose-500/80 text-white')}
        {img.promoted_dataset_id != null && badge('⬆', 'bg-indigo-500/80 text-white')}
        {img.flags.map((f) => badge(FLAG_LABEL[f]?.slice(0, 2) || f, 'bg-black/60 text-amber-200'))}
        {img.face_cluster != null && badge(`👤${img.face_cluster}`, 'bg-black/60 text-sky-200')}
        {img.style_cluster != null && badge(`🎨${img.style_cluster}`, 'bg-black/60 text-fuchsia-200')}
        {img.dup_group != null && badge(`≈${img.dup_group}`, 'bg-black/60 text-fuchsia-200')}
        {img.semantic_dup_group != null && badge(`✂${img.semantic_dup_group}`, 'bg-black/60 text-orange-200')}
        {img.caption && badge('🏷️', 'bg-black/60 text-emerald-200')}
      </span>
      <a href={`/api/bank/${bankId}/file/${img.id}`} target="_blank" rel="noreferrer"
        title="Open the original file" aria-label={`Open ${img.name} full size`}
        className="absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[11px] text-white no-underline hover:bg-black/80">⛶</a>
    </li>
  )
}

export default function BankWorkspace({ bankId, onBack, onGone }) {
  const toast = useToast()
  const { caps } = useCapabilities()
  const [payload, setPayload] = useState(null)
  const [filter, setFilter] = useState({ status: null, flag: null, cluster: null,
    style: null, subfolder: null, search: null })
  const [searchText, setSearchText] = useState('')
  const [subfolders, setSubfolders] = useState([])
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState({ images: [], total: 0 })
  const [selected, setSelected] = useState(() => new Set())
  const [promoteOpen, setPromoteOpen] = useState(false)
  const [launchOpen, setLaunchOpen] = useState(false)
  const [dismissedReportAt, setDismissedReportAt] = useState(null)
  const [rejectFlags, setRejectFlags] = useState(() => new Set(['blur', 'uniform']))
  const [showAutoReject, setShowAutoReject] = useState(false)
  const [tileSize, setTileSize] = useState('M')
  const activityWasLive = useRef(false)

  const refreshPayload = useCallback(async () => {
    try {
      const d = await apiFetch(`/api/bank/${bankId}`)
      setPayload(d)
      return d
    } catch (e) {
      if (String(e?.message || '').includes('not found')) { onGone?.(); return null }
      return null
    }
  }, [bankId, onGone])

  const filterParams = useCallback((f) => {
    const params = {}
    if (f.status) params.status = f.status
    if (f.flag) params.flag = f.flag
    if (f.cluster != null) params.cluster = String(f.cluster)
    if (f.style != null) params.style = String(f.style)
    // subfolder is a string facet where '' is meaningful (bank root) — send it
    // whenever it isn't null, empty string included.
    if (f.subfolder != null) params.subfolder = f.subfolder
    if (f.search) params.search = f.search
    return params
  }, [])

  const refreshImages = useCallback(async (f = filter, off = offset) => {
    const params = { ...filterParams(f), offset: String(off), limit: String(PAGE_SIZE) }
    try {
      const d = await apiFetch(`/api/bank/${bankId}/images?${new URLSearchParams(params)}`)
      setPage(d)
    } catch { /* transient — next poll retries */ }
  }, [bankId, filter, offset, filterParams])

  useEffect(() => {
    refreshPayload(); refreshImages()
    apiFetch(`/api/bank/${bankId}/subfolders`)
      .then((d) => setSubfolders(d.subfolders || []))
      .catch(() => setSubfolders([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankId])

  // Poll while a job runs; refresh the grid once when it lands.
  const live = payload?.activity && !payload.activity.finished
  useEffect(() => {
    if (!live) {
      if (activityWasLive.current) {
        activityWasLive.current = false
        refreshImages()
        if (payload?.activity?.error) toast.error(`Job failed — ${payload.activity.error}`)
        else if (payload?.activity?.cancelled && payload?.activity?.detail)
          toast.info(payload.activity.detail)   // Stopped — N cached (M remaining)…
        else if (payload?.activity?.detail) toast.success(payload.activity.detail)
      }
      return undefined
    }
    activityWasLive.current = true
    const t = setInterval(refreshPayload, 2000)
    return () => clearInterval(t)
  }, [live, refreshPayload, refreshImages, toast, payload?.activity?.error,
      payload?.activity?.cancelled, payload?.activity?.detail])

  const setF = (patch) => {
    const f = { ...filter, ...patch }
    setFilter(f); setOffset(0); setSelected(new Set())
    refreshImages(f, 0)
  }

  // Debounce the search box, then apply it as a filter (page 1, selection cleared).
  useEffect(() => {
    const term = searchText.trim()
    if ((filter.search || '') === term) return undefined
    const t = setTimeout(() => setF({ search: term || null }), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText])
  const goto = (off) => { setOffset(off); refreshImages(filter, off) }

  const act = async (fn, okMsg) => {
    try {
      const d = await fn()
      if (okMsg) toast.success(okMsg)
      await refreshPayload(); await refreshImages()
      return d
    } catch (e) {
      toast.error(e?.message || 'Action failed.')
      return null
    }
  }

  const startScan = (rescan) => act(
    () => postJson(`/api/bank/${bankId}/scan`, { rescan: !!rescan }), null)
  const startFaces = () => act(() => postJson(`/api/bank/${bankId}/faces`, {}), null)
  const startScore = () => act(() => postJson(`/api/bank/${bankId}/score`, {}), null)
  const startSemanticDedup = () => act(
    () => postJson(`/api/bank/${bankId}/semantic-dedup`, {}), null)
  const startWatermark = () => act(() => postJson(`/api/bank/${bankId}/watermark`, {}), null)
  const startCaption = () => act(
    () => postJson(`/api/bank/${bankId}/caption`,
      selected.size ? { image_ids: [...selected] } : {}), null)
  const cancelJob = () => act(() => postJson(`/api/bank/${bankId}/cancel`, {}), null)
  const startPipeline = async (config) => {
    setLaunchOpen(false)
    await act(() => postJson(`/api/bank/${bankId}/pipeline`, config),
      '🚀 Launch all started — you can walk away; Stop any time.')
  }

  const batchStatus = async (ids, status) => {
    if (!ids.length) return
    await act(() => postJson(`/api/bank/${bankId}/images/status`, { ids, status }),
      `${ids.length} image(s) → ${status}`)
    setSelected(new Set())
  }

  const applyAutoReject = async () => {
    setShowAutoReject(false)
    const flags = [...rejectFlags]
    const d = await act(() => postJson(`/api/bank/${bankId}/apply-flags`, { flags }), null)
    if (d?.rejected) {
      const n = Object.values(d.rejected).reduce((a, b) => a + b, 0)
      toast.success(`Auto-reject: ${n} image(s) rejected (${flags.join(', ')}). Manual ✓/✕ untouched.`)
    }
  }

  const selectAllCurrent = async () => {
    try {
      const ids = await fetchAllIds(bankId, filterParams(filter))
      setSelected(new Set(ids))
      toast.info(`${ids.length} image(s) selected (whole filter, all pages).`)
    } catch (e) {
      toast.error(e?.message || 'Selection failed.')
    }
  }

  const counts = payload?.counts
  const flags = payload?.flags || {}
  const clusters = payload?.clusters || []
  const styleClusters = payload?.style_clusters || []
  const visionReady = !!caps.ollama?.vision_model_ready
  const scored = counts?.scored || 0
  const watermarkScanned = counts?.watermark_scanned || 0
  // Score flags only make sense once their pass ran; watermark is its own pass.
  const availableScoreFlags = SCORE_REJECT_FLAGS.filter(
    (f) => (f === 'watermark' ? watermarkScanned : scored) > 0)
  const canPromote = (counts?.keep || 0) > 0 || selected.size > 0

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={onBack}
          className="rounded-md border border-border px-2 py-1 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
          ← Banks
        </button>
        <h1 className="text-lg font-bold text-content">🗃️ {payload?.name || `Bank #${bankId}`}</h1>
        <span className="px-1.5 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.625rem] font-semibold uppercase tracking-wide">Beta</span>
        <span className="truncate font-mono text-xs text-content-subtle" title={payload?.source_path}>
          {payload?.source_path}
        </span>
      </header>

      {counts && (
        <p className="text-sm text-content-muted">
          <span className="font-semibold text-content">{counts.total}</span> images ·
          {' '}{counts.scanned} scanned ·
          {scored > 0 && <> {scored} scored ·</>}
          {watermarkScanned > 0 && <> {watermarkScanned} watermark-checked ·</>}
          {' '}{counts.pending} undecided ·
          {' '}<span className="text-emerald-300">{counts.keep} kept</span> ·
          {' '}<span className="text-rose-300">{counts.reject} rejected</span> ·
          {' '}<span className="text-indigo-300">{counts.promoted} promoted</span>
        </p>
      )}

      <ProgressBar activity={payload?.activity} onCancel={cancelJob} />

      {!live && payload?.pipeline_report
        && payload.pipeline_report.finished_at !== dismissedReportAt && (
        <PipelineReport report={payload.pipeline_report}
          onDismiss={() => setDismissedReportAt(payload.pipeline_report.finished_at)} />
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setLaunchOpen(true)} disabled={live || !(counts?.total > 0)}
          title="Run the whole triage in one go — scan, auto-reject, score, watermarks, group by person and (optionally) caption. Start it and walk away."
          className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-bold text-white shadow disabled:opacity-50">
          🚀 Launch all…
        </button>
        <span aria-hidden className="mx-0.5 h-5 w-px bg-border" />
        <button type="button" onClick={() => startScan(false)} disabled={live}
          title="Score every unscanned image (sharpness/noise/flat/size), hash it and group near-duplicates — CPU only, runs in the background"
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          🔎 Scan quality
        </button>
        {(counts?.scanned || 0) > 0 && (
          <button type="button" onClick={() => startScan(true)} disabled={live}
            title="Re-score everything (e.g. after files changed on disk)"
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
            Rescan all
          </button>
        )}
        <button type="button" onClick={startFaces} disabled={live || !caps.face_scoring}
          title={caps.face_scoring
            ? 'Detect the dominant face of every non-rejected image and cluster the bank by person (no reference needed). CPU, can take a while on thousands of images.'
            : 'Install the Quality tools (Setup) to sort by person'}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          👥 Group by person
        </button>
        <button type="button" onClick={startScore} disabled={live || !caps.bank_scoring}
          title={caps.bank_scoring
            ? 'Rate every non-rejected image for aesthetics (1–10), flag NSFW, and group by visual style — one CLIP pass. Powers a smarter "keep best". GPU when available; runs in the background.'
            : 'Install the Bank scoring extra (Setup ▸ Quality tools) to score aesthetics / NSFW / style'}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          ✨ Score{!caps.bank_scoring && ' (needs setup)'}
        </button>
        <button type="button" onClick={startWatermark} disabled={live || !visionReady}
          title={visionReady
            ? 'Scan every non-rejected image for an overlaid watermark/logo/URL with the same Qwen3-VL detector the datasets use (detection only — the bank never edits your files). GPU vision pass.'
            : 'Pull the vision model (Settings ▸ Captioning & quality) to scan for watermarks'}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          🚩 Find watermarks{!visionReady && ' (needs setup)'}
        </button>
        <button type="button" onClick={startCaption} disabled={live}
          title={selected.size
            ? `Caption the ${selected.size} selected image(s) with your caption engine (Settings ▸ Captioning & quality). Captions become searchable and follow the images when you promote them to a dataset.`
            : 'Caption every not-yet-captioned image (skips rejected) with your caption engine. Captions become searchable tags and follow the images when you promote them to a dataset. Select images first to caption just those.'}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          🏷️ Caption{selected.size ? ` ${selected.size} selected` : ' all'}
        </button>
        <button type="button" onClick={startSemanticDedup} disabled={live || scored === 0}
          title={scored > 0
            ? 'Group crops and re-compressed variants of the SAME shot the exact-duplicate hash misses — reuses the ✨ Score embeddings, so it costs no extra GPU time. Review them under the ✂ Same shot chip.'
            : 'Run ✨ Score first — semantic near-duplicates reuse its embeddings'}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          ✂ Find crops &amp; variants{scored === 0 && ' (needs Score)'}
        </button>
        <div className="relative">
          <button type="button" onClick={() => setShowAutoReject((v) => !v)} disabled={live}
            aria-expanded={showAutoReject}
            title="Bulk-reject the still-undecided images carrying the chosen quality flags"
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
            🧹 Auto-reject flagged…
          </button>
          {showAutoReject && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowAutoReject(false)} aria-hidden />
              <div className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                <p className="text-xs text-content-muted">
                  Rejects the UNDECIDED images with these flags. Your manual ✓/✕ are never changed;
                  everything stays reversible (nothing is deleted from disk).
                </p>
                {[...QUALITY_REJECT_FLAGS, ...availableScoreFlags].map((f) => (
                  <label key={f} className="flex items-center gap-2 text-sm text-content">
                    <input type="checkbox" checked={rejectFlags.has(f)}
                      onChange={(e) => setRejectFlags((prev) => {
                        const next = new Set(prev)
                        if (e.target.checked) next.add(f); else next.delete(f)
                        return next
                      })} />
                    {FLAG_LABEL[f]} <span className="text-content-subtle">({flags[f] ?? 0} flagged)</span>
                  </label>
                ))}
                <button type="button" onClick={applyAutoReject} disabled={!rejectFlags.size}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
                  Reject them
                </button>
              </div>
            </>
          )}
        </div>
        <button type="button" onClick={() => setPromoteOpen(true)} disabled={live || !canPromote}
          title={canPromote ? 'Copy the kept selection into a dataset' : 'Keep some images first'}
          className="ml-auto rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
          ⬆ Promote to dataset…
        </button>
      </div>

      {/* Person clusters (after the face pass) */}
      {clusters.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-content-subtle">
            People ({clusters.length} cluster{clusters.length > 1 ? 's' : ''} — biggest first)
          </p>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {clusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ cluster: filter.cluster === c.id ? null : c.id, flag: null })}
                  title={`Show person #${c.id} (${c.size} image(s))`}
                  className={`relative block overflow-hidden rounded-lg border ${filter.cluster === c.id
                    ? 'border-indigo-400 ring-2 ring-indigo-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`} alt={`Person ${c.id}`}
                      loading="lazy" className="h-16 w-16 object-cover" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] font-semibold text-white">
                    #{c.id} · {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Style clusters (after the scoring pass) — group screenshots/memes vs photoreal */}
      {styleClusters.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-content-subtle">
            Styles ({styleClusters.length} group{styleClusters.length > 1 ? 's' : ''} — biggest first)
          </p>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {styleClusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ style: filter.style === c.id ? null : c.id, flag: null, cluster: null })}
                  title={`Show style group #${c.id} (${c.size} image(s))`}
                  className={`relative block overflow-hidden rounded-lg border ${filter.style === c.id
                    ? 'border-fuchsia-400 ring-2 ring-fuchsia-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`} alt={`Style ${c.id}`}
                      loading="lazy" className="h-16 w-16 object-cover" />
                  )}
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-center text-[10px] font-semibold text-white">
                    🎨{c.id} · {c.size}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Subfolder scoping (a Telegram export nests one folder per chat/date) */}
      {subfolders.length > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <label className="text-xs font-semibold uppercase tracking-wide text-content-subtle">
            Subfolder
          </label>
          <select value={filter.subfolder ?? '__all__'}
            onChange={(e) => setF({ subfolder: e.target.value === '__all__' ? null : e.target.value })}
            className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-content">
            <option value="__all__">All subfolders</option>
            {subfolders.map((s) => (
              <option key={s.name || '__root__'} value={s.name}>
                {s.name === '' ? '(bank root)' : s.name} · {s.count}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Full-text search over captions + file paths */}
      <div className="relative max-w-md">
        <span aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-content-subtle">🔍</span>
        <input type="search" value={searchText} onChange={(e) => setSearchText(e.target.value)}
          placeholder="Search captions and file names… (e.g. red dress)"
          aria-label="Search the bank by caption or file name"
          className="w-full rounded-md border border-border bg-surface py-1.5 pl-8 pr-8 text-sm text-content placeholder:text-content-subtle" />
        {searchText && (
          <button type="button" onClick={() => setSearchText('')} aria-label="Clear search"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-content-subtle hover:text-content">✕</button>
        )}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip active={!filter.status && !filter.flag && filter.cluster == null && filter.style == null}
          onClick={() => setF({ status: null, flag: null, cluster: null, style: null })}>All</Chip>
        <Chip active={filter.status === 'pending'} onClick={() => setF({ status: filter.status === 'pending' ? null : 'pending' })}>Undecided</Chip>
        <Chip active={filter.status === 'keep'} onClick={() => setF({ status: filter.status === 'keep' ? null : 'keep' })}>✓ Kept</Chip>
        <Chip active={filter.status === 'reject'} onClick={() => setF({ status: filter.status === 'reject' ? null : 'reject' })}>✕ Rejected</Chip>
        <span aria-hidden className="mx-1 h-4 w-px bg-border" />
        {['blur', 'noise', 'uniform', 'small', 'unreadable'].map((f) => (
          <Chip key={f} active={filter.flag === f}
            onClick={() => setF({ flag: filter.flag === f ? null : f })}
            title="Sorted worst-first">
            {FLAG_LABEL[f]} {flags[f] ?? 0}
          </Chip>
        ))}
        <Chip active={filter.flag === 'clean'} onClick={() => setF({ flag: filter.flag === 'clean' ? null : 'clean' })}>✨ Clean</Chip>
        {/* Score-derived flags — only surfaced once their pass has produced data. */}
        {availableScoreFlags.map((f) => (
          <Chip key={f} active={filter.flag === f}
            onClick={() => setF({ flag: filter.flag === f ? null : f, cluster: null, style: null })}
            title={f === 'watermark' ? 'Overlaid watermark detected' : 'Sorted worst-first'}>
            {FLAG_LABEL[f]} {flags[f] ?? 0}
          </Chip>
        ))}
        <Chip active={filter.flag === 'dups'} onClick={() => setF({ flag: filter.flag === 'dups' ? null : 'dups', cluster: null })}
          title="Exact / resized duplicate groups (perceptual hash) with their resolution panel">
          ≈ Duplicates {payload?.dup?.unresolved ?? 0}
        </Chip>
        {(payload?.semantic_dup?.groups ?? 0) > 0 && (
          <Chip active={filter.flag === 'semantic_dups'}
            onClick={() => setF({ flag: filter.flag === 'semantic_dups' ? null : 'semantic_dups', cluster: null })}
            title="Semantic near-duplicates — same shot, different crop/compression — with their resolution panel">
            ✂ Same shot {payload?.semantic_dup?.unresolved ?? 0}
          </Chip>
        )}
        {payload?.faces_scanned > 0 && (
          <Chip active={filter.flag === 'no_face'} onClick={() => setF({ flag: filter.flag === 'no_face' ? null : 'no_face' })}>
            🚫👤 No face
          </Chip>
        )}
        <span className="ml-auto" />
        <button type="button" onClick={() => setTileSize((s) => (s === 'M' ? 'S' : 'M'))}
          className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">
          {tileSize === 'M' ? 'Small tiles' : 'Medium tiles'}
        </button>
      </div>

      {/* Selection bar */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-content-muted">{selected.size} selected</span>
        <button type="button" onClick={selectAllCurrent}
          className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
          Select all in filter
        </button>
        {selected.size > 0 && (
          <>
            <button type="button" onClick={() => setSelected(new Set())}
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">Clear</button>
            <button type="button" onClick={() => batchStatus([...selected], 'keep')}
              className="rounded-md border border-emerald-400/50 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-200">✓ Keep</button>
            <button type="button" onClick={() => batchStatus([...selected], 'reject')}
              className="rounded-md border border-rose-400/50 bg-rose-500/10 px-2 py-0.5 text-xs font-semibold text-rose-200">✕ Reject</button>
            <button type="button" onClick={() => batchStatus([...selected], 'pending')}
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">↺ Undecided</button>
          </>
        )}
      </div>

      {filter.flag === 'dups' ? (
        <DupGroupsPanel bankId={bankId} live={live} kind="exact"
          onChanged={() => { refreshPayload(); refreshImages() }} />
      ) : filter.flag === 'semantic_dups' ? (
        <DupGroupsPanel bankId={bankId} live={live} kind="semantic"
          onChanged={() => { refreshPayload(); refreshImages() }} />
      ) : (
        <>
          <ul className={`grid gap-2 ${tileSize === 'S'
            ? 'grid-cols-4 sm:grid-cols-6 lg:grid-cols-8'
            : 'grid-cols-3 sm:grid-cols-4 lg:grid-cols-6'}`}>
            {page.images.map((img) => (
              <Tile key={img.id} img={img} bankId={bankId} size={tileSize}
                selected={selected.has(img.id)}
                onToggle={() => setSelected((prev) => {
                  const next = new Set(prev)
                  if (next.has(img.id)) next.delete(img.id); else next.add(img.id)
                  return next
                })} />
            ))}
          </ul>
          {page.total === 0 && (
            <p className="text-sm text-content-muted">Nothing matches this filter.</p>
          )}
          {page.total > PAGE_SIZE && (
            <nav className="flex items-center gap-3 text-sm" aria-label="Grid pages">
              <button type="button" disabled={offset === 0} onClick={() => goto(Math.max(0, offset - PAGE_SIZE))}
                className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">← Prev</button>
              <span className="text-content-muted">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total}
              </span>
              <button type="button" disabled={offset + PAGE_SIZE >= page.total}
                onClick={() => goto(offset + PAGE_SIZE)}
                className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">Next →</button>
            </nav>
          )}
        </>
      )}

      {promoteOpen && (
        <PromoteDialog bankId={bankId} keepCount={counts?.keep || 0}
          selectedIds={[...selected]}
          onClose={() => setPromoteOpen(false)}
          onStarted={() => { setPromoteOpen(false); refreshPayload() }} />
      )}

      {launchOpen && (
        <LaunchAllDialog caps={caps} visionReady={visionReady}
          onClose={() => setLaunchOpen(false)} onLaunch={startPipeline} />
      )}
    </div>
  )
}
