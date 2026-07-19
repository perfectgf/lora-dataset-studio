import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { useI18n } from '../../i18n/I18nContext'
import DupGroupsPanel from './DupGroupsPanel'
import PromoteDialog from './PromoteDialog'

const PAGE_SIZE = 120

const FLAG_META = {
  blur: ['🌫', 'blur'], noise: ['📺', 'noise'], uniform: ['⬜', 'uniform'],
  small: ['📐', 'small'], unreadable: ['❌', 'unreadable'],
  // V2 scoring flags (aesthetic · NSFW · watermark passes).
  low_aesthetic: ['💔', 'lowAesthetic'], nsfw: ['🔞', 'nsfw'], watermark: ['🚩', 'watermark'],
}
const flagLabel = (t, flag) => FLAG_META[flag]
  ? `${FLAG_META[flag][0]} ${t(`bank.flags.${FLAG_META[flag][1]}`)}`
  : flag
const flagIcon = (flag) => FLAG_META[flag]?.[0] || flag
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

function ProgressBar({ activity, onCancel }) {
  const { t } = useI18n()
  if (!activity || activity.finished) return null
  const { kind, done, total, detail } = activity
  const pct = total > 0 ? Math.round((100 * done) / total) : null
  return (
    <div className="flex items-center gap-3 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm">
      <span aria-hidden>⏳</span>
      <span className="text-content">
        {t(`bank.activity.${['scan', 'faces', 'score', 'watermark', 'promote'].includes(kind)
          ? kind : 'job'}`)} {t('bank.activity.running')} —
        {' '}{done}{total ? ` / ${total}` : ''}{detail ? ` · ${detail}` : ''}
      </span>
      {pct != null && (
        <div className="h-1.5 w-40 overflow-hidden rounded bg-surface-raised" role="progressbar"
          aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-full bg-amber-400" style={{ width: `${pct}%` }} />
        </div>
      )}
      <button type="button" onClick={onCancel}
        className="ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
        {t('common.cancel')}
      </button>
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
  const { t } = useI18n()
  const badge = (txt, cls) => (
    <span className={`rounded px-1 py-px text-[10px] font-semibold leading-none ${cls}`}>{txt}</span>
  )
  return (
    <li className={`relative overflow-hidden rounded-lg border border-border bg-surface ${STATUS_RING[img.status] || ''}`}>
      <button type="button" onClick={onToggle}
        title={`${img.name} — ${img.width || '?'}×${img.height || '?'}`
          + (img.blur_score != null ? ` · ${t('bank.metrics.sharpness')} ${Math.round(img.blur_score)}` : '')
          + (img.aesthetic_score != null ? ` · ${t('bank.metrics.aesthetic')} ${img.aesthetic_score.toFixed(1)}` : '')
          + (img.nsfw_score != null ? ` · NSFW ${Math.round(img.nsfw_score * 100)}%` : '')
          + (img.face_cluster ? ` · ${t('bank.metrics.person')} #${img.face_cluster}` : '')
          + (img.style_cluster ? ` · ${t('bank.metrics.style')} #${img.style_cluster}` : '')}
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
        {img.flags.map((f) => badge(flagIcon(f), 'bg-black/60 text-amber-200'))}
        {img.face_cluster != null && badge(`👤${img.face_cluster}`, 'bg-black/60 text-sky-200')}
        {img.style_cluster != null && badge(`🎨${img.style_cluster}`, 'bg-black/60 text-fuchsia-200')}
        {img.dup_group != null && badge(`≈${img.dup_group}`, 'bg-black/60 text-fuchsia-200')}
      </span>
      <a href={`/api/bank/${bankId}/file/${img.id}`} target="_blank" rel="noreferrer"
        title={t('bank.workspace.openOriginal')}
        aria-label={t('bank.workspace.openFullSize', { name: img.name })}
        className="absolute bottom-1 right-1 rounded bg-black/60 px-1 text-[11px] text-white no-underline hover:bg-black/80">⛶</a>
    </li>
  )
}

export default function BankWorkspace({ bankId, onBack, onGone }) {
  const toast = useToast()
  const { caps } = useCapabilities()
  const { t } = useI18n()
  const [payload, setPayload] = useState(null)
  const [filter, setFilter] = useState({ status: null, flag: null, cluster: null,
    style: null, subfolder: null })
  const [subfolders, setSubfolders] = useState([])
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState({ images: [], total: 0 })
  const [selected, setSelected] = useState(() => new Set())
  const [promoteOpen, setPromoteOpen] = useState(false)
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
        if (payload?.activity?.error) {
          toast.error(t('bank.activity.failed', { error: payload.activity.error }))
        }
        else if (payload?.activity?.detail) toast.success(payload.activity.detail)
      }
      return undefined
    }
    activityWasLive.current = true
    const pollTimer = setInterval(refreshPayload, 2000)
    return () => clearInterval(pollTimer)
  }, [live, refreshPayload, refreshImages, toast, t, payload?.activity?.error, payload?.activity?.detail])

  const setF = (patch) => {
    const f = { ...filter, ...patch }
    setFilter(f); setOffset(0); setSelected(new Set())
    refreshImages(f, 0)
  }
  const goto = (off) => { setOffset(off); refreshImages(filter, off) }

  const act = async (fn, okMsg) => {
    try {
      const d = await fn()
      if (okMsg) toast.success(okMsg)
      await refreshPayload(); await refreshImages()
      return d
    } catch (e) {
      toast.error(e?.message || t('bank.workspace.actionFailed'))
      return null
    }
  }

  const startScan = (rescan) => act(
    () => postJson(`/api/bank/${bankId}/scan`, { rescan: !!rescan }), null)
  const startFaces = () => act(() => postJson(`/api/bank/${bankId}/faces`, {}), null)
  const startScore = () => act(() => postJson(`/api/bank/${bankId}/score`, {}), null)
  const startWatermark = () => act(() => postJson(`/api/bank/${bankId}/watermark`, {}), null)
  const cancelJob = () => act(() => postJson(`/api/bank/${bankId}/cancel`, {}), null)

  const batchStatus = async (ids, status) => {
    if (!ids.length) return
    await act(() => postJson(`/api/bank/${bankId}/images/status`, { ids, status }),
      t('bank.workspace.statusChanged', {
        count: ids.length,
        status: t(`bank.status.${status}`),
      }))
    setSelected(new Set())
  }

  const applyAutoReject = async () => {
    setShowAutoReject(false)
    const flags = [...rejectFlags]
    const d = await act(() => postJson(`/api/bank/${bankId}/apply-flags`, { flags }), null)
    if (d?.rejected) {
      const n = Object.values(d.rejected).reduce((a, b) => a + b, 0)
      toast.success(t('bank.workspace.autoRejectResult', {
        count: n, flags: flags.map((f) => flagLabel(t, f)).join(', '),
      }))
    }
  }

  const selectAllCurrent = async () => {
    try {
      const ids = await fetchAllIds(bankId, filterParams(filter))
      setSelected(new Set(ids))
      toast.info(t('bank.workspace.selectedAll', { count: ids.length }))
    } catch (e) {
      toast.error(e?.message || t('bank.workspace.selectionFailed'))
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
          ← {t('bank.page.title')}
        </button>
        <h1 className="text-lg font-bold text-content">
          🗃️ {payload?.name || t('bank.workspace.fallbackName', { id: bankId })}
        </h1>
        <span className="px-1.5 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.625rem] font-semibold uppercase tracking-wide">{t('common.beta')}</span>
        <span className="truncate font-mono text-xs text-content-subtle" title={payload?.source_path}>
          {payload?.source_path}
        </span>
      </header>

      {counts && (
        <p className="text-sm text-content-muted">
          <span className="font-semibold text-content">{t('bank.counts.images', { count: counts.total })}</span> ·{' '}
          {t('bank.counts.scanned', { count: counts.scanned })} ·
          {scored > 0 && <> {t('bank.counts.scored', { count: scored })} ·</>}
          {watermarkScanned > 0 && <> {t('bank.counts.watermarkChecked', { count: watermarkScanned })} ·</>}
          {' '}{t('bank.counts.undecided', { count: counts.pending })} ·{' '}
          <span className="text-emerald-300">{t('bank.counts.kept', { count: counts.keep })}</span> ·{' '}
          <span className="text-rose-300">{t('bank.counts.rejected', { count: counts.reject })}</span> ·{' '}
          <span className="text-indigo-300">{t('bank.counts.promoted', { count: counts.promoted })}</span>
        </p>
      )}

      <ProgressBar activity={payload?.activity} onCancel={cancelJob} />

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => startScan(false)} disabled={live}
          title={t('bank.workspace.scanTitle')}
          className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
          🔎 {t('bank.workspace.scan')}
        </button>
        {(counts?.scanned || 0) > 0 && (
          <button type="button" onClick={() => startScan(true)} disabled={live}
            title={t('bank.workspace.rescanTitle')}
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
            {t('bank.workspace.rescan')}
          </button>
        )}
        <button type="button" onClick={startFaces} disabled={live || !caps.face_scoring}
          title={caps.face_scoring
            ? t('bank.workspace.peopleTitle')
            : t('bank.workspace.peopleSetup')}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          👥 {t('bank.workspace.groupPeople')}
        </button>
        <button type="button" onClick={startScore} disabled={live || !caps.bank_scoring}
          title={caps.bank_scoring
            ? t('bank.workspace.scoreTitle')
            : t('bank.workspace.scoreSetup')}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          ✨ {t('bank.workspace.score')}{!caps.bank_scoring && ` (${t('bank.workspace.needsSetup')})`}
        </button>
        <button type="button" onClick={startWatermark} disabled={live || !visionReady}
          title={visionReady
            ? t('bank.workspace.watermarkTitle')
            : t('bank.workspace.watermarkSetup')}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
          🚩 {t('bank.workspace.findWatermarks')}{!visionReady && ` (${t('bank.workspace.needsSetup')})`}
        </button>
        <div className="relative">
          <button type="button" onClick={() => setShowAutoReject((v) => !v)} disabled={live}
            aria-expanded={showAutoReject}
            title={t('bank.workspace.autoRejectTitle')}
            className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50 hover:bg-surface">
            🧹 {t('bank.workspace.autoReject')}
          </button>
          {showAutoReject && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowAutoReject(false)} aria-hidden />
              <div className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface p-3 shadow-xl space-y-2">
                <p className="text-xs text-content-muted">
                  {t('bank.workspace.autoRejectHelp')}
                </p>
                {[...QUALITY_REJECT_FLAGS, ...availableScoreFlags].map((f) => (
                  <label key={f} className="flex items-center gap-2 text-sm text-content">
                    <input type="checkbox" checked={rejectFlags.has(f)}
                      onChange={(e) => setRejectFlags((prev) => {
                        const next = new Set(prev)
                        if (e.target.checked) next.add(f); else next.delete(f)
                        return next
                      })} />
                    {flagLabel(t, f)}{' '}
                    <span className="text-content-subtle">
                      ({t('bank.workspace.flagged', { count: flags[f] ?? 0 })})
                    </span>
                  </label>
                ))}
                <button type="button" onClick={applyAutoReject} disabled={!rejectFlags.size}
                  className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
                  {t('bank.workspace.rejectThem')}
                </button>
              </div>
            </>
          )}
        </div>
        <button type="button" onClick={() => setPromoteOpen(true)} disabled={live || !canPromote}
          title={canPromote ? t('bank.workspace.promoteTitle') : t('bank.workspace.promoteDisabled')}
          className="ml-auto rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
          ⬆ {t('bank.workspace.promote')}…
        </button>
      </div>

      {/* Person clusters (after the face pass) */}
      {clusters.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-content-subtle">
            {t('bank.workspace.peopleSummary', { count: clusters.length })}
          </p>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {clusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ cluster: filter.cluster === c.id ? null : c.id, flag: null })}
                  title={t('bank.workspace.showPerson', { id: c.id, count: c.size })}
                  className={`relative block overflow-hidden rounded-lg border ${filter.cluster === c.id
                    ? 'border-indigo-400 ring-2 ring-indigo-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`}
                      alt={t('bank.workspace.personAlt', { id: c.id })}
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
            {t('bank.workspace.stylesSummary', { count: styleClusters.length })}
          </p>
          <ul className="flex gap-2 overflow-x-auto pb-1">
            {styleClusters.map((c) => (
              <li key={c.id} className="shrink-0">
                <button type="button" onClick={() => setF({ style: filter.style === c.id ? null : c.id, flag: null, cluster: null })}
                  title={t('bank.workspace.showStyle', { id: c.id, count: c.size })}
                  className={`relative block overflow-hidden rounded-lg border ${filter.style === c.id
                    ? 'border-fuchsia-400 ring-2 ring-fuchsia-400' : 'border-border'}`}>
                  {c.cover_image_id != null && (
                    <img src={`/api/bank/${bankId}/thumb/${c.cover_image_id}`}
                      alt={t('bank.workspace.styleAlt', { id: c.id })}
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
            {t('bank.workspace.subfolder')}
          </label>
          <select value={filter.subfolder ?? '__all__'}
            onChange={(e) => setF({ subfolder: e.target.value === '__all__' ? null : e.target.value })}
            className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-content">
            <option value="__all__">{t('bank.workspace.allSubfolders')}</option>
            {subfolders.map((s) => (
              <option key={s.name || '__root__'} value={s.name}>
                {s.name === '' ? t('bank.workspace.bankRoot') : s.name} · {s.count}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip active={!filter.status && !filter.flag && filter.cluster == null && filter.style == null}
          onClick={() => setF({ status: null, flag: null, cluster: null, style: null })}>{t('common.all')}</Chip>
        <Chip active={filter.status === 'pending'} onClick={() => setF({ status: filter.status === 'pending' ? null : 'pending' })}>{t('bank.status.pending')}</Chip>
        <Chip active={filter.status === 'keep'} onClick={() => setF({ status: filter.status === 'keep' ? null : 'keep' })}>✓ {t('bank.status.keep')}</Chip>
        <Chip active={filter.status === 'reject'} onClick={() => setF({ status: filter.status === 'reject' ? null : 'reject' })}>✕ {t('bank.status.reject')}</Chip>
        <span aria-hidden className="mx-1 h-4 w-px bg-border" />
        {['blur', 'noise', 'uniform', 'small', 'unreadable'].map((f) => (
          <Chip key={f} active={filter.flag === f}
            onClick={() => setF({ flag: filter.flag === f ? null : f })}
            title={t('bank.workspace.sortedWorst')}>
            {flagLabel(t, f)} {flags[f] ?? 0}
          </Chip>
        ))}
        <Chip active={filter.flag === 'clean'} onClick={() => setF({ flag: filter.flag === 'clean' ? null : 'clean' })}>✨ {t('bank.flags.clean')}</Chip>
        {/* Score-derived flags — only surfaced once their pass has produced data. */}
        {availableScoreFlags.map((f) => (
          <Chip key={f} active={filter.flag === f}
            onClick={() => setF({ flag: filter.flag === f ? null : f, cluster: null, style: null })}
            title={f === 'watermark' ? t('bank.workspace.watermarkDetected') : t('bank.workspace.sortedWorst')}>
            {flagLabel(t, f)} {flags[f] ?? 0}
          </Chip>
        ))}
        <Chip active={filter.flag === 'dups'} onClick={() => setF({ flag: filter.flag === 'dups' ? null : 'dups', cluster: null })}
          title={t('bank.workspace.duplicatesTitle')}>
          ≈ {t('bank.workspace.duplicates')} {payload?.dup?.unresolved ?? 0}
        </Chip>
        {payload?.faces_scanned > 0 && (
          <Chip active={filter.flag === 'no_face'} onClick={() => setF({ flag: filter.flag === 'no_face' ? null : 'no_face' })}>
            🚫👤 {t('bank.workspace.noFace')}
          </Chip>
        )}
        <span className="ml-auto" />
        <button type="button" onClick={() => setTileSize((s) => (s === 'M' ? 'S' : 'M'))}
          className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">
          {tileSize === 'M' ? t('bank.workspace.smallTiles') : t('bank.workspace.mediumTiles')}
        </button>
      </div>

      {/* Selection bar */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-content-muted">{t('bank.workspace.selected', { count: selected.size })}</span>
        <button type="button" onClick={selectAllCurrent}
          className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
          {t('bank.workspace.selectAll')}
        </button>
        {selected.size > 0 && (
          <>
            <button type="button" onClick={() => setSelected(new Set())}
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">{t('common.clear')}</button>
            <button type="button" onClick={() => batchStatus([...selected], 'keep')}
              className="rounded-md border border-emerald-400/50 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-200">✓ {t('bank.actions.keep')}</button>
            <button type="button" onClick={() => batchStatus([...selected], 'reject')}
              className="rounded-md border border-rose-400/50 bg-rose-500/10 px-2 py-0.5 text-xs font-semibold text-rose-200">✕ {t('bank.actions.reject')}</button>
            <button type="button" onClick={() => batchStatus([...selected], 'pending')}
              className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">↺ {t('bank.actions.undecided')}</button>
          </>
        )}
      </div>

      {filter.flag === 'dups' ? (
        <DupGroupsPanel bankId={bankId} live={live}
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
            <p className="text-sm text-content-muted">{t('bank.workspace.noMatches')}</p>
          )}
          {page.total > PAGE_SIZE && (
            <nav className="flex items-center gap-3 text-sm" aria-label={t('bank.pagination.gridPages')}>
              <button type="button" disabled={offset === 0} onClick={() => goto(Math.max(0, offset - PAGE_SIZE))}
                className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">← {t('bank.pagination.prev')}</button>
              <span className="text-content-muted">
                {t('bank.pagination.range', {
                  from: offset + 1, to: Math.min(offset + PAGE_SIZE, page.total), total: page.total,
                })}
              </span>
              <button type="button" disabled={offset + PAGE_SIZE >= page.total}
                onClick={() => goto(offset + PAGE_SIZE)}
                className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">{t('bank.pagination.next')} →</button>
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
    </div>
  )
}
