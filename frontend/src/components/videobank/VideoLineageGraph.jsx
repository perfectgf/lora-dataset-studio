import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Trash2 } from 'lucide-react'
import { buildLineageGraph, CARD_W } from '../../utils/lineageGraph'
import { GraphCard, CheckpointPill } from '../dataset/lineageNodes'
import { LineageEdgeDefs, LineageEdges } from '../dataset/lineageEdges'
import { clampPopoverToViewport, POPOVER_W } from '../dataset/checkpointPopover.js'
import { useFocusTrap } from '../../hooks/useFocusTrap'
import { fmtSize } from './videoCheckpoints'
import {
  MUTED_CLS, ROW_CLS, nodeGroup, pillActionModel, pillPreview, pillStep, videoDeployHint,
} from './videoLineage'

// Same fit rules as the image graph (RunLineageGraph): shrink to the panel's
// width down to this scale, then pan; never taller than this before scrolling.
const MIN_SCALE = 0.5
const MAX_H = 560

/** Keep menu text and touch targets independent of the graph's zoom and clip. */
function FloatingVideoMenu({ anchor, onClose, children }) {
  const ref = useRef(null)
  const [position, setPosition] = useState({ left: 8, top: 8, width: POPOVER_W })
  useFocusTrap(ref, true)
  useLayoutEffect(() => {
    const place = () => {
      const next = clampPopoverToViewport(anchor,
        { width: window.innerWidth, height: window.innerHeight },
        { height: ref.current?.getBoundingClientRect().height || 264 })
      setPosition((prev) => prev.left === next.left && prev.top === next.top && prev.width === next.width
        ? prev : next)
    }
    place()
    const observer = new ResizeObserver(place)
    observer.observe(ref.current)
    window.addEventListener('resize', place)
    return () => { observer.disconnect(); window.removeEventListener('resize', place) }
  }, [anchor])
  useEffect(() => {
    const outside = (e) => { if (!ref.current?.contains(e.target)) onClose() }
    const escape = (e) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    ref.current?.querySelector('button[aria-label="Close"]')?.focus()
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [onClose])
  return createPortal(
    <div ref={ref} data-probe-layer data-probe-chrome="video-checkpoint-menu"
      style={{ ...position, position: 'fixed', zIndex: 1100,
        maxHeight: 'calc(100dvh - 16px)', overflowY: 'auto' }}>
      {children}
    </div>, document.body)
}

/** 📦 The actions popover of ONE pill on the video graph — the list's verbs,
 * decided by the list's model (`pillActionModel`), at the pill. Presentational:
 * it draws what `a` says and calls the handlers it is given. Its rows are the
 * image popover's rows (⬇, ▶, 📦/⏏, ⓘ, 🗑) with one difference the video lane
 * owns: ⬇ is one link PER FILE, because a Wan pair is two files at one step. */
export function VideoCheckpointPopover({
  node, pill, a, busy = null, onDeploy, onUndeploy, onDelete, onContinue, onDetails,
  onPlaySample, onClose,
}) {
  if (!a) return null
  const g = nodeGroup(node)
  const s = pillStep(pill)
  const rowBusy = typeof busy === 'string' && busy.startsWith(`${a.key}:`)
  const preview = pillPreview(pill)
  return (
    <div className="lds-ck-popover rounded-lg border border-indigo-400/40 bg-surface-overlay p-2 shadow-xl"
      role="dialog" aria-label={`Checkpoint ${a.label} actions`}
      onPointerDown={(e) => e.stopPropagation()}>
      <div className="mb-1.5 flex items-center gap-1.5">
        <span className="min-w-0 truncate text-content text-[0.6875rem] font-semibold tabular-nums">{a.label}</span>
        {a.deployed && (
          <span className="shrink-0 rounded bg-emerald-500/15 px-1 py-px text-emerald-200 text-[0.5rem] font-semibold uppercase">
            Deployed
          </span>
        )}
        <button type="button" onClick={onClose}
          className="ml-auto min-h-10 min-w-10 shrink-0 text-content-subtle hover:text-content text-[0.75rem] lg:min-h-0 lg:min-w-0"
          aria-label="Close">✕</button>
      </div>
      <div className="flex flex-col gap-1">
        {a.files.map((f) => (
          <a key={f.filename} href={f.url} download title={f.filename} onClick={onClose}
            aria-label={`Download ${f.filename}`}
            className={ROW_CLS + ' min-w-0 border-emerald-500/40 bg-emerald-600/15 text-emerald-100 no-underline hover:bg-emerald-600/25'}>
            <span aria-hidden>⬇</span> <span className="truncate">{f.short}</span>
            {f.size ? <span className="shrink-0 text-emerald-200/70">{fmtSize(f.size)}</span> : null}
          </a>
        ))}
        {preview?.url && (
          <button type="button" onClick={() => { onPlaySample?.(node, pill); onClose?.() }}
            title="Play the sample ai-toolkit rendered at this step"
            className={ROW_CLS + ' border-sky-400/40 bg-sky-500/15 text-sky-100 hover:bg-sky-500/25'}>
            <span aria-hidden>🎬</span> Play sample{preview.count > 1 ? `s (${preview.count})` : ''}
          </button>
        )}
        {a.continue.ok ? (
          <button type="button" disabled={rowBusy} onClick={() => { onContinue?.(g, s, node, pill); onClose?.() }}
            className={ROW_CLS + ' border-indigo-400/40 bg-indigo-500/15 text-indigo-100 hover:bg-indigo-500/25'}>
            <span aria-hidden>▶</span> Continue from here
          </button>
        ) : (
          <span className={MUTED_CLS}><span aria-hidden>▶</span> {a.continue.reason}</span>
        )}
        {a.deployed ? (a.undeploy?.ok ? (
          <button type="button" disabled={rowBusy} onClick={() => onUndeploy?.(g, s, node, pill)}
            className={ROW_CLS + ' border-emerald-500/40 bg-emerald-600/5 text-emerald-200/90 hover:bg-emerald-600/20'}>
            <span aria-hidden>⏏</span> Undeploy
          </button>
        ) : (
          <span className={MUTED_CLS}><span aria-hidden>⏏</span> {a.undeploy?.reason}</span>
        )) : (a.deploy?.ok ? (
          <button type="button" disabled={rowBusy} onClick={() => onDeploy?.(g, s, node, pill)}
            className={ROW_CLS + ' border-primary/40 bg-primary/20 text-white hover:bg-primary/30'}>
            <span aria-hidden>📦</span> Deploy → {a.deploy.folder}
          </button>
        ) : (
          <span className={MUTED_CLS}><span aria-hidden>📦</span> {a.deploy?.reason}</span>
        ))}
        {a.details && (
          <button type="button" onClick={() => { onDetails?.(node); onClose?.() }}
            className={ROW_CLS + ' border-border bg-app/60 text-content hover:border-indigo-400/50'}>
            <span aria-hidden>ⓘ</span> Details
          </button>
        )}
        {a.del.ok ? (
          <button type="button" disabled={rowBusy} onClick={() => onDelete?.(g, s, node, pill)}
            title={a.del.title}
            className="mt-1 flex min-h-10 items-center gap-1.5 border-t border-border px-2 pt-1.5 pb-0.5 text-left text-content-subtle text-[0.625rem] hover:text-rose-200 disabled:opacity-60 lg:min-h-0">
            <Trash2 aria-hidden="true" className="h-3.5 w-3.5" /> {a.del.label}
          </button>
        ) : (
          <span className={MUTED_CLS + ' mt-1'}><span aria-hidden>🗑</span> {a.del.reason}</span>
        )}
      </div>
    </div>
  )
}

/** 🌳 The ◉ Graph of a video dataset's runs.
 *
 * The DRAWING is the image lane's, verbatim: `buildLineageGraph` lays the tree
 * out, `GraphCard` draws a run, `CheckpointPill` a save, `LineageEdges` the
 * continuation curves — one rendering for both surfaces, so a lineage reads
 * the same whichever dataset it belongs to. What is this lane's own is what a
 * click DOES: a pill opens the video popover (per STEP, the list's verbs), a
 * card opens the run's details, a thumbnail plays the training sample. There
 * is no compare, no notes and no "Generate previews" bar — see PREVIEWS_NOTE. */
export default function VideoLineageGraph({
  datasetId, tree, ctx = {}, busy = null,
  onDeploy, onUndeploy, onDelete, onContinue, onDetails, onPlaySample,
}) {
  const [bigPreviews, setBigPreviews] = useState(() => {
    try { return localStorage.getItem('lds.videoGraphBigPreviews') === '1' } catch { return false }
  })
  const toggleBigPreviews = useCallback(() => {
    setBigPreviews((v) => {
      const next = !v
      try { localStorage.setItem('lds.videoGraphBigPreviews', next ? '1' : '0') } catch { /* ignore */ }
      return next
    })
  }, [])
  const g = useMemo(() => buildLineageGraph(tree, { bigPreviews }), [tree, bigPreviews])
  const scrollRef = useRef(null)
  const [scale, setScale] = useState(1)
  const [hoverId, setHoverId] = useState(null)
  // The open popover: `laid` identifies the drawn pill, `pill` carries its
  // files, and `anchor` positions the menu in viewport pixels.
  const [openCk, setOpenCk] = useState(null)
  const closePopover = useCallback(() => setOpenCk(null), [])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el || !g.width) return undefined
    const measure = () => {
      const avail = el.clientWidth || g.width
      setScale(Math.max(MIN_SCALE, Math.min(1, (avail - 4) / g.width)))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [g.width])

  // Drag-to-pan when the tree overflows — a light grab, not a zoom UI.
  const drag = useRef(null)
  const onPointerDown = useCallback((e) => {
    const el = scrollRef.current
    if (!el) return
    if (!e.target.closest('.lds-ckpill') && !e.target.closest('.lds-ck-popover')) setOpenCk(null)
    const overflow = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1
    if (!overflow || e.target.closest('.lds-gcard') || e.target.closest('.lds-ckpill')
        || e.target.closest('.lds-ck-popover')) return
    drag.current = { x: e.clientX, y: e.clientY, l: el.scrollLeft, t: el.scrollTop }
    el.setPointerCapture?.(e.pointerId)
  }, [])
  const onPointerMove = useCallback((e) => {
    const el = scrollRef.current
    if (!el || !drag.current) return
    el.scrollLeft = drag.current.l - (e.clientX - drag.current.x)
    el.scrollTop = drag.current.t - (e.clientY - drag.current.y)
  }, [])
  const endDrag = useCallback((e) => {
    drag.current = null
    scrollRef.current?.releasePointerCapture?.(e.pointerId)
  }, [])

  if (!g.nodes.length) return null

  const litNodes = new Set()
  if (hoverId != null) {
    litNodes.add(hoverId)
    for (const a of (g.ancestorsOf.get(hoverId) || [])) litNodes.add(a)
  }
  const isLit = (id) => litNodes.has(id)
  // The tree's own pill behind a laid-out one: same step, same finality.
  const originalOf = (node, laid) => (node.checkpoints || [])
    .find((c) => c.step === laid.step && !!c.final === !!laid.final) || laid
  const vw = g.width * scale
  const vh = g.height * scale

  return (
    <>
      <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[0.625rem] text-content-subtle">
        <button type="button" onClick={toggleBigPreviews} aria-pressed={bigPreviews}
          title={bigPreviews ? 'Back to compact pills' : 'Enlarge the sample stills to compare steps at a glance'}
          className={'rounded-md border px-2 py-0.5 text-[0.625rem] font-semibold transition-colors '
            + (bigPreviews
              ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-100 '
              : 'border-border bg-app/60 text-content-muted hover:text-content ')}>
          🔍 Big previews
        </button>
        <span>Click a save for its actions · a run card for its details · a still to play the sample</span>
      </div>
      <div ref={scrollRef} className="lds-lgraph-scroll relative overflow-auto rounded-xl"
        style={{ maxHeight: MAX_H }} data-probe-panel="video-lineage-graph"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove}
        onPointerUp={endDrag} onPointerCancel={endDrag}>
        <svg className="lds-lgraph block" width={vw} height={vh}
          viewBox={`0 0 ${g.width} ${g.height}`} style={{ minHeight: Math.min(vh, MAX_H) }}
          role="img" aria-label={`Lineage graph: ${g.nodes.length} runs`}>
          <LineageEdgeDefs />
          <LineageEdges edges={g.edges} isLit={isLit} />
          <g>
            {g.nodes.map((n) => (
              <foreignObject key={n.node.record_id} className="lds-gnode overflow-visible"
                x={n.x} y={n.y} width={CARD_W} height={n.cellH}
                style={{ '--enter-delay': `${Math.min(n.depth, 8) * 90 + 40}ms` }}
                onPointerEnter={() => setHoverId(n.node.record_id)}
                onPointerLeave={() => setHoverId((cur) => (cur === n.node.record_id ? null : cur))}>
                <div style={{ position: 'relative', width: CARD_W, height: n.cellH }}>
                  <GraphCard node={n.node} lit={isLit(n.node.record_id)} annotated={false}
                    compareRole={null}
                    onSelect={n.node.source === 'cloud' && typeof onDetails === 'function'
                      ? (node) => onDetails(node) : undefined} />
                  {n.checkpoints.map((p) => (
                    <CheckpointPill key={`${p.step}-${p.filename ?? p.x}`}
                      pill={p} offX={p.x - n.x} offY={p.y - n.y}
                      active={openCk?.laid === p} selected={false}
                      preview={pillPreview(originalOf(n.node, p))} big={bigPreviews}
                      resultNoun="sample" resultIcon="🎬" deployHint={videoDeployHint}
                      onOpen={(laid, event) => {
                        const rect = event.currentTarget.getBoundingClientRect()
                        setOpenCk({ node: n.node, laid, pill: originalOf(n.node, laid),
                          anchor: { x: rect.left + rect.width / 2, y: rect.bottom } })
                      }}
                      onOpenGallery={typeof onPlaySample === 'function'
                        ? (laid) => onPlaySample(n.node, originalOf(n.node, laid)) : undefined}
                      onZoomPreview={typeof onPlaySample === 'function'
                        ? () => onPlaySample(n.node, originalOf(n.node, p)) : undefined} />
                  ))}
                </div>
              </foreignObject>
            ))}
          </g>
        </svg>
      </div>
      {openCk && (
        <FloatingVideoMenu anchor={openCk.anchor} onClose={closePopover}>
          <VideoCheckpointPopover node={openCk.node} pill={openCk.pill}
            a={pillActionModel(datasetId, openCk.node, openCk.pill, ctx)}
            busy={busy}
            onDeploy={onDeploy} onUndeploy={onUndeploy} onDelete={onDelete}
            onContinue={onContinue} onDetails={onDetails} onPlaySample={onPlaySample}
            onClose={closePopover} />
        </FloatingVideoMenu>
      )}
    </>
  )
}
