import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { buildLineageGraph, CARD_W, CARD_H, PILL_W, PILL_H } from '../../utils/lineageGraph';
import { resumeCaption } from '../../utils/lineageTree';
import { famLabel, StatusDot, SavesChip } from './lineageChrome';
import LineageDetailPanel from './LineageDetailPanel';
import LineageDiffPanel from './LineageDiffPanel';
import { noteBadge, toggleDiffSelection } from './lineageDetail.js';

/* ◉ Graph view of a run's lineage — the showcase rendering. A tidy left-to-right
   tree: the root on the left, each continuation one generation to the right,
   forks stacking. Cards carry the same vocabulary as the list (status dot, ☁/💻,
   family, steps, v{n}, 💾), the current run wears an indigo glow, and the runs
   are joined by flowing bezier edges whose gradient runs parent→child.

   Under each run sit its CHECKPOINTS as sober pills (step · 💾). A continuation's
   run→run edge starts from the exact pill it resumed from, so the graph reads
   "this run started from THIS checkpoint". Click a pill for its actions
   (⬇ download, ▶ continue from here). The trunk (root→current) is drawn brighter;
   a superseded branch is dashed and dimmed. Hover any run to light its whole path
   back to the root. SVG-native (no graph library); geometry comes from
   utils/lineageGraph.js so the pills line up exactly with the edge anchors. */

const MIN_SCALE = 0.5;   // shrink to fit down to here, then pan instead
const MAX_H = 560;       // the panel never grows taller than this before it pans

/** One run as a fixed-size card. Mirrors the list card's content, sized to the
 *  graph's card box; sits at the top of the run's cell (pills go below). */
function GraphCard({ node, lit, annotated, compareRole, onSelect }) {
  const cur = node.is_current;
  const dim = node.checkpoint_ready === false;
  const clickable = typeof onSelect === 'function';
  return (
    <div
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? (e) => onSelect(node, e) : undefined}
      onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(node, e); } } : undefined}
      title={clickable ? 'Click to inspect · Shift-click to compare' : undefined}
      style={{ height: CARD_H }}
      className={'lds-gcard flex w-full flex-col justify-center gap-1 rounded-xl border px-2.5 py-1.5 '
        + (cur
          ? 'lds-gcard-current border-indigo-400/70 bg-indigo-500/10 ring-1 ring-indigo-400/30 '
          : dim
            ? 'border-border bg-app/40 '
            : 'border-border bg-surface-raised ')
        + (lit && !cur ? 'ring-1 ring-indigo-300/40 border-indigo-400/50 ' : '')
        + (compareRole ? 'ring-2 ring-amber-400/70 border-amber-400/60 ' : '')
        + (clickable ? 'cursor-pointer' : '')}>
      <div className="flex min-w-0 items-center gap-1.5">
        <StatusDot status={node.status} />
        <span className="shrink-0 font-mono text-content-muted text-[0.625rem]">
          <span aria-hidden>{node.source === 'cloud' ? '☁' : '💻'}</span>{' '}
          #{node.source === 'cloud' && node.run_id ? node.run_id : node.record_id}
        </span>
        <span className={`min-w-0 truncate text-[0.75rem] font-semibold ${dim ? 'text-content-muted' : 'text-content'}`}
          title={`${famLabel(node.train_type)}${node.variant ? ` · ${node.variant}` : ''}`}>
          {famLabel(node.train_type)}{node.variant ? <span className="font-normal text-content-muted"> · {node.variant}</span> : null}
        </span>
        {cur && (
          <span className="shrink-0 rounded-full bg-indigo-500/25 px-1.5 py-0.5 text-indigo-100 text-[0.5rem] font-bold uppercase tracking-wider">
            this run
          </span>
        )}
        {annotated && (
          <span aria-hidden title="Has notes" className="shrink-0 text-amber-300 text-[0.625rem] leading-none">●</span>
        )}
        {compareRole && (
          <span title={`Selected for compare (${compareRole})`}
            className="shrink-0 rounded-full bg-amber-500/25 px-1.5 py-0.5 text-amber-100 text-[0.5rem] font-bold uppercase tracking-wider">
            {compareRole}
          </span>
        )}
        <span className="ml-auto shrink-0"><SavesChip node={node} /></span>
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-content-subtle text-[0.5625rem]">
        {node.version != null && (
          <span className="rounded bg-app/60 px-1 py-px font-medium text-content-muted">v{node.version}</span>
        )}
        {node.steps ? <span className="tabular-nums">{node.steps.toLocaleString()} steps</span> : null}
        {resumeCaption(node) && (
          <span className="inline-flex items-center gap-0.5">
            <span aria-hidden className="text-[0.625rem] leading-none">↳</span>{resumeCaption(node)}
          </span>
        )}
        {node.origin_unknown && (
          <span className="italic" title="This run resumed from an earlier checkpoint, but its source run predates lineage tracking">
            origin not recorded
          </span>
        )}
      </div>
    </div>
  );
}

/** One checkpoint as a compact pill: its step, 💾, a ✓ for the final save, and an
 *  indigo ring when it's the point another run branched off. Clicking opens the
 *  pill's actions. Absolutely positioned inside the run cell at the exact box the
 *  layout computed, so the edge anchored on it lands dead-centre. */
function CheckpointPill({ pill, offX, offY, active, onOpen }) {
  const gone = pill.present === false;
  return (
    <button type="button"
      onClick={(e) => { e.stopPropagation(); onOpen(pill); }}
      title={`Checkpoint at step ${pill.step}${pill.final ? ' — final' : ''}${pill.isResumeSource ? ' — a run continued from here' : ''}`}
      style={{ position: 'absolute', left: offX, top: offY, width: PILL_W, height: PILL_H }}
      className={'lds-ckpill flex items-center justify-center gap-0.5 rounded-md border text-[0.5625rem] font-medium tabular-nums transition-colors '
        + (gone
          ? 'border-dashed border-border bg-transparent text-content-subtle '
          : pill.final
            ? 'border-emerald-400/50 bg-emerald-500/10 text-emerald-200 '
            : 'border-border bg-app/70 text-content-muted hover:border-indigo-400/50 hover:text-content ')
        + (pill.isResumeSource ? 'ring-1 ring-indigo-400/60 border-indigo-400/60 ' : '')
        + (active ? 'ring-2 ring-indigo-400/80 ' : '')}>
      {pill.final && <span aria-hidden className="text-emerald-300">✓</span>}
      <span>{pill.step >= 1000 && pill.step % 1000 === 0 ? `${pill.step / 1000}k` : pill.step}</span>
      <span aria-hidden className="opacity-70">💾</span>
    </button>
  );
}

export default function RunLineageGraph({ tree, onSelect, onContinueCheckpoint }) {
  const g = useMemo(() => buildLineageGraph(tree), [tree]);
  const scrollRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [hoverId, setHoverId] = useState(null);
  // The open checkpoint popover: { node, pill } | null.
  const [openCk, setOpenCk] = useState(null);
  const closePopover = useCallback(() => setOpenCk(null), []);
  // The Lab detail panel's open node (click a run card to inspect its config).
  const [openNode, setOpenNode] = useState(null);
  // Bounded-to-2 "compare" selection (record ids) — a DISTINCT interaction from
  // the single-run inspector: SHIFT-click toggles a run in/out of the compare
  // set; a plain click still opens the inspector (slice-1 behaviour untouched).
  const [selectedForDiff, setSelectedForDiff] = useState([]);
  const handleNodeClick = useCallback((node, e) => {
    if (e && e.shiftKey) {
      setSelectedForDiff((sel) => toggleDiffSelection(sel, node.record_id));
      return;   // compare only — don't open the inspector or jump the Runs hub
    }
    setOpenNode(node);
    if (typeof onSelect === 'function') onSelect(node);   // keep the Runs-hub jump
  }, [onSelect]);
  // record_id -> node, so the two picked ids resolve to the nodes the diff reads.
  const nodeById = useMemo(() => {
    const m = new Map();
    for (const n of g.nodes) m.set(n.node.record_id, n.node);
    return m;
  }, [g.nodes]);
  // Note edits happen in the panel; mirror them here (record_id -> updated node)
  // so the ● badge lights live without a full refetch of the graph.
  const [noteEdits, setNoteEdits] = useState({});
  const handleNodeChanged = useCallback((updated) => {
    setNoteEdits((m) => ({ ...m, [updated.record_id]: updated }));
    setOpenNode((cur) => (cur && cur.record_id === updated.record_id ? updated : cur));
  }, []);

  // Fit horizontally to the panel, shrinking no further than MIN_SCALE (then the
  // panel pans). Re-measured on resize so it always poses well in a screenshot.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || !g.width) return;
    const measure = () => {
      const avail = el.clientWidth || g.width;
      const s = Math.max(MIN_SCALE, Math.min(1, (avail - 4) / g.width));
      setScale(s);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [g.width]);

  // Drag-to-pan when the tree overflows the panel — a light grab, not a zoom UI.
  const drag = useRef(null);
  const onPointerDown = useCallback((e) => {
    const el = scrollRef.current;
    if (!el) return;
    // A press outside a pill/card/popover dismisses an open popover.
    if (!e.target.closest('.lds-ckpill') && !e.target.closest('.lds-ck-popover')) setOpenCk(null);
    const overflow = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
    if (!overflow || e.target.closest('.lds-gcard') || e.target.closest('.lds-ckpill')
        || e.target.closest('.lds-ck-popover')) return; // let cards/pills take clicks
    drag.current = { x: e.clientX, y: e.clientY, l: el.scrollLeft, t: el.scrollTop };
    el.setPointerCapture?.(e.pointerId);
    el.classList.add('is-grabbing');
  }, []);
  const onPointerMove = useCallback((e) => {
    const el = scrollRef.current;
    if (!el || !drag.current) return;
    el.scrollLeft = drag.current.l - (e.clientX - drag.current.x);
    el.scrollTop = drag.current.t - (e.clientY - drag.current.y);
  }, []);
  const endDrag = useCallback((e) => {
    const el = scrollRef.current;
    drag.current = null;
    el?.classList.remove('is-grabbing');
    el?.releasePointerCapture?.(e.pointerId);
  }, []);

  if (!g.nodes.length) return null;

  // A node is "lit" when it's the hovered run or one of its ancestors; an edge is
  // lit when both its ends are — so hover traces the path back to the root.
  const litNodes = new Set();
  if (hoverId != null) {
    litNodes.add(hoverId);
    for (const a of (g.ancestorsOf.get(hoverId) || [])) litNodes.add(a);
  }
  const isLit = (id) => litNodes.has(id);

  // Which compare slot a run holds, if any: first pick = A, second = B.
  const diffRole = (id) => {
    const i = selectedForDiff.indexOf(id);
    return i === 0 ? 'A' : i === 1 ? 'B' : null;
  };

  const vw = g.width * scale, vh = g.height * scale;
  const capped = Math.min(vh, MAX_H);
  // Can this checkpoint be continued from? Only cloud runs carry a run_id and the
  // Runs hub's Continue flow is cloud-only — mirror that here (a local run shows
  // download only). TODO(lineage): once local resume is wired into this view and
  // generations can be launched from a node (with their results shown, and a
  // Test-Studio graph), extend this popover with those actions.
  const canContinue = (node) => typeof onContinueCheckpoint === 'function'
    && node.source === 'cloud' && node.run_id != null && node.status === 'done';

  return (
    <>
    <div className="mb-1.5 flex items-center justify-end gap-2 text-[0.625rem] text-content-subtle">
      {selectedForDiff.length === 0 ? (
        <span><span className="font-semibold">⇧ Shift-click</span> two runs to compare their settings</span>
      ) : (
        <>
          <span className="text-amber-200">
            {selectedForDiff.length === 1 ? 'Shift-click another run to compare' : 'Comparing two runs →'}
          </span>
          <button type="button" onClick={() => setSelectedForDiff([])}
            className="underline decoration-dotted hover:text-content">Clear</button>
        </>
      )}
    </div>
    <div
      ref={scrollRef}
      className="lds-lgraph-scroll relative overflow-auto rounded-xl"
      style={{ maxHeight: MAX_H }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}>
      <svg
        className="lds-lgraph block"
        width={vw} height={vh}
        viewBox={`0 0 ${g.width} ${g.height}`}
        style={{ minHeight: capped }}
        role="img"
        aria-label={`Lineage graph: ${g.nodes.length} runs`}>
        <defs>
          {/* edges flow left→right = parent→child, so a horizontal gradient in
              the path's own box paints the direction of descent. */}
          <linearGradient id="lds-edge-normal" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="rgb(148 163 184)" stopOpacity="0.15" />
            <stop offset="1" stopColor="rgb(203 213 225)" stopOpacity="0.4" />
          </linearGradient>
          <linearGradient id="lds-edge-spine" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#6366f1" stopOpacity="0.6" />
            <stop offset="1" stopColor="#a5b4fc" stopOpacity="0.98" />
          </linearGradient>
          <linearGradient id="lds-edge-super" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#f59e0b" stopOpacity="0.12" />
            <stop offset="1" stopColor="#fbbf24" stopOpacity="0.5" />
          </linearGradient>
          <filter id="lds-edge-glow" x="-20%" y="-40%" width="140%" height="180%">
            <feGaussianBlur stdDeviation="2.2" result="b" />
            <feMerge>
              <feMergeNode in="b" /><feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Glow halo underneath the trunk (root→current), so even short hops read
            as a lit ribbon. Drawn first, then the crisp cores on top. */}
        <g fill="none" strokeLinecap="round" aria-hidden>
          {g.edges.map((e) => {
            if (!(e.onSpine || (isLit(e.parentId) && isLit(e.childId))) || e.superseded) return null;
            return (
              <path key={`glow-${e.parentId}-${e.childId}`}
                d={e.d} stroke="url(#lds-edge-spine)" strokeWidth="5"
                opacity="0.5" filter="url(#lds-edge-glow)" />
            );
          })}
        </g>
        <g fill="none" strokeLinecap="round">
          {g.edges.map((e, i) => {
            const lit = isLit(e.parentId) && isLit(e.childId);
            const spine = e.onSpine || lit;
            const grad = e.superseded ? 'lds-edge-super' : spine ? 'lds-edge-spine' : 'lds-edge-normal';
            return (
              <path key={`${e.parentId}-${e.childId}`}
                className="lds-ledge"
                d={e.d}
                stroke={`url(#${grad})`}
                strokeWidth={spine ? 2.6 : 1.5}
                strokeDasharray={e.superseded ? '2 4' : undefined}
                pathLength="1"
                style={{ '--draw-delay': `${Math.min(i, 10) * 60 + 120}ms` }} />
            );
          })}
        </g>

        <g>
          {g.nodes.map((n) => (
            <foreignObject key={n.node.record_id}
              className="lds-gnode overflow-visible"
              x={n.x} y={n.y} width={CARD_W} height={n.cellH}
              style={{ '--enter-delay': `${Math.min(n.depth, 8) * 90 + 40}ms` }}
              onPointerEnter={() => setHoverId(n.node.record_id)}
              onPointerLeave={() => setHoverId((cur) => (cur === n.node.record_id ? null : cur))}>
              <div style={{ position: 'relative', width: CARD_W, height: n.cellH }}>
                <GraphCard node={n.node} lit={isLit(n.node.record_id)}
                  annotated={noteBadge(noteEdits[n.node.record_id] || n.node)}
                  compareRole={diffRole(n.node.record_id)}
                  onSelect={handleNodeClick} />
                {n.checkpoints.map((p) => (
                  <CheckpointPill key={`${p.step}-${p.filename ?? p.x}`}
                    pill={p} offX={p.x - n.x} offY={p.y - n.y}
                    active={openCk?.pill === p}
                    onOpen={(pill) => setOpenCk({ node: n.node, pill })} />
                ))}
              </div>
            </foreignObject>
          ))}
        </g>

        {/* Actions popover — drawn last so it sits above every node. OPAQUE
            surface (bg-surface-overlay) so the graph behind never shows through.
            Flips ABOVE the pill when there's no room below (bottom rows), and is
            clamped horizontally, so the scroll panel never clips it. */}
        {openCk && (() => {
          const POP_W = 210, POP_H = 112;
          const below = openCk.pill.y + PILL_H + 4;
          const py = below + POP_H > g.height ? Math.max(0, openCk.pill.y - POP_H - 4) : below;
          const px = Math.max(0, Math.min(openCk.pill.x, g.width - POP_W));
          return (
          <foreignObject className="lds-gnode overflow-visible"
            x={px} y={py} width={POP_W + 10} height={POP_H + 8}>
            <div className="lds-ck-popover w-[210px] rounded-lg border border-indigo-400/40 bg-surface-overlay p-2 shadow-xl"
              onPointerDown={(e) => e.stopPropagation()}>
              <div className="mb-1.5 flex items-center gap-1.5">
                <span className="text-content text-[0.6875rem] font-semibold tabular-nums">
                  Step {openCk.pill.step.toLocaleString()}
                </span>
                {openCk.pill.final && (
                  <span className="rounded bg-emerald-500/15 px-1 py-px text-emerald-200 text-[0.5rem] font-semibold uppercase">final</span>
                )}
                <button type="button" onClick={closePopover}
                  className="ml-auto text-content-subtle hover:text-content text-[0.75rem]" aria-label="Close">✕</button>
              </div>
              <div className="flex flex-col gap-1">
                {openCk.pill.download_url ? (
                  <a href={openCk.pill.download_url} download
                    onClick={closePopover}
                    className="flex items-center gap-1.5 rounded-md border border-emerald-500/40 bg-emerald-600/15 px-2 py-1 text-emerald-100 text-[0.6875rem] font-medium no-underline hover:bg-emerald-600/25">
                    <span aria-hidden>⬇</span> Download
                  </a>
                ) : (
                  <span className="rounded-md border border-border bg-app/40 px-2 py-1 text-content-subtle text-[0.625rem]">
                    Download unavailable for this save
                  </span>
                )}
                {canContinue(openCk.node) && (
                  <button type="button"
                    onClick={() => { onContinueCheckpoint(openCk.node, openCk.pill); closePopover(); }}
                    className="flex items-center gap-1.5 rounded-md border border-indigo-400/40 bg-indigo-500/15 px-2 py-1 text-indigo-100 text-[0.6875rem] font-medium hover:bg-indigo-500/25">
                    <span aria-hidden>▶</span> Continue from here
                  </button>
                )}
              </div>
            </div>
          </foreignObject>
          );
        })()}
      </svg>
    </div>
    {/* The right rail hosts ONE drawer at a time: two picked runs → the compare
        diff; otherwise the slice-1 single-run inspector (openNode is preserved
        underneath, so closing the diff returns to whatever was inspected). */}
    {selectedForDiff.length === 2 ? (
      <LineageDiffPanel
        a={nodeById.get(selectedForDiff[0])}
        b={nodeById.get(selectedForDiff[1])}
        onClose={() => setSelectedForDiff([])} />
    ) : (
      <LineageDetailPanel node={openNode} onClose={() => setOpenNode(null)}
        onNodeChanged={handleNodeChanged} />
    )}
    </>
  );
}
