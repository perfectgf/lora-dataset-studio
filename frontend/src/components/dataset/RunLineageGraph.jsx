import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { buildLineageGraph, CARD_W, CARD_H } from '../../utils/lineageGraph';
import { resumeCaption } from '../../utils/lineageTree';
import { famLabel, StatusDot, SavesChip } from './lineageChrome';

/* ◉ Graph view of a run's lineage — the showcase rendering. A tidy left-to-right
   tree: the root on the left, each continuation one generation to the right,
   forks stacking. Cards carry the same vocabulary as the list (status dot, ☁/💻,
   family, steps, v{n}, 💾), the current run wears an indigo glow, and the runs
   are joined by flowing bezier edges whose gradient runs parent→child. The trunk
   (root→current) is drawn brighter; a superseded branch is dashed and dimmed.
   Hover any run to light its whole path back to the root. SVG-native (no graph
   library) so every pixel is ours; geometry comes from utils/lineageGraph.js. */

const MIN_SCALE = 0.5;   // shrink to fit down to here, then pan instead
const MAX_H = 480;       // the panel never grows taller than this before it pans

/** One run as a fixed-size card inside its <foreignObject>. Mirrors the list
 *  card's content, sized to the graph's card box. */
function GraphCard({ node, lit, onSelect }) {
  const cur = node.is_current;
  const dim = node.checkpoint_ready === false;
  const clickable = typeof onSelect === 'function';
  return (
    <div
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? () => onSelect(node) : undefined}
      onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(node); } } : undefined}
      title={clickable ? 'Jump to this run' : undefined}
      className={'lds-gcard flex h-full w-full flex-col justify-center gap-1 rounded-xl border px-2.5 py-1.5 '
        + (cur
          ? 'lds-gcard-current border-indigo-400/70 bg-indigo-500/10 ring-1 ring-indigo-400/30 '
          : dim
            ? 'border-border bg-app/40 '
            : 'border-border bg-surface-raised ')
        + (lit && !cur ? 'ring-1 ring-indigo-300/40 border-indigo-400/50 ' : '')
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

export default function RunLineageGraph({ tree, onSelect }) {
  const g = useMemo(() => buildLineageGraph(tree), [tree]);
  const scrollRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [hoverId, setHoverId] = useState(null);

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
    const overflow = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
    if (!overflow || e.target.closest('.lds-gcard')) return; // let cards take clicks
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

  const vw = g.width * scale, vh = g.height * scale;
  const capped = Math.min(vh, MAX_H);

  return (
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
              x={n.x} y={n.y} width={CARD_W} height={CARD_H}
              style={{ '--enter-delay': `${Math.min(n.depth, 8) * 90 + 40}ms` }}
              onPointerEnter={() => setHoverId(n.node.record_id)}
              onPointerLeave={() => setHoverId((cur) => (cur === n.node.record_id ? null : cur))}>
              <GraphCard node={n.node} lit={isLit(n.node.record_id)} onSelect={onSelect} />
            </foreignObject>
          ))}
        </g>
      </svg>
    </div>
  );
}
