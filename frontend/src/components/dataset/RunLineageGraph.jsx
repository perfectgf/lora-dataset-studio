import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { buildLineageGraph, CARD_W, CARD_H } from '../../utils/lineageGraph';
import { resumeCaption } from '../../utils/lineageTree';
import { famLabel, StatusDot, SavesChip } from './lineageChrome';
import LineageDetailPanel from './LineageDetailPanel';
import LineageDiffPanel from './LineageDiffPanel';
import { noteBadge, toggleDiffSelection } from './lineageDetail.js';
import { removeRunFromTree } from '../../utils/runDeletable.js';
import { postJson } from '../../api/fetchClient';
import { loraFolderLabel } from '../../utils/checkpointBrowser';
import { useToast } from '../common/Toast';
import {
  checkpointKey, toggleCheckpointSelection, selectedCheckpointRefs,
  describePreviewSelection, parseSeedInput,
  checkpointDeployed, lineageImportPayload,
} from './lineagePreview.js';

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

/** One checkpoint as a compact pill: its step, a ✓ for the final save, an indigo
 *  ring when it's the point another run branched off, and — the Lab flagship — a
 *  select checkbox (deployed checkpoints only) plus its inline generated preview
 *  (thumbnail when done, a ◌ while it renders, a ⚠ if it failed). Clicking the
 *  body opens the pill's actions; the checkbox toggles it into the shared-prompt
 *  generation batch. Absolutely positioned at the exact box the layout computed. */
function CheckpointPill({ pill, offX, offY, active, selected, preview, big, onOpen, onToggleSelect, onZoomPreview }) {
  const gone = pill.present === false;
  const st = preview?.status || null;
  const label = pill.step >= 1000 && pill.step % 1000 === 0 ? `${pill.step / 1000}k` : pill.step;
  const zoom = (e) => { e.stopPropagation(); onZoomPreview?.(preview.url, pill.step); };
  const zoomKey = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); zoom(e); } };
  const shellCls = 'lds-ckpill rounded-md border transition-colors '
    + (gone
      ? 'border-dashed border-border bg-transparent text-content-subtle '
      : pill.final
        ? 'border-emerald-400/50 bg-emerald-500/10 text-emerald-200 '
        : 'border-border bg-app/70 text-content-muted hover:border-indigo-400/50 hover:text-content ')
    + (pill.isResumeSource ? 'ring-1 ring-indigo-400/60 border-indigo-400/60 ' : '')
    + (selected ? 'ring-2 ring-indigo-400/80 border-indigo-400/70 ' : active ? 'ring-2 ring-indigo-400/80 ' : '');
  const openTitle = `Checkpoint at step ${pill.step}${pill.final ? ' — final' : ''}${pill.isResumeSource ? ' — a run continued from here' : ''}${st ? ` — preview ${st}` : ''}`;
  return (
    <div style={{ position: 'absolute', left: offX, top: offY, width: pill.w, height: pill.h }}
      className="lds-ckpill-wrap">
      {big ? (
        // 🔍 Big-preview tile: a large generated image on top (ComfyUI-style — click
        // it to view full-screen), with a step label strip underneath that opens
        // the pill's actions. The whole tile still opens the popover except the
        // image, which zooms.
        <button type="button"
          onClick={(e) => { e.stopPropagation(); onOpen(pill); }}
          title={openTitle}
          style={{ width: pill.w, height: pill.h }}
          className={shellCls + ' flex w-full flex-col overflow-hidden text-[0.625rem] font-medium tabular-nums'}>
          <div className="relative min-h-0 flex-1 w-full">
            {preview?.url ? (
              <img src={preview.url} alt={`Preview at step ${pill.step}`}
                role="button" tabIndex={0} title="Click to view this preview full-screen"
                onClick={zoom} onKeyDown={zoomKey}
                className="h-full w-full cursor-zoom-in object-cover hover:opacity-90" />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-base">
                {st === 'pending' ? <span aria-hidden title="Generating preview…" className="animate-pulse text-indigo-300">◌</span>
                  : st === 'failed' ? <span aria-hidden title="Preview failed" className="text-amber-300">⚠</span>
                  : <span aria-hidden className="opacity-50">💾</span>}
              </div>
            )}
          </div>
          <span className="flex shrink-0 items-center justify-center gap-0.5 border-t border-border bg-black/20 py-0.5 leading-none">
            {pill.final && <span aria-hidden className="text-emerald-300">✓</span>}
            <span>{label}</span>
          </span>
        </button>
      ) : (
        <button type="button"
          onClick={(e) => { e.stopPropagation(); onOpen(pill); }}
          title={openTitle}
          style={{ width: pill.w, height: pill.h }}
          className={shellCls + ' flex w-full items-center justify-center gap-0.5 text-[0.5625rem] font-medium tabular-nums'}>
          {pill.final && <span aria-hidden className="text-emerald-300">✓</span>}
          <span>{label}</span>
          {preview?.url ? (
            // The thumbnail is tiny by necessity (the pill is 60×20). Clicking it
            // opens the preview LARGE in a lightbox — a DISTINCT action from the
            // pill's popover, so stopPropagation keeps the two from colliding.
            <img src={preview.url} alt={`Preview at step ${pill.step}`} width={14} height={14}
              role="button" tabIndex={0}
              title="Click to view this preview large"
              onClick={zoom} onKeyDown={zoomKey}
              className="ml-0.5 h-3.5 w-3.5 shrink-0 cursor-zoom-in rounded-sm object-cover ring-1 ring-black/30 hover:ring-indigo-400/80" />
          ) : st === 'pending' ? (
            <span aria-hidden title="Generating preview…" className="ml-0.5 animate-pulse text-indigo-300">◌</span>
          ) : st === 'failed' ? (
            <span aria-hidden title="Preview failed" className="ml-0.5 text-amber-300">⚠</span>
          ) : (
            <span aria-hidden className="opacity-70">💾</span>
          )}
        </button>
      )}
      {/* Select for the shared-prompt preview batch — deployed checkpoints only
          (nothing to load otherwise). A corner box; clicking it never opens the
          popover. Slightly larger in big mode so it stays clickable on the tile. */}
      {pill.testable && typeof onToggleSelect === 'function' && (
        <button type="button" role="checkbox" aria-checked={selected}
          aria-label={`Select step ${pill.step} for preview`}
          title={selected ? 'Selected for preview' : 'Select for preview'}
          onClick={(e) => { e.stopPropagation(); onToggleSelect(pill); }}
          style={{ position: 'absolute', left: -5, top: -5 }}
          className={'lds-cksel flex items-center justify-center rounded-[3px] border leading-none shadow-sm '
            + (big ? 'h-5 w-5 text-[0.6875rem] ' : 'h-3.5 w-3.5 text-[0.5rem] ')
            + (selected ? 'border-indigo-400 bg-indigo-500 text-white ' : 'border-border bg-surface-overlay text-transparent hover:border-indigo-400/70 ')}>
          ✓
        </button>
      )}
    </div>
  );
}

export default function RunLineageGraph({ tree, onSelect, onContinueCheckpoint, refetchTree }) {
  const toast = useToast();
  // Runs removed in-session (a gone run deleted from the detail panel) drop from
  // the graph without a full refetch; children re-root via removeRunFromTree.
  const [deletedIds, setDeletedIds] = useState([]);
  const shownTree = useMemo(
    () => deletedIds.reduce((t, id) => removeRunFromTree(t, id), tree),
    [tree, deletedIds]);
  // 🔍 Big-preview mode: enlarge the generated thumbnails into ComfyUI-style tiles
  // so epochs compare at a glance without opening each. Persisted; default compact.
  const [bigPreviews, setBigPreviews] = useState(() => {
    try { return localStorage.getItem('lds.graphBigPreviews') === '1'; } catch { return false; }
  });
  const toggleBigPreviews = useCallback(() => {
    setBigPreviews((v) => {
      const next = !v;
      try { localStorage.setItem('lds.graphBigPreviews', next ? '1' : '0'); } catch { /* ignore */ }
      return next;
    });
  }, []);
  const g = useMemo(() => buildLineageGraph(shownTree, { bigPreviews }), [shownTree, bigPreviews]);
  const scrollRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [hoverId, setHoverId] = useState(null);
  // The open checkpoint popover: { node, pill } | null.
  const [openCk, setOpenCk] = useState(null);
  const closePopover = useCallback(() => setOpenCk(null), []);
  // 📦 Deploying a checkpoint straight from its pill popover (Import → loras/…).
  const [importing, setImporting] = useState(false);
  // A preview thumbnail opened LARGE in a lightbox: { url, step } | null.
  const [bigPreview, setBigPreview] = useState(null);
  const zoomPreview = useCallback((url, step) => setBigPreview({ url, step }), []);
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
  // A gone run removed from the panel: drop it from the graph and close the panel.
  const handleNodeDeleted = useCallback((recordId) => {
    setDeletedIds((ids) => (ids.includes(recordId) ? ids : [...ids, recordId]));
    setOpenNode(null);
  }, []);

  // --- Lab inline generation (slice 3) --------------------------------------
  // Checked checkpoints (Set of `${record_id}:${step}`) get ONE shared prompt +
  // seed and a strength-1.0 preview each, produced by the reused Test-Studio
  // engine. `pillByKey` resolves a key to {record_id, step, testable} so the
  // request carries only deployable picks and the bar can say why it's disabled.
  const [selectedCk, setSelectedCk] = useState(() => new Set());
  const [genPrompt, setGenPrompt] = useState('');
  const [genSeed, setGenSeed] = useState('');
  const [gen, setGen] = useState({ busy: false, error: null, note: null });
  // Optimistic + polled preview overlay, key -> { status, url }, so a pill shows
  // ◌ pending the moment a job is queued and flips to the thumbnail on its own.
  const [previewOverlay, setPreviewOverlay] = useState({});
  const pollRef = useRef(null);

  const pillByKey = useMemo(() => {
    const m = new Map();
    for (const n of g.nodes) {
      for (const p of n.checkpoints) {
        m.set(checkpointKey(n.node.record_id, p.step),
          { record_id: n.node.record_id, step: p.step, testable: p.testable === true });
      }
    }
    return m;
  }, [g.nodes]);
  const previewOf = useCallback((recordId, pill) => {
    const o = previewOverlay[checkpointKey(recordId, pill.step)];
    if (o) return o;
    if (pill.preview_status || pill.preview_url) return { status: pill.preview_status, url: pill.preview_url };
    return null;
  }, [previewOverlay]);
  const toggleCk = useCallback((recordId, pill) => {
    setSelectedCk((sel) => toggleCheckpointSelection(sel, checkpointKey(recordId, pill.step)));
  }, []);
  const sel = describePreviewSelection(selectedCk, pillByKey);
  const datasetId = g.nodes[0]?.node.dataset_id ?? null;

  // Merge fresh preview state from a refetched tree into the overlay (a pill
  // reads the overlay first), then stop polling once nothing is pending.
  const mergeFromTree = useCallback((t) => {
    const next = {};
    let stillPending = false;
    for (const node of (t?.nodes || [])) {
      for (const c of (node.checkpoints || [])) {
        if (!c.preview_status && !c.preview_url) continue;
        next[checkpointKey(node.record_id, c.step)] = { status: c.preview_status, url: c.preview_url };
        if (c.preview_status === 'pending') stillPending = true;
      }
    }
    setPreviewOverlay((cur) => ({ ...cur, ...next }));
    return stillPending;
  }, []);

  // 📦 Import → loras/<family>: deploy THIS checkpoint into ComfyUI straight from
  // its pill, closing the see-it → use-it loop without leaving the graph. Uses
  // postJson (CSRF header + one-shot refresh) — a bare fetch is rejected 400 by
  // Flask-WTF, the same trap that broke browser Generate. The payload mirrors the
  // flat checkpoint list EXACTLY (a cloud pill rides its cloud_run_id). On success
  // we refetch the lineage so the freshly-deployed pill flips to `testable` (✓
  // Deployed + eligible for inline Generate).
  const handleImport = useCallback(async (node, pill) => {
    const body = lineageImportPayload(node, pill);
    if (datasetId == null || !body) return;
    setImporting(true);
    try {
      const d = await postJson(`/api/dataset/${datasetId}/train/import`, body);
      toast.success(d?.note || `LoRA imported: ${d?.dest || pill.filename}`);
      setOpenCk(null);
      if (typeof refetchTree === 'function') {
        try { const t = await refetchTree(); if (t) mergeFromTree(t); } catch { /* the list still updated server-side */ }
      }
    } catch (e) {
      toast.error(e?.message || 'Import failed');
    } finally {
      setImporting(false);
    }
  }, [datasetId, refetchTree, mergeFromTree, toast]);

  const handleGenerate = useCallback(async () => {
    const refs = selectedCheckpointRefs(selectedCk, pillByKey);
    if (!refs.length || datasetId == null) return;
    const seedParsed = parseSeedInput(genSeed);
    if (seedParsed.error) { setGen({ busy: false, error: seedParsed.error, note: null }); return; }
    // Family = the family of the first selected checkpoint's run (a lineage is one
    // dataset; the engine can't mix families, so all picks share it).
    const firstNode = g.nodes.find((n) => n.node.record_id === refs[0].record_id);
    const family = firstNode?.node.train_type || null;
    setGen({ busy: true, error: null, note: null });
    try {
      // postJson (not raw fetch) so the state-changing POST carries the X-CSRFToken
      // header + the app's one-shot CSRF-refresh retry — a bare fetch is rejected
      // 400 (CSRF missing) by Flask-WTF, which is exactly what broke browser Generate.
      const body = await postJson(`/api/dataset/${datasetId}/lineage/previews`, {
        prompt: genPrompt || null, seed: seedParsed.seed, family, checkpoints: refs,
      });
      // Optimistically mark the queued checkpoints as rendering, clear the picks.
      setPreviewOverlay((cur) => {
        const nx = { ...cur };
        for (const r of refs) nx[checkpointKey(r.record_id, r.step)] = { status: 'pending', url: null };
        return nx;
      });
      const skipped = (body.skipped || []).length;
      setGen({ busy: false, error: null,
        note: `Generating ${body.queued} preview${body.queued > 1 ? 's' : ''}${skipped ? ` · ${skipped} skipped (not deployed)` : ''}` });
      setSelectedCk(new Set());
      // Poll the lineage for the finished images if the parent gave us a refetch.
      if (typeof refetchTree === 'function') {
        if (pollRef.current) clearInterval(pollRef.current);
        let tries = 0;
        pollRef.current = setInterval(async () => {
          tries += 1;
          let t = null;
          try { t = await refetchTree(); } catch { /* transient */ }
          const pending = t ? mergeFromTree(t) : true;
          if ((!pending || tries >= 15) && pollRef.current) {
            clearInterval(pollRef.current); pollRef.current = null;
          }
        }, 4000);
      }
    } catch (e) {
      setGen({ busy: false, error: e?.message || 'Generation failed', note: null });
    }
  }, [selectedCk, pillByKey, datasetId, genSeed, genPrompt, g.nodes, refetchTree, mergeFromTree]);

  useLayoutEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // Esc closes the preview lightbox from anywhere (a window listener, not div
  // focus — the backdrop div isn't reliably focused when the image is clicked).
  useLayoutEffect(() => {
    if (!bigPreview) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') setBigPreview(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [bigPreview]);

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
  // Continue from a checkpoint of any STOPPED cloud run. A run that failed (e.g.
  // 'pod did not become ready in time') can still hold a valid harvested save, so
  // for a non-'done' run we additionally require THIS pill to be present
  // (downloadable). An actively-running run is never offered Continue.
  const canContinue = (node, pill) => typeof onContinueCheckpoint === 'function'
    && node.source === 'cloud' && node.run_id != null
    && (node.status === 'done'
        || (['error', 'error_pod_kept', 'stopped', 'failed'].includes(node.status)
            && !!pill?.download_url));

  return (
    <>
    <div className="mb-1.5 flex items-center justify-end gap-2 text-[0.625rem] text-content-subtle">
      {/* 🔍 Big-preview mode: enlarge the generated tiles to compare epochs at a
          glance (ComfyUI-style), no clicking each. Persisted; default compact. */}
      <button type="button" onClick={toggleBigPreviews}
        aria-pressed={bigPreviews}
        title={bigPreviews ? 'Back to compact pills' : 'Enlarge the generated previews to compare checkpoints at a glance'}
        className={'mr-auto rounded-md border px-2 py-0.5 text-[0.625rem] font-semibold transition-colors '
          + (bigPreviews
            ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-100 '
            : 'border-border bg-app/60 text-content-muted hover:text-content ')}>
        🔍 Big previews
      </button>
      {selectedForDiff.length === 0 ? (
        <span><span className="font-semibold">⇧ Shift-click</span> two runs to compare · check <span aria-hidden>☑</span> checkpoints to preview them</span>
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
    {/* 🎨 Generation bar — appears once a checkpoint is checked. ONE shared prompt
        + seed renders a strength-1.0 preview per selected checkpoint (reusing the
        Test-Studio engine), so a LoRA's epoch-by-epoch evolution reads at a glance.
        Disabled with an honest reason when the picks aren't deployable. */}
    {selectedCk.size > 0 && (
      <div className="lds-lgen mb-2 rounded-xl border border-indigo-400/40 bg-indigo-500/5 p-2.5">
        <div className="mb-1.5 flex items-center gap-2 text-[0.6875rem]">
          <span className="font-semibold text-content">🎨 Generate previews</span>
          <span className="text-content-muted">{sel.testableCount} checkpoint{sel.testableCount !== 1 ? 's' : ''}, one shared prompt + seed, strength 1.0</span>
          <button type="button" onClick={() => setSelectedCk(new Set())}
            className="ml-auto text-content-subtle underline decoration-dotted hover:text-content">Clear</button>
        </div>
        <textarea value={genPrompt} onChange={(e) => setGenPrompt(e.target.value)}
          rows={2} placeholder="Shared prompt — leave blank to use the dataset's identity prompt (trigger)"
          className="w-full resize-y rounded-md border border-border bg-app/60 px-2 py-1.5 text-[0.6875rem] text-content placeholder:text-content-subtle focus:border-indigo-400/60 focus:outline-none" />
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-1 text-[0.625rem] text-content-muted">
            Seed
            <input value={genSeed} onChange={(e) => setGenSeed(e.target.value)}
              inputMode="numeric" placeholder="random"
              className="w-24 rounded-md border border-border bg-app/60 px-1.5 py-1 text-[0.6875rem] tabular-nums text-content placeholder:text-content-subtle focus:border-indigo-400/60 focus:outline-none" />
          </label>
          <button type="button" onClick={handleGenerate} disabled={!sel.enabled || gen.busy}
            className={'rounded-md px-3 py-1 text-[0.6875rem] font-semibold '
              + (sel.enabled && !gen.busy
                ? 'bg-indigo-500 text-white hover:bg-indigo-400 '
                : 'cursor-not-allowed bg-app/60 text-content-subtle ')}>
            {gen.busy ? 'Generating…' : 'Generate'}
          </button>
          {sel.hint && <span className="text-[0.625rem] text-amber-200/90">{sel.hint}</span>}
          {gen.error && <span className="text-[0.625rem] text-red-300">{gen.error}</span>}
          {gen.note && !gen.error && <span className="text-[0.625rem] text-emerald-300">{gen.note}</span>}
        </div>
      </div>
    )}
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
                    selected={selectedCk.has(checkpointKey(n.node.record_id, p.step))}
                    preview={previewOf(n.node.record_id, p)}
                    big={bigPreviews}
                    onOpen={(pill) => setOpenCk({ node: n.node, pill })}
                    onToggleSelect={(pill) => toggleCk(n.node.record_id, pill)}
                    onZoomPreview={zoomPreview} />
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
          const POP_W = 210, POP_H = 152;
          const below = openCk.pill.y + openCk.pill.h + 4;
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
                {canContinue(openCk.node, openCk.pill) && (
                  <button type="button"
                    onClick={() => { onContinueCheckpoint(openCk.node, openCk.pill); closePopover(); }}
                    className="flex items-center gap-1.5 rounded-md border border-indigo-400/40 bg-indigo-500/15 px-2 py-1 text-indigo-100 text-[0.6875rem] font-medium hover:bg-indigo-500/25">
                    <span aria-hidden>▶</span> Continue from here
                  </button>
                )}
                {/* 📦 Import → loras/<family>: deploy on the spot. Already-deployed
                    pills show "✓ Deployed" instead (nothing to do twice). Only
                    importable pills (a file + a resolvable run) offer the button. */}
                {checkpointDeployed(openCk.pill) ? (
                  <span className="flex items-center gap-1.5 rounded-md border border-emerald-500/40 bg-emerald-600/10 px-2 py-1 text-emerald-200 text-[0.6875rem] font-medium">
                    <span aria-hidden>✓</span> Deployed
                  </span>
                ) : lineageImportPayload(openCk.node, openCk.pill) ? (
                  <button type="button" disabled={importing}
                    onClick={() => handleImport(openCk.node, openCk.pill)}
                    title={`Deploy this checkpoint into ComfyUI's ${loraFolderLabel(openCk.node.train_type)} folder so you can test and generate with it`}
                    className="flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/20 px-2 py-1 text-white text-[0.6875rem] font-medium hover:bg-primary/30 disabled:cursor-not-allowed disabled:opacity-50">
                    <span aria-hidden>📦</span> {importing ? 'Importing…' : `Import → ${loraFolderLabel(openCk.node.train_type)}`}
                  </button>
                ) : null}
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
        onNodeChanged={handleNodeChanged} onNodeDeleted={handleNodeDeleted} />
    )}
    {/* 🔍 Preview lightbox — a checkpoint's generated image LARGE, so epochs read
        in ComfyUI spirit (the pill thumbnails are only 14px). Esc / backdrop /
        image click closes. Fixed + high z-index so it floats over everything. */}
    {bigPreview && (
      <div role="dialog" aria-modal="true" aria-label={`Preview at step ${bigPreview.step}`}
        className="fixed inset-0 z-[9997] flex flex-col items-center justify-center bg-black/90 p-4"
        onClick={() => setBigPreview(null)}>
        <button type="button" onClick={(e) => { e.stopPropagation(); setBigPreview(null); }}
          title="Close (Esc)" aria-label="Close preview"
          className="absolute top-3 right-3 z-10 flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20">✕</button>
        <img src={bigPreview.url} alt={`Generated preview at step ${bigPreview.step}`}
          onClick={(e) => e.stopPropagation()}
          className="max-h-[88vh] max-w-full select-none rounded-lg object-contain shadow-2xl" />
        <span className="mt-2 text-white/70 text-[0.75rem] tabular-nums">Step {bigPreview.step.toLocaleString?.() ?? bigPreview.step} · click outside or Esc to close</span>
      </div>
    )}
    </>
  );
}
