import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { buildLineageGraph, CARD_W } from '../../utils/lineageGraph';
import { GraphCard, CheckpointPill } from './lineageNodes';
import { LineageEdgeDefs, LineageEdges } from './lineageEdges';
import LineageDetailPanel from './LineageDetailPanel';
import LineageDiffPanel from './LineageDiffPanel';
import CheckpointActionsPopover from './CheckpointActionsPopover';
import PreviewLightbox from './PreviewLightbox';
import CheckpointGalleryPanel from '../shared/CheckpointGalleryPanel';
import { checkpointPopoverPlacement, POPOVER_H, POPOVER_W } from './checkpointPopover.js';
import { noteBadge, toggleDiffSelection } from './lineageDetail.js';
import { removeRunFromTree } from '../../utils/runDeletable.js';
import { postJson } from '../../api/fetchClient';
import { loraFolderLabel } from '../../utils/checkpointBrowser';
import { useCheckpointActions } from '../../hooks/useCheckpointActions';
import {
  checkpointKey, toggleCheckpointSelection, selectedCheckpointRefs,
  describePreviewSelection, parseSeedInput,
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

export default function RunLineageGraph({ tree, onSelect, onContinueCheckpoint,
  continueSource = 'cloud', refetchTree, bestSettingsLora = null }) {
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
  // A preview thumbnail opened LARGE in a lightbox: { url, step } | null.
  const [bigPreview, setBigPreview] = useState(null);
  // 🖼 The gallery of ONE checkpoint: { recordId, step } | null. The compact pill
  // no longer carries a 14-px thumbnail (illegible at that size, and its badge
  // collided with the neighbouring pill's); it carries a results COUNT, and this
  // is where the count leads — the images at a size where they can be judged.
  const [gallery, setGallery] = useState(null);
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
  // ⓘ Details, from the shared popover: the same drawer, opened because it was
  // asked for. This mount ALSO keeps opening it on a card click — its cards
  // double as the Runs-hub jump, and taking that away would be a regression on a
  // surface nobody complained about. The canvas, whose cards do nothing else,
  // routes the click to the popover instead.
  const handleOpenDetails = useCallback((node) => setOpenNode(node), []);
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

  // 📦 Deploy and 🗑/⏏ delete are the SHARED checkpoint actions (they run the
  // same routes, with the same payloads and the same confirmation, on the LoRA
  // Canvas). All this mount adds is what "re-read the lineage" means here: a
  // refetch whose fresh preview state is merged into the overlay, so a
  // just-deployed pill flips to ✓ Deployed without a reload.
  const onCheckpointChanged = useCallback(async () => {
    if (typeof refetchTree !== 'function') return;
    const t = await refetchTree();
    if (t) mergeFromTree(t);
  }, [refetchTree, mergeFromTree]);
  const { importing, deleting, deployCheckpoint, deleteCheckpoint } = useCheckpointActions({
    onChanged: onCheckpointChanged, bestSettingsLora,
  });
  const handleImport = useCallback(async (node, pill) => {
    if (await deployCheckpoint(datasetId, node, pill)) setOpenCk(null);
  }, [deployCheckpoint, datasetId]);
  const handleDeleteCheckpoint = useCallback(async (node, pill) => {
    if (await deleteCheckpoint(datasetId, node, pill)) setOpenCk(null);
  }, [deleteCheckpoint, datasetId]);

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
        <span><span className="font-semibold">⇧ Shift-click</span> two runs to compare · tick the <span aria-hidden>☑</span> corner box on an <span className="font-semibold">imported</span> checkpoint to preview it (import one with <span aria-hidden>📦</span> first)</span>
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
                ? 'bg-indigo-500 text-gray-950 hover:bg-indigo-400 '
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
        <LineageEdgeDefs />
        <LineageEdges edges={g.edges} isLit={isLit} />

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
                    onOpenGallery={(pill) => setGallery({ recordId: n.node.record_id, step: pill.step })}
                    onZoomPreview={zoomPreview} />
                ))}
              </div>
            </foreignObject>
          ))}
        </g>

        {/* Actions popover — drawn last so it sits above every node. THE shared
            checkpoint popover (CheckpointActionsPopover), the very same component
            the LoRA Canvas floats over its board: one set of actions, one set of
            labels, one place where "which file does this delete" is decided.
            Only the PLACEMENT is this surface's own — inside the <svg>, in world
            units, flipped above the pill when there is no room below and clamped
            horizontally so the scroll panel never clips it. */}
        {openCk && (() => {
          const at = checkpointPopoverPlacement(openCk.pill, g);
          return (
          <foreignObject className="lds-gnode overflow-visible"
            x={at.x} y={at.y} width={POPOVER_W + 10} height={POPOVER_H + 8}>
            <div style={{ width: POPOVER_W }}>
              <CheckpointActionsPopover
                node={openCk.node} pill={openCk.pill}
                continueSource={continueSource}
                folderLabel={loraFolderLabel(openCk.node.train_type)}
                importing={importing} deleting={deleting}
                onContinue={typeof onContinueCheckpoint === 'function' ? onContinueCheckpoint : undefined}
                onDeploy={handleImport}
                onDelete={handleDeleteCheckpoint}
                onDetails={handleOpenDetails}
                onClose={closePopover} />
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
        in ComfyUI spirit (the pill thumbnails are only 14px). Shared with the
        canvas, where the same thumbnail used to be clickable and do nothing. */}
    <PreviewLightbox target={bigPreview} onClose={() => setBigPreview(null)} />
    {/* 🖼 Everything one checkpoint ever produced — the same panel the canvas
        opens, so the results of a generation are reachable from either surface. */}
    <CheckpointGalleryPanel target={gallery} onClose={() => setGallery(null)}
      onDeleted={() => { Promise.resolve(refetchTree?.()).catch(() => {}); }} />
    </>
  );
}
