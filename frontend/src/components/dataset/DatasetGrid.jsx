import { useEffect, useMemo, useState } from 'react';
import DatasetGridItem from './DatasetGridItem';

const DEFAULT_GREEN = 0.50;

/* Auto-triage (A2): pre-mark the UNDECIDED scorable images by face-score
   threshold — score >= t -> keep, below -> reject. Client-side derivation from
   the payload the grid already has; applies through the same batch endpoint as
   the manual multi-select. Manual keep/reject decisions are never touched. */
function AutoTriageBar({ images, faceThresholds, onBatch, busy }) {
  const [t, setT] = useState(() => faceThresholds?.green ?? DEFAULT_GREEN);
  const [lastRun, setLastRun] = useState(null);
  const candidates = useMemo(
    () => images.filter((i) => i.status === 'pending' && i.filename
      && i.face_state === 'scorable' && i.face_score != null),
    [images]);
  if (!candidates.length) return null;
  const keepIds = candidates.filter((i) => i.face_score >= t).map((i) => i.id);
  const rejectIds = candidates.filter((i) => i.face_score < t).map((i) => i.id);
  const apply = async () => {
    const kept = keepIds.length ? await onBatch(keepIds, 'keep', { silent: true }) : 0;
    const rejected = rejectIds.length ? await onBatch(rejectIds, 'reject', { silent: true }) : 0;
    setLastRun({ kept, rejected });
  };
  return (
    <div className="flex items-center gap-3 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
      <span className="text-content text-sm font-semibold shrink-0">🎯 Auto-triage</span>
      <label className="flex items-center gap-2 text-xs text-content-muted">
        keep&nbsp;≥
        <input type="range" min="0.30" max="0.70" step="0.01" value={t}
          onChange={(e) => { setT(parseFloat(e.target.value)); setLastRun(null); }}
          aria-label="Face-score threshold for auto-triage" className="w-36" />
        <span className="font-mono text-content w-10">{t.toFixed(2)}</span>
      </label>
      <span className="text-xs text-content-subtle">
        → would keep {keepIds.length} · reject {rejectIds.length} (of {candidates.length} undecided)
      </span>
      <button type="button" onClick={apply} disabled={busy || !candidates.length}
        title="Pre-marks only UNDECIDED analyzed images — your manual ✓/✕ choices are never changed"
        className="ml-auto px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
        Apply
      </button>
      {lastRun && (
        <span className="text-xs text-emerald-400">✓ {lastRun.kept} kept · {lastRun.rejected} rejected</span>
      )}
    </div>
  );
}

export default function DatasetGrid({ images, datasetId, onStatus, onCaption, onCrop, onDelete,
                                      onRegenerate, onView, onBatch, busy, nonces, faceThresholds }) {
  const [selected, setSelected] = useState(() => new Set());
  // Prune ids that vanished (deleted / poll refresh) so stale selections can't act.
  useEffect(() => {
    setSelected((prev) => {
      const alive = new Set(images.filter((i) => i.filename).map((i) => i.id));
      const next = new Set([...prev].filter((id) => alive.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [images]);

  if (!images || !images.length) {
    return <p className="text-content-subtle text-xs">No images — generate variations or import photos.</p>;
  }
  const selectable = images.filter((i) => i.filename);
  const ids = [...selected];
  const toggle = (id) => setSelected((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const act = async (action) => {
    if (action === 'delete'
        && !window.confirm(`Permanently delete the ${ids.length} selected image(s) (files included)?`)) return;
    await onBatch(ids, action);
    setSelected(new Set());
  };
  const batchBtn = 'px-2.5 py-1 rounded-lg text-xs font-semibold disabled:opacity-40';

  return (
    <div className="flex flex-col gap-2">
      {onBatch && (
        <AutoTriageBar images={images} faceThresholds={faceThresholds} onBatch={onBatch} busy={busy} />
      )}
      {onBatch && (
        <div className="flex items-center gap-2 flex-wrap text-xs">
          {selected.size === 0 ? (
            <>
              <span className="text-content-subtle">Tick images to curate them in bulk —</span>
              <button type="button" onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
                className="text-content-muted underline hover:text-content">select all ({selectable.length})</button>
            </>
          ) : (
            <div role="toolbar" aria-label="Bulk actions on the selection"
              className="flex items-center gap-2 flex-wrap rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-2.5 py-1.5 w-full">
              <span className="text-content font-semibold">{selected.size} selected</span>
              <button type="button" disabled={busy} onClick={() => act('keep')}
                className={`${batchBtn} bg-green-600/80 text-white`}>✓ Keep</button>
              <button type="button" disabled={busy} onClick={() => act('reject')}
                className={`${batchBtn} bg-red-600/80 text-white`}>✕ Reject</button>
              <button type="button" disabled={busy} onClick={() => act('pending')}
                title="Back to undecided" className={`${batchBtn} bg-surface text-content border border-border`}>↺ Undecide</button>
              <button type="button" disabled={busy} onClick={() => act('clear_caption')}
                title="Delete the selected images' captions (the Caption button then regenerates them)"
                className={`${batchBtn} bg-surface text-content border border-border`}>🧹 Clear captions</button>
              <button type="button" disabled={busy} onClick={() => act('delete')}
                className={`${batchBtn} bg-red-500/15 border border-red-500/40 text-red-300`}>🗑 Delete</button>
              <span className="ml-auto flex gap-2">
                <button type="button" onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
                  className="text-content-muted underline hover:text-content">all ({selectable.length})</button>
                <button type="button" onClick={() => setSelected(new Set())}
                  className="text-content-muted underline hover:text-content">none</button>
              </span>
            </div>
          )}
        </div>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {images.map((img) => (
          <DatasetGridItem key={img.id} img={img} datasetId={datasetId} onStatus={onStatus} onCaption={onCaption}
            onCrop={onCrop} onDelete={onDelete} onRegenerate={onRegenerate} onView={onView}
            selected={selected.has(img.id)} onToggleSelect={onBatch ? toggle : undefined}
            nonce={(nonces && nonces[img.id]) || 0} faceThresholds={faceThresholds} />
        ))}
      </div>
    </div>
  );
}
