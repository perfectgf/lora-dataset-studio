import { useEffect, useMemo, useState } from 'react';
import DatasetGridItem from './DatasetGridItem';

const DEFAULT_GREEN = 0.50;

// 3-4 line plain-language explanation for the 🎯 panel's "?" button.
const AUTO_TRIAGE_HELP = [
  'Marks the UNDECIDED, face-scored images: keep when the face similarity is ≥ the threshold, reject below it.',
  'It never deletes anything and never touches your manual ✓/✕ — those are left as-is and drop out of a Re-apply.',
  'Images with no score (face too small / no face detected) are skipped — judge those by eye.',
  'After an Apply, move the slider and Re-apply to re-sort everything it triaged this session at the new threshold.',
];

// Thumbnail size (S/M/L): 3 crans plutôt qu'un slider (fragile à la souris, pas
// de granularité utile ici). Persisté en préférence GLOBALE (pas par dataset —
// même pattern que `datasetGenerator`) : c'est un réglage d'affichage, pas une
// donnée du dataset. M = comportement historique (grid-cols-2/3/4) inchangé.
// L réduit les colonnes pour de vraies grandes tuiles (juger une composition
// verticale/horizontale avant crop) ; S en ajoute pour un survol dense.
const TILE_SIZE_KEY = 'datasetGridTileSize';
const TILE_SIZE_COLS = {
  S: 'grid-cols-2 sm:grid-cols-4 lg:grid-cols-6',
  M: 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-4',
  L: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
};
const TILE_SIZE_TITLE = {
  S: 'Small tiles — more per row, quick overview',
  M: 'Medium tiles (default)',
  L: 'Large tiles — see the full composition before you crop/keep/reject',
};

// Discreet segmented S/M/L control, not a slider (mouse-fragile, no useful
// granularity for 3 steps). Lives in the grid header, next to "select all".
function TileSizeControl({ size, onChange, className = '' }) {
  return (
    <div role="group" aria-label="Thumbnail size" className={`flex items-center gap-1 shrink-0 ${className}`}>
      <span aria-hidden className="text-content-subtle text-xs">🔳</span>
      {['S', 'M', 'L'].map((s) => (
        <button key={s} type="button" onClick={() => onChange(s)}
          aria-pressed={size === s} title={TILE_SIZE_TITLE[s]}
          aria-label={`${TILE_SIZE_TITLE[s]}${size === s ? ' (active)' : ''}`}
          className={`w-6 h-6 rounded-md border text-[0.6875rem] font-semibold transition-colors ${
            size === s
              ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
              : 'border-border bg-surface text-content-muted hover:bg-surface-raised'}`}>
          {s}
        </button>
      ))}
    </div>
  );
}

/* Auto-triage (A2): pre-mark scorable images by face-score threshold —
   score >= t -> keep, below -> reject. It marks the currently UNDECIDED scorable
   images AND re-owns the images IT decided earlier in this session, so the panel
   stays after an Apply and is replayable: move the slider, Re-apply, and the
   whole {previously auto-triaged} ∪ {new undecided} set is re-sorted at the new
   threshold. A manual ✓/✕ made after an auto-triage releases that image from the
   replay set — its status no longer matches what auto-triage assigned it — so
   manual decisions are never re-flipped. (Blind spot: manually re-affirming the
   SAME status auto-triage already set is indistinguishable and may be re-sorted.)
   Client-side derivation from the payload the grid already has; applies through
   the same batch endpoint as the manual multi-select, which already allows a
   direct keep<->reject switch (no backend change). */
function AutoTriageBar({ images, datasetId, faceThresholds, onBatch, busy }) {
  const [t, setT] = useState(() => faceThresholds?.green ?? DEFAULT_GREEN);
  // Session memory: image id -> the status auto-triage last assigned it
  // ('keep'|'reject'). An image whose CURRENT status still equals this value is
  // still "owned" by auto-triage and re-enters a Re-apply; if the user changed it
  // by hand since, its status diverges and it drops out (manual decision wins).
  const [owned, setOwned] = useState({});
  const [lastRun, setLastRun] = useState(null); // {kept, rejected, t} of the last Apply
  const [showHelp, setShowHelp] = useState(false);

  // A different dataset = a fresh session (the component isn't remounted on a
  // dataset switch — no key on <DatasetGrid> — so reset explicitly, like the
  // sibling per-dataset states elsewhere in the workspace).
  useEffect(() => {
    setOwned({});
    setLastRun(null);
    setShowHelp(false);
    setT(faceThresholds?.green ?? DEFAULT_GREEN);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  const isScorable = (i) => i.filename && i.face_state === 'scorable' && i.face_score != null;
  // Always-eligible: the undecided scorable images.
  const pending = useMemo(
    () => images.filter((i) => i.status === 'pending' && isScorable(i)), [images]);
  // Still owned by auto-triage: present, scorable, and status unchanged since we set it.
  const ownedImgs = useMemo(
    () => images.filter((i) => isScorable(i) && owned[i.id] != null && i.status === owned[i.id]),
    [images, owned]);
  // Replay scope = new undecided ∪ still-owned (disjoint: a 'pending' status can
  // never equal an owned 'keep'/'reject').
  const replay = useMemo(() => [...pending, ...ownedImgs], [pending, ownedImgs]);

  // Keep the panel while there is anything to triage OR anything it still owns.
  if (!replay.length) return null;

  const isReplay = lastRun != null; // at least one Apply already happened this session
  const keepTargets = replay.filter((i) => i.face_score >= t);
  const rejectTargets = replay.filter((i) => i.face_score < t);
  // Only flip the images that aren't already at their target status (no-op churn).
  const keepIds = keepTargets.filter((i) => i.status !== 'keep').map((i) => i.id);
  const rejectIds = rejectTargets.filter((i) => i.status !== 'reject').map((i) => i.id);
  const nothingToDo = !keepIds.length && !rejectIds.length;

  const apply = async () => {
    if (keepIds.length) await onBatch(keepIds, 'keep', { silent: true });
    if (rejectIds.length) await onBatch(rejectIds, 'reject', { silent: true });
    // Re-own the WHOLE replay scope at this threshold (incl. images left unchanged)
    // and forget any previously-owned image no longer in scope (manual override).
    const next = {};
    keepTargets.forEach((i) => { next[i.id] = 'keep'; });
    rejectTargets.forEach((i) => { next[i.id] = 'reject'; });
    setOwned(next);
    setLastRun({ kept: keepTargets.length, rejected: rejectTargets.length, t });
  };

  return (
    <div className="relative flex items-center gap-3 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
      <span className="text-content text-sm font-semibold shrink-0">🎯 Auto-triage</span>
      <button type="button" onClick={() => setShowHelp((v) => !v)}
        aria-expanded={showHelp} aria-label="What does auto-triage do?"
        title="What does auto-triage do?"
        className="shrink-0 w-5 h-5 -ml-1 rounded-full border border-border bg-surface-raised text-content-muted text-xs font-bold leading-none hover:text-content hover:bg-surface">
        ?
      </button>
      {showHelp && (
        <>
          {/* Transparent backdrop: an outside click dismisses the popover. */}
          <div className="fixed inset-0 z-40" onClick={() => setShowHelp(false)} aria-hidden />
          <div role="tooltip"
            className="absolute z-50 top-full left-2 mt-1 w-80 max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-surface p-3 shadow-xl flex flex-col gap-1.5">
            {AUTO_TRIAGE_HELP.map((line) => (
              <p key={line} className="text-[11px] leading-snug text-content-muted">{line}</p>
            ))}
          </div>
        </>
      )}
      <label className="flex items-center gap-2 text-xs text-content-muted">
        keep&nbsp;≥
        <input type="range" min="0.30" max="0.70" step="0.01" value={t}
          onChange={(e) => setT(parseFloat(e.target.value))}
          aria-label="Face-score threshold for auto-triage" className="w-36" />
        <span className="font-mono text-content w-10">{t.toFixed(2)}</span>
      </label>
      <span className="text-xs text-content-subtle">
        → keep {keepTargets.length} · reject {rejectTargets.length}
        {isReplay
          ? ` (re-sort ${ownedImgs.length}${pending.length ? ` + ${pending.length} new` : ''})`
          : ` (of ${replay.length} undecided)`}
      </span>
      <button type="button" onClick={apply} disabled={busy || nothingToDo}
        title="Marks only scored images — your manual ✓/✕ choices are never changed"
        className="ml-auto px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
        {isReplay ? 'Re-apply' : 'Apply'}
      </button>
      {lastRun && (
        <span className="text-xs text-emerald-400">
          ✓ applied: kept {lastRun.kept} · rejected {lastRun.rejected} at ≥ {lastRun.t.toFixed(2)}
        </span>
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
  // Thumbnail size: a UI preference, not dataset data — persisted globally
  // (same lazy-init + effect pattern as `datasetGenerator` in VariationCatalog).
  // Runs before the early return below so hook order stays stable.
  const [tileSize, setTileSize] = useState(() => {
    try {
      const v = localStorage.getItem(TILE_SIZE_KEY);
      return v === 'S' || v === 'M' || v === 'L' ? v : 'M';
    } catch { return 'M'; }
  });
  useEffect(() => {
    try { localStorage.setItem(TILE_SIZE_KEY, tileSize); } catch { /* ignore — private mode */ }
  }, [tileSize]);

  if (!images || !images.length) {
    return (
      <p id="ds-images-review" tabIndex={-1} data-workspace-focus
        className="text-content-subtle text-xs scroll-mt-20">
        No images — generate variations or import photos.
      </p>
    );
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
    <div id="ds-images-review" tabIndex={-1} data-workspace-focus
      className="flex flex-col gap-2 scroll-mt-20">
      {onBatch && (
        <AutoTriageBar images={images} datasetId={datasetId} faceThresholds={faceThresholds} onBatch={onBatch} busy={busy} />
      )}
      <div id="ds-images-bulk" tabIndex={-1}
        className="flex items-center gap-2 flex-wrap text-xs scroll-mt-20">
        {onBatch && (
          selected.size === 0 ? (
            <>
              <span className="text-content-subtle">Tick images to curate them in bulk —</span>
              <button type="button" data-workspace-focus
                onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
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
          )
        )}
        <TileSizeControl size={tileSize} onChange={setTileSize} className="ml-auto" />
      </div>
      <div className={`grid ${TILE_SIZE_COLS[tileSize]} gap-2`}>
        {images.map((img) => (
          <DatasetGridItem key={img.id} img={img} datasetId={datasetId} onStatus={onStatus} onCaption={onCaption}
            onCrop={onCrop} onDelete={onDelete} onRegenerate={onRegenerate} onView={onView}
            selected={selected.has(img.id)} onToggleSelect={onBatch ? toggle : undefined}
            nonce={(nonces && nonces[img.id]) || 0} faceThresholds={faceThresholds} tileSize={tileSize} />
        ))}
      </div>
    </div>
  );
}
