import { useEffect, useMemo, useRef, useState } from 'react';
import DatasetGridItem from './DatasetGridItem';
import TileSizeControl from '../shared/TileSizeControl';
import { isSmallImageRescueRow } from '../../utils/smallImageRescue';
import {
  partitionKleinImproveSelection,
  runSequentialKleinImprove,
} from '../../utils/kleinBulkImprove';
import { useToast } from '../common/Toast';
import { useI18n } from '../../i18n/I18nContext';

const DEFAULT_GREEN = 0.50;

// 3-4 line plain-language explanation for the 🎯 panel's "?" button.
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
  const { t: translate } = useI18n();
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
      <span className="text-content text-sm font-semibold shrink-0">
        🎯 {translate('workspace.imageBulk.autoTriage.title')}
      </span>
      <button type="button" onClick={() => setShowHelp((v) => !v)}
        aria-expanded={showHelp} aria-label={translate('workspace.imageBulk.autoTriage.what')}
        title={translate('workspace.imageBulk.autoTriage.what')}
        className="shrink-0 w-5 h-5 -ml-1 rounded-full border border-border bg-surface-raised text-content-muted text-xs font-bold leading-none hover:text-content hover:bg-surface">
        ?
      </button>
      {showHelp && (
        <>
          {/* Transparent backdrop: an outside click dismisses the popover. */}
          <div className="fixed inset-0 z-40" onClick={() => setShowHelp(false)} aria-hidden />
          <div role="tooltip"
            className="absolute z-50 top-full left-2 mt-1 w-80 max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-surface p-3 shadow-xl flex flex-col gap-1.5">
            {[1, 2, 3, 4].map((line) => (
              <p key={line} className="text-[11px] leading-snug text-content-muted">
                {translate(`workspace.imageBulk.autoTriage.help.${line}`)}
              </p>
            ))}
          </div>
        </>
      )}
      <label className="flex items-center gap-2 text-xs text-content-muted">
        {translate('workspace.imageBulk.autoTriage.keepAtLeast')}&nbsp;≥
        <input type="range" min="0.30" max="0.70" step="0.01" value={t}
          onChange={(e) => setT(parseFloat(e.target.value))}
          aria-label={translate('workspace.imageBulk.autoTriage.threshold')} className="w-36" />
        <span className="font-mono text-content w-10">{t.toFixed(2)}</span>
      </label>
      <span className="text-xs text-content-subtle">
        → {translate('workspace.imageBulk.autoTriage.outcome', {
          kept: keepTargets.length,
          rejected: rejectTargets.length,
        })}
        {isReplay
          ? ` (${translate('workspace.imageBulk.autoTriage.resort', {
              count: ownedImgs.length,
              newCount: pending.length,
            })})`
          : ` (${translate('workspace.imageBulk.autoTriage.undecided', { count: replay.length })})`}
      </span>
      <button type="button" onClick={apply} disabled={busy || nothingToDo}
        title={translate('workspace.imageBulk.autoTriage.applyTitle')}
        className="ml-auto px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
        {translate(isReplay
          ? 'workspace.imageBulk.autoTriage.reapply'
          : 'workspace.imageBulk.autoTriage.apply')}
      </button>
      {lastRun && (
        <span className="text-xs text-emerald-400">
          ✓ {translate('workspace.imageBulk.autoTriage.applied', {
            kept: lastRun.kept,
            rejected: lastRun.rejected,
            threshold: lastRun.t.toFixed(2),
          })}
        </span>
      )}
    </div>
  );
}

export default function DatasetGrid({ images, datasetId, onStatus, onCaption, onCrop, onDelete,
                                      onMirror, onRegenerate, onView, onBatch, busy, nonces,
                                      mirroringIds, faceThresholds, datasetKind = 'character',
                                      onImprove, onRefresh, kleinAvailable = false,
                                      eligibilityImages, dualCaptions = false }) {
  const toast = useToast();
  const { t } = useI18n();
  const [selected, setSelected] = useState(() => new Set());
  const [bulkImprove, setBulkImprove] = useState(null); // {running, done, total}
  const datasetIdRef = useRef(datasetId);
  const bulkImproveRunRef = useRef(0);
  // Update during render (not one effect later) so a completion microtask for
  // dataset A immediately sees navigation to B. The token also protects A→B→A.
  datasetIdRef.current = datasetId;
  const bulkBusy = busy || Boolean(bulkImprove?.running);
  useEffect(() => {
    bulkImproveRunRef.current += 1;
    setSelected(new Set());
    setBulkImprove(null);
  }, [datasetId]);
  // Prune ids that vanished (deleted / poll refresh) so stale selections can't act.
  useEffect(() => {
    setSelected((prev) => {
      const alive = new Set(images
        .filter((i) => i.filename && !isSmallImageRescueRow(i))
        .map((i) => i.id));
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
        {t('workspace.images.empty')}
      </p>
    );
  }
  // Rescue winners remain editable one-by-one (caption/crop), but their paired
  // provenance makes generic bulk status/delete unsafe. Never select them here.
  const selectable = images.filter((i) => i.filename && !isSmallImageRescueRow(i));
  const ids = [...selected];
  const improveUniverse = Array.isArray(eligibilityImages) ? eligibilityImages : images;
  const improveSelection = partitionKleinImproveSelection(improveUniverse, ids);
  const exclusionReason = (reason) => {
    const key = {
      'image no longer exists': 'missing',
      'image file is not ready': 'fileNotReady',
      'resolve the Klein rescue pair first': 'rescuePair',
      'already an improvement candidate': 'alreadyCandidate',
      'an improvement is already pending review': 'pendingReview',
    }[reason];
    return key ? t(`workspace.imageBulk.reasons.${key}`) : reason;
  };
  const exclusionSummary = [...improveSelection.excluded.reduce((counts, item) => {
    counts.set(item.reason, (counts.get(item.reason) || 0) + 1);
    return counts;
  }, new Map())].map(([reason, count]) => `${count} ${exclusionReason(reason)}`).join(' · ');
  const toggle = (id) => setSelected((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const act = async (action) => {
    if (bulkBusy) return;
    if (action === 'delete'
        && !window.confirm(t('workspace.imageBulk.deleteConfirm', { count: ids.length }))) return;
    await onBatch(ids, action);
    setSelected(new Set());
  };
  const improveSelected = async () => {
    const { eligible, excluded } = partitionKleinImproveSelection(improveUniverse, ids);
    if (!onImprove || !kleinAvailable || !eligible.length || bulkBusy) return;
    const skipped = excluded.length
      ? `\n\n${t('workspace.imageBulk.improveSkipped', {
          count: excluded.length,
          reasons: exclusionSummary,
        })}`
      : '';
    if (!window.confirm(
      t('workspace.imageBulk.improveConfirm', { count: eligible.length })
      + `${skipped}\n\n${t('workspace.imageBulk.originalsUnchanged')}`,
    )) return;
    const batchDatasetId = datasetId;
    const runToken = ++bulkImproveRunRef.current;
    const isCurrentBatch = () => (
      datasetIdRef.current === batchDatasetId && bulkImproveRunRef.current === runToken
    );
    setBulkImprove({ running: true, done: 0, total: eligible.length });
    let result = { succeeded: [], failed: [] };
    let unexpectedError = null;
    let refreshFailed = false;
    try {
      result = await runSequentialKleinImprove(
        eligible,
        (imageId) => onImprove(imageId, { silent: true, refreshAfter: false }),
        ({ done, total }) => {
          if (isCurrentBatch()) setBulkImprove({ running: true, done, total });
        },
      );
    } catch (error) {
      unexpectedError = error;
    } finally {
      // The requests may finish after navigation. Do not fetch A or mutate B's
      // selection/progress/toasts from this stale batch.
      if (!isCurrentBatch()) return;
      try { await onRefresh?.(batchDatasetId); } catch { refreshFailed = true; }
      if (!isCurrentBatch()) return;
      setSelected(new Set());
      setBulkImprove({ running: false, done: eligible.length, total: eligible.length });
      if (unexpectedError) {
        toast.error(t('workspace.imageBulk.toast.stopped', {
          error: unexpectedError.message || t('workspace.imageBulk.toast.unknownError'),
        }));
      } else if (result.failed.length || refreshFailed) {
        toast.warning(t('workspace.imageBulk.toast.partial', {
          succeeded: result.succeeded.length,
          total: eligible.length,
          failed: result.failed.length,
          refresh: refreshFailed ? t('workspace.imageBulk.toast.refreshFailed') : '',
        }));
      } else {
        toast.success(t('workspace.imageBulk.toast.queued', { count: result.succeeded.length }));
      }
    }
  };
  const batchBtn = 'px-2.5 py-1 rounded-lg text-xs font-semibold disabled:opacity-40';

  return (
    <div id="ds-images-review" tabIndex={-1} data-workspace-focus
      className="flex flex-col gap-2 scroll-mt-20">
      {onBatch && (
        <AutoTriageBar images={images.filter((image) => !isSmallImageRescueRow(image))}
          datasetId={datasetId} faceThresholds={faceThresholds} onBatch={onBatch} busy={bulkBusy} />
      )}
      <div id="ds-images-bulk" tabIndex={-1}
        className="flex items-center gap-2 flex-wrap text-xs scroll-mt-20">
        {onBatch && (
          selected.size === 0 ? (
            <>
              <span className="text-content-subtle">{t('workspace.imageBulk.hint')}</span>
              <button type="button" data-workspace-focus
                disabled={bulkBusy}
                onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
                className="text-content-muted underline hover:text-content disabled:opacity-40">
                {t('workspace.imageBulk.selectAll', { count: selectable.length })}
              </button>
            </>
          ) : (
            <div role="toolbar" aria-label={t('workspace.imageBulk.toolbar')}
              className="flex items-center gap-2 flex-wrap rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-2.5 py-1.5 w-full">
              <span className="text-content font-semibold">
                {t('workspace.imageBulk.selected', { count: selected.size })}
              </span>
              <button type="button" disabled={bulkBusy} onClick={() => act('keep')}
                className={`${batchBtn} bg-green-600/80 text-white`}>✓ {t('workspace.imageBulk.keep')}</button>
              <button type="button" disabled={bulkBusy} onClick={() => act('reject')}
                className={`${batchBtn} bg-red-600/80 text-white`}>✕ {t('workspace.imageBulk.reject')}</button>
              <button type="button" disabled={bulkBusy} onClick={() => act('pending')}
                title={t('workspace.imageBulk.undecideTitle')}
                className={`${batchBtn} bg-surface text-content border border-border`}>
                ↺ {t('workspace.imageBulk.undecide')}
              </button>
              <button type="button" disabled={bulkBusy} onClick={() => act('clear_caption')}
                title={t('workspace.imageBulk.clearCaptionsTitle')}
                className={`${batchBtn} bg-surface text-content border border-border`}>
                🧹 {t('workspace.imageBulk.clearCaptions')}
              </button>
              {onImprove && (
                <button type="button" onClick={improveSelected}
                  disabled={bulkBusy || !kleinAvailable || !improveSelection.eligible.length}
                  title={!kleinAvailable
                    ? t('workspace.imageBulk.kleinUnavailable')
                    : improveSelection.eligible.length
                      ? t('workspace.imageBulk.improveTitle', { exclusions: exclusionSummary })
                      : t('workspace.imageBulk.noneEligible', { exclusions: exclusionSummary })}
                  className={`${batchBtn} border border-indigo-400/50 bg-indigo-500/20 text-indigo-100`}>
                  {bulkImprove?.running
                    ? `✨ ${t('workspace.imageBulk.improving', {
                        done: bulkImprove.done,
                        total: bulkImprove.total,
                      })}`
                    : `✨ ${t('workspace.imageBulk.improve', {
                        count: improveSelection.eligible.length,
                      })}`}
                </button>
              )}
              {onImprove && improveSelection.excluded.length > 0 && (
                <span className="text-content-subtle" title={exclusionSummary}>
                  {t('workspace.imageBulk.notEligible', { count: improveSelection.excluded.length })}
                </span>
              )}
              <button type="button" disabled={bulkBusy} onClick={() => act('delete')}
                className={`${batchBtn} bg-red-500/15 border border-red-500/40 text-red-300`}>
                🗑 {t('workspace.imageBulk.delete')}
              </button>
              <span className="ml-auto flex gap-2">
                <button type="button" disabled={bulkBusy}
                  onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
                  className="text-content-muted underline hover:text-content disabled:opacity-40">
                  {t('workspace.imageBulk.all', { count: selectable.length })}
                </button>
                <button type="button" disabled={bulkBusy} onClick={() => setSelected(new Set())}
                  className="text-content-muted underline hover:text-content disabled:opacity-40">
                  {t('workspace.imageBulk.none')}
                </button>
              </span>
            </div>
          )
        )}
        <TileSizeControl size={tileSize} onChange={setTileSize} titles={{
          S: t('workspace.imageBulk.tileSize.small'),
          M: t('workspace.imageBulk.tileSize.medium'),
          L: t('workspace.imageBulk.tileSize.large'),
        }} className="ml-auto" />
      </div>
      <div className={`grid ${TILE_SIZE_COLS[tileSize]} gap-2`}>
        {images.map((img) => (
          <DatasetGridItem key={img.id} img={img} datasetId={datasetId} onStatus={onStatus} onCaption={onCaption}
            onCrop={onCrop} onDelete={onDelete} onMirror={onMirror}
            mirrorBusy={Boolean(mirroringIds?.has(img.id))} busy={bulkBusy}
            onRegenerate={onRegenerate} onView={onView}
            selected={selected.has(img.id)}
            onToggleSelect={onBatch && !isSmallImageRescueRow(img) ? toggle : undefined}
            nonce={(nonces && nonces[img.id]) || 0} faceThresholds={faceThresholds}
            tileSize={tileSize} datasetKind={datasetKind} dualCaptions={dualCaptions} />
        ))}
      </div>
    </div>
  );
}
