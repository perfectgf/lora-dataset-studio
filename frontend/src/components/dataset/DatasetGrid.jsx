import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import DatasetGridItem from './DatasetGridItem';
import TileSizeControl from '../shared/TileSizeControl';
import KleinImproveNote from './KleinImproveNote';
import { isSmallImageRescueRow } from '../../utils/smallImageRescue';
import { partitionKleinImproveSelection } from '../../utils/kleinBulkImprove';
import { improvementStateByParent } from './improveCandidates.js';
import { GRID_PAGE_SIZE, clampPage, pageOfIndex, pageSlice } from './gridPaging.js';
import {
  availableImproveEngines,
  describeImproveLaunch,
  improveBatchLabel,
  improveConfirmMessage,
  improveEngineBlockedReason,
} from '../../utils/improveEngines';
import { useCapabilities } from '../../context/CapabilitiesContext';
import { useToast } from '../common/Toast';
import { autoTriageAvailable, autoTriageEmptyReason } from './faceScoringGate.js';
import { bulkActionMessage, createBulkActionGate } from './bulkActionGate.js';
import { READS_STAY_OPEN, datasetBusyReason } from './datasetBusyReason.js';
import {
  autoTriageFailureMessage,
  autoTriageOwnershipForResult,
  createAutoTriageRunGate,
  runAutoTriageBatches,
  updateAutoTriageRuns,
} from './autoTriageApply.js';

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
function AutoTriageBar({ images, allImages, datasetId, faceThresholds, onBatch, busy,
                         applying, onApplyingChange }) {
  const autoTriageRunGateRef = useRef(null);
  if (!autoTriageRunGateRef.current) {
    autoTriageRunGateRef.current = createAutoTriageRunGate(datasetId);
  }
  // Invalidate the previous dataset synchronously during render. A passive
  // effect is too late: its old promise may settle between commit and effect.
  autoTriageRunGateRef.current.syncDataset(datasetId);
  const [t, setT] = useState(() => faceThresholds?.green ?? DEFAULT_GREEN);
  // Session memory: image id -> the status auto-triage last assigned it
  // ('keep'|'reject'). An image whose CURRENT status still equals this value is
  // still "owned" by auto-triage and re-enters a Re-apply; if the user changed it
  // by hand since, its status diverges and it drops out (manual decision wins).
  const [owned, setOwned] = useState({});
  const [lastRun, setLastRun] = useState(null); // {kept, rejected, t} of the last Apply
  const [applyFailure, setApplyFailure] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  // A different dataset = a fresh session (the component isn't remounted on a
  // dataset switch — no key on <DatasetGrid> — so reset explicitly, like the
  // sibling per-dataset states elsewhere in the workspace).
  useEffect(() => {
    setOwned({});
    setLastRun(null);
    setApplyFailure(null);
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
  // With nothing to do it STAYS and says why: three unrelated situations used to
  // render as the same silent absence (see autoTriageEmptyReason).
  if (!replay.length) {
    const empty = autoTriageEmptyReason(images, allImages);
    if (!empty) return null;
    return (
      <div className="flex items-center gap-2 flex-wrap rounded-lg border border-dashed border-border bg-surface px-3 py-2">
        <span className="text-content-muted text-sm font-semibold shrink-0">🎯 Auto-triage</span>
        <span role="status" className="text-xs text-content-subtle">{empty.message}</span>
      </div>
    );
  }

  const isReplay = lastRun != null; // at least one Apply already happened this session
  const keepTargets = replay.filter((i) => i.face_score >= t);
  const rejectTargets = replay.filter((i) => i.face_score < t);
  // Only flip the images that aren't already at their target status (no-op churn).
  const keepIds = keepTargets.filter((i) => i.status !== 'keep').map((i) => i.id);
  const rejectIds = rejectTargets.filter((i) => i.status !== 'reject').map((i) => i.id);
  const nothingToDo = !keepIds.length && !rejectIds.length;

  const apply = async () => {
    const runGate = autoTriageRunGateRef.current;
    if (busy || runGate.hasActive(datasetId)) return;
    const token = runGate.begin(datasetId);
    if (!token) return;
    onApplyingChange(datasetId, token, true);
    setApplyFailure(null);
    try {
      const result = await runAutoTriageBatches({
        onBatch,
        keepIds,
        rejectIds,
        shouldContinue: () => runGate.isCurrent(token),
      });
      if (!runGate.isCurrent(token)) return;
      const next = autoTriageOwnershipForResult({
        result, keepTargets, rejectTargets, previousOwnership: owned,
      });
      if (!result.ok) {
        runGate.commit(token, () => {
          // A successful first POST is already durable even if the second one
          // failed. Keep those rows in the replay set so Retry/Re-apply can
          // still re-sort them after refresh changed pending -> keep/reject.
          setOwned(next);
          setApplyFailure(result);
        });
        return;
      }
      // Re-own the WHOLE replay scope at this threshold (incl. images left unchanged)
      // and forget any previously-owned image no longer in scope (manual override).
      runGate.commit(token, () => {
        setOwned(next);
        setLastRun({ kept: keepTargets.length, rejected: rejectTargets.length, t });
      });
    } finally {
      // `finish` settles this token's dataset even after navigation, but invokes
      // its callback only while the token still belongs to the visible session.
      // The parent's token-aware Map means settling A can never unlock B.
      const settled = runGate.finish(token, () => {});
      if (settled) onApplyingChange(token.datasetId, token, false);
    }
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
            className="absolute z-50 top-full left-2 mt-1 w-80 max-w-[calc(100vw-2rem)] rounded-lg border border-border bg-surface-overlay p-3 shadow-xl flex flex-col gap-1.5">
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
      <button type="button" onClick={apply} disabled={busy || applying || nothingToDo}
        title="Marks only scored images — your manual ✓/✕ choices are never changed"
        className="ml-auto px-3 py-1 rounded-lg bg-surface-raised border border-border text-content text-xs font-semibold disabled:opacity-40 hover:bg-surface">
        {applying ? 'Applying…' : isReplay ? 'Re-apply' : 'Apply'}
      </button>
      <span role="status" aria-live="polite" aria-atomic="true"
        className="text-xs text-emerald-400">
        {applying
          ? 'Applying auto-triage…'
          : lastRun && !applyFailure
            ? `✓ applied: kept ${lastRun.kept} · rejected ${lastRun.rejected} at ≥ ${lastRun.t.toFixed(2)}`
            : ''}
      </span>
      {applyFailure && (
        <span role="alert" className="text-xs text-rose-300">
          {autoTriageFailureMessage(applyFailure)}
        </span>
      )}
    </div>
  );
}

/* Page navigator for the image wall — deliberately the same shape and wording as
   the Bank grid's (← Prev · "1–500 of 6211" · Next →), because it is the same
   gesture on the same kind of surface. Rendered ABOVE and BELOW the grid: a full
   page is tens of thousands of pixels tall, and a bottom-only pager would mean
   scrolling all the way back up to move on. Buttons are min-h-11 (44 px) so the
   whole thing is usable with a thumb at 400 px, where the row wraps. */
function GridPager({ view, onGo, where }) {
  if (!view.paged) return null;
  const btn = 'min-h-11 rounded-lg border border-border bg-surface px-3 py-1 '
    + 'text-content font-semibold disabled:opacity-40 hover:bg-surface-raised';
  return (
    <nav aria-label={`Image grid pages (${where})`}
      className="flex flex-wrap items-center gap-2 text-xs">
      <button type="button" disabled={view.page === 0} onClick={() => onGo(view.page - 1)}
        className={btn}>← Prev</button>
      <button type="button" disabled={view.page >= view.pages - 1} onClick={() => onGo(view.page + 1)}
        className={btn}>Next →</button>
      <span className="text-content-muted tabular-nums">
        {view.from}–{view.to} of {view.total}
      </span>
      <span className="text-content-subtle tabular-nums">
        page {view.page + 1}/{view.pages}
      </span>
    </nav>
  );
}

export default function DatasetGrid({ images, datasetId, onStatus, onCaption, onCrop, onDelete,
                                      onMirror, onRegenerate, onScoreFace, scoringFaceIds, onReimprove, onView, onBatch, busy,
                                      /* Queue-lane gates (GitHub #44). `busy` is every pass and still guards
                                         every WRITE; these two are only what refuses a NEW queued job, and there
                                         are TWO of them because the launches they gate are different work:
                                         `improveBusy` for the ✨ batch and the per-tile re-improve (the backend
                                         answers 409 to a second improve batch), `generateBusy` for 🔄 / ✏️ retry,
                                         which enqueue a 'generate' the backend accepts freely. Merging them is
                                         exactly the bug this pair exists to prevent: the tiles then read the
                                         improve gate and stay grey for the whole improve batch — #44's own
                                         symptom, on the surface its release note promised. Both default to
                                         `busy`, so a caller that passes neither keeps the old blanket. */
                                      improveBusy = undefined, generateBusy = undefined,
                                      /* And the third question, for the writes that CURATE an image
                                         (keep/reject, caption, crop, mirror, rotate, delete, score,
                                         watermark): is a pass running that owns the rows? Queued work
                                         does not, and every one of those writes is defended server-side
                                         already. Defaults to `busy` like its two siblings. */
                                      curationBusy = undefined, nonces,
                                      mirroringIds, faceThresholds, datasetKind = 'character',
                                      onImproveBatch, kleinAvailable = false,
                                      eligibilityImages, dualCaptions = false,
                                      subjectType = '',
                                      // Server's reason why face scoring can't run here
                                      // (string) or null — auto-triage acts on face
                                      // scores, so it stands down when they can't be
                                      // trusted. Existing scores are NOT deleted.
                                      faceScoringBlocked = null,
                                      activity = null,
                                      // Which image the lightbox is currently on
                                      // (null when it is closed).
                                      viewingImageId = null,
                                      onBulkBusyChange }) {
  const toast = useToast();
  const { caps } = useCapabilities();
  const [selected, setSelected] = useState(() => new Set());
  const bulkActionGateRef = useRef(null);
  if (!bulkActionGateRef.current) bulkActionGateRef.current = createBulkActionGate();
  const [bulkAction, setBulkAction] = useState(null);
  // Track Auto-triage by dataset AND opaque run token. Navigation can leave a
  // request settling in A while B is open; only A stays locked, and a late A
  // finish cannot delete a newer token for either dataset.
  const [autoTriageRuns, setAutoTriageRuns] = useState(() => new Map());
  const setAutoTriageRun = useCallback((runDatasetId, token, active) => {
    setAutoTriageRuns((previous) => (
      updateAutoTriageRuns(previous, runDatasetId, token, active)
    ));
  }, []);
  // Only the LAUNCH request is tracked locally; the batch's own progress comes
  // from the server (`activity`), so it survives a reload and a closed tab.
  const [launchingImprove, setLaunchingImprove] = useState(false);
  const improveLabel = improveBatchLabel(activity);
  // Klein is always offered (its button carries the reason when it can't run);
  // SeedVR2 joins the toolbar only once it is actually installed — until then it
  // is a Setup task, not a choice.
  const improveEngines = useMemo(() => availableImproveEngines(caps), [caps]);
  // Which SOURCE tiles have an upscale waiting for review (or still rendering).
  // Computed ONCE for the whole grid: per tile it would be a scan of every
  // sibling on every render, and a 400-image dataset renders a lot.
  const improvementStates = useMemo(
    () => improvementStateByParent(images), [images]);
  const autoTriageApplying = autoTriageRuns.has(datasetId);
  const bulkBusy = busy || launchingImprove || !!bulkAction || autoTriageApplying;
  // Same three local locks, but over the queue-lane gate: launching an improve
  // or a retry while a generation batch is still running is exactly what #44
  // asked for, and the backend queue serializes them anyway.
  const localLaunchLock = launchingImprove || !!bulkAction || autoTriageApplying;
  const improveLaunchBusy = (improveBusy ?? busy) || localLaunchLock;
  const curationWriteBusy = (curationBusy ?? busy) || localLaunchLock;
  const generateLaunchBusy = (generateBusy ?? busy) || localLaunchLock;
  /* WHAT BLOCKS WHAT — three answers, not one.
     `bulkBusy` blocks WRITES: a running pass owns the pixels, the statuses and
     the files, and a second writer would race it. That has never been in doubt.
     `selectionLocked` blocks the TICK BOXES, and only for the actions that are
     spending the selection right now (a bulk keep/reject/delete, an improve
     launch, an auto-triage apply) — each of those snapshots the ids at click
     time and CLEARS the selection when it returns, so a selection made while
     one is in flight would be silently thrown away. A dataset pass
     (`busy`: generation, captioning, watermark scan…) is deliberately NOT in
     that list: it was handed its own list of images when it started and cannot
     be shifted by anything ticked afterwards.
     READS are in neither list — inspecting an image writes nothing. */
  const selectionLocked = launchingImprove || !!bulkAction || autoTriageApplying;
  // The sentence a refused write shows, instead of going quietly grey. `activity`
  // is the server's snapshot (name, progress, time left); a purely local lock
  // has none and still gets an honest generic line.
  const busyReason = bulkBusy ? datasetBusyReason(busy ? activity : null) : null;
  useEffect(() => {
    onBulkBusyChange?.(bulkBusy);
    return () => onBulkBusyChange?.(false);
  }, [bulkBusy, onBulkBusyChange]);
  // Which page of the filtered list is on screen. Only the RENDERING is paged:
  // the selection, the bulk actions, auto-triage and every count keep reading
  // the full list (see gridPaging.js).
  const [page, setPage] = useState(0);
  useEffect(() => {
    setSelected(new Set());
    setLaunchingImprove(false);
    setPage(0);
  }, [datasetId]);
  // A filter, a delete or a status flip can shrink the list under the page the
  // user is standing on; land them on the last real page instead of a blank
  // grid that would read as data loss.
  useEffect(() => {
    setPage((p) => clampPage(p, (images || []).length));
  }, [images]);
  // The lightbox's ⟨ / ⟩ walk the WHOLE filtered list (see lightboxNavigation.js),
  // so they can carry the user off this page. Follow them, silently and without
  // scrolling — the grid is behind an overlay; this is only so that closing it
  // reveals the page the inspected image is actually on. No-op when the id is
  // already on this page, which is every step inside a page.
  useEffect(() => {
    if (viewingImageId == null) return;
    const index = (images || []).findIndex((i) => i && i.id === viewingImageId);
    if (index < 0) return;
    setPage((p) => (pageOfIndex(index) === p ? p : pageOfIndex(index)));
  }, [viewingImageId, images]);
  // Prune ids that vanished (deleted / poll refresh) so stale selections can't act.
  useEffect(() => {
    setSelected((prev) => {
      // The delete endpoint is synchronous, but refreshes the dataset before
      // its promise returns. Keep the exact selection visible until that return;
      // the action's own success path clears it immediately afterwards.
      if (bulkActionGateRef.current.active?.action === 'delete') return prev;
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
        No images — generate variations or import photos.
      </p>
    );
  }
  // Rescue winners remain editable one-by-one (caption/crop), but their paired
  // provenance makes generic bulk status/delete unsafe. Never select them here.
  const selectable = images.filter((i) => i.filename && !isSmallImageRescueRow(i));
  // What is actually mounted. Everything below still reasons about `images`.
  const view = pageSlice(images, page, GRID_PAGE_SIZE);
  const goToPage = (next) => {
    setPage(clampPage(next, images.length));
    // Land at the top of the grid: the tiles under the cursor are now different
    // images, and finishing a page leaves you tens of thousands of pixels down
    // a page that no longer exists.
    try {
      document.getElementById('ds-images-review')?.scrollIntoView({ block: 'start' });
    } catch { /* no DOM (tests) — the page still changed, which is what matters */ }
  };
  const ids = [...selected];
  const improveUniverse = Array.isArray(eligibilityImages) ? eligibilityImages : images;
  // The dataset BEFORE the grid filters — `images` is what the filters left. Auto
  // triage needs both to tell "nothing to do" from "your filter is hiding it".
  const unfilteredImages = Array.isArray(eligibilityImages) ? eligibilityImages : images;
  const improveSelection = partitionKleinImproveSelection(improveUniverse, ids);
  const exclusionSummary = [...improveSelection.excluded.reduce((counts, item) => {
    counts.set(item.reason, (counts.get(item.reason) || 0) + 1);
    return counts;
  }, new Map())].map(([reason, count]) => `${count} ${reason}`).join(' · ');
  const toggle = (id) => {
    if (selectionLocked) return;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };
  const act = async (action) => {
    if (bulkBusy || bulkActionGateRef.current.active || !ids.length) return;
    if (action === 'delete'
        && !window.confirm(`Permanently delete the ${ids.length} selected image(s) (files included)?`)) return;
    const actionIds = [...ids];
    const token = bulkActionGateRef.current.begin(action, actionIds.length);
    if (!token) return;
    setBulkAction(token);
    try {
      const affected = await onBatch(actionIds, action);
      // A resolved numeric result is the hook's success contract. On rejection
      // or an explicit failure, keep the selection ready for a retry.
      if (typeof affected === 'number') setSelected(new Set());
    } finally {
      bulkActionGateRef.current.finish(token);
      setBulkAction(null);
    }
  };
  // Hand the whole selection to the server in ONE call. The client keeps the
  // eligibility partition (it already holds the rows, and the confirm must state
  // what will be skipped); the server re-checks it and owns the pacing.
  const improveSelected = async (engineId) => {
    const { eligible, excluded } = partitionKleinImproveSelection(improveUniverse, ids);
    if (!onImproveBatch || !eligible.length || improveLaunchBusy) return;
    if (improveEngineBlockedReason(engineId, {
      caps, engines: caps?.engines, eligibleCount: eligible.length,
    })) return;
    if (!window.confirm(improveConfirmMessage(engineId, {
      eligibleCount: eligible.length,
      excludedCount: excluded.length,
      exclusionSummary,
    }))) return;
    setLaunchingImprove(true);
    try {
      const result = await onImproveBatch(eligible.map((image) => image.id), engineId);
      if (result?.ok) {
        setSelected(new Set());
        toast.success(describeImproveLaunch(result));
      }
      // A failed launch already surfaced its own error toast (and the selection is
      // kept so the user can retry once the reason is fixed).
    } catch (error) {
      toast.error(`Could not start the improvement batch: ${error?.message || 'unknown error'}`);
    } finally {
      setLaunchingImprove(false);
    }
  };
  const batchBtn = 'px-2.5 py-1 rounded-lg text-xs font-semibold disabled:opacity-40';

  return (
    <div id="ds-images-review" tabIndex={-1} data-workspace-focus
      className="flex flex-col gap-2 scroll-mt-20">
      {/* Says what the pass BLOCKS, once, in words — the per-button titles name
          which pass, and a title is not readable on a touch screen at all. */}
      {busyReason && (
        <p role="status" aria-live="polite"
          className="rounded-lg border border-amber-400/30 bg-amber-400/5 px-2.5 py-1.5 text-[11px] leading-snug text-amber-100/90">
          <span aria-hidden="true">🔒 </span>{READS_STAY_OPEN}
        </p>
      )}
      {onBatch && autoTriageAvailable(faceScoringBlocked) && (
        <AutoTriageBar images={images.filter((image) => !isSmallImageRescueRow(image))}
          allImages={unfilteredImages.filter((image) => !isSmallImageRescueRow(image))}
          datasetId={datasetId} faceThresholds={faceThresholds} onBatch={onBatch}
          busy={bulkBusy} applying={autoTriageApplying} onApplyingChange={setAutoTriageRun} />
      )}
      <div id="ds-images-bulk" tabIndex={-1}
        className="flex items-center gap-2 flex-wrap text-xs scroll-mt-20">
        {onBatch && (
          selected.size === 0 ? (
            <>
              <span className="text-content-subtle">Tick images to curate them in bulk —</span>
              {/* The grid shows one page at a time, so this button has to say
                  what it takes: EVERY image the current filters show, not the
                  500 on screen (same rule as the Bank's "whole filter, all
                  pages"). A selection also survives a page change. */}
              <button type="button" data-workspace-focus
                disabled={bulkBusy}
                title="Selects every image the current filters show — all pages, not just the tiles on screen"
                onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
                className="text-content-muted underline hover:text-content disabled:opacity-40">select all ({selectable.length})</button>
            </>
          ) : (
            <div role="toolbar" aria-label="Bulk actions on the selection"
              className="flex items-center gap-2 flex-wrap rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-2.5 py-1.5 w-full">
              <span className="text-content font-semibold">{selected.size} selected</span>
              {bulkAction && (
                <span role="status" aria-live="polite" aria-atomic="true"
                  className="rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-amber-200">
                  {bulkActionMessage(bulkAction)}
                </span>
              )}
              <button type="button" disabled={curationWriteBusy} onClick={() => act('keep')}
                className={`${batchBtn} bg-green-600/80 text-white`}>✓ Keep</button>
              <button type="button" disabled={curationWriteBusy} onClick={() => act('reject')}
                className={`${batchBtn} bg-red-600/80 text-white`}>✕ Reject</button>
              <button type="button" disabled={curationWriteBusy} onClick={() => act('pending')}
                title="Back to undecided" className={`${batchBtn} bg-surface text-content border border-border`}>↺ Undecide</button>
              <button type="button" disabled={curationWriteBusy} onClick={() => act('clear_caption')}
                title="Delete the selected images' captions (the Caption button then regenerates them)"
                className={`${batchBtn} bg-surface text-content border border-border`}>🧹 Clear captions</button>
              {/* One button per engine that can actually run, because the two
                  passes are a CHOICE, not two qualities of the same thing:
                  Klein re-renders detail (and can move skin and colour),
                  SeedVR2 resolves it and leaves the look alone (issue #32,
                  SurpassHR). Each button's title carries that sentence, so the
                  difference is readable before the click, not after a ruined
                  batch. They wrap onto their own line on a narrow screen — the
                  toolbar is already flex-wrap. */}
              {onImproveBatch && improveEngines.map((engine) => {
                const blocked = improveEngineBlockedReason(engine.id, {
                  caps, engines: caps?.engines,
                  eligibleCount: improveSelection.eligible.length,
                });
                return (
                  <button key={engine.id} type="button"
                    onClick={() => improveSelected(engine.id)}
                    disabled={improveLaunchBusy || !!improveLabel || !!blocked}
                    title={blocked
                      ? `${blocked}${exclusionSummary ? ` ${exclusionSummary}.` : ''}`
                      : `${engine.summary} Runs in the background, a few at a time — survives a page reload.${exclusionSummary ? ` Excluded: ${exclusionSummary}.` : ''}`}
                    className={`${batchBtn} border border-indigo-400/50 bg-indigo-500/20 text-indigo-100`}>
                    {improveLabel || `${engine.emoji} ${engine.action} (${improveSelection.eligible.length})`}
                  </button>
                );
              })}
              {onImproveBatch && improveSelection.excluded.length > 0 && (
                <span className="text-content-subtle" title={exclusionSummary}>
                  {improveSelection.excluded.length} not eligible
                </span>
              )}
              <button type="button" disabled={curationWriteBusy} onClick={() => act('delete')}
                className={`${batchBtn} bg-red-500/15 border border-red-500/40 text-red-300`}>
                {bulkAction?.action === 'delete' ? bulkActionMessage(bulkAction) : '🗑 Delete'}
              </button>
              <span className="ml-auto flex gap-2">
                <button type="button" disabled={bulkBusy}
                  title="Selects every image the current filters show — all pages, not just the tiles on screen"
                  onClick={() => setSelected(new Set(selectable.map((i) => i.id)))}
                  className="text-content-muted underline hover:text-content disabled:opacity-40">all ({selectable.length})</button>
                <button type="button" disabled={bulkBusy} onClick={() => setSelected(new Set())}
                  className="text-content-muted underline hover:text-content disabled:opacity-40">none</button>
              </span>
              {/* A bulk improve is where the damage scales: the instruction that
                  turned ONE anime tile realistic is about to run on the whole
                  selection. w-full = its own line under the buttons, at every
                  width (Qeeyana, Reddit). */}
              {/* The anime-realism warning is about KLEIN's rewrite specifically —
                  a SeedVR2 pass does not restyle, so the note follows Klein. */}
              {onImproveBatch && kleinAvailable && improveSelection.eligible.length > 0 && (
                <KleinImproveNote subjectType={subjectType} datasetId={datasetId}
                  className="w-full border-t border-indigo-400/20 pt-1.5" />
              )}
            </div>
          )
        )}
        <TileSizeControl size={tileSize} onChange={setTileSize} titles={TILE_SIZE_TITLE} className="ml-auto" />
      </div>
      <GridPager view={view} onGo={goToPage} where="top" />
      <div className={`grid ${TILE_SIZE_COLS[tileSize]} gap-2`}>
        {view.items.map((img) => (
          <DatasetGridItem key={img.id} img={img} datasetId={datasetId} onStatus={onStatus} onCaption={onCaption}
            improvementState={improvementStates.get(img.id)}
            onCrop={onCrop} onDelete={onDelete} onMirror={onMirror}
            mirrorBusy={Boolean(mirroringIds?.has(img.id))} busy={bulkBusy}
            improveBusy={improveLaunchBusy} generateBusy={generateLaunchBusy}
            curationBusy={curationWriteBusy}
            busyReason={busyReason}
            onScoreFace={onScoreFace} scoreFaceBusy={Boolean(scoringFaceIds?.has(img.id))}
            faceScoringBusy={Boolean(scoringFaceIds?.size)}
            faceScoringBlocked={faceScoringBlocked}
            /* onRegenerate, onView and onReimprove are handed over UNCONDITIONALLY.
               Withholding a handler while its button's `disabled` says otherwise makes
               the button light up and do nothing — it happened to onView first, and it
               happened again here the moment 🔄 moved off `bulkBusy`. The refusal
               belongs in ONE place, the `disabled` the tile computes from the lane
               gates above, which already covers every case that must be refused. */
            onRegenerate={onRegenerate} onReimprove={onReimprove} onView={onView}
            selected={selected.has(img.id)}
            onToggleSelect={onBatch && !selectionLocked && !isSmallImageRescueRow(img) ? toggle : undefined}
            nonce={(nonces && nonces[img.id]) || 0} faceThresholds={faceThresholds}
            tileSize={tileSize} datasetKind={datasetKind} dualCaptions={dualCaptions} />
        ))}
      </div>
      <GridPager view={view} onGo={goToPage} where="bottom" />
    </div>
  );
}
