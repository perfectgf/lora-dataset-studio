import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, ArrowUp, Ban, CheckCircle2, Copy, Download, Drama, Eraser, Filter, FolderOpen, Globe, Loader2, Package, PenLine, PersonStanding, RefreshCw, Save, Scissors, Search, Settings, SlidersHorizontal, Sparkles, Trash2, X } from 'lucide-react';
import { useNavigate, useSearchParams } from 'react-router';
import CompositionBar from './CompositionBar';
import CoveragePanel from './CoveragePanel';
import ClassifyFramingButton from './ClassifyFramingButton';
import ReferencePanel from './ReferencePanel';
import TrainingPanel from './TrainingPanel';
import { fmt } from '../../utils/studioFormat';
import ImportDropzone from './ImportDropzone';
import ConceptSourcesPanel from './ConceptSourcesPanel';
import BankImportPanel from './BankImportPanel';
import DatasetFolderNote from './DatasetFolderNote';
import { isDatasetImportBlocked, isStopGenerationBlocked } from './scraperState';
import { holdsLocalGpu } from '../../utils/activityLanes.js';
import { IMPROVE_DERIVATION } from './improveCandidates.js';
import { faceAnalysisState, faceAnalysisLabel } from './faceScoringGate.js';
import DatasetGrid from './DatasetGrid';
import { datasetBusyReason } from './datasetBusyReason.js';
import KleinModelSetting from '../shared/KleinModelSetting';
import SmallImageRescueReview from './SmallImageRescueReview';
import CaptionToolsBar from './CaptionToolsBar';
import CaptionOptionsPopover from './CaptionOptionsPopover';
import { recaptionConfirmation } from './captionCategory';
import { defaultEditEngine } from './referenceEdit';
import { localEngineUnavailableReason, hasComfyui } from '../../utils/localEngineReason.js';
import { captionEnginesSummary, CAPTION_ENGINE_WHY } from '../../utils/captionEngines.js';
// …and the per-image half of the same question, for the captions listed in full here.
import { captionOriginInfo } from '../../utils/captionOrigin.js';
import { extraRefCropSource } from './extraRefs';
import DatasetSettingsModal from './DatasetSettingsModal';
import DatasetToBankDialog from './DatasetToBankDialog';
import TextScanDialog from './TextScanDialog';
import WatermarkReviewLightbox, { buildWatermarkRecap } from './WatermarkReviewLightbox';
// Lazily loaded: each renders behind a condition (a click opens it) or a
// hidden section, so its code leaves the DatasetPage entry chunk - which
// sat alone above vite's 500 kB warning. WatermarkReviewLightbox stays
// static: its buildWatermarkRecap named export is used at runtime here,
// and a static import would pin the module into this chunk anyway.
const VariationCatalog = lazy(() => import('./VariationCatalog'));
const CropModal = lazy(() => import('./CropModal'));
const ReferenceEditModal = lazy(() => import('./ReferenceEditModal'));
const DatasetLightbox = lazy(() => import('./DatasetLightbox'));
const PublishHfModal = lazy(() => import('./PublishHfModal'));
import {
  summarizeFlagged, rejectableFlagged, rejectFlaggedConfirmText, flaggedSourceNote,
} from './watermarkFlagged.js';
import { useToast } from '../common/Toast';
import { pickNativeFolder, FolderBrowserModal } from '../common/FolderPicker';
import { useCapabilities } from '../../context/CapabilitiesContext';
import InstallRunner from '../setup/InstallRunner';
import GuidedChecklist from './GuidedChecklist';
import NextStepCard from './NextStepCard';
import TrainingReadiness from './TrainingReadiness';
import useGuidedFlow from '../../hooks/useGuidedFlow';
import { filterImages, normalizeTag } from '../../utils/tagFilter';
import { summarizeGeneration, refusalHeadline, failureHeadline } from './generationOutcome.js';
import {
  GRID_STATUS_FILTERS, DEFAULT_GRID_STATUS_FILTER,
  filterImagesByStatus, gridStatusFilterCounts, normalizeGridStatusFilter,
} from '../../utils/gridStatusFilter';
import {
  DEFAULT_DATASET_SORT, datasetSortOptions, normalizeDatasetSort, sortDatasetImages,
} from '../../utils/gridSort';
import {
  buildSmallImageRescuePairs,
  filterSmallImageRescueGrid,
  isSmallImageRescueRow,
} from '../../utils/smallImageRescue';
import { describeDerivedComparison } from '../../utils/derivedCompare';
import { WORKSPACE_SECTIONS, SECTION_FOR_TARGET } from './workspaceSections';
import { postJson, putJson } from '../../api/fetchClient';
import { datasetToBankRequest, datasetToBankUrl } from './datasetToBank';
import { HelpBadge } from '../../help/HelpMode';
import { requestHelpTip } from '../../help/helpTips';
import { useDatasetCameraAngles } from '../../hooks/useDatasetCameraAngles';
import { datasetCameraRefusal } from '../../utils/cameraAngles';
import { openCollapsedAncestors } from '../../help/revealTarget';
import {
  PANEL_STATUS,
  getWorkspacePanel,
  getWorkspacePanelStatus,
  getWorkspacePanels,
  resolveWorkspaceLocation,
  withWorkspaceLocation,
} from './workspaceNavigation';

const EMPTY_IMAGES = Object.freeze([]);

// Grid decision filter (All / Undecided / Kept / Rejected / Improve candidates):
// a VIEW preference, not dataset data, so it is persisted globally — same lazy-init
// + effect pattern as `datasetGridTileSize` (DatasetGrid.jsx) / `cloudRunsRecentCollapsed`.
// Because it SURVIVES a reload, the filtered-view banner above the grid is not
// cosmetic: it is the only thing that stops a persisted filter from reading as
// "my images are gone" when the workspace is reopened.
const GRID_STATUS_FILTER_KEY = 'datasetGridStatusFilter';
// Grid ordering (default / face similarity ↓ / ↑) — same persisted-view-preference
// pattern. The KEY and the stored ids are stable handles: never rename either.
const GRID_SORT_KEY = 'datasetGridSort';

// Style partagé des items du menu « ⋯ More » du header (actions secondaires).
const MENU_ITEM = 'min-h-10 lg:min-h-0 w-full flex items-center gap-2 text-left px-2.5 py-1.5 rounded-md text-sm text-content hover:bg-surface-raised disabled:opacity-40';

/* En-tête de section (miroir visuel du SectionHeader de Settings, en h2 : le h1
   de la page reste le nom du dataset) : eyebrow mono + titre + description. */
function SectionHeading({ id, eyebrow, title, description, badge }) {
  return (
    <div id={id} tabIndex={-1}>
      <p className="m-0 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">{eyebrow}</p>
      <h2 className="m-0 mt-0.5 flex items-center gap-2 text-content text-base font-semibold">{title}{badge}</h2>
      {description && <p className="m-0 mt-0.5 text-content-muted text-[0.75rem] leading-relaxed">{description}</p>}
    </div>
  );
}

/* Pastille de compte dans la sidebar — sobre : ambre = action attendue (triage,
   watermarks, fuites), indigo pulsé = travail en cours (générations), neutre =
   simple info. Jamais couleur seule : le sr-only épelle le sens. */
function NavBadge({ badge }) {
  if (!badge) return null;
  const cls = badge.tone === 'amber' ? 'border-amber-400/50 bg-amber-500/15 text-amber-200'
    : badge.tone === 'indigo' ? 'border-indigo-400/50 bg-indigo-500/15 text-indigo-200'
    : 'border-border bg-surface-raised text-content-subtle';
  return (
    <span
      className={`ml-auto shrink-0 rounded-full border px-1.5 py-px text-[0.625rem] font-semibold tabular-nums ${cls} ${badge.pulse ? 'animate-pulse' : ''}`}>
      <span aria-hidden>{badge.n}</span>
      <span className="sr-only"> — {badge.srLabel}</span>
    </span>
  );
}

/* Decision filter above the grid — isolate what still needs a ✓/✕, what was kept
   or rejected, or the Klein improvement candidates, without hunting through the
   whole set. It narrows the list the grid receives, so auto-triage, "select all"
   and every bulk action then operate on exactly what is on screen. Composes with
   the caption tag filter (both apply). */
function GridStatusFilter({ value, counts, onChange }) {
  return (
    <div className="flex items-center gap-1.5 flex-wrap text-xs" role="group"
      aria-label="Filter the grid by decision">
      <span className="text-content-subtle shrink-0">Show</span>
      {GRID_STATUS_FILTERS.map((f) => {
        const on = f.id === value;
        return (
          <button key={f.id} type="button" onClick={() => onChange(f.id)}
            aria-pressed={on} title={f.title}
            className={`min-h-10 lg:min-h-0 px-2 py-0.5 rounded-full border text-[0.6875rem] font-semibold tabular-nums ${
              on ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-100'
                : 'border-border bg-surface text-content-muted hover:text-content'}`}>
            {f.label} ({counts[f.id] ?? 0})
          </button>
        );
      })}
      <HelpBadge topic="action-grid-status-filter" />
    </div>
  );
}

/* Two ways to arrange the grid, and they answer different questions.
   MEASURED (face similarity to the reference — asked for by nofaceman, Discord)
   ranks the whole grid best-to-worst; CATEGORICAL (shot type — asked for by
   .samexit, Discord) does not rank anything, it puts every face shot in one run,
   every bust in the next, so near-identical images can be compared side by side
   instead of scattered through the arrival order. Nothing else is offered: the
   aesthetic/sharpness scores live on bank images, which have their own sort.
   Sorting NARROWS nothing: it composes with both grid filters, and since the
   grid derives its tiles AND its "select all" from the same list, the selection
   follows the order on screen. Images a pass never reached go last, both ways;
   an entry whose pass has not run is greyed out naming that pass, rather than
   silently reordering nothing. `shrink-0` + a bounded width keep it usable when
   the toolbar wraps at 400 px. */
function GridSortSelect({ value, images, onChange }) {
  return (
    <label className="flex shrink-0 items-center gap-1 text-xs text-content-subtle">
      Sort
      <select value={value} onChange={(e) => onChange(e.target.value)}
        aria-label="Sort the grid"
        title="Order the images by face similarity to your reference, or group them by shot type. Images the pass never reached sink to the end."
        className="max-w-[13rem] rounded-md border border-border bg-surface px-2 py-0.5 text-[0.6875rem] text-content">
        {datasetSortOptions(images).map((o) => (
          <option key={o.id} value={o.id} disabled={o.disabled} title={o.title}>
            {o.label}
          </option>
        ))}
      </select>
      <HelpBadge topic="action-grid-sort" />
    </label>
  );
}

/* Loud banner sitting directly above the grid whenever a tag filter is active,
   so the user can NEVER mistake a filtered view for "images disappeared". Shows
   every active exclusion (⊘) / inclusion (◉ only) as a removable chip, the live
   "showing N of M" count, and a one-click "clear all". Session-only state lives
   in the parent workspace (transient view, not persisted). */
function GridFilterBar({
  excludes, includes, statusLabel, coverageLabel, shown, total,
  onRemoveExclude, onRemoveInclude, onRemoveStatus, onRemoveCoverage, onClearAll,
}) {
  return (
    <div role="status" data-probe-chrome="filter-bar"
      className="flex items-center gap-2 flex-wrap rounded-lg border-2 border-amber-400/50 bg-amber-400/10 px-3 py-2">
      <span className="inline-flex items-center gap-1.5 text-amber-200 text-sm font-semibold shrink-0"><Filter aria-hidden="true" className="h-4 w-4" /> Filtered view</span>
      <span className="text-content-muted text-xs tabular-nums shrink-0">
        showing {shown} of {total}
      </span>
      <div className="flex items-center gap-1.5 flex-wrap">
        {statusLabel && (
          <span
            className="inline-flex items-center gap-1 rounded-full border border-amber-400/50 bg-amber-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-amber-100">
            <span aria-hidden>◧</span> {statusLabel} only
            <button type="button" onClick={onRemoveStatus}
              aria-label="Show images with any decision again"
              className="min-h-10 min-w-10 lg:min-h-0 lg:min-w-0 lg:w-4 lg:h-4 grid place-items-center rounded-full hover:bg-amber-500/30"><X aria-hidden="true" className="h-3 w-3" /></button>
          </span>
        )}
        {coverageLabel && (
          <span
            className="inline-flex items-center gap-1 rounded-full border border-emerald-400/50 bg-emerald-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-emerald-100">
            <Search aria-hidden="true" className="h-3 w-3" /> {coverageLabel}
            <button type="button" onClick={onRemoveCoverage}
              aria-label="Stop showing only the images from that coverage chip"
              className="min-h-10 min-w-10 lg:min-h-0 lg:min-w-0 lg:w-4 lg:h-4 grid place-items-center rounded-full hover:bg-emerald-500/30"><X aria-hidden="true" className="h-3 w-3" /></button>
          </span>
        )}
        {excludes.map((t) => (
          <span key={`x-${t}`}
            className="inline-flex items-center gap-1 rounded-full border border-rose-400/50 bg-rose-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-rose-200">
            <span aria-hidden>⊘</span> {t}
            <button type="button" onClick={() => onRemoveExclude(t)}
              aria-label={`Stop hiding images tagged ${t}`}
              className="min-h-10 min-w-10 lg:min-h-0 lg:min-w-0 lg:w-4 lg:h-4 grid place-items-center rounded-full hover:bg-rose-500/30"><X aria-hidden="true" className="h-3 w-3" /></button>
          </span>
        ))}
        {includes.map((t) => (
          <span key={`i-${t}`}
            className="inline-flex items-center gap-1 rounded-full border border-indigo-400/50 bg-indigo-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-indigo-200">
            <span aria-hidden>◉</span> only {t}
            <button type="button" onClick={() => onRemoveInclude(t)}
              aria-label={`Stop isolating images tagged ${t}`}
              className="min-h-10 min-w-10 lg:min-h-0 lg:min-w-0 lg:w-4 lg:h-4 grid place-items-center rounded-full hover:bg-indigo-500/30"><X aria-hidden="true" className="h-3 w-3" /></button>
          </span>
        ))}
      </div>
      <button type="button" onClick={onClearAll}
        className="ml-auto shrink-0 text-content-muted underline hover:text-content text-xs">
        clear all
      </button>
    </div>
  );
}

export default function DatasetWorkspace({ ds, onBack }) {
  const navigate = useNavigate();
  const toast = useToast();
  const { caps, loading: capsLoading, refresh: refreshCaps } = useCapabilities();
  // 📷 With the other hooks, above the loading return — rules-of-hooks.
  const shootDatasetViews = useDatasetCameraAngles();
  const d = ds.data;
  // 🎭 Analyze faces: state + tooltip derived from the SERVER's verdict
  // (`face_scoring_blocked`, a sentence or null) — see faceScoringGate.js. The UI
  // never decides on its own which subjects InsightFace can read.
  const faceAnalysis = faceAnalysisState({
    blockedReason: d?.face_scoring_blocked,
    hasRef: !!d?.ref_filename,
    busy: ds.busy,
    // InsightFace/antelopev2 are an optional extra: without them the button used
    // to look alive and only failed on click. Now it names the gap and links Setup.
    capable: !!caps?.face_scoring,
    capsLoading,
  });
  const [cropImg, setCropImg] = useState(null);
  // 🔍 The coverage chip the user clicked, or null. Session-only like the tag
  // filters: it is a way of LOOKING at the set, not a property of it, and one
  // that would be actively misleading to restore days later against a dataset
  // whose captions have since been rewritten. {key, label, ids}.
  const [coverageFilter, setCoverageFilter] = useState(null);
  const [captionOptionsOpen, setCaptionOptionsOpen] = useState(false);
  // Frozen snapshot of the flagged queue when review mode opens (null = closed).
  const [reviewQueue, setReviewQueue] = useState(null);
  // What the last "Reject all flagged" actually did, in the SERVER's number, and
  // the way back. The undo exists (Show ▸ Rejected → ✓ Keep) but it only exists
  // for the user if it is named at the moment it becomes useful.
  const [rejectFlaggedNote, setRejectFlaggedNote] = useState('');
  const zipInput = useRef(null);   // hidden input for "Import dataset (ZIP)"
  const [refCrop, setRefCrop] = useState(false);
  const [refEdit, setRefEdit] = useState(false);
  // Filename of the extra reference being cropped (extras have no numeric id).
  const [extraRefCrop, setExtraRefCrop] = useState(null);
  const [viewImg, setViewImg] = useState(null);
  const [gridBulkBusy, setGridBulkBusy] = useState(false);
  useEffect(() => {
    if (gridBulkBusy) setViewImg(null);
  }, [gridBulkBusy]);
  useEffect(() => { setGridBulkBusy(false); }, [d?.id]);
  const [captionMode, setCaptionMode] = useState(null);   // null → défaut auto selon train_type
  const [showLeaks, setShowLeaks] = useState(false);       // liste dépliée des captions qui fuient
  const [captionToolsOpen, setCaptionToolsOpen] = useState(false);
  const [installInpaintOpen, setInstallInpaintOpen] = useState(false);  // panneau d'install LaMa
  const [watermarkMethod, setWatermarkMethod] = useState('lama');  // moteur d'inpaint batch : lama | klein
  /* What 🧽 Clean aims at: every flagged page ('all'), only the 🔤 text-flagged
     ones ('text'), or only the 🚩 watermark-flagged ones ('watermark'). The
     split is BY PAGE — a page carrying both counts as text (its zones share one
     channel, so one page is never split between two runs). Bank parity: its
     panel offers the same three, in the same words. */
  const [watermarkTarget, setWatermarkTarget] = useState('all');
  // 🔤 Find text launch window (full parity with the bank's). Hook up here
  // with its siblings — the counts it prices live next to watermarkDetected,
  // past the early returns.
  const [textScanOpen, setTextScanOpen] = useState(false);
  const [savingAllowCrop, setSavingAllowCrop] = useState(false);  // write-through of the auto-crop pref
  const [checkpointCount, setCheckpointCount] = useState(0);
  const [checkpointHost, setCheckpointHost] = useState(null);
  const [trainingNavigation, setTrainingNavigation] = useState({ ready: false, queueCount: 0 });
  // « Continue anyway » : ack remonté par la pastille de préparation (garde-fou
  // qualité contournable) → débloque le bouton Train du panneau et voyage jusqu'au
  // launch (allow_not_ready). Le serveur reste l'autorité.
  const [notReadyAck, setNotReadyAck] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [importToBankOpen, setImportToBankOpen] = useState(false);
  const [publishHfOpen, setPublishHfOpen] = useState(false);
  const [folderBrowseOpen, setFolderBrowseOpen] = useState(false);  // in-app folder browser (native-dialog fallback)
  // Grid tag-filter (session-only): tags whose images are hidden (exclude) or the
  // ONLY tags allowed through (include). Both are normalized (trim+lowercase).
  const [excludeTags, setExcludeTags] = useState([]);
  const [includeTags, setIncludeTags] = useState([]);
  // Grid decision filter — PERSISTED (see GRID_STATUS_FILTER_KEY), unlike the
  // session-only tag filter: isolating the undecided pile is a working mode you
  // keep across reloads, not a one-off lookup.
  const [statusFilter, setStatusFilter] = useState(() => {
    try { return normalizeGridStatusFilter(localStorage.getItem(GRID_STATUS_FILTER_KEY)); }
    catch { return DEFAULT_GRID_STATUS_FILTER; }
  });
  useEffect(() => {
    try { localStorage.setItem(GRID_STATUS_FILTER_KEY, statusFilter); }
    catch { /* ignore — private mode */ }
  }, [statusFilter]);
  // Grid sort — PERSISTED like the decision filter: "show me the least-alike
  // first" is a working mode you keep across reloads, not a one-off lookup. The
  // stored value is normalised on read, so an id from an older build (or a
  // hand-edited localStorage) degrades to the default order instead of freezing
  // the grid on a sort nothing implements.
  const [gridSort, setGridSort] = useState(() => {
    try { return normalizeDatasetSort(localStorage.getItem(GRID_SORT_KEY)); }
    catch { return DEFAULT_DATASET_SORT; }
  });
  useEffect(() => {
    try { localStorage.setItem(GRID_SORT_KEY, gridSort); }
    catch { /* ignore — private mode */ }
  }, [gridSort]);
  const [searchParams, setSearchParams] = useSearchParams();
  // "Allow auto-crop" is a persisted preference (Settings ▸ Watermark inpainting). The
  // batch Clean bar shows the SAME setting and writes it through here, so there's one
  // source of truth: Clean then reads it server-side. Best-effort — a failed save leaves
  // the toggle where it was and tells the user, never silently diverging from Settings.
  const allowAutoCrop = caps.watermark_allow_crop !== false;
  const setWatermarkAllowCrop = useCallback(async (value) => {
    setSavingAllowCrop(true);
    try {
      await putJson('/api/settings', { config: { watermark: { allow_crop: Boolean(value) } } });
      await refreshCaps(true);
    } catch {
      toast.error('Could not save the auto-crop preference');
    } finally {
      setSavingAllowCrop(false);
    }
  }, [refreshCaps, toast]);
  const navImages = d?.images || EMPTY_IMAGES;
  const navContext = useMemo(() => ({
    kind: d?.kind || 'character',
    hasSelectableImages: filterSmallImageRescueGrid(navImages)
      .some((image) => Boolean(image.filename)),
    hasKeptImages: navImages.some((image) => image.status === 'keep'),
    hasCaptionedKept: navImages.some(
      (image) => image.status === 'keep' && Boolean((image.caption || '').trim()),
    ),
    hasLeakMetadata: Boolean(d?.caption_leak),
    watermarkDetected: navImages.filter((image) => image.watermark_state === 'detected').length,
    watermarkRejectable: rejectableFlagged(navImages).length,
    smallImageRescue: buildSmallImageRescuePairs(navImages).filter((pair) => !pair.resolved).length,
    unused: navImages.filter((image) => (image.status === 'reject' || image.status === 'failed')
      && !isSmallImageRescueRow(image)).length,
    hfPublish: Boolean(caps.hf_publish),
    trainingVisible: Boolean(caps.training_visible),
    trainingStatusReady: !caps.training_visible || trainingNavigation.ready,
    trainingQueueCount: trainingNavigation.queueCount,
    studioVisible: Boolean(caps.studio_visible),
  }), [d, navImages, caps.hf_publish, caps.training_visible, caps.studio_visible, trainingNavigation]);
  const workspaceLocation = resolveWorkspaceLocation(searchParams, navContext);
  const section = workspaceLocation.section;
  const panel = workspaceLocation.panel;

  const writeWorkspaceLocation = useCallback((sectionId, panelId = null, replace = false) => {
    setSearchParams(
      (previous) => withWorkspaceLocation(previous, sectionId, panelId),
      { replace },
    );
  }, [setSearchParams]);

  const focusRequestedRef = useRef(false);
  const [landingRequest, setLandingRequest] = useState(0);

  const setSection = useCallback((sectionId) => {
    focusRequestedRef.current = false;
    writeWorkspaceLocation(sectionId, null, false);
  }, [writeWorkspaceLocation]);

  const navigateToPanel = useCallback((sectionId, panelId) => {
    if (getWorkspacePanelStatus(sectionId, panelId, navContext) !== PANEL_STATUS.AVAILABLE) return;
    focusRequestedRef.current = true;
    setLandingRequest((value) => value + 1);
    writeWorkspaceLocation(sectionId, panelId, false);
  }, [navContext, writeWorkspaceLocation]);

  const clearActivePanel = useCallback(() => {
    focusRequestedRef.current = false;
    writeWorkspaceLocation(section, null, true);
  }, [section, writeWorkspaceLocation]);

  useEffect(() => {
    if (!d || workspaceLocation.pending || !workspaceLocation.needsNormalization) return;
    setSearchParams(
      (previous) => withWorkspaceLocation(previous, workspaceLocation.section, workspaceLocation.panel),
      { replace: true },
    );
  }, [d, workspaceLocation.pending, workspaceLocation.needsNormalization,
      workspaceLocation.section, workspaceLocation.panel, setSearchParams]);

  useEffect(() => {
    const destination = panel ? getWorkspacePanel(section, panel) : null;
    if (destination?.reveal === 'caption-leak') setShowLeaks(true);
    if (destination?.reveal === 'caption-tools') setCaptionToolsOpen(true);
  }, [section, panel]);

  const onRevealOpenChange = useCallback((panelId, nextOpen, setter) => {
    setter(nextOpen);
    if (!nextOpen && panel === panelId) clearActivePanel();
  }, [panel, clearActivePanel]);
  // Hooks must run unconditionally on every render — deriveSteps() null-guards `d`,
  // so this is safe to call before the loading early-return below.
  const { steps, nextStep } = useGuidedFlow(d, caps, checkpointCount);
  // Filters are per-dataset & transient — drop them when switching datasets so they
  // never leak from one dataset to the next.
  useEffect(() => { setExcludeTags([]); setIncludeTags([]); }, [d?.id]);

  // The landing effect below keys on the dataset IDENTITY (id), never the polled
  // `d` object. While a generation runs, useDataset re-fetches every 4 s and hands
  // back a brand-new object each time; depending on `d` would re-run the landing
  // and re-fire scrollIntoView on every poll, yanking the view back to the active
  // panel's anchor (`ds-add-reference` when the Reference panel is selected). The
  // id is stable across polls, still flips on null→loaded and on a dataset switch,
  // and the effect's own MutationObserver/rAF/timeout already handle a target that
  // appears slightly after the data does.
  const datasetId = d?.id ?? null;

  useEffect(() => {
    if (datasetId == null || !panel || workspaceLocation.pending) return undefined;
    const destination = getWorkspacePanel(section, panel);
    if (!destination) return undefined;
    const revealReady = destination.reveal === 'caption-leak'
      ? showLeaks
      : destination.reveal === 'caption-tools'
        ? captionToolsOpen
        : true;
    if (!revealReady) return undefined;
    let finished = false;
    let observer;
    let timer;
    let frame;

    const land = () => {
      if (finished) return true;
      const target = document.getElementById(destination.targetId);
      if (!target) return false;
      // A panel can live inside a collapsed disclosure (the "More ways out"
      // <details> in Import & export). Chromium still reports a client rect for
      // that subtree, so the landing below would happily scroll+ring something
      // the user cannot see. Open the disclosures on the way up first, starting
      // at the PARENT: a target that IS a <details> (Advanced options,
      // Checkpoints) is React-controlled and stays driven by its own state,
      // which the reveal guard right after this waits for.
      openCollapsedAncestors(target.parentElement);
      if (target.getClientRects().length === 0) return false;
      if ((destination.reveal === 'training-advanced'
          || destination.reveal === 'training-checkpoints') && !target.open) return false;
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      target.classList.remove('gf-highlight');
      void target.offsetWidth;
      target.classList.add('gf-highlight');
      window.setTimeout(() => target.classList.remove('gf-highlight'), 1500);
      if (focusRequestedRef.current) {
        const focusableSelector = [
          'button:not([disabled])', 'a[href]', 'input:not([disabled])',
          'select:not([disabled])', 'textarea:not([disabled])', 'summary',
          '[tabindex]:not([tabindex="-1"])',
        ].join(', ');
        const preferred = destination.focusSelector
          ? target.querySelector(destination.focusSelector)
          : target.querySelector('[data-workspace-focus]');
        const fallback = target.matches(focusableSelector)
          ? target
          : target.querySelector(focusableSelector);
        let focusTarget = preferred?.matches?.(':not(:disabled)') ? preferred : fallback;
        if (!focusTarget) {
          const nativeControl = target.matches('button, input, select, textarea');
          if (!nativeControl) {
            if (!target.hasAttribute('tabindex')) target.tabIndex = -1;
            focusTarget = target;
          } else {
            focusTarget = document.getElementById(`ds-section-${section}-heading`);
          }
        }
        focusTarget?.focus({ preventScroll: true });
        focusRequestedRef.current = false;
      }
      finished = true;
      observer?.disconnect();
      if (timer) window.clearTimeout(timer);
      return true;
    };

    if (!land()) {
      observer = new MutationObserver(land);
      observer.observe(document.body, {
        childList: true, subtree: true, attributes: true,
        attributeFilter: ['class', 'open'],
      });
      frame = requestAnimationFrame(() => {
        frame = undefined;
        land();
      });
      timer = window.setTimeout(() => {
        if (finished) return;
        observer.disconnect();
        const shouldFocus = focusRequestedRef.current;
        focusRequestedRef.current = false;
        writeWorkspaceLocation(section, null, true);
        if (shouldFocus) {
          document.getElementById(`ds-section-${section}-heading`)?.focus({ preventScroll: true });
        }
      }, 2000);
    }

    return () => {
      finished = true;
      observer?.disconnect();
      if (frame !== undefined) cancelAnimationFrame(frame);
      if (timer) window.clearTimeout(timer);
    };
  }, [datasetId, section, panel, workspaceLocation.pending, landingRequest,
      showLeaks, captionToolsOpen, writeWorkspaceLocation]);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const parentChip = document.querySelector(`[data-mobile-section="${section}"]`);
      const childChip = panel
        ? document.querySelector(`[data-mobile-panel="${panel}"]`)
        : null;
      for (const chip of [parentChip, childChip]) {
        if (chip?.getClientRects().length) {
          chip.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
        }
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [section, panel]);

  // One-time contextual tips (best-effort; shown once ever, independent of Help
  // mode). Each fires when the user first reaches the relevant surface.
  // MUST live above the `!d` early return: hooks after a conditional return
  // change the hook count between the Loading render and the loaded one
  // (React #310 crash — caught by runtime verification).
  const leakingCount = ((d && d.images) || []).filter((i) => i.leak).length;
  useEffect(() => { if (d && section === 'add') requestHelpTip('add-images-visit'); }, [d, section]);
  useEffect(() => { if (leakingCount >= 1) requestHelpTip('leak-panel-visible'); }, [leakingCount]);
  useEffect(() => { if (settingsOpen) requestHelpTip('dataset-settings-open'); }, [settingsOpen]);

  // ⏹ Stop generation: the cancel happens server-side WITHIN the request (the
  // in-flight rows are dropped before it answers), so a local flag covering the
  // round-trip is the honest feedback — without it the click looked like a no-op
  // and users re-clicked, believing Stop was broken. The ✨ improve batch keeps
  // its own longer-lived server `cancelling` (its worker unwinds asynchronously).
  // Above the early return for the same hook-count reason as the tips above.
  const [stoppingGeneration, setStoppingGeneration] = useState(false);
  const stopGeneration = useCallback(async () => {
    setStoppingGeneration(true);
    // finally, not "on success": a failed cancel must give the button back.
    try { await ds.cancelPending(); } finally { setStoppingGeneration(false); }
  }, [ds]);

  if (!d) return <p className="text-content-subtle text-sm">Loading…</p>;

  const images = d.images || [];
  const rescuePairs = buildSmallImageRescuePairs(images);
  const unresolvedRescuePairs = rescuePairs.filter((pair) => !pair.resolved);
  // An unresolved pair is intentionally absent from the generic grid/bulk
  // controls: only the atomic side-by-side resolver may decide it. Once resolved,
  // the chosen keep + rejected counterpart return to the regular dataset view.
  const unresolvedRescueIds = new Set(unresolvedRescuePairs.flatMap(
    (pair) => [pair.original.id, pair.candidate.id],
  ));
  const rescueGridImages = filterSmallImageRescueGrid(images);
  const rescueReviewCount = unresolvedRescuePairs.length;
  // Dataset CONCEPT : on masque tout ce qui est identité/visage (référence, générateur
  // de variations, analyse faciale, badge de fuite, composition, flux guidé) — il ne
  // reste que import brut → curation → caption (inversée) → entraînement.
  // Style follows the same compact layout as concept (no face/reference tools),
  // but keeps its own always-on semantics and requires content-only captions.
  const isConcept = d.kind === 'concept';
  const isStyle = d.kind === 'style';
  const isConceptual = isConcept || isStyle;
  // Leak check is KIND-specific (see the caption-leak panel): character flags identity,
  // concept flags the caption NAMING the concept (must bind to the trigger), style never
  // (its subjects' description IS the content). `isConceptual` is layout-only.
  // Fidélité corps : captions bannissent aussi les marques corporelles, composition
  // cible plus de bustes/corps, import plein cadre par défaut.
  const bodyFid = d.fidelity === 'body';
  const kept = images.filter((i) => i.status === 'keep').length;
  const unused = images.filter((i) => (i.status === 'reject' || i.status === 'failed')
    && !isSmallImageRescueRow(i)).length;
  const keptUncaptioned = images.filter((i) => i.status === 'keep' && !i.caption).length;
  const keptCaptioned = kept - keptUncaptioned;
  // Hand-written captions ('asserted'): a forced 🔄 Re-caption spares them, so the
  // confirm and the button must count only what the pass will actually rewrite.
  const keptAsserted = images.filter((i) => i.status === 'keep' && i.caption
    && i.caption_origin === 'asserted').length;
  const recaptionable = keptCaptioned - keptAsserted;
  // Captions that still leak identity/concept — the Identity-leak panel lists them for
  // in-place edit AND targeted 🔄 Re-caption (per row + a "Re-caption all leaking" header).
  const leakingImages = images.filter((i) => i.leak);
  // Overlaid watermarks still awaiting removal → drives the "🧽 Clean (N)" button.
  const watermarkDetected = images.filter((i) => i.watermark_state === 'detected').length;
  // …and how many of those are 🔤 text-flagged pages — the "What to clean"
  // selector prices its three choices from this split, and only appears when
  // there IS a split (with no text page the three choices collapse into one).
  const textFlaggedDetected = images.filter(
    (i) => i.watermark_state === 'detected' && i.text_state === 'detected').length;
  const cleanTargetCount = watermarkTarget === 'text' ? textFlaggedDetected
    : watermarkTarget === 'watermark' ? watermarkDetected - textFlaggedDetected
    : watermarkDetected;
  // 🔤 window pricing: the kept pile the pass actually reads — dismissed rows
  // are the machine's no-go, already-answered rows only re-enter through the
  // redo line.
  const keptForText = images.filter(
    (i) => i.status === 'keep' && i.watermark_state !== 'dismissed');
  const textToRead = keptForText.filter(
    (i) => !i.text_state || i.text_state === 'error').length;
  // …and what that pile is really made of: what a bulk reject would move, what it
  // would refuse to touch, which detector judged, and how many carry no position.
  const flagged = summarizeFlagged(images);
  const flaggedNote = flaggedSourceNote(flagged);
  // Style de caption : défaut AUTO (SDXL booru-native → booru tags ; sinon prose),
  // surchargé par le sélecteur. Anima est HYBRIDE (les deux formes sont natives) :
  // le défaut prose n'est qu'un point de départ, basculer sur booru est légitime et
  // ne déclenche aucun garde-fou au lancement.
  const effCaptionMode = captionMode || (d.train_type === 'sdxl' ? 'booru' : 'prose');
  // ── Grid tag-filter (session-only) ──────────────────────────────────────────
  // A tag is toggled in its list and mutually excluded from the other (a tag can't
  // be both hidden and isolated). Match mode follows the caption style so booru
  // captions match a whole tag, prose captions a whole word (see utils/tagFilter).
  const toggleTag = (setSelf, setOther) => (raw) => {
    const t = normalizeTag(raw);
    if (!t) return;
    setOther((prev) => prev.filter((x) => x !== t));
    setSelf((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]));
  };
  const toggleExclude = toggleTag(setExcludeTags, setIncludeTags);
  const toggleInclude = toggleTag(setIncludeTags, setExcludeTags);
  const clearFilters = () => {
    setExcludeTags([]); setIncludeTags([]); setStatusFilter(DEFAULT_GRID_STATUS_FILTER);
    setCoverageFilter(null);
  };
  const tagFiltersActive = excludeTags.length > 0 || includeTags.length > 0;
  const statusFilterActive = statusFilter !== DEFAULT_GRID_STATUS_FILTER;
  const filtersActive = tagFiltersActive || statusFilterActive || !!coverageFilter;
  const statusFilterLabel = GRID_STATUS_FILTERS.find((f) => f.id === statusFilter)?.label || '';
  const statusFilterOpts = { unresolvedRescueIds };
  const statusCounts = gridStatusFilterCounts(rescueGridImages, statusFilterOpts);
  // The list actually rendered by the grid. Filtering here means select-all,
  // auto-triage and every bulk action operate ONLY on the visible images. The
  // Caption-tools counts keep using the full `images` list (global, never lies).
  // Decision filter first, tag filter on top — the two compose, order-independent.
  // The sort runs LAST and only reorders: membership is entirely the filters'
  // business, so "shown / total" and every bulk action keep meaning what they say.
  const filteredGridImages = filterImages(
    filterImagesByStatus(rescueGridImages, statusFilter, statusFilterOpts),
    { excludes: excludeTags, includes: includeTags, mode: effCaptionMode },
  );
  // 🔍 A coverage chip narrows to the exact images that bucket counted. It is an
  // id list rather than a re-run of the lexicon in the browser: the panel and the
  // grid then cannot disagree about what "profile 3" means, and the app keeps ONE
  // place where a caption is read for variety (services/caption_coverage.py).
  // Third filter, same rank as the other two — so it also narrows what select-all
  // and every bulk action see, which is the point of picking a bucket.
  const coverageIds = coverageFilter ? new Set(coverageFilter.ids) : null;
  const gridImages = sortDatasetImages(coverageIds
    ? filteredGridImages.filter((i) => coverageIds.has(i.id))
    : filteredGridImages, gridSort);
  // Outcome tally for the refusal notice below the progress banner. Counted from
  // the rows already polled — no extra request, and it survives a page reload
  // because it reads the stored fail_kind rather than a one-shot end-of-batch
  // event nobody would be watching when the batch actually ends.
  // Deliberately NOT memoised: this line sits after an early return, and a hook
  // here throws "Rendered more hooks than during the previous render" the moment
  // a dataset is opened. It is one O(n) pass over a list this component already
  // walks a dozen times unmemoised.
  const outcome = summarizeGeneration(images);
  const refusalNote = refusalHeadline(outcome);
  const pending = images.filter((i) => i.status === 'pending' && !i.filename
    && !unresolvedRescueIds.has(i.id)).length;
  // The same rows minus the ✨ improve candidates. `pending` counts everything in
  // flight, which is right for the progress banner ("3 generating…"); it is wrong
  // for the ⚡ Generate double-click guard, which asks "is a GENERATION batch
  // already running?". While the button was greyed during an improve batch the
  // question never came up — now that it is clickable, which is the whole point
  // of #44, the first click of the nominal path met a dialog announcing a batch
  // that did not exist.
  const pendingGenerations = images.filter((i) => i.status === 'pending' && !i.filename
    && i.derivation_kind !== IMPROVE_DERIVATION
    && !unresolvedRescueIds.has(i.id)).length;
  const triage = images.filter((i) => i.status === 'pending' && i.filename
    && !unresolvedRescueIds.has(i.id)).length;   // generated/imported, awaiting ✓/✕

  const toggleLeakReview = () => {
    onRevealOpenChange('leak-review', !showLeaks, setShowLeaks);
  };

  /* Saut vers une ancre gf-* (checklist, NextStep, « Fix → » du preflight) :
     on bascule d'abord la sidebar sur la section qui l'héberge, puis on scrolle
     + flash quand l'ancre est VISIBLE (les sections inactives restent montées
     mais display:none — getClientRects() vide tant que la bascule n'a pas peint). */
  const jumpTo = (step) => {
    setSection(SECTION_FOR_TARGET[step.targetId] || 'images');
    let tries = 0;
    const attempt = () => {
      const el = document.getElementById(step.targetId);
      if (!el || el.getClientRects().length === 0) {
        if (tries++ < 20) requestAnimationFrame(attempt);
        return;
      }
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      const b = el.querySelector('button:not([disabled])'); if (b) b.focus({ preventScroll: true });
      // Flash the landed-on block (gf-highlight, index.css) so the eye finds it.
      // remove + reflow restarts the animation when the same step is clicked twice.
      el.classList.remove('gf-highlight');
      void el.offsetWidth;
      el.classList.add('gf-highlight');
      window.setTimeout(() => el.classList.remove('gf-highlight'), 1500);
    };
    requestAnimationFrame(attempt);
  };
  const nextAction = () => {
    if (!nextStep) return;
    if (nextStep.id === 'caption') { ds.caption(effCaptionMode); return; }
    if (nextStep.id === 'finish' && !caps.training_visible) { ds.exportZip(); return; }
    if (nextStep.id === 'studio') { navigate(`/studio?dataset=${d.id}`); return; }
    jumpTo(nextStep);
  };
  const nextActionLabel = !nextStep ? '' : {
    reference: 'Go to reference', generate: 'Go to generation', curate: 'Review the grid',
    caption: 'Caption the kept ones',
    finish: caps.training_visible ? 'Go to training' : `Export ZIP (${kept})`,
    studio: 'Open Studio',
  }[nextStep.id];
  // Keep the inspected image in sync with poll refreshes (label/status updates).
  const viewImgLive = viewImg ? {
    ...(images.find((i) => i.id === viewImg.id) || viewImg),
    _rescueReviewPreview: !!viewImg._rescueReviewPreview,
  } : null;
  const viewImgImproving = viewImgLive ? images.some((image) => (
    image.derivation_kind === 'klein_image_improve'
      && image.parent_image_id === viewImgLive.id
      && image.status === 'pending'
  )) : false;
  const viewImgImprovementReady = viewImgLive ? images.some((image) => (
    image.derivation_kind === 'klein_image_improve'
      && image.parent_image_id === viewImgLive.id
      && image.status === 'pending'
      && !!image.filename
  )) : false;
  // A candidate (manual improve, small-image rescue) can be judged NEXT TO the
  // original it was made from. Resolved here because the parent is another row
  // of the same payload, which the lightbox never receives.
  const viewImgComparison = viewImgLive
    ? describeDerivedComparison(viewImgLive, images)
    : null;
  const canImproveViewImg = !!viewImgLive
    && !viewImgLive._rescueReviewPreview
    && !isSmallImageRescueRow(viewImgLive)
    && viewImgLive.derivation_kind !== 'klein_image_improve';
  // 📷 Eligibility decided HERE like improve's, so an ineligible picture shows
  // no button at all. The rescue preview is excluded for the same reason it is
  // everywhere: it is half of a Curation decision, not a library picture.
  const canCameraViewImg = !!viewImgLive
    && !viewImgLive._rescueReviewPreview
    && !isSmallImageRescueRow(viewImgLive)
    && datasetCameraRefusal(viewImgLive) === null;

  // Import to bank — the reverse of promoting bank images into a dataset. Both
  // choices retain Dataset-owned metadata; the default restores compatible
  // final-file technical analysis, while Start fresh skips only prior reuse.
  // Either way the images are COPIED into a bank-owned folder, never shared.
  const importToBank = async ({ name, preserveAnalysis }) => {
    const result = await postJson(datasetToBankUrl(),
      datasetToBankRequest(d.id, name, preserveAnalysis));
    if (!result?.ok) throw new Error(result?.error || 'Could not create the bank');
    toast.success(`Importing ${kept} image(s) into the bank — copying in the background`);
    // The bank page picks its open bank from localStorage (it has no :id route),
    // so preselect the new one before navigating rather than landing on the list.
    try { localStorage.setItem('bankCurrentId', String(result.id)); } catch { /* ignore */ }
    setImportToBankOpen(false);
    navigate('/bank');
  };

  // Export ZIP — shared by the header CTA and the Import & export row.
  // Guard-rails: untriaged images are silently EXCLUDED from the zip. Missing
  // captions are a REFUSAL YOU CAN OVERRIDE, never a wall: a Style set with no
  // captions used to be a hard toast.error with no way past it, which blocked a
  // legitimate need — getting the bare images out to caption them in another
  // tool (reported by Qeeyana on Reddit). The reason for the guard-rail is now
  // said out loud, cancelling still walks you to the captions, and the way back
  // in is named so the trip has an end.
  const exportZipGuarded = () => {
    if (triage && !window.confirm(`${triage} image(s) still await triage (✓/✕) and will NOT be in the ZIP. Export anyway?`)) return;
    if (isStyle && keptUncaptioned) {
      const ok = window.confirm(
        `${keptUncaptioned} kept image(s) have no caption.\n\n`
        + 'A Style LoRA learns "everything the captions do NOT name", so an empty '
        + '.txt teaches it nothing — that is why this is normally blocked, and the '
        + 'ZIP will contain those empty files.\n\n'
        + 'OK to export anyway — e.g. to caption the images in another tool and '
        + 'bring the .txt files back with 📦 Import dataset, which drops them onto '
        + 'these same images.\n'
        + 'Cancel to caption them here instead.');
      if (!ok) { jumpTo({ targetId: 'gf-captions' }); return; }
      ds.exportZip();
      return;
    }
    if (keptUncaptioned && !window.confirm(`${keptUncaptioned} kept image(s) without a caption (trigger only). Export anyway?`)) return;
    ds.exportZip();
  };
  // The folder lives on the machine running the app, so a browser file-picker
  // can't reach it. Try the server's native "choose a folder" dialog first;
  // when the server has no desktop (LAN/tablet, Linux/vast.ai) fall back to the
  // in-app folder browser.
  const importFolderPrompt = async () => {
    const r = await pickNativeFolder();
    if (r.available) {
      if (r.path) ds.importDatasetFolder(r.path);  // r.cancelled → user backed out
    } else {
      setFolderBrowseOpen(true);
    }
  };

  // Amber "in progress" banner text. Captioning keeps its richer live count (derived
  // from the images themselves). Otherwise, when a server-side batch is running
  // (restored from ds.activity after a reload too), name it and show done/total —
  // e.g. "Scanning for watermarks… 12/64". CPU passes (face analysis, watermark
  // clean) don't pause ComfyUI, so their note omits that claim.
  const act = ds.activity;
  const importBusy = isDatasetImportBlocked({ localBusy: ds.localBusy, activity: act });
  // Which activities actually hold ComfyUI, and therefore fight the import's
  // auto head-crop for the exclusive GPU vision window. This used to name
  // 'generate' by hand; when the import gate above widened to the whole queue
  // lane, an import started during an ✨ improve batch opened, ran, and died on
  // a 503 "GPU busy" at the crop. `holdsLocalGpu` now answers for every
  // queue-lane kind, and fails safe as local on an unknown engine.
  const visionImportBusy = holdsLocalGpu(act);
  const activityBanner = ds.captioning
    ? `${act?.detail || `Captioning in progress — ${keptCaptioned}/${kept} captioned…`} ComfyUI is paused.`
    : (() => {
        if (act) {
          const prog = act.total ? ` ${act.done}/${act.total}` : '';
          // Passes that DON'T claim "ComfyUI is paused": the CPU ones, plus
          // 'generate' (engine-dependent — Nano Banana / ChatGPT don't touch
          // ComfyUI, and the Klein case is obvious from the tiles appearing).
          // 'analyze_faces' is CPU by default and then claims nothing — but it
          // has a GPU lane (face_scoring.device), and THAT one really does hold
          // the exclusive window. The pass advertises its engine so this line
          // never has to guess.
          const cpu = (act.kind === 'analyze_faces' && act.engine !== 'cuda')
            || (act.kind === 'watermark_clean' && !String(act.detail || '').includes('GPU'))
            || act.kind === 'generate'
            // Same reasoning as 'generate': the improve batch feeds ComfyUI, and
            // the candidates appearing in the grid say so better than a claim.
            || act.kind === 'improve'
            // Editing the reference is an API call (ChatGPT / Nano Banana) — no GPU,
            // ComfyUI is never touched, so never claim it is paused.
            || act.kind === 'edit_reference'
            // Dataset → Bank is a reserved filesystem copy. It blocks edits to
            // keep one coherent source generation, but does not touch the GPU.
            || act.kind === 'bank_export'
            || act.kind === 'bank_import'
            || act.kind === 'training_export'
            || act.kind === 'backup';
          const label = {
            watermark_detect: `Scanning for watermarks…${prog}`,
            watermark_clean: `Cleaning watermarks…${prog}`,
            text_detect: `Reading burned-in text…${prog}`,
            caption: `Captioning…${prog}`,
            recaption: `Re-captioning…${prog}`,
            analyze_faces: `Analyzing faces…${prog}`,
            classify: `Classifying framing…${prog}`,
            generate: `Generating variations…${prog}`,
            improve: `Queuing improvements…${prog}`,
            edit_reference: 'Editing reference…',
            bank_export: `Copying into a Bank…${prog}`,
            bank_import: `Copying images from a Bank…${prog}`,
            training_export: 'Freezing the Dataset for training…',
            backup: `Creating portable backup…${prog}`,
          }[act.kind];
          if (label) {
            // Copy/freeze details are stable phase names, while done/total lives
            // beside them. Prefer the count-aware labels so progress stays visible.
            const detailed = ['bank_export', 'bank_import', 'training_export']
              .includes(act.kind) ? label : (act.detail || label);
            return `${detailed}${cpu ? '' : ' ComfyUI is paused during the pass.'}`;
          }
        }
        return 'GPU processing in progress (analysis / cropping / captioning)… ComfyUI is paused during the pass.';
      })();

  // ── Sidebar : pastilles par section — ambre quand une action attend l'utilisateur,
  //    indigo pulsé quand des générations tournent, neutre pour l'info « à faire ».
  // The engine line for the caption pass that just ran — empty string (falsy) when
  // no pass ran in this session, when the backend sent no counts, or when the run
  // belongs to ANOTHER dataset: the id check is what stops this line from describing
  // someone else's pass after a dataset switch.
  const lastCaptionEngines = ds.lastCaptionRun && ds.lastCaptionRun.datasetId === d.id
    ? captionEnginesSummary(ds.lastCaptionRun.engines) : '';

  const navBadges = {
    images: triage > 0
      ? { n: triage, tone: 'amber', srLabel: `${triage} image(s) awaiting keep/reject` } : null,
    add: pending > 0
      ? { n: pending, tone: 'indigo', pulse: true, srLabel: `${pending} generation(s) in progress` } : null,
    curation: watermarkDetected + rescueReviewCount > 0
      ? {
          n: watermarkDetected + rescueReviewCount,
          tone: 'amber',
          srLabel: `${watermarkDetected} watermark(s) and ${rescueReviewCount} Klein rescue pair(s) to review`,
        } : null,
    captions: (!isStyle && (d.caption_leak?.leaking ?? 0) > 0)
      ? { n: d.caption_leak.leaking, tone: 'amber', srLabel: `${d.caption_leak.leaking} caption(s) leaking` }
      : keptUncaptioned > 0
        ? { n: keptUncaptioned, tone: 'subtle', srLabel: `${keptUncaptioned} kept image(s) without a caption` } : null,
    export: null,
    training: null,
  };

  const activePanels = getWorkspacePanels(section, navContext);

  const panelNavItem = (sectionId, destination, chip = false) => {
    const isActive = sectionId === section && destination.id === panel;
    const className = chip
      ? `min-h-10 shrink-0 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs ${
          isActive
            ? 'border-indigo-400/60 bg-indigo-500/15 text-indigo-100'
            : 'border-border text-content-subtle hover:text-content'}`
      : `relative w-full rounded-md py-1.5 pl-8 pr-3 text-left text-xs ${
          isActive
            ? 'bg-indigo-500/10 text-indigo-200'
            : 'text-content-subtle hover:bg-surface hover:text-content-muted'}`;
    return (
      <button type="button"
        onClick={() => navigateToPanel(sectionId, destination.id)}
        aria-current={isActive ? 'location' : undefined}
        data-mobile-panel={chip ? destination.id : undefined}
        className={className}>
        {!chip && isActive && (
          <span aria-hidden className="absolute bottom-1.5 left-4 top-1.5 w-px rounded bg-indigo-400" />
        )}
        {destination.title}
      </button>
    );
  };

  // Un item de la sidebar : rail vertical desktop (chip=false) ou chip du bandeau
  // horizontal mobile (chip=true) — mêmes classes que la sidebar de Settings.
  const navItem = (s, chip) => {
    const isActive = s.id === section;
    const base = chip
      ? `flex min-h-10 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${
          isActive ? 'border-border-strong bg-surface-raised text-content' : 'border-border text-content-muted hover:text-content'}`
      : `relative flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm font-medium ${
          isActive ? 'bg-surface-raised text-content' : 'text-content-muted hover:bg-surface hover:text-content'}`;
    return (
      <button type="button" onClick={() => setSection(s.id)}
        aria-current={isActive ? 'page' : undefined}
        aria-expanded={isActive}
        aria-controls={isActive
          ? `${chip ? 'dataset-mobile-panels' : 'dataset-nav-panels'}-${s.id}`
          : undefined}
        data-mobile-section={chip ? s.id : undefined}
        className={base}>
        {!chip && isActive && (
          <span aria-hidden className="absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded bg-gradient-primary" />
        )}
        <s.icon aria-hidden="true" className="h-4 w-4 shrink-0" />
        <span>{s.title}</span>
        <NavBadge badge={navBadges[s.id]} />
        {!chip && <span aria-hidden className="text-content-subtle text-[0.625rem]">{isActive ? '▾' : '▸'}</span>}
      </button>
    );
  };

  const sectionMeta = Object.fromEntries(WORKSPACE_SECTIONS.map((s) => [s.id, s]));
  const heading = (id) => {
    const s = sectionMeta[id];
    return <SectionHeading id={`ds-section-${id}-heading`} eyebrow={s.eyebrow} title={s.title}
      badge={<HelpBadge topic={`workspace-${id}`} />}
      description={isStyle && id === 'add'
        ? 'Import varied images that share the aesthetic; subject and scene diversity keep the Style LoRA composable.'
        : isConceptual && s.conceptDescription ? s.conceptDescription : s.description} />;
  };
  // Sections inactives : montées mais masquées (display:none) — les polls et
  // états internes survivent au changement de section (le poll 10 s du
  // TrainingPanel fait AVANCER la file d'entraînement côté serveur, il ne doit
  // jamais s'arrêter parce qu'on regarde la grille ; idem sélection de la
  // grille, panneaux dépliés, catalogue de variations).
  const sectionCls = (id) => (section === id ? 'flex flex-col gap-3' : 'hidden');

  // Discreet entry point kept in "Add images" after the scraper moved to its own
  // 🕸 Scrape destination — preserves the build-flow's discoverability without the
  // long accordion. Navigates to the Scrape section and focuses its gallery-URL input.
  const scrapeLink = (
    <button type="button" onClick={() => navigateToPanel('scrape', 'scan')}
      className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-left text-content-muted hover:text-content hover:bg-surface-raised transition-colors">
      <Globe aria-hidden="true" className="h-4 w-4" />
      <span className="text-sm font-medium">Scrape images from the web</span>
      <span className="text-content-subtle text-[0.6875rem]">scan a gallery URL, pick images, import full-frame</span>
      <span aria-hidden className="ml-auto text-content-subtle">→</span>
    </button>
  );

  // Third source next to the dropzone and the scraper: an already-triaged bank.
  // The server's promote path does the copying (normalize + dedup vs THIS
  // dataset) in a background job — we only pick the bank and refresh at the end.
  const bankImport = (
    <BankImportPanel datasetId={d.id} disabled={importBusy}
      onImported={() => ds.refresh()} />
  );

  return (
    <div className="flex flex-col gap-3">
      {/* ---- Header : identité du dataset + UNE action primaire (Export ZIP).
           Les actions secondaires de PARAMÉTRAGE (settings, fidélité) vivent dans
           le menu « ⋯ More » ; les actions de DONNÉES (backup, import-fusion,
           publish) vivent dans la section « Import & export » de la sidebar. ---- */}
      {/* relative z-30 : le header est un flex item ; sans stacking-context propre,
          le z-20 du menu « ⋯ More » resterait piégé sous les frères plus bas. */}
      <div data-probe-chrome="header" className="relative z-30 flex items-center gap-x-2 gap-y-1 flex-wrap">
        <button type="button" onClick={onBack}
          className="min-h-10 lg:min-h-0 flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-sm transition-colors">
          <ArrowLeft aria-hidden="true" className="h-4 w-4" /> Datasets
        </button>
        <h1 className="text-content font-bold">{d.name}</h1>
        {isStyle ? (
          <span title="This Style LoRA is always active when loaded; adjust its LoRA weight to control the effect."
            className="flex items-center gap-1 px-2 py-0.5 rounded-lg border border-cyan-400/40 bg-cyan-500/10 text-cyan-200 text-[0.6875rem]">
            always-on style · no trigger
          </span>
        ) : (
          <button type="button"
            onClick={() => { try { navigator.clipboard.writeText(d.trigger_word || ''); } catch { /* ignore */ } }}
            title="Copy the trigger word (to put in your prompts)"
            className="min-h-10 lg:min-h-0 flex items-center gap-1 px-2 py-0.5 rounded-lg border border-indigo-400/40 bg-indigo-500/10 text-[0.6875rem]">
            <span className="text-content-subtle">trigger:</span>
            <code className="text-indigo-300 font-semibold">{d.trigger_word || '—'}</code>
            <Copy aria-hidden="true" className="h-3 w-3 text-content-subtle" />
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button type="button" disabled={!kept} onClick={exportZipGuarded}
            className="min-h-10 lg:min-h-0 px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
            <Download aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Export ZIP ({kept})
          </button>
          {/* summary en display:flex → pas de marqueur natif ; les items restent
              montés en permanence (details ne fait que masquer l'affichage). */}
          <details className="relative">
            <summary
              title="More dataset actions — edit settings, body fidelity"
              className="min-h-10 lg:min-h-0 flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-sm cursor-pointer select-none">
              ⋯ More
            </summary>
            <div data-probe-chrome="more-menu" data-probe-panel="more-menu"
              className="absolute right-0 top-full mt-1 z-20 w-72 rounded-lg border border-border bg-surface-overlay shadow-xl p-1.5 flex flex-col gap-0.5">
              <button type="button" onClick={() => setSettingsOpen(true)}
                title={isStyle ? 'Edit the Style dataset name and review its always-on behavior.' : 'Edit the dataset name, trigger word, and (for concept datasets) the concept description that drives the caption avoid-list.'}
                className={MENU_ITEM}>
                <Settings aria-hidden="true" className="h-4 w-4" /> Edit settings
                <span className="ml-auto text-content-subtle text-[0.625rem]">
                  {isStyle ? 'name · always-on' : `name · trigger${isConcept ? ' · concept' : ''}`}
                </span>
              </button>
              {!isConceptual && (
                <button type="button" disabled={ds.busy}
                  onClick={() => ds.setDatasetFidelity?.(bodyFid ? 'face' : 'body')}
                  title={bodyFid
                    ? 'Body fidelity ON: captions also omit tattoos/scars/marks (they bind to the trigger), composition targets more bust/body shots, imports keep the full frame by default. Click to go back to face-only.'
                    : 'Face-only fidelity (default): the LoRA learns the face; body shape follows the prompt. Click for FULL-BODY fidelity (body shape & marks bind to the trigger too).'}
                  className={`${MENU_ITEM} ${bodyFid ? 'text-emerald-300' : ''}`}>
                  <PersonStanding aria-hidden="true" className="h-4 w-4" /> Body fidelity
                  <span className={`ml-auto text-[0.625rem] ${bodyFid ? 'text-emerald-300 font-semibold' : 'text-content-subtle'}`}>
                    {bodyFid ? '✓ on' : 'off'}
                  </span>
                </button>
              )}
            </div>
          </details>
        </div>
      </div>

      {/* Where the images actually are. Shown here, at the top, because the
          question "where is this on disk?" is asked before anything else — and
          because the answer used to be findable only by hand, which is how a
          dataset folder ended up pasted into a bank. */}
      <DatasetFolderNote path={d.storage_path} />

      {/* Two-column workspace: the persistent section sidebar on the left (with the
          guided Progress checklist below it for character datasets), the ACTIVE
          section's content on the right. On mobile the sidebar folds into a
          horizontal chip rail — same responsive pattern as the Settings page. */}
      <div className="lg:grid lg:grid-cols-[15rem_minmax(0,1fr)] lg:gap-4 lg:items-start">
        <aside>
          {/* Mobile: horizontal chip rail.

              `relative` is NOT decoration — without it the whole page renders
              at ~73% on a phone. `overflow-x-auto` clips a descendant only when
              the scroller is also its containing block, and a static box never
              is. The NavBadge counts carry an `.sr-only` label, which Tailwind
              implements as `position: absolute` with no offsets: it therefore
              resolves against the document, keeps its static position out at
              the far end of a rail 1123 px wide, and escapes the clip. The
              document then measures ~598 px against a 440 px viewport, mobile
              Safari shrinks the page to fit, and every bar on screen — header
              included — draws at 73% of the screen with dead space beside it.
              Nothing overflows visibly, because the escapee is a 1 px box no
              one can see. */}
          <nav aria-label="Dataset sections" data-probe-chrome="sections"
            className="relative -mx-4 overflow-x-auto px-4 pb-1 lg:hidden">
            <ul className="m-0 flex list-none gap-2 p-0">
              {WORKSPACE_SECTIONS.map((s) => <li key={s.id}>{navItem(s, true)}</li>)}
            </ul>
          </nav>
          {activePanels.length > 0 && (
            /* On a phone held sideways (≤ 500 px of fold) this second rail is the
               one thing to give back: the section buttons above still reach every
               panel, this rail only jumps inside the open one. */
            <nav aria-label={`${sectionMeta[section].title} destinations`} data-probe-chrome="destinations"
              className="relative -mx-4 overflow-x-auto px-4 pb-1 lg:hidden [@media(max-height:500px)]:hidden">
              <ul id={`dataset-mobile-panels-${section}`} className="m-0 flex list-none gap-2 p-0">
                {activePanels.map((destination) => (
                  <li key={destination.id}>{panelNavItem(section, destination, true)}</li>
                ))}
              </ul>
            </nav>
          )}
          {/* Desktop: sticky rail + guided progress below it */}
          <div data-probe-panel="sections-rail" className="hidden lg:sticky lg:top-20 lg:flex lg:flex-col lg:gap-3">
            <nav aria-label="Dataset sections">
              <p className="m-0 px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">Dataset</p>
              <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
                {WORKSPACE_SECTIONS.map((s) => {
                  const isActive = s.id === section;
                  const destinations = isActive ? getWorkspacePanels(s.id, navContext) : [];
                  return (
                    <li key={s.id}>
                      {navItem(s, false)}
                      {isActive && (
                        <ul id={`dataset-nav-panels-${s.id}`}
                          className="m-0 ml-4 flex list-none flex-col gap-0.5 border-l border-border py-1 pl-1 p-0">
                          {destinations.map((destination) => (
                            <li key={destination.id}>{panelNavItem(s.id, destination, false)}</li>
                          ))}
                        </ul>
                      )}
                    </li>
                  );
                })}
              </ul>
            </nav>
            {!isConceptual && (
              <GuidedChecklist steps={steps} currentId={nextStep ? nextStep.id : null} onJump={jumpTo} />
            )}
          </div>
        </aside>

        <div className="flex flex-col gap-3 min-w-0 mt-1 lg:mt-0">
          {/* ---- Bandeaux GLOBAUX : visibles quelle que soit la section active
               (une passe GPU ou un batch de générations concernent tout l'écran). ---- */}
          {!isConceptual && (
            <NextStepCard step={nextStep} trainMode={!!caps.training_visible} busy={ds.busy}
              totalImages={images.length} onAction={nextAction} actionLabel={nextActionLabel} />
          )}

          {ds.busy && (
            <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2">
              <span className="inline-block w-4 h-4 border-2 border-amber-400/40 border-t-amber-400 rounded-full animate-spin" aria-hidden />
              <span className="text-content text-sm">{activityBanner}</span>
              {(act?.kind === 'caption' || act?.kind === 'recaption') && (
                <button type="button" onClick={ds.cancelCaption} disabled={!!act?.cancelling}
                  title="Stops after the current image finishes — captions already written are kept; the rest stays uncaptioned."
                  className="ml-auto shrink-0 px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-bold disabled:opacity-50 disabled:cursor-not-allowed">
                  {act?.cancelling ? 'Stopping…' : '⏹ Stop'}
                </button>
              )}
              {/* Same seam, same promise, for the watermark scan: it was the one
                  long dataset pass with no way out but closing the tab. */}
              {act?.kind === 'watermark_detect' && (
                <button id="ds-watermark-scan-stop" type="button"
                  onClick={ds.cancelWatermarkScan} disabled={!!act?.cancelling}
                  title="Stops after the current image finishes — every watermark already found is kept; run 🧽 Find watermarks again to finish the rest."
                  className="ml-auto shrink-0 px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-bold disabled:opacity-50 disabled:cursor-not-allowed">
                  {act?.cancelling ? 'Stopping…' : '⏹ Stop'}
                </button>
              )}
              {/* …and for the 🔤 text scan, which reads whole banks of pages. */}
              {act?.kind === 'text_detect' && (
                <button id="ds-text-scan-stop" type="button"
                  onClick={ds.cancelTextScan} disabled={!!act?.cancelling}
                  title="Stops after the current image finishes — every zone already found is kept; run 🔤 Find text again to finish the rest."
                  className="ml-auto shrink-0 px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-bold disabled:opacity-50 disabled:cursor-not-allowed">
                  {act?.cancelling ? 'Stopping…' : '⏹ Stop'}
                </button>
              )}
            </div>
          )}

          {/* Stop is also the batch's Stop: a server-side ✨ improve run keeps feeding
              the queue, so the banner must stay reachable even between two waves
              (a moment when nothing is in flight). */}
          {(pending > 0 || act?.kind === 'improve') && (
            <div className="flex items-center gap-3 rounded-lg border-2 border-indigo-400/60 bg-indigo-500/15 px-3 py-2.5">
              <span className="animate-pulse text-lg" aria-hidden>⏳</span>
              <div className="flex flex-col">
                <span className="text-content text-sm font-semibold">
                  {act?.kind === 'improve' && act.total
                    ? `${act.done}/${act.total} improvement(s) queued — ${pending} generating…`
                    : `${pending} generation(s) in progress…`}
                </span>
                <span className="text-content-subtle text-[0.6875rem]">
                  {act?.kind === 'improve'
                    ? 'Runs on the server — you can close this tab. Stop ends the whole batch, not just what is in flight.'
                    : 'First results look wrong? Stop now — the remaining API calls are skipped (not billed).'}
                </span>
              </div>
              <button type="button" onClick={stopGeneration}
                disabled={isStopGenerationBlocked({
                  busy: ds.busy, activity: act, cancelling: stoppingGeneration })}
                title="Cancels every generation still in flight (and stops a running improvement batch); finished images stay."
                className="ml-auto shrink-0 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-bold disabled:opacity-40">
                {stoppingGeneration || (act?.cancelling && act?.kind === 'improve')
                  ? 'Stopping…' : '⏹ Stop generation'}
              </button>
            </div>
          )}

          {/* Provider refusals, counted and named ONCE — because the alternative
              is what shipped before: N missing images, each explained only by a
              9px clamped line inside its own failed tile, which reads as "the app
              generated fewer images than I asked for". Shown whenever refused rows
              exist (not just right after a batch): it is derived from stored rows,
              so it survives a reload and clears when they are purged or retried.
              Collapsed by default — the count is the headline, the policy detail
              is one tap away and does not eat a 400 px screen. */}
          {refusalNote && (
            <div className="rounded-lg border-2 border-amber-400/60 bg-amber-500/10 px-3 py-2.5">
              <div className="flex items-start gap-2">
                <Ban aria-hidden="true" className="h-4 w-4 shrink-0 mt-0.5 text-amber-300" />
                <div className="min-w-0 flex flex-col gap-1">
                  <span className="text-content text-sm font-semibold break-words">
                    {refusalNote}
                  </span>
                  {failureHeadline(outcome) && (
                    <span className="text-content-subtle text-[0.6875rem] break-words">
                      {failureHeadline(outcome)}
                    </span>
                  )}
                  <details className="text-content-subtle text-[0.6875rem]">
                    <summary className="cursor-pointer text-amber-300">
                      Why, and what this does and does not mean
                    </summary>
                    <div className="mt-1 flex flex-col gap-1 break-words">
                      <p className="m-0">
                        {outcome.refusedEngines.includes('nanobanana')
                          ? 'Gemini (Nano Banana) screens the image it just produced'
                          : 'The provider screens the image it just produced'}
                        {' '}and answers with no image when that screen trips.
                        The screen is <strong>not configurable</strong>: the safety
                        settings the API exposes act on the prompt, not on the
                        returned image, so LDS has no switch to turn it off.
                      </p>
                      <p className="m-0">
                        It refuses ordinary requests too, and it is not consistent —
                        the same prompt can pass on one attempt and be refused on the
                        next. That is why nothing here tells you to reword or retry:
                        neither is a fix, and we will not pretend otherwise.
                      </p>
                      <p className="m-0">
                        Google&apos;s usage policy does not allow adult content on
                        this engine, and repeated attempts can lead to restrictions
                        on your Google account. LDS never sends NSFW variations to an
                        API engine — those run only on the local Klein engine.
                      </p>
                    </div>
                  </details>
                </div>
              </div>
            </div>
          )}

          {/* ============ 🖼️ Images — la grille : triage ✓/✕, filtres, tri auto. */}
          <div className={sectionCls('images')}>
            {heading('images')}
            <p className="m-0 text-content-subtle text-[0.75rem] tabular-nums">
              {rescueGridImages.length} image(s) · {kept} kept
              {triage > 0 ? <> · <span className="text-amber-300">{triage} awaiting ✓/✕</span></> : ''}
              {rescueReviewCount > 0
                ? <> · <span className="text-indigo-300">{rescueReviewCount} Klein rescue pair(s) in Curation</span></>
                : ''}
              {kept > 0 ? ` · ${keptCaptioned}/${kept} captioned` : ''}
              {watermarkDetected > 0 ? ` · ${watermarkDetected} watermark(s) flagged` : ''}
            </p>
            <div id="gf-images" className="scroll-mt-20 flex flex-col gap-2">
              {rescueGridImages.length > 0 && (
                // One wrapping row at 400 px: the decision chips flow, the Sort
                // control drops onto its own line instead of forcing a scrollbar.
                <div data-probe-chrome="grid-toolbar" className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                  <GridStatusFilter value={statusFilter} counts={statusCounts}
                    onChange={setStatusFilter} />
                  <GridSortSelect value={gridSort} images={rescueGridImages}
                    onChange={setGridSort} />
                </div>
              )}
              {filtersActive && (
                <GridFilterBar excludes={excludeTags} includes={includeTags}
                  statusLabel={statusFilterActive ? statusFilterLabel : ''}
                  coverageLabel={coverageFilter ? coverageFilter.label : ''}
                  shown={gridImages.length} total={rescueGridImages.length}
                  onRemoveExclude={toggleExclude} onRemoveInclude={toggleInclude}
                  onRemoveStatus={() => setStatusFilter(DEFAULT_GRID_STATUS_FILTER)}
                  onRemoveCoverage={() => setCoverageFilter(null)}
                  onClearAll={clearFilters} />
              )}
              {filtersActive && gridImages.length === 0 ? (
                // Filtered down to nothing: say so plainly (the grid's own "no images"
                // empty-state would read as "everything's gone", which would be a lie).
                <p className="rounded-lg border border-border bg-surface px-3 py-4 text-center text-content-subtle text-sm">
                  No images match the active filters —{' '}
                  <button type="button" onClick={clearFilters} className="underline hover:text-content">clear all</button>{' '}
                  to see all {rescueGridImages.length} again.
                </p>
              ) : (
                <DatasetGrid images={gridImages} datasetId={d.id} onStatus={ds.setStatus} onCaption={ds.setCaption}
                  onCrop={setCropImg} onDelete={ds.deleteImage}
                  onMirror={ds.mirrorImage} mirroringIds={ds.mirroringIds}
                  onScoreFace={ds.scoreFace} scoringFaceIds={ds.scoringFaceIds}
                  onRegenerate={(id, loraStrength, prompt, opts) => ds.regenerate(id, loraStrength, prompt, opts)}
                  onReimprove={ds.reimproveImage} onView={setViewImg}
                  /* ⟨ / ⟩ cross page boundaries, so the page underneath follows:
                     closing the lightbox must not leave you on a page that no
                     longer holds the image you were looking at. */
                  viewingImageId={viewImg?.id ?? null}
                  onBatch={ds.batchImages} busy={ds.busy}
                  improveBusy={ds.improveBusy} generateBusy={ds.generationBusy} curationBusy={ds.curationBusy}
                  onBulkBusyChange={setGridBulkBusy}
                  onImproveBatch={ds.improveBatch} activity={act}
                          subjectType={d.subject_type || 'human'}
                  eligibilityImages={images}
                  nonces={ds.nonces} faceThresholds={d.face_thresholds} datasetKind={d.kind || 'character'}
                  faceScoringBlocked={d.face_scoring_blocked}
                  dualCaptions={Boolean(d.dual_captions)} />
              )}
            </div>
          </div>

          {/* ============ 📸 Add images — constituer le dataset. Concept : sources
               scrapées + import brut. Personnage : référence puis génération/import. */}
          <div className={sectionCls('add')}>
            {heading('add')}
            {isConceptual ? (
              // Concept : pas de photo de référence ni de générateur — on peuple le
              // dataset par upload manuel et/ou via le scraper, qui vit désormais dans
              // sa propre section 🕸 Scrape (lien discret ci-dessous).
              <div id="gf-reference" className="scroll-mt-20 flex flex-col gap-2">
                {scrapeLink}
                <div id="ds-add-import" tabIndex={-1} className="scroll-mt-20">
                  <ImportDropzone onImport={(f) => ds.importFiles(f)} busy={importBusy} visionBusy={visionImportBusy} />
                </div>
                {bankImport}
              </div>
            ) : (
              <>
                <div id="gf-reference" className="scroll-mt-20">
                  <div id="ds-add-reference" tabIndex={-1} className="scroll-mt-20 flex flex-col gap-1">
                    <span className="text-content-subtle text-[0.6875rem]">
                      one clear photo of the face — every generated variation starts from it
                    </span>
                    <ReferencePanel refFilename={d.ref_filename} datasetId={d.id} onSetRef={ds.setRef}
                      onCropRef={() => setRefCrop(true)} onEditRef={() => setRefEdit(true)} busy={ds.busy} importBusy={importBusy} visionBusy={visionImportBusy} nonce={ds.refNonce}
                      extraRefs={d.ref_extra_filenames || []}
                      onAddExtraRef={ds.addExtraRef} onRemoveExtraRef={ds.removeExtraRef}
                      onCropExtraRef={(fn) => setExtraRefCrop(fn)}
                      subjectType={d.subject_type || 'human'}
                      referenceEdit={d.reference_edit} />
                  </div>
                </div>

                <div id="gf-generate" className="scroll-mt-20 flex flex-col gap-2">
                  <CompositionBar composition={d.composition} upscaled={d.composition_upscaled} bodyFidelity={bodyFid} />
                  {/* The bar above counts shot types against a target and can go
                      fully green on a set that is the same pose, one outfit, one
                      light — none of which it counts. This is that second
                      question, read from the captions already written. */}
                  {/* A chip is a place to go, not only a number: clicking one
                      narrows the grid to exactly the images it counted and jumps
                      to it — the panel lives in Add images, the grid in Images,
                      and a filter applied to a screen you cannot see would look
                      like nothing happened. */}
                  <CoveragePanel datasetId={d.id} refreshKey={images.length}
                    onPick={(pick) => { setCoverageFilter(pick); setSection('images'); }} />
                  {/* Images imported WITHOUT head-crop have no shot type, so they count
                      for nothing in the bar above (the default on body-fidelity datasets:
                      a whole drag-and-drop import can leave it at 0). A crop clears the
                      stored shot type for the same reason — those pixels changed. The
                      vision pass that fills them in lives right here, under the bar
                      that shows the gap. */}
                  <ClassifyFramingButton images={images} ollama={caps.ollama} capsLoading={capsLoading}
                    busy={ds.busy} activity={act} onClassify={(n) => ds.classify(n)} />
                  <div id="ds-add-generate" tabIndex={-1} className="scroll-mt-20">
                    <Suspense fallback={null}>
                    <VariationCatalog key={`vc-${d.id}-${bodyFid}`} datasetId={d.id} busy={ds.generationBusy}
                      generating={act && act.kind === 'generate' ? act : null}
                      onGenerate={(...args) => {
                        // Guard-rail: a batch is already in flight — launching another one
                        // on top is usually an accidental double-click, not a plan.
                        if (pendingGenerations > 0 && !window.confirm(
                          `A generation batch is already running (${pendingGenerations} in flight).\n\nLaunch another one anyway?`)) return;
                        ds.generate(...args);
                      }}
                      hasRef={!!d.ref_filename} composition={d.composition} images={images}
                      bodyFidelity={bodyFid}
                      promptSuffix={d.prompt_suffix || ''}
                      promptSuffixes={d.prompt_suffixes || null}
                      onSaveSuffixes={(patch) => ds.updateSettings(patch, { quiet: true })}
                      subjectType={d.subject_type || 'human'}
                      onSaveSubjectType={(st) => ds.updateSettings({ subject_type: st }, { quiet: true })} />
                    </Suspense>
                  </div>
                  {/* Head-crop optional: ON tags framing='face' at import (I2); OFF keeps
                      the original framing so bust/body photos import as-is. Body-fidelity
                      datasets default OFF (full frames are the point) — key remounts the
                      dropzone so the default follows a fidelity switch. */}
                  <div id="ds-add-import" tabIndex={-1} className="scroll-mt-20">
                    <ImportDropzone key={`${d.id}-${bodyFid}`} onImport={(f, o) => ds.importFiles(f, o)}
                      busy={importBusy} visionBusy={visionImportBusy} cropOption defaultCrop={!bodyFid} />
                  </div>
                  {/* Scraper moved to its own 🕸 Scrape destination — keep a discreet
                      link here so the build flow still surfaces it without burying the
                      reference/generate flow under a long accordion. */}
                  {scrapeLink}
                  {bankImport}
                </div>
              </>
            )}
          </div>

          {/* ============ 🕸 Scrape — its own destination now (moved out of "Add
               images"): scan a gallery URL → pick thumbnails → import full-frame, then
               crop each tile manually (✂ on the card). One ConceptSourcesPanel serves
               every dataset kind. */}
          <div className={sectionCls('scrape')}>
            {heading('scrape')}
            <div id="ds-scrape-scan" tabIndex={-1} className="scroll-mt-20">
              <ConceptSourcesPanel key={`scraper-${d.id}`} datasetId={d.id}
                onImport={ds.scrapeImport} busy={importBusy} />
            </div>
          </div>

          {/* ============ 🧹 Curation — passes de qualité sur les images gardées :
               ressemblance faciale, watermarks (find → clean → review), purge. */}
          <div className={sectionCls('curation')}>
            {heading('curation')}
            <div id="gf-curation" className="scroll-mt-20 flex flex-col gap-2">
              <SmallImageRescueReview images={images} datasetId={d.id}
                onResolve={ds.resolveSmallImageRescue}
                onPreview={(image) => setViewImg({ ...image, _rescueReviewPreview: true })}
                nonces={ds.nonces} />
              <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
                {!isConceptual && (
                  <button id="ds-curation-face-analysis" type="button" data-workspace-focus
                    onClick={ds.analyzeFaces} disabled={faceAnalysis.disabled}
                    title={faceAnalysis.title}
                    className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border scroll-mt-20">
                    <Drama aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />
                    {ds.analyzing
                      ? `Analyzing…${act?.kind === 'analyze_faces' && act.total ? ` ${act.done}/${act.total}` : ''}`
                      : faceAnalysisLabel(d.face_scoring_scope)}
                  </button>
                )}
                {/* A greyed button whose reason lives in a tooltip is invisible on a
                    phone (no hover) — the pass must SAY why it can't run, in place.
                    When the reason is a missing install, the sentence is also the
                    way OUT: a link to the Setup step that installs it. */}
                {!isConceptual && faceAnalysis.blocked && (
                  <p className="m-0 basis-full text-sky-300/90 text-[0.6875rem]">
                    ℹ {faceAnalysis.reason}
                    {faceAnalysis.setupRoute && (
                      <> — <a href={faceAnalysis.setupRoute} className="underline">open Setup</a></>
                    )}
                  </p>
                )}
                <div id="ds-curation-watermarks" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                {/* Watermark auto-correction (V1): find overlaid site logos/URLs/usernames on
                    the kept images, then Clean them (border → crop, small off-center → LaMa
                    inpaint, on-subject → manual review). Applies to any dataset kind. */}
                <button type="button" data-workspace-focus onClick={ds.findWatermarks} disabled={ds.busy}
                  title="Scans the kept images for overlaid watermarks/logos/URLs added on top of the photo (deletes nothing)"
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  <Eraser aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />
                  {ds.watermarking
                    ? `Scanning…${act?.kind === 'watermark_detect' && act.total ? ` ${act.done}/${act.total}` : ''}`
                    : 'Find watermarks'}
                </button>
                {/* 🔤 The other detection feeding the same funnel: burned-in text
                    (speech bubbles, subtitles, captions, sound effects) read by
                    the RapidOCR engine the video lane ships — CPU only. Zones
                    land in the same mask channel, so the same Clean repaints
                    them. Greyed with the install route when the extra is out. */}
                <button type="button" data-workspace-focus onClick={() => setTextScanOpen(true)}
                  disabled={ds.busy || !caps.video_text}
                  title={caps.video_text
                    ? 'Opens the launch window: try a sample, tune the sensitivity, then scan — zones land in the mask 🧽 Clean repaints (deletes nothing, CPU only)'
                    : 'Reads burned-in text on the kept images. Install "Burned-in text" from Setup first.'}
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  🔤 {ds.textScanning
                    ? `Reading…${act?.kind === 'text_detect' && act.total ? ` ${act.done}/${act.total}` : ''}`
                    : 'Find text…'}
                </button>
                {textScanOpen && (
                  <TextScanDialog
                    onClose={() => setTextScanOpen(false)}
                    onLaunch={(opts) => ds.findText(opts)}
                    toRead={textToRead}
                    rereadable={keptForText.length}
                    sensitivity={caps.text_scan_score_min}
                    live={ds.busy}
                    /* The window polls this dataset's /text/preview itself —
                       reading the payload here only refreshed when the run
                       returned, leaving the strip empty for the whole scan. */
                    datasetId={d.id} />
                )}
                <HelpBadge topic="action-watermark-clean" />
                {watermarkDetected > 0 && (
                  <>
                  {/* Inpaint engine: LaMa (fast, non-generative) vs Klein (masked
                      Flux.2 inpaint — better on complex texture AND makes on-subject
                      marks actionable, but GPU + slower). Klein is greyed until ComfyUI
                      + the Klein models are ready (caps.watermark_klein). */}
                  <div role="group" aria-label="Watermark inpaint method"
                    className="flex items-center rounded-lg border border-border bg-surface p-0.5 text-xs">
                    <button type="button" aria-pressed={watermarkMethod === 'lama'}
                      onClick={() => setWatermarkMethod('lama')} disabled={ds.busy}
                      title="LaMa: fast, non-generative. Crops border marks, repaints small off-center marks; on-subject marks go to manual review."
                      className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkMethod === 'lama'
                        ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                      LaMa <span className="font-normal opacity-70">fast</span>
                    </button>
                    <button type="button" aria-pressed={watermarkMethod === 'klein'}
                      onClick={() => setWatermarkMethod('klein')} disabled={ds.busy || !caps.watermark_klein}
                      title={caps.watermark_klein
                        ? 'Klein: masked Flux.2 inpaint (crop-and-stitch). Better on skin/fabric/busy backgrounds and can clean marks ON the subject. Uses the GPU via ComfyUI — slower.'
                        : (localEngineUnavailableReason('klein', caps)
                          || 'Klein inpaint needs ComfyUI running + the Klein models installed (Setup ▸ ComfyUI).')}
                      className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkMethod === 'klein'
                        ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                      Klein <span className="font-normal opacity-70">quality</span>
                    </button>
                  </div>
                  {/* 🔤/🚩 WHAT to clean — only offered once Find text flagged
                      something (no text page = the three choices collapse into
                      one, a dead dial). Same three choices, same words as the
                      bank panel. */}
                  {textFlaggedDetected > 0 && (
                    <div role="group" aria-label="What to clean"
                      className="flex items-center rounded-lg border border-border bg-surface p-0.5 text-xs">
                      <button type="button" aria-pressed={watermarkTarget === 'all'}
                        onClick={() => setWatermarkTarget('all')} disabled={ds.busy}
                        title="Repaint every flagged page — text and watermarks alike."
                        className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkTarget === 'all'
                          ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                        Both
                      </button>
                      <button type="button" aria-pressed={watermarkTarget === 'text'}
                        onClick={() => setWatermarkTarget('text')} disabled={ds.busy}
                        title="Only pages 🔤 Find text flagged. A page carrying both a watermark and text counts here — one page is never split between two runs."
                        className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkTarget === 'text'
                          ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                        🔤 Text
                      </button>
                      <button type="button" aria-pressed={watermarkTarget === 'watermark'}
                        onClick={() => setWatermarkTarget('watermark')} disabled={ds.busy}
                        title="Only pages 🚩 flagged with no text flag on them."
                        className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkTarget === 'watermark'
                          ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                        🚩 Marks
                      </button>
                    </div>
                  )}
                  {/* A clean OVERWRITES the image, so the model that repaints it is
                      the one lane whose swap can never be spotted afterwards. It is
                      the dataset's model, like improve and generation — named here,
                      changeable here. basis-full: its own row rather than a squeezed
                      column at 400 px. */}
                  {watermarkMethod === 'klein' && (
                    <KleinModelSetting datasetId={d.id} className="basis-full" />
                  )}
                  {/* Allow auto-crop: the SAME persisted preference as Settings ▸ Watermark
                      inpainting (write-through). Off → a border mark is repainted (LaMa/
                      Klein) instead of cropped; the per-image review can still override it. */}
                  <button type="button" role="switch" aria-checked={allowAutoCrop}
                    onClick={() => setWatermarkAllowCrop(!allowAutoCrop)}
                    disabled={ds.busy || savingAllowCrop}
                    title={allowAutoCrop
                      ? 'Auto-crop ON: watermarks in a border are cropped off (no invented pixels). Click to repaint them instead. Saved as a preference.'
                      : 'Auto-crop OFF: border watermarks are repainted (LaMa/Klein) instead of cropped. Click to allow cropping again. Saved as a preference.'}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold disabled:opacity-40 ${allowAutoCrop
                      ? 'border-border bg-surface text-content-subtle hover:text-content'
                      : 'border-amber-400/50 bg-amber-500/10 text-amber-200'}`}>
                    <Scissors aria-hidden="true" className="h-3.5 w-3.5" />{allowAutoCrop ? "Auto-crop on" : "Auto-crop off"}
                  </button>
                  <button type="button"
                    onClick={() => { requestHelpTip('watermark-batch-clean'); ds.cleanWatermarks(watermarkMethod, watermarkTarget); }}
                    disabled={ds.busy || savingAllowCrop || cleanTargetCount === 0}
                    title={watermarkMethod === 'klein'
                      ? (allowAutoCrop
                        ? 'Removes them with masked Flux.2 Klein inpaint: border marks are cropped, every other mark (off-center AND on-subject) is repainted then composited back — only the mark changes'
                        : 'Auto-crop off: EVERY mark (border included) is repainted with masked Flux.2 Klein inpaint then composited back — only the mark changes')
                      : caps.watermark_inpaint
                      ? (allowAutoCrop
                        ? 'Removes them: border marks are cropped, small off-center marks are inpainted (LaMa), on-subject marks are flagged for manual review'
                        : 'Auto-crop off: border marks are repainted (LaMa) instead of cropped; large/on-subject marks are flagged for manual review')
                      : 'Removes border marks by cropping. Inpainting (LaMa) needs a one-time install — use Install inpainting next to this button; off-center marks are skipped until then'}
                    className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
                    <Eraser aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Clean ({cleanTargetCount})
                  </button>
                  </>
                )}
                {/* Per-image control: step through the flagged images full-screen, see each
                    detected box, and Clean / dismiss (false positive) / reject one by one.
                    The auto-detect has false positives, so this hands the final call to the
                    user (crucial after the "64/75 flagged" real-dataset run). */}
                {watermarkDetected > 0 && (
                  <button id="ds-curation-review-flagged" type="button" data-workspace-focus
                    disabled={ds.busy}
                    onClick={() => setReviewQueue(images.filter((i) => i.watermark_state === 'detected'))}
                    title="Step through the flagged images one by one — see each detected box and Clean, dismiss a false positive, or reject"
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40 scroll-mt-20">
                    <Search aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Review flagged ({watermarkDetected})
                  </button>
                )}
                {/* The shortcut past that review, because it was asked for — and
                    it says what it costs. The count is `flagged.rejectable`, NOT
                    watermarkDetected: a small-image rescue row in the batch makes
                    the SERVER refuse the whole request (400, zero rejected) and a
                    'failed' row is skipped, so promising the bigger number would
                    be the bank's "5 930 → 0" defect all over again. Red-tinted
                    like 🧹 Purge, never the primary colour: this is the shortcut,
                    the per-image review is still the recommended path. */}
                {flagged.rejectable > 0 && (
                  <button id="ds-curation-reject-flagged" type="button" data-workspace-focus
                    disabled={ds.busy}
                    onClick={async () => {
                      if (!window.confirm(rejectFlaggedConfirmText(flagged))) return;
                      const affected = await ds.batchImages(flagged.rejectableIds, 'reject');
                      // The server's number, not ours — it is the one that counts
                      // rows it actually wrote (see batch_image_action).
                      setRejectFlaggedNote(affected
                        ? `✓ rejected ${affected} — undo with Show ▸ Rejected in the grid, then ✓ Keep`
                        : '');
                    }}
                    title="The detector is a review flag, not a verdict — it does flag clean images sometimes, which is what Review flagged is for. This rejects them all at once instead; rejected images stay on disk and can be brought back."
                    className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm disabled:opacity-40 scroll-mt-20">
                    <X aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Reject all flagged ({flagged.rejectable})
                  </button>
                )}
                {/* Two numbers that differ must never be shown as one. */}
                {flagged.rejectable > 0 && flagged.heldBack > 0 && (
                  <p className="m-0 basis-full text-content-subtle text-[0.6875rem]">
                    {flagged.rejectable} of {flagged.flagged} flagged can be rejected in bulk —
                    the rest are small-image rescue pairs or failed rows, settled in their own review.
                  </p>
                )}
                {rejectFlaggedNote && (
                  <p className="m-0 basis-full text-emerald-300/90 text-[0.6875rem]">
                    {rejectFlaggedNote}
                  </p>
                )}
                {/* Who judged, and how many carry no position. Silent on the
                    ordinary run (one source, every flag located). */}
                {flaggedNote && (
                  <p className="m-0 basis-full text-content-subtle text-[0.6875rem]">
                    ℹ {flaggedNote}
                  </p>
                )}
                {/* The ONLY way to re-judge images ruled false positives. Without
                    it, changing the detector leaves a residue of verdicts from the
                    old one that nothing can ever touch again — a setting that
                    visibly changes nothing. Shown only when there IS a residue, so
                    the row does not grow on the ordinary dataset. */}
                {flagged.dismissed > 0 && (
                  <button type="button" disabled={ds.busy}
                    onClick={() => ds.findWatermarks({ includeDismissed: true })}
                    title="Re-examines the images you ruled false positives too. Use it after changing the watermark detector — a normal scan skips them forever."
                    className="px-3 py-1.5 rounded-lg bg-surface border border-dashed border-border text-content-subtle text-sm disabled:opacity-40 hover:text-content">
                    ⟲ Rescan incl. dismissed ({flagged.dismissed})
                  </button>
                )}
                {/* Watermark inpainting (LaMa) needs one extra ML package (simple-lama-
                    inpainting). Show a scoped installer RIGHT HERE — where the lack is
                    met — instead of sending the user back to Setup's whole ML-extras
                    step. Toggles a panel below (InstallRunner does the polling +
                    progress + manual-command fallback); on success caps re-fetch and
                    this affordance disappears. */}
                {!caps.watermark_inpaint && (
                  <button type="button" onClick={() => setInstallInpaintOpen((v) => !v)}
                    aria-expanded={installInpaintOpen}
                    title="Install the watermark-inpainting package (LaMa) so off-center marks can be repainted instead of only cropped. One-time download (~hundreds of MB)."
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-dashed border-amber-400/50 bg-amber-500/5 text-amber-200/90 text-sm hover:bg-amber-500/10">
                    <Download aria-hidden="true" className="h-4 w-4" /> Install inpainting
                    <span className="text-content-subtle text-[0.625rem] font-normal">one-time · ~hundreds of MB</span>
                    <span aria-hidden className="text-content-subtle text-xs">{installInpaintOpen ? '▴' : '▾'}</span>
                  </button>
                )}
                </div>
              </div>

              {/* Scoped watermark-inpainting installer. Reuses the Setup InstallRunner
                  (same polling / live progress / manual-command fallback). onDone
                  force-refreshes capabilities → watermark_inpaint flips true without a
                  restart or the 600 s probe TTL (the backend drops the import cache on
                  success), and the affordance above unmounts on its own. */}
              {installInpaintOpen && !caps.watermark_inpaint && (
                <div className="rounded-lg border border-amber-400/40 bg-amber-500/5 p-3 flex flex-col gap-2">
                  <div className="flex items-start gap-2">
                    <Eraser aria-hidden="true" className="h-5 w-5 shrink-0 text-amber-300" />
                    <div className="flex flex-col">
                      <span className="text-amber-200 text-sm font-semibold">Install watermark inpainting (LaMa)</span>
                      <span className="text-content-subtle text-[0.6875rem]">
                        Adds the <code className="text-amber-200/90">simple-lama-inpainting</code> package
                        (pulls a CPU torch — one-time download, ~hundreds of MB). No restart, no GPU:
                        once done, the Clean pass inpaints small off-center marks instead of skipping them.
                      </span>
                    </div>
                    <button type="button" onClick={() => setInstallInpaintOpen(false)}
                      className="ml-auto shrink-0 text-content-subtle hover:text-content text-sm"
                      aria-label="Close the inpainting installer"><X aria-hidden="true" className="h-4 w-4" /></button>
                  </div>
                  <InstallRunner action="watermark_inpaint" buttonLabel="Download & install"
                    onDone={() => refreshCaps(true)} />
                </div>
              )}

              {/* Nettoyage définitif des rejetées/échouées (ex-item du menu ⋯ More :
                  c'est une action de curation, elle vit avec les autres). */}
              {unused > 0 && (
                <div id="ds-curation-rejected-cleanup" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-20">
                  <button type="button" data-workspace-focus disabled={ds.busy}
                    onClick={() => {
                      if (window.confirm(`Permanently delete the ${unused} rejected/failed image(s) (files included)?`)) ds.purgeUnused();
                    }}
                    title="Permanently delete rejected and failed images"
                    className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm disabled:opacity-40">
                    <Trash2 aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Purge rejected/failed ({unused})
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    frees disk space — rejected images never train either way
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* ============ ✍️ Captions — générer/regénérer les captions, surveiller
               les fuites (identité/concept), outils de masse (find/replace, tags). */}
          <div className={sectionCls('captions')}>
            {heading('captions')}
            <div id="gf-captions" className="scroll-mt-20 flex flex-col gap-2">
              {/* WHO wrote the captions of the pass that just ran. The default engine
                  setting is "Auto", which silently chains JoyCaption and the Ollama
                  vision model — two different writing styles — and the result used to
                  be reported as a bare count. Shown HERE, under the buttons that
                  produced it, not only in Settings. Wraps freely: it must stay
                  readable at 400px. In-session only; it disappears on reload. */}
              {lastCaptionEngines && (
                <p title={CAPTION_ENGINE_WHY}
                  className="break-words rounded-lg border border-border bg-surface px-3 py-1.5 text-[0.75rem] text-content-muted">
                  <PenLine aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Last pass: {ds.lastCaptionRun.captioned} caption(s) — {lastCaptionEngines}
                </p>
              )}
              <div id="ds-captions-generate" tabIndex={-1}
                className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-20">
                {!isConceptual && (
                  <select value={effCaptionMode} onChange={(e) => setCaptionMode(e.target.value)} disabled={ds.busy}
                    title={d.train_type === 'anima'
                      ? "Caption style — Anima reads BOTH: booru tags and natural language are first-class on this model. Prose is only the default; switching to Booru tags trains fine and is never flagged as a mismatch."
                      : "Caption style — Prose (Z-Image) or Booru tags (SDXL booru-native, e.g. bigLove). Defaults to auto based on the dataset's type."}
                    className="px-2 py-1.5 rounded-lg bg-surface border border-border text-content text-[0.8125rem] disabled:opacity-40">
                    <option value="prose">Prose</option>
                    <option value="booru">Booru tags</option>
                  </select>
                )}
                <button type="button" data-workspace-focus
                  onClick={() => ds.caption(effCaptionMode)} disabled={ds.busy}
                  className="px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
                  <Sparkles aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />
                  {ds.captioning ? `${keptCaptioned}/${kept} captioned…` : 'Caption the kept ones'}
                </button>
                <HelpBadge topic="action-caption-generate" />
                <button type="button" disabled={ds.busy || !recaptionable}
                  onClick={() => {
                    if (window.confirm(recaptionConfirmation(d.kind || 'character', recaptionable, keptAsserted))) ds.recaption(effCaptionMode);
                  }}
                  title={keptCaptioned && !recaptionable
                    ? 'Every existing caption is hand-written — a forced pass never rewrites those. Edit or clear them on their tiles instead.'
                    : isConcept
                    ? "Re-generates every caption while keeping the recurring concept unspoken"
                    : isStyle
                      ? "Re-generates every caption as content-only text without naming the aesthetic"
                      : "Re-generates every caption without describing identity (face/hair)"}
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  <RefreshCw aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Re-caption
                </button>
                <button type="button" data-workspace-focus
                  onClick={() => setCaptionOptionsOpen(true)} disabled={ds.busy}
                  title="Choose the caption engine, Ollama model and vocabulary, pull a new model, and add custom instructions — for this dataset"
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  <Settings aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Options
                </button>
                <HelpBadge topic="action-caption-options" />
                {/* Caption-leak badge — KIND-aware. character: identity words
                    (hair/face/skin); concept: the caption NAMING the concept (must bind to
                    the trigger, not the words); style: not applicable (the subjects'
                    description IS the content). BOTH count states are clickable → the same
                    explainer panel; the count is spelled out ("N checked") so a green 0
                    reads as a REAL result, not a scan that never ran. */}
                {isStyle ? (
                  <span className={`ml-auto text-[0.8125rem] ${keptUncaptioned ? 'text-amber-300' : 'text-emerald-400'}`}
                    title="Every Style image needs a content-only caption: describe subject, action and setting, but do not name the aesthetic, medium or artist. No activation trigger is added.">
                    {keptUncaptioned
                      ? `${keptUncaptioned} missing · content-only captions required · no trigger`
                      : `${keptCaptioned}/${kept} content-only captions · no trigger`}
                  </span>
                ) : d.caption_leak && (
                  d.caption_leak.captioned > 0 ? (
                    <button id="ds-captions-leak-review" type="button" data-workspace-focus
                      onClick={toggleLeakReview}
                      aria-expanded={showLeaks}
                      title={d.caption_leak.leaking === 0
                        ? (isConcept
                            ? "0 captions name the concept — it binds to the trigger. Click for what was checked and why."
                            : "0 captions describe hair/face/skin — identity binds to the trigger. Click for what was checked and why.")
                        : (isConcept
                            ? "These captions name the concept → it won't bind to the trigger. Click to see what's watched and fix them here."
                            : "These captions mention hair/face/skin → identity won't bind to the trigger. Click to see what's watched and fix them here.")}
                      className={`ml-auto text-[0.8125rem] underline decoration-dashed scroll-mt-20 ${
                        d.caption_leak.leaking === 0
                          ? 'text-emerald-400 decoration-emerald-400/40'
                          : 'text-amber-400 decoration-amber-400/50'}`}>
                      {d.caption_leak.leaking === 0
                        ? `0 ${isConcept ? 'concept' : 'identity'} leaks · ${d.caption_leak.captioned} captions checked`
                        : `${d.caption_leak.leaking}/${d.caption_leak.captioned} captions leak ${isConcept ? 'the concept' : 'identity'}`}
                      {' '}{showLeaks ? '▴' : '▾'}
                    </button>
                  ) : kept > 0 ? (
                    <button id="ds-captions-leak-review" type="button" data-workspace-focus
                      onClick={toggleLeakReview}
                      aria-expanded={showLeaks}
                      title={`The ${isConcept ? 'concept' : 'identity'}-leak scan runs on captions. Caption the kept images first. Click to learn what it checks.`}
                      className="ml-auto text-content-subtle text-[0.8125rem] underline decoration-dashed decoration-border scroll-mt-20">
                      {isConcept ? 'concept' : 'identity'}-leak scan: no captions yet {showLeaks ? '▴' : '▾'}
                    </button>
                  ) : null
                )}
              </div>

              {/* Caption-leak explainer + triage. Opened from the badge in EITHER state:
                  it says what a leak is (kind-specific: identity vs the concept itself),
                  WHAT was checked (so a green 0 is a real result, not a check that never
                  ran), why 0 is normal — and, when there ARE leaks, the offending captions
                  editable IN PLACE (saves on blur, like the grid). Style sets have no leak
                  concept, so the panel only opens for character/concept. */}
              {showLeaks && !isStyle && (
                <div className="rounded-lg border border-border bg-surface-raised p-3 flex flex-col gap-3 text-[0.75rem]">
                  <div className="flex items-start gap-2">
                    <Drama aria-hidden="true" className="h-5 w-5 shrink-0 text-content-muted" />
                    <div className="flex flex-col gap-1">
                      <span className="text-content font-semibold text-sm">{isConcept ? 'Concept-leak check' : 'Identity-leak check'}</span>
                      {isConcept ? (
                        <p className="m-0 text-content-muted leading-relaxed">
                          A <strong className="text-content">concept leak</strong> is a word in a caption
                          that names <em>the concept itself</em> — the recurring element every image in
                          the set shares. On a concept LoRA these words must stay OUT of the captions:
                          they bind the concept to the text instead of to your trigger word{' '}
                          <code className="text-indigo-300">{d.trigger_word || 'your trigger'}</code>.
                          Describe the person and scene freely, but leave the concept
                          {d.concept_desc ? <> (<em className="text-content-muted">{d.concept_desc}</em>)</> : null}
                          {' '}unspoken so it binds to the trigger, not the caption.
                        </p>
                      ) : (
                        <p className="m-0 text-content-muted leading-relaxed">
                          An <strong className="text-content">identity leak</strong> is a word in a caption
                          that describes <em>who the person is</em> — the traits currently set to Omit
                          (face, eyes, skin, and by default hair). On a character LoRA those words
                          must stay OUT of the captions: they dilute the identity into the text instead
                          of binding it to your trigger word{' '}
                          <code className="text-indigo-300">{d.trigger_word || 'your trigger'}</code>.
                          Flip a family to Describe in Captions ▸ Options when you want that look
                          prompt-controllable.
                        </p>
                      )}
                    </div>
                    <button type="button" onClick={toggleLeakReview}
                      className="ml-auto shrink-0 text-content-subtle hover:text-content text-sm" aria-label="Close"><X aria-hidden="true" className="h-4 w-4" /></button>
                  </div>

                  {/* What was checked — the numbers behind the badge. */}
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-content-subtle tabular-nums">
                    <span><strong className="text-content-muted">{d.caption_leak?.captioned ?? 0}</strong> captions checked</span>
                    <span className={d.caption_leak?.leaking ? 'text-amber-300' : 'text-emerald-400'}>
                      <strong>{d.caption_leak?.leaking ?? 0}</strong> leaking
                    </span>
                    <span className="text-content-subtle/70">re-scanned live on every caption change</span>
                  </div>

                  {/* Words the detector watches. Concept: derived from the description
                      (its words + their basic lexical field); character: the fixed regex. */}
                  <div className="flex flex-col gap-1">
                    <span className="text-content-subtle">Words watched for:</span>
                    {isConcept ? (
                      <p className="m-0 text-content-muted leading-relaxed">
                        The words of the concept description
                        {d.concept_desc ? <> (<em className="text-content-muted">{d.concept_desc}</em>)</> : null}
                        {' '}and their basic lexical field — the body parts and positions it refers to
                        (e.g. a leg pose also watches <em>knees, feet, thighs, lifted, raised</em>), so a
                        periphrase can’t sneak the concept back into the caption.
                      </p>
                    ) : (
                      <div className="flex flex-wrap gap-1.5">
                        {(d.caption_leak?.watched?.length
                          ? d.caption_leak.watched
                          : ['hair', 'eye colour', 'skin · complexion · freckles',
                             'jawline · eyebrows · facial features', 'face shape',
                             ...(bodyFid ? ['tattoos · scars · piercings (body fidelity)'] : [])]
                        ).map((c) => (
                          <span key={c} className="rounded-full bg-surface border border-border px-2 py-0.5 text-content-muted text-[0.6875rem]">{c}</span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Why a green 0 is expected, not suspicious. */}
                  {d.caption_leak?.captioned === 0 ? (
                    <p className="m-0 text-content-subtle leading-relaxed">
                      Nothing checked yet — the scan runs on captions. Caption the kept images first.
                    </p>
                  ) : d.caption_leak?.leaking === 0 ? (
                    <p className="m-0 text-emerald-400/90 leading-relaxed">
                      {isConcept
                        ? <><CheckCircle2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />All clear — every caption describes the scene while leaving the concept
                          unspoken, so it will bind to your trigger. It’s a real result on {d.caption_leak?.captioned} caption(s),
                          not a check that didn’t run.</>
                        : <><CheckCircle2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />All clear — and this is expected. The app’s captioner is built to describe pose,
                          clothing, setting and framing but never the person’s identity, so a clean character
                          set genuinely reads 0. It’s a real result on {d.caption_leak?.captioned} caption(s),
                          not a check that didn’t run.</>}
                    </p>
                  ) : (
                    <div className="rounded-lg border border-amber-400/40 bg-amber-500/5 p-2.5 flex flex-col gap-2">
                      {/* Targeted re-caption is scoped, so it stays disabled only while a
                          full batch/other vision pass is running (ds.captioning) or another
                          targeted row is in flight — the offending row shows its own spinner. */}
                      {(() => {
                        const recaptionLocked = ds.busy || ds.captioning || ds.recaptioningIds.size > 0;
                        return (
                          <div className="flex items-start justify-between gap-2 flex-wrap">
                            <span className="text-amber-300 text-[0.8125rem] font-semibold">
                              {isConcept
                                ? <>Captions naming the concept ({d.caption_leak?.leaking}) — remove the concept words, or Re-caption. Edits save when you click away.</>
                                : <>Captions leaking identity ({d.caption_leak?.leaking}) — remove the highlighted words, or Re-caption. Edits save when you click away.</>}
                              <HelpBadge topic="action-recaption-targeted" className="ml-1" />
                            </span>
                            {leakingImages.length > 1 && (
                              <button type="button"
                                disabled={recaptionLocked}
                                onClick={() => ds.recaptionImages(leakingImages.map((i) => i.id), effCaptionMode)}
                                title={isConcept
                                  ? 'Re-generate every leaking caption while keeping the concept unspoken'
                                  : 'Re-generate every leaking caption without describing identity (face/hair)'}
                                className="shrink-0 px-2.5 py-1 rounded-lg bg-amber-500/15 text-amber-200 text-[0.75rem] font-semibold border border-amber-400/40 hover:bg-amber-500/25 disabled:opacity-40">
                                <RefreshCw aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Re-caption all leaking ({leakingImages.length})
                              </button>
                            )}
                          </div>
                        );
                      })()}
                      {leakingImages.map((img) => {
                        const rowBusy = ds.recaptioningIds.has(img.id);
                        const recaptionLocked = ds.busy || ds.captioning || ds.recaptioningIds.size > 0;
                        return (
                          <div key={img.id} className="flex gap-2 items-start">
                            <img src={`/api/dataset/${d.id}/img/${encodeURIComponent(img.filename)}`}
                              alt={img.variation_label || 'dataset image'} loading="lazy"
                              className="w-14 h-14 rounded-lg object-cover shrink-0 bg-black" />
                            <div className="flex-1 min-w-0 flex flex-col gap-1">
                              <textarea defaultValue={img.caption || ''} rows={2}
                                key={`${img.id}:${img.caption || ''}`}
                                onBlur={(e) => {
                                  if (e.target.value !== (img.caption || '')) ds.setCaption(img.id, e.target.value);
                                }}
                                aria-label={`Caption of image ${img.id}`}
                                className="w-full bg-app/60 border border-amber-400/30 rounded px-2 py-1 text-[0.6875rem] text-content resize-y" />
                              {/* WHO WROTE THE LEAKING SENTENCE. This list is read
                                  caption by caption to decide what to redo, and the
                                  'auto' backend chains two engines inside one run —
                                  so "which engine keeps leaking" is answerable here
                                  and was not. Silent when the author was never
                                  recorded (that is not "a model wrote it"). */}
                              {captionOriginInfo(img.caption_origin).known && (
                                <span className="text-[0.625rem] text-content-subtle"
                                  title={captionOriginInfo(img.caption_origin).title}>
                                  {captionOriginInfo(img.caption_origin).short}
                                </span>
                              )}
                              <button type="button"
                                disabled={recaptionLocked}
                                onClick={() => ds.recaptionImages([img.id], effCaptionMode)}
                                title={isConcept
                                  ? 'Re-generate this caption while keeping the concept unspoken'
                                  : 'Re-generate this caption without describing identity (face/hair)'}
                                className="self-start px-2 py-0.5 rounded-lg bg-surface text-content text-[0.6875rem] border border-border hover:bg-surface-raised disabled:opacity-40">
                                {rowBusy
                                  ? <><Loader2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px] animate-spin" />Re-captioning…</>
                                  : <><RefreshCw aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Re-caption</>}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                      {leakingImages.length === 0 && (
                        <p className="m-0 text-emerald-400 text-[0.8125rem]"><CheckCircle2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />All clear — no leaking caption left.</p>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div id="ds-captions-tools" tabIndex={-1} className="scroll-mt-20">
                <CaptionToolsBar images={images} kind={d.kind || 'character'} mode={effCaptionMode}
                  excludes={excludeTags} includes={includeTags}
                  onExclude={toggleExclude} onInclude={toggleInclude}
                  onReplace={ds.replaceCaptions}
                  onWriteFiles={ds.writeCaptionFiles} onOpenFolder={ds.openDatasetFolder}
                  busy={ds.busy}
                  open={captionToolsOpen}
                  onOpenChange={(open) => onRevealOpenChange('tools', open, setCaptionToolsOpen)} />
              </div>
              {filtersActive && (
                <p className="m-0 text-content-subtle text-[0.6875rem]">
                  <Filter aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />A grid filter is active — the filtered grid lives in{' '}
                  <button type="button" onClick={() => setSection('images')}
                    className="underline hover:text-content">Images</button>
                  {' '}(showing {gridImages.length} of {rescueGridImages.length}).
                </p>
              )}
            </div>
          </div>

          {/* ============ 📦 Import & export — fusionner un dataset existant ;
               sortir celui-ci (ZIP d'entraînement, backup portable, HF Hub). */}
          <div className={sectionCls('export')}>
            {heading('export')}
            <div id="gf-export" className="scroll-mt-20 flex flex-col gap-2">
              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">Bring images in</span>
              <div id="ds-export-import" tabIndex={-1}
                className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-20">
                <button type="button" data-workspace-focus
                  onClick={() => zipInput.current?.click()} disabled={importBusy}
                  title="Merge an existing training dataset into this one: a ZIP of images with kohya-style same-name .txt captions (any folder layout). Aspect kept, perceptual duplicates skipped."
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                  <Package aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Import dataset (ZIP)
                </button>
                <button type="button" disabled={importBusy} onClick={importFolderPrompt}
                  title="Merge an existing training dataset already on this machine's disk: a folder of images with kohya-style same-name .txt captions (subfolders included). Aspect kept, perceptual duplicates skipped."
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                  <FolderOpen aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Import from folder…
                </button>
                <span className="text-content-subtle text-[0.6875rem]">
                  merges images + same-name .txt captions in — duplicates are skipped
                </span>
              </div>
              {/* The round trip existed and nothing said so: export → caption in
                  another tool → import back. It only became a real trip once a
                  re-imported .txt landed on the image ALREADY here instead of
                  being dropped with its duplicate (see _merge_training_images).
                  Reported by Qeeyana (Reddit). */}
              <p id="ds-caption-elsewhere" tabIndex={-1}
                className="scroll-mt-20 rounded-lg border border-border bg-surface px-3 py-2 text-[0.6875rem] text-content-muted">
                <span className="font-medium text-content">Want to caption in another tool?</span>{' '}
                Export the ZIP below, caption it wherever you like, then bring the same
                folder back through Import dataset: images already here are not
                duplicated, and their new <code>.txt</code> captions land on them.
                A caption you already wrote here is never overwritten.
              </p>
              <input ref={zipInput} type="file" accept=".zip,application/zip" className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) ds.importDatasetZip(f);
                  e.target.value = '';
                }} />

              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">Get this dataset out</span>
              <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
                <div id="ds-export-training-zip" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                  <button type="button" data-workspace-focus={kept ? '' : undefined}
                    disabled={!kept} onClick={exportZipGuarded}
                    className="px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
                    <Download aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Export ZIP ({kept})
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    kept images + captions, training-ready (kohya layout)
                  </span>
                </div>
                {/* One primary way out (the training ZIP) stays in the open; the
                    other three are occasional, so they live behind one disclosure
                    instead of four buttons competing for the same glance. A
                    <details> and not a floating menu: the workspace landing
                    (openCollapsedAncestors in `land`) opens collapsed ancestors on
                    jump, so the sidebar links to Import to bank / Backup /
                    Hugging Face keep working. Do NOT make this a controlled
                    <details> without teaching `land` about it. */}
                <details className="rounded-lg border border-border bg-surface-raised">
                  <summary className="flex items-center gap-2 px-2.5 py-1.5 text-[0.6875rem] text-content-muted hover:text-content cursor-pointer select-none">
                    ⋯ More ways out
                    <span className="text-content-subtle">
                      bank · portable backup{caps.hf_publish && kept > 0 ? ' · Hugging Face' : ''}
                    </span>
                  </summary>
                  <div className="flex flex-col gap-2 px-2.5 pb-2.5 pt-1">
                    <div id="ds-export-to-bank" tabIndex={-1}
                      className="flex items-center gap-2 flex-wrap scroll-mt-20">
                      <button type="button" data-workspace-focus disabled={!kept}
                        onClick={() => setImportToBankOpen(true)}
                        title="Turn this dataset back into a bank: its kept images are COPIED into a bank of their own. Both choices keep Dataset-owned captions, curation, framing, watermark and provenance. By default compatible final-file technical analysis is restored; Start fresh skips only reuse of prior analysis. This dataset is never touched."
                        className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                        <ArrowUp aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Import to bank
                      </button>
                      <span className="text-content-subtle text-[0.6875rem]">
                        both choices keep Dataset metadata — restore compatible analysis, or start fresh analysis
                      </span>
                    </div>
                    <div id="ds-export-backup" tabIndex={-1}
                      className="flex items-center gap-2 flex-wrap scroll-mt-20">
                      <button type="button" data-workspace-focus onClick={ds.exportBackup}
                        title="Full portable backup: all images with statuses, captions, scores and settings — restore it on any machine from the Datasets page."
                        className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                        <Save aria-hidden="true" className="mr-1.5 inline h-4 w-4 align-[-2px]" />Backup
                      </button>
                      <span className="text-content-subtle text-[0.6875rem]">
                        portable copy — restore it on any machine from the Datasets page
                      </span>
                    </div>
                    {caps.hf_publish && kept > 0 && (
                      <div id="ds-export-hugging-face" tabIndex={-1}
                        className="flex items-center gap-2 flex-wrap scroll-mt-20">
                        <button type="button" data-workspace-focus
                          onClick={() => setPublishHfOpen(true)}
                          title="Publish this dataset (kept images + captions) as a dataset repo on the Hugging Face Hub. Private by default; you choose the license and confirm you have the right to share."
                          className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                          🤗 Publish to Hugging Face
                        </button>
                        <span className="text-content-subtle text-[0.6875rem]">
                          dataset repo on the Hub — private by default
                        </span>
                      </div>
                    )}
                  </div>
                </details>
              </div>
            </div>
          </div>

          {/* ============ 🎓 Training — readiness, launch, progress and options. */}
          <div className={sectionCls('training')}>
            {heading('training')}
            <div id="gf-training" className="scroll-mt-20 flex flex-col gap-2">
              <div id="ds-training-launch" tabIndex={-1}
                className="flex flex-col gap-2 scroll-mt-20">
                {/* Pastille de préparation (miroir du preflight) : refreshKey borné aux
                    compteurs pertinents → pas de re-fetch à chaque poll du dataset. */}
                {caps.training_visible && (
                  <TrainingReadiness datasetId={d.id} trainType={d.train_type} variant={d.train_variant}
                    refreshKey={`${kept}|${keptCaptioned}|${pending}|${triage}|${d.caption_leak?.leaking ?? ''}`}
                    onJump={(targetId) => jumpTo({ targetId })}
                    onOverrideChange={setNotReadyAck} />
                )}
                <TrainingPanel ds={ds} keptCount={kept} kind={d.kind} allowNotReady={notReadyAck}
                  sectionVisible={section === 'training'}
                  onCheckpointsChange={setCheckpointCount}
                  checkpointHost={checkpointHost}
                  navigationPanel={section === 'training' && panel === 'advanced' ? panel : null}
                  onNavigationStateChange={setTrainingNavigation}
                  onPanelOpenChange={(panelId, open) => {
                    if (!open && section === 'training' && panel === panelId) clearActivePanel();
                  }} />
              </div>
            </div>
          </div>

          {/* The TrainingPanel stays mounted exactly once; its checkpoint manager
              portals into this first-class stage so the queue poller is not duplicated. */}
          <div className={sectionCls('checkpoints')}>
            {heading('checkpoints')}
            <div id="gf-checkpoints" className="scroll-mt-20 flex flex-col gap-2">
              <div id="ds-checkpoints-manager" ref={setCheckpointHost} tabIndex={-1}
                className="scroll-mt-20" />
            </div>
          </div>

          {/* ============ 🎛️ Studio — final stage and dedicated-page launcher. */}
          <div className={sectionCls('studio')}>
            {heading('studio')}
            <div id="gf-studio" className="scroll-mt-20 flex flex-col gap-2">
              {caps.studio_visible ? (
                <button id="ds-studio-launcher" type="button" data-workspace-focus
                  onClick={() => navigate(`/studio?dataset=${d.id}`)}
                  className="flex items-center gap-2 rounded-lg border border-purple-500/30 bg-purple-500/5 px-3 py-2.5 text-left hover:bg-purple-500/10 transition-colors scroll-mt-20">
                  <SlidersHorizontal aria-hidden="true" className="h-4 w-4" />
                  <span className="text-content font-semibold text-sm">LoRA testing studio</span>
                  {d.best_settings && (
                    <span className="text-amber-300 text-[0.6875rem]" title="Saved winning settings">
                      ★ {fmt(d.best_settings.strength)}
                    </span>
                  )}
                  <span className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold">
                    ⤢ Open Studio
                  </span>
                </button>
              ) : (
                <p className="m-0 rounded-lg border border-border bg-surface px-3 py-2 text-content-muted text-sm">
                  Configure ComfyUI in Settings to use the LoRA testing Studio.
                </p>
              )}
              <div><HelpBadge topic="action-studio-open" /></div>
            </div>
          </div>
        </div>{/* /right column */}
      </div>{/* /workspace grid */}

      {/* Every dialog below renders behind a click; the lazy ones load
          their chunk on that first click, a frame nobody can see. */}
      <Suspense fallback={null}>
      {cropImg && cropImg.filename && (
        <CropModal imageUrl={`/api/dataset/${d.id}/img/${encodeURIComponent(cropImg.filename)}${
          ds.nonces?.[cropImg.id] ? `?v=${ds.nonces[cropImg.id]}` : ''}`}
          onCancel={() => setCropImg(null)}
          onConfirm={async (box) => { await ds.crop(cropImg.id, box); setCropImg(null); }} />
      )}
      {refCrop && d.ref_filename && (
        // Feed the crop editor the full-frame ORIGINAL (when kept) so the box can widen
        // back out — not just tighten the already-cropped square. Legacy datasets with
        // no stored original fall back to the cropped ref (can only tighten, as before).
        <CropModal imageUrl={`/api/dataset/${d.id}/img/${encodeURIComponent(d.ref_original_filename || d.ref_filename)}`}
          // 1:1 is the historical default; the editor itself still offers its ratio
          // buttons when a different manual crop is wanted.
          defaultAspect={1}
          onCancel={() => setRefCrop(false)}
          onConfirm={async (box) => { await ds.cropRef(box); setRefCrop(false); }}
          onReset={d.ref_original_filename
            ? async () => { await ds.recropRefAuto(); setRefCrop(false); }
            : undefined} />
      )}
      {refEdit && d.ref_filename && (
        <ReferenceEditModal datasetId={d.id} refFilename={d.ref_filename} nonce={ds.refNonce}
          // The local engines are free but not universal: the modal is handed the
          // SAME capabilities the generation panel reads, so it can offer them
          // when this ComfyUI can run them, explain the one missing action when it
          // nearly can, and drop them entirely on an install that has no ComfyUI.
          defaultEngine={defaultEditEngine(window.localStorage,
            (e) => !localEngineUnavailableReason(e, caps))}
          comfyuiConfigured={hasComfyui(caps)}
          engineAvailable={caps.engines || {}}
          engineReason={(e) => localEngineUnavailableReason(e, caps)}
          datasetExtraCount={(d.ref_extra_filenames || []).length}
          liveActivity={ds.activity} referenceEdit={d.reference_edit}
          onEdit={ds.editReference} onRetry={ds.retryReferenceEdit}
          canRetry={ds.canRetryReferenceEdit}
          onKeep={ds.keepEditedReference} onDiscard={ds.discardEditedReference}
          onClose={() => setRefEdit(false)} />
      )}
      {extraRefCrop && extraRefCropSource(d.ref_extra_filenames, d.ref_extra_crop_sources, extraRefCrop) && (
        // Same editor as the primary reference, fed the extra's full-frame ORIGINAL
        // when one is kept. No ratio preset: extras are bust/body shots kept in their
        // own aspect, and nothing downstream needs a square. No "Reset to auto" —
        // extras are deliberately imported without the head-crop vision pass.
        <CropModal imageUrl={`/api/dataset/${d.id}/img/${encodeURIComponent(
          extraRefCropSource(d.ref_extra_filenames, d.ref_extra_crop_sources, extraRefCrop))}${
          ds.refNonce ? `?v=${ds.refNonce}` : ''}`}
          onCancel={() => setExtraRefCrop(null)}
          onConfirm={async (box) => { await ds.cropExtraRef(extraRefCrop, box); setExtraRefCrop(null); }} />
      )}
      {viewImgLive && (
        <DatasetLightbox img={viewImgLive} datasetId={d.id}
          nonce={(ds.nonces && ds.nonces[viewImgLive.id]) || 0}
          compare={viewImgComparison}
          parentNonce={(ds.nonces && viewImgComparison?.parent
            && ds.nonces[viewImgComparison.parent.id]) || 0}
          // The SECOND comparison — against the dataset's reference photo — is
          // decided inside the lightbox: unlike the derived parent it is not a
          // row of the payload, just this dataset's own filename, and it is
          // served by the very same /img/<name> endpoint. `refNonce` is the
          // reference's own cache buster (a crop or a re-upload bumps it).
          refFilename={d.ref_filename || ''}
          refNonce={ds.refNonce || 0}
          onClose={() => setViewImg(null)}
          onMirror={viewImgLive._rescueReviewPreview ? undefined : ds.mirrorImage}
          onRotate={viewImgLive._rescueReviewPreview ? undefined : ds.rotateImage}
          mirrorBusy={Boolean(ds.mirroringIds?.has(viewImgLive.id))}
          // The lightbox hands back WHICH engine was pressed; a single-✨
          // surface passes none and the improve.engine setting decides.
          onImprove={canImproveViewImg
            ? ((imageId, engine) => ds.improveImage(imageId, { engine }))
            : undefined}
          onCameraAngles={canCameraViewImg ? shootDatasetViews : undefined}
          /* ⟨ / ⟩ walk `gridImages` — the filtered, sorted list the grid shows,
             the SAME array it is handed below. Not `images` (the raw payload):
             ⟩ would then land on a picture the current filters hide, behind an
             overlay that gives no way to notice. The rescue-review preview
             opens on a Curation pair, not on a position in that list, so it
             gets no arrows rather than arrows into someone else's sequence. */
          images={viewImgLive._rescueReviewPreview ? null : gridImages}
          onNavigate={viewImgLive._rescueReviewPreview ? null : setViewImg}
          /* ✓ Keep / ✕ Reject from inside the lightbox — the SAME write the
             grid tile makes, so a verdict taken while looking at the picture
             full size is the very same fact as one taken on the thumbnail.
             The rescue-review preview gets none: that pair is resolved in
             Curation, and a status written behind its back would decide half
             of a decision it owns. */
          onStatus={viewImgLive._rescueReviewPreview ? null : ds.setStatus}
          /* 🚩 Mark a watermark — the answer to a detector MISS, opened from the
             image the user is looking at rather than from the flagged queue they
             are precisely NOT in. It reuses the review overlay on a one-image
             queue: same editor, same Clean/Dismiss afterwards, nothing forked.
             The rescue-review preview is excluded like every other write. */
          onMarkWatermark={viewImgLive._rescueReviewPreview
            ? undefined
            : ((image) => { setViewImg(null); setReviewQueue([image]); })}
          /* ✦ Repair acts HERE now, without the detour through the watermark
             review. Same call the review screen makes, so both surfaces share
             one lane, one undo and one set of refusals. */
          onRepair={viewImgLive._rescueReviewPreview
            ? undefined
            : ((id, prompt, boxes, mask) => ds.repairImageRegion(id, prompt, boxes, mask))}
          onUndoRepair={viewImgLive._rescueReviewPreview
            ? undefined
            : ((id) => ds.undoImageRepair(id))}
          improvePending={viewImgImproving}
          improveReady={viewImgImprovementReady}
          busy={ds.busy || gridBulkBusy}
          improveBusy={ds.improveBusy || gridBulkBusy}
          curationBusy={ds.curationBusy || gridBulkBusy}
          // The refused writes in there name the pass that holds them, exactly
          // like the tiles behind the lightbox.
          busyReason={(ds.busy || gridBulkBusy) ? datasetBusyReason(ds.busy ? act : null) : null}
          kleinAvailable={Boolean(caps.engines?.klein)}
          subjectType={d.subject_type || 'human'}
          onCrop={viewImgLive._rescueReviewPreview
            ? undefined
            : (img) => { setViewImg(null); setCropImg(img); }} />
      )}
      {settingsOpen && (
        <DatasetSettingsModal d={d} busy={ds.busy}
          onSave={ds.updateSettings} onClose={() => setSettingsOpen(false)} />
      )}
      {importToBankOpen && (
        <DatasetToBankDialog datasetName={d.name} keptCount={kept}
          onClose={() => setImportToBankOpen(false)} onStart={importToBank} />
      )}
      {publishHfOpen && (
        <PublishHfModal datasetId={d.id} onClose={() => setPublishHfOpen(false)} />
      )}
      {folderBrowseOpen && (
        <FolderBrowserModal
          onPick={(p) => ds.importDatasetFolder(p, { silent: true })}
          onClose={() => setFolderBrowseOpen(false)} />
      )}
      {captionOptionsOpen && (
        <CaptionOptionsPopover datasetId={d.id} trainType={d.train_type} kind={d.kind}
          onClose={() => setCaptionOptionsOpen(false)} />
      )}
      {reviewQueue && reviewQueue.length > 0 && (
        <WatermarkReviewLightbox
          datasetId={d.id}
          queue={reviewQueue}
          caps={caps}
          nonces={ds.nonces}
          onSaveRegions={(id, regions) => ds.saveWatermarkRegions(id, regions)}
          onClean={(id, method, allowCrop) => ds.cleanWatermarkImages([id], method, allowCrop)}
          onRestore={(id) => ds.restoreWatermarkImage(id)}
          onDismiss={(id) => ds.dismissWatermarks([id])}
          onReject={(id) => ds.setStatus(id, 'reject')}
          onRepair={(id, prompt, boxes, mask) => ds.repairImageRegion(id, prompt, boxes, mask)}
          onUndoRepair={(id) => ds.undoImageRepair(id)}
          onClose={(recap) => {
            setReviewQueue(null);
            const summary = recap || buildWatermarkRecap({});
            if (summary) toast.success(`Review done — ${summary}`);
          }} />
      )}
      </Suspense>
    </div>
  );
}
