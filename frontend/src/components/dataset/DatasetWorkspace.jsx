import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import CompositionBar from './CompositionBar';
import ReferencePanel from './ReferencePanel';
import VariationCatalog from './VariationCatalog';
import TrainingPanel from './TrainingPanel';
import { fmt } from '../../utils/studioFormat';
import ImportDropzone from './ImportDropzone';
import ConceptSourcesPanel from './ConceptSourcesPanel';
import { isDatasetImportBlocked } from './scraperState';
import DatasetGrid from './DatasetGrid';
import SmallImageRescueReview from './SmallImageRescueReview';
import CaptionToolsBar from './CaptionToolsBar';
import CaptionOptionsPopover from './CaptionOptionsPopover';
import { recaptionConfirmation } from './captionCategory';
import CropModal from './CropModal';
import DatasetLightbox from './DatasetLightbox';
import DatasetSettingsModal from './DatasetSettingsModal';
import PublishHfModal from './PublishHfModal';
import WatermarkReviewLightbox, { buildWatermarkRecap } from './WatermarkReviewLightbox';
import { useToast } from '../common/Toast';
import { pickNativeFolder, FolderBrowserModal } from '../common/FolderPicker';
import { useCapabilities } from '../../context/CapabilitiesContext';
import InstallRunner from '../setup/InstallRunner';
import GuidedChecklist from './GuidedChecklist';
import NextStepCard from './NextStepCard';
import TrainingReadiness from './TrainingReadiness';
import useGuidedFlow from '../../hooks/useGuidedFlow';
import { filterImages, normalizeTag } from '../../utils/tagFilter';
import {
  buildSmallImageRescuePairs,
  filterSmallImageRescueGrid,
  isSmallImageRescueRow,
} from '../../utils/smallImageRescue';
import { WORKSPACE_SECTIONS, SECTION_FOR_TARGET } from './workspaceSections';
import { putJson } from '../../api/fetchClient';
import { HelpBadge } from '../../help/HelpMode';
import { requestHelpTip } from '../../help/helpTips';
import { useI18n } from '../../i18n/I18nContext';
import {
  PANEL_STATUS,
  getWorkspacePanel,
  getWorkspacePanelStatus,
  getWorkspacePanels,
  resolveWorkspaceLocation,
  withWorkspaceLocation,
} from './workspaceNavigation';

const EMPTY_IMAGES = Object.freeze([]);

// Style partagé des items du menu « ⋯ More » du header (actions secondaires).
const MENU_ITEM = 'w-full flex items-center gap-2 text-left px-2.5 py-1.5 rounded-md text-sm text-content hover:bg-surface-raised disabled:opacity-40';

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

/* Loud banner sitting directly above the grid whenever a tag filter is active,
   so the user can NEVER mistake a filtered view for "images disappeared". Shows
   every active exclusion (⊘) / inclusion (◉ only) as a removable chip, the live
   "showing N of M" count, and a one-click "clear all". Session-only state lives
   in the parent workspace (transient view, not persisted). */
function GridFilterBar({ excludes, includes, shown, total, onRemoveExclude, onRemoveInclude, onClearAll }) {
  const { t } = useI18n();
  return (
    <div role="status"
      className="flex items-center gap-2 flex-wrap rounded-lg border-2 border-amber-400/50 bg-amber-400/10 px-3 py-2">
      <span className="text-amber-200 text-sm font-semibold shrink-0">🔎 {t('workspace.filters.title')}</span>
      <span className="text-content-muted text-xs tabular-nums shrink-0">
        {t('workspace.filters.showing', { shown, total })}
      </span>
      <div className="flex items-center gap-1.5 flex-wrap">
        {excludes.map((tag) => (
          <span key={`x-${tag}`}
            className="inline-flex items-center gap-1 rounded-full border border-rose-400/50 bg-rose-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-rose-200">
            <span aria-hidden>⊘</span> {tag}
            <button type="button" onClick={() => onRemoveExclude(tag)}
              aria-label={t('workspace.filters.stopHiding', { tag })}
              className="w-4 h-4 grid place-items-center rounded-full hover:bg-rose-500/30">✕</button>
          </span>
        ))}
        {includes.map((tag) => (
          <span key={`i-${tag}`}
            className="inline-flex items-center gap-1 rounded-full border border-indigo-400/50 bg-indigo-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-indigo-200">
            <span aria-hidden>◉</span> {t('workspace.filters.only', { tag })}
            <button type="button" onClick={() => onRemoveInclude(tag)}
              aria-label={t('workspace.filters.stopIsolating', { tag })}
              className="w-4 h-4 grid place-items-center rounded-full hover:bg-indigo-500/30">✕</button>
          </span>
        ))}
      </div>
      <button type="button" onClick={onClearAll}
        className="ml-auto shrink-0 text-content-muted underline hover:text-content text-xs">
        {t('workspace.filters.clearAll')}
      </button>
    </div>
  );
}

export default function DatasetWorkspace({ ds, onBack }) {
  const navigate = useNavigate();
  const toast = useToast();
  const { t } = useI18n();
  const { caps, refresh: refreshCaps } = useCapabilities();
  const d = ds.data;
  const [cropImg, setCropImg] = useState(null);
  const [captionOptionsOpen, setCaptionOptionsOpen] = useState(false);
  // Frozen snapshot of the flagged queue when review mode opens (null = closed).
  const [reviewQueue, setReviewQueue] = useState(null);
  const zipInput = useRef(null);   // hidden input for "Import dataset (ZIP)"
  const [refCrop, setRefCrop] = useState(false);
  const [viewImg, setViewImg] = useState(null);
  const [captionMode, setCaptionMode] = useState(null);   // null → défaut auto selon train_type
  const [showLeaks, setShowLeaks] = useState(false);       // liste dépliée des captions qui fuient
  const [captionToolsOpen, setCaptionToolsOpen] = useState(false);
  const [installInpaintOpen, setInstallInpaintOpen] = useState(false);  // panneau d'install LaMa
  const [watermarkMethod, setWatermarkMethod] = useState('lama');  // moteur d'inpaint batch : lama | klein
  const [savingAllowCrop, setSavingAllowCrop] = useState(false);  // write-through of the auto-crop pref
  const [checkpointCount, setCheckpointCount] = useState(0);
  const [checkpointHost, setCheckpointHost] = useState(null);
  const [trainingNavigation, setTrainingNavigation] = useState({ ready: false, queueCount: 0 });
  // « Continue anyway » : ack remonté par la pastille de préparation (garde-fou
  // qualité contournable) → débloque le bouton Train du panneau et voyage jusqu'au
  // launch (allow_not_ready). Le serveur reste l'autorité.
  const [notReadyAck, setNotReadyAck] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [publishHfOpen, setPublishHfOpen] = useState(false);
  const [folderBrowseOpen, setFolderBrowseOpen] = useState(false);  // in-app folder browser (native-dialog fallback)
  // Grid tag-filter (session-only): tags whose images are hidden (exclude) or the
  // ONLY tags allowed through (include). Both are normalized (trim+lowercase).
  const [excludeTags, setExcludeTags] = useState([]);
  const [includeTags, setIncludeTags] = useState([]);
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
      toast.error(t('workspace.toast.autoCropSaveFailed'));
    } finally {
      setSavingAllowCrop(false);
    }
  }, [refreshCaps, t, toast]);
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
      if (!target || target.getClientRects().length === 0) return false;
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

  if (!d) return <p className="text-content-subtle text-sm">{t('common.loading')}</p>;

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
  // Captions that still leak identity/concept — the Identity-leak panel lists them for
  // in-place edit AND targeted 🔄 Re-caption (per row + a "Re-caption all leaking" header).
  const leakingImages = images.filter((i) => i.leak);
  // Overlaid watermarks still awaiting removal → drives the "🧽 Clean (N)" button.
  const watermarkDetected = images.filter((i) => i.watermark_state === 'detected').length;
  // Style de caption : défaut AUTO (SDXL booru-native → booru tags ; sinon prose), surchargé par le sélecteur.
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
  const clearFilters = () => { setExcludeTags([]); setIncludeTags([]); };
  const filtersActive = excludeTags.length > 0 || includeTags.length > 0;
  // The list actually rendered by the grid. Filtering here means select-all,
  // auto-triage and every bulk action operate ONLY on the visible images. The
  // Caption-tools counts keep using the full `images` list (global, never lies).
  const gridImages = filterImages(rescueGridImages, {
    excludes: excludeTags, includes: includeTags, mode: effCaptionMode,
  });
  const pending = images.filter((i) => i.status === 'pending' && !i.filename
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
    reference: `📸 ${t('workspace.next.reference')}`,
    generate: `⚡ ${t('workspace.next.generate')}`,
    curate: `🖼️ ${t('workspace.next.curate')}`,
    caption: `✨ ${t('workspace.next.caption')}`,
    finish: caps.training_visible
      ? `🎓 ${t('workspace.next.training')}`
      : `⬇ ${t('workspace.actions.exportZip')} (${kept})`,
    studio: `🎛️ ${t('workspace.next.studio')}`,
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
  const canImproveViewImg = !!viewImgLive
    && !viewImgLive._rescueReviewPreview
    && !isSmallImageRescueRow(viewImgLive)
    && viewImgLive.derivation_kind !== 'klein_image_improve';

  // Export ZIP — shared by the header CTA and the Import & export row.
  // Guard-rails: untriaged images are silently EXCLUDED from the zip. Style
  // captions are mandatory and have no trigger-only fallback.
  const exportZipGuarded = () => {
    if (triage && !window.confirm(t('workspace.confirm.exportWithTriage', { count: triage }))) return;
    if (isStyle && keptUncaptioned) {
      toast.error(t('workspace.toast.styleCaptionsMissing', { count: keptUncaptioned }));
      jumpTo({ targetId: 'gf-captions' });
      return;
    }
    if (keptUncaptioned && !window.confirm(t('workspace.confirm.exportWithoutCaptions', {
      count: keptUncaptioned,
    }))) return;
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
  // Unknown / legacy engine values fail safe as local: only these two API
  // engines are guaranteed not to share ComfyUI VRAM with vision auto-crop.
  const visionImportBusy = act?.kind === 'generate'
    && !['nanobanana', 'chatgpt'].includes(String(act?.engine || '').toLowerCase());
  const activityBanner = ds.captioning
    ? `${act?.detail || t('workspace.activity.captioningProgress', {
      done: keptCaptioned,
      total: kept,
    })} ${t('workspace.activity.comfyPaused')}`
    : (() => {
        if (act) {
          const prog = act.total ? ` ${act.done}/${act.total}` : '';
          // Passes that DON'T claim "ComfyUI is paused": the CPU ones, plus
          // 'generate' (engine-dependent — Nano Banana / ChatGPT don't touch
          // ComfyUI, and the Klein case is obvious from the tiles appearing).
          const cpu = act.kind === 'analyze_faces'
            || (act.kind === 'watermark_clean' && !String(act.detail || '').includes('GPU'))
            || act.kind === 'generate';
          const label = {
            watermark_detect: `${t('workspace.activity.watermarkDetect')}${prog}`,
            watermark_clean: `${t('workspace.activity.watermarkClean')}${prog}`,
            caption: `${t('workspace.activity.caption')}${prog}`,
            recaption: `${t('workspace.activity.recaption')}${prog}`,
            analyze_faces: `${t('workspace.activity.faceAnalysis')}${prog}`,
            classify: `${t('workspace.activity.classify')}${prog}`,
            generate: `${t('workspace.activity.generate')}${prog}`,
          }[act.kind];
          if (label) {
            const detailed = act.detail || label;
            return `${detailed}${cpu ? '' : ` ${t('workspace.activity.comfyPausedDuringPass')}`}`;
          }
        }
        return t('workspace.activity.gpuProcessing');
      })();

  // ── Sidebar : pastilles par section — ambre quand une action attend l'utilisateur,
  //    indigo pulsé quand des générations tournent, neutre pour l'info « à faire ».
  const navBadges = {
    images: triage > 0
      ? { n: triage, tone: 'amber', srLabel: t('workspace.badges.awaitingTriage', { count: triage }) } : null,
    add: pending > 0
      ? { n: pending, tone: 'indigo', pulse: true, srLabel: t('workspace.badges.generating', { count: pending }) } : null,
    curation: watermarkDetected + rescueReviewCount > 0
      ? {
          n: watermarkDetected + rescueReviewCount,
          tone: 'amber',
          srLabel: t('workspace.badges.curation', {
            watermarks: watermarkDetected,
            rescues: rescueReviewCount,
          }),
        } : null,
    captions: (!isStyle && (d.caption_leak?.leaking ?? 0) > 0)
      ? { n: d.caption_leak.leaking, tone: 'amber', srLabel: t('workspace.badges.leaking', {
        count: d.caption_leak.leaking,
      }) }
      : keptUncaptioned > 0
        ? { n: keptUncaptioned, tone: 'subtle', srLabel: t('workspace.badges.uncaptioned', {
          count: keptUncaptioned,
        }) } : null,
    export: null,
    training: null,
  };

  const activePanels = getWorkspacePanels(section, navContext);

  const panelNavItem = (sectionId, destination, chip = false) => {
    const isActive = sectionId === section && destination.id === panel;
    const className = chip
      ? `shrink-0 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs ${
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
        {t(`workspace.sections.${sectionId}.panels.${destination.id}`)}
      </button>
    );
  };

  // Un item de la sidebar : rail vertical desktop (chip=false) ou chip du bandeau
  // horizontal mobile (chip=true) — mêmes classes que la sidebar de Settings.
  const navItem = (s, chip) => {
    const isActive = s.id === section;
    const base = chip
      ? `flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium ${
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
        <span aria-hidden>{s.icon}</span>
        <span>{t(`workspace.sections.${s.id}.title`)}</span>
        <NavBadge badge={navBadges[s.id]} />
        {!chip && <span aria-hidden className="text-content-subtle text-[0.625rem]">{isActive ? '▾' : '▸'}</span>}
      </button>
    );
  };

  const sectionMeta = Object.fromEntries(WORKSPACE_SECTIONS.map((s) => [s.id, s]));
  const heading = (id) => {
    const s = sectionMeta[id];
    return <SectionHeading id={`ds-section-${id}-heading`}
      eyebrow={t(`workspace.sections.${id}.eyebrow`)}
      title={t(`workspace.sections.${id}.title`)}
      badge={<HelpBadge topic={`workspace-${id}`} />}
      description={isStyle && id === 'add'
        ? t('workspace.sections.add.styleDescription')
        : isConceptual && s.conceptDescription
          ? t(`workspace.sections.${id}.conceptDescription`)
          : t(`workspace.sections.${id}.description`)} />;
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
      <span aria-hidden>🕸</span>
      <span className="text-sm font-medium">{t('workspace.scrapeLink.title')}</span>
      <span className="text-content-subtle text-[0.6875rem]">{t('workspace.scrapeLink.description')}</span>
      <span aria-hidden className="ml-auto text-content-subtle">→</span>
    </button>
  );

  return (
    <div className="flex flex-col gap-3">
      {/* ---- Header : identité du dataset + UNE action primaire (Export ZIP).
           Les actions secondaires de PARAMÉTRAGE (settings, fidélité) vivent dans
           le menu « ⋯ More » ; les actions de DONNÉES (backup, import-fusion,
           publish) vivent dans la section « Import & export » de la sidebar. ---- */}
      {/* relative z-30 : le header est un flex item ; sans stacking-context propre,
          le z-20 du menu « ⋯ More » resterait piégé sous les frères plus bas. */}
      <div className="relative z-30 flex items-center gap-2 flex-wrap">
        <button type="button" onClick={onBack}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-sm transition-colors">
          ← {t('nav.datasets')}
        </button>
        <h1 className="text-content font-bold">{d.name}</h1>
        {isStyle ? (
          <span title={t('workspace.header.styleTitle')}
            className="flex items-center gap-1 px-2 py-0.5 rounded-lg border border-cyan-400/40 bg-cyan-500/10 text-cyan-200 text-[0.6875rem]">
            {t('workspace.header.alwaysOnStyle')}
          </span>
        ) : (
          <button type="button"
            onClick={() => { try { navigator.clipboard.writeText(d.trigger_word || ''); } catch { /* ignore */ } }}
            title={t('workspace.header.copyTrigger')}
            className="flex items-center gap-1 px-2 py-0.5 rounded-lg border border-indigo-400/40 bg-indigo-500/10 text-[0.6875rem]">
            <span className="text-content-subtle">{t('workspace.header.trigger')}:</span>
            <code className="text-indigo-300 font-semibold">{d.trigger_word || '—'}</code>
            <span aria-hidden className="text-content-subtle">⧉</span>
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button type="button" disabled={!kept} onClick={exportZipGuarded}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            ⬇ {t('workspace.actions.exportZip')} ({kept})
          </button>
          {/* summary en display:flex → pas de marqueur natif ; les items restent
              montés en permanence (details ne fait que masquer l'affichage). */}
          <details className="relative">
            <summary
              title={t('workspace.header.moreTitle')}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-sm cursor-pointer select-none">
              ⋯ {t('workspace.header.more')}
            </summary>
            <div className="absolute right-0 top-full mt-1 z-20 w-72 rounded-lg border border-border bg-surface-overlay shadow-xl p-1.5 flex flex-col gap-0.5">
              <button type="button" onClick={() => setSettingsOpen(true)}
                title={isStyle ? t('workspace.header.editStyleTitle') : t('workspace.header.editSettingsTitle')}
                className={MENU_ITEM}>
                ⚙️ {t('workspace.header.editSettings')}
                <span className="ml-auto text-content-subtle text-[0.625rem]">
                  {isStyle
                    ? t('workspace.header.styleSettingsSummary')
                    : t('workspace.header.settingsSummary', {
                      concept: isConcept ? t('workspace.header.conceptSuffix') : '',
                    })}
                </span>
              </button>
              {!isConceptual && (
                <button type="button" disabled={ds.busy}
                  onClick={() => ds.setDatasetFidelity?.(bodyFid ? 'face' : 'body')}
                  title={bodyFid
                    ? t('workspace.header.bodyFidelityOnTitle')
                    : t('workspace.header.bodyFidelityOffTitle')}
                  className={`${MENU_ITEM} ${bodyFid ? 'text-emerald-300' : ''}`}>
                  🧍 {t('workspace.header.bodyFidelity')}
                  <span className={`ml-auto text-[0.625rem] ${bodyFid ? 'text-emerald-300 font-semibold' : 'text-content-subtle'}`}>
                    {bodyFid ? `✓ ${t('workspace.header.on')}` : t('workspace.header.off')}
                  </span>
                </button>
              )}
            </div>
          </details>
        </div>
      </div>

      {/* Two-column workspace: the persistent section sidebar on the left (with the
          guided Progress checklist below it for character datasets), the ACTIVE
          section's content on the right. On mobile the sidebar folds into a
          horizontal chip rail — same responsive pattern as the Settings page. */}
      <div className="lg:grid lg:grid-cols-[15rem_minmax(0,1fr)] lg:gap-4 lg:items-start">
        <aside>
          {/* Mobile: horizontal chip rail */}
          <nav aria-label={t('workspace.navigation.sections')} className="-mx-4 overflow-x-auto px-4 pb-2 lg:hidden">
            <ul className="m-0 flex list-none gap-2 p-0">
              {WORKSPACE_SECTIONS.map((s) => <li key={s.id}>{navItem(s, true)}</li>)}
            </ul>
          </nav>
          {activePanels.length > 0 && (
            <nav aria-label={t('workspace.navigation.destinations', {
              section: t(`workspace.sections.${section}.title`),
            })}
              className="-mx-4 -mt-1 overflow-x-auto px-4 pb-3 lg:hidden">
              <ul id={`dataset-mobile-panels-${section}`} className="m-0 flex list-none gap-2 p-0">
                {activePanels.map((destination) => (
                  <li key={destination.id}>{panelNavItem(section, destination, true)}</li>
                ))}
              </ul>
            </nav>
          )}
          {/* Desktop: sticky rail + guided progress below it */}
          <div className="hidden lg:sticky lg:top-20 lg:flex lg:flex-col lg:gap-3">
            <nav aria-label={t('workspace.navigation.sections')}>
              <p className="m-0 px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">
                {t('workspace.navigation.dataset')}
              </p>
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
            </div>
          )}

          {pending > 0 && (
            <div className="flex items-center gap-3 rounded-lg border-2 border-indigo-400/60 bg-indigo-500/15 px-3 py-2.5">
              <span className="animate-pulse text-lg" aria-hidden>⏳</span>
              <div className="flex flex-col">
                <span className="text-content text-sm font-semibold">
                  {t('workspace.generation.inProgress', { count: pending })}
                </span>
                <span className="text-content-subtle text-[0.6875rem]">
                  {t('workspace.generation.stopHint')}
                </span>
              </div>
              <button type="button" onClick={ds.cancelPending} disabled={ds.cancellingGeneration}
                title={t('workspace.generation.stopTitle')}
                className="ml-auto shrink-0 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-bold disabled:opacity-40">
                ⏹ {t(ds.cancellingGeneration
                  ? 'workspace.generation.stopping'
                  : 'workspace.generation.stop')}
              </button>
            </div>
          )}

          {/* ============ 🖼️ Images — la grille : triage ✓/✕, filtres, tri auto. */}
          <div className={sectionCls('images')}>
            {heading('images')}
            <p className="m-0 text-content-subtle text-[0.75rem] tabular-nums">
              {t('workspace.images.summary', { total: rescueGridImages.length, kept })}
              {triage > 0 ? <> · <span className="text-amber-300">{t('workspace.images.awaiting', { count: triage })}</span></> : ''}
              {rescueReviewCount > 0
                ? <> · <span className="text-indigo-300">{t('workspace.images.rescuePairs', { count: rescueReviewCount })}</span></>
                : ''}
              {kept > 0 ? ` · ${t('workspace.images.captioned', { done: keptCaptioned, total: kept })}` : ''}
              {watermarkDetected > 0 ? ` · ${t('workspace.images.watermarks', { count: watermarkDetected })}` : ''}
            </p>
            <div id="gf-images" className="scroll-mt-20 flex flex-col gap-2">
              {filtersActive && (
                <GridFilterBar excludes={excludeTags} includes={includeTags}
                  shown={gridImages.length} total={rescueGridImages.length}
                  onRemoveExclude={toggleExclude} onRemoveInclude={toggleInclude}
                  onClearAll={clearFilters} />
              )}
              {filtersActive && gridImages.length === 0 ? (
                // Filtered down to nothing: say so plainly (the grid's own "no images"
                // empty-state would read as "everything's gone", which would be a lie).
                <p className="rounded-lg border border-border bg-surface px-3 py-4 text-center text-content-subtle text-sm">
                  {t('workspace.images.noFilterMatches')}{' '}
                  <button type="button" onClick={clearFilters} className="underline hover:text-content">{t('workspace.images.clearAll')}</button>{' '}
                  {t('workspace.images.seeAllAgain', { count: images.length })}
                </p>
              ) : (
                <DatasetGrid images={gridImages} datasetId={d.id} onStatus={ds.setStatus} onCaption={ds.setCaption}
                  onCrop={setCropImg} onDelete={ds.deleteImage}
                  onMirror={ds.mirrorImage} mirroringIds={ds.mirroringIds}
                  onRegenerate={(id, loraStrength, prompt) => ds.regenerate(id, loraStrength, prompt)} onView={setViewImg}
                  onBatch={ds.batchImages} busy={ds.busy}
                  onImprove={ds.improveImage} onRefresh={ds.refresh}
                  kleinAvailable={Boolean(caps.engines?.klein)}
                  eligibilityImages={images}
                  nonces={ds.nonces} faceThresholds={d.face_thresholds} datasetKind={d.kind || 'character'}
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
              </div>
            ) : (
              <>
                <div id="gf-reference" className="scroll-mt-20">
                  <div id="ds-add-reference" tabIndex={-1} className="scroll-mt-20 flex flex-col gap-1">
                    <span className="text-content-subtle text-[0.6875rem]">
                      {t('workspace.reference.intro')}
                    </span>
                    <ReferencePanel refFilename={d.ref_filename} datasetId={d.id} onSetRef={ds.setRef}
                      onCropRef={() => setRefCrop(true)} busy={ds.busy} importBusy={importBusy} visionBusy={visionImportBusy} nonce={ds.refNonce}
                      extraRefs={d.ref_extra_filenames || []}
                      onAddExtraRef={ds.addExtraRef} onRemoveExtraRef={ds.removeExtraRef} />
                  </div>
                </div>

                <div id="gf-generate" className="scroll-mt-20 flex flex-col gap-2">
                  <CompositionBar composition={d.composition} upscaled={d.composition_upscaled} bodyFidelity={bodyFid} />
                  <div id="ds-add-generate" tabIndex={-1} className="scroll-mt-20">
                    <VariationCatalog key={`vc-${d.id}-${bodyFid}`} busy={ds.busy}
                      generating={act && act.kind === 'generate' ? act : null}
                      onCancelGeneration={ds.cancelPending}
                      cancellingGeneration={ds.cancellingGeneration}
                      onGenerate={(...args) => {
                        // Guard-rail: a batch is already in flight — launching another one
                        // on top is usually an accidental double-click, not a plan.
                        if (pending > 0 && !window.confirm(
                          t('workspace.confirm.generationAlreadyRunning', { count: pending }))) return;
                        ds.generate(...args);
                      }}
                      hasRef={!!d.ref_filename} composition={d.composition} images={images}
                      bodyFidelity={bodyFid}
                      promptSuffix={d.prompt_suffix || ''}
                      promptSuffixes={d.prompt_suffixes || null}
                      onSaveSuffixes={(patch) => ds.updateSettings(patch, { quiet: true })} />
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
                    onClick={ds.analyzeFaces} disabled={ds.busy || !d.ref_filename}
                    title={d.ref_filename
                      ? t('workspace.curationTools.faceTitle')
                      : t('workspace.curationTools.referenceFirst')}
                    className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border scroll-mt-20">
                    {ds.analyzing
                      ? `🎭 ${t('workspace.curationTools.analyzing')}${act?.kind === 'analyze_faces' && act.total ? ` ${act.done}/${act.total}` : ''}`
                      : `🎭 ${t('workspace.curationTools.analyzeFaces')}`}
                  </button>
                )}
                <div id="ds-curation-watermarks" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                {/* Watermark auto-correction (V1): find overlaid site logos/URLs/usernames on
                    the kept images, then Clean them (border → crop, small off-center → LaMa
                    inpaint, on-subject → manual review). Applies to any dataset kind. */}
                <button type="button" data-workspace-focus onClick={ds.findWatermarks} disabled={ds.busy}
                  title={t('workspace.curationTools.findTitle')}
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  {ds.watermarking
                    ? `🧽 ${t('workspace.curationTools.scanning')}${act?.kind === 'watermark_detect' && act.total ? ` ${act.done}/${act.total}` : ''}`
                    : `🧽 ${t('workspace.curationTools.findWatermarks')}`}
                </button>
                <HelpBadge topic="action-watermark-clean" />
                {watermarkDetected > 0 && (
                  <>
                  {/* Inpaint engine: LaMa (fast, non-generative) vs Klein (masked
                      Flux.2 inpaint — better on complex texture AND makes on-subject
                      marks actionable, but GPU + slower). Klein is greyed until ComfyUI
                      + the Klein models are ready (caps.watermark_klein). */}
                  <div role="group" aria-label={t('workspace.curationTools.method')}
                    className="flex items-center rounded-lg border border-border bg-surface p-0.5 text-xs">
                    <button type="button" aria-pressed={watermarkMethod === 'lama'}
                      onClick={() => setWatermarkMethod('lama')} disabled={ds.busy}
                      title={t('workspace.curationTools.lamaTitle')}
                      className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkMethod === 'lama'
                        ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                      LaMa <span className="font-normal opacity-70">{t('workspace.curationTools.fast')}</span>
                    </button>
                    <button type="button" aria-pressed={watermarkMethod === 'klein'}
                      onClick={() => setWatermarkMethod('klein')} disabled={ds.busy || !caps.watermark_klein}
                      title={caps.watermark_klein
                        ? t('workspace.curationTools.kleinTitle')
                        : t('workspace.curationTools.kleinUnavailable')}
                      className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkMethod === 'klein'
                        ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                      Klein <span className="font-normal opacity-70">{t('workspace.curationTools.quality')}</span>
                    </button>
                  </div>
                  {/* Allow auto-crop: the SAME persisted preference as Settings ▸ Watermark
                      inpainting (write-through). Off → a border mark is repainted (LaMa/
                      Klein) instead of cropped; the per-image review can still override it. */}
                  <button type="button" role="switch" aria-checked={allowAutoCrop}
                    onClick={() => setWatermarkAllowCrop(!allowAutoCrop)}
                    disabled={ds.busy || savingAllowCrop}
                    title={allowAutoCrop
                      ? t('workspace.curationTools.autoCropOnTitle')
                      : t('workspace.curationTools.autoCropOffTitle')}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold disabled:opacity-40 ${allowAutoCrop
                      ? 'border-border bg-surface text-content-subtle hover:text-content'
                      : 'border-amber-400/50 bg-amber-500/10 text-amber-200'}`}>
                    {allowAutoCrop
                      ? `✂ ${t('workspace.curationTools.autoCropOn')}`
                      : `✂ ${t('workspace.curationTools.autoCropOff')}`}
                  </button>
                  <button type="button"
                    onClick={() => { requestHelpTip('watermark-batch-clean'); ds.cleanWatermarks(watermarkMethod); }}
                    disabled={ds.busy || savingAllowCrop}
                    title={watermarkMethod === 'klein'
                      ? (allowAutoCrop
                        ? t('workspace.curationTools.cleanKleinCrop')
                        : t('workspace.curationTools.cleanKleinNoCrop'))
                      : caps.watermark_inpaint
                      ? (allowAutoCrop
                        ? t('workspace.curationTools.cleanLamaCrop')
                        : t('workspace.curationTools.cleanLamaNoCrop'))
                      : t('workspace.curationTools.cleanCropOnly')}
                    className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
                    🧽 {t('workspace.curationTools.clean', { count: watermarkDetected })}
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
                    title={t('workspace.curationTools.reviewTitle')}
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40 scroll-mt-20">
                    🔍 {t('workspace.curationTools.review', { count: watermarkDetected })}
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
                    title={t('workspace.curationTools.installTitle')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-dashed border-amber-400/50 bg-amber-500/5 text-amber-200/90 text-sm hover:bg-amber-500/10">
                    ⬇ {t('workspace.curationTools.install')}
                    <span className="text-content-subtle text-[0.625rem] font-normal">
                      {t('workspace.curationTools.installSize')}
                    </span>
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
                    <span aria-hidden className="text-lg leading-none">🧽</span>
                    <div className="flex flex-col">
                      <span className="text-amber-200 text-sm font-semibold">
                        {t('workspace.curationTools.installPanelTitle')}
                      </span>
                      <span className="text-content-subtle text-[0.6875rem]">
                        {t('workspace.curationTools.installDescriptionBefore')}{' '}
                        <code className="text-amber-200/90">simple-lama-inpainting</code>{' '}
                        {t('workspace.curationTools.installDescriptionAfter')}
                      </span>
                    </div>
                    <button type="button" onClick={() => setInstallInpaintOpen(false)}
                      className="ml-auto shrink-0 text-content-subtle hover:text-content text-sm"
                      aria-label={t('workspace.curationTools.closeInstaller')}>✕</button>
                  </div>
                  <InstallRunner action="watermark_inpaint"
                    buttonLabel={`⬇ ${t('workspace.curationTools.downloadInstall')}`}
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
                      if (window.confirm(t('workspace.curationTools.purgeConfirm', { count: unused }))) ds.purgeUnused();
                    }}
                    title={t('workspace.curationTools.purgeTitle')}
                    className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm disabled:opacity-40">
                    🧹 {t('workspace.curationTools.purge', { count: unused })}
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    {t('workspace.curationTools.purgeHelp')}
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
              <div id="ds-captions-generate" tabIndex={-1}
                className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-20">
                {!isConceptual && (
                  <select value={effCaptionMode} onChange={(e) => setCaptionMode(e.target.value)} disabled={ds.busy}
                    title={t('workspace.captions.modeTitle')}
                    className="px-2 py-1.5 rounded-lg bg-surface border border-border text-content text-[0.8125rem] disabled:opacity-40">
                    <option value="prose">📝 {t('workspace.captions.prose')}</option>
                    <option value="booru">🏷️ {t('workspace.captions.booru')}</option>
                  </select>
                )}
                <button type="button" data-workspace-focus
                  onClick={() => ds.caption(effCaptionMode)} disabled={ds.busy}
                  className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                  {ds.captioning
                    ? `✨ ${t('workspace.captions.captioning', { done: keptCaptioned, total: kept })}`
                    : `✨ ${t('workspace.captions.generate')}`}
                </button>
                <HelpBadge topic="action-caption-generate" />
                <button type="button" disabled={ds.busy || !keptCaptioned}
                  onClick={() => {
                    if (window.confirm(t(`workspace.captions.recaptionConfirm.${d.kind || 'character'}`, {
                      count: keptCaptioned,
                    }))) ds.recaption(effCaptionMode);
                  }}
                  title={isConcept
                    ? t('workspace.captions.recaptionTitle.concept')
                    : isStyle
                      ? t('workspace.captions.recaptionTitle.style')
                      : t('workspace.captions.recaptionTitle.character')}
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  🔄 {t('workspace.captions.recaption')}
                </button>
                <button type="button" data-workspace-focus
                  onClick={() => setCaptionOptionsOpen(true)} disabled={ds.busy}
                  title="Choose the caption engine, Ollama model and vocabulary, pull a new model, and add custom instructions — for this dataset"
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  ⚙️ Options
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
                    title={t('workspace.captions.styleStatusTitle')}>
                    {keptUncaptioned
                      ? `⚠ ${t('workspace.captions.styleMissing', { count: keptUncaptioned })}`
                      : `✅ ${t('workspace.captions.styleComplete', {
                          done: keptCaptioned,
                          total: kept,
                        })}`}
                  </span>
                ) : d.caption_leak && (
                  d.caption_leak.captioned > 0 ? (
                    <button id="ds-captions-leak-review" type="button" data-workspace-focus
                      onClick={toggleLeakReview}
                      aria-expanded={showLeaks}
                      title={d.caption_leak.leaking === 0
                        ? (isConcept
                            ? t('workspace.captions.leak.cleanTitleConcept')
                            : t('workspace.captions.leak.cleanTitleIdentity'))
                        : (isConcept
                            ? t('workspace.captions.leak.foundTitleConcept')
                            : t('workspace.captions.leak.foundTitleIdentity'))}
                      className={`ml-auto text-[0.8125rem] underline decoration-dashed scroll-mt-20 ${
                        d.caption_leak.leaking === 0
                          ? 'text-emerald-400 decoration-emerald-400/40'
                          : 'text-amber-400 decoration-amber-400/50'}`}>
                      {d.caption_leak.leaking === 0
                        ? `✅ ${t(isConcept
                            ? 'workspace.captions.leak.cleanConcept'
                            : 'workspace.captions.leak.cleanIdentity', {
                            count: d.caption_leak.captioned,
                          })}`
                        : `⚠️ ${t(isConcept
                            ? 'workspace.captions.leak.foundConcept'
                            : 'workspace.captions.leak.foundIdentity', {
                            leaking: d.caption_leak.leaking,
                            total: d.caption_leak.captioned,
                          })}`}
                      {' '}{showLeaks ? '▴' : '▾'}
                    </button>
                  ) : kept > 0 ? (
                    <button id="ds-captions-leak-review" type="button" data-workspace-focus
                      onClick={toggleLeakReview}
                      aria-expanded={showLeaks}
                      title={t(isConcept
                        ? 'workspace.captions.leak.noneTitleConcept'
                        : 'workspace.captions.leak.noneTitleIdentity')}
                      className="ml-auto text-content-subtle text-[0.8125rem] underline decoration-dashed decoration-border scroll-mt-20">
                      {t(isConcept
                        ? 'workspace.captions.leak.noneConcept'
                        : 'workspace.captions.leak.noneIdentity')} {showLeaks ? '▴' : '▾'}
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
                    <span aria-hidden className="text-base leading-none">🎭</span>
                    <div className="flex flex-col gap-1">
                      <span className="text-content font-semibold text-sm">
                        {t(isConcept
                          ? 'workspace.captions.leak.conceptTitle'
                          : 'workspace.captions.leak.identityTitle')}
                      </span>
                      {isConcept ? (
                        <p className="m-0 text-content-muted leading-relaxed">
                          {t('workspace.captions.leak.conceptHelp', {
                            trigger: d.trigger_word || t('workspace.captions.leak.yourTrigger'),
                            concept: d.concept_desc || t('common.none'),
                          })}
                        </p>
                      ) : (
                        <p className="m-0 text-content-muted leading-relaxed">
                          {t('workspace.captions.leak.identityHelp', {
                            trigger: d.trigger_word || t('workspace.captions.leak.yourTrigger'),
                          })}
                        </p>
                      )}
                    </div>
                    <button type="button" onClick={toggleLeakReview}
                      className="ml-auto shrink-0 text-content-subtle hover:text-content text-sm"
                      aria-label={t('common.close')}>✕</button>
                  </div>

                  {/* What was checked — the numbers behind the badge. */}
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-content-subtle tabular-nums">
                    <span>{t('workspace.captions.leak.checked', {
                      count: d.caption_leak?.captioned ?? 0,
                    })}</span>
                    <span className={d.caption_leak?.leaking ? 'text-amber-300' : 'text-emerald-400'}>
                      {t('workspace.captions.leak.leaking', { count: d.caption_leak?.leaking ?? 0 })}
                    </span>
                    <span className="text-content-subtle/70">{t('workspace.captions.leak.liveScan')}</span>
                  </div>

                  {/* Words the detector watches. Concept: derived from the description
                      (its words + their basic lexical field); character: the fixed regex. */}
                  <div className="flex flex-col gap-1">
                    <span className="text-content-subtle">{t('workspace.captions.leak.watched')}</span>
                    {isConcept ? (
                      <p className="m-0 text-content-muted leading-relaxed">
                        {t('workspace.captions.leak.conceptWatched', {
                          concept: d.concept_desc || t('common.none'),
                        })}
                      </p>
                    ) : (
                      <div className="flex flex-wrap gap-1.5">
                        {['hair', 'eyes', 'skin', 'features', 'faceShape',
                          ...(bodyFid ? ['bodyMarks'] : [])].map((key) => (
                          <span key={key} className="rounded-full bg-surface border border-border px-2 py-0.5 text-content-muted text-[0.6875rem]">
                            {t(`workspace.captions.leak.watchTerms.${key}`)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Why a green 0 is expected, not suspicious. */}
                  {d.caption_leak?.captioned === 0 ? (
                    <p className="m-0 text-content-subtle leading-relaxed">
                      {t('workspace.captions.leak.nothingChecked')}
                    </p>
                  ) : d.caption_leak?.leaking === 0 ? (
                    <p className="m-0 text-emerald-400/90 leading-relaxed">
                      {isConcept
                        ? `✅ ${t('workspace.captions.leak.allClearConcept', {
                            count: d.caption_leak?.captioned,
                          })}`
                        : `✅ ${t('workspace.captions.leak.allClearIdentity', {
                            count: d.caption_leak?.captioned,
                          })}`}
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
                                ? t('workspace.captions.leak.fixConcept', { count: d.caption_leak?.leaking })
                                : t('workspace.captions.leak.fixIdentity', { count: d.caption_leak?.leaking })}
                              <HelpBadge topic="action-recaption-targeted" className="ml-1" />
                            </span>
                            {leakingImages.length > 1 && (
                              <button type="button"
                                disabled={recaptionLocked}
                                onClick={() => ds.recaptionImages(leakingImages.map((i) => i.id), effCaptionMode)}
                                title={isConcept
                                  ? t('workspace.captions.leak.recaptionAllTitleConcept')
                                  : t('workspace.captions.leak.recaptionAllTitleIdentity')}
                                className="shrink-0 px-2.5 py-1 rounded-lg bg-amber-500/15 text-amber-200 text-[0.75rem] font-semibold border border-amber-400/40 hover:bg-amber-500/25 disabled:opacity-40">
                                🔄 {t('workspace.captions.leak.recaptionAll', { count: leakingImages.length })}
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
                              alt={img.variation_label || t('workspace.captions.datasetImage')} loading="lazy"
                              className="w-14 h-14 rounded-lg object-cover shrink-0 bg-black" />
                            <div className="flex-1 min-w-0 flex flex-col gap-1">
                              <textarea defaultValue={img.caption || ''} rows={2}
                                key={`${img.id}:${img.caption || ''}`}
                                onBlur={(e) => {
                                  if (e.target.value !== (img.caption || '')) ds.setCaption(img.id, e.target.value);
                                }}
                                aria-label={t('workspace.captions.captionOfImage', { id: img.id })}
                                className="w-full bg-app/60 border border-amber-400/30 rounded px-2 py-1 text-[0.6875rem] text-content resize-y" />
                              <button type="button"
                                disabled={recaptionLocked}
                                onClick={() => ds.recaptionImages([img.id], effCaptionMode)}
                                title={isConcept
                                  ? t('workspace.captions.leak.recaptionOneTitleConcept')
                                  : t('workspace.captions.leak.recaptionOneTitleIdentity')}
                                className="self-start px-2 py-0.5 rounded-lg bg-surface text-content text-[0.6875rem] border border-border hover:bg-surface-raised disabled:opacity-40">
                                {rowBusy
                                  ? `⏳ ${t('workspace.captions.recaptioning')}`
                                  : `🔄 ${t('workspace.captions.recaption')}`}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                      {leakingImages.length === 0 && (
                        <p className="m-0 text-emerald-400 text-[0.8125rem]">
                          ✅ {t('workspace.captions.leak.noneLeft')}
                        </p>
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
                  🔎 {t('workspace.captions.filterActive')}{' '}
                  <button type="button" onClick={() => setSection('images')}
                    className="underline hover:text-content">
                    {t('workspace.sections.images.title')}
                  </button>
                  {' '}({t('workspace.filters.showing', {
                    shown: gridImages.length,
                    total: images.length,
                  })}).
                </p>
              )}
            </div>
          </div>

          {/* ============ 📦 Import & export — fusionner un dataset existant ;
               sortir celui-ci (ZIP d'entraînement, backup portable, HF Hub). */}
          <div className={sectionCls('export')}>
            {heading('export')}
            <div id="gf-export" className="scroll-mt-20 flex flex-col gap-2">
              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">
                {t('workspace.dataTransfer.bringIn')}
              </span>
              <div id="ds-export-import" tabIndex={-1}
                className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-20">
                <button type="button" data-workspace-focus
                  onClick={() => zipInput.current?.click()} disabled={importBusy}
                  title={t('workspace.dataTransfer.importZipTitle')}
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                  📦 {t('workspace.dataTransfer.importZip')}
                </button>
                <button type="button" disabled={importBusy} onClick={importFolderPrompt}
                  title={t('workspace.dataTransfer.importFolderTitle')}
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                  📂 {t('workspace.dataTransfer.importFolder')}
                </button>
                <span className="text-content-subtle text-[0.6875rem]">
                  {t('workspace.dataTransfer.importHelp')}
                </span>
              </div>
              <input ref={zipInput} type="file" accept=".zip,application/zip" className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) ds.importDatasetZip(f);
                  e.target.value = '';
                }} />

              <span className="text-content-subtle text-[0.625rem] uppercase tracking-wide">
                {t('workspace.dataTransfer.getOut')}
              </span>
              <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
                <div id="ds-export-training-zip" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                  <button type="button" data-workspace-focus={kept ? '' : undefined}
                    disabled={!kept} onClick={exportZipGuarded}
                    className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                    ⬇ {t('workspace.dataTransfer.exportZip', { count: kept })}
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    {t('workspace.dataTransfer.exportZipHelp')}
                  </span>
                </div>
                <div id="ds-export-backup" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                  <button type="button" data-workspace-focus onClick={ds.exportBackup}
                    title={t('workspace.dataTransfer.backupTitle')}
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                    💾 {t('workspace.dataTransfer.backup')}
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    {t('workspace.dataTransfer.backupHelp')}
                  </span>
                </div>
                {caps.hf_publish && kept > 0 && (
                  <div id="ds-export-hugging-face" tabIndex={-1}
                    className="flex items-center gap-2 flex-wrap scroll-mt-20">
                    <button type="button" data-workspace-focus
                      onClick={() => setPublishHfOpen(true)}
                      title={t('workspace.dataTransfer.publishTitle')}
                      className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                      🤗 {t('workspace.dataTransfer.publish')}
                    </button>
                    <span className="text-content-subtle text-[0.6875rem]">
                      {t('workspace.dataTransfer.publishHelp')}
                    </span>
                  </div>
                )}
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
                  <span aria-hidden>🎛️</span>
                  <span className="text-content font-semibold text-sm">
                    {t('workspace.studioLauncher.title')}
                  </span>
                  {d.best_settings && (
                    <span className="text-amber-300 text-[0.6875rem]"
                      title={t('workspace.studioLauncher.savedSettings')}>
                      ★ {fmt(d.best_settings.strength)}
                    </span>
                  )}
                  <span className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-xs font-semibold">
                    ⤢ {t('workspace.studioLauncher.open')}
                  </span>
                </button>
              ) : (
                <p className="m-0 rounded-lg border border-border bg-surface px-3 py-2 text-content-muted text-sm">
                  {t('workspace.studioLauncher.configure')}
                </p>
              )}
              <div><HelpBadge topic="action-studio-open" /></div>
            </div>
          </div>
        </div>{/* /right column */}
      </div>{/* /workspace grid */}

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
          defaultAspect={1}
          onCancel={() => setRefCrop(false)}
          onConfirm={async (box) => { await ds.cropRef(box); setRefCrop(false); }}
          onReset={d.ref_original_filename
            ? async () => { await ds.recropRefAuto(); setRefCrop(false); }
            : undefined} />
      )}
      {viewImgLive && (
        <DatasetLightbox img={viewImgLive} datasetId={d.id}
          nonce={(ds.nonces && ds.nonces[viewImgLive.id]) || 0}
          onClose={() => setViewImg(null)}
          onMirror={viewImgLive._rescueReviewPreview ? undefined : ds.mirrorImage}
          mirrorBusy={Boolean(ds.mirroringIds?.has(viewImgLive.id))}
          onImprove={canImproveViewImg ? ds.improveImage : undefined}
          improvePending={viewImgImproving}
          improveReady={viewImgImprovementReady}
          busy={ds.busy}
          kleinAvailable={Boolean(caps.engines?.klein)}
          onCrop={viewImgLive._rescueReviewPreview
            ? undefined
            : (img) => { setViewImg(null); setCropImg(img); }} />
      )}
      {settingsOpen && (
        <DatasetSettingsModal d={d} busy={ds.busy}
          onSave={ds.updateSettings} onClose={() => setSettingsOpen(false)} />
      )}
      {publishHfOpen && (
        <PublishHfModal datasetId={d.id} onClose={() => setPublishHfOpen(false)} />
      )}
      {folderBrowseOpen && (
        <FolderBrowserModal
          onPick={(p) => ds.importDatasetFolder(p)}
          onClose={() => setFolderBrowseOpen(false)} />
      )}
      {captionOptionsOpen && (
        <CaptionOptionsPopover datasetId={d.id}
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
          onClose={(recap) => {
            setReviewQueue(null);
            const summary = recap || buildWatermarkRecap({});
            if (summary) toast.success(`Review done — ${summary}`);
          }} />
      )}
    </div>
  );
}
