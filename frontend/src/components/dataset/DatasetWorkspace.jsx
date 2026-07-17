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
import { recaptionConfirmation } from './captionCategory';
import CropModal from './CropModal';
import DatasetLightbox from './DatasetLightbox';
import DatasetSettingsModal from './DatasetSettingsModal';
import PublishHfModal from './PublishHfModal';
import WatermarkReviewLightbox, { buildWatermarkRecap } from './WatermarkReviewLightbox';
import { useToast } from '../common/Toast';
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
  return (
    <div role="status"
      className="flex items-center gap-2 flex-wrap rounded-lg border-2 border-amber-400/50 bg-amber-400/10 px-3 py-2">
      <span className="text-amber-200 text-sm font-semibold shrink-0">🔎 Filtered view</span>
      <span className="text-content-muted text-xs tabular-nums shrink-0">
        showing {shown} of {total}
      </span>
      <div className="flex items-center gap-1.5 flex-wrap">
        {excludes.map((t) => (
          <span key={`x-${t}`}
            className="inline-flex items-center gap-1 rounded-full border border-rose-400/50 bg-rose-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-rose-200">
            <span aria-hidden>⊘</span> {t}
            <button type="button" onClick={() => onRemoveExclude(t)}
              aria-label={`Stop hiding images tagged ${t}`}
              className="w-4 h-4 grid place-items-center rounded-full hover:bg-rose-500/30">✕</button>
          </span>
        ))}
        {includes.map((t) => (
          <span key={`i-${t}`}
            className="inline-flex items-center gap-1 rounded-full border border-indigo-400/50 bg-indigo-500/15 pl-2 pr-1 py-0.5 text-[0.6875rem] text-indigo-200">
            <span aria-hidden>◉</span> only {t}
            <button type="button" onClick={() => onRemoveInclude(t)}
              aria-label={`Stop isolating images tagged ${t}`}
              className="w-4 h-4 grid place-items-center rounded-full hover:bg-indigo-500/30">✕</button>
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
  const { caps, refresh: refreshCaps } = useCapabilities();
  const d = ds.data;
  const [cropImg, setCropImg] = useState(null);
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
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [publishHfOpen, setPublishHfOpen] = useState(false);
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

  useEffect(() => {
    if (!d || !panel || workspaceLocation.pending) return undefined;
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
  }, [d, section, panel, workspaceLocation.pending, landingRequest,
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
    reference: '📸 Go to reference', generate: '⚡ Go to generation', curate: '🖼️ Review the grid',
    caption: '✨ Caption the kept ones',
    finish: caps.training_visible ? '🎓 Go to training' : `⬇ Export ZIP (${kept})`,
    studio: '🎛️ Open Studio',
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
    if (triage && !window.confirm(`${triage} image(s) still await triage (✓/✕) and will NOT be in the ZIP. Export anyway?`)) return;
    if (isStyle && keptUncaptioned) {
      toast.error(`Style training needs a content-only caption for every kept image — ${keptUncaptioned} still missing.`);
      jumpTo({ targetId: 'gf-captions' });
      return;
    }
    if (keptUncaptioned && !window.confirm(`${keptUncaptioned} kept image(s) without a caption (trigger only). Export anyway?`)) return;
    ds.exportZip();
  };
  // The folder lives on the machine running the app, so a browser file-picker
  // can't select it — the user pastes the path instead.
  const importFolderPrompt = () => {
    const p = window.prompt(
      'Path of the dataset folder on this machine (images + same-name .txt captions):');
    if (p && p.trim()) ds.importDatasetFolder(p.trim());
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
    ? `${act?.detail || `Captioning in progress — ${keptCaptioned}/${kept} captioned…`} ComfyUI is paused.`
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
            watermark_detect: `Scanning for watermarks…${prog}`,
            watermark_clean: `Cleaning watermarks…${prog}`,
            caption: `Captioning…${prog}`,
            recaption: `Re-captioning…${prog}`,
            analyze_faces: `Analyzing faces…${prog}`,
            classify: `Classifying framing…${prog}`,
            generate: `Generating variations…${prog}`,
          }[act.kind];
          if (label) {
            const detailed = act.detail || label;
            return `${detailed}${cpu ? '' : ' ComfyUI is paused during the pass.'}`;
          }
        }
        return 'GPU processing in progress (analysis / cropping / captioning)… ComfyUI is paused during the pass.';
      })();

  // ── Sidebar : pastilles par section — ambre quand une action attend l'utilisateur,
  //    indigo pulsé quand des générations tournent, neutre pour l'info « à faire ».
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
        {destination.title}
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
      <span aria-hidden>🕸</span>
      <span className="text-sm font-medium">Scrape images from the web</span>
      <span className="text-content-subtle text-[0.6875rem]">scan a gallery URL, pick images, import full-frame</span>
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
          ← Datasets
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
            className="flex items-center gap-1 px-2 py-0.5 rounded-lg border border-indigo-400/40 bg-indigo-500/10 text-[0.6875rem]">
            <span className="text-content-subtle">trigger:</span>
            <code className="text-indigo-300 font-semibold">{d.trigger_word || '—'}</code>
            <span aria-hidden className="text-content-subtle">⧉</span>
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button type="button" disabled={!kept} onClick={exportZipGuarded}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            ⬇ Export ZIP ({kept})
          </button>
          {/* summary en display:flex → pas de marqueur natif ; les items restent
              montés en permanence (details ne fait que masquer l'affichage). */}
          <details className="relative">
            <summary
              title="More dataset actions — edit settings, body fidelity"
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-sm cursor-pointer select-none">
              ⋯ More
            </summary>
            <div className="absolute right-0 top-full mt-1 z-20 w-72 rounded-lg border border-border bg-surface-overlay shadow-xl p-1.5 flex flex-col gap-0.5">
              <button type="button" onClick={() => setSettingsOpen(true)}
                title={isStyle ? 'Edit the Style dataset name and review its always-on behavior.' : 'Edit the dataset name, trigger word, and (for concept datasets) the concept description that drives the caption avoid-list.'}
                className={MENU_ITEM}>
                ⚙️ Edit settings
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
                  🧍 Body fidelity
                  <span className={`ml-auto text-[0.625rem] ${bodyFid ? 'text-emerald-300 font-semibold' : 'text-content-subtle'}`}>
                    {bodyFid ? '✓ on' : 'off'}
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
          <nav aria-label="Dataset sections" className="-mx-4 overflow-x-auto px-4 pb-2 lg:hidden">
            <ul className="m-0 flex list-none gap-2 p-0">
              {WORKSPACE_SECTIONS.map((s) => <li key={s.id}>{navItem(s, true)}</li>)}
            </ul>
          </nav>
          {activePanels.length > 0 && (
            <nav aria-label={`${sectionMeta[section].title} destinations`}
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
            </div>
          )}

          {pending > 0 && (
            <div className="flex items-center gap-3 rounded-lg border-2 border-indigo-400/60 bg-indigo-500/15 px-3 py-2.5">
              <span className="animate-pulse text-lg" aria-hidden>⏳</span>
              <div className="flex flex-col">
                <span className="text-content text-sm font-semibold">
                  {pending} generation(s) in progress…
                </span>
                <span className="text-content-subtle text-[0.6875rem]">
                  First results look wrong? Stop now — the remaining API calls are skipped (not billed).
                </span>
              </div>
              <button type="button" onClick={ds.cancelPending} disabled={ds.busy}
                title="Cancels every generation still in flight; finished images stay."
                className="ml-auto shrink-0 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-bold disabled:opacity-40">
                ⏹ Stop generation
              </button>
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
                  No images match the active filter{excludeTags.length + includeTags.length > 1 ? 's' : ''} —{' '}
                  <button type="button" onClick={clearFilters} className="underline hover:text-content">clear all</button>{' '}
                  to see all {images.length} again.
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
                  nonces={ds.nonces} faceThresholds={d.face_thresholds} datasetKind={d.kind || 'character'} />
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
                      one clear photo of the face — every generated variation starts from it
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
                      onGenerate={(...args) => {
                        // Guard-rail: a batch is already in flight — launching another one
                        // on top is usually an accidental double-click, not a plan.
                        if (pending > 0 && !window.confirm(
                          `A generation batch is already running (${pending} in flight).\n\nLaunch another one anyway?`)) return;
                        ds.generate(...args);
                      }}
                      hasRef={!!d.ref_filename} composition={d.composition} images={images}
                      bodyFidelity={bodyFid} />
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
                    title={d.ref_filename ? "Scores each image's facial resemblance vs the reference (deletes nothing)" : "Set a reference photo first"}
                    className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border scroll-mt-20">
                    {ds.analyzing
                      ? `🎭 Analyzing…${act?.kind === 'analyze_faces' && act.total ? ` ${act.done}/${act.total}` : ''}`
                      : '🎭 Analyze faces'}
                  </button>
                )}
                <div id="ds-curation-watermarks" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                {/* Watermark auto-correction (V1): find overlaid site logos/URLs/usernames on
                    the kept images, then Clean them (border → crop, small off-center → LaMa
                    inpaint, on-subject → manual review). Applies to any dataset kind. */}
                <button type="button" data-workspace-focus onClick={ds.findWatermarks} disabled={ds.busy}
                  title="Scans the kept images for overlaid watermarks/logos/URLs added on top of the photo (deletes nothing)"
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  {ds.watermarking
                    ? `🧽 Scanning…${act?.kind === 'watermark_detect' && act.total ? ` ${act.done}/${act.total}` : ''}`
                    : '🧽 Find watermarks'}
                </button>
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
                        : 'Klein inpaint needs ComfyUI running + the Klein models installed (Setup ▸ ComfyUI).'}
                      className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${watermarkMethod === 'klein'
                        ? 'bg-amber-500/25 text-amber-100' : 'text-content-subtle hover:text-content'}`}>
                      Klein <span className="font-normal opacity-70">quality</span>
                    </button>
                  </div>
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
                    {allowAutoCrop ? '✂ Auto-crop on' : '✂ Auto-crop off'}
                  </button>
                  <button type="button"
                    onClick={() => { requestHelpTip('watermark-batch-clean'); ds.cleanWatermarks(watermarkMethod); }}
                    disabled={ds.busy || savingAllowCrop}
                    title={watermarkMethod === 'klein'
                      ? (allowAutoCrop
                        ? 'Removes them with masked Flux.2 Klein inpaint: border marks are cropped, every other mark (off-center AND on-subject) is repainted then composited back — only the mark changes'
                        : 'Auto-crop off: EVERY mark (border included) is repainted with masked Flux.2 Klein inpaint then composited back — only the mark changes')
                      : caps.watermark_inpaint
                      ? (allowAutoCrop
                        ? 'Removes them: border marks are cropped, small off-center marks are inpainted (LaMa), on-subject marks are flagged for manual review'
                        : 'Auto-crop off: border marks are repainted (LaMa) instead of cropped; large/on-subject marks are flagged for manual review')
                      : 'Removes border marks by cropping. Inpainting (LaMa) needs a one-time install — use ⬇ Install inpainting next to this button; off-center marks are skipped until then'}
                    className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
                    🧽 Clean ({watermarkDetected})
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
                    🔍 Review flagged ({watermarkDetected})
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
                    ⬇ Install inpainting
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
                    <span aria-hidden className="text-lg leading-none">🧽</span>
                    <div className="flex flex-col">
                      <span className="text-amber-200 text-sm font-semibold">Install watermark inpainting (LaMa)</span>
                      <span className="text-content-subtle text-[0.6875rem]">
                        Adds the <code className="text-amber-200/90">simple-lama-inpainting</code> package
                        (pulls a CPU torch — one-time download, ~hundreds of MB). No restart, no GPU:
                        once done, ⬇ inpaints small off-center marks instead of skipping them.
                      </span>
                    </div>
                    <button type="button" onClick={() => setInstallInpaintOpen(false)}
                      className="ml-auto shrink-0 text-content-subtle hover:text-content text-sm"
                      aria-label="Close the inpainting installer">✕</button>
                  </div>
                  <InstallRunner action="watermark_inpaint" buttonLabel="⬇ Download & install"
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
                    🧹 Purge rejected/failed ({unused})
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
              <div id="ds-captions-generate" tabIndex={-1}
                className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2 scroll-mt-20">
                {!isConceptual && (
                  <select value={effCaptionMode} onChange={(e) => setCaptionMode(e.target.value)} disabled={ds.busy}
                    title="Caption style — Prose (Z-Image) or Booru tags (SDXL booru-native, e.g. bigLove). Defaults to auto based on the dataset's type."
                    className="px-2 py-1.5 rounded-lg bg-surface border border-border text-content text-[0.8125rem] disabled:opacity-40">
                    <option value="prose">📝 Prose</option>
                    <option value="booru">🏷️ Booru tags</option>
                  </select>
                )}
                <button type="button" data-workspace-focus
                  onClick={() => ds.caption(effCaptionMode)} disabled={ds.busy}
                  className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                  {ds.captioning ? `✨ ${keptCaptioned}/${kept} captioned…` : '✨ Caption the kept ones'}
                </button>
                <HelpBadge topic="action-caption-generate" />
                <button type="button" disabled={ds.busy || !keptCaptioned}
                  onClick={() => {
                    if (window.confirm(recaptionConfirmation(d.kind || 'character', keptCaptioned))) ds.recaption(effCaptionMode);
                  }}
                  title={isConcept
                    ? "Re-generates every caption while keeping the recurring concept unspoken"
                    : isStyle
                      ? "Re-generates every caption as content-only text without naming the aesthetic"
                      : "Re-generates every caption without describing identity (face/hair)"}
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  🔄 Re-caption
                </button>
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
                      ? `⚠ ${keptUncaptioned} missing · content-only captions required · no trigger`
                      : `✅ ${keptCaptioned}/${kept} content-only captions · no trigger`}
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
                        ? `✅ 0 ${isConcept ? 'concept' : 'identity'} leaks · ${d.caption_leak.captioned} captions checked`
                        : `⚠️ ${d.caption_leak.leaking}/${d.caption_leak.captioned} captions leak ${isConcept ? 'the concept' : 'identity'}`}
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
                    <span aria-hidden className="text-base leading-none">🎭</span>
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
                          that describes <em>who the person is</em> — hair, eye or skin colour, facial
                          features. On a character LoRA these words must stay OUT of the captions: they
                          dilute the identity into the text instead of binding it to your trigger word{' '}
                          <code className="text-indigo-300">{d.trigger_word || 'your trigger'}</code>.
                        </p>
                      )}
                    </div>
                    <button type="button" onClick={toggleLeakReview}
                      className="ml-auto shrink-0 text-content-subtle hover:text-content text-sm" aria-label="Close">✕</button>
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
                        {['hair', 'eye colour', 'skin · complexion · freckles',
                          'jawline · eyebrows · facial features', 'face shape',
                          ...(bodyFid ? ['tattoos · scars · piercings (body fidelity)'] : [])].map((c) => (
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
                        ? <>✅ All clear — every caption describes the scene while leaving the concept
                          unspoken, so it will bind to your trigger. It’s a real result on {d.caption_leak?.captioned} caption(s),
                          not a check that didn’t run.</>
                        : <>✅ All clear — and this is expected. The app’s captioner is built to describe pose,
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
                                ? <>Captions naming the concept ({d.caption_leak?.leaking}) — remove the concept words, or 🔄 Re-caption. Edits save when you click away.</>
                                : <>Captions leaking identity ({d.caption_leak?.leaking}) — remove the highlighted words, or 🔄 Re-caption. Edits save when you click away.</>}
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
                                🔄 Re-caption all leaking ({leakingImages.length})
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
                              <button type="button"
                                disabled={recaptionLocked}
                                onClick={() => ds.recaptionImages([img.id], effCaptionMode)}
                                title={isConcept
                                  ? 'Re-generate this caption while keeping the concept unspoken'
                                  : 'Re-generate this caption without describing identity (face/hair)'}
                                className="self-start px-2 py-0.5 rounded-lg bg-surface text-content text-[0.6875rem] border border-border hover:bg-surface-raised disabled:opacity-40">
                                {rowBusy ? '⏳ Re-captioning…' : '🔄 Re-caption'}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                      {leakingImages.length === 0 && (
                        <p className="m-0 text-emerald-400 text-[0.8125rem]">✅ All clear — no leaking caption left.</p>
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
                  🔎 A tag filter is active — the filtered grid lives in{' '}
                  <button type="button" onClick={() => setSection('images')}
                    className="underline hover:text-content">Images</button>
                  {' '}(showing {gridImages.length} of {images.length}).
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
                  📦 Import dataset (ZIP)
                </button>
                <button type="button" disabled={importBusy} onClick={importFolderPrompt}
                  title="Merge an existing training dataset already on this machine's disk: a folder of images with kohya-style same-name .txt captions (subfolders included). Aspect kept, perceptual duplicates skipped."
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                  📂 Import from folder…
                </button>
                <span className="text-content-subtle text-[0.6875rem]">
                  merges images + same-name .txt captions in — duplicates are skipped
                </span>
              </div>
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
                    className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                    ⬇ Export ZIP ({kept})
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    kept images + captions, training-ready (kohya layout)
                  </span>
                </div>
                <div id="ds-export-backup" tabIndex={-1}
                  className="flex items-center gap-2 flex-wrap scroll-mt-20">
                  <button type="button" data-workspace-focus onClick={ds.exportBackup}
                    title="Full portable backup: all images with statuses, captions, scores and settings — restore it on any machine from the Datasets page."
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                    💾 Backup
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
                    onJump={(targetId) => jumpTo({ targetId })} />
                )}
                <TrainingPanel ds={ds} keptCount={kept} kind={d.kind}
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
                  <span className="text-content font-semibold text-sm">LoRA testing studio</span>
                  {d.best_settings && (
                    <span className="text-amber-300 text-[0.6875rem]" title="Saved winning settings">
                      ★ {fmt(d.best_settings.strength)}
                    </span>
                  )}
                  <span className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-xs font-semibold">
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
