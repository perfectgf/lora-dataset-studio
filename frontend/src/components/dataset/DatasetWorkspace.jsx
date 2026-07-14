import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import CompositionBar from './CompositionBar';
import ReferencePanel from './ReferencePanel';
import VariationCatalog from './VariationCatalog';
import TrainingPanel from './TrainingPanel';
import { fmt } from '../../utils/studioFormat';
import ImportDropzone from './ImportDropzone';
import ConceptSourcesPanel from './ConceptSourcesPanel';
import DatasetGrid from './DatasetGrid';
import CaptionToolsBar from './CaptionToolsBar';
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
import { WORKSPACE_SECTIONS, SECTION_FOR_TARGET, isWorkspaceSection } from './workspaceSections';

// Style partagé des items du menu « ⋯ More » du header (actions secondaires).
const MENU_ITEM = 'w-full flex items-center gap-2 text-left px-2.5 py-1.5 rounded-md text-sm text-content hover:bg-surface-raised disabled:opacity-40';

/* En-tête de section (miroir visuel du SectionHeader de Settings, en h2 : le h1
   de la page reste le nom du dataset) : eyebrow mono + titre + description. */
function SectionHeading({ eyebrow, title, description }) {
  return (
    <div>
      <p className="m-0 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">{eyebrow}</p>
      <h2 className="m-0 mt-0.5 text-content text-base font-semibold">{title}</h2>
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
  const [installInpaintOpen, setInstallInpaintOpen] = useState(false);  // panneau d'install LaMa
  const [checkpointCount, setCheckpointCount] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [publishHfOpen, setPublishHfOpen] = useState(false);
  // Grid tag-filter (session-only): tags whose images are hidden (exclude) or the
  // ONLY tags allowed through (include). Both are normalized (trim+lowercase).
  const [excludeTags, setExcludeTags] = useState([]);
  const [includeTags, setIncludeTags] = useState([]);
  // ── Section active de la sidebar — persistée dans la query du hash
  //    (#/datasets?section=captions) pour survivre au reload, comme les pages
  //    Settings/Guide persistent la leur dans le path. Valeur inconnue → défaut.
  const [searchParams, setSearchParams] = useSearchParams();
  const rawSection = searchParams.get('section');
  const section = isWorkspaceSection(rawSection) ? rawSection : 'images';
  const setSection = (id) => {
    if (id === section) return;
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      p.set('section', id);
      return p;
    });
  };
  // Hooks must run unconditionally on every render — deriveSteps() null-guards `d`,
  // so this is safe to call before the loading early-return below.
  const { steps, nextStep } = useGuidedFlow(d, caps, checkpointCount);
  // Filters are per-dataset & transient — drop them when switching datasets so they
  // never leak from one dataset to the next.
  useEffect(() => { setExcludeTags([]); setIncludeTags([]); }, [d?.id]);
  if (!d) return <p className="text-content-subtle text-sm">Loading…</p>;

  const images = d.images || [];
  // Dataset CONCEPT : on masque tout ce qui est identité/visage (référence, générateur
  // de variations, analyse faciale, badge de fuite, composition, flux guidé) — il ne
  // reste que import brut → curation → caption (inversée) → entraînement.
  // 'style' suit le même chemin UI que concept : pas de référence/visage/composition,
  // juste import brut → curation → caption (contenu pur, optionnelle) → entraînement.
  const concept = d.kind === 'concept' || d.kind === 'style';
  // Leak check is KIND-specific (see the caption-leak panel): character flags identity,
  // concept flags the caption NAMING the concept (must bind to the trigger), style never
  // (its subjects' description IS the content). `concept` above stays "concept OR style"
  // for the shared layout gating.
  const isConcept = d.kind === 'concept';
  const isStyle = d.kind === 'style';
  // Fidélité corps : captions bannissent aussi les marques corporelles, composition
  // cible plus de bustes/corps, import plein cadre par défaut.
  const bodyFid = d.fidelity === 'body';
  const kept = images.filter((i) => i.status === 'keep').length;
  const unused = images.filter((i) => i.status === 'reject' || i.status === 'failed').length;
  const keptUncaptioned = images.filter((i) => i.status === 'keep' && !i.caption).length;
  const keptCaptioned = kept - keptUncaptioned;
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
  const gridImages = filterImages(images, {
    excludes: excludeTags, includes: includeTags, mode: effCaptionMode });
  const pending = images.filter((i) => i.status === 'pending' && !i.filename).length;
  const triage = images.filter((i) => i.status === 'pending' && i.filename).length;   // generated, awaiting ✓/✕

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
  const viewImgLive = viewImg ? (images.find((i) => i.id === viewImg.id) || viewImg) : null;

  // Export ZIP — shared by the header CTA and the Import & export row.
  // Guard-rails: untriaged images are silently EXCLUDED from the zip,
  // and uncaptioned kept ones export as trigger-only.
  const exportZipGuarded = () => {
    if (triage && !window.confirm(`${triage} image(s) still await triage (✓/✕) and will NOT be in the ZIP. Export anyway?`)) return;
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
  const activityBanner = ds.captioning
    ? `Captioning in progress — ${keptCaptioned}/${kept} captioned… ComfyUI is paused.`
    : (() => {
        if (act) {
          const prog = act.total ? ` ${act.done}/${act.total}` : '';
          // Passes that DON'T claim "ComfyUI is paused": the CPU ones, plus
          // 'generate' (engine-dependent — Nano Banana / ChatGPT don't touch
          // ComfyUI, and the Klein case is obvious from the tiles appearing).
          const cpu = act.kind === 'analyze_faces' || act.kind === 'watermark_clean'
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
          if (label) return `${label}${cpu ? '' : ' ComfyUI is paused during the pass.'}`;
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
    curation: watermarkDetected > 0
      ? { n: watermarkDetected, tone: 'amber', srLabel: `${watermarkDetected} watermark(s) to review` } : null,
    captions: (!isStyle && (d.caption_leak?.leaking ?? 0) > 0)
      ? { n: d.caption_leak.leaking, tone: 'amber', srLabel: `${d.caption_leak.leaking} caption(s) leaking` }
      : keptUncaptioned > 0
        ? { n: keptUncaptioned, tone: 'subtle', srLabel: `${keptUncaptioned} kept image(s) without a caption` } : null,
    export: null,
    training: null,
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
      <button key={s.id} type="button" onClick={() => setSection(s.id)}
        aria-current={isActive ? 'page' : undefined} className={base}>
        {!chip && isActive && (
          <span aria-hidden className="absolute bottom-1.5 left-0 top-1.5 w-0.5 rounded bg-gradient-primary" />
        )}
        <span aria-hidden>{s.icon}</span>
        <span>{s.title}</span>
        <NavBadge badge={navBadges[s.id]} />
      </button>
    );
  };

  const sectionMeta = Object.fromEntries(WORKSPACE_SECTIONS.map((s) => [s.id, s]));
  const heading = (id) => {
    const s = sectionMeta[id];
    return <SectionHeading eyebrow={s.eyebrow} title={s.title}
      description={concept && s.conceptDescription ? s.conceptDescription : s.description} />;
  };
  // Sections inactives : montées mais masquées (display:none) — les polls et
  // états internes survivent au changement de section (le poll 10 s du
  // TrainingPanel fait AVANCER la file d'entraînement côté serveur, il ne doit
  // jamais s'arrêter parce qu'on regarde la grille ; idem sélection de la
  // grille, panneaux dépliés, catalogue de variations).
  const sectionCls = (id) => (section === id ? 'flex flex-col gap-3' : 'hidden');

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
        <button type="button"
          onClick={() => { try { navigator.clipboard.writeText(d.trigger_word || ''); } catch { /* ignore */ } }}
          title="Copy the trigger word (to put in your prompts)"
          className="flex items-center gap-1 px-2 py-0.5 rounded-lg border border-indigo-400/40 bg-indigo-500/10 text-[0.6875rem]">
          <span className="text-content-subtle">trigger:</span>
          <code className="text-indigo-300 font-semibold">{d.trigger_word || '—'}</code>
          <span aria-hidden className="text-content-subtle">⧉</span>
        </button>
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
            <div className="absolute right-0 top-full mt-1 z-20 w-72 rounded-lg border border-border bg-surface shadow-xl p-1.5 flex flex-col gap-0.5">
              <button type="button" onClick={() => setSettingsOpen(true)}
                title="Edit the dataset name, trigger word, and (for concept datasets) the concept description that drives the caption avoid-list."
                className={MENU_ITEM}>
                ⚙️ Edit settings
                <span className="ml-auto text-content-subtle text-[0.625rem]">
                  name · trigger{concept ? ' · concept' : ''}
                </span>
              </button>
              {!concept && (
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
          <nav aria-label="Dataset sections" className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-3 lg:hidden">
            {WORKSPACE_SECTIONS.map((s) => navItem(s, true))}
          </nav>
          {/* Desktop: sticky rail + guided progress below it */}
          <div className="hidden lg:sticky lg:top-20 lg:flex lg:flex-col lg:gap-3">
            <nav aria-label="Dataset sections">
              <p className="m-0 px-3 pb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-content-subtle">Dataset</p>
              <div className="flex flex-col gap-0.5">
                {WORKSPACE_SECTIONS.map((s) => navItem(s, false))}
              </div>
            </nav>
            {!concept && (
              <GuidedChecklist steps={steps} currentId={nextStep ? nextStep.id : null} onJump={jumpTo} />
            )}
          </div>
        </aside>

        <div className="flex flex-col gap-3 min-w-0 mt-1 lg:mt-0">
          {/* ---- Bandeaux GLOBAUX : visibles quelle que soit la section active
               (une passe GPU ou un batch de générations concernent tout l'écran). ---- */}
          {!concept && (
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
              {images.length} image(s) · {kept} kept
              {triage > 0 ? <> · <span className="text-amber-300">{triage} awaiting ✓/✕</span></> : ''}
              {kept > 0 ? ` · ${keptCaptioned}/${kept} captioned` : ''}
              {watermarkDetected > 0 ? ` · ${watermarkDetected} watermark(s) flagged` : ''}
            </p>
            <div id="gf-images" className="scroll-mt-20 flex flex-col gap-2">
              {filtersActive && (
                <GridFilterBar excludes={excludeTags} includes={includeTags}
                  shown={gridImages.length} total={images.length}
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
                  onRegenerate={(id, loraStrength, prompt) => ds.regenerate(id, loraStrength, prompt)} onView={setViewImg}
                  onBatch={ds.batchImages} busy={ds.busy}
                  nonces={ds.nonces} faceThresholds={d.face_thresholds} />
              )}
            </div>
          </div>

          {/* ============ 📸 Add images — constituer le dataset. Concept : sources
               scrapées + import brut. Personnage : référence puis génération/import. */}
          <div className={sectionCls('add')}>
            {heading('add')}
            {concept ? (
              // Concept : pas de photo de référence ni de générateur — on peuple le dataset
              // en scannant des galeries (ConceptSourcesPanel) et/ou par upload manuel.
              <div id="gf-reference" className="scroll-mt-20 flex flex-col gap-2">
                <ConceptSourcesPanel onImport={ds.scrapeImport} busy={ds.busy} />
                <ImportDropzone onImport={(f) => ds.importFiles(f)} busy={ds.busy} />
              </div>
            ) : (
              <>
                <div id="gf-reference" className="scroll-mt-20 flex flex-col gap-1">
                  <span className="text-content-subtle text-[0.6875rem]">
                    one clear photo of the face — every generated variation starts from it
                  </span>
                  <ReferencePanel refFilename={d.ref_filename} datasetId={d.id} onSetRef={ds.setRef}
                    onCropRef={() => setRefCrop(true)} busy={ds.busy} nonce={ds.refNonce}
                    extraRefs={d.ref_extra_filenames || []}
                    onAddExtraRef={ds.addExtraRef} onRemoveExtraRef={ds.removeExtraRef} />
                </div>

                <div id="gf-generate" className="scroll-mt-20 flex flex-col gap-2">
                  <CompositionBar composition={d.composition} upscaled={d.composition_upscaled} bodyFidelity={bodyFid} />
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
                  {/* Head-crop optional: ON tags framing='face' at import (I2); OFF keeps
                      the original framing so bust/body photos import as-is. Body-fidelity
                      datasets default OFF (full frames are the point) — key remounts the
                      dropzone so the default follows a fidelity switch. */}
                  <ImportDropzone key={`${d.id}-${bodyFid}`} onImport={(f, o) => ds.importFiles(f, o)}
                    busy={ds.busy} cropOption defaultCrop={!bodyFid} />
                  {/* Scraper (character datasets too): scan a gallery URL → pick → import
                      full-frame — then crop each tile manually (✂ on the card). Collapsed
                      by default to keep the reference/generate flow prominent. */}
                  <details className="rounded-lg border border-border bg-surface open:pb-3">
                    <summary className="cursor-pointer select-none px-3 py-2 text-sm text-content font-semibold">
                      🕸 Scrape images from the web
                      <span className="ml-2 font-normal text-content-subtle text-[0.6875rem]">
                        scan a gallery URL, pick images, import full-frame — crop them afterwards
                      </span>
                    </summary>
                    <div className="px-3">
                      <ConceptSourcesPanel onImport={ds.scrapeImport} busy={ds.busy} />
                    </div>
                  </details>
                </div>
              </>
            )}
          </div>

          {/* ============ 🧹 Curation — passes de qualité sur les images gardées :
               ressemblance faciale, watermarks (find → clean → review), purge. */}
          <div className={sectionCls('curation')}>
            {heading('curation')}
            <div id="gf-curation" className="scroll-mt-20 flex flex-col gap-2">
              <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
                {!concept && (
                  <button type="button" onClick={ds.analyzeFaces} disabled={ds.busy || !d.ref_filename}
                    title={d.ref_filename ? "Scores each image's facial resemblance vs the reference (deletes nothing)" : "Set a reference photo first"}
                    className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                    {ds.analyzing
                      ? `🎭 Analyzing…${act?.kind === 'analyze_faces' && act.total ? ` ${act.done}/${act.total}` : ''}`
                      : '🎭 Analyze faces'}
                  </button>
                )}
                {/* Watermark auto-correction (V1): find overlaid site logos/URLs/usernames on
                    the kept images, then Clean them (border → crop, small off-center → LaMa
                    inpaint, on-subject → manual review). Applies to any dataset kind. */}
                <button type="button" onClick={ds.findWatermarks} disabled={ds.busy}
                  title="Scans the kept images for overlaid watermarks/logos/URLs added on top of the photo (deletes nothing)"
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40 border border-border">
                  {ds.watermarking
                    ? `🧽 Scanning…${act?.kind === 'watermark_detect' && act.total ? ` ${act.done}/${act.total}` : ''}`
                    : '🧽 Find watermarks'}
                </button>
                {watermarkDetected > 0 && (
                  <button type="button" onClick={ds.cleanWatermarks} disabled={ds.busy}
                    title={caps.watermark_inpaint
                      ? 'Removes them: border marks are cropped, small off-center marks are inpainted (LaMa), on-subject marks are flagged for manual review'
                      : 'Removes border marks by cropping. Inpainting (LaMa) needs a one-time install — use ⬇ Install inpainting next to this button; off-center marks are skipped until then'}
                    className="px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-400/40 text-amber-200 text-sm font-semibold disabled:opacity-40">
                    🧽 Clean ({watermarkDetected})
                  </button>
                )}
                {/* Per-image control: step through the flagged images full-screen, see each
                    detected box, and Clean / dismiss (false positive) / reject one by one.
                    The auto-detect has false positives, so this hands the final call to the
                    user (crucial after the "64/75 flagged" real-dataset run). */}
                {watermarkDetected > 0 && (
                  <button type="button" disabled={ds.busy}
                    onClick={() => setReviewQueue(images.filter((i) => i.watermark_state === 'detected'))}
                    title="Step through the flagged images one by one — see each detected box and Clean, dismiss a false positive, or reject"
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
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
                <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
                  <button type="button" disabled={ds.busy}
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
              <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
                {!concept && (
                  <select value={effCaptionMode} onChange={(e) => setCaptionMode(e.target.value)} disabled={ds.busy}
                    title="Caption style — Prose (Z-Image) or Booru tags (SDXL booru-native, e.g. bigLove). Defaults to auto based on the dataset's type."
                    className="px-2 py-1.5 rounded-lg bg-surface border border-border text-content text-[0.8125rem] disabled:opacity-40">
                    <option value="prose">📝 Prose</option>
                    <option value="booru">🏷️ Booru tags</option>
                  </select>
                )}
                <button type="button" onClick={() => ds.caption(effCaptionMode)} disabled={ds.busy}
                  className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                  {ds.captioning ? `✨ ${keptCaptioned}/${kept} captioned…` : '✨ Caption the kept ones'}
                </button>
                <button type="button" disabled={ds.busy || !keptCaptioned}
                  onClick={() => {
                    if (window.confirm(`Re-captioning overwrites the ${keptCaptioned} existing caption(s) (new prompt, no face description). Continue?`)) ds.recaption(effCaptionMode);
                  }}
                  title="Re-generates all captions with the prompt that doesn't describe identity (face/hair)"
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
                  <span className="ml-auto text-content-subtle text-[0.8125rem]"
                    title="A style LoRA describes its subjects freely — those words are the controllable content, not a leak. No leak check applies to a style set.">
                    leak check: not applicable to a style set
                  </span>
                ) : d.caption_leak && (
                  d.caption_leak.captioned > 0 ? (
                    <button type="button" onClick={() => setShowLeaks((v) => !v)}
                      aria-expanded={showLeaks}
                      title={d.caption_leak.leaking === 0
                        ? (isConcept
                            ? "0 captions name the concept — it binds to the trigger. Click for what was checked and why."
                            : "0 captions describe hair/face/skin — identity binds to the trigger. Click for what was checked and why.")
                        : (isConcept
                            ? "These captions name the concept → it won't bind to the trigger. Click to see what's watched and fix them here."
                            : "These captions mention hair/face/skin → identity won't bind to the trigger. Click to see what's watched and fix them here.")}
                      className={`ml-auto text-[0.8125rem] underline decoration-dashed ${
                        d.caption_leak.leaking === 0
                          ? 'text-emerald-400 decoration-emerald-400/40'
                          : 'text-amber-400 decoration-amber-400/50'}`}>
                      {d.caption_leak.leaking === 0
                        ? `✅ 0 ${isConcept ? 'concept' : 'identity'} leaks · ${d.caption_leak.captioned} captions checked`
                        : `⚠️ ${d.caption_leak.leaking}/${d.caption_leak.captioned} captions leak ${isConcept ? 'the concept' : 'identity'}`}
                      {' '}{showLeaks ? '▴' : '▾'}
                    </button>
                  ) : kept > 0 ? (
                    <button type="button" onClick={() => setShowLeaks((v) => !v)}
                      aria-expanded={showLeaks}
                      title={`The ${isConcept ? 'concept' : 'identity'}-leak scan runs on captions. Caption the kept images first. Click to learn what it checks.`}
                      className="ml-auto text-content-subtle text-[0.8125rem] underline decoration-dashed decoration-border">
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
                    <button type="button" onClick={() => setShowLeaks(false)}
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
                      <span className="text-amber-300 text-[0.8125rem] font-semibold">
                        {isConcept
                          ? <>Captions naming the concept ({d.caption_leak?.leaking}) — remove the concept words, or 🔄 Re-caption. Edits save when you click away.</>
                          : <>Captions leaking identity ({d.caption_leak?.leaking}) — remove the highlighted words, or 🔄 Re-caption. Edits save when you click away.</>}
                      </span>
                      {images.filter((i) => i.leak).map((img) => (
                        <div key={img.id} className="flex gap-2 items-start">
                          <img src={`/api/dataset/${d.id}/img/${encodeURIComponent(img.filename)}`}
                            alt={img.variation_label || 'dataset image'} loading="lazy"
                            className="w-14 h-14 rounded-lg object-cover shrink-0 bg-black" />
                          <textarea defaultValue={img.caption || ''} rows={2}
                            onBlur={(e) => {
                              if (e.target.value !== (img.caption || '')) ds.setCaption(img.id, e.target.value);
                            }}
                            aria-label={`Caption of image ${img.id}`}
                            className="flex-1 bg-app/60 border border-amber-400/30 rounded px-2 py-1 text-[0.6875rem] text-content resize-y" />
                        </div>
                      ))}
                      {images.filter((i) => i.leak).length === 0 && (
                        <p className="m-0 text-emerald-400 text-[0.8125rem]">✅ All clear — no leaking caption left.</p>
                      )}
                    </div>
                  )}
                </div>
              )}

              <CaptionToolsBar images={images} trainType={d.train_type} mode={effCaptionMode}
                excludes={excludeTags} includes={includeTags}
                onExclude={toggleExclude} onInclude={toggleInclude}
                onReplace={ds.replaceCaptions}
                onWriteFiles={ds.writeCaptionFiles} onOpenFolder={ds.openDatasetFolder} busy={ds.busy} />
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
              <div className="flex items-center gap-2 flex-wrap rounded-lg border border-border bg-surface px-3 py-2">
                <button type="button" onClick={() => zipInput.current?.click()} disabled={ds.busy}
                  title="Merge an existing training dataset into this one: a ZIP of images with kohya-style same-name .txt captions (any folder layout). Aspect kept, perceptual duplicates skipped."
                  className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm disabled:opacity-40">
                  📦 Import dataset (ZIP)
                </button>
                <button type="button" disabled={ds.busy} onClick={importFolderPrompt}
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
                <div className="flex items-center gap-2 flex-wrap">
                  <button type="button" disabled={!kept} onClick={exportZipGuarded}
                    className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
                    ⬇ Export ZIP ({kept})
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    kept images + captions, training-ready (kohya layout)
                  </span>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <button type="button" onClick={ds.exportBackup}
                    title="Full portable backup: all images with statuses, captions, scores and settings — restore it on any machine from the Datasets page."
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm">
                    💾 Backup
                  </button>
                  <span className="text-content-subtle text-[0.6875rem]">
                    portable copy — restore it on any machine from the Datasets page
                  </span>
                </div>
                {caps.hf_publish && kept > 0 && (
                  <div className="flex items-center gap-2 flex-wrap">
                    <button type="button" onClick={() => setPublishHfOpen(true)}
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

          {/* ============ 🎓 Training — readiness, panneau complet, Studio de test. */}
          <div className={sectionCls('training')}>
            {heading('training')}
            <div id="gf-training" className="scroll-mt-20 flex flex-col gap-2">
              {/* Pastille de préparation (miroir du preflight) : refreshKey borné aux
                  compteurs pertinents → pas de re-fetch à chaque poll du dataset. */}
              {caps.training_visible && (
                <TrainingReadiness datasetId={d.id} trainType={d.train_type}
                  refreshKey={`${kept}|${keptCaptioned}|${pending}|${triage}|${d.caption_leak?.leaking ?? ''}`}
                  onJump={(targetId) => jumpTo({ targetId })} />
              )}
              <TrainingPanel ds={ds} keptCount={kept} kind={d.kind} onCheckpointsChange={setCheckpointCount} />

              {/* Lanceur du Studio de test LoRA : page dédiée plein écran /studio?dataset=
                  (le LoRA du dataset y est pré-coché). Le dataset ouvert est persisté
                  (useDataset) → « ← Retour au Dataset Maker » rouvre ce workspace.
                  Hidden when ComfyUI isn't reachable — the Studio needs it to generate. */}
              {caps.studio_visible && (
                <button type="button" onClick={() => navigate(`/studio?dataset=${d.id}`)}
                  className="flex items-center gap-2 rounded-lg border border-purple-500/30 bg-purple-500/5 px-3 py-2.5 text-left hover:bg-purple-500/10 transition-colors">
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
              )}
            </div>
          </div>
        </div>{/* /right column */}
      </div>{/* /workspace grid */}

      {cropImg && cropImg.filename && (
        <CropModal imageUrl={`/api/dataset/${d.id}/img/${encodeURIComponent(cropImg.filename)}`}
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
          onCrop={(img) => { setViewImg(null); setCropImg(img); }} />
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
          onClean={(id) => ds.cleanWatermarkImages([id])}
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
