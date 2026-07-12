import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
import { useCapabilities } from '../../context/CapabilitiesContext';
import GuidedChecklist from './GuidedChecklist';
import NextStepCard from './NextStepCard';
import TrainingReadiness from './TrainingReadiness';
import useGuidedFlow from '../../hooks/useGuidedFlow';

/* Chaque étape du workflow est une CARTE distincte plutôt qu'un simple titre
   flottant : un bandeau-titre teinté (numéro, ou ✓ vert une fois l'étape faite)
   posé sur un corps encadré. La carte COURANTE porte un anneau indigo + une
   pastille « You are here ». Une étape TERMINÉE se replie sur son bandeau (avec
   un résumé) — repli calculé à l'ouverture du dataset, l'étape en cours restant
   toujours dépliée. L'état ouvert/fermé est piloté par le parent (`open`) pour
   qu'un saut depuis la checklist puisse ré-ouvrir une carte repliée. `id` +
   `scroll-mt` restent sur la <section> (saut + flash gf-highlight). */
function StepSection({ n, title, help, id, active, done, summary, open, onToggle, children }) {
  const collapsible = !active;   // l'étape en cours ne se replie jamais
  return (
    <section id={id}
      className={`scroll-mt-20 rounded-lg border bg-surface overflow-hidden transition-colors ${
        active ? 'border-primary/50 ring-1 ring-primary/25' : 'border-border'}`}>
      <button type="button"
        onClick={collapsible ? onToggle : undefined}
        aria-expanded={collapsible ? open : undefined}
        className={`w-full text-left flex items-center gap-2.5 flex-wrap px-4 py-2.5 border-b bg-surface-raised transition-colors ${
          active ? 'border-primary/40 cursor-default' : 'border-border hover:bg-white/[0.06]'}`}>
        <span aria-hidden
          className={`grid place-items-center w-7 h-7 shrink-0 rounded-full border text-sm font-bold ${
            done ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-300'
              : active ? 'bg-primary/25 border-primary/60 text-indigo-200'
                : 'bg-primary/10 border-primary/30 text-indigo-300'}`}>
          {done ? '✓' : n}
        </span>
        <h2 className="m-0 text-content font-semibold text-sm">{title}</h2>
        {/* Déplié → l'aide pédagogique ; replié → le résumé de ce qui est fait. */}
        {open
          ? (help && <span className="text-content-subtle text-[0.6875rem]">{help}</span>)
          : (summary && <span className="text-content-muted text-[0.6875rem] tabular-nums">{summary}</span>)}
        <span className="ml-auto shrink-0 flex items-center gap-2">
          {active && (
            <span className="rounded-full bg-primary/15 border border-primary/40 px-2 py-0.5 text-indigo-200 text-[0.625rem] font-semibold uppercase tracking-wide">
              You are here
            </span>
          )}
          {collapsible && (
            <span aria-hidden className="text-content-subtle text-xs">{open ? '▾' : '▸'}</span>
          )}
        </span>
      </button>
      {open && (
        <div className="flex flex-col gap-2 p-4">
          {children}
        </div>
      )}
    </section>
  );
}

// Style partagé des items du menu « ⋯ More » du header (actions secondaires).
const MENU_ITEM = 'w-full flex items-center gap-2 text-left px-2.5 py-1.5 rounded-md text-sm text-content hover:bg-surface-raised disabled:opacity-40';

export default function DatasetWorkspace({ ds, onBack }) {
  const navigate = useNavigate();
  const { caps } = useCapabilities();
  const d = ds.data;
  const [cropImg, setCropImg] = useState(null);
  const zipInput = useRef(null);   // hidden input for "Import dataset (ZIP)"
  const [refCrop, setRefCrop] = useState(false);
  const [viewImg, setViewImg] = useState(null);
  const [showImages, setShowImages] = useState(true);
  const [captionMode, setCaptionMode] = useState(null);   // null → défaut auto selon train_type
  const [showLeaks, setShowLeaks] = useState(false);       // liste dépliée des captions qui fuient
  const [checkpointCount, setCheckpointCount] = useState(0);
  // Repli des cartes d'étape : override manuel par section id (gf-*). Vide au
  // départ → chaque carte suit son défaut (repliée si terminée & non-courante).
  // Un clic sur le bandeau écrit ici ; un saut depuis la checklist force l'ouverture.
  const [openMap, setOpenMap] = useState({});
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Hooks must run unconditionally on every render — deriveSteps() null-guards `d`,
  // so this is safe to call before the loading early-return below.
  const { steps, nextStep } = useGuidedFlow(d, caps, checkpointCount);
  // Changer de dataset repart d'un repli « propre » (les overrides visent des ids
  // partagés entre datasets, sinon ils fuiteraient d'un dataset à l'autre).
  useEffect(() => { setOpenMap({}); }, [d?.id]);
  // Pendant une passe de captioning, épingle la section images ouverte : sinon,
  // dès que la DERNIÈRE caption tombe, stepDone['gf-images'] bascule à true et la
  // carte se replie toute seule (sectionOpen retombe sur !stepDone) — la grille
  // « Dataset images » disparaît au lieu de se remplir en place. On respecte un
  // override explicite déjà posé (l'utilisateur l'avait replié → on ne force pas).
  useEffect(() => {
    if (ds.captioning) setOpenMap((m) => ('gf-images' in m ? m : { ...m, 'gf-images': true }));
  }, [ds.captioning]);
  if (!d) return <p className="text-content-subtle text-sm">Loading…</p>;

  const images = d.images || [];
  // Dataset CONCEPT : on masque tout ce qui est identité/visage (référence, générateur
  // de variations, analyse faciale, badge de fuite, composition, flux guidé) — il ne
  // reste que import brut → curation → caption (inversée) → entraînement.
  // 'style' suit le même chemin UI que concept : pas de référence/visage/composition,
  // juste import brut → curation → caption (contenu pur, optionnelle) → entraînement.
  const concept = d.kind === 'concept' || d.kind === 'style';
  // Fidélité corps : captions bannissent aussi les marques corporelles, composition
  // cible plus de bustes/corps, import plein cadre par défaut.
  const bodyFid = d.fidelity === 'body';
  const kept = images.filter((i) => i.status === 'keep').length;
  const unused = images.filter((i) => i.status === 'reject' || i.status === 'failed').length;
  const keptUncaptioned = images.filter((i) => i.status === 'keep' && !i.caption).length;
  const keptCaptioned = kept - keptUncaptioned;
  // Style de caption : défaut AUTO (SDXL booru-native → booru tags ; sinon prose), surchargé par le sélecteur.
  const effCaptionMode = captionMode || (d.train_type === 'sdxl' ? 'booru' : 'prose');
  const pending = images.filter((i) => i.status === 'pending' && !i.filename).length;
  const triage = images.filter((i) => i.status === 'pending' && i.filename).length;   // generated, awaiting ✓/✕

  // ── État « terminé » + résumé par carte (bandeau ✓ vert + repli). Un id de
  //    section (gf-*) est replié par défaut quand il est fini et non-courant. ──
  const stepDone = {
    'gf-reference': concept ? images.length > 0 : !!d.ref_filename,
    'gf-generate': images.length > 0,
    'gf-images': kept > 0 && triage === 0 && keptUncaptioned === 0,
    'gf-training': checkpointCount > 0,
  };
  const stepSummary = {
    'gf-reference': concept ? `${images.length} image(s)` : 'Reference set',
    'gf-generate': `${images.length} image(s)`,
    'gf-images': kept > 0
      ? `${kept} kept${keptUncaptioned === 0 ? ' · all captioned' : ` · ${keptCaptioned}/${kept} captioned`}`
      : 'nothing kept yet',
    'gf-training': checkpointCount > 0 ? `${checkpointCount} checkpoint(s)` : 'not trained yet',
  };
  const isActive = (id) => nextStep?.targetId === id;
  // Ouverte si : c'est l'étape courante ; sinon l'override manuel ; sinon le
  // défaut (dépliée tant que non terminée, repliée une fois terminée).
  const sectionOpen = (id) => isActive(id) || (id in openMap ? openMap[id] : !stepDone[id]);
  const toggleSection = (id) => setOpenMap((m) => ({ ...m, [id]: !sectionOpen(id) }));
  // Toutes les props d'état d'une carte pour un id de section donné.
  const cardProps = (id) => ({
    id, active: isActive(id), done: stepDone[id], summary: stepSummary[id],
    open: sectionOpen(id), onToggle: () => toggleSection(id),
  });

  const jumpTo = (step) => {
    // Ré-ouvre une carte repliée avant d'y sauter (une étape terminée l'est).
    setOpenMap((m) => ({ ...m, [step.targetId]: true }));
    const el = document.getElementById(step.targetId);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    const b = el.querySelector('button:not([disabled])'); if (b) b.focus({ preventScroll: true });
    // Flash the landed-on section (gf-highlight, index.css) so the eye finds it.
    // remove + reflow restarts the animation when the same step is clicked twice.
    el.classList.remove('gf-highlight');
    void el.offsetWidth;
    el.classList.add('gf-highlight');
    window.setTimeout(() => el.classList.remove('gf-highlight'), 1500);
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

  return (
    <div className="flex flex-col gap-3">
      {/* ---- Header : identité du dataset + UNE action primaire (Export ZIP).
           Les actions secondaires (backup, import-fusion, fidélité, purge)
           vivent dans le menu discret « ⋯ More » — un clic pour les atteindre,
           hors du chemin d'un premier utilisateur. ---- */}
      <div className="flex items-center gap-2 flex-wrap">
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
          <button type="button" disabled={!kept}
            onClick={() => {
              // Guard-rails: untriaged images are silently EXCLUDED from the zip,
              // and uncaptioned kept ones export as trigger-only.
              const triage = images.filter((i) => i.status === 'pending' && i.filename).length;
              if (triage && !window.confirm(`${triage} image(s) still await triage (✓/✕) and will NOT be in the ZIP. Export anyway?`)) return;
              if (keptUncaptioned && !window.confirm(`${keptUncaptioned} kept image(s) without a caption (trigger only). Export anyway?`)) return;
              ds.exportZip();
            }}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            ⬇ Export ZIP ({kept})
          </button>
          {/* summary en display:flex → pas de marqueur natif ; les items restent
              montés en permanence (details ne fait que masquer l'affichage). */}
          <details className="relative">
            <summary
              title="More dataset actions — backup, merge another dataset in, fidelity, cleanup"
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content hover:bg-surface-raised text-sm cursor-pointer select-none">
              ⋯ More
            </summary>
            <div className="absolute right-0 top-full mt-1 z-20 w-72 rounded-lg border border-border bg-surface shadow-xl p-1.5 flex flex-col gap-0.5">
              <button type="button" onClick={ds.exportBackup}
                title="Full portable backup: all images with statuses, captions, scores and settings — restore it on any machine from the Datasets page."
                className={MENU_ITEM}>
                💾 Backup
                <span className="ml-auto text-content-subtle text-[0.625rem]">portable copy</span>
              </button>
              <button type="button" onClick={() => zipInput.current?.click()} disabled={ds.busy}
                title="Merge an existing training dataset into this one: a ZIP of images with kohya-style same-name .txt captions (any folder layout). Aspect kept, perceptual duplicates skipped."
                className={MENU_ITEM}>
                📦 Import dataset
                <span className="ml-auto text-content-subtle text-[0.625rem]">merge a ZIP in</span>
              </button>
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
              {unused > 0 && (
                <button type="button" disabled={ds.busy}
                  onClick={() => {
                    if (window.confirm(`Permanently delete the ${unused} rejected/failed image(s) (files included)?`)) ds.purgeUnused();
                  }}
                  title="Permanently delete rejected and failed images"
                  className="w-full flex items-center gap-2 text-left px-2.5 py-1.5 rounded-md text-sm text-red-300 hover:bg-red-500/10 disabled:opacity-40">
                  🧹 Purge rejected/failed ({unused})
                </button>
              )}
            </div>
          </details>
          <input ref={zipInput} type="file" accept=".zip,application/zip" className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) ds.importDatasetZip(f);
              e.target.value = '';
            }} />
        </div>
      </div>

      {/* Two-column workspace: a sticky vertical progress checklist on the left,
          the dataset content on the right. Concept datasets have no reference/
          generate flow, so they keep the single-column layout (the wrapper is
          display:contents -> no visual change, no sidebar). */}
      <div className={concept ? 'contents' : 'grid grid-cols-1 lg:grid-cols-[15rem_minmax(0,1fr)] gap-3 items-start'}>
        {!concept && (
          // Sticky checklist spans the viewport height and centers its content
          // vertically (flex-col + justify-center) so, after jumping to a section,
          // it stays mid-screen instead of stranded in the top-left corner. The
          // nav still stretches to the full 15rem column (default align stretch).
          <aside className="lg:sticky lg:top-0 lg:h-screen lg:flex lg:flex-col lg:justify-center">
            <GuidedChecklist steps={steps} currentId={nextStep ? nextStep.id : null} onJump={jumpTo} />
          </aside>
        )}
        <div className="flex flex-col gap-5 min-w-0">
          {!concept && (
            <NextStepCard step={nextStep} trainMode={!!caps.training_visible} busy={ds.busy}
              totalImages={images.length} onAction={nextAction} actionLabel={nextActionLabel} />
          )}

          {ds.busy && (
            <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2">
              <span className="inline-block w-4 h-4 border-2 border-amber-400/40 border-t-amber-400 rounded-full animate-spin" aria-hidden />
              <span className="text-content text-sm">
                {ds.captioning
                  ? `Captioning in progress — ${keptCaptioned}/${kept} captioned… ComfyUI is paused.`
                  : 'GPU processing in progress (analysis / cropping / captioning)… ComfyUI is paused during the pass.'}
              </span>
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

          {/* ============ Étape 1 (+2) : constituer le dataset. Concept : sources
               scrapées + import brut. Personnage : référence puis génération/import. */}
          {concept ? (
            // Concept : pas de photo de référence ni de générateur — on peuple le dataset
            // en scannant des galeries (ConceptSourcesPanel) et/ou par upload manuel.
            <StepSection n={1} {...cardProps('gf-reference')}
              title="Add images"
              help="a concept LoRA learns from real images — scrape galleries or drop photos">
              <ConceptSourcesPanel onImport={ds.scrapeImport} busy={ds.busy} />
              <ImportDropzone onImport={(f) => ds.importFiles(f)} busy={ds.busy} />
            </StepSection>
          ) : (
            <>
              <StepSection n={1} {...cardProps('gf-reference')}
                title="Reference photo"
                help="one clear photo of the face — every generated variation starts from it">
                <ReferencePanel refFilename={d.ref_filename} datasetId={d.id} onSetRef={ds.setRef}
                  onCropRef={() => setRefCrop(true)} busy={ds.busy} nonce={ds.refNonce}
                  extraRefs={d.ref_extra_filenames || []}
                  onAddExtraRef={ds.addExtraRef} onRemoveExtraRef={ds.removeExtraRef} />
              </StepSection>

              <StepSection n={2} {...cardProps('gf-generate')}
                title="Add images"
                help="generate AI variations of the reference — and mix in a few real photos if you have them">
                <CompositionBar composition={d.composition} bodyFidelity={bodyFid} />
                <VariationCatalog key={`vc-${d.id}-${bodyFid}`} busy={ds.busy}
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
              </StepSection>
            </>
          )}

          {/* ============ Curation + caption : trier ✓/✕ puis légender les gardées.
               La barre d'outils caption (mode, lancer, re-caption, analyse, fuite)
               vit ICI, à côté de la grille qu'elle concerne — plus dans le header. */}
          <StepSection n={concept ? 2 : 3} {...cardProps('gf-images')}
            title="Curate & caption"
            help="keep ✓ the good shots, reject ✕ the rest — then caption the kept ones (captions are what training reads)">

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
                className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                🔄 Re-caption
              </button>
              {!concept && (
                <button type="button" onClick={ds.analyzeFaces} disabled={ds.busy || !d.ref_filename}
                  title={d.ref_filename ? "Scores each image's facial resemblance vs the reference (deletes nothing)" : "Set a reference photo first"}
                  className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
                  {ds.analyzing ? '🎭 Analyzing…' : '🎭 Analyze faces'}
                </button>
              )}
              {!concept && d.caption_leak && d.caption_leak.captioned > 0 && (
                d.caption_leak.leaking === 0 ? (
                  <span className="ml-auto text-emerald-400 text-[0.8125rem]"
                    title="No caption describes hair/face/skin — identity binds to the trigger.">
                    ✅ 0 leak ({d.caption_leak.captioned})
                  </span>
                ) : (
                  <button type="button" onClick={() => setShowLeaks((v) => !v)}
                    aria-expanded={showLeaks}
                    title="These captions mention hair/face/skin → identity won't bind to the trigger. Click to list and fix them here."
                    className="ml-auto text-amber-400 text-[0.8125rem] underline decoration-amber-400/50 decoration-dashed">
                    ⚠️ {d.caption_leak.leaking}/{d.caption_leak.captioned} identity leak {showLeaks ? '▴' : '▾'}
                  </button>
                )
              )}
            </div>

            {/* Identity-leak triage list: every leaking caption editable IN PLACE
                (saves on blur, like the grid) — no more hunting through the tiles. */}
            {showLeaks && !concept && (
              <div className="rounded-lg border border-amber-400/40 bg-amber-500/5 p-3 flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-amber-300 text-sm font-semibold">⚠️ Captions leaking identity</span>
                  <span className="text-content-subtle text-[0.6875rem]">
                    they mention hair / face / skin{bodyFid ? ' / body marks' : ''} — the identity
                    won't bind to the trigger. Edit here (saves when you click away) or 🔄 Re-caption.
                  </span>
                  <button type="button" onClick={() => setShowLeaks(false)}
                    className="ml-auto text-content-subtle hover:text-content text-sm" aria-label="Close the leak list">✕</button>
                </div>
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
                  <p className="text-emerald-400 text-[0.8125rem]">✅ All clear — no leaking caption left.</p>
                )}
              </div>
            )}

            <button type="button" onClick={() => setShowImages((v) => !v)} aria-expanded={showImages}
              className="flex items-center gap-2 text-left text-content font-semibold text-sm">
              <span aria-hidden>🖼️</span> Dataset images
              <span className="text-content-subtle text-[0.6875rem] font-normal">{images.length}</span>
              <span aria-hidden className="ml-auto text-content-subtle">{showImages ? '▾' : '▸'}</span>
            </button>
            {showImages && (
              <CaptionToolsBar images={images} trainType={d.train_type} onReplace={ds.replaceCaptions} busy={ds.busy} />
            )}
            {showImages && (
              <DatasetGrid images={d.images} datasetId={d.id} onStatus={ds.setStatus} onCaption={ds.setCaption}
                onCrop={setCropImg} onDelete={ds.deleteImage}
                onRegenerate={(id, loraStrength, prompt) => ds.regenerate(id, loraStrength, prompt)} onView={setViewImg}
                onBatch={ds.batchImages} busy={ds.busy}
                nonces={ds.nonces} faceThresholds={d.face_thresholds} />
            )}
          </StepSection>

          {/* ============ Entraînement (et lanceur du Studio de test). */}
          <StepSection n={concept ? 3 : 4} {...cardProps('gf-training')}
            title="Train"
            help="turn the kept & captioned images into a LoRA — or export the ZIP (top right) to train elsewhere">
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
          </StepSection>
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
    </div>
  );
}
