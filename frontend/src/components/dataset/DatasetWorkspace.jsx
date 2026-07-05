import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CompositionBar from './CompositionBar';
import ReferencePanel from './ReferencePanel';
import VariationCatalog from './VariationCatalog';
import TrainingPanel from './TrainingPanel';
import { fmt } from '../../utils/studioFormat';
import ImportDropzone from './ImportDropzone';
import DatasetGrid from './DatasetGrid';
import CropModal from './CropModal';
import DatasetLightbox from './DatasetLightbox';
import { useCapabilities } from '../../context/CapabilitiesContext';
import GuidedStepper from './GuidedStepper';
import NextStepCard from './NextStepCard';
import useGuidedFlow from '../../hooks/useGuidedFlow';

export default function DatasetWorkspace({ ds, onBack }) {
  const navigate = useNavigate();
  const { caps } = useCapabilities();
  const d = ds.data;
  const [cropImg, setCropImg] = useState(null);
  const [refCrop, setRefCrop] = useState(false);
  const [viewImg, setViewImg] = useState(null);
  const [showImages, setShowImages] = useState(true);
  const [captionMode, setCaptionMode] = useState(null);   // null → défaut auto selon train_type
  const [checkpointCount, setCheckpointCount] = useState(0);
  // Hooks must run unconditionally on every render — deriveSteps() null-guards `d`,
  // so this is safe to call before the loading early-return below.
  const { steps, nextStep } = useGuidedFlow(d, caps, checkpointCount);
  if (!d) return <p className="text-content-subtle text-sm">Loading…</p>;

  const images = d.images || [];
  const kept = images.filter((i) => i.status === 'keep').length;
  const unused = images.filter((i) => i.status === 'reject' || i.status === 'failed').length;
  const keptUncaptioned = images.filter((i) => i.status === 'keep' && !i.caption).length;
  const keptCaptioned = kept - keptUncaptioned;
  // Style de caption : défaut AUTO (SDXL booru-native → booru tags ; sinon prose), surchargé par le sélecteur.
  const effCaptionMode = captionMode || (d.train_type === 'sdxl' ? 'booru' : 'prose');
  const pending = images.filter((i) => i.status === 'pending' && !i.filename).length;
  const jumpTo = (step) => {
    const el = document.getElementById(step.targetId);
    if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      const b = el.querySelector('button:not([disabled])'); if (b) b.focus({ preventScroll: true }); }
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
        <div className="ml-auto flex gap-2">
          {unused > 0 && (
            <button type="button" disabled={ds.busy}
              onClick={() => {
                if (window.confirm(`Permanently delete the ${unused} rejected/failed image(s) (files included)?`)) ds.purgeUnused();
              }}
              title="Permanently delete rejected and failed images"
              className="px-3 py-1.5 rounded-lg bg-red-500/15 border border-red-500/40 text-red-300 text-sm disabled:opacity-40">
              🧹 Purge ({unused})
            </button>
          )}
          <select value={effCaptionMode} onChange={(e) => setCaptionMode(e.target.value)} disabled={ds.busy}
            title="Caption style — Prose (Z-Image) or Booru tags (SDXL booru-native, e.g. bigLove). Defaults to auto based on the dataset's type."
            className="px-2 py-1.5 rounded-lg bg-surface border border-border text-content text-[0.8125rem] disabled:opacity-40">
            <option value="prose">📝 Prose</option>
            <option value="booru">🏷️ Booru tags</option>
          </select>
          <button type="button" onClick={() => ds.caption(effCaptionMode)} disabled={ds.busy}
            className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
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
          <button type="button" onClick={ds.analyzeFaces} disabled={ds.busy || !d.ref_filename}
            title={d.ref_filename ? "Scores each image's facial resemblance vs the reference (deletes nothing)" : "Set a reference photo first"}
            className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40">
            {ds.analyzing ? '🎭 Analyzing…' : '🎭 Analyze faces'}
          </button>
          {d.caption_leak && d.caption_leak.captioned > 0 && (
            d.caption_leak.leaking === 0 ? (
              <span className="self-center text-emerald-400 text-[0.8125rem]"
                title="No caption describes hair/face/skin — identity binds to the trigger.">
                ✅ 0 leak ({d.caption_leak.captioned})
              </span>
            ) : (
              <span className="self-center text-amber-400 text-[0.8125rem]"
                title="These captions mention hair/face/skin → identity won't bind to the trigger. Re-caption or edit them.">
                ⚠️ {d.caption_leak.leaking}/{d.caption_leak.captioned} identity leak
              </span>
            )
          )}
          <button type="button" disabled={!kept}
            onClick={() => {
              if (keptUncaptioned && !window.confirm(`${keptUncaptioned} kept image(s) without a caption (trigger only). Export anyway?`)) return;
              ds.exportZip();
            }}
            className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            ⬇ Export ZIP ({kept})
          </button>
        </div>
      </div>

      <GuidedStepper steps={steps} currentId={nextStep ? nextStep.id : null} onJump={jumpTo} />
      <NextStepCard step={nextStep} trainMode={!!caps.training_visible} busy={ds.busy}
        totalImages={images.length} onAction={nextAction} actionLabel={nextActionLabel} />

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
        <div className="flex items-center gap-2 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-3 py-2">
          <span className="animate-pulse" aria-hidden>⏳</span>
          <span className="text-content text-sm">{pending} Klein generation(s) in progress…</span>
          <button type="button" onClick={ds.cancelPending} disabled={ds.busy}
            className="ml-auto px-2.5 py-1 rounded-lg bg-red-600/80 text-white text-xs font-semibold disabled:opacity-40">
            Cancel all
          </button>
        </div>
      )}

      <CompositionBar composition={d.composition} />

      <div id="gf-reference" className="grid grid-cols-1 lg:grid-cols-2 gap-3 scroll-mt-4">
        <ReferencePanel refFilename={d.ref_filename} datasetId={d.id} onSetRef={ds.setRef}
          onCropRef={() => setRefCrop(true)} busy={ds.busy} nonce={ds.refNonce}
          extraRefs={d.ref_extra_filenames || []}
          onAddExtraRef={ds.addExtraRef} onRemoveExtraRef={ds.removeExtraRef} />
        {/* No auto-classify: the backend tags framing='face' at import time (I2). */}
        <ImportDropzone onImport={(f) => ds.importFiles(f)} busy={ds.busy} />
      </div>

      <div id="gf-generate" className="scroll-mt-4">
        <VariationCatalog onGenerate={ds.generate} busy={ds.busy} hasRef={!!d.ref_filename}
          composition={d.composition} />
      </div>

      <div id="gf-training" className="scroll-mt-4">
        <TrainingPanel ds={ds} keptCount={kept} onCheckpointsChange={setCheckpointCount} />
      </div>

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

      <div id="gf-images" className="flex flex-col gap-2 scroll-mt-4">
        <button type="button" onClick={() => setShowImages((v) => !v)} aria-expanded={showImages}
          className="flex items-center gap-2 text-left text-content font-semibold text-sm">
          <span aria-hidden>🖼️</span> Dataset images
          <span className="text-content-subtle text-[0.6875rem] font-normal">{images.length}</span>
          <span aria-hidden className="ml-auto text-content-subtle">{showImages ? '▾' : '▸'}</span>
        </button>
        {showImages && (
          <DatasetGrid images={d.images} datasetId={d.id} onStatus={ds.setStatus} onCaption={ds.setCaption}
            onCrop={setCropImg} onDelete={ds.deleteImage}
            onRegenerate={(id) => ds.regenerate(id)} onView={setViewImg}
            nonces={ds.nonces} faceThresholds={d.face_thresholds} />
        )}
      </div>

      {cropImg && cropImg.filename && (
        <CropModal imageUrl={`/api/dataset/${d.id}/img/${encodeURIComponent(cropImg.filename)}`}
          onCancel={() => setCropImg(null)}
          onConfirm={async (box) => { await ds.crop(cropImg.id, box); setCropImg(null); }} />
      )}
      {refCrop && d.ref_filename && (
        <CropModal imageUrl={`/api/dataset/${d.id}/img/${encodeURIComponent(d.ref_filename)}`}
          onCancel={() => setRefCrop(false)}
          onConfirm={async (box) => { await ds.cropRef(box); setRefCrop(false); }} />
      )}
      {viewImgLive && (
        <DatasetLightbox img={viewImgLive} datasetId={d.id}
          nonce={(ds.nonces && ds.nonces[viewImgLive.id]) || 0}
          onClose={() => setViewImg(null)} />
      )}
    </div>
  );
}
