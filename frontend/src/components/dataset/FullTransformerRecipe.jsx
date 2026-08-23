// react-frontend/src/components/dataset/FullTransformerRecipe.jsx
// The dense (full-transformer) advanced recipe card and its base picker,
// moved VERBATIM from TrainingPanel.jsx (2026-08-24, panel decomposition
// slice 1). TrainingPanel re-exports both components, so every existing
// import path keeps working.
import { useEffect, useState } from 'react';
import Fp8QuantizeTool from './Fp8QuantizeTool';
import { UseDatasetCaptionsButton } from './UseDatasetCaptionsButton';
import { baseOptionSuffix } from './trainingFamilyScope.js';

// « Custom weights… » : valeur-sentinelle de l'entrée du sélecteur de base qui
// révèle le champ chemin. Les familles qui l'exposent + celles honorant VAE/TE
// (miroir de CUSTOM_WEIGHTS_FAMILIES / VAE_TE_OVERRIDE_FAMILIES côté serveur ;
// base-info les renvoie, ces défauts ne servent qu'avant son chargement).
export const CUSTOM_BASE_SENTINEL = '__custom_weights__';
export const DEFAULT_CUSTOM_FAMILIES = ['sdxl', 'krea', 'flux', 'flux2klein'];

// FULL_TRANSFORMER_ADVANCED_RECIPE_START
/** The dense Krea recipe stays server-owned, but not all of it is a constraint.
 *
 * The values that changed the OUTPUT rather than whether the run fits in 80 GB
 * are editable here: preview prompts (the generic defaults showed nothing about
 * the actual dataset), learning rate, resolution, the checkpoint-every / keep
 * pair — which is also what the Hugging Face storage forecast multiplies, so it
 * states the delivery size right next to the control — and the three quality
 * levers (images per step, learning-rate schedule, noise schedule).
 *
 * "Images per step" is gradient accumulation in the user's words. It is the only
 * lever here whose cost is money rather than memory, so the multiplier it
 * implies is printed next to it and turns amber once it is above 1: a rented
 * 80 GB GPU is billed by the hour, and nobody should learn that from an invoice.
 *
 * Everything else is locked and SAYS SO: optimizer, batch size, dtype and
 * gradient checkpointing are the geometry that makes a 12B transformer trainable
 * on one 80 GB card. Changing any of them turns a working run into an
 * out-of-memory crash an hour in, on a rented GPU.
 */
const DENSE_BOUNDS_FALLBACK = {
  lr: 1e-6, lrMin: 1e-7, lrMax: 5e-6,
  resolution: 1024, resolutionChoices: [768, 1024],
  saveEvery: 250, saveEveryMin: 100, saveEveryMax: 5000,
  keeps: 1, keepsMax: 3,
  gradAccum: 1, gradAccumChoices: [1, 2, 4, 8],
  lrSchedule: 'constant',
  lrScheduleChoices: ['constant', 'constant_with_warmup', 'cosine'],
  warmup: 100, warmupMin: 10, warmupMax: 1000,
  timestepType: 'linear', timestepTypeChoices: ['linear', 'sigmoid', 'weighted'],
};

// User-facing wording for ai-toolkit's own value names. The STORED value stays
// ai-toolkit's (no alias to maintain, cf. CLAUDE.md rule 7) — only the label is
// translated, and each one says what it does rather than what it is called.
const DENSE_LR_SCHEDULE_LABELS = {
  constant: 'Constant (default)',
  constant_with_warmup: 'Warm up, then constant',
  cosine: 'Cosine decay to zero',
};
const DENSE_TIMESTEP_LABELS = {
  linear: 'Linear (default)',
  sigmoid: 'Sigmoid — favours mid noise levels',
  weighted: 'Weighted — same draw, bell-curve loss weighting',
};

const fmtGB = (bytes) => (
  typeof bytes === 'number' && bytes > 0 ? `${(bytes / 1e9).toFixed(1)} GB` : null
);

// Exported for the render contract test. Nothing else imports it: a settings
// card whose JSX is never executed by a test is a card that can ship with a
// crash in it, and source-text assertions do not execute anything.
export function FullTransformerAdvancedRecipe({
  stepsOverride, setStepsOverride, disabled = false,
  adv = null, saveAdv = null,
  samplePromptsText = '', setSamplePromptsText = null, saveSamplePrompts = null,
  samplePromptsDefault = [], maxSamplePrompts = 8,
  // The dataset's own images (kept ones carry the captions the 🎲 button draws
  // from) and the one callback that writes AND persists the textarea.
  datasetImages = [], applySamplePrompts = null,
  quantizeTarget = null, suggestedQuantizePath = '',
  // The base the emitted config will actually carry. Computed, never a
  // literal: this card used to state "Official Krea 2 Raw" over a recipe that
  // can now be Turbo or a local checkpoint.
  baseSummary = 'official Krea 2 Raw',
}) {
  const explicitSteps = String(stepsOverride || '').trim();
  const factClass = 'rounded-lg border border-sky-400/20 bg-app/45 px-2.5 py-2';
  const b = DENSE_BOUNDS_FALLBACK;
  const lr = adv?.dense_lr ?? b.lr;
  const lrMin = adv?.dense_lr_min ?? b.lrMin;
  const lrMax = adv?.dense_lr_max ?? b.lrMax;
  const resolution = adv?.dense_resolution ?? b.resolution;
  const resolutionChoices = adv?.dense_resolution_choices ?? b.resolutionChoices;
  const saveEvery = adv?.dense_save_every ?? b.saveEvery;
  const saveEveryMin = adv?.dense_save_every_min ?? b.saveEveryMin;
  const saveEveryMax = adv?.dense_save_every_max ?? b.saveEveryMax;
  const keeps = adv?.dense_max_step_saves ?? b.keeps;
  const keepsMax = adv?.dense_max_step_saves_max ?? b.keepsMax;
  const plan = adv?.dense_storage_plan || null;
  const hint = adv?.dense_inference_hint || null;
  const fp8 = adv?.dense_fp8_export !== false;
  const keepMaster = adv?.dense_keep_bf16 !== false;
  const gradAccum = adv?.dense_grad_accum ?? b.gradAccum;
  const gradAccumChoices = adv?.dense_grad_accum_choices ?? b.gradAccumChoices;
  const timeMultiplier = adv?.dense_time_multiplier ?? gradAccum;
  const lrSchedule = adv?.dense_lr_schedule ?? b.lrSchedule;
  const lrScheduleChoices = adv?.dense_lr_schedule_choices ?? b.lrScheduleChoices;
  const warmup = adv?.dense_warmup ?? b.warmup;
  const warmupMin = adv?.dense_warmup_min ?? b.warmupMin;
  const warmupMax = adv?.dense_warmup_max ?? b.warmupMax;
  const warmupApplies = adv?.dense_warmup_applies ?? (lrSchedule === 'constant_with_warmup');
  const timestepType = adv?.dense_timestep_type ?? b.timestepType;
  const timestepTypeChoices = adv?.dense_timestep_type_choices ?? b.timestepTypeChoices;
  const [lrDraft, setLrDraft] = useState(String(lr));
  const [saveDraft, setSaveDraft] = useState(String(saveEvery));
  const [warmupDraft, setWarmupDraft] = useState(String(warmup));
  useEffect(() => { setLrDraft(String(lr)); }, [lr]);
  useEffect(() => { setSaveDraft(String(saveEvery)); }, [saveEvery]);
  useEffect(() => { setWarmupDraft(String(warmup)); }, [warmup]);
  const patch = (values) => { if (saveAdv) saveAdv(values); };
  const commitLr = () => {
    const value = Number(lrDraft);
    if (!Number.isFinite(value) || value < lrMin || value > lrMax) {
      setLrDraft(String(lr));
      return;
    }
    if (value !== lr) patch({ dense_lr: value });
  };
  const commitSaveEvery = () => {
    const value = Number(saveDraft);
    if (!Number.isInteger(value) || value < saveEveryMin || value > saveEveryMax) {
      setSaveDraft(String(saveEvery));
      return;
    }
    if (value !== saveEvery) patch({ dense_save_every: value });
  };
  const commitWarmup = () => {
    const value = Number(warmupDraft);
    if (!Number.isInteger(value) || value < warmupMin || value > warmupMax) {
      setWarmupDraft(String(warmup));
      return;
    }
    if (value !== warmup) patch({ dense_warmup: value });
  };
  // min-w-0 on both the control and its label: a <select> sizes itself on its
  // WIDEST option ("Sigmoid — favour…" is 303 px), and a flex item defaults to
  // min-width:auto, so without this the row cannot shrink and spills off a
  // 400 px screen instead of wrapping.
  const controlClass = 'min-w-0 rounded border border-sky-300/40 bg-app/70 px-2 py-1 text-content tabular-nums disabled:opacity-50';

  return (
    <section aria-label="Krea 2 full-model recipe"
      className="rounded-xl border border-sky-400/35 bg-sky-500/[0.07] p-3 text-[0.75rem]">
      <div className="flex flex-col gap-1">
        <span className="font-semibold text-sky-100">Recipe sent to AI Toolkit</span>
        <span className="text-sky-200/85 leading-relaxed">
          The values below the line are yours to change. The ones above are locked because they
          are what makes a 12B transformer fit on one 80 GB card — LoRA/LoKr presets and settings
          are not shown or applied here.
        </span>
      </div>

      <dl className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className={factClass}>
          <dt className="text-content-subtle text-[0.625rem] uppercase">Model</dt>
          <dd className="m-0 mt-0.5 text-content">
            {baseSummary} · full transformer · unquantized
          </dd>
        </div>
        <div className={factClass}>
          <dt className="text-content-subtle text-[0.625rem] uppercase">Locked · batch &amp; precision</dt>
          <dd className="m-0 mt-0.5 text-content">Batch 1 · bf16 — the 80 GB budget has no room for more</dd>
        </div>
        <div className={factClass}>
          <dt className="text-content-subtle text-[0.625rem] uppercase">Locked · optimizer</dt>
          <dd className="m-0 mt-0.5 text-content">Adafactor — Adam-family states would not fit in memory</dd>
        </div>
        <div className={factClass}>
          <dt className="text-content-subtle text-[0.625rem] uppercase">Locked · memory</dt>
          <dd className="m-0 mt-0.5 text-content">Gradient checkpointing · cached latents + text embeddings</dd>
        </div>
        <div className={factClass}>
          <dt className="text-content-subtle text-[0.625rem] uppercase">Cloud requirements</dt>
          <dd className="m-0 mt-0.5 text-content">80 GB VRAM GPU · at least 200 GB disk</dd>
        </div>
        <div className={factClass}>
          <dt className="text-content-subtle text-[0.625rem] uppercase">Delivery</dt>
          <dd className="m-0 mt-0.5 text-content">
            {fp8
              ? (keepMaster
                ? 'Private Hugging Face repo · bf16 master + fp8 export for ComfyUI'
                : 'Private Hugging Face repo · fp8 export only (no re-training later)')
              : 'Private Hugging Face repo · bf16 master only'}
          </dd>
        </div>
      </dl>

      {hint?.note && (
        <p className="m-0 mt-3 rounded-lg border border-amber-300/35 bg-amber-400/10 px-2.5 py-2 text-amber-100 leading-relaxed">
          ⚠ {hint.note}
        </p>
      )}

      <div className="mt-3 border-t border-sky-300/25 pt-3 flex flex-col gap-2">
        <span className="font-semibold text-sky-100">Editable</span>

        <label className="flex flex-wrap items-center gap-2 rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <span className="font-semibold">Steps</span>
          <input type="number" min={500} step={100} value={stepsOverride}
            onChange={(event) => setStepsOverride(event.target.value)}
            disabled={disabled}
            placeholder="adaptive"
            aria-label="Full-model training steps (leave empty for an adaptive target)"
            className={`w-[6rem] ${controlClass}`} />
          <span className="text-sky-100/80">
            {explicitSteps ? `${explicitSteps} target steps` : 'empty = server-calculated adaptive target'}
          </span>
        </label>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Learning rate</span>
            <input type="number" step="1e-7" min={lrMin} max={lrMax} value={lrDraft}
              onChange={(event) => setLrDraft(event.target.value)}
              onBlur={commitLr} disabled={disabled}
              aria-label="Full-model learning rate"
              className={`w-[7rem] ${controlClass}`} />
          </label>
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Resolution</span>
            <select value={String(resolution)} disabled={disabled}
              onChange={(event) => patch({ dense_resolution: Number(event.target.value) })}
              aria-label="Full-model training resolution"
              className={controlClass}>
              {resolutionChoices.map((value) => (
                <option key={value} value={String(value)}>{value} px</option>
              ))}
            </select>
          </label>
          <span className="basis-full text-sky-200/70 text-[0.6875rem]">
            {lrMin.toExponential(0)}–{lrMax.toExponential(0)} · default 1e-6. 768 px trains faster
            and cheaper than the 1024 px default, at lower fidelity.
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Images per step</span>
            <select value={String(gradAccum)} disabled={disabled}
              onChange={(event) => patch({ dense_grad_accum: Number(event.target.value) })}
              aria-label="How many images each optimizer step learns from"
              className={controlClass}>
              {gradAccumChoices.map((value) => (
                <option key={value} value={String(value)}>{value}</option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Noise schedule</span>
            <select value={timestepType} disabled={disabled}
              onChange={(event) => patch({ dense_timestep_type: event.target.value })}
              aria-label="Full-model timestep distribution"
              className={controlClass}>
              {timestepTypeChoices.map((value) => (
                <option key={value} value={value}>
                  {DENSE_TIMESTEP_LABELS[value] || value}
                </option>
              ))}
            </select>
          </label>
          {/* The bill, next to the control that sets it. Gradient accumulation
              is the one lever here that buys quality with money rather than
              memory, and a rented GPU is billed by the hour. */}
          <span className={`basis-full text-[0.6875rem] ${
            timeMultiplier > 1 ? 'text-amber-100/90' : 'text-sky-200/70'}`}>
            {timeMultiplier > 1
              ? `Each step learns from ${gradAccum} images instead of 1 — steadier training on a `
                + `big dataset, but the run takes about ${timeMultiplier}× as long, so the rented `
                + `GPU costs about ${timeMultiplier}× as much. Same checkpoints, same storage.`
              : 'One image per step is the default. Raising it averages several images into each '
                + 'update — steadier on a large dataset, but it multiplies the run time and the '
                + 'pod bill by the same number.'}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Learning-rate schedule</span>
            <select value={lrSchedule} disabled={disabled}
              onChange={(event) => patch({ dense_lr_schedule: event.target.value })}
              aria-label="Full-model learning-rate schedule"
              className={controlClass}>
              {lrScheduleChoices.map((value) => (
                <option key={value} value={value}>
                  {DENSE_LR_SCHEDULE_LABELS[value] || value}
                </option>
              ))}
            </select>
          </label>
          {/* Warmup steps only reach the trainer on the one schedule that
              accepts them; the server gates it the same way. */}
          {warmupApplies && (
            <label className="flex min-w-0 items-center gap-2">
              <span className="font-semibold">Warm up over</span>
              <input type="number" min={warmupMin} max={warmupMax} step={10} value={warmupDraft}
                onChange={(event) => setWarmupDraft(event.target.value)}
                onBlur={commitWarmup} disabled={disabled}
                aria-label="Full-model warmup length in steps"
                className={`w-[5.5rem] ${controlClass}`} />
              <span className="text-sky-100/80">steps</span>
            </label>
          )}
          <span className="basis-full text-sky-200/70 text-[0.6875rem]">
            Constant is what shipped. Warming up eases the first steps instead of hitting a 12B
            model at full rate from step 1; cosine decay fades the rate to zero by the last step,
            which settles detail late in the run. Neither changes what the run delivers or costs.
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Checkpoint every</span>
            <input type="number" min={saveEveryMin} max={saveEveryMax} step={50} value={saveDraft}
              onChange={(event) => setSaveDraft(event.target.value)}
              onBlur={commitSaveEvery} disabled={disabled}
              aria-label="Full-model checkpoint interval in steps"
              className={`w-[6rem] ${controlClass}`} />
            <span className="text-sky-100/80">steps</span>
          </label>
          <label className="flex min-w-0 items-center gap-2">
            <span className="font-semibold">Keep</span>
            <select value={String(keeps)} disabled={disabled}
              onChange={(event) => patch({ dense_max_step_saves: Number(event.target.value) })}
              aria-label="How many full-model checkpoints to keep"
              className={controlClass}>
              {Array.from({ length: keepsMax }, (_, i) => i + 1).map((value) => (
                <option key={value} value={String(value)}>{value}</option>
              ))}
            </select>
          </label>
          <span className="basis-full text-amber-100/90 text-[0.6875rem]">
            {plan
              ? `Each checkpoint is about ${fmtGB(plan.checkpoint_bytes) || '26 GB'}${
                plan.fp8_typical_bytes ? `, plus a ~${fmtGB(plan.fp8_typical_bytes)} fp8 export` : ''
              } — this run will need about ${fmtGB(plan.peak_bytes) || '26 GB'} of PRIVATE Hugging Face storage.`
              : 'Each checkpoint is about 26 GB of private Hugging Face storage.'}
          </span>
        </div>

        <div className="rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <label className="flex flex-col gap-1">
            <span className="font-semibold">Preview prompts</span>
            <textarea rows={4} value={samplePromptsText} disabled={disabled}
              onChange={(event) => setSamplePromptsText?.(event.target.value)}
              onBlur={() => saveSamplePrompts?.()}
              placeholder={samplePromptsDefault.join('\n')}
              aria-label="Full-model preview prompts, one per line"
              className="w-full min-w-0 rounded border border-sky-300/40 bg-app/70 px-2 py-1 text-content text-[0.75rem] font-mono disabled:opacity-50" />
          </label>
          <UseDatasetCaptionsButton images={datasetImages} max={maxSamplePrompts}
            disabled={disabled} onPick={applySamplePrompts} className="mt-1" />
          <p className="m-0 mt-1 text-sky-200/70 text-[0.6875rem]">
            One per line, up to {maxSamplePrompts}. Empty = the generic defaults, which show nothing
            about this dataset — these images are the only way to judge the run while it costs money.
            Use <code>{'{trigger}'}</code> where the subject belongs.
          </p>
        </div>

        <label className="flex flex-wrap items-center gap-2 rounded-lg border border-sky-300/30 bg-sky-400/10 px-3 py-2 text-sky-50">
          <input type="checkbox" checked={keepMaster} disabled={disabled || !fp8}
            onChange={(event) => patch({ dense_keep_bf16: event.target.checked })}
            className="accent-sky-400" />
          <span className="font-semibold">Keep the bf16 master next to the fp8 export</span>
          <span className="basis-full text-sky-200/70 text-[0.6875rem]">
            fp8 is a one-way, inference-only export: without the master this model can never be
            continued, re-trained or merged. Turning this off halves the storage and closes that door.
          </span>
        </label>

        {/* THE surface for the conversion. `target` is what turns it from "paste
            a path" into one click: the model this dataset's run delivered, named
            by the same `hf_weight_filename` the artifact card above lists — so
            the card and the operation can never designate different checkpoints
            out of a repository that holds several 26 GB files. `suggestedPath`
            pre-fills the manual field with the custom base already on screen,
            when there is one. */}
        <Fp8QuantizeTool disabled={disabled} target={quantizeTarget}
          suggestedPath={suggestedQuantizePath} />
      </div>
    </section>
  );
}
// FULL_TRANSFORMER_ADVANCED_RECIPE_END

/** The base a FULL-MODEL run fine-tunes: the Raw/Turbo switch, the Krea 2
 * checkpoints installed on this machine, and a local file.
 *
 * It exists because the family/variant/base controls live in the LoRA-only
 * branch of the Advanced section, so a dense recipe had no visible way to
 * choose anything — the owner's report was literally "I still can't see where
 * to put the turbo option". The values are the same state the LoRA lane
 * writes (one stored column each), so nothing new is persisted and no alias is
 * owed.
 *
 * It is a top-level component rather than inline JSX for one reason: inline
 * JSX inside the panel is unreachable for a test. The panel only enters
 * full-model mode from an effect, and effects do not run under
 * renderToStaticMarkup, so this markup could never be EXECUTED by the suite —
 * exactly the shape that shipped two white screens before (see
 * tests/support/mountJsx.mjs). As its own component it is mounted for real, in
 * each of its three states. */
export function DenseBasePicker({
  variant, setVariant, base, setBase, customBase, setCustomBase,
  currentBases = [], customSupported = false, baseNote = null,
  baseSummary = 'official Krea 2 Raw', busy = false,
}) {
  // A picked checkpoint IS the base: the backend resolver returns it whatever
  // the variant says, so a live Raw/Turbo switch would offer a choice with no
  // effect — the exact class of lie this lane is being corrected for.
  const customPicked = !!String(base || '').trim();
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-sky-400/25 bg-app/40 px-3 py-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-content-muted text-[0.625rem] uppercase">
          Base to fine-tune
        </span>
        <select value={variant} onChange={(e) => setVariant(e.target.value)}
          disabled={busy || customPicked}
          aria-label="Krea 2 base for full-model training"
          title="Raw is Krea's official recommendation. Turbo is allowed and untested for full-model training — the notice above says exactly what is unknown."
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-50">
          <option value="base">Raw (recommended)</option>
          <option value="turbo">Turbo (few-step)</option>
        </select>
        <select value={customBase ? CUSTOM_BASE_SENTINEL : base}
          onChange={(e) => {
            const v = e.target.value;
            if (v === CUSTOM_BASE_SENTINEL) { setCustomBase(true); setBase(''); }
            else { setCustomBase(false); setBase(v); }
          }}
          disabled={busy}
          aria-label="Full-model base checkpoint"
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] max-w-[230px]">
          <option value="">Official Krea 2 — by variant</option>
          {currentBases.filter((b) => b.value).map((b) => (
            <option key={b.value} value={b.value}>
              {b.label}{baseOptionSuffix(b)}
            </option>
          ))}
          {customSupported && (
            <option value={CUSTOM_BASE_SENTINEL}>Custom weights… (local file)</option>
          )}
        </select>
      </div>
      {customBase && customSupported && (
        <input type="text" value={base} onChange={(e) => setBase(e.target.value)}
          disabled={busy}
          spellCheck={false}
          placeholder={'C:\\path\\to\\your-krea2-model.safetensors'}
          aria-label="Full-model custom weights path"
          className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] font-mono w-full max-w-[520px]" />
      )}
      {baseNote && (
        <span className={`text-[0.625rem] leading-relaxed ${
          baseNote.level === 'error' ? 'text-red-300' : 'text-amber-300'}`}>
          {baseNote.level === 'error' ? '⛔' : '⚠️'} {baseNote.text}
        </span>
      )}
      <span className="text-content-subtle text-[0.625rem] leading-relaxed">
        This run will train <b className="text-content-muted font-medium">{baseSummary}</b>.
        {customPicked
          ? ' A local checkpoint IS the base, so the Raw/Turbo switch does not apply to it. It travels to the rented GPU through a private repository on your own Hugging Face account.'
          : ' Raw is the non-distilled checkpoint Krea recommends fine-tuning.'}
        {' '}A ComfyUI-scaled fp8 export cannot be loaded for training and is refused with its reason —
        pick the bf16/fp16 build of the same model.
      </span>
    </div>
  );
}
