// react-frontend/src/components/dataset/studio/StudioGenerationSettings.jsx
/**
 * StudioGenerationSettings shares GLOBAL run settings between multi-LoRA comparison and full
 * single-LoRA Studio, matching Generate without its prompt builder. The matrix remains
 * LoRA/strength; aspect/CFG/steps axes are handled elsewhere. Family-gated StudioSections offer:
 * FORMAT for all through ResolutionSelector/resolution_tier; SAMPLING for Krea with allowed
 * sampler/scheduler and empty Auto; DETAIL for SDXL with DetailDaemon 0-1; ENGINE for Krea with
 * precision, finishing and permanent_loras; NEGATIVE when supported by the server. This component owns
 * state, persists by storagePrefix, and emits normalized snake_case /run settings via onChange.
 * Omit empty fields to retain backend defaults; server also gates by family. Props: family,
 * storagePrefix, optional permanentLoras candidates [{filename,label|displayName,triggerWord}],
 * and stable onChange. Single-LoRA Studio supplies scoped candidates; comparison derives them from
 * /api/index_config krea_loras when absent.
 */
import { useEffect, useMemo, useState } from 'react';
import ResolutionSelector from '../../shared/ResolutionSelector';
import LockableSlider from '../../shared/LockableSlider';
import ZImageLoraConfig from '../../shared/ZImageLoraConfig';
import StudioSection from './StudioSection';
import { MemberOptions } from '../../patreon/MemberPreviews.jsx';
import { showsNegativePrompt } from '../../../utils/studioFamilySettings';
import {
  KREA_SAMPLER_PRESETS_FALLBACK, presetChoice, splitSamplerChoice,
} from '../../../utils/kreaSamplerChoice';
import {
  FINISH_REFERENCE, HIRES_SCALE_CHOICES, finishPayload, fmtScale,
  hiresDefaultLabel, hiresIsOn, hiresPayload, normaliseHiresDefaults,
} from '../../../utils/studioFinishKnobs';

// Fallback until /api/index_config loads must mirror backend KREA_ALLOWED_* lists; authoritative
// values come from config.krea_samplers/schedulers.
const KREA_SAMPLERS_FALLBACK = ['er_sde', 'euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_sde', 'res_multistep', 'deis', 'ddim', 'uni_pc'];
const KREA_SCHEDULERS_FALLBACK = ['simple', 'sgm_uniform', 'beta', 'normal', 'ddim_uniform', 'kl_optimal', 'linear_quadratic'];

// KREA_WEIGHT_DTYPE_HELPERS_START
const KREA_DEFAULT_WEIGHT_DTYPE = 'fp8_e4m3fn';
const KREA_WEIGHT_DTYPES = Object.freeze([
  'default',
  'fp8_e4m3fn',
  'fp8_e4m3fn_fast',
  'fp8_e5m2',
]);
const KREA_LEGACY_FP8_DTYPES = KREA_WEIGHT_DTYPES.filter((dtype) => dtype !== 'default');

// `wdt` predates the FP8-safe Krea default. Its old implicit `default` value must
// therefore migrate to FP8, while `wdt_v2=default` is an explicit user choice.
const resolveKreaWeightDtype = (versionedValue, legacyValue) => (
  KREA_WEIGHT_DTYPES.includes(versionedValue)
    ? versionedValue
    : (KREA_LEGACY_FP8_DTYPES.includes(legacyValue)
      ? legacyValue
      : KREA_DEFAULT_WEIGHT_DTYPE)
);
// KREA_WEIGHT_DTYPE_HELPERS_END
//
// Studio aspects mirror backend TEST_ASPECTS, with the ratio name ResolutionSelector needs to
// display ACTUAL generated dimensions.
const STUDIO_ASPECTS = [
  { key: '9:16', label: 'Tall', ratio: 'tall' },
  { key: '3:4', label: 'Portrait', ratio: 'portrait' },
  { key: '1:1', label: 'Square', ratio: 'square' },
  { key: '4:3', label: 'Landscape', ratio: 'landscape' },
  { key: '16:9', label: 'Wide', ratio: 'widescreen' },
];

const basename = (p) => String(p || '').split(/[\\/]/).pop();

export default function StudioGenerationSettings({ family = 'zimage', capabilities = null, cfgs = null, storagePrefix = 'studioGen', permanentLoras = null, aspectPicker = false, onChange }) {
  // `cfgs` = the CFG values the run will use: Krea's negative only acts above CFG 1.
  const hasNegative = showsNegativePrompt(capabilities, family, cfgs);
  const isSdxl = family === 'sdxl';
  const isKrea = family === 'krea';

  // Namespaced localStorage helpers provide lazy initialization and VALUE persistence.
  // LockableSlider persists only its lock, so values are handled here.
  const k = (name) => `${storagePrefix}_${name}`;
  const load = (name, fallback, parse = (v) => v) => {
    try { const v = localStorage.getItem(k(name)); return v === null ? fallback : parse(v); }
    catch { return fallback; }
  };
  const save = (name, value) => { try { localStorage.setItem(k(name), String(value)); } catch { /* private mode */ } };

  // State persisted under storagePrefix.
  const [resolutionTier, setResolutionTierS] = useState(() => load('tier', 'standard'));
  // Resolution multiplier 1.0-1.9 applies to the selected tier. Default 1.0 preserves tier
  // dimensions for compatibility. Clamp on read and write.
  const [resolutionMultiplier, setResolutionMultiplierS] = useState(
    () => load('resmult', 1.0, parseFloat));
  // Run aspect for comparison only; full Studio uses it as a matrix AXIS through AxisPickers.
  // Default 9:16 matches backend DEFAULT_ASPECT previously applied when omitted.
  const [aspect, setAspectS] = useState(() => load('aspect', '9:16'));
  const [negative, setNegativeS] = useState(() => load('negative', ''));
  const [detailAmount, setDetailAmountS] = useState(() => load('detail', 0.21, parseFloat));
  const [sampler, setSamplerS] = useState(() => load('sampler', ''));
  const [scheduler, setSchedulerS] = useState(() => load('scheduler', ''));
  const [weightDtype, setWeightDtypeS] = useState(() => resolveKreaWeightDtype(
    load('wdt_v2', null),
    load('wdt', null),
  ));
  // PER-RUN Krea hi-res second pass and finishing. Empty hiresScale defers to the Settings default
  // named in the menu; string 1 disables it for this run; other strings carry the factor from
  // select. Both finishing passes default to 0/off because Studio cells have no Settings default
  // for them. utils/studioFinishKnobs.js defines the three wire forms.
  const [hiresScale, setHiresScaleS] = useState(() => load('hiresScale', ''));
  // null means never edited here: display the Settings default and OMIT hires_denoise so its
  // rewrite setting applies. Hardcoded 0.5 previously overrode krea_hires.denoise=0.7 while the
  // menu still claimed the Settings default was 1.5x with rewrite 0.7.
  const [hiresDenoise, setHiresDenoiseS] = useState(() => load('hiresDenoise', null, parseFloat));
  const [finishSharpen, setFinishSharpenS] = useState(() => load('finSharpen', 0, parseFloat));
  const [finishGrain, setFinishGrainS] = useState(() => load('finGrain', 0, parseFloat));
  const [permStack, setPermStack] = useState([]);   // Reported by ZImageLoraConfig.
  // Setters persist values as they update, matching RunSetupPanel/SettingsPanel.
  const setResolutionTier = (v) => { setResolutionTierS(v); save('tier', v); };
  const setResolutionMultiplier = (v) => {
    const m = Math.max(1.0, Math.min(1.9, Number(v) || 1.0));
    setResolutionMultiplierS(m); save('resmult', m);
  };
  const setAspect = (v) => { setAspectS(v); save('aspect', v); };
  const setNegative = (v) => { setNegativeS(v); save('negative', v); };
  const setDetailAmount = (v) => { setDetailAmountS(v); save('detail', v); };
  const setSampler = (v) => { setSamplerS(v); save('sampler', v); };
  const setScheduler = (v) => { setSchedulerS(v); save('scheduler', v); };
  const setWeightDtype = (v) => { setWeightDtypeS(v); save('wdt_v2', v); };
  const setHiresScale = (v) => { setHiresScaleS(v); save('hiresScale', v); };
  const setHiresDenoise = (v) => { setHiresDenoiseS(v); save('hiresDenoise', v); };
  const setFinishSharpen = (v) => { setFinishSharpenS(v); save('finSharpen', v); };
  const setFinishGrain = (v) => { setFinishGrainS(v); save('finGrain', v); };

  // Krea configuration: sampler/scheduler and always-on LoRA candidates.
  // Fetch only for Krea; other families need nothing from /config.
  const [config, setConfig] = useState(null);
  useEffect(() => {
    if (!isKrea) return undefined;
    let cancelled = false;
    fetch('/api/index_config', { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled && d) setConfig(d); })
      .catch(() => { /* Use the fallback allowlist below. */ });
    return () => { cancelled = true; };
  }, [isKrea]);

  const kreaSamplers = config?.krea_samplers?.length ? config.krea_samplers : KREA_SAMPLERS_FALLBACK;
  const kreaSchedulers = config?.krea_schedulers?.length ? config.krea_schedulers : KREA_SCHEDULERS_FALLBACK;
  // Custom sampler presets come from /api/index_config like the lists above. Fallback handles only
  // older servers without this field, offering exactly the presets those servers know.
  const kreaSamplerPresets = config?.krea_sampler_presets?.length
    ? config.krea_sampler_presets : KREA_SAMPLER_PRESETS_FALLBACK;
  // Numeric Settings hi-res defaults let the menu state its actual values. Memoize: this object is
  // a dependency of the settings-reporting effect, and a fresh object would trigger parent
  // setState, rerender and another fresh object indefinitely. Fall back to off because older
  // servers never add this pass.
  const hiresDefaults = useMemo(
    () => normaliseHiresDefaults(config?.krea_hires_defaults), [config]);

  // Always-on LoRA candidates come from the full Studio's family-scoped payload, otherwise
  // config.krea_loras. Exclude lora_* trained characters because they are a test AXIS, mirroring
  // backend permanent_lora_candidates.
  const permCandidates = useMemo(() => {
    if (!isKrea) return [];
    if (permanentLoras != null) {
      return permanentLoras.map((l) => ({
        filename: l.filename, displayName: l.displayName || l.label || basename(l.filename), triggerWord: l.triggerWord,
      }));
    }
    return (config?.krea_loras || [])
      .filter((l) => !basename(l.filename).toLowerCase().startsWith('lora_'))
      .map((l) => ({ filename: l.filename, displayName: l.displayName || basename(l.filename), triggerWord: l.triggerWord }));
  }, [isKrea, permanentLoras, config]);

  // Emit normalized snake_case /run settings, OMITTING empty values to preserve backend defaults.
  // onChange should be stable, such as parent setState, to avoid loops; include it in dependencies
  // defensively.
  useEffect(() => {
    const s = { resolution_tier: resolutionTier, resolution_multiplier: resolutionMultiplier };
    // Comparison's global aspect becomes a one-value matrix axis. NEVER emit it in full Studio
    // with aspectPicker=false, where AxisPickers owns the test axis; overriding it here would
    // break the matrix.
    if (aspectPicker && aspect) s.aspects = [aspect];
    if (hasNegative) {
      const neg = negative.trim();
      if (neg) s.negative = neg;
    }
    if (isSdxl) {
      s.detail_amount = detailAmount;
    }
    if (isKrea) {
      // ONE choice maps to TWO fields. Putting a preset name into sampler would give ComfyUI an
      // unknown sampler and reject the whole graph. See utils/kreaSamplerChoice.js.
      const samplerChoice = splitSamplerChoice(sampler);
      if (samplerChoice.sampler) s.sampler = samplerChoice.sampler;
      if (samplerChoice.sampler_preset) s.sampler_preset = samplerChoice.sampler_preset;
      if (scheduler) s.scheduler = scheduler;
      s.weight_dtype = weightDtype;
      // Hi-res fix uses three wire forms: deferred, explicitly off or a value. Finishing omits off
      // keys. Keep these rules in utils/studioFinishKnobs.js so node --test can exercise them.
      Object.assign(s, hiresPayload({ scale: hiresScale, denoise: hiresDenoise }, hiresDefaults));
      Object.assign(s, finishPayload({ sharpen: finishSharpen, grain: finishGrain }));
      // Split always-on stack: batch-checked LoRAs become server-managed with/without test AXES;
      // others apply to EVERY cell as before.
      const alwaysOn = permStack.filter((e) => !e.batch)
        .map(({ filename, strength }) => ({ filename, strength }));
      const batched = permStack.filter((e) => e.batch)
        .map(({ filename, strength }) => ({ filename, strength }));
      if (alwaysOn.length) s.permanent_loras = alwaysOn;
      if (batched.length) s.batch_loras = batched;
    }
    onChange?.(s);
  }, [hasNegative, isSdxl, isKrea, resolutionTier, resolutionMultiplier, aspectPicker, aspect, negative, detailAmount, sampler, scheduler,
      weightDtype, permStack, onChange,
      hiresScale, hiresDenoise, hiresDefaults, finishSharpen, finishGrain]);

  return (
    <div className="flex flex-col gap-2">
      {/*
       * FORMAT for all families: run SIZE, plus comparison's RATIO. Full Studio keeps ratio as an
       * AxisPickers test axis, so no picker here.
       */}
      <StudioSection title="Format" storageKey={k('sec_format')} anchorId="st-format">
        {aspectPicker && (
          <>
            <span className="text-content-muted text-[0.625rem] uppercase">Aspect ratio</span>
            <div className="grid grid-cols-5 gap-1.5">
              {STUDIO_ASPECTS.map((a) => (
                <button key={a.key} type="button" onClick={() => setAspect(a.key)}
                  aria-pressed={aspect === a.key}
                  className={`flex flex-col items-center gap-0.5 py-1.5 px-1 rounded-[10px] border transition-all duration-150 ${aspect === a.key
                    ? 'border-primary/70 bg-primary/15 text-white'
                    : 'border-white/10 bg-white/[0.04] text-content-muted'}`}>
                  <span className="text-[0.6875rem] font-semibold">{a.key}</span>
                  <span className="text-[0.5625rem] opacity-60">{a.label}</span>
                </button>
              ))}
            </div>
          </>
        )}
        <span className="text-content-muted text-[0.625rem] uppercase">Resolution</span>
        <ResolutionSelector value={resolutionTier} onChange={setResolutionTier}
          aspectRatio={aspectPicker
            ? (STUDIO_ASPECTS.find((a) => a.key === aspect)?.ratio || 'square')
            : 'square'}
          maxLongSide={family === 'sdxl' ? 1024 : undefined}
          maxPixels={capabilities?.max_pixels}
          multiplier={resolutionMultiplier} onMultiplierChange={setResolutionMultiplier} />
        <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-0.5">
          {aspectPicker
            ? 'Output size — the ratio above sets the proportions. Standard ≈ 1 MP.'
            : 'Output size (the aspect axis sets the proportions). Standard ≈ 1 MP.'}
        </span>
      </StudioSection>

      {/* SAMPLING (krea) — sampler/scheduler (whitelist backend, '' = Auto). */}
      {isKrea && (
        <StudioSection title="Sampling" storageKey={k('sec_sampling')} anchorId="st-sampling">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide">
              Sampler
              <select
                value={sampler}
                onChange={(e) => setSampler(e.target.value)}
                aria-label="Krea sampler"
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="">Auto (er_sde)</option>
                <optgroup label="ComfyUI samplers">
                  {kreaSamplers.map((s) => (<option key={s} value={s}>{s}</option>))}
                </optgroup>
                <optgroup label="Preset sampler (tuned for 8 steps)">
                  {kreaSamplerPresets.map((p) => (
                    <option key={p} value={presetChoice(p)}>{p}</option>
                  ))}
                </optgroup>
              </select>
            </label>
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide">
              Scheduler
              <select
                value={scheduler}
                onChange={(e) => setScheduler(e.target.value)}
                aria-label="Krea scheduler"
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="">Auto (simple)</option>
                {kreaSchedulers.map((s) => (<option key={s} value={s}>{s}</option>))}
              </select>
            </label>
          </div>

          {/*
           * HI-RES FIX: Krea second pass uses core LatentUpscaleBy and KSampler nodes. Empty
           * follows the Settings default named in the option; string 1 disables the pass even if
           * Settings says 1.5; otherwise use the run factor.
           */}
          <div className="mt-2 pt-2 border-t border-white/10 flex flex-col gap-2">
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide">
              Hi-res fix (second pass)
              <select
                value={hiresScale}
                onChange={(e) => setHiresScale(e.target.value)}
                aria-label="Krea hi-res fix (second sampling pass)"
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="">{hiresDefaultLabel(hiresDefaults)}</option>
                <option value="1">Off for this run</option>
                {HIRES_SCALE_CHOICES.map((v) => (
                  <option key={v} value={String(v)}>{fmtScale(v)} the latent</option>
                ))}
              </select>
            </label>
            <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
              Samples small, upscales the latent, re-samples at the larger size — the model
              draws the detail instead of interpolating it. 1.5× ≈ 2.25× the pixels and time.
            </span>
            {hiresIsOn(hiresScale, hiresDefaults) && (
              <>
                <LockableSlider
                  label="How much the second pass may rewrite"
                  value={hiresDenoise ?? hiresDefaults.denoise}
                  min="0.05" max="1" step="0.05"
                  storageKey={k('hires_denoise_lock')}
                  onChange={(e) => setHiresDenoise(parseFloat(e.target.value))}
                />
                <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
                  0.5 = keeps the composition, rewrites the texture · near 1 = a different
                  picture at the larger size
                </span>
              </>
            )}
          </div>
        </StudioSection>
      )}

      {/* SDXL DETAIL controls DetailDaemon intensity, separate from the second-pass steps axis. */}
      {isSdxl && (
        <StudioSection title="Detail" storageKey={k('sec_detail')} anchorId="st-detail">
          <LockableSlider
            label="Detail (Daemon intensity)"
            value={detailAmount}
            min="0" max="1" step="0.01"
            storageKey={k('detail_lock')}
            onChange={(e) => setDetailAmount(parseFloat(e.target.value))}
          />
          <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
            0.21 = SDXL default · ≤0.25 safe · ↑ more detail (HDR/grain risk)
          </span>
        </StudioSection>
      )}

      {/* ENGINE (krea) — precision + finition + LoRA always-on. */}
      {isKrea && (
        <StudioSection title="Engine" storageKey={k('sec_engine')} anchorId="st-engine">
          {/* Loader precision, finishing and always-on LoRAs. */}
          <div className="flex flex-col gap-2.5">
            {/* Loader precision: node 20 weight_dtype. */}
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide mt-1">
              Precision
              <select
                value={weightDtype}
                onChange={(e) => setWeightDtype(e.target.value)}
                aria-label="Krea loader precision (weight dtype)"
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="default">ComfyUI default (auto · dtype varies)</option>
                <option value="fp8_e4m3fn">FP8 e4m3fn (recommended)</option>
                <option value="fp8_e4m3fn_fast">Fast+ (fp8 fast)</option>
                <option value="fp8_e5m2">fp8 e5m2 (wide range)</option>
              </select>
            </label>
            <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
              FP8 e4m3fn is the Krea-safe default. “ComfyUI default” delegates the dtype to the checkpoint and may use much more VRAM; try it only as a compatibility fallback.
            </span>
          </div>

          {/*
           * FINISHING applies app-side to rendered cells through utils/photo_finish: sharpening
           * and grain. No ColorMatch because Studio txt2img has no before image to match. Zero
           * disables the pass, omits its payload key and stores NULL.
           */}
          <div className="mt-2 pt-2 border-t border-white/10 flex flex-col gap-2.5">
            <span className="text-[0.6875rem] text-content-muted uppercase tracking-wide">
              Finishing (after render)
            </span>
            <LockableSlider
              label="Sharpen"
              value={finishSharpen}
              min="0" max="1.5" step="0.05"
              storageKey={k('fin_sharpen_lock')}
              format={(v) => (Number(v) > 0 ? v : 'off')}
              onChange={(e) => setFinishSharpen(parseFloat(e.target.value))}
            />
            <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
              0 = off · {FINISH_REFERENCE.sharpen} = reference · local contrast at 1 px, the
              octave diffusion leaves empty — past ~1 the halo reads as an outline
            </span>
            <LockableSlider
              label="Film grain"
              value={finishGrain}
              min="0" max="0.05" step="0.002"
              storageKey={k('fin_grain_lock')}
              format={(v) => (Number(v) > 0 ? v : 'off')}
              onChange={(e) => setFinishGrain(parseFloat(e.target.value))}
            />
            <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
              0 = off · {FINISH_REFERENCE.grain} = reference (±2.5 levels: texture, never noise) ·
              what stops a render looking plastic
            </span>
          </div>

          {/*
           * Always-on style/utility LoRAs apply to EVERY cell rather than a test axis.
           * ZImageLoraConfig persists itself and reports the enabled stack.
           */}
          {permCandidates.length > 0 && (
            <div className="mt-2 pt-2 border-t border-white/10">
              <ZImageLoraConfig
                loras={permCandidates}
                onChange={setPermStack}
                storageKey={k('perm')}
                label="Always-on LoRAs (every cell · ⚖ batch = tested as an axis)"
                emptyHint="No always-on LoRA for this pipeline."
                krea
                batchToggle
              />
            </div>
          )}
        </StudioSection>
      )}

      {/* Only offer a negative prompt when the selected pipeline can apply it. */}
      {hasNegative && (
        <StudioSection title="Negative" storageKey={k('sec_negative')} defaultOpen={false} anchorId="st-negative">
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-[0.625rem] uppercase">Negative prompt (optional)</span>
            <textarea
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              rows={3}
              placeholder="Leave empty for the pipeline default…"
              aria-label="Negative prompt"
              className="rounded-lg border border-border bg-app/60 px-2.5 py-1.5 text-content text-sm resize-y min-h-[4rem]"
            />
            {capabilities?.negative_prompt_needs_cfg && (
              <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70">
                Applies to cells above CFG 1. At CFG 1 the sampler ignores it.
              </span>
            )}
          </label>
        </StudioSection>
      )}

      {/* Members previews of this family's options (free installations only). */}
      <MemberOptions variant="section" storageKey={k('sec_members')}
        placements={['studio.options', `studio.options.${family}`]} />
    </div>
  );
}
