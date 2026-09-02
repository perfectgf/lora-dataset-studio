// react-frontend/src/components/dataset/studio/StudioGenerationSettings.jsx
/**
 * StudioGenerationSettings — bloc PARTAGÉ de réglages de génération GLOBAUX du run,
 * inséré dans les DEUX asides du Studio (comparaison ≥2 LoRA ET studio riche 1 LoRA).
 * Parité avec la page Generate, SANS le prompt builder. Tout est un réglage GLOBAL
 * par run (la matrice de test reste LoRA × strength ; les axes aspect/cfg/steps
 * restent gérés ailleurs — on ne les duplique pas ici).
 *
 * Sections (via <StudioSection>), conditionnées à la famille (`family`) :
 *   • FORMAT   (toutes)      → <ResolutionSelector> → resolution_tier (fast|standard|hq|max)
 *   • SAMPLING (krea)        → sampler + scheduler (whitelist backend, '' = Auto)
 *   • DETAIL   (sdxl)        → detail_amount (DetailDaemon, 0–1)
 *   • ENGINE   (krea)        → precision, finition,
 *                              + pile LoRA « always-on » (permanent_loras)
 *   • NEGATIVE (zimage)      → negative (textarea)
 *
 * Composant AUTONOME : garde son propre état, le persiste en localStorage (namespacé
 * par `storagePrefix`), et remonte vers le parent un objet `settings` NORMALISÉ
 * (snake_case = contrat des routes /run) via `onChange`. Les champs vides sont OMIS
 * (le backend garde alors ses défauts) ; chaque champ est gaté PAR FAMILLE côté serveur.
 *
 * Props :
 *   family          'zimage'|'sdxl'|'krea'
 *   storagePrefix   préfixe des clés localStorage (namespace par contexte/famille)
 *   permanentLoras  (optionnel) candidats always-on [{filename,label|displayName,triggerWord}]
 *                   — fourni par le studio riche (payload d.permanent_loras). Absent en
 *                   comparaison → on dérive la liste depuis /api/index_config (krea_loras).
 *   onChange        (settings) => void  (idéalement un setState stable du parent)
 */
import { useEffect, useMemo, useState } from 'react';
import ResolutionSelector from '../../shared/ResolutionSelector';
import LockableSlider from '../../shared/LockableSlider';
import ZImageLoraConfig from '../../shared/ZImageLoraConfig';
import StudioSection from './StudioSection';
import {
  KREA_SAMPLER_PRESETS_FALLBACK, presetChoice, splitSamplerChoice,
} from '../../../utils/kreaSamplerChoice';
import {
  FINISH_REFERENCE, HIRES_SCALE_CHOICES, finishPayload, fmtScale,
  hiresDefaultLabel, hiresIsOn, hiresPayload, normaliseHiresDefaults,
} from '../../../utils/studioFinishKnobs';

// Repli si /api/index_config n'est pas encore chargé (doit refléter la whitelist
// backend KREA_ALLOWED_* — la liste réelle vient de config.krea_samplers/schedulers).
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

// Formats du Studio (whitelist backend TEST_ASPECTS) + le nom de ratio attendu
// par <ResolutionSelector> pour afficher les VRAIES dimensions générées.
const STUDIO_ASPECTS = [
  { key: '9:16', label: 'Tall', ratio: 'tall' },
  { key: '3:4', label: 'Portrait', ratio: 'portrait' },
  { key: '1:1', label: 'Square', ratio: 'square' },
  { key: '4:3', label: 'Landscape', ratio: 'landscape' },
  { key: '16:9', label: 'Wide', ratio: 'widescreen' },
];

const basename = (p) => String(p || '').split(/[\\/]/).pop();

export default function StudioGenerationSettings({ family = 'zimage', storagePrefix = 'studioGen', permanentLoras = null, aspectPicker = false, onChange }) {
  const isZ = family === 'zimage';
  const isSdxl = family === 'sdxl';
  const isKrea = family === 'krea';

  // Helpers localStorage namespacés (init paresseuse + persistance des VALEURS ;
  // LockableSlider ne persiste que son verrou, pas la valeur → on s'en charge).
  const k = (name) => `${storagePrefix}_${name}`;
  const load = (name, fallback, parse = (v) => v) => {
    try { const v = localStorage.getItem(k(name)); return v === null ? fallback : parse(v); }
    catch { return fallback; }
  };
  const save = (name, value) => { try { localStorage.setItem(k(name), String(value)); } catch { /* private mode */ } };

  // --- État (persisté, namespacé par storagePrefix) ---------------------------
  const [resolutionTier, setResolutionTierS] = useState(() => load('tier', 'standard'));
  // Multiplicateur de résolution (1.0–1.9) appliqué au palier choisi. Défaut 1.0 =
  // taille du palier inchangée (rétrocompatible). Clampé au chargement + à l'écriture.
  const [resolutionMultiplier, setResolutionMultiplierS] = useState(
    () => load('resmult', 1.0, parseFloat));
  // Format du run (mode comparaison uniquement — dans le studio riche, le ratio
  // est un AXE de la matrice via AxisPickers). Défaut = 9:16, le DEFAULT_ASPECT
  // que le backend appliquait déjà en silence quand rien n'était envoyé.
  const [aspect, setAspectS] = useState(() => load('aspect', '9:16'));
  const [negative, setNegativeS] = useState(() => load('negative', ''));
  const [detailAmount, setDetailAmountS] = useState(() => load('detail', 0.21, parseFloat));
  const [sampler, setSamplerS] = useState(() => load('sampler', ''));
  const [scheduler, setSchedulerS] = useState(() => load('scheduler', ''));
  const [weightDtype, setWeightDtypeS] = useState(() => resolveKreaWeightDtype(
    load('wdt_v2', null),
    load('wdt', null),
  ));
  // Hi-res fix (2e passe Krea) et finition, PAR RUN. `hiresScale` '' = laisser
  // le défaut Settings (nommé dans le menu) ; '1' = off pour ce run ; sinon le
  // facteur, en chaîne comme le <select> le rend. Les deux passes de finition
  // partent à 0 = off : une cellule Studio n'a pas de défaut Settings à suivre.
  // Les trois formes de fil sont décidées dans utils/studioFinishKnobs.js.
  const [hiresScale, setHiresScaleS] = useState(() => load('hiresScale', ''));
  // null = "never touched here" : le curseur AFFICHE le défaut Settings et le
  // payload n'envoie PAS hires_denoise, donc le rewrite du réglage s'applique.
  // Un 0.5 en dur ici écrasait silencieusement un krea_hires.denoise à 0.7 tout
  // en affichant « Settings default (1.5×, rewrite 0.7) » dans le menu.
  const [hiresDenoise, setHiresDenoiseS] = useState(() => load('hiresDenoise', null, parseFloat));
  const [finishSharpen, setFinishSharpenS] = useState(() => load('finSharpen', 0, parseFloat));
  const [finishGrain, setFinishGrainS] = useState(() => load('finGrain', 0, parseFloat));
  const [permStack, setPermStack] = useState([]);   // remonté par ZImageLoraConfig

  // Setters qui persistent en même temps (miroir du pattern RunSetupPanel/SettingsPanel).
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

  // --- Config Krea (sampler/scheduler + candidats LoRA always-on) --------------
  // Fetch uniquement en Krea (les autres familles n'ont besoin de rien de /config).
  const [config, setConfig] = useState(null);
  useEffect(() => {
    if (!isKrea) return undefined;
    let cancelled = false;
    fetch('/api/index_config', { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (!cancelled && d) setConfig(d); })
      .catch(() => { /* fallback whitelist en dur ci-dessous */ });
    return () => { cancelled = true; };
  }, [isKrea]);

  const kreaSamplers = config?.krea_samplers?.length ? config.krea_samplers : KREA_SAMPLERS_FALLBACK;
  const kreaSchedulers = config?.krea_schedulers?.length ? config.krea_schedulers : KREA_SCHEDULERS_FALLBACK;
  // Presets du sampler maison. Servis par /api/index_config comme les deux listes
  // ci-dessus ; le repli couvre le seul cas où le serveur est trop ancien pour les
  // connaître — et lui offrir ceux qu'il connaît est alors exactement juste.
  const kreaSamplerPresets = config?.krea_sampler_presets?.length
    ? config.krea_sampler_presets : KREA_SAMPLER_PRESETS_FALLBACK;
  // Le défaut Settings du hi-res fix, en nombres, pour que l'option « défaut »
  // du menu DISE ce qu'elle vaut. Mémoïsé : cet objet est une dépendance de
  // l'effet qui remonte `settings`, et un objet neuf à chaque rendu relancerait
  // l'effet → setState du parent → rendu → objet neuf : la boucle. Repli = off
  // (un serveur trop ancien pour l'envoyer n'ajoute jamais la passe).
  const hiresDefaults = useMemo(
    () => normaliseHiresDefaults(config?.krea_hires_defaults), [config]);

  // Candidats LoRA « always-on » : liste fournie par le studio riche (family-scopée,
  // payload) sinon dérivée de config.krea_loras (on écarte les `lora_*` = perso entraînés,
  // qui sont un AXE de test, pas un always-on — miroir de permanent_lora_candidates backend).
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

  // --- Remontée du `settings` normalisé (snake_case = contrat /run) ------------
  // On OMET les vides (le backend garde ses défauts). `onChange` doit être stable
  // (setState du parent) — sinon boucle ; deps incluent onChange par prudence.
  useEffect(() => {
    const s = { resolution_tier: resolutionTier, resolution_multiplier: resolutionMultiplier };
    // Format global du run (comparaison) : axe à 1 seule valeur côté matrice.
    // JAMAIS émis en studio riche (aspectPicker=false) — là, le ratio est un axe
    // de test choisi via AxisPickers et l'écraser ici casserait la matrice.
    if (aspectPicker && aspect) s.aspects = [aspect];
    if (isZ) {
      const neg = negative.trim();
      if (neg) s.negative = neg;
    }
    if (isSdxl) {
      s.detail_amount = detailAmount;
    }
    if (isKrea) {
      // UN choix, DEUX champs : un preset écrit dans `sampler` serait un nom de
      // sampler inconnu de ComfyUI et le graphe entier serait refusé. Voir
      // utils/kreaSamplerChoice.js.
      const samplerChoice = splitSamplerChoice(sampler);
      if (samplerChoice.sampler) s.sampler = samplerChoice.sampler;
      if (samplerChoice.sampler_preset) s.sampler_preset = samplerChoice.sampler_preset;
      if (scheduler) s.scheduler = scheduler;
      s.weight_dtype = weightDtype;
      // Hi-res fix + finition : trois formes de fil pour le premier (différé /
      // off explicite / valeur), clés OMISES quand off pour la seconde — décidé
      // dans utils/studioFinishKnobs.js, pas ici, pour que node --test le voie.
      Object.assign(s, hiresPayload({ scale: hiresScale, denoise: hiresDenoise }, hiresDefaults));
      Object.assign(s, finishPayload({ sharpen: finishSharpen, grain: finishGrain }));
      // Pile always-on scindée : ☑ batch → AXE de test (cellules avec/sans, géré
      // serveur), sinon appliqué à CHAQUE cellule comme avant.
      const alwaysOn = permStack.filter((e) => !e.batch)
        .map(({ filename, strength }) => ({ filename, strength }));
      const batched = permStack.filter((e) => e.batch)
        .map(({ filename, strength }) => ({ filename, strength }));
      if (alwaysOn.length) s.permanent_loras = alwaysOn;
      if (batched.length) s.batch_loras = batched;
    }
    onChange?.(s);
  }, [isZ, isSdxl, isKrea, resolutionTier, resolutionMultiplier, aspectPicker, aspect, negative, detailAmount, sampler, scheduler,
      weightDtype, permStack, onChange,
      hiresScale, hiresDenoise, hiresDefaults, finishSharpen, finishGrain]);

  return (
    <div className="flex flex-col gap-2">
      {/* FORMAT (toutes familles) — SIZE du run + (comparaison) le RATIO. Dans le
          studio riche le ratio reste un axe de test (AxisPickers) → pas de picker ici. */}
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

          {/* HI-RES FIX — 2e passe Krea (LatentUpscaleBy + KSampler, nœuds du cœur).
              '' = le défaut Settings, NOMMÉ dans l'option ; '1' = off pour ce run,
              qui bat un défaut Settings à 1.5 ; sinon le facteur du run. */}
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

      {/* DETAIL (sdxl) — intensité DetailDaemon (distincte du steps pass 2 = axe). */}
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
          {/* Précision du loader, finition, LoRA always-on. */}
          <div className="flex flex-col gap-2.5">
            {/* Précision du loader (node 20 weight_dtype). */}
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

          {/* FINITION — côté app, sur la cellule rendue (utils/photo_finish) :
              sharpen + grain. Pas de ColorMatch ici : une cellule Studio est du
              txt2img, il n'y a pas d'image « d'avant » à laquelle se recaler.
              0 = off, clé omise du payload (NULL sur la ligne). */}
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

          {/* LoRA « always-on » (style/utilitaire) : appliqués à CHAQUE cellule (pas un
              axe de test). ZImageLoraConfig se persiste seul et remonte la pile activée. */}
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

      {/* NEGATIVE (zimage) — prompt négatif global du run. */}
      {isZ && (
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
          </label>
        </StudioSection>
      )}
    </div>
  );
}
