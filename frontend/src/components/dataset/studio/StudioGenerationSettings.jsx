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
 *   • ENGINE   (krea)        → rebalance(+strength), enhancer(+strength), precision,
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
import { useI18n } from '../../../i18n/I18nContext';

// Repli si /api/index_config n'est pas encore chargé (doit refléter la whitelist
// backend KREA_ALLOWED_* — la liste réelle vient de config.krea_samplers/schedulers).
const KREA_SAMPLERS_FALLBACK = ['er_sde', 'euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_sde', 'res_multistep', 'deis', 'ddim', 'uni_pc'];
const KREA_SCHEDULERS_FALLBACK = ['simple', 'sgm_uniform', 'beta', 'normal', 'ddim_uniform', 'kl_optimal', 'linear_quadratic'];

// Formats du Studio (whitelist backend TEST_ASPECTS) + le nom de ratio attendu
// par <ResolutionSelector> pour afficher les VRAIES dimensions générées.
const STUDIO_ASPECTS = [
  { key: '9:16', labelKey: 'tall', ratio: 'tall' },
  { key: '3:4', labelKey: 'portrait', ratio: 'portrait' },
  { key: '1:1', labelKey: 'square', ratio: 'square' },
  { key: '4:3', labelKey: 'landscape', ratio: 'landscape' },
  { key: '16:9', labelKey: 'wide', ratio: 'widescreen' },
];

const basename = (p) => String(p || '').split(/[\\/]/).pop();

export default function StudioGenerationSettings({ family = 'zimage', storagePrefix = 'studioGen', permanentLoras = null, aspectPicker = false, onChange }) {
  const { t } = useI18n();
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
  // Format du run (mode comparaison uniquement — dans le studio riche, le ratio
  // est un AXE de la matrice via AxisPickers). Défaut = 9:16, le DEFAULT_ASPECT
  // que le backend appliquait déjà en silence quand rien n'était envoyé.
  const [aspect, setAspectS] = useState(() => load('aspect', '9:16'));
  const [negative, setNegativeS] = useState(() => load('negative', ''));
  const [detailAmount, setDetailAmountS] = useState(() => load('detail', 0.21, parseFloat));
  const [sampler, setSamplerS] = useState(() => load('sampler', ''));
  const [scheduler, setSchedulerS] = useState(() => load('scheduler', ''));
  const [weightDtype, setWeightDtypeS] = useState(() => load('wdt', 'default'));
  const [rebalanceOn, setRebalanceOnS] = useState(() => load('rebalance', true, (v) => v === 'true'));
  const [rebalanceStrength, setRebalanceStrengthS] = useState(() => load('rebalanceStr', 4.0, parseFloat));
  const [enhancerOn, setEnhancerOnS] = useState(() => load('enhancer', false, (v) => v === 'true'));
  const [enhancerStrength, setEnhancerStrengthS] = useState(() => load('enhancerStr', 1.0, parseFloat));
  const [permStack, setPermStack] = useState([]);   // remonté par ZImageLoraConfig

  // Setters qui persistent en même temps (miroir du pattern RunSetupPanel/SettingsPanel).
  const setResolutionTier = (v) => { setResolutionTierS(v); save('tier', v); };
  const setAspect = (v) => { setAspectS(v); save('aspect', v); };
  const setNegative = (v) => { setNegativeS(v); save('negative', v); };
  const setDetailAmount = (v) => { setDetailAmountS(v); save('detail', v); };
  const setSampler = (v) => { setSamplerS(v); save('sampler', v); };
  const setScheduler = (v) => { setSchedulerS(v); save('scheduler', v); };
  const setWeightDtype = (v) => { setWeightDtypeS(v); save('wdt', v); };
  const setRebalanceOn = (v) => { setRebalanceOnS(v); save('rebalance', v); };
  const setRebalanceStrength = (v) => { setRebalanceStrengthS(v); save('rebalanceStr', v); };
  const setEnhancerOn = (v) => { setEnhancerOnS(v); save('enhancer', v); };
  const setEnhancerStrength = (v) => { setEnhancerStrengthS(v); save('enhancerStr', v); };

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
    const s = { resolution_tier: resolutionTier };
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
      if (sampler) s.sampler = sampler;
      if (scheduler) s.scheduler = scheduler;
      s.weight_dtype = weightDtype;
      s.rebalance = rebalanceOn;
      s.rebalance_strength = rebalanceStrength;
      s.enhancer = enhancerOn;
      s.enhancer_strength = enhancerStrength;
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
  }, [isZ, isSdxl, isKrea, resolutionTier, aspectPicker, aspect, negative, detailAmount, sampler, scheduler,
      weightDtype, rebalanceOn, rebalanceStrength, enhancerOn, enhancerStrength, permStack, onChange]);

  return (
    <div className="flex flex-col gap-2">
      {/* FORMAT (toutes familles) — SIZE du run + (comparaison) le RATIO. Dans le
          studio riche le ratio reste un axe de test (AxisPickers) → pas de picker ici. */}
      <StudioSection title={t('studio.generation.format')} storageKey={k('sec_format')} anchorId="st-format">
        {aspectPicker && (
          <>
            <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.generation.aspectRatio')}</span>
            <div className="grid grid-cols-5 gap-1.5">
              {STUDIO_ASPECTS.map((a) => (
                <button key={a.key} type="button" onClick={() => setAspect(a.key)}
                  aria-pressed={aspect === a.key}
                  className={`flex flex-col items-center gap-0.5 py-1.5 px-1 rounded-[10px] border transition-all duration-150 ${aspect === a.key
                    ? 'border-primary/70 bg-primary/15 text-white'
                    : 'border-white/10 bg-white/[0.04] text-content-muted'}`}>
                  <span className="text-[0.6875rem] font-semibold">{a.key}</span>
                  <span className="text-[0.5625rem] opacity-60">{t(`studio.generation.aspect.${a.labelKey}`)}</span>
                </button>
              ))}
            </div>
          </>
        )}
        <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.generation.resolution')}</span>
        <ResolutionSelector value={resolutionTier} onChange={setResolutionTier}
          aspectRatio={aspectPicker
            ? (STUDIO_ASPECTS.find((a) => a.key === aspect)?.ratio || 'square')
            : 'square'}
          maxLongSide={family === 'sdxl' ? 1024 : undefined} />
        <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-0.5">
          {aspectPicker
            ? t('studio.generation.outputSizeWithRatio')
            : t('studio.generation.outputSizeWithAxis')}
        </span>
      </StudioSection>

      {/* SAMPLING (krea) — sampler/scheduler (whitelist backend, '' = Auto). */}
      {isKrea && (
        <StudioSection title={t('studio.generation.sampling')} storageKey={k('sec_sampling')} anchorId="st-sampling">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide">
              {t('studio.generation.sampler')}
              <select
                value={sampler}
                onChange={(e) => setSampler(e.target.value)}
                aria-label={t('studio.generation.kreaSampler')}
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="">{t('studio.generation.auto')} (er_sde)</option>
                {kreaSamplers.map((s) => (<option key={s} value={s}>{s}</option>))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide">
              {t('studio.generation.scheduler')}
              <select
                value={scheduler}
                onChange={(e) => setScheduler(e.target.value)}
                aria-label={t('studio.generation.kreaScheduler')}
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="">{t('studio.generation.auto')} (simple)</option>
                {kreaSchedulers.map((s) => (<option key={s} value={s}>{s}</option>))}
              </select>
            </label>
          </div>
        </StudioSection>
      )}

      {/* DETAIL (sdxl) — intensité DetailDaemon (distincte du steps pass 2 = axe). */}
      {isSdxl && (
        <StudioSection title={t('studio.generation.detail')} storageKey={k('sec_detail')} anchorId="st-detail">
          <LockableSlider
            label={t('studio.generation.detailIntensity')}
            value={detailAmount}
            min="0" max="1" step="0.01"
            storageKey={k('detail_lock')}
            onChange={(e) => setDetailAmount(parseFloat(e.target.value))}
          />
          <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
            {t('studio.generation.detailHint')}
          </span>
        </StudioSection>
      )}

      {/* ENGINE (krea) — rebalance + enhancer + precision + LoRA always-on. */}
      {isKrea && (
        <StudioSection title={t('studio.generation.engine')} storageKey={k('sec_engine')} anchorId="st-engine">
          {/* NSFW / texture rebalance (node 30). Miroir exact du mode Generate. */}
          <label className="flex items-center justify-between gap-2 text-[0.6875rem] text-content-muted uppercase tracking-wide cursor-pointer">
            <span>{t('studio.generation.rebalance')}</span>
            <input
              type="checkbox"
              checked={rebalanceOn}
              onChange={(e) => setRebalanceOn(e.target.checked)}
              aria-label={t('studio.generation.rebalanceLabel')}
              className="accent-primary w-4 h-4"
            />
          </label>
          <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
            {t('studio.generation.rebalanceHint')}
          </span>
          {rebalanceOn && (
            <>
              <LockableSlider
                label={t('studio.generation.strength')}
                value={rebalanceStrength}
                min="1" max="8" step="0.5"
                storageKey={k('rebalance_lock')}
                onChange={(e) => setRebalanceStrength(parseFloat(e.target.value))}
              />
              <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">{t('studio.generation.rebalanceStrengthHint')}</span>
            </>
          )}

          {/* Krea2T Enhancer (patcher texte-adhérence, indépendant du rebalance). */}
          <div className="mt-2 pt-2 border-t border-white/10 flex flex-col gap-2.5">
            <label className="flex items-center justify-between gap-2 text-[0.6875rem] text-content-muted uppercase tracking-wide cursor-pointer">
              <span>{t('studio.generation.enhancer')}</span>
              <input
                type="checkbox"
                checked={enhancerOn}
                onChange={(e) => setEnhancerOn(e.target.checked)}
                aria-label={t('studio.generation.enhancerLabel')}
                className="accent-primary w-4 h-4"
              />
            </label>
            <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
              {t('studio.generation.enhancerHint')}
            </span>
            {enhancerOn && (
              <>
                <LockableSlider
                  label={t('studio.generation.strength')}
                  value={enhancerStrength}
                  min="0" max="2" step="0.1"
                  storageKey={k('enhancer_lock')}
                  onChange={(e) => setEnhancerStrength(parseFloat(e.target.value))}
                />
                <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">{t('studio.generation.enhancerStrengthHint')}</span>
              </>
            )}

            {/* Précision du loader (node 20 weight_dtype). */}
            <label className="flex flex-col gap-1 text-[0.6875rem] text-content-muted uppercase tracking-wide mt-1">
              {t('studio.generation.precision')}
              <select
                value={weightDtype}
                onChange={(e) => setWeightDtype(e.target.value)}
                aria-label={t('studio.generation.precisionLabel')}
                className="w-full bg-app/60 border border-border rounded-md px-2 py-1.5 text-content text-[0.8125rem] focus:border-primary focus:outline-none normal-case tracking-normal"
              >
                <option value="default">bf16 ({t('studio.generation.highPrecision')})</option>
                <option value="fp8_e4m3fn">{t('studio.generation.fast')} (fp8)</option>
                <option value="fp8_e4m3fn_fast">{t('studio.generation.fastPlus')} (fp8 fast)</option>
                <option value="fp8_e5m2">fp8 e5m2 ({t('studio.generation.wideRange')})</option>
              </select>
            </label>
            <span className="normal-case tracking-normal text-[0.625rem] text-content-muted/70 -mt-1">
              {t('studio.generation.precisionHint')}
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
                label={t('studio.generation.alwaysOn')}
                emptyHint={t('studio.generation.noAlwaysOn')}
                krea
                batchToggle
              />
            </div>
          )}
        </StudioSection>
      )}

      {/* NEGATIVE (zimage) — prompt négatif global du run. */}
      {isZ && (
        <StudioSection title={t('studio.generation.negative')} storageKey={k('sec_negative')} defaultOpen={false} anchorId="st-negative">
          <label className="flex flex-col gap-1">
            <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.generation.negativeOptional')}</span>
            <textarea
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              rows={3}
              placeholder={t('studio.generation.negativePlaceholder')}
              aria-label={t('studio.generation.negativePrompt')}
              className="rounded-lg border border-border bg-app/60 px-2.5 py-1.5 text-content text-sm resize-y min-h-[4rem]"
            />
          </label>
        </StudioSection>
      )}
    </div>
  );
}
