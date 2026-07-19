// Axes optionnels du balayage : modèle Z-Image (select), formats / CFG / steps (multi-toggles).
// Extrait behavior-preserving de LoraTestStudio.jsx (blocs modèle/formats/CFG/steps).
// Chaque bloc conserve sa garde de rendu d'origine :
//   - modèle : z_models tableau de longueur > 1
//   - formats : aspects tableau de longueur > 1
//   - CFG : cfgChoices est un tableau
//   - steps : stepsChoices est un tableau
import { useI18n } from '../../../i18n/I18nContext';

export default function AxisPickers({
  zModels, effectiveModels, onToggleModel,
  aspects, effectiveAspects, onToggleAspect,
  cfgChoices, effectiveCfgs, onToggleCfg, defaultCfg,
  stepsChoices, effectiveSteps, onToggleStep, defaultSteps,
  // SDXL uniquement : 2e passe (detail daemon). Absent (null) pour Z-Image.
  steps2Choices, effectiveSteps2, onToggleStep2, defaultSteps2,
  fmt,
}) {
  const { t } = useI18n();
  // En SDXL (2 passes), le 1er picker devient « pass 1 (classic) » ; sinon « Steps ».
  const hasPass2 = Array.isArray(steps2Choices);
  return (
    <>
      {Array.isArray(zModels) && zModels.length > 1 && (
        <div className="flex flex-col gap-1">
          {/* Libellé générique : la liste vient du payload PAR FAMILLE (Z-Image,
              checkpoints SDXL, ou « Official + UNET Krea locaux »). */}
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.axes.baseModel')}</span>
          <div className="flex gap-2 flex-wrap">
            {zModels.map((m) => (
              <button key={m.value} type="button" onClick={() => onToggleModel(m.value)}
                aria-pressed={effectiveModels.includes(m.value)}
                className={`px-2.5 py-1 rounded-lg border text-[0.75rem] transition-colors ${
                  effectiveModels.includes(m.value)
                    ? 'border-purple-400/60 bg-purple-500/20 text-purple-200 font-semibold'
                    : 'border-border bg-surface text-content-muted'}`}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {Array.isArray(aspects) && aspects.length > 1 && (
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.axes.formats')}</span>
          <div className="flex gap-2 flex-wrap">
            {aspects.map((a) => (
              <button key={a} type="button" onClick={() => onToggleAspect(a)}
                aria-pressed={effectiveAspects.includes(a)}
                className={`px-2.5 py-1 rounded-lg border text-[0.75rem] tabular-nums transition-colors ${
                  effectiveAspects.includes(a)
                    ? 'border-purple-400/60 bg-purple-500/20 text-purple-200 font-semibold'
                    : 'border-border bg-surface text-content-muted'}`}>
                {a}
              </button>
            ))}
          </div>
        </div>
      )}

      {Array.isArray(cfgChoices) && (
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.axes.cfg', { value: fmt(defaultCfg ?? 1.0) })}</span>
          <div className="flex gap-2 flex-wrap">
            {cfgChoices.map((v) => (
              <button key={v} type="button" onClick={() => onToggleCfg(v)}
                aria-pressed={effectiveCfgs.includes(v)}
                className={`px-2.5 py-1 rounded-lg border text-[0.75rem] tabular-nums transition-colors ${
                  effectiveCfgs.includes(v)
                    ? 'border-purple-400/60 bg-purple-500/20 text-purple-200 font-semibold'
                    : 'border-border bg-surface text-content-muted'}`}>
                {fmt(v)}
              </button>
            ))}
          </div>
        </div>
      )}

      {Array.isArray(stepsChoices) && (
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">
            {hasPass2
              ? t('studio.axes.stepsPass1', { value: defaultSteps ?? 8 })
              : t('studio.axes.steps', { value: defaultSteps ?? 8 })}
          </span>
          <div className="flex gap-2 flex-wrap">
            {stepsChoices.map((v) => (
              <button key={v} type="button" onClick={() => onToggleStep(v)}
                aria-pressed={effectiveSteps.includes(v)}
                className={`px-2.5 py-1 rounded-lg border text-[0.75rem] tabular-nums transition-colors ${
                  effectiveSteps.includes(v)
                    ? 'border-purple-400/60 bg-purple-500/20 text-purple-200 font-semibold'
                    : 'border-border bg-surface text-content-muted'}`}>
                {v}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* SDXL : 2e passe (detail daemon, node 57 du workflow HQ). Affichée seulement
          quand le backend la propose (steps2Choices non-null = dataset SDXL). */}
      {hasPass2 && (
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.axes.stepsPass2', { value: defaultSteps2 ?? 8 })}</span>
          <div className="flex gap-2 flex-wrap">
            {steps2Choices.map((v) => (
              <button key={v} type="button" onClick={() => onToggleStep2(v)}
                aria-pressed={effectiveSteps2.includes(v)}
                className={`px-2.5 py-1 rounded-lg border text-[0.75rem] tabular-nums transition-colors ${
                  effectiveSteps2.includes(v)
                    ? 'border-amber-400/60 bg-amber-500/20 text-amber-200 font-semibold'
                    : 'border-border bg-surface text-content-muted'}`}>
                {v}
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
