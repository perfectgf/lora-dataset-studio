// Axes optionnels du balayage : modèle Z-Image (select), formats / CFG / steps (multi-toggles).
// Extrait behavior-preserving de LoraTestStudio.jsx (blocs modèle/formats/CFG/steps).
// Chaque bloc conserve sa garde de rendu d'origine :
//   - modèle : z_models tableau de longueur > 1
//   - formats : aspects tableau de longueur > 1
//   - CFG : cfgChoices est un tableau
//   - steps : stepsChoices est un tableau
export default function AxisPickers({
  zModels, effectiveModels, onToggleModel,
  aspects, effectiveAspects, onToggleAspect,
  cfgChoices, effectiveCfgs, onToggleCfg, defaultCfg,
  stepsChoices, effectiveSteps, onToggleStep, defaultSteps,
  // SDXL uniquement : 2e passe (detail daemon). Absent (null) pour Z-Image.
  steps2Choices, effectiveSteps2, onToggleStep2, defaultSteps2,
  fmt,
}) {
  // En SDXL (2 passes), le 1er picker devient « pass 1 (classic) » ; sinon « Steps ».
  const hasPass2 = Array.isArray(steps2Choices);
  return (
    <>
      {Array.isArray(zModels) && zModels.length > 1 && (
        <div className="flex flex-col gap-1">
          <span className="text-content-muted text-[0.625rem] uppercase">Base Z-Image model (multi)</span>
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
          <span className="text-content-muted text-[0.625rem] uppercase">Image formats (multi)</span>
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
          <span className="text-content-muted text-[0.625rem] uppercase">CFG (multi) — default {fmt(defaultCfg ?? 1.0)}</span>
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
            {hasPass2 ? 'Steps · pass 1 — classic (multi)' : 'Steps (multi)'} — default {defaultSteps ?? 8}
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
          <span className="text-content-muted text-[0.625rem] uppercase">Steps · pass 2 — detail daemon (multi) — default {defaultSteps2 ?? 8}</span>
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
