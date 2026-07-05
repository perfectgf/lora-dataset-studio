import { useEffect, useState } from 'react';
import { DEFAULT_STRENGTHS } from '../components/dataset/studio/constants';

const rollSeed = () => Math.floor(Math.random() * 2 ** 31);

/**
 * État de formulaire du Studio de test LoRA + dérivations (valeurs « effective »)
 * + toggles. Extrait 1:1 de l'ancien LoraTestStudio.jsx.
 *
 * Les sélections sont PERSISTÉES dans localStorage par dataset : un refresh de la
 * page retrouve les derniers paramètres (checkpoints, strengths, prompt, modèle,
 * formats/cfg/steps, verrou seed, gén/config). Clé namespacée `studioForm_v1_<id>`.
 *
 * `d` = payload de useLoraTestStudio (peut être null au 1er render).
 * `datasetId` = id du dataset (namespace de persistance).
 */
export function useStudioForm(d, datasetId, family = null) {
  // Persistance namespacée par dataset ET par famille : chaque pipeline (ZIT/SDXL/Krea)
  // garde ses propres axes (checkpoints/strengths/modèle…). Le composant studio est
  // remonté quand la famille change → ce hook re-lit la bonne clé au montage.
  const [persistKey] = useState(() => `studioForm_v1_${datasetId || 'x'}_${family || 'default'}`);
  // Lecture unique au montage (lazy) — restaure les derniers paramètres.
  const [initial] = useState(() => {
    try { return JSON.parse(localStorage.getItem(`studioForm_v1_${datasetId || 'x'}_${family || 'default'}`)) || {}; }
    catch { return {}; }
  });

  const [selCps, setSelCps] = useState(initial.selCps ?? null);              // null = tous cochés
  const [selSts, setSelSts] = useState(initial.selSts ?? DEFAULT_STRENGTHS);
  const [seed, setSeed] = useState(() => initial.seed ?? rollSeed());
  const [seedLocked, setSeedLocked] = useState(initial.seedLocked ?? false);
  const [genCount, setGenCount] = useState(initial.genCount ?? 1);
  const [promptText, setPromptText] = useState(initial.promptText ?? null);  // null = suit d.prompt
  const [selModels, setSelModels] = useState(initial.selModels ?? null);
  const [selAspects, setSelAspects] = useState(initial.selAspects ?? null);
  const [selCfgs, setSelCfgs] = useState(initial.selCfgs ?? null);
  const [selSteps, setSelSteps] = useState(initial.selSteps ?? null);
  const [selSteps2, setSelSteps2] = useState(initial.selSteps2 ?? null);  // SDXL : pass 2 (detail daemon)

  // Persiste les sélections à chaque changement (refresh-safe, par dataset).
  useEffect(() => {
    try {
      localStorage.setItem(persistKey, JSON.stringify({
        selCps, selSts, seed, seedLocked, genCount, promptText, selModels, selAspects, selCfgs, selSteps, selSteps2,
      }));
    } catch { /* quota / private mode — la persistance est best-effort */ }
  }, [persistKey, selCps, selSts, seed, seedLocked, genCount, promptText, selModels, selAspects, selCfgs, selSteps, selSteps2]);

  const checkpoints = d?.checkpoints || [];
  const allFns = checkpoints.map((c) => c.filename);
  // Filtre les checkpoints persistés qui n'existent plus (dataset modifié depuis).
  const chosenCps = (selCps ?? allFns).filter((fn) => allFns.includes(fn));
  const effectivePrompt = promptText ?? (d?.prompt || '');
  const effectiveModels = selModels ?? (d?.z_models?.[0]?.value ? [d.z_models[0].value] : []);
  const effectiveAspects = selAspects ?? (d?.default_aspect ? [d.default_aspect] : ['9:16']);
  const effectiveCfgs = selCfgs ?? (d?.default_cfg != null ? [d.default_cfg] : [1.0]);
  const effectiveSteps = selSteps ?? (d?.default_steps != null ? [d.default_steps] : [8]);
  // Pass 2 (detail daemon) : SDXL uniquement. Z-Image → default_steps2 null → axe vide
  // (×1 dans le compteur, pas envoyé au backend).
  const effectiveSteps2 = selSteps2 ?? (d?.default_steps2 != null ? [d.default_steps2] : []);
  const total = chosenCps.length * selSts.length * effectiveAspects.length
    * effectiveCfgs.length * effectiveSteps.length * Math.max(1, effectiveSteps2.length)
    * Math.max(1, effectiveModels.length);

  const toggleCp = (fn) =>
    setSelCps((cur) => {
      const base = cur ?? allFns;
      return base.includes(fn) ? base.filter((f) => f !== fn) : [...base, fn];
    });
  const toggleSt = (s) =>
    setSelSts((cur) => (cur.includes(s) ? cur.filter((v) => v !== s) : [...cur, s].sort((a, b) => a - b)));
  // Toggle qui garde au moins une valeur (formats/cfg/steps).
  const _toggleKeep = (setter, getEff) => (v) =>
    setter((cur) => {
      const base = cur ?? getEff();
      const next = base.includes(v) ? base.filter((x) => x !== v) : [...base, v].sort((a, b) => a - b);
      return next.length ? next : base;
    });
  const toggleAspect = (a) =>
    setSelAspects((cur) => {
      const base = cur ?? effectiveAspects;
      const next = base.includes(a) ? base.filter((v) => v !== a) : [...base, a];
      return next.length ? next : base;
    });
  const toggleCfg = _toggleKeep(setSelCfgs, () => effectiveCfgs);
  const toggleStep = _toggleKeep(setSelSteps, () => effectiveSteps);
  const toggleStep2 = _toggleKeep(setSelSteps2, () => effectiveSteps2);
  // Modèles = chaînes (pas de tri numérique) ; garde au moins un modèle sélectionné.
  const toggleModel = (m) =>
    setSelModels((cur) => {
      const base = cur ?? effectiveModels;
      const next = base.includes(m) ? base.filter((x) => x !== m) : [...base, m];
      return next.length ? next : base;
    });

  // Seed auto à chaque lancement sauf verrou. Renvoie la seed à utiliser.
  const nextSeed = () => {
    const s = seedLocked ? seed : rollSeed();
    if (!seedLocked) setSeed(s);
    return s;
  };

  return {
    selSts, seed, seedLocked, genCount, promptText, selModels,
    chosenCps, effectivePrompt, effectiveModels, effectiveAspects, effectiveCfgs, effectiveSteps, effectiveSteps2, total,
    setSelSts, setSeed, setSeedLocked, setGenCount, setPromptText,
    toggleCp, toggleSt, toggleAspect, toggleCfg, toggleStep, toggleStep2, toggleModel, rollSeed, nextSeed,
  };
}
