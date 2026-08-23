import { useEffect, useRef, useState } from 'react';
import { DEFAULT_STRENGTHS } from '../components/dataset/studio/constants';
import { defaultCfgFor, defaultStepsFor, mixedModelDefaults } from '../utils/studioModelDefaults';
import {
  addGuestCheckpoint, chosenCheckpoints, normalizeGuestCheckpoints,
  removeGuestCheckpoint,
} from '../utils/studioGuestCheckpoints';

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
/* `pinnedCheckpoints` (optionnel) : la liste de checkpoints est IMPOSÉE par
   l'appelant au lieu d'être cochée dans le picker. C'est la seule chose que le
   ◉ LoRA Canvas fait autrement que le Studio de test — là-bas les checkpoints se
   choisissent en cliquant les pastilles des nœuds, éventuellement sur plusieurs
   datasets. Tout le reste (modèle, format, cfg, steps, seed, ×N, réglages
   globaux) passe par exactement ce hook et exactement ce composant, donc les
   deux écrans ne peuvent pas diverger. */
export function useStudioForm(d, datasetId, family = null,
  { pinnedCheckpoints = null, preselectBase = null } = {}) {
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
  // Guest files (not in this dataset's pool). Independent of selCps: null
  // selCps still means "all of mine" without silently ticking every guest.
  const [guestCps, setGuestCps] = useState(() => normalizeGuestCheckpoints(initial.guestCps));
  const [selGuests, setSelGuests] = useState(initial.selGuests ?? null);
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

  /* `preselectBase` — a base named in the URL (`/studio?base=…`), used by the
     full-model card to open ON the model you asked to test.

     Three deliberate choices:
       • it wins over the persisted selection, ONCE. Same rule as the LoRA
         picker's URL preselection: an explicit navigation is a fresher intent
         than what localStorage remembers, but only for the arrival, so the very
         next click of the user is never fought over;
       • it clears the persisted CFG/steps, and that is the whole point for an
         undistilled base. A session spent on a Turbo checkpoint leaves cfg 1 /
         8 steps behind; inherited by a Raw full model they render a blurry
         sketch that reads as "the training failed". Cleared, the axes re-seed
         from `model_defaults` — the run's OWN sample settings;
       • it waits for the base to actually be in `z_models`. Selecting a value
         the payload does not know would be silently dropped by the backend
         whitelist and fall back to the first base, i.e. generate on the WRONG
         model while the UI claims otherwise. */
  const preselectedBaseRef = useRef(false);
  useEffect(() => {
    if (preselectedBaseRef.current || !preselectBase) return;
    if (!(d?.z_models || []).some((m) => m.value === preselectBase)) return;
    preselectedBaseRef.current = true;
    setSelModels([preselectBase]);
    setSelCfgs(null);
    setSelSteps(null);
  }, [preselectBase, d?.z_models]);

  // Persiste les sélections à chaque changement (refresh-safe, par dataset).
  useEffect(() => {
    try {
      localStorage.setItem(persistKey, JSON.stringify({
        selCps, guestCps, selGuests, selSts, seed, seedLocked, genCount, promptText,
        selModels, selAspects, selCfgs, selSteps, selSteps2,
      }));
    } catch { /* quota / private mode — la persistance est best-effort */ }
  }, [persistKey, selCps, guestCps, selGuests, selSts, seed, seedLocked, genCount, promptText, selModels, selAspects, selCfgs, selSteps, selSteps2]);

  const checkpoints = d?.checkpoints || [];
  const allFns = checkpoints.map((c) => c.filename);
  // Filtre les checkpoints persistés qui n'existent plus (dataset modifié depuis).
  // Checkpoints imposés (canvas) → ils sont la sélection, telle quelle. Ne PAS
  // les filtrer sur `allFns` : ils viennent de plusieurs datasets, alors que
  // `d.checkpoints` est la liste d'un seul. Guests are a second list: null
  // selCps still means "all of mine", never "all guests too".
  const chosenCps = chosenCheckpoints({
    mineFns: allFns, selCps, guests: guestCps, selGuests, pinned: pinnedCheckpoints,
  });
  const effectivePrompt = promptText ?? (d?.prompt || '');
  // Défaut = 1re entrée de la liste — y compris « Official » (value '' , Krea) pour
  // que la puce par défaut apparaisse pressée ; le backend mappe '' → défaut câblé.
  const effectiveModels = selModels ?? (d?.z_models?.length ? [d.z_models[0].value] : []);
  const effectiveAspects = selAspects ?? (d?.default_aspect ? [d.default_aspect] : ['9:16']);
  // CFG/steps par MODÈLE DE BASE (bobba84, GitHub #18) : Z-Image Base n'est pas
  // distillé et ne doit pas hériter des réglages Turbo (cfg 1, 8 steps), qui ruinent
  // son rendu. `selCfgs`/`selSteps` non nuls = l'utilisateur a choisi → jamais
  // réécrit ; le défaut par modèle ne s'applique qu'à l'axe encore intact.
  const modelDefaultCfg = defaultCfgFor(d, effectiveModels);
  const modelDefaultSteps = defaultStepsFor(d, effectiveModels);
  const effectiveCfgs = selCfgs ?? [modelDefaultCfg];
  const effectiveSteps = selSteps ?? [modelDefaultSteps];
  // Pass 2 (detail daemon) : SDXL uniquement. Z-Image → default_steps2 null → axe vide
  // (×1 dans le compteur, pas envoyé au backend).
  const effectiveSteps2 = selSteps2 ?? (d?.default_steps2 != null ? [d.default_steps2] : []);
  // Everything the axes multiply EXCEPT the checkpoints and the strength sweep.
  // 🧬 Blend collapses those two into one configuration (each LoRA carries its own
  // weight, they all load in the same image), so its cell count is exactly this —
  // exposed rather than divided back out of `total`, which would be a lie the
  // moment one of the two factors is zero.
  const axisTotal = effectiveAspects.length * effectiveCfgs.length * effectiveSteps.length
    * Math.max(1, effectiveSteps2.length) * Math.max(1, effectiveModels.length);
  const total = chosenCps.length * selSts.length * axisTotal;

  const toggleCp = (fn) =>
    setSelCps((cur) => {
      const base = cur ?? allFns;
      return base.includes(fn) ? base.filter((f) => f !== fn) : [...base, fn];
    });
  const addGuest = (filename) => {
    const fn = String(filename || '').trim();
    if (!fn) return;
    if (allFns.includes(fn)) {
      setSelCps((cur) => {
        const base = cur ?? allFns;
        return base.includes(fn) ? base : [...base, fn];
      });
      return;
    }
    setGuestCps((cur) => addGuestCheckpoint(cur, fn, allFns));
    setSelGuests((cur) => {
      if (cur == null) return null;
      return cur.includes(fn) ? cur : [...cur, fn];
    });
  };
  const removeGuest = (filename) => {
    setGuestCps((cur) => removeGuestCheckpoint(cur, filename));
    setSelGuests((cur) => (cur == null ? null : cur.filter((f) => f !== filename)));
  };
  const toggleGuest = (fn) =>
    setSelGuests((cur) => {
      const fns = guestCps.map((g) => g.filename);
      const base = cur ?? fns;
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
    chosenCps, effectivePrompt, effectiveModels, effectiveAspects, effectiveCfgs, effectiveSteps, effectiveSteps2, total, axisTotal,
    // Défauts DU MODÈLE sélectionné (pour l'étiquette « default … » des pickers).
    modelDefaultCfg, modelDefaultSteps,
    mixedModelDefaults: mixedModelDefaults(d, effectiveModels),
    guestCps,
    setSelSts, setSeed, setSeedLocked, setGenCount, setPromptText,
    toggleCp, addGuest, removeGuest, toggleGuest,
    toggleSt, toggleAspect, toggleCfg, toggleStep, toggleStep2, toggleModel, rollSeed, nextSeed,
  };
}
