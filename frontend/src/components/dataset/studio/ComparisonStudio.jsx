// react-frontend/src/components/dataset/studio/ComparisonStudio.jsx
/**
 * Studio de COMPARAISON multi-LoRA (≥2 LoRA cochés). Branche « comparaison » de
 * StudioShell.
 *
 * Flux : règle un run (StudioRunSetup) sur la `selection` reçue → POST
 * /api/studio/run → useStudioRun(run_id) pilote l'affichage (poll + vote +
 * cancel/resume). Grille colonnes = LoRA × lignes = strength (LoraComparisonGrid),
 * panneau « 🏆 Classement LoRA » (data.lora_ranking). Vote rapide (file + swipe)
 * et lightbox réutilisent useQuickVote / QuickVoteModal / StudioResultViewer.
 *
 * Le LoraPicker reste dans StudioShell (partagé avec la branche 1-LoRA) ; ici on
 * reçoit la sélection figée et on pilote uniquement le run.
 */
import { useEffect, useMemo, useState } from 'react';
import { postJson } from '../../../api/fetchClient';
import { useToast } from '../../common/Toast';
import { useStudioRun } from '../../../hooks/useStudioRun';
import { useQuickVote } from '../../../hooks/useQuickVote';
import { fmt } from '../../../utils/studioFormat';
import { flipOrder } from './flipOrder';
import { DEFAULT_STRENGTHS, FAMILY_LABELS } from './constants';
import { blendConfigCount, buildSelectionsPayload, combineBlocker } from './loraStack';
import { axisPayload, axisTotal, effectiveAxis, toggleAxisValue } from './studioAxes';
import { mergeBatches } from './promptBatch';
import AxisPickers from './AxisPickers';
import { isStackRun, stackMembers } from './stackResults';
import StudioRunSetup from './StudioRunSetup';
import { readInjectTrigger, writeInjectTrigger } from './triggerPref';
import LoraStackPanel from './LoraStackPanel';
import StackCompositionPanel from './StackCompositionPanel';
import StackVariantsGrid from './StackVariantsGrid';
import StudioGenerationSettings from './StudioGenerationSettings';
import StudioActionBar from './StudioActionBar';
import StudioPreflightBanner from './StudioPreflightBanner';
import LoraComparisonGrid from './LoraComparisonGrid';
import LoraRankingPanel from './LoraRankingPanel';
import RunSelector from './RunSelector';
import QuickVoteModal from './QuickVoteModal';
import StudioResultViewer from './StudioResultViewer';

const rollSeed = () => Math.floor(Math.random() * 2 ** 31);

export default function ComparisonStudio({ selection, baseModels = [], axes = null,
  modelDefaults = null, runType = 'zimage', baseNote = null }) {
  const toast = useToast();

  // --- Réglages du run (persistés : recharger la page ne les perd plus) --------
  const [strengths, setStrengths] = useState(() => {
    try {
      const v = JSON.parse(localStorage.getItem('studioComp_strengths') || 'null');
      return Array.isArray(v) && v.length ? v : DEFAULT_STRENGTHS;
    } catch { return DEFAULT_STRENGTHS; }
  });
  const [prompt, setPrompt] = useState(() => {
    try { return localStorage.getItem('studioComp_prompt') || ''; } catch { return ''; }
  });
  // 🔤 Case « Trigger word » — MÊME préférence que le panneau du Test Studio et
  // le canvas (module partagé triggerPref) : décocher ici vaut partout.
  const [injectTrigger, setInjectTrigger] = useState(readInjectTrigger);
  const toggleInjectTrigger = (v) => {
    setInjectTrigger(v);
    writeInjectTrigger(v);
  };
  const [seed, setSeed] = useState(() => rollSeed());

  // 📝 LE LOT DE PROMPTS de cette surface. Il vit ICI et pas dans le panneau parce
  // que c'est ce composant qui construit le corps du POST — même règle que
  // RunSetupPanel sur les deux autres surfaces de lancement.
  // Délibérément NON persisté : un lot est l'intention d'UN lancement ; le
  // retrouver coché après un rechargement multiplierait un run qu'on croyait simple.
  const [historyBatch, setHistoryBatch] = useState([]);
  const [civitaiPicks, setCivitaiPicks] = useState([]);
  const toggleIn = (set) => (v) => set((cur) => (
    cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]));
  // Dédoublonné : un prompt Civitai déjà lancé est DANS l'historique, donc
  // cochable des deux côtés — deux cellules identiques seraient du GPU perdu.
  const pickedPrompts = mergeBatches(historyBatch, civitaiPicks);
  // 'compare' (historique : un LoRA seul par cellule) ou 'combine' (pile : tous les
  // LoRA cochés dans la MÊME image, chacun à son poids). Persisté comme le reste.
  const [mode, setMode] = useState(() => {
    try { return localStorage.getItem('studioComp_mode') === 'combine' ? 'combine' : 'compare'; }
    catch { return 'compare'; }
  });
  // Poids par LoRA de la pile, indexés par `${dataset_id}:${checkpoint}` → un poids
  // réglé survit au décochage d'un AUTRE LoRA.
  const [stackWeights, setStackWeights] = useState(() => {
    try { return JSON.parse(localStorage.getItem('studioComp_weights') || '{}') || {}; }
    catch { return {}; }
  });
  // Les poids COCHÉS par LoRA (balayage 🧬). Clé neuve : rien de ce qui est déjà
  // stocké ne change de sens, et une install qui n'en a pas lit {} = aucune case
  // cochée = les curseurs gouvernent, exactement comme avant.
  const [stackSets, setStackSets] = useState(() => {
    try { return JSON.parse(localStorage.getItem('studioComp_weightSets') || '{}') || {}; }
    catch { return {}; }
  });
  const toggleStackChip = (k, w) => setStackSets((cur) => {
    const list = Array.isArray(cur[k]) ? cur[k] : [];
    const next = list.includes(w) ? list.filter((v) => v !== w) : [...list, w];
    return { ...cur, [k]: next };
  });
  const [count, setCount] = useState(() => {
    try { return Math.max(1, parseInt(localStorage.getItem('studioComp_count'), 10) || 1); } catch { return 1; }
  });
  // 🎛 Axes de rendu — CFG, steps, et la 2e passe SDXL. Ils MANQUAIENT ici : cette
  // branche ne pouvait pas régler les steps alors que le studio mono-LoRA et le
  // panneau du canvas le pouvaient. Clés localStorage NEUVES : rien de ce qui est
  // déjà stocké ne change de sens, et une install qui n'en a pas lit null =
  // « le défaut de la famille », c'est-à-dire ce qu'elle lançait hier.
  const readAxis = (key) => {
    try {
      const v = JSON.parse(localStorage.getItem(key) || 'null');
      return Array.isArray(v) && v.length ? v : null;
    } catch { return null; }
  };
  // Modèle de base sélectionné : défaut = 1er de la liste fournie par le parent.
  // Se réinitialise quand baseModels change (changement de runType).
  // ⚠ DÉCLARÉ AVANT les axes : leurs défauts en dépendent (cf. `baseDefaults`
  // plus bas), et une `const` lue avant sa déclaration est une ReferenceError au
  // rendu — pas une valeur `undefined` qu'on rattraperait avec un `??`.
  const [selectedBase, setSelectedBase] = useState('');
  useEffect(() => {
    setSelectedBase(baseModels.length > 0 ? baseModels[0].filename : '');
  }, [baseModels]);
  const [selCfgs, setSelCfgs] = useState(() => readAxis('studioComp_cfgs'));
  const [selSteps, setSelSteps] = useState(() => readAxis('studioComp_steps'));
  const [selSteps2, setSelSteps2] = useState(() => readAxis('studioComp_steps2'));
  // Le défaut d'un axe dépend de la BASE, pas seulement de la famille : une base
  // non distillée (Z-Image Base, un modèle complet Krea 2 Raw) rend une esquisse
  // floue avec le cfg 1 / 8 steps de la Turbo. Le studio mono-LoRA lisait déjà
  // `model_defaults` ; cette branche ne le recevait pas et héritait donc des
  // mauvais chiffres pour exactement les modèles qui coûtent le plus cher à
  // tester. Un axe que l'utilisateur a touché (selCfgs non nul) n'est jamais
  // réécrit — ici comme là-bas.
  // `selectedBase` vaut '' quand la famille n'offre aucune alternative — et la clé
  // '' du payload EST celle du défaut élu. L'ignorer renvoyait une base non
  // distillée sur les chiffres Turbo (cfg 1 / 8 steps), l'esquisse floue de #18.
  const baseDefaults = (modelDefaults ? modelDefaults[selectedBase || ''] : null) || null;
  const defaultCfg = baseDefaults?.cfg ?? axes?.default_cfg;
  const defaultSteps = baseDefaults?.steps ?? axes?.default_steps;
  const effectiveCfgs = effectiveAxis(selCfgs, defaultCfg);
  const effectiveSteps = effectiveAxis(selSteps, defaultSteps);
  // ⚠ La 2e passe appartient au workflow SDXL. Sans cette garde, une sélection
  // laissée par un run SDXL suivrait l'utilisateur dans un run Z-Image (les
  // réglages sont persistés, la famille change avec les cases cochées) et
  // enverrait un axe que cette famille n'a pas. Famille sans 2e passe = axe vide.
  const effectiveSteps2 = axes?.steps2_choices
    ? effectiveAxis(selSteps2, axes.default_steps2) : [];
  useEffect(() => {
    try {
      localStorage.setItem('studioComp_cfgs', JSON.stringify(selCfgs));
      localStorage.setItem('studioComp_steps', JSON.stringify(selSteps));
      localStorage.setItem('studioComp_steps2', JSON.stringify(selSteps2));
    } catch { /* private mode */ }
  }, [selCfgs, selSteps, selSteps2]);
  useEffect(() => {
    try {
      localStorage.setItem('studioComp_strengths', JSON.stringify(strengths));
      localStorage.setItem('studioComp_prompt', prompt);
      localStorage.setItem('studioComp_count', String(count));
      localStorage.setItem('studioComp_mode', mode);
      localStorage.setItem('studioComp_weights', JSON.stringify(stackWeights));
      localStorage.setItem('studioComp_weightSets', JSON.stringify(stackSets));
    } catch { /* private mode */ }
  }, [strengths, prompt, count, mode, stackWeights, stackSets]);
  const [launching, setLaunching] = useState(false);
  // 409 `studio_missing` au lancement (P0-a) → bandeau des modèles/nodes manquants.
  const [preflight, setPreflight] = useState(null);
  // 409 `studio_arch_mismatch` : checkpoint dont l'arch RÉELLE contredit la famille.
  const [archMismatch, setArchMismatch] = useState(null);
  // Réglages de génération GLOBAUX (parité Generate) remontés par StudioGenerationSettings.
  // Objet snake_case déjà prêt à fusionner dans le POST /run (voir launch()).
  const [genSettings, setGenSettings] = useState({});
  const toggleStrength = (s) =>
    setStrengths((cur) => (cur.includes(s) ? cur.filter((v) => v !== s) : [...cur, s].sort((a, b) => a - b)));


  // --- Run piloté --------------------------------------------------------------
  const [runId, setRunId] = useState(null);
  const run = useStudioRun(runId);
  const data = run.data;
  const loras = data?.loras || [];
  const cells = useMemo(() => data?.cells || [], [data]);

  const vote = useQuickVote(run.rate);
  const [lbImg, setLbImg] = useState(null);
  const [showResults, setShowResults] = useState(true);
  const rateLightbox = (id, nv) => {
    run.rate(id, nv);
    setLbImg((p) => (p && p.id === id ? { ...p, rating: nv } : p));
  };

  // Les cellules RÉELLEMENT à l'écran. Sur une pile ce sont celles de toutes les
  // variantes de poids affichées, pas seulement celles du run ouvert : le vote rapide
  // et la lightbox doivent porter sur ce que l'utilisateur voit, sinon « 3 à voter »
  // en annonce 3 alors que 6 tuiles non votées sont sous ses yeux.
  const displayedCells = useMemo(() => {
    const variantCells = (data?.stack_variants || []).flatMap((v) => v.cells || []);
    if (!variantCells.length) return cells;
    const seen = new Set(variantCells.map((c) => c.id));
    return [...variantCells, ...cells.filter((c) => !seen.has(c.id))];
  }, [cells, data]);

  const unvoted = useMemo(
    () => displayedCells.filter((c) => c.status === 'done' && c.filename && !c.rating),
    [displayedCells],
  );
  const greens = useMemo(
    () => displayedCells.filter((c) => c.status === 'done' && c.filename && c.rating === 1),
    [displayedCells],
  );

  // Set navigable de la lightbox : les strengths d'un même rendu (même LoRA + même
  // seed) adjacentes → LoRA (dataset_id) → aspect → seed → STRENGTH en dernier. Ici
  // les cellules sont live (déjà dans ce composant) → on passe le set directement.
  // `displayedCells` et non `cells` : sur une pile, ouvrir une image d'une AUTRE
  // variante donnerait sinon une lightbox sans flèches (index -1 dans le set).
  const navImages = useMemo(
    () => flipOrder(displayedCells,
      (c) => [c.dataset_id ?? 0, c.aspect || '', c.seed ?? 0, c.strength ?? 0]),
    [displayedCells],
  );

  const combine = mode === 'combine';
  const combineBlocked = combine ? combineBlocker(selection) : null;

  // --- Vue PILE ---------------------------------------------------------------
  // Décidée par le RUN AFFICHÉ, pas par la bascule Compare/Blend : on peut ouvrir
  // une pile lancée hier alors que la bascule est repassée sur Compare, et l'inverse.
  const shownStack = useMemo(() => stackMembers(data), [data]);
  const showStackView = isStackRun(data);
  const [savingBest, setSavingBest] = useState(false);
  const [bestSavedAt, setBestSavedAt] = useState(null);
  // Changer de run efface la confirmation : « ★ Saved » sous une AUTRE pile que celle
  // qu'on vient d'épingler serait un mensonge.
  useEffect(() => { setBestSavedAt(null); }, [runId]);

  const saveStackBest = async ({ dataset_id: dsId, ...body }) => {
    setSavingBest(true);
    try {
      await postJson(`/api/dataset/${dsId}/lora-test/best`, body);
      setBestSavedAt(Date.now());
      toast.success('★ Stack weights saved as the best setting');
    } catch (e) {
      toast.error(e.message || 'Could not save the best setting');
    } finally {
      setSavingBest(false);
    }
  };

  // « Use these weights » : recharge les poids d'une variante dans les curseurs. Les
  // clés sont celles de loraStack.stackKey, donc les sliders les relisent tels quels.
  const useVariantWeights = (map) => {
    if (!map || Object.keys(map).length === 0) {
      toast.error('This run did not record enough to reload its weights');
      return;
    }
    setStackWeights((cur) => ({ ...cur, ...map }));
    setMode('combine');
    toast.success('Weights loaded — adjust them and run again to add a variant');
  };

  const launch = async () => {
    if (!selection.length || combineBlocked) return;
    if (!combine && !strengths.length) return;
    setLaunching(true);
    try {
      const body = {
        selections: buildSelectionsPayload(selection, { combine, weights: stackWeights, sets: stackSets }),
        // En combine chaque LoRA porte son poids : l'axe strengths n'est plus envoyé
        // (le backend le remplace par le poids du LoRA de tête).
        ...(combine ? { combine: true } : { strengths }),
        seed,
        count,
        // Base du run : '' (entrée « Official », Krea) ou rien de coché → absent,
        // le backend garde alors le défaut de la famille (UNET câblé / 1er modèle).
        z_model: selectedBase || undefined,
        // Réglages globaux (resolution_tier, negative/sampler/detail/…),
        // déjà gatés PAR FAMILLE côté backend — champs vides absents = défauts gardés.
        ...genSettings,
        // 🎛 CFG / steps / 2e passe, EN DERNIER : ce que l'utilisateur vient de
        // régler dans le panneau gagne sur tout réglage global homonyme. La route
        // les acceptait déjà et l'engine les balayait déjà — ce qui manquait était
        // ce corps, qui ne les portait pas, et le panneau, qui ne les offrait pas.
        ...axisPayload({ cfgs: effectiveCfgs, steps: effectiveSteps, steps2: effectiveSteps2 }),
      };
      if (prompt.trim()) body.prompt = prompt.trim();
      // 📝 L'axe lot. Rien de coché ⇒ clé absente : le corps reste octet pour
      // octet celui d'avant cette feature.
      if (pickedPrompts.length) body.prompts = [...pickedPrompts];
      // Case « Trigger word » décochée → prompt envoyé tel quel. Absent quand
      // cochée : le corps reste octet pour octet celui d'avant.
      if (!injectTrigger) body.inject_trigger = false;
      const dResp = await postJson('/api/studio/run', body);
      // Keep this defensive path even though apiFetch currently throws on
      // non-2xx: alternate clients/tests may return the structured 409 body.
      // Never announce success or retain a bogus run id in that case.
      if (!dResp?.ok) {
        let errorBody = dResp;
        if (typeof dResp?.json === 'function') {
          try { errorBody = await dResp.json(); } catch { errorBody = {}; }
        }
        setPreflight(errorBody?.studio_missing || null);
        setArchMismatch(errorBody?.studio_arch_mismatch || null);
        toast.error(errorBody?.error || 'Error on launch');
        return;
      }
      toast.success(`${dResp.created} generation(s) queued (seed ${dResp.seed}${dResp.count > 1 ? ` ×${dResp.count}` : ''})`);
      setRunId(dResp.run_id);
      setSeed(rollSeed());
      setPreflight(null);
      setArchMismatch(null);
    } catch (e) {
      // apiFetch throws on non-2xx; a 409 carries the itemized manques on e.body (P0-a)
      // or a wrong-arch checkpoint on e.body.studio_arch_mismatch.
      setPreflight(e?.body?.studio_missing || null);
      setArchMismatch(e?.body?.studio_arch_mismatch || null);
      toast.error(e.message || 'Error on launch');
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
      <aside className="flex flex-col gap-3 lg:sticky lg:top-16 lg:max-h-[calc(100vh-7rem)] lg:overflow-auto">
        {/* Ce que le défaut de base a d'anormal — affiché même sans sélecteur :
            l'install qui n'a qu'une base est celle qui doit le lire. */}
        {baseNote && (
          <p className="m-0 rounded-lg border border-amber-400/30 bg-amber-400/5 px-3 py-2
                        text-[0.6875rem] leading-snug text-amber-300/80 break-words">
            {baseNote}
          </p>
        )}
        {/* Picker de base — toutes familles. Krea : l'endpoint ne renvoie une liste
            (défaut élu + alternatives) que si des UNET Krea locaux existent ; sinon
            vide → le sélecteur reste caché (défaut élu appliqué au node 20). */}
        {baseModels.length > 0 && (
          <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface p-3">
            <span className="text-content-muted text-[0.625rem] uppercase">
              Base model ({FAMILY_LABELS[runType] || 'Z-Image'})
            </span>
            <select
              value={selectedBase}
              onChange={(e) => setSelectedBase(e.target.value)}
              aria-label="Base model for this run"
              className="rounded border border-border bg-app/60 px-1.5 py-1 text-content text-sm"
            >
              {baseModels.map((m) => (
                <option key={m.filename} value={m.filename}>{m.label}</option>
              ))}
            </select>
          </div>
        )}
        <LoraStackPanel selection={selection} mode={mode} onMode={setMode}
          weights={stackWeights}
          sets={stackSets}
          onToggleChip={toggleStackChip}
          count={count}
          secondsPerImage={axes?.seconds_per_image ?? null}
          injectTrigger={injectTrigger}
          onWeight={(k, v) => setStackWeights((cur) => ({ ...cur, [k]: v }))} />
        <div id="st-setup" className="scroll-mt-16">
          <StudioRunSetup
            selectionCount={selection.length}
            strengths={strengths}
            onToggleStrength={toggleStrength}
            prompt={prompt}
            onPrompt={setPrompt}
            seed={seed}
            onReroll={() => setSeed(rollSeed())}
            count={count}
            onCount={setCount}
            onLaunch={launch}
            launching={launching}
            gpuBusy={data?.gpu_busy}
            batchMult={1 + ((genSettings.batch_loras || []).length)}
            combine={combine}
            combineBlocked={combineBlocked}
            configCount={blendConfigCount(selection, { weights: stackWeights, sets: stackSets })}
            axisTotal={axisTotal({ cfgs: effectiveCfgs, steps: effectiveSteps, steps2: effectiveSteps2 })}
            secondsPerImage={axes?.seconds_per_image ?? null}
            injectTrigger={injectTrigger}
            onInjectTrigger={toggleInjectTrigger}
            batchPrompts={historyBatch}
            onToggleBatchPrompt={toggleIn(setHistoryBatch)}
            onClearBatchPrompts={() => setHistoryBatch([])}
            civitaiPicks={civitaiPicks}
            onToggleCivitaiPick={toggleIn(setCivitaiPicks)}
            onClearCivitaiPicks={() => setCivitaiPicks([])}
            pickedPrompts={pickedPrompts}
            /* 🎛 Les axes de rendu, dans le panneau de réglage du run et pas dans
               un bloc à part : c'est le même geste que choisir une strength. Le
               MÊME composant que le studio mono-LoRA et le canvas — base et format
               sont déjà réglés au-dessus ici, donc ces deux blocs restent muets. */
            axisSlot={axes ? (
              /* defaultCfg/defaultSteps = ceux de la BASE choisie (repli sur la
                 famille) : la puce marquée « défaut » doit désigner la valeur
                 réellement lancée, sinon elle désigne celle qui rend flou. */
              <AxisPickers
                zModels={null} effectiveModels={[]} onToggleModel={() => {}}
                aspects={null} effectiveAspects={[]} onToggleAspect={() => {}}
                cfgChoices={axes.cfg_choices} effectiveCfgs={effectiveCfgs}
                onToggleCfg={(v) => setSelCfgs((cur) => toggleAxisValue(effectiveAxis(cur, defaultCfg), v))}
                defaultCfg={defaultCfg}
                stepsChoices={axes.steps_choices} effectiveSteps={effectiveSteps}
                onToggleStep={(v) => setSelSteps((cur) => toggleAxisValue(effectiveAxis(cur, defaultSteps), v))}
                defaultSteps={defaultSteps}
                steps2Choices={axes.steps2_choices} effectiveSteps2={effectiveSteps2}
                onToggleStep2={(v) => setSelSteps2((cur) => toggleAxisValue(effectiveAxis(cur, axes.default_steps2), v))}
                defaultSteps2={axes.default_steps2}
                fmt={fmt}
              />
            ) : null}
          />
        </div>
        {/* Réglages de génération globaux (parité Generate, hors prompt builder).
            key=runType → remonte proprement au changement de famille (état/localStorage
            namespacés par famille). aspectPicker : en comparaison le ratio n'est pas
            un axe → choix GLOBAL du format ici (envoyé comme axe à 1 valeur). */}
        <StudioGenerationSettings
          key={runType}
          family={runType}
          storagePrefix={`studioGenComp_${runType}`}
          aspectPicker
          onChange={setGenSettings}
        />
        {/* Une pile n'a qu'un LoRA « testé » : son classement par-LoRA n'a qu'une
            ligne et n'apprend rien. À sa place, ce qui la définit — sa composition. */}
        {showStackView ? (
          <StackCompositionPanel members={shownStack} onSaveBest={saveStackBest}
            saving={savingBest} savedAt={bestSavedAt}
            // La vérité du RUN affiché, pas de la case : une seule cellule
            // False suffit (toutes le sont sur un run décoché).
            injectTrigger={!cells.some((c) => c.inject_trigger === false)} />
        ) : (
          <LoraRankingPanel ranking={data?.lora_ranking} />
        )}
      </aside>

      <main id="st-results" className="flex flex-col gap-3 min-w-0 scroll-mt-16">
        <StudioPreflightBanner missing={preflight} archMismatch={archMismatch}
          onDismiss={() => { setPreflight(null); setArchMismatch(null); }} />
        {data?.comfyui_recovery?.requires_comfyui_restart_confirmation && (
          <div className="flex items-center gap-2 flex-wrap rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
            <span aria-hidden>⚠</span>
            <span className="text-content text-sm">A ComfyUI submission has an unknown outcome. Restart ComfyUI first, then confirm it here; the paused cell will become resumable.</span>
            <button type="button" disabled={run.confirmingComfyuiRestart}
              onClick={run.confirmComfyuiRestart}
              className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
              {run.confirmingComfyuiRestart ? 'Confirming…' : '✓ J’ai redémarré ComfyUI'}
            </button>
          </div>
        )}

        {data?.pending > 0 && (
          <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-raised px-3 py-2" role="status">
            <span className="inline-block w-4 h-4 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
            <span className="text-content text-sm">
              {data.generating ?? data.running ?? 0} generating · {data.queued ?? data.pending} queued
            </span>
            <button type="button" onClick={run.cancel}
              className="ml-auto px-2.5 py-1 rounded-lg bg-red-600/80 text-white text-xs font-semibold">
              Stop (resumable)
            </button>
          </div>
        )}
        {!data?.pending && data?.resumable > 0 && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
            <span aria-hidden>⏸</span>
            <span className="text-content text-sm">{data.resumable} stopped cell(s) — resumable with their settings</span>
            <button type="button" disabled={!!data?.gpu_busy} onClick={run.resume}
              className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
              ▶ Resume the test
            </button>
          </div>
        )}

        {!runId ? (
          <p className="text-content-subtle text-sm rounded-lg border border-border bg-surface px-3 py-6 text-center">
            Set up the run on the left then “🚀 Run the test”{combine
              ? ` to render the ${selection.length} LoRAs together in one image.`
              : ` to compare the ${selection.length} LoRAs side by side.`}
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            <RunSelector
              runs={[]}
              activeRunKey={null}
              onSelect={() => {}}
              unvotedCount={unvoted.length}
              onStartVote={() => vote.startVoting(unvoted)}
              greenCount={greens.length}
              onStartReVote={() => vote.startVoting(greens, '♻️ Reconfirm the 👍')}
              displayedCount={cells.length}
              showResults={showResults}
              onToggleResults={() => setShowResults((v) => !v)}
            />
            {showResults && (showStackView ? (
              <StackVariantsGrid members={shownStack} variants={data?.stack_variants}
                onRate={run.rate} onOpen={setLbImg} onSelectRun={setRunId}
                onUseWeights={useVariantWeights} />
            ) : (
              <LoraComparisonGrid loras={loras} cells={cells} onRate={run.rate} onOpen={setLbImg} />
            ))}
          </div>
        )}
      </main>

      <QuickVoteModal vote={vote} datasetId={vote.current?.dataset_id} fmt={fmt} />
      {lbImg && (
        <StudioResultViewer img={lbImg} items={navImages}
          onRate={rateLightbox} onNavigate={setLbImg} onClose={() => setLbImg(null)} />
      )}

      {/* Barre de commande fixe : Run toujours visible + raccourcis de sections. */}
      <StudioActionBar
        shortcuts={[
          { id: 'st-loras', emoji: '🧬', label: 'LoRAs' },
          { id: 'st-setup', emoji: '📝', label: 'Prompt & seed' },
          { id: 'st-format', emoji: '📐', label: 'Format' },
          ...(runType === 'krea' ? [
            { id: 'st-sampling', emoji: '🎛️', label: 'Sampling' },
            { id: 'st-engine', emoji: '⚙️', label: 'Engine' },
          ] : []),
          ...(runType === 'sdxl' ? [{ id: 'st-detail', emoji: '✨', label: 'Detail' }] : []),
          ...(runType === 'zimage' ? [{ id: 'st-negative', emoji: '🚫', label: 'Negative' }] : []),
          { id: 'st-results', emoji: '🖼️', label: 'Results' },
        ]}
        canRun={!!selection.length && !!strengths.length && !launching && !data?.gpu_busy}
        running={launching}
        onRun={launch}
      />
    </div>
  );
}
