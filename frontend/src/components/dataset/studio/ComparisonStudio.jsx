// react-frontend/src/components/dataset/studio/ComparisonStudio.jsx
/**
 * Studio de COMPARAISON multi-LoRA (≥2 LoRA cochés). Branche « comparaison » de
 * StudioShell.
 *
 * Flux : règle un run (StudioRunSetup) sur la `selection` reçue → POST
 * /api/studio/run → useStudioRun(run_id) pilote l'affichage (poll + vote +
 * cancel/resume). Grille colonnes = LoRA × lignes = strength (LoraComparisonGrid),
 * panneau « 🏆 Classement LoRA » (data.lora_ranking). Vote rapide (file + swipe)
 * et lightbox réutilisent useQuickVote / QuickVoteModal / ResultLightbox.
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
import StudioRunSetup from './StudioRunSetup';
import StudioGenerationSettings from './StudioGenerationSettings';
import StudioActionBar from './StudioActionBar';
import StudioPreflightBanner from './StudioPreflightBanner';
import LoraComparisonGrid from './LoraComparisonGrid';
import LoraRankingPanel from './LoraRankingPanel';
import RunSelector from './RunSelector';
import QuickVoteModal from './QuickVoteModal';
import ResultLightbox from './ResultLightbox';
import { useI18n } from '../../../i18n/I18nContext';

const rollSeed = () => Math.floor(Math.random() * 2 ** 31);

export default function ComparisonStudio({ selection, baseModels = [], runType = 'zimage' }) {
  const toast = useToast();
  const { t } = useI18n();

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
  const [seed, setSeed] = useState(() => rollSeed());
  const [count, setCount] = useState(() => {
    try { return Math.max(1, parseInt(localStorage.getItem('studioComp_count'), 10) || 1); } catch { return 1; }
  });
  useEffect(() => {
    try {
      localStorage.setItem('studioComp_strengths', JSON.stringify(strengths));
      localStorage.setItem('studioComp_prompt', prompt);
      localStorage.setItem('studioComp_count', String(count));
    } catch { /* private mode */ }
  }, [strengths, prompt, count]);
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

  // Modèle de base sélectionné : défaut = 1er de la liste fournie par le parent.
  // Se réinitialise quand baseModels change (changement de runType).
  const [selectedBase, setSelectedBase] = useState('');
  useEffect(() => {
    setSelectedBase(baseModels.length > 0 ? baseModels[0].filename : '');
  }, [baseModels]);

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

  const unvoted = useMemo(
    () => cells.filter((c) => c.status === 'done' && c.filename && !c.rating),
    [cells],
  );
  const greens = useMemo(
    () => cells.filter((c) => c.status === 'done' && c.filename && c.rating === 1),
    [cells],
  );

  // Set navigable de la lightbox : les strengths d'un même rendu (même LoRA + même
  // seed) adjacentes → LoRA (dataset_id) → aspect → seed → STRENGTH en dernier. Ici
  // les cellules sont live (déjà dans ce composant) → on passe le set directement.
  const navImages = useMemo(
    () => flipOrder(cells, (c) => [c.dataset_id ?? 0, c.aspect || '', c.seed ?? 0, c.strength ?? 0]),
    [cells],
  );

  const launch = async () => {
    if (!selection.length || !strengths.length) return;
    setLaunching(true);
    try {
      const body = {
        selections: selection.map((s) => ({ dataset_id: s.dataset_id, checkpoint: s.checkpoint })),
        strengths,
        seed,
        count,
        // Base du run : '' (entrée « Official », Krea) ou rien de coché → absent,
        // le backend garde alors le défaut de la famille (UNET câblé / 1er modèle).
        z_model: selectedBase || undefined,
        // Réglages globaux (resolution_tier, negative/sampler/detail/rebalance/…),
        // déjà gatés PAR FAMILLE côté backend — champs vides absents = défauts gardés.
        ...genSettings,
      };
      if (prompt.trim()) body.prompt = prompt.trim();
      const dResp = await postJson('/api/studio/run', body);
      toast.success(t('studio.run.queued', {
        count: dResp.created,
        seed: dResp.seed,
        multiplier: dResp.count > 1 ? ` ×${dResp.count}` : '',
      }));
      setRunId(dResp.run_id);
      setSeed(rollSeed());
      setPreflight(null);
      setArchMismatch(null);
    } catch (e) {
      // apiFetch throws on non-2xx; a 409 carries the itemized manques on e.body (P0-a)
      // or a wrong-arch checkpoint on e.body.studio_arch_mismatch.
      setPreflight(e?.body?.studio_missing || null);
      setArchMismatch(e?.body?.studio_arch_mismatch || null);
      toast.error(e.message || t('studio.run.launchFailed'));
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4 items-start">
      <aside className="flex flex-col gap-3 lg:sticky lg:top-16 lg:max-h-[calc(100vh-7rem)] lg:overflow-auto">
        {/* Picker de base — toutes familles. Krea : l'endpoint ne renvoie une liste
            (Official + alternatives) que si des UNET Krea locaux existent ; sinon
            vide → le sélecteur reste caché (défaut câblé du workflow). */}
        {baseModels.length > 0 && (
          <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface p-3">
            <span className="text-content-muted text-[0.625rem] uppercase">
              {t('studio.comparison.baseModel', { family: FAMILY_LABELS[runType] || 'Z-Image' })}
            </span>
            <select
              value={selectedBase}
              onChange={(e) => setSelectedBase(e.target.value)}
              aria-label={t('studio.comparison.baseModelLabel')}
              className="rounded border border-border bg-app/60 px-1.5 py-1 text-content text-sm"
            >
              {baseModels.map((m) => (
                <option key={m.filename} value={m.filename}>{m.label}</option>
              ))}
            </select>
          </div>
        )}
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
        <LoraRankingPanel ranking={data?.lora_ranking} />
      </aside>

      <main id="st-results" className="flex flex-col gap-3 min-w-0 scroll-mt-16">
        <StudioPreflightBanner missing={preflight} archMismatch={archMismatch}
          onDismiss={() => { setPreflight(null); setArchMismatch(null); }} />
        {data?.pending > 0 && (
          <div className="flex items-center gap-2 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-3 py-2" role="status">
            <span className="inline-block w-4 h-4 border-2 border-indigo-400/40 border-t-indigo-400 rounded-full animate-spin" aria-hidden />
            <span className="text-content text-sm">
              {t('studio.run.progress', {
                generating: data.generating ?? data.running ?? 0,
                queued: data.queued ?? data.pending,
              })}
            </span>
            <button type="button" onClick={run.cancel}
              className="ml-auto px-2.5 py-1 rounded-lg bg-red-600/80 text-white text-xs font-semibold">
              {t('studio.run.stopResumable')}
            </button>
          </div>
        )}
        {!data?.pending && data?.resumable > 0 && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2" role="status">
            <span aria-hidden>⏸</span>
            <span className="text-content text-sm">{t('studio.run.stopped', { count: data.resumable })}</span>
            <button type="button" disabled={!!data?.gpu_busy} onClick={run.resume}
              className="ml-auto px-2.5 py-1 rounded-lg bg-gradient-primary text-white text-xs font-semibold disabled:opacity-40">
              ▶ {t('studio.run.resume')}
            </button>
          </div>
        )}

        {!runId ? (
          <p className="text-content-subtle text-sm rounded-lg border border-border bg-surface px-3 py-6 text-center">
            {t('studio.comparison.empty', { count: selection.length })}
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
              onStartReVote={() => vote.startVoting(greens, t('studio.results.reconfirmTitle'))}
              displayedCount={cells.length}
              showResults={showResults}
              onToggleResults={() => setShowResults((v) => !v)}
            />
            {showResults && (
              <LoraComparisonGrid loras={loras} cells={cells} onRate={run.rate} onOpen={setLbImg} />
            )}
          </div>
        )}
      </main>

      <QuickVoteModal vote={vote} datasetId={vote.current?.dataset_id} fmt={fmt} />
      {lbImg && (
        <ResultLightbox img={lbImg} items={navImages} datasetId={lbImg.dataset_id}
          onRate={rateLightbox} onNavigate={setLbImg} onClose={() => setLbImg(null)} fmt={fmt} />
      )}

      {/* Barre de commande fixe : Run toujours visible + raccourcis de sections. */}
      <StudioActionBar
        shortcuts={[
          { id: 'st-loras', emoji: '🧬', label: t('studio.shortcuts.loras') },
          { id: 'st-setup', emoji: '📝', label: t('studio.shortcuts.promptSeed') },
          { id: 'st-format', emoji: '📐', label: t('studio.shortcuts.format') },
          ...(runType === 'krea' ? [
            { id: 'st-sampling', emoji: '🎛️', label: t('studio.shortcuts.sampling') },
            { id: 'st-engine', emoji: '⚙️', label: t('studio.shortcuts.engine') },
          ] : []),
          ...(runType === 'sdxl' ? [{ id: 'st-detail', emoji: '✨', label: t('studio.shortcuts.detail') }] : []),
          ...(runType === 'zimage' ? [{ id: 'st-negative', emoji: '🚫', label: t('studio.shortcuts.negative') }] : []),
          { id: 'st-results', emoji: '🖼️', label: t('studio.shortcuts.results') },
        ]}
        canRun={!!selection.length && !!strengths.length && !launching && !data?.gpu_busy}
        running={launching}
        onRun={launch}
      />
    </div>
  );
}
