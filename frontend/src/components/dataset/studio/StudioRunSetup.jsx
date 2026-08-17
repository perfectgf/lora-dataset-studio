// react-frontend/src/components/dataset/studio/StudioRunSetup.jsx
/**
 * Panneau de réglage d'un run du Studio autonome : strengths à balayer, prompt
 * libre, seed (+ relance aléatoire), nombre d'images par config. Affiche le COÛT
 * GPU (nombre de cellules = LoRA × strengths × count) AVANT lancement, puis le
 * bouton « 🚀 Lancer le test ».
 *
 * État local minimal (inspiré de RunSetupPanel/useStudioForm mais autonome — pas
 * lié à un dataset précis puisqu'on teste plusieurs LoRA). Le parent (StudioShell)
 * possède la sélection de LoRA et déclenche le POST.
 */
import { useCallback, useEffect, useState } from 'react';
import { STRENGTH_CHOICES } from './constants';
import { fmt } from '../../../utils/studioFormat';
import { postJson } from '../../../api/fetchClient';
import StrengthPicker from './StrengthPicker';
import RecentPrompts from './RecentPrompts';
import DescribeImageModal from './DescribeImageModal';
import DatasetCaptionControl from './DatasetCaptionControl';
import EnhancePromptButton from './EnhancePromptButton';
import { cellCount } from './loraStack';
import { heavyRunConfirm, heavyRunNotice, runCost } from './runCost';

export default function StudioRunSetup({
  selectionCount, strengths, onToggleStrength,
  prompt, onPrompt, seed, onReroll, count, onCount,
  onLaunch, launching, gpuBusy, batchMult = 1, combine = false, combineBlocked = null,
  configCount = 1,
  // 🎛 Axes de rendu (CFG / steps / 2e passe) rendus par l'appelant. `axisTotal`
  // est le facteur qu'ils ajoutent à la grille : le compteur de coût DOIT le
  // porter, sinon le panneau annonce 6 cellules et la file en reçoit 18.
  axisSlot = null, axisTotal = 1,
  // ⏱ Rythme MESURÉ de la machine (médiane du backend). null → repli affiché
  // avec un « ~ », jamais un chiffre précis inventé.
  secondsPerImage = null,
}) {
  // batchMult = 1 + nb de LoRA cochés « ⚖ batch » (axe sans/avec) — le backend
  // multiplie les cellules d'autant, le compteur de coût doit suivre.
  // En mode « pile » (combine) l'axe strengths disparaît : chaque LoRA porte son
  // propre poids. La pile vaut UNE configuration — ou `configCount` quand des
  // cases de poids sont cochées et que le lancement balaye leurs combinaisons.
  const cells = cellCount({
    selectionCount, strengthCount: strengths.length, count, batchMult, combine, configCount,
    axisTotal,
  });
  const canLaunch = cells > 0 && !launching && !gpuBusy && !combineBlocked;
  // ⏱ Même règle que l'autre panneau : on chiffre, on demande UNE fois, on
  // n'interdit jamais.
  const cost = runCost(cells, secondsPerImage);
  const launchGuarded = () => {
    if (cost.heavy && !window.confirm(heavyRunConfirm(cost))) return;
    onLaunch();
  };

  // Prompts de test récents GLOBAUX (tous datasets — la comparaison n'en avait
  // aucun avant). Rechargé après un lancement (nouveau prompt mémorisé) et après
  // une suppression.
  const [recentPrompts, setRecentPrompts] = useState([]);
  const [describeOpen, setDescribeOpen] = useState(false);
  const applyDescription = (text) => {
    if (prompt && prompt.trim()
      && !window.confirm('Replace the current prompt with the described one?')) return;
    onPrompt(text);
  };
  const applyCaption = (text) => {
    if (prompt && prompt.trim()
      && !window.confirm('Replace the current prompt with a random caption drawn from your locked source?')) return;
    onPrompt(text);
  };
  const loadRecent = useCallback(() => {
    fetch('/api/studio/recent-prompts', { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.ok) setRecentPrompts(d.prompts || []); })
      .catch(() => { /* menu facultatif — silencieux */ });
  }, []);
  useEffect(() => { loadRecent(); }, [loadRecent, launching]);
  const deleteRecent = useCallback(async (p) => {
    await postJson('/api/studio/recent-prompts/delete', { prompt: p }).catch(() => {});
    loadRecent();
  }, [loadRecent]);

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-3">
      {gpuBusy && (
        <p className="m-0 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-red-300 text-sm" role="status">
          {gpuBusy}
        </p>
      )}

      {!combine && (
        <StrengthPicker choices={STRENGTH_CHOICES} selected={strengths} onToggle={onToggleStrength} fmt={fmt} />
      )}

      {/* 🎛 CFG / steps / 2e passe. En 🧬 Blend l'axe des strengths disparaît (chaque
          LoRA porte son poids) mais celui-ci reste : les steps sont un réglage de
          RENDU, pas de LoRA — les cacher là était le bug signalé. */}
      {axisSlot}

      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <label htmlFor="studio-run-prompt" className="text-content-muted text-[0.625rem] uppercase">
            Prompt (optional)
          </label>
          <div className="flex max-w-full flex-wrap items-center justify-end gap-1">
            <DatasetCaptionControl onCaption={applyCaption} />
            <EnhancePromptButton prompt={prompt} onResult={onPrompt} />
          <button type="button" onClick={() => setDescribeOpen(true)}
            title="Describe an image into a test prompt (vision model)"
            className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
            🔎 Describe
          </button>
          </div>
        </div>
        <textarea id="studio-run-prompt" value={prompt} onChange={(e) => onPrompt(e.target.value)} rows={5}
          placeholder="Leave empty for the LoRA's default prompt…"
          className="rounded-lg border border-border bg-app/60 px-2.5 py-1.5 text-content text-sm resize-y min-h-[7rem]" />
      </div>
      <DescribeImageModal open={describeOpen} onClose={() => setDescribeOpen(false)}
        onResult={applyDescription} />

      {recentPrompts.length > 0 && (
        <RecentPrompts items={recentPrompts} datasetId={null} selectedPrompt={prompt}
          onPick={onPrompt} onDelete={deleteRecent} />
      )}

      <div className="flex items-center gap-2 flex-wrap">
        <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
          <span className="uppercase">Seed</span>
          <span className="tabular-nums text-content px-2 py-0.5 rounded border border-border bg-app/60">{seed}</span>
          <button type="button" onClick={onReroll} aria-label="New random seed"
            title="New random seed"
            className="px-2 py-0.5 rounded border border-border bg-surface text-content hover:bg-surface-raised">🎲</button>
        </label>

        <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
          <span className="uppercase">Images / config</span>
          <select value={count} onChange={(e) => onCount(Number(e.target.value))}
            aria-label="Number of images per configuration"
            className="rounded border border-border bg-app/60 px-1.5 py-0.5 text-content">
            {[1, 2, 3, 4].map((n) => <option key={n} value={n}>×{n}</option>)}
          </select>
        </label>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-content-subtle text-[0.6875rem]"
          title={combine
            ? `GPU cost: ${configCount} weight combination(s) of ${selectionCount} LoRAs × images per config`
            : `GPU cost: checked LoRAs × strengths × images per config${batchMult > 1 ? ` × ${batchMult} (⚖ batch axis: without + with each checked LoRA)` : ''}${axisTotal > 1 ? ` × ${axisTotal} (🎛 CFG / steps axes)` : ''}`}>
          {combine
            ? (
              <>
                {configCount > 1
                  ? <>{configCount} weight combos of {selectionCount} LoRA</>
                  : <>1 stack of {selectionCount} LoRA</>} × {count}
              </>
            )
            : <>{selectionCount} LoRA × {strengths.length} strength × {count}</>}
          {batchMult > 1 && <span className="text-amber-300"> × {batchMult} ⚖</span>}
          {axisTotal > 1 && <span className="text-purple-300"> × {axisTotal} 🎛</span>} ={' '}
          <span className={`tabular-nums font-semibold ${cells > 0 ? 'text-content' : 'text-content-subtle'}`}>{cells}</span>{' '}
          cell(s) to generate
          {cells > 0 && (
            <span className="text-content-subtle">
              {' '}· {cost.measured ? '' : '~'}{cost.label}
            </span>
          )}
        </span>
        <button type="button" onClick={launchGuarded} disabled={!canLaunch}
          aria-label="Run the test"
          className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {launching ? '…' : '🚀 Run the test'}
        </button>
      </div>
      {cost.heavy && (
        <p data-testid="heavy-run-notice"
          className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-2.5 py-1.5 text-[0.6875rem] text-amber-200"
          role="status">
          <span aria-hidden>⏱</span> {heavyRunNotice(cost)}
        </p>
      )}
      {selectionCount === 0 && (
        <p className="m-0 text-amber-300 text-[0.6875rem]">Check at least one LoRA above.</p>
      )}
      {combineBlocked && (
        <p className="m-0 text-amber-300 text-[0.6875rem]" role="status">{combineBlocked}</p>
      )}
    </div>
  );
}
