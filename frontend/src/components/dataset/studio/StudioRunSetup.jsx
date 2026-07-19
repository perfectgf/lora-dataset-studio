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
import { useI18n } from '../../../i18n/I18nContext';

export default function StudioRunSetup({
  selectionCount, strengths, onToggleStrength,
  prompt, onPrompt, seed, onReroll, count, onCount,
  onLaunch, launching, gpuBusy, batchMult = 1,
}) {
  const { t } = useI18n();
  // batchMult = 1 + nb de LoRA cochés « ⚖ batch » (axe sans/avec) — le backend
  // multiplie les cellules d'autant, le compteur de coût doit suivre.
  const cells = selectionCount * strengths.length * count * batchMult;
  const canLaunch = selectionCount > 0 && strengths.length > 0 && cells > 0 && !launching && !gpuBusy;

  // Prompts de test récents GLOBAUX (tous datasets — la comparaison n'en avait
  // aucun avant). Rechargé après un lancement (nouveau prompt mémorisé) et après
  // une suppression.
  const [recentPrompts, setRecentPrompts] = useState([]);
  const [describeOpen, setDescribeOpen] = useState(false);
  const applyDescription = (text) => {
    if (prompt && prompt.trim()
      && !window.confirm(t('studio.setup.replaceDescription'))) return;
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

      <StrengthPicker choices={STRENGTH_CHOICES} selected={strengths} onToggle={onToggleStrength} fmt={fmt} />

      <label className="flex flex-col gap-1">
        <span className="flex items-center justify-between gap-2">
          <span className="text-content-muted text-[0.625rem] uppercase">{t('studio.setup.promptOptional')}</span>
          <button type="button" onClick={() => setDescribeOpen(true)}
            title={t('studio.setup.describeTitle')}
            className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
            🔎 {t('studio.prompt.describe')}
          </button>
        </span>
        <textarea value={prompt} onChange={(e) => onPrompt(e.target.value)} rows={5}
          placeholder={t('studio.setup.promptPlaceholder')}
          className="rounded-lg border border-border bg-app/60 px-2.5 py-1.5 text-content text-sm resize-y min-h-[7rem]" />
      </label>
      <DescribeImageModal open={describeOpen} onClose={() => setDescribeOpen(false)}
        onResult={applyDescription} />

      {recentPrompts.length > 0 && (
        <RecentPrompts items={recentPrompts} datasetId={null} selectedPrompt={prompt}
          onPick={onPrompt} onDelete={deleteRecent} />
      )}

      <div className="flex items-center gap-2 flex-wrap">
        <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
          <span className="uppercase">{t('studio.seed.seed')}</span>
          <span className="tabular-nums text-content px-2 py-0.5 rounded border border-border bg-app/60">{seed}</span>
          <button type="button" onClick={onReroll} aria-label={t('studio.setup.newSeed')}
            title={t('studio.setup.newSeed')}
            className="px-2 py-0.5 rounded border border-border bg-surface text-content hover:bg-surface-raised">🎲</button>
        </label>

        <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
          <span className="uppercase">{t('studio.setup.imagesPerConfig')}</span>
          <select value={count} onChange={(e) => onCount(Number(e.target.value))}
            aria-label={t('studio.setup.imagesPerConfigLabel')}
            className="rounded border border-border bg-app/60 px-1.5 py-0.5 text-content">
            {[1, 2, 3, 4].map((n) => <option key={n} value={n}>×{n}</option>)}
          </select>
        </label>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-content-subtle text-[0.6875rem]"
          title={t('studio.setup.gpuCost', {
            batch: batchMult > 1 ? t('studio.setup.batchCost', { count: batchMult }) : '',
          })}>
          {selectionCount} LoRA × {strengths.length} {t('studio.setup.strengthUnit')} × {count}
          {batchMult > 1 && <span className="text-amber-300"> × {batchMult} ⚖</span>} ={' '}
          <span className={`tabular-nums font-semibold ${cells > 0 ? 'text-content' : 'text-content-subtle'}`}>{cells}</span>{' '}
          {t('studio.setup.cellsToGenerate')}
        </span>
        <button type="button" onClick={onLaunch} disabled={!canLaunch}
          aria-label={t('studio.actions.runTest')}
          className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {launching ? '…' : `🚀 ${t('studio.actions.runTest')}`}
        </button>
      </div>
      {selectionCount === 0 && (
        <p className="m-0 text-amber-300 text-[0.6875rem]">{t('studio.setup.selectOne')}</p>
      )}
    </div>
  );
}
