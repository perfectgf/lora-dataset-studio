// react-frontend/src/components/dataset/studio/StudioRunSetup.jsx
/**
 * Standalone Studio run setup: strength sweep, free prompt, seed/reroll and images per
 * configuration. Show GPU cell count (LoRAs x strengths x count) BEFORE launch. Minimal local
 * state follows RunSetupPanel/useStudioForm but is not bound to one dataset because multiple LoRAs
 * are tested. Parent StudioShell owns LoRA selection and triggers POST.
 */
import { useCallback, useEffect, useState } from 'react';
import { Dices } from 'lucide-react';
import { STRENGTH_CHOICES } from './constants';
import { fmt } from '../../../utils/studioFormat';
import { postJson } from '../../../api/fetchClient';
import StrengthPicker from './StrengthPicker';
import RecentPrompts from './RecentPrompts';
import DescribeImageModal from './DescribeImageModal';
import DatasetCaptionControl from './DatasetCaptionControl';
import EnhancePromptButton from './EnhancePromptButton';
import CivitaiBrowserButton from './CivitaiBrowserButton';
import { MemberPreviewButton } from '../../patreon/MemberPreviews.jsx';
import { cellCount } from './loraStack';
import { launchText as batchLaunchText } from './promptBatch';
import { heavyRunConfirm, heavyRunNotice, runCost } from './runCost';

export default function StudioRunSetup({
  selectionCount, strengths, onToggleStrength,
  prompt, onPrompt, seed, onReroll, count, onCount,
  onLaunch, launching, gpuBusy, batchMult = 1, combine = false, combineBlocked = null,
  configCount = 1,
  // Caller renders CFG/steps/second-pass axes. axisTotal is their grid multiplier and MUST affect
  // cost; otherwise the panel could promise 6 cells while queueing 18.
  axisSlot = null, axisTotal = 1,
  // MEASURED machine pace from the backend median. null uses an explicitly approximate fallback,
  // never invented precision.
  secondsPerImage = null,
  // Trigger word optionally prefixes each LoRA's trigger. Pass through only; caller owns state and
  // persistence.
  injectTrigger = true, onInjectTrigger = null,
  // PROMPT BATCH was missing here: create_comparison_run already accepted prompts, but POST
  // /api/studio/run did not forward it and this panel had no control. History, scenes and Civitai
  // could not batch here despite support on both other launch surfaces. Pass through to
  // ComparisonStudio, which owns state and request construction.
  batchPrompts = null, onToggleBatchPrompt = null, onClearBatchPrompts = null,
  civitaiPicks = null, onToggleCivitaiPick = null, onClearCivitaiPicks = null,
  // Use the MERGED, deduplicated launch batch for cost and button text, not the raw total of
  // checked boxes.
  pickedPrompts = null,
}) {
  // batchMult is 1 plus the count of with/without batch LoRAs, matching backend cell
  // multiplication. Combine mode has no strengths axis: each LoRA has its own weight and the stack
  // is ONE configuration, or configCount when sweeping checked weights. Each batch prompt adds
  // another pass over that grid; show its cost BEFORE clicking.
  const picked = Array.isArray(pickedPrompts) ? pickedPrompts : [];
  const cells = cellCount({
    selectionCount, strengthCount: strengths.length, count, batchMult, combine, configCount,
    axisTotal, promptCount: Math.max(1, picked.length),
  });
  const canLaunch = cells > 0 && !launching && !gpuBusy && !combineBlocked;
  // Same long-run rule as the other panel: estimate, ask ONCE, never forbid.
  const cost = runCost(cells, secondsPerImage);
  const launchGuarded = () => {
    if (cost.heavy && !window.confirm(heavyRunConfirm(cost))) return;
    onLaunch();
  };

  // GLOBAL recent test prompts across datasets, previously absent in comparison. Reload after
  // launch saves a prompt and after deletion.
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
    // Deletion removes the prompt AND its images. Also remove it from the batch or launch would
    // use an invisible row. The other surface guards with visibleBatch because history comes from
    // its parent; local history must remove it here at its deletion point.
    if (typeof onToggleBatchPrompt === 'function'
      && (batchPrompts || []).includes(p)) onToggleBatchPrompt(p);
    loadRecent();
  }, [loadRecent, onToggleBatchPrompt, batchPrompts]);

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

      {/*
       * CFG, steps and second pass remain in Blend even though strengths disappear into per-LoRA
       * weights. Steps are a RENDER setting, not a LoRA setting; hiding them here was the reported
       * bug.
       */}
      {axisSlot}

      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="studio-run-prompt" className="text-content-muted text-[0.625rem] uppercase">
              Prompt (optional)
            </label>
            {onInjectTrigger && (
              <label className="flex items-center gap-1 text-content-subtle text-[0.625rem] cursor-pointer"
                title="Prefix each LoRA's trigger word to this prompt when generating. Uncheck to send the prompt exactly as written — useful when a render keeps typing the trigger back (speech bubbles, signs) or for pure style/scene tests.">
                <input type="checkbox" checked={injectTrigger}
                  onChange={(e) => onInjectTrigger(e.target.checked)} />
                Trigger word
              </label>
            )}
          </div>
          <div className="flex max-w-full flex-wrap items-center justify-end gap-1">
            <DatasetCaptionControl onCaption={applyCaption} />
            <EnhancePromptButton prompt={prompt} onResult={onPrompt} />
          <button type="button" onClick={() => setDescribeOpen(true)}
            title="Describe an image into a test prompt (vision model)"
            className="px-2 py-0.5 rounded border border-border bg-surface text-content-subtle text-[0.625rem] hover:text-content">
            🔎 Describe
          </button>
          <CivitaiBrowserButton prompt={prompt} onPrompt={onPrompt}
            picks={civitaiPicks} onTogglePick={onToggleCivitaiPick} />
          <MemberPreviewButton placement="studio.prompt" />
          </div>
        </div>
        <textarea id="studio-run-prompt" value={prompt} onChange={(e) => onPrompt(e.target.value)} rows={5}
          placeholder="Leave empty for the LoRA's default prompt…"
          className="rounded-lg border border-border bg-app/60 px-2.5 py-1.5 text-content text-sm resize-y min-h-[7rem]" />
      </div>
      <DescribeImageModal open={describeOpen} onClose={() => setDescribeOpen(false)}
        onResult={applyDescription} />

      {Array.isArray(civitaiPicks) && civitaiPicks.length > 0 && (
        <p className="m-0 flex flex-wrap items-center gap-1.5 text-content-subtle text-[0.5625rem]">
          <span className="rounded bg-purple-500/20 px-1.5 py-0.5 font-semibold text-purple-200 tabular-nums">
            🌐 {civitaiPicks.length} Civitai prompt{civitaiPicks.length === 1 ? '' : 's'} in the batch
          </span>
          {onClearCivitaiPicks && (
            <button type="button" onClick={onClearCivitaiPicks}
              className="inline-flex min-h-10 items-center px-1 underline decoration-dotted hover:text-content lg:min-h-0 lg:px-0">
              Clear
            </button>
          )}
        </p>
      )}

      {recentPrompts.length > 0 && (
        <RecentPrompts items={recentPrompts} datasetId={null} selectedPrompt={prompt}
          onPick={onPrompt} onDelete={deleteRecent}
          batch={batchPrompts} onToggleBatch={onToggleBatchPrompt}
          onClearBatch={onClearBatchPrompts} />
      )}

      <div className="flex items-center gap-2 flex-wrap">
        <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
          <span className="uppercase">Seed</span>
          <span className="tabular-nums text-content px-2 py-0.5 rounded border border-border bg-app/60">{seed}</span>
          <button type="button" onClick={onReroll} aria-label="New random seed"
            title="New random seed"
            className="px-2 py-0.5 rounded border border-border bg-surface text-content hover:bg-surface-raised"><Dices aria-hidden="true" className="h-3.5 w-3.5" /></button>
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
            : `GPU cost: checked LoRAs × strengths × images per config${batchMult > 1 ? ` × ${batchMult} (⚖ batch axis: without + with each checked LoRA)` : ''}${axisTotal > 1 ? ` × ${axisTotal} (🎛 CFG / steps axes)` : ''}${picked.length > 1 ? ` × ${picked.length} (📝 prompt batch: one image set per ticked prompt)` : ''}`}>
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
          {axisTotal > 1 && <span className="text-purple-300"> × {axisTotal} 🎛</span>}
          {picked.length > 1 && (
            <span className="text-purple-200" data-testid="studio-prompt-axis"> × {picked.length} 📝</span>
          )} ={' '}
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
          className="ml-auto px-4 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
          {launching ? '…' : (batchLaunchText('🚀 Run the test', picked) ?? '🚀 Run the test')}
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
