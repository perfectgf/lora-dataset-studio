/** ▶ Continue training — flexible resume for a finished run (local or cloud).
 * Replaces the old fixed « +1000 » confirm/prompt: pick how many more steps, WHICH
 * checkpoint to resume from (default = latest, but an earlier, less-cooked epoch is
 * the whole point — « step 750 beat the over-cooked 1000 »), and optionally adjust
 * the handful of settings a resume can safely change (cadence + preview prompts).
 *
 * Purely presentational and props-driven so the local panel and the cloud Runs hub
 * share one dialog. onResolve(payload | null): payload =
 * { extraSteps, fromStep, overrides } (fromStep null = resume from the latest,
 * in place), or null on cancel. */
import { useEffect, useMemo, useState } from 'react';
import { HelpBadge } from '../../help/HelpMode';

const SAVE_CHOICES = [250, 500, 1000];
const SAMPLE_EVERY_CHOICES = [100, 250, 500, 1000];
// Mirrors the backend's _TIMESTEP_TYPE_CHOICES. '' = keep the run's weighting.
const TIMESTEP_CHOICES = ['sigmoid', 'linear', 'weighted', 'shift'];

export default function ContinueDialog({
  context,                 // short run identity, e.g. "Lola — Z-Image · Turbo"
  where = 'local',         // 'local' | 'cloud' — only tweaks the seeding note wording
  checkpoints = [],        // [{ step, final?, best? }] — the run's saves
  bestStep = null,         // optional « Find best epoch » recommendation to flag
  settings = {},           // inherited effective settings shown as the starting point
  defaultExtra = 1000,
  busy = false,
  onResolve,
}) {
  // Distinct steps, ascending; the newest is the default resume point.
  const steps = useMemo(
    () => [...new Set(checkpoints.map((c) => c.step))].filter((s) => s > 0).sort((a, b) => a - b),
    [checkpoints]);
  const latest = steps.length ? steps[steps.length - 1] : 0;

  const [fromStep, setFromStep] = useState(latest);
  const [extra, setExtra] = useState(String(defaultExtra));
  const [showSettings, setShowSettings] = useState(false);

  const inheritedSave = SAVE_CHOICES.includes(settings.save_every) ? settings.save_every : 250;
  const inheritedSampleEvery =
    SAMPLE_EVERY_CHOICES.includes(settings.sample_every) ? settings.sample_every : 250;
  const inheritedPrompts = Array.isArray(settings.sample_prompts)
    ? settings.sample_prompts.join('\n') : '';
  const inheritedTimestep = TIMESTEP_CHOICES.includes(settings.timestep_type)
    ? settings.timestep_type : '';
  const [saveEvery, setSaveEvery] = useState(inheritedSave);
  const [sampleEvery, setSampleEvery] = useState(inheritedSampleEvery);
  const [prompts, setPrompts] = useState('');   // blank = keep the run's prompts
  const [timestep, setTimestep] = useState(inheritedTimestep); // '' = keep current

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onResolve(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onResolve]);

  const extraNum = Math.max(100, parseInt(extra, 10) || 0);
  const target = fromStep + extraNum;
  const isEarlier = fromStep < latest;

  const submit = () => {
    const overrides = {};
    if (saveEvery !== inheritedSave) overrides.save_every = Number(saveEvery);
    if (sampleEvery !== inheritedSampleEvery) overrides.sample_every = Number(sampleEvery);
    if (prompts.trim() !== '') {
      overrides.sample_prompts = prompts.split('\n').map((s) => s.trim()).filter(Boolean);
    }
    if (timestep !== '' && timestep !== inheritedTimestep) overrides.timestep_type = timestep;
    onResolve({
      extraSteps: extraNum,
      // null when the newest checkpoint is chosen → the historical in-place resume.
      fromStep: isEarlier ? fromStep : null,
      overrides: Object.keys(overrides).length ? overrides : undefined,
    });
  };

  const stepLabel = (s) => {
    const tags = [];
    if (s === latest) tags.push('latest');
    if (s === bestStep) tags.push('best');
    return `step ${s}${tags.length ? ` — ${tags.join(', ')}` : ''}`;
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Continue training"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) onResolve(null); }}>
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-indigo-400/40 bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-indigo-300 font-semibold"><span aria-hidden>▶</span> Continue training</span>
          <HelpBadge topic="continue-training" />
          {context && <span className="text-content-subtle text-[0.75rem] truncate">{context}</span>}
          <button type="button" onClick={() => onResolve(null)}
            className="ml-auto text-content-subtle hover:text-content" aria-label="Cancel">✕</button>
        </div>

        {/* Resume FROM which checkpoint. */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Resume from</span>
            <select value={String(fromStep)} onChange={(e) => setFromStep(Number(e.target.value))}
              aria-label="Checkpoint to resume from"
              className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
              {steps.length === 0 && <option value="0">no checkpoint</option>}
              {steps.slice().reverse().map((s) => (
                <option key={s} value={String(s)}>{stepLabel(s)}</option>
              ))}
            </select>
          </div>
          <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
            <b className="text-content-muted font-medium">Why:</b> a later epoch can be over-cooked — resume from the
            one that held up best (the 🏆 <b>best</b> tag, when scored).
          </span>
        </div>

        {/* How many more steps. */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-content text-[0.75rem] w-28 shrink-0">Extra steps</span>
            <input type="number" min="100" step="100" value={extra}
              onChange={(e) => setExtra(e.target.value)}
              aria-label="Additional steps to train"
              className="w-28 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] tabular-nums" />
            <span className="text-content-muted text-[0.6875rem] tabular-nums">
              → target step {target}
            </span>
          </div>
          {isEarlier && (
            <span className="text-amber-300/90 text-[0.6875rem] leading-relaxed">
              Restarts from step {fromStep}: the run’s later checkpoints are set aside (kept
              {where === 'cloud' ? ' in this run’s staging' : ' on disk'}, recoverable), and the
              continuation writes its own saves.
            </span>
          )}
        </div>

        {/* Optional, folded: the settings a resume can safely change. */}
        <div className="flex flex-col gap-1">
          <button type="button" onClick={() => setShowSettings((v) => !v)}
            className="self-start text-indigo-300 hover:text-indigo-200 text-[0.75rem] font-medium">
            {showSettings ? '▾' : '▸'} Adjust settings (optional)
          </button>
          {showSettings && (
            <div className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface-raised p-2.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Save checkpoint</span>
                <select value={String(saveEvery)} onChange={(e) => setSaveEvery(Number(e.target.value))}
                  aria-label="Checkpoint frequency"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  {SAVE_CHOICES.map((n) => <option key={n} value={String(n)}>every {n} steps</option>)}
                </select>
                <span className="text-content-subtle text-[0.625rem]">how often a checkpoint is written</span>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Preview every</span>
                <select value={String(sampleEvery)} onChange={(e) => setSampleEvery(Number(e.target.value))}
                  aria-label="Preview sample frequency"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  {SAMPLE_EVERY_CHOICES.map((n) => <option key={n} value={String(n)}>every {n} steps</option>)}
                </select>
                <span className="text-content-subtle text-[0.625rem]">preview images cadence</span>
              </div>
              <label className="flex flex-col gap-1">
                <span className="text-content text-[0.75rem]">Preview prompts</span>
                <textarea value={prompts} onChange={(e) => setPrompts(e.target.value)} rows={3}
                  placeholder={inheritedPrompts || 'one prompt per line — blank keeps the run’s prompts'}
                  aria-label="Preview sample prompts, one per line"
                  className="px-2 py-1.5 rounded-lg border border-border bg-surface text-content text-[0.6875rem] font-mono leading-relaxed resize-y placeholder:text-content-subtle" />
                <span className="text-content-subtle text-[0.625rem]">test images only — never affects the weights</span>
              </label>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Timestep weighting</span>
                <select value={timestep} onChange={(e) => setTimestep(e.target.value)}
                  aria-label="Timestep weighting for the continuation"
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]">
                  <option value="">keep current</option>
                  {TIMESTEP_CHOICES.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
                <span className="text-content-subtle text-[0.625rem]">advanced — SDXL ignores it</span>
              </div>
              <span className="text-content-subtle text-[0.625rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why timestep:</b> a known two-phase recipe trains balanced
                first, then continues with a different noise-level emphasis to polish fine texture — changing it here is
                a deliberate recipe change for the extra steps, applied via this dataset&apos;s settings.
              </span>
              <span className="text-content-subtle text-[0.625rem] leading-relaxed">
                Only cadence, preview prompts and the timestep weighting can change on a resume — rank, base, optimizer
                and the like are locked to the checkpoint being continued.
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 pt-1">
          <button type="button" onClick={() => onResolve(null)}
            className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm">Cancel</button>
          <button type="button" onClick={submit} disabled={busy || latest === 0}
            className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {busy ? 'Starting…' : `Continue → ${target}`}
          </button>
        </div>
      </div>
    </div>
  );
}
