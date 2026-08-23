/** ▶ Continue training — flexible resume for a finished run (local or cloud).
 * Replaces the old fixed « +1000 » confirm/prompt: pick how many more steps, WHICH
 * checkpoint to resume from (default = latest, but an earlier, less-cooked epoch is
 * the whole point — « step 750 beat the over-cooked 1000 »), and optionally adjust
 * the handful of settings a resume can safely change. Full-state continuation
 * keeps both save/sample cadence and the training trajectory locked.
 *
 * Purely presentational and props-driven so the local panel and the cloud Runs hub
 * share one dialog. onResolve(payload | null): payload =
 * { extraSteps, fromStep, overrides, lane, resumeMode, stateBundleId } (fromStep
 * null = resume from the latest), or null on cancel. overrides may
 * carry save_every / sample_every / sample_prompts / timestep_type / lr_factor
 * — the safe subset a resume can change;
 * lr_factor (0.5/0.1) scales the run's current LR and is omitted for an adaptive
 * (Prodigy) run. `settings` supplies optimizer + learning_rate so the LR hint and
 * the Prodigy-disabled state are truthful.
 *
 * `lane` ('local' | 'cloud') says WHERE the continuation runs. A mount that passes
 * `lanes` lets the user choose it — a checkpoint is a file, so a run trained here
 * can be finished on a rented GPU and vice-versa; a mount that doesn't gets its
 * own `where` back and can ignore the field. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { HelpBadge } from '../../help/HelpMode';
import {
  defaultResumeMode,
  fullStateUnavailableReason,
  preferredCheckpointForStep,
} from '../../utils/trainingResumeState.js';
import { initialResumeStep, resolveInitialLane, submitBlockedReason } from './lineageContinue.js';
import {
  TRANSPORTS, TRANSPORT_LABELS, closedRoads, initialTransport, rateNote,
  transportBlockedReason, transportOption, transportSummary,
} from './podTransportChoice.js';

const SAVE_CHOICES = [250, 500, 1000];
const SAMPLE_EVERY_CHOICES = [100, 250, 500, 1000];
// Mirrors the backend's _TIMESTEP_TYPE_CHOICES. '' = keep the run's weighting.
const TIMESTEP_CHOICES = ['sigmoid', 'linear', 'weighted', 'shift'];
// LR factors applied to the run's CURRENT rate (backend _RESUME_LR_FACTORS).
// 1 = keep current (sent as no override). Mirrors the timestep low-noise recipe.
const LR_FACTOR_CHOICES = [
  { value: 1, label: 'keep current' },
  { value: 0.5, label: 'half (polish)' },
  { value: 0.1, label: 'tenth (gentle finish)' },
];
const DEFAULT_LR = 1e-4;   // backend _DEFAULT_LR for every non-adaptive optimizer

// Compact scientific form for a LR (5e-5, 1e-5, 2.5e-5) — trims a bare ".0".
const fmtLR = (lr) => Number(lr).toExponential(1).replace('.0e', 'e');

export default function ContinueDialog({
  context,                 // short run identity, e.g. "Lola — Z-Image · Turbo"
  where = 'local',         // 'local' | 'cloud' — the lane, and the seeding note wording
  checkpoints = [],        // [{ step, final?, best? }] — the run's saves
  bestStep = null,         // optional « Find best epoch » recommendation to flag
  initialFromStep = null,  // pre-select a specific checkpoint (◉ Graph "continue from here")
  // OPT-IN lane picker: { local: {available, reason}, cloud: {available, reason} }.
  // Absent (the Runs hub) → no picker at all and the dialog behaves exactly as
  // before. Present (the dataset panel) → the user chooses where the continuation
  // RUNS, independently of where the source run trained; `where` seeds the default.
  lanes = null,
  // OPT-IN transport picker, for a FULL MODEL only: the backend's resume plan
  // ({ default_transport, options: [{transport, available, reason, bytes,
  // seconds, gpu_cost, rate_* }] }). Absent → no picker and the dialog behaves
  // exactly as before. A ~26 GB checkpoint can reach the pod two ways, and the
  // difference between them is hours of a GPU that is rented and idle the whole
  // time — a choice nobody should make without seeing the price.
  transportPlan = null,
  settings = {},           // inherited effective settings shown as the starting point
  defaultExtra = 1000,
  busy = false,
  // A refusal from the LAST attempt, rendered inside the dialog. The hosts keep
  // the dialog open on refusal now: the lane, the resume checkpoint, the extra
  // steps and the five folded settings are exactly what the user has to change
  // to get past it, and they used to be wiped before the message arrived. null
  // while nothing has been refused.
  error = null,
  onResolve,
}) {
  // Distinct steps, ascending; the newest is the default resume point.
  const steps = useMemo(
    () => [...new Set(checkpoints.map((c) => c.step))].filter((s) => s > 0).sort((a, b) => a - b),
    [checkpoints]);
  const latest = steps.length ? steps[steps.length - 1] : 0;

  // Open on the requested checkpoint when it's a real save of this run, else the
  // newest (the historical default) — the rule is unit-tested in lineageContinue.js.
  const [fromStep, setFromStep] = useState(() => initialResumeStep(initialFromStep, steps));
  const [extra, setExtra] = useState(String(defaultExtra));
  const [showSettings, setShowSettings] = useState(false);
  // WHERE the continuation runs. Without `lanes` this is just `where` and nothing
  // is rendered — the historical single-lane dialog.
  const [lane, setLane] = useState(() => resolveInitialLane(where, lanes));
  const laneState = (id) => (lanes ? (lanes[id] || {}) : {});
  const laneBlocked = !!lanes && laneState(lane).available === false;
  // HOW the checkpoint gets to the pod. Seeded from the backend's default, but
  // never onto a road it would immediately have to disable.
  const [transport, setTransport] = useState(() => initialTransport(transportPlan));
  useEffect(() => { setTransport(initialTransport(transportPlan)); }, [transportPlan]);
  const transportReason = transportPlan
    ? transportBlockedReason(transportPlan, transport) : null;
  const blockedReason = submitBlockedReason({
    latest, laneBlocked, laneReason: laneState(lane).reason, lane })
    || transportReason;
  const selectedCheckpoint = useMemo(
    () => preferredCheckpointForStep(checkpoints, fromStep),
    [checkpoints, fromStep]);
  const fullStateReason = fullStateUnavailableReason(selectedCheckpoint, lane);
  const fullStateAvailable = !fullStateReason;
  const [resumeMode, setResumeMode] = useState(
    () => defaultResumeMode(selectedCheckpoint, lane));
  const trajectoryLocked = resumeMode === 'full_state';

  // A bundle belongs to one checkpoint and the current cloud image cannot use
  // one. Changing either selection therefore recomputes the safe default rather
  // than carrying a now-invalid full-state choice to another save/lane.
  useEffect(() => {
    setResumeMode(defaultResumeMode(selectedCheckpoint, lane));
  }, [fromStep, lane, selectedCheckpoint?.resume_state?.bundle_id, fullStateAvailable]);

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
  // Preview steps / CFG (#46). Blank = keep whatever the run already uses. NOT
  // gated by `trajectoryLocked`: they change how a preview image is rendered,
  // never the loop or the loader, so a full-state resume can honour them — and
  // a resume is exactly when you have SEEN the previews and know they are
  // unreadable.
  const [sampleSteps, setSampleSteps] = useState('');
  const [sampleGuidance, setSampleGuidance] = useState('');
  const [timestep, setTimestep] = useState(inheritedTimestep); // '' = keep current
  const [lrFactor, setLrFactor] = useState(1);  // 1 = keep the run's LR

  // Prodigy drives its own LR (lr=1) → scaling it is meaningless: the knob is shown
  // but disabled with the reason, never silently hidden. Everyone else has a real
  // base rate (an explicit learning_rate from a prior resume, else the 1e-4 default).
  const isAdaptiveLR = String(settings.optimizer || '').startsWith('prodigy');
  const currentLR = typeof settings.learning_rate === 'number' && settings.learning_rate > 0
    ? settings.learning_rate : DEFAULT_LR;

  /* ONE way out, and it is closed while a request is in flight.
     Two reasons, both paid for once: a dismissal during the post would leave the
     launch running with nothing on screen to report it, and the dataset panel
     raises its preflight report ON TOP of this dialog — a single Escape reached
     both window listeners and cancelled the two at once. */
  const dismiss = () => { if (!busy) onResolve(null); };

  /* With "Adjust settings" unfolded the dialog is taller than a 400-px screen
     and scrolls INSIDE itself, so a refusal can land outside the viewport —
     MEASURED on a 400-px capture with the section open. The message and the
     button that produced it are the last two blocks, so scrolling the card to
     its end shows both. (scrollIntoView({block:'nearest'}) does not: it moves
     the minimum, which left the message half-cut under the fold.) */
  const cardRef = useRef(null);
  useEffect(() => {
    const card = cardRef.current;
    if (error && card) card.scrollTop = card.scrollHeight;
  }, [error]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') dismiss(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onResolve]);  // eslint-disable-line react-hooks/exhaustive-deps

  const extraNum = Math.max(100, parseInt(extra, 10) || 0);
  const target = fromStep + extraNum;
  const isEarlier = fromStep < latest;

  const submit = () => {
    const overrides = {};
    if (!trajectoryLocked && saveEvery !== inheritedSave) {
      overrides.save_every = Number(saveEvery);
    }
    if (!trajectoryLocked && sampleEvery !== inheritedSampleEvery) {
      overrides.sample_every = Number(sampleEvery);
    }
    if (prompts.trim() !== '') {
      overrides.sample_prompts = prompts.split('\n').map((s) => s.trim()).filter(Boolean);
    }
    if (sampleSteps.trim() !== '' && Number.isFinite(Number(sampleSteps))) {
      overrides.sample_steps = Number(sampleSteps);
    }
    if (sampleGuidance.trim() !== '' && Number.isFinite(Number(sampleGuidance))) {
      overrides.sample_guidance = Number(sampleGuidance);
    }
    if (!trajectoryLocked && timestep !== '' && timestep !== inheritedTimestep) {
      overrides.timestep_type = timestep;
    }
    // A real LR reduction only — Prodigy (adaptive) never sends it, and 1 = keep.
    if (!trajectoryLocked && lrFactor !== 1 && !isAdaptiveLR) {
      overrides.lr_factor = lrFactor;
    }
    onResolve({
      extraSteps: extraNum,
      // null when the newest checkpoint is chosen → the historical in-place resume.
      fromStep: isEarlier ? fromStep : null,
      overrides: Object.keys(overrides).length ? overrides : undefined,
      // Where it runs. Always sent; a caller that offers no picker gets `where`
      // back and can ignore it.
      lane,
      // Which road the 26 GB takes back to the pod. Undefined when no picker was
      // offered, so the backend keeps its own default (the Hugging Face copy).
      transport: transportPlan ? transport : undefined,
      // Never infer this server-side: legacy checkpoints and current cloud pods
      // are weights-only, while a verified exact bundle opts into the bridge.
      resumeMode,
      stateBundleId: resumeMode === 'full_state'
        ? selectedCheckpoint?.resume_state?.bundle_id : null,
    });
  };

  const stepLabel = (s) => {
    const tags = [];
    if (s === latest) tags.push('latest');
    if (s === bestStep) tags.push('best');
    const state = preferredCheckpointForStep(checkpoints, s)?.resume_state;
    if (['ready', 'complete'].includes(state?.status) && state?.integrity === 'verified'
        && state?.state_level === 'exact') tags.push('full state');
    else tags.push('weights only');
    return `step ${s}${tags.length ? ` — ${tags.join(', ')}` : ''}`;
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Continue training"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) dismiss(); }}>
      <div ref={cardRef}
        className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-indigo-400/40 bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-indigo-300 font-semibold"><span aria-hidden>▶</span> Continue training</span>
          <HelpBadge topic="continue-training" />
          {context && <span className="text-content-subtle text-[0.75rem] truncate">{context}</span>}
          <button type="button" onClick={dismiss} disabled={busy}
            className="ml-auto text-content-subtle hover:text-content disabled:opacity-40" aria-label="Cancel">✕</button>
        </div>

        {/* WHERE the continuation runs — offered only when the mount passes
            `lanes`. A checkpoint is just a file: a run trained here can be
            finished on a rented GPU, and a cloud run's epoch can be finished on
            this machine. An unusable lane is disabled WITH its reason, never
            hidden. */}
        {lanes && (
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-content text-[0.75rem] w-28 shrink-0">Run it</span>
              <div role="radiogroup" aria-label="Where to run the continuation"
                className="flex items-center gap-1 rounded-lg border border-border bg-surface p-0.5">
                {[['local', '💻 Local'], ['cloud', '☁ Cloud']].map(([id, label]) => {
                  const st = laneState(id);
                  const off = st.available === false;
                  return (
                    <button key={id} type="button" role="radio" aria-checked={lane === id}
                      disabled={off} onClick={() => setLane(id)}
                      title={off ? st.reason || undefined : `Continue on the ${id === 'cloud' ? 'rented cloud GPU' : 'local GPU'}`}
                      className={'px-2.5 py-1 rounded-md text-[0.75rem] font-semibold '
                        + (lane === id
                          ? 'bg-indigo-500/25 text-indigo-100 border border-indigo-400/50 '
                          : 'text-content-muted hover:text-content border border-transparent ')
                        + (off ? 'opacity-40 cursor-not-allowed ' : '')}>
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>
            {laneState(lane).reason && (
              <span className="text-amber-300/90 text-[0.6875rem] leading-relaxed">
                {laneState(lane).reason}
              </span>
            )}
          </div>
        )}

        {/* HOW the checkpoint reaches the pod — full models only.
            A ~26 GB file has two roads and they are not interchangeable: the
            pod can pull it from Hugging Face over a datacenter link in minutes,
            or this computer can push it up over the user's own uplink in hours.
            The pod is RENTED AND BILLED either way, so the slow road costs real
            money for a GPU that computes nothing — which is why the price sits
            under the buttons and not in a tooltip. */}
        {transportPlan && (
          <div className="flex flex-col gap-1">
            <div className="flex items-start gap-2 flex-wrap">
              <span className="text-content text-[0.75rem] w-28 shrink-0 pt-1">Send it via</span>
              <div role="radiogroup" aria-label="How the checkpoint reaches the pod"
                className="flex items-center gap-1 rounded-lg border border-border bg-surface p-0.5 flex-wrap">
                {TRANSPORTS.map((id) => {
                  const option = transportOption(transportPlan.options, id);
                  const off = option.available === false;
                  return (
                    <button key={id} type="button" role="radio" aria-checked={transport === id}
                      disabled={off} onClick={() => setTransport(id)}
                      title={off ? option.reason || undefined : undefined}
                      className={'px-2.5 py-1 rounded-md text-[0.75rem] font-semibold '
                        + (transport === id
                          ? 'bg-indigo-500/25 text-indigo-100 border border-indigo-400/50 '
                          : 'text-content-muted hover:text-content border border-transparent ')
                        + (off ? 'opacity-40 cursor-not-allowed ' : '')}>
                      {TRANSPORT_LABELS[id]}
                    </button>
                  );
                })}
              </div>
            </div>
            {/* Every road that is CLOSED says so on screen, whether or not it is
                the one selected. Putting that in the button's `title` would be
                greyed out with no reason for anyone who is not hovering a mouse
                — which is the exact complaint this picker answers. */}
            {closedRoads(transportPlan).map(({ transport: id, label, reason }) => (
              <span key={id}
                className="text-amber-300/90 text-[0.6875rem] leading-relaxed break-words">
                {label} is unavailable: {reason}
              </span>
            ))}
            {/* The numbers for the SELECTED road — shown only when it is a road
                that can actually be taken. */}
            {!transportReason && (
                <div className="flex flex-col gap-0.5 sm:pl-[7.5rem]">
                  <span className="text-content text-[0.6875rem] leading-relaxed break-words">
                    {transportSummary(transportOption(transportPlan.options, transport))}
                  </span>
                  <span className="text-content-subtle text-[0.6875rem] leading-relaxed break-words">
                    {rateNote(transportOption(transportPlan.options, transport))}
                  </span>
                  {transport === 'direct' && (
                    <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                      Sent in pieces: if it stops, continuing this run again picks
                      up where it left off instead of starting over.
                    </span>
                  )}
                  {transport === 'hub' && (
                    <span className="text-content-subtle text-[0.6875rem] leading-relaxed">
                      Fast, and the weights pass through Hugging Face on the way.
                    </span>
                  )}
                </div>
              )}
          </div>
        )}

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

        {/* HOW state is restored. A verified exact bundle is the local default;
            every legacy/corrupt/cloud case visibly falls back to weights-only
            with the concrete reason, never by implication. */}
        <div className="flex flex-col gap-1">
          <span className="text-content text-[0.75rem]">Resume mode</span>
          <div role="radiogroup" aria-label="Training state to restore"
            className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-raised p-2.5">
            <label className={'flex items-start gap-2 text-[0.75rem] '
              + (fullStateAvailable ? 'text-content' : 'text-content-subtle')}>
              <input type="radio" name="resume-mode" value="full_state"
                checked={resumeMode === 'full_state'}
                disabled={!fullStateAvailable}
                onChange={() => setResumeMode('full_state')} />
              <span>
                <b>Full state</b> — weights, optimizer, scheduler, RNG and next batch
              </span>
            </label>
            <label className="flex items-start gap-2 text-content text-[0.75rem]">
              <input type="radio" name="resume-mode" value="weights_only"
                checked={resumeMode === 'weights_only'}
                onChange={() => setResumeMode('weights_only')} />
              <span><b>Weights only</b> — starts fresh optimizer/scheduler state</span>
            </label>
            {fullStateReason && (
              <span className="text-amber-300/90 text-[0.6875rem] leading-relaxed">
                {fullStateReason}
              </span>
            )}
          </div>
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
              {lane === 'cloud'
                ? `Restarts from step ${fromStep} on a fresh pod: this checkpoint is uploaded and`
                  + ' trained further — every save you have here stays exactly where it is.'
                : `Restarts from step ${fromStep}: the run’s later checkpoints are set aside (kept`
                  + (where === 'cloud' ? ' in this run’s staging' : ' on disk')
                  + ', recoverable), and the continuation writes its own saves.'}
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
                <select value={String(trajectoryLocked ? inheritedSave : saveEvery)}
                  onChange={(e) => setSaveEvery(Number(e.target.value))}
                  disabled={trajectoryLocked}
                  aria-label="Checkpoint frequency"
                  title={trajectoryLocked
                    ? 'Full state preserves the checkpoint save cadence'
                    : undefined}
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-40">
                  {SAVE_CHOICES.map((n) => <option key={n} value={String(n)}>every {n} steps</option>)}
                </select>
                <span className="text-content-subtle text-[0.625rem]">how often a checkpoint is written</span>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Preview every</span>
                <select value={String(trajectoryLocked ? inheritedSampleEvery : sampleEvery)}
                  onChange={(e) => setSampleEvery(Number(e.target.value))}
                  disabled={trajectoryLocked}
                  aria-label="Preview sample frequency"
                  title={trajectoryLocked
                    ? 'Full state preserves the checkpoint preview cadence'
                    : undefined}
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-40">
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
                <span className="text-content text-[0.75rem] w-28 shrink-0">Preview quality</span>
                <label className="flex items-center gap-1.5">
                  <input type="number" min="1" max="60" step="1" value={sampleSteps}
                    onChange={(e) => setSampleSteps(e.target.value)}
                    placeholder="keep"
                    aria-label="Preview steps"
                    className="w-16 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
                  <span className="text-content-muted text-[0.625rem]">steps</span>
                </label>
                <label className="flex items-center gap-1.5">
                  <input type="number" min="1" max="20" step="0.5" value={sampleGuidance}
                    onChange={(e) => setSampleGuidance(e.target.value)}
                    placeholder="keep"
                    aria-label="Preview guidance scale"
                    className="w-16 px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem]" />
                  <span className="text-content-muted text-[0.625rem]">CFG</span>
                </label>
                <span className="text-content-subtle text-[0.625rem]">preview rendering only — allowed even on full state</span>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Timestep weighting</span>
                <select value={timestep} onChange={(e) => setTimestep(e.target.value)}
                  disabled={trajectoryLocked}
                  aria-label="Timestep weighting for the continuation"
                  title={trajectoryLocked
                    ? 'Full state preserves the checkpoint training trajectory'
                    : undefined}
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-40">
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
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-content text-[0.75rem] w-28 shrink-0">Learning rate</span>
                <select value={String(lrFactor)} onChange={(e) => setLrFactor(Number(e.target.value))}
                  disabled={isAdaptiveLR || trajectoryLocked}
                  aria-label="Learning rate for the continuation"
                  title={trajectoryLocked
                    ? 'Full state preserves the checkpoint training trajectory'
                    : (isAdaptiveLR
                      ? 'Prodigy adapts the learning rate itself (lr=1) — there is no base rate to scale'
                      : undefined)}
                  className="px-2 py-1 rounded-lg border border-border bg-surface text-content text-[0.75rem] disabled:opacity-40">
                  {LR_FACTOR_CHOICES.map((c) => <option key={c.value} value={String(c.value)}>{c.label}</option>)}
                </select>
                <span className="text-content-muted text-[0.625rem] tabular-nums">
                  {isAdaptiveLR
                    ? 'Prodigy — adaptive'
                    : (lrFactor === 1 ? `keeps ${fmtLR(currentLR)}` : `→ ${fmtLR(currentLR * lrFactor)}`)}
                </span>
              </div>
              <span className="text-content-subtle text-[0.625rem] leading-relaxed">
                <b className="text-content-muted font-medium">Why LR:</b> resume the epoch that held up best and finish
                gentler — a smaller rate polishes texture without moving the identity, the LR pendant of the low-noise
                timestep recipe. The values are factors of this run&apos;s current rate.
              </span>
              {/* This was already the contract; it is now also what the backend
                  enforces. Naming the actual rank makes it checkable at a glance
                  — the Estelle run continued a rank-64 LoRA while the dataset had
                  been edited to rank 32, and nothing on screen said which won. */}
              <span className="text-content-subtle text-[0.625rem] leading-relaxed">
                {trajectoryLocked
                  ? 'Full state keeps the learning rate, timestep trajectory and save/preview cadence exact; only preview prompts can change. '
                  : 'Weights-only resume may also change timestep weighting and learning rate. '}
                Rank, base, optimizer and the like are locked to the checkpoint being continued
                {settings?.rank
                  ? <> (rank {settings.rank}{settings.alpha ? ` · alpha ${settings.alpha}` : ''}, inherited from it)</>
                  : null}.
              </span>
            </div>
          )}
        </div>

        {/* A disabled ▶ Continue that explains nothing reads as a broken button.
            The blocked lane already prints its reason above; this covers the
            state that printed none at all. */}
        {blockedReason && (
          <span className="text-amber-300/90 text-[0.6875rem] leading-relaxed">{blockedReason}</span>
        )}

        {/* The LAST attempt's refusal, right above the button that produced it —
            and the inputs it is about are still filled in. Scrolls on its own
            rather than pushing ▶ Continue off a 400-px screen: a backend
            refusal can be several lines ("no local checkpoint at step 750
            (available: 250, 500)"), and an error that hides the retry button is
            the same dead end wearing a different coat. */}
        {error && (
          <div role="alert"
            /* shrink-0: the card is a flex column with a max height, so a flex
               child is free to be SQUASHED — with the settings unfolded the
               message rendered as a 20-px sliver of clipped text (measured). */
            className="shrink-0 rounded-lg border border-red-500/40 bg-red-500/10 px-2.5 py-2 max-h-28 overflow-y-auto">
            <span className="block text-red-200 text-[0.6875rem] leading-relaxed whitespace-pre-wrap break-words">
              {error}
            </span>
            <span className="block text-content-subtle text-[0.625rem] mt-1">
              Your choices are kept — adjust and try again.
            </span>
          </div>
        )}

        <div className="flex items-center gap-2 pt-1">
          <button type="button" onClick={dismiss} disabled={busy}
            className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm disabled:opacity-40">Cancel</button>
          <button type="button" onClick={submit}
            disabled={busy || latest === 0 || laneBlocked
              || (resumeMode === 'full_state' && !fullStateAvailable)}
            title={laneBlocked ? laneState(lane).reason || undefined : undefined}
            className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {busy ? 'Starting…' : `Continue → ${target}`}
          </button>
        </div>
      </div>
    </div>
  );
}
