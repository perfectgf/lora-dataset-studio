import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import { apiFetch } from '../../api/fetchClient';
import { HelpBadge } from '../../help/HelpMode';
import { postJson } from '../../hooks/useDataset';
import useHubPresence from '../../hooks/useHubPresence';
import Fp8QuantizeTool from './Fp8QuantizeTool';
import LoraMergeTool from './LoraMergeTool';
import {
  denseActions, denseFileRows, denseGuidanceLine, denseHubLine, denseModelTitle,
  denseStudioTarget, denseWhereChip, fmtBytes, STUDIO_NEEDS_A_LORA,
} from './denseModels';

/** Full models, in the panel where every other trained thing already lives.
 *
 * WHY THIS IS A SECOND LANE AND NOT A FEW MORE ROWS
 * -------------------------------------------------
 * The list above deploys LoRA adapters into ComfyUI's `loras/<family>`. A dense
 * run does not produce an adapter, and the backend guard that keeps it out of
 * that list is right to. Folding a 26 GB transformer into those rows would have
 * given it the adapter's verbs — "Import →", "Undeploy" — for an operation that
 * must never happen.
 *
 * So a full model gets the verbs it actually has: quantize the master into the
 * fp8 file ComfyUI loads, put THAT where ComfyUI looks, and throw either one in
 * the trash. The master is not sendable from here by construction: there is no
 * control that does it and no endpoint behind one.
 */
const TONE = {
  ok: 'border-emerald-400/40 bg-emerald-500/10 text-emerald-200',
  info: 'border-sky-400/40 bg-sky-500/10 text-sky-200',
  error: 'border-rose-400/45 bg-rose-500/10 text-rose-200',
  muted: 'border-border text-content-subtle',
};

const LINE_TONE = {
  ok: 'text-emerald-200',
  error: 'text-rose-200',
  muted: 'text-content-subtle',
};

function Chip({ tone = 'muted', title = '', children }) {
  return (
    <span title={title}
      className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[0.625rem] font-medium ${TONE[tone] || TONE.muted}`}>
      {children}
    </span>
  );
}

/** The Hugging Face line of a card: the repository, and ONE sentence chosen
 *  from what is actually known about it.
 *
 *  Exported so a test can render it in every state. The bug it replaces was
 *  invisible to a payload test and to a source-text test alike: the status came
 *  from `entry.hub.status` and the sentence came from `rows.length`, so the DB
 *  could say "missing" while the paragraph next to it said "the model is there"
 *  — two assertions, one screen, and nothing that ever compared them.
 *
 *  Wraps and breaks at 400 px: a repository id is long, and a card that scrolls
 *  sideways on a phone hides the very sentence this exists to show. */
function HubLine({ entry, presence = null }) {
  const line = denseHubLine(entry, presence);
  if (!line) return null;
  return (
    // `status`, not `alert`: the sentence appears when the check lands, and a
    // panel with seven gone repositories would otherwise interrupt a screen
    // reader seven times over something nobody has to act on this second.
    <p className={`m-0 mt-1.5 text-[0.625rem] leading-snug ${LINE_TONE[line.tone] || LINE_TONE.muted}`}
      role={line.state === 'gone' ? 'status' : undefined}>
      <span className="text-content-subtle">Backup: </span>
      {line.url ? (
        <a href={line.url} target="_blank" rel="noreferrer"
          className="break-all text-primary underline">{line.repoId}</a>
      ) : <span className="break-all font-mono">{line.repoId}</span>}
      <span className="text-content-subtle"> · {line.stateLabel}</span>
      <span className="block">{line.text}</span>
    </p>
  );
}

/** One artifact line. Stacks on a phone and wraps its filename: a dense weight
 *  name is long, and a card that scrolls sideways at 400 px hides the size and
 *  the state — the two things the row exists to show. */
function FileRow({ row, children }) {
  return (
    <li className="rounded-md border border-border bg-app/60 px-2 py-1.5">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className="text-content text-[0.6875rem] font-semibold">{row.label}</span>
        {row.bytes ? (
          <span className="text-content-subtle text-[0.625rem]">{fmtBytes(row.bytes)}</span>
        ) : null}
        <Chip tone={row.state === 'in-comfyui' ? 'ok'
          : (row.state === 'delivered' ? 'info' : 'muted')}>{row.stateLabel}</Chip>
      </div>
      <p className="m-0 mt-0.5 break-all font-mono text-content-muted text-[0.625rem]">
        {row.filename}
      </p>
      <p className="m-0 mt-0.5 text-content-subtle text-[0.625rem] leading-snug">
        {row.role}
      </p>
      {row.choice ? (
        <p className="m-0 mt-0.5 text-content-subtle text-[0.625rem] leading-snug">
          Takes {row.choice}.
        </p>
      ) : null}
      {children}
    </li>
  );
}

/** The confirm step of "Send to ComfyUI": link or copy, where, and what it costs.
 *  A hard link is instantaneous and free — saying so is the difference between
 *  "this will take a while" and one click. */
function SendPlan({ plan, busy, onSend, onCancel }) {
  if (!plan) return null;
  if (!plan.ok) {
    return (
      <p className="m-0 mt-1 text-amber-200 text-[0.625rem] leading-snug" role="alert">
        ⚠ {plan.error}
      </p>
    );
  }
  const linked = plan.method === 'linked';
  return (
    <div className="mt-1 rounded-md border border-primary/40 bg-primary/10 px-2 py-1.5 text-[0.625rem] leading-relaxed">
      <p className="m-0">
        {linked
          ? 'Links the file into '
          : `Copies ${fmtBytes(plan.total_bytes)} into `}
        <span className="break-all font-mono">{plan.destination_dir}</span>.
        {linked ? ' Same drive, so it takes no extra disk space and is instant.' : ''}
      </p>
      <p className={`m-0 mt-0.5 ${plan.destination_dir_kind === 'comfyui' ? 'opacity-85' : 'text-amber-200'}`}>
        {plan.destination_dir_kind === 'comfyui'
          ? `✓ ${plan.destination_dir_note} — ComfyUI lists it after a refresh.`
          : `⚠ ${plan.destination_dir_note}.`}
      </p>
      {plan.enough_space === false && (
        <p className="m-0 mt-0.5 text-rose-200" role="alert">
          ✗ Not enough space: {fmtBytes(plan.free_bytes)} free, {fmtBytes(plan.required_bytes)} needed.
        </p>
      )}
      <div className="mt-1 flex flex-wrap gap-1.5">
        <button type="button" onClick={onSend} disabled={busy || plan.enough_space === false}
          className="rounded-md border border-primary/50 bg-primary/20 px-2.5 py-1 font-semibold text-white hover:bg-primary/30 disabled:opacity-40">
          {busy ? 'Working…' : (linked ? 'Link it' : 'Copy it')}
        </button>
        <button type="button" onClick={onCancel}
          className="rounded-md border border-border px-2.5 py-1 font-medium text-content-muted hover:bg-app">
          Cancel
        </button>
      </div>
    </div>
  );
}

export default function DenseModelsPanel({ datasetId, models = [], onChanged = null,
  hubPresenceOverride = null }) {
  // Asked once, after this panel has already painted. Until it answers, every
  // card reads from the RECORD and says so — no sentence below depends on this
  // arriving. `hubPresenceOverride` is the seam the render tests use: effects
  // never run under renderToStaticMarkup, so the states that matter most (a
  // repository verified gone) would otherwise be unreachable from a test.
  const fetched = useHubPresence(models.map((m) => m.run_id));
  const hubPresence = hubPresenceOverride || fetched;
  const [plans, setPlans] = useState({});          // run_id -> send plan
  const [quantizeFor, setQuantizeFor] = useState(null);
  const [mergeFor, setMergeFor] = useState(null);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState(null);
  const pollRef = useRef(null);

  const stop = () => { clearInterval(pollRef.current); pollRef.current = null; };

  const poll = useCallback(() => {
    apiFetch('/api/tools/dense-send/status')
      .then((s) => {
        setJob(s);
        if (s?.status !== 'sending') { stop(); onChanged?.(); }
      })
      .catch(() => {});
  }, [onChanged]);

  // A copy outlives a visit to this tab. Adopt one already running rather than
  // offering to start a second — the server refuses that anyway.
  useEffect(() => {
    let alive = true;
    apiFetch('/api/tools/dense-send/status')
      .then((s) => {
        if (!alive || s?.status !== 'sending') return;
        setJob(s);
        stop();
        pollRef.current = setInterval(poll, 2000);
      })
      .catch(() => {});
    return () => { alive = false; stop(); };
  }, [poll]);

  if (!models.length) return null;

  const askSend = async (runId) => {
    setBusy(true);
    const plan = await postJson(`/api/dataset/${datasetId}/train/dense/send-plan`,
      { run_id: runId });
    setPlans((p) => ({ ...p, [runId]: plan }));
    setBusy(false);
  };

  const doSend = async (runId) => {
    setBusy(true);
    const res = await postJson(`/api/dataset/${datasetId}/train/dense/send`,
      { run_id: runId });
    setBusy(false);
    setPlans((p) => ({ ...p, [runId]: null }));
    if (!res?.ok) {
      setJob({ status: 'error', error: res?.error || 'The model could not be sent.' });
      return;
    }
    setJob(res.job || { status: res.status });
    if (res.status === 'sending') {
      stop();
      pollRef.current = setInterval(poll, 2000);
    } else {
      onChanged?.();
    }
  };

  const trash = async (runId, filename, label) => {
    if (!window.confirm(
      `Move ${filename} to the app trash?\n\n`
      + `It is the ${label} of run #${runId}. Nothing is deleted permanently — `
      + 'you can restore it from the trash.')) return;
    setBusy(true);
    const res = await postJson(`/api/dataset/${datasetId}/train/dense/delete`,
      { run_id: runId, filename });
    setBusy(false);
    if (!res?.ok) {
      setJob({ status: 'error', error: res?.error || 'That file could not be moved.' });
      return;
    }
    onChanged?.();
  };

  return (
    <section className="rounded-lg border border-sky-300/30 bg-sky-400/5 px-3 py-2"
      aria-labelledby="ds-dense-models-title">
      <h4 id="ds-dense-models-title" className="m-0 flex flex-wrap items-center gap-x-2 text-content text-[0.8125rem] font-semibold">
        <span>🧱 Full models</span>
        <HelpBadge topic="workspace-dense-models" />
        <span className="font-normal text-content-subtle text-[0.6875rem]">
          {models.length} run{models.length > 1 ? 's' : ''} trained the whole model, not an adapter
        </span>
      </h4>

      <ul className="m-0 mt-2 flex list-none flex-col gap-2 p-0">
        {models.map((entry) => {
          const presence = hubPresence[entry.run_id] || null;
          const where = denseWhereChip(entry, presence);
          const rows = denseFileRows(entry);
          const actions = denseActions(entry, presence);
          const guidance = denseGuidanceLine(entry.inference_hint);
          const studio = denseStudioTarget(entry);
          const plan = plans[entry.run_id];
          return (
            <li key={entry.run_id}
              className="rounded-lg border border-border bg-surface px-2.5 py-2">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="text-content text-[0.75rem] font-semibold">
                  {denseModelTitle(entry)}
                </span>
                <Chip tone={where.tone} title={where.title}>{where.label}</Chip>
                {entry.version ? (
                  <span className="text-content-subtle text-[0.625rem]">v{entry.version}</span>
                ) : null}
                {entry.steps ? (
                  <span className="text-content-subtle text-[0.625rem]">{entry.steps} steps</span>
                ) : null}
              </div>

              {guidance && (
                <p className="m-0 mt-1 rounded-md border border-amber-400/30 bg-amber-500/10 px-2 py-1 text-amber-100 text-[0.625rem] leading-snug">
                  This is an undistilled model: sample it at <b>{guidance}</b>. The
                  family’s few-step Turbo defaults render mush on it.
                </p>
              )}

              {rows.length > 0 && (
                <ul className="m-0 mt-1.5 flex list-none flex-col gap-1.5 p-0">
                  {rows.map((row) => (
                    <FileRow key={row.kind} row={row}>
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        {row.kind === 'fp8' && actions.send && (
                          <button type="button" onClick={() => askSend(entry.run_id)}
                            disabled={busy || !actions.send.enabled}
                            title="Put this file where ComfyUI looks for diffusion models"
                            className="rounded-md border border-primary/40 bg-primary/20 px-2 py-0.5 text-[0.625rem] font-semibold text-white hover:bg-primary/30 disabled:opacity-40">
                            {actions.send.label}
                          </button>
                        )}
                        {entry.can_delete && (
                          <button type="button"
                            onClick={() => trash(entry.run_id, row.filename, row.label)}
                            disabled={busy}
                            title="Move this file to the app trash — recoverable"
                            className="rounded-md border border-border px-2 py-0.5 text-[0.625rem] font-medium text-content-muted hover:bg-app disabled:opacity-40">
                            🗑 Trash
                          </button>
                        )}
                      </div>
                      {row.kind === 'fp8' && plan !== undefined && plan !== null && (
                        <SendPlan plan={plan} busy={busy}
                          onSend={() => doSend(entry.run_id)}
                          onCancel={() => setPlans((p) => ({ ...p, [entry.run_id]: null }))} />
                      )}
                    </FileRow>
                  ))}
                </ul>
              )}

              <HubLine entry={entry} presence={presence} />

              {entry.trainer && (
                <p className="m-0 mt-0.5 break-all text-content-subtle text-[0.5625rem] leading-snug">
                  Trained on <span className="font-mono">{entry.trainer}</span>
                </p>
              )}

              {actions.activeNote && (
                <p className="m-0 mt-1 text-amber-200 text-[0.625rem]">{actions.activeNote}</p>
              )}

              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {actions.quantize && (
                  <button type="button"
                    onClick={() => setQuantizeFor(
                      quantizeFor?.run_id === entry.run_id ? null : entry)}
                    // A button whose own promise is "quantizing fetches it
                    // first" cannot be offered once the repository it would
                    // fetch from has been measured gone: clicking it can only
                    // fail, and the reason belongs here, before the click.
                    disabled={busy || !actions.quantize.enabled}
                    title={actions.quantize.reason || undefined}
                    className="rounded-md border border-sky-300/40 bg-sky-400/15 px-2.5 py-1 text-[0.6875rem] font-semibold text-sky-50 hover:bg-sky-400/25 disabled:opacity-40">
                    {quantizeFor?.run_id === entry.run_id ? 'Hide' : actions.quantize.label}
                  </button>
                )}
                {/* Only for a master that is HERE. Merging reads the whole
                    checkpoint tensor by tensor, so a model that exists only in a
                    Hugging Face repo has nothing to merge into yet — offering the
                    button anyway would be a refusal dressed as an action. */}
                {entry.master?.path && (
                  <button type="button"
                    onClick={() => setMergeFor(
                      mergeFor?.run_id === entry.run_id ? null : entry)}
                    disabled={busy}
                    title="Fold a LoRA into this model's weights and write a new full model"
                    className="rounded-md border border-sky-300/40 bg-sky-400/15 px-2.5 py-1 text-[0.6875rem] font-semibold text-sky-50 hover:bg-sky-400/25 disabled:opacity-40">
                    {mergeFor?.run_id === entry.run_id ? 'Hide' : '🧬 Merge a LoRA in'}
                  </button>
                )}
              </div>

              {actions.quantize?.reason && (
                <p className="m-0 mt-1 text-content-subtle text-[0.625rem] leading-snug">
                  {actions.quantize.reason}
                </p>
              )}

              {mergeFor?.run_id === entry.run_id && (
                <div className="mt-1.5 rounded-md border border-sky-300/30 bg-app/50 px-2 py-1.5">
                  <LoraMergeTool framed={false} family={entry.train_type}
                    base={entry.master.path}
                    baseLabel="this run’s full model" />
                </div>
              )}

              {/* The honest limit, next to the thing that would otherwise look
                  broken: the Test Studio is entered through a LoRA of this
                  dataset, so a dataset trained only as a full model cannot open
                  it. Said here rather than discovered on an empty screen. */}
              {studio && (
                <>
                  <Link to={`/studio?dataset=${studio.datasetId}`
                    + `&family=${encodeURIComponent(studio.family)}`
                    + `&base=${encodeURIComponent(studio.base)}`}
                    className="mt-1.5 inline-block rounded-md border border-primary/40 bg-primary/15 px-2.5 py-1 text-[0.6875rem] font-semibold text-content hover:bg-primary/25">
                    🧪 Test in Studio
                  </Link>
                  <p className="m-0 mt-1 text-content-subtle text-[0.625rem] leading-snug">
                    Opens the Studio on this model, with its own sample settings
                    already filled in. {STUDIO_NEEDS_A_LORA}
                  </p>
                </>
              )}

              {quantizeFor?.run_id === entry.run_id && (
                <div className="mt-1.5 rounded-md border border-sky-300/30 bg-app/50 px-2 py-1.5">
                  <Fp8QuantizeTool framed={false} manualPath={false}
                    target={{
                      label: 'This run’s full model',
                      name: entry.master?.filename || entry.hub?.weight_filename || '',
                      sizeBytes: entry.master?.size_bytes || 0,
                      family: entry.train_type,
                      path: entry.master?.path || null,
                      repoId: entry.master ? null : (entry.hub?.repo_id || null),
                      filename: entry.master ? null : (entry.hub?.weight_filename || null),
                    }} />
                </div>
              )}
            </li>
          );
        })}
      </ul>

      {job?.status === 'sending' && (
        <p className="m-0 mt-2 text-content-muted text-[0.625rem]" role="status">
          → Copying {job.filename} into {job.destination_dir} —{' '}
          {fmtBytes(job.done_bytes)} of {fmtBytes(job.total_bytes)}.
        </p>
      )}
      {job?.status === 'done' && (
        <p className="m-0 mt-2 text-emerald-200 text-[0.625rem]" role="status">
          ✓ {job.filename} is in {job.destination_dir}
          {job.method === 'linked' ? ' (linked — no extra disk space used)' : ''}. ComfyUI
          lists it after a refresh.
        </p>
      )}
      {job?.status === 'error' && (
        <p className="m-0 mt-2 text-rose-200 text-[0.625rem]" role="alert">✗ {job.error}</p>
      )}
    </section>
  );
}
