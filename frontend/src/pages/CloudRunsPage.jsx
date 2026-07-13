import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import TrainingProgress from '../components/dataset/TrainingProgress';

/* Dedicated hub for cloud training runs across ALL datasets: watch the ones in
   progress (live progress + samples), stop them, and download finished LoRAs —
   without hunting through each dataset's panel. Polls the aggregate
   /train/cloud/runs endpoint (actives + recent history + budget summary). */

const POLL_MS = 5000;
const FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1' };

const STATUS_STYLE = {
  done: 'text-emerald-300 border-emerald-400/40 bg-emerald-500/10',
  error: 'text-rose-300 border-rose-400/40 bg-rose-500/10',
  error_pod_kept: 'text-amber-200 border-amber-400/40 bg-amber-500/10',
  stopped: 'text-content-muted border-border bg-surface',
};
const statusStyle = (s) =>
  STATUS_STYLE[s] || 'text-sky-300 border-sky-400/40 bg-sky-500/10';

function timeAgo(iso) {
  if (!iso) return '';
  // backend timestamps are naive UTC (isoformat of utcnow) — pin to UTC.
  const t = new Date(/[Z+]/.test(iso) ? iso : `${iso}Z`).getTime();
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function famLabel(f) { return FAMILY_LABEL[f] || f || 'LoRA'; }

/* One compact line: the EFFECTIVE ai-toolkit settings this launch used
   (snapshotted at launch by the provenance registry). Absent on rows that
   predate the snapshot feature. */
function settingsLine(run) {
  const s = run.settings;
  if (!s) return null;
  return [
    s.rank ? `rank ${s.rank}${s.alpha ? `/${s.alpha}` : ''}` : null,
    Array.isArray(s.resolution) ? `${s.resolution.join('+')} px` : null,
    run.steps ? `${run.steps} steps` : null,
    s.save_every ? `save ${s.save_every}` : null,
    s.optimizer && s.optimizer !== 'adamw8bit' ? s.optimizer : null,
    s.lr_scheduler || null,
    s.dropout ? `dropout ${s.dropout}` : null,
    s.timestep_type || null,
    run.variant ? (run.variant === 'base' ? 'Raw' : run.variant) : null,
    run.masked === false ? 'unmasked' : 'masked',
  ].filter(Boolean).join(' · ');
}

function checkpointHref(run) {
  const qs = new URLSearchParams();
  if (run.train_type) qs.set('train_type', run.train_type);
  // run_id: THIS row's file — with several finished runs of a family in the
  // history, family resolution alone would serve the newest run's checkpoint.
  if (run.run_id) qs.set('run_id', String(run.run_id));
  return `/api/dataset/${run.dataset_id}/train/cloud/checkpoint?${qs.toString()}`;
}

export default function CloudRunsPage() {
  const toast = useToast();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [stopping, setStopping] = useState({});     // run_id -> bool

  const poll = useCallback(async () => {
    try {
      const r = await fetch('/api/dataset/train/cloud/runs?limit=15', { credentials: 'include' });
      if (r.ok) setData(await r.json());
    } catch { /* transient — next tick retries */ }
  }, []);

  useEffect(() => {
    let alive = true;
    let t;
    const tick = async () => { await poll(); if (alive) t = setTimeout(tick, POLL_MS); };
    tick();
    return () => { alive = false; clearTimeout(t); };
  }, [poll]);

  const openDataset = (id) => {
    try { localStorage.setItem('datasetCurrentId', String(id)); } catch { /* ignore */ }
    navigate('/datasets');
  };

  const stop = async (run) => {
    const who = run.dataset_name || run.run_name || `run #${run.run_id}`;
    if (!window.confirm(`Stop the cloud run for « ${who} »?\n\n`
      + 'The pod is terminated. Any checkpoint reached so far is still downloaded '
      + 'and importable — you only lose the remaining steps.')) return;
    setStopping((m) => ({ ...m, [run.run_id]: true }));
    try {
      const d = await postJson('/api/dataset/train/cloud/stop', { run_id: run.run_id });
      if (d.ok === false) toast.error('Could not stop the run — it may have already finished.');
      else toast.info('Stopping the run — the pod is winding down…');
      poll();
    } finally {
      setStopping((m) => ({ ...m, [run.run_id]: false }));
    }
  };

  // ↻ Retry of a failed run: fresh pod, exact same settings as the failed
  // launch (steps/variant/family/masked/GPU class) — the two field failures
  // (vanished vast offer, pod never ready) are transient by nature.
  const [retrying, setRetrying] = useState({});      // run_id -> bool
  const retry = async (run) => {
    setRetrying((m) => ({ ...m, [run.run_id]: true }));
    try {
      const d = await postJson('/api/dataset/train/cloud/retry', { run_id: run.run_id });
      if (d.ok === false) toast.error(d.error || 'Retry failed');
      else toast.success('Run relaunched — provisioning a fresh pod…');
      poll();
    } finally {
      setRetrying((m) => ({ ...m, [run.run_id]: false }));
    }
  };

  // ▶ Continue a finished cloud run: fresh pod, same settings, resuming from the
  // run's last harvested checkpoint for `extra` more steps (ai-toolkit
  // auto-resume — the monitor seeds the checkpoint onto the pod before start).
  const [continuing, setContinuing] = useState({});   // run_id -> bool
  const continueRun = async (run) => {
    const raw = window.prompt('Additional steps to train from the last checkpoint:', '1000');
    if (raw == null) return;                            // cancelled
    const extra = parseInt(raw, 10);
    if (!Number.isFinite(extra) || extra <= 0) {
      toast.error('Enter a positive number of extra steps.');
      return;
    }
    setContinuing((m) => ({ ...m, [run.run_id]: true }));
    try {
      const d = await postJson('/api/dataset/train/cloud/continue',
        { run_id: run.run_id, extra_steps: extra });
      if (d.ok === false) toast.error(d.error || 'Continue failed');
      else toast.success(`Continuing from step ${d.resumed_from} → ${d.target_steps} on a fresh pod…`);
      poll();
    } finally {
      setContinuing((m) => ({ ...m, [run.run_id]: false }));
    }
  };

  const configured = data?.configured;
  const actives = data?.actives || [];
  const recent = data?.recent || [];
  const limit = data?.limit || 1;
  const budget = data?.monthly_budget || 0;
  const spent = data?.month_spend || 0;

  return (
    <section className="flex flex-col gap-5">
      <header className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="m-0 text-content text-xl font-bold">
            <span aria-hidden>🏋️</span> Training runs
          </h1>
          {/* Escape hatch to the provider: see the pod's own console (billing,
              logs, manual destroy) when something looks off app-side. */}
          <a href="https://cloud.vast.ai/instances/" target="_blank" rel="noreferrer"
            className="ml-auto text-xs font-medium text-sky-300 underline hover:text-sky-200">
            Open the vast.ai console ↗
          </a>
        </div>
        <p className="m-0 text-content-muted text-sm">
          Every training in one place — cloud and local: watch progress, stop a run,
          download a finished LoRA, and see the exact settings each launch used.
        </p>
      </header>

      {data && !configured && (
        <div className="rounded-lg border border-border bg-surface p-4 text-content-muted text-sm">
          Cloud training isn’t configured yet. Add your vast.ai API key in{' '}
          <button type="button" onClick={() => navigate('/settings')}
            className="text-sky-300 underline hover:text-sky-200">Settings</button>{' '}
          to rent GPUs on demand.
        </div>
      )}

      {configured && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm">
          <span className="text-content">
            <b className="tabular-nums">{actives.length}</b>
            <span className="text-content-muted">/{limit} active</span>
          </span>
          <span className="text-content-muted tabular-nums">
            ${data.total_price_per_hour || 0}/h total
          </span>
          <span className="text-content-muted tabular-nums">
            this month: ${spent.toFixed(2)}{budget > 0 ? ` of $${budget.toFixed(2)}` : ' (no budget cap)'}
          </span>
        </div>
      )}

      {/* Active runs */}
      <div className="flex flex-col gap-3">
        <h2 className="m-0 text-content-muted text-xs font-semibold uppercase tracking-wide">
          In progress
        </h2>
        {/* Live LOCAL training — its own card next to the cloud actives. */}
        {data?.local_active?.current && (
          <div className="flex flex-col gap-2 rounded-xl border border-violet-500/30 bg-violet-500/5 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span aria-hidden>💻</span>
              <button type="button" onClick={() => openDataset(data.local_active.current.dataset_id)}
                title="Open this dataset"
                className="text-content font-semibold text-sm hover:underline">
                {data.local_active.current.name || `Dataset #${data.local_active.current.dataset_id}`}
              </button>
              <span className="rounded border border-violet-400/40 bg-violet-500/10 px-1.5 py-0.5 text-violet-200 text-[0.625rem] uppercase">
                local · training
              </span>
              {data.local_active.error && (
                <span className="text-rose-300 text-[0.625rem]">{data.local_active.error}</span>
              )}
              <button type="button" onClick={() => openDataset(data.local_active.current.dataset_id)}
                className="ml-auto px-2 py-1 rounded-lg text-content-muted hover:text-content text-xs">
                Open dataset ↗
              </button>
            </div>
            <TrainingProgress datasetId={data.local_active.current.dataset_id} />
          </div>
        )}
        {!data ? (
          <p className="m-0 text-content-subtle text-sm">Loading…</p>
        ) : actives.length === 0 ? (
          !data.local_active && (
            <p className="m-0 text-content-subtle text-sm">
              No run in progress. Launch one from a dataset’s training panel.
            </p>
          )
        ) : (
          actives.map((run) => (
            <div key={run.run_id}
              className="flex flex-col gap-2 rounded-xl border border-sky-500/30 bg-sky-500/5 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => openDataset(run.dataset_id)}
                  title="Open this dataset"
                  className="text-content font-semibold text-sm hover:underline">
                  {run.dataset_name || run.run_name || `Dataset #${run.dataset_id}`}
                </button>
                <span className="rounded border border-border bg-surface px-1.5 py-0.5 text-content-muted text-[0.625rem] uppercase">
                  {famLabel(run.train_type)}
                </span>
                {run.version && (
                  <span className="rounded border border-border bg-surface px-1.5 py-0.5 text-content-subtle text-[0.625rem]"
                    title="Dataset version this run trains on">
                    v{run.version}
                  </span>
                )}
                <span className={`rounded border px-1.5 py-0.5 text-[0.625rem] ${statusStyle(run.status)}`}>
                  {run.status}
                </span>
                <span className="text-content-subtle text-[0.625rem]">{timeAgo(run.created_at)}</span>
                <span className="ml-auto text-content-muted text-[0.6875rem] tabular-nums">
                  {run.gpu ? `${run.gpu} · ` : ''}{run.price_per_hour != null ? `$${run.price_per_hour}/h · ` : ''}
                  ~${run.cost_estimate} so far
                </span>
              </div>

              <TrainingProgress datasetId={run.dataset_id} trainType={run.train_type} cloud />

              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => stop(run)} disabled={stopping[run.run_id]}
                  className="px-3 py-1.5 rounded-lg bg-red-600/80 text-white text-xs font-semibold disabled:opacity-40">
                  {stopping[run.run_id] ? 'Stopping…' : 'Stop run'}
                </button>
                {run.checkpoint_ready && (
                  <a href={checkpointHref(run)}
                    className="px-3 py-1.5 rounded-lg border border-emerald-400/40 bg-emerald-500/10 text-emerald-200 text-xs font-semibold no-underline">
                    ⬇ Download the LoRA
                  </a>
                )}
                <span className="ml-auto flex items-center gap-2">
                  {/* Per-run escape hatch to this pod's provider console (billing,
                      logs, manual destroy). The vast instance id, when known, goes
                      in the tooltip so it's findable in the console's instance list. */}
                  <a href="https://cloud.vast.ai/instances/" target="_blank" rel="noreferrer"
                    title={run.vast_instance_id
                      ? `vast.ai instance ${run.vast_instance_id} — provider console (billing, logs, manual destroy)`
                      : 'vast.ai console — billing, logs, manual destroy'}
                    className="px-2 py-1 rounded-lg text-sky-300 hover:text-sky-200 text-xs no-underline">
                    vast.ai console ↗
                  </a>
                  <button type="button" onClick={() => openDataset(run.dataset_id)}
                    className="px-2 py-1 rounded-lg text-content-muted hover:text-content text-xs">
                    Open dataset ↗
                  </button>
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* A pod kept alive for manual recovery bills until reaped — call it out. */}
      {recent.some((r) => r.status === 'error_pod_kept') && (
        <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-amber-200 text-xs">
          ⚠ A finished run kept its pod for manual checkpoint recovery — it keeps billing until reaped. Download its LoRA below, then it is cleaned up automatically after the recovery window.
        </div>
      )}

      {/* Recent history */}
      {recent.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <h2 className="m-0 text-content-muted text-xs font-semibold uppercase tracking-wide">
              Recent
            </h2>
            <button type="button"
              onClick={async () => {
                if (!window.confirm('Move the staging folders of all FINISHED runs to the trash?\n\nDataset copies, samples and checkpoint duplicates already imported. Active runs and pods kept for recovery are spared. Recoverable until you empty the trash in Settings.')) return;
                const d = await postJson('/api/dataset/train/cloud/purge', {});
                if (d.ok) toast.info(`Cleaned ${d.purged_runs} run(s) — ${(d.freed_bytes / 1e9).toFixed(1)} GB moved to the trash.`);
                poll();
              }}
              className="ml-auto px-2.5 py-1 rounded-lg bg-red-500/10 border border-red-500/30 text-red-200 text-xs font-semibold">
              🧹 Clean finished runs
            </button>
          </div>
          <div className="flex flex-col divide-y divide-border rounded-lg border border-border bg-surface">
            {recent.map((run, i) => (
              <div key={run.run_id ? `c${run.run_id}` : `l${run.dataset_id}-${run.created_at || i}`}
                className="flex flex-wrap items-center gap-2 px-3 py-2 text-sm">
                <span aria-hidden title={run.source === 'cloud' ? 'Cloud run (vast.ai)' : 'Local run'}>
                  {run.source === 'cloud' ? '☁️' : '💻'}
                </span>
                <button type="button" onClick={() => openDataset(run.dataset_id)}
                  className="text-content font-medium hover:underline">
                  {run.dataset_name || run.run_name || `Dataset #${run.dataset_id}`}
                </button>
                <span className="text-content-subtle text-[0.625rem] uppercase">{famLabel(run.train_type)}</span>
                {run.version && (
                  <span className="text-content-subtle text-[0.625rem]" title="Dataset version">v{run.version}</span>
                )}
                <span className={`rounded border px-1.5 py-0.5 text-[0.625rem] ${statusStyle(run.status)}`}>
                  {run.status}
                </span>
                <span className="text-content-subtle text-[0.625rem]">{timeAgo(run.finished_at || run.created_at)}</span>
                {run.error && (run.status === 'error' || run.status === 'error_pod_kept') && (
                  <span className="text-content-subtle text-[0.625rem] truncate max-w-[16rem]" title={run.error}>
                    — {run.error}
                  </span>
                )}
                <span className="ml-auto text-content-muted text-[0.6875rem] tabular-nums">
                  {run.gpu ? `${run.gpu} · ` : ''}${run.cost_estimate}
                </span>
                {run.checkpoint_ready && (
                  <a href={checkpointHref(run)}
                    className="px-2 py-1 rounded-lg border border-emerald-400/40 bg-emerald-500/10 text-emerald-200 text-xs font-semibold no-underline">
                    ⬇ LoRA
                  </a>
                )}
                {run.status === 'error' && (
                  <button type="button" onClick={() => retry(run)} disabled={!!retrying[run.run_id]}
                    title="Relaunch this run with the same settings on a fresh pod"
                    className="px-2 py-1 rounded-lg border border-primary/40 bg-primary/15 text-white text-xs font-semibold disabled:opacity-50">
                    {retrying[run.run_id] ? '↻ Retrying…' : '↻ Retry'}
                  </button>
                )}
                {run.source === 'cloud' && run.status === 'done' && run.checkpoint_ready && (
                  <button type="button" onClick={() => continueRun(run)} disabled={!!continuing[run.run_id]}
                    title="Resume training from this run's last checkpoint for more steps, on a fresh pod"
                    className="px-2 py-1 rounded-lg border border-sky-400/40 bg-sky-500/10 text-sky-200 text-xs font-semibold disabled:opacity-50">
                    {continuing[run.run_id] ? '▶ Continuing…' : '▶ Continue (+1000)'}
                  </button>
                )}
                {settingsLine(run) && (
                  <span className="w-full text-content-subtle text-[0.625rem]"
                    title="The effective ai-toolkit settings this launch used">
                    ⚙ {settingsLine(run)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
