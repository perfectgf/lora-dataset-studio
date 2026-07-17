import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import TrainingProgress from '../components/dataset/TrainingProgress';
import {
  canStopLocalRun,
  isTrainingRecipeReplayBlocked,
  retryRequest,
  runRetryKey,
  trainingRunVariantLabel,
} from '../utils/trainingRuns';

/* Dedicated hub for cloud training runs across ALL datasets: watch the ones in
   progress (live progress + samples), stop them, and download finished LoRAs —
   without hunting through each dataset's panel. Polls the aggregate
   /train/cloud/runs endpoint (actives + recent history + budget summary). */

const POLL_MS = 5000;
const FAMILY_LABEL = { zimage: 'Z-Image', krea: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein' };

// "Recent" history collapse: a UI preference, not run data — persisted globally
// (same lazy-init + effect pattern as `datasetGridTileSize` in DatasetGrid.jsx /
// `datasetGenerator` in VariationCatalog.jsx). Default open = today's behavior.
const RECENT_COLLAPSED_KEY = 'cloudRunsRecentCollapsed';

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

function AutoRetryBadges({ run }) {
  return (
    <>
      {run.auto_retry_of != null && (
        <span
          className="rounded border border-sky-400/40 bg-sky-500/10 px-1.5 py-0.5 text-sky-200 text-[0.625rem]"
          title={`Automatic retry of cloud run #${run.auto_retry_of}`}>
          ↻ automatic retry {run.auto_retry_count || 1}/1
        </span>
      )}
      {run.auto_retry_run_id != null && (
        <span
          className="rounded border border-violet-400/40 bg-violet-500/10 px-1.5 py-0.5 text-violet-200 text-[0.625rem]"
          title={`Automatically relaunched as cloud run #${run.auto_retry_run_id}`}>
          ↻ auto-retried as #{run.auto_retry_run_id}
        </span>
      )}
    </>
  );
}

function RecipeWarning({ run }) {
  if (!run.recipe_warning) return null;
  const replayBlocked = isTrainingRecipeReplayBlocked(run);
  return (
    <div role="alert"
      className="w-full rounded-md border border-amber-400/40 bg-amber-500/10 px-2.5 py-2 text-amber-200 text-[0.6875rem] leading-relaxed">
      <span className="font-semibold">⚠ Z-Image recipe warning:</span> {run.recipe_warning}
      {replayBlocked && (
        <span className="font-semibold"> Retry and Continue are disabled; start a fresh validated run.</span>
      )}
    </div>
  );
}

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
    trainingRunVariantLabel(run.train_type, run.variant),
    run.masked === false ? 'unmasked' : 'masked',
  ].filter(Boolean).join(' · ');
}

function checkpointHref(run) {
  const qs = new URLSearchParams();
  if (run.train_type) qs.set('train_type', run.train_type);
  if (run.variant) qs.set('variant', run.variant);
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
  const [stoppingLocal, setStoppingLocal] = useState(false);
  // React disables the button on the next render. The ref also closes the tiny
  // gap before that render, so a fast double-click cannot send two kill calls.
  const stoppingLocalRef = useRef(false);
  const [recentCollapsed, setRecentCollapsed] = useState(() => {
    try { return localStorage.getItem(RECENT_COLLAPSED_KEY) === '1'; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem(RECENT_COLLAPSED_KEY, recentCollapsed ? '1' : '0'); } catch { /* ignore — private mode */ }
  }, [recentCollapsed]);

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

  const stopLocal = async () => {
    const local = data?.local_active;
    if (!canStopLocalRun(local) || stoppingLocalRef.current) return;
    const who = local.current.name || `dataset #${local.current.dataset_id}`;
    if (!window.confirm(`Stop the local run for « ${who} »?\n\n`
      + 'The training process is terminated and the pending local training queue is cleared. '
      + 'Checkpoints already saved remain available.')) return;

    stoppingLocalRef.current = true;
    setStoppingLocal(true);
    try {
      const d = await postJson('/api/dataset/train/stop', {
        dataset_id: local.current.dataset_id,
        run_token: local.current.run_token,
      });
      if (d.ok === false) {
        toast.error(d.error || 'Could not stop the local run — it may have already finished.');
        return;
      }
      // The stop endpoint is synchronous: once it answers, the process is gone
      // and the backend flag is clear. Remove the live card immediately instead
      // of waiting up to POLL_MS for the next refresh.
      setData((current) => current ? { ...current, local_active: null } : current);
      toast.success('Local training stopped — ComfyUI is re-enabled.');
    } catch (error) {
      toast.error(error?.message
        ? `Could not stop the local run: ${error.message}`
        : 'Could not stop the local run. Please try again.');
    } finally {
      await poll();
      stoppingLocalRef.current = false;
      setStoppingLocal(false);
    }
  };

  // ↻ Retry of a failed run: exact same settings as the failed launch
  // (steps/variant/family/masked, + GPU class for cloud). Cloud runs replay
  // their pod params on a fresh pod; a LOCAL run replays its stamped provenance
  // record through launch_training (normal preflight, GPU-collision refusal).
  const [retrying, setRetrying] = useState({});      // runRetryKey -> bool
  const retry = async (run) => {
    if (isTrainingRecipeReplayBlocked(run)) {
      toast.error('This run uses an incompatible legacy Z-Image recipe. Start a fresh validated run instead.');
      return;
    }
    const req = retryRequest(run);
    if (!req) return;
    const isLocal = run.source === 'local';
    const key = runRetryKey(run);
    setRetrying((m) => ({ ...m, [key]: true }));
    try {
      const d = await postJson(req.url, req.body);
      if (d.ok === false) toast.error(d.error || 'Retry failed');
      else toast.success(isLocal
        ? 'Run relaunched locally — watch it under In progress…'
        : 'Run relaunched — provisioning a fresh pod…');
      poll();
    } finally {
      setRetrying((m) => ({ ...m, [key]: false }));
    }
  };

  // ▶ Continue a finished cloud run: fresh pod, same settings, resuming from the
  // run's last harvested checkpoint for `extra` more steps (ai-toolkit
  // auto-resume — the monitor seeds the checkpoint onto the pod before start).
  const [continuing, setContinuing] = useState({});   // run_id -> bool
  const continueRun = async (run) => {
    if (isTrainingRecipeReplayBlocked(run)) {
      toast.error('This checkpoint uses an incompatible legacy Z-Image recipe and cannot be continued safely.');
      return;
    }
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

  // ⎘ Share config: download a paste-safe .txt of every setting this launch
  // sent to ai-toolkit (recipe sharing / help threads). Fetch-then-blob so a
  // 404/500 surfaces as a toast instead of navigating to an error page.
  const shareConfig = async (run) => {
    if (!run.share_key) return;
    try {
      const r = await fetch(`/api/dataset/train/runs/${encodeURIComponent(run.share_key)}/share`,
        { credentials: 'include' });
      if (!r.ok) { toast.error('Could not build the config file — please retry.'); return; }
      const blob = await r.blob();
      const cd = r.headers.get('Content-Disposition') || '';
      const m = /filename="?([^"]+)"?/.exec(cd);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = m ? m[1] : 'lds-config.txt';
      a.click();
      URL.revokeObjectURL(a.href);
    } catch {
      toast.error('Could not download the config file.');
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
              <span className="ml-auto flex items-center gap-2">
                {canStopLocalRun(data.local_active) && (
                  <button type="button" onClick={stopLocal} disabled={stoppingLocal}
                    title="Stop this local training process; checkpoints already saved are kept"
                    className="px-3 py-1 rounded-lg bg-red-600/80 text-white text-xs font-semibold disabled:opacity-40">
                    {stoppingLocal ? 'Stopping…' : 'Stop run'}
                  </button>
                )}
                {data.local_active.share_key && (
                  <button type="button" onClick={() => shareConfig(data.local_active)}
                    title="Download this run's full settings as a paste-safe text file (recipe / help thread)"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-xs font-semibold">
                    ⎘ Share config
                  </button>
                )}
                <button type="button" onClick={() => openDataset(data.local_active.current.dataset_id)}
                  className="px-2 py-1 rounded-lg text-content-muted hover:text-content text-xs">
                  Open dataset ↗
                </button>
              </span>
            </div>
            <RecipeWarning run={{ ...data.local_active, ...data.local_active.current }} />
            <TrainingProgress datasetId={data.local_active.current.dataset_id}
              base={data.local_active.current.base_model}
              trainType={data.local_active.current.train_type}
              variant={data.local_active.current.variant} />
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
                <AutoRetryBadges run={run} />
                <span className="text-content-subtle text-[0.625rem]">{timeAgo(run.created_at)}</span>
                <span className="ml-auto text-content-muted text-[0.6875rem] tabular-nums">
                  {run.gpu ? `${run.gpu} · ` : ''}{run.price_per_hour != null ? `$${run.price_per_hour}/h · ` : ''}
                  ~${run.cost_estimate} so far
                </span>
              </div>

              <RecipeWarning run={run} />
              <TrainingProgress datasetId={run.dataset_id} trainType={run.train_type} variant={run.variant} cloud />

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
                {run.share_key && (
                  <button type="button" onClick={() => shareConfig(run)}
                    title="Download this run's full settings as a paste-safe text file (recipe / help thread)"
                    className="px-2 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-xs font-semibold">
                    ⎘ Share config
                  </button>
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
            <h2 className="m-0">
              <button type="button" onClick={() => setRecentCollapsed((v) => !v)}
                aria-expanded={!recentCollapsed}
                className="flex items-center gap-1.5 text-content-muted hover:text-content text-xs font-semibold uppercase tracking-wide">
                <span aria-hidden className="text-[0.625rem] leading-none">{recentCollapsed ? '▸' : '▾'}</span>
                Recent{recent.length ? ` (${recent.length})` : ''}
                <span className="sr-only">{recentCollapsed ? ' — collapsed' : ' — expanded'}</span>
              </button>
            </h2>
            {!recentCollapsed && (
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
            )}
          </div>
          {!recentCollapsed && (
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
                <AutoRetryBadges run={run} />
                <span className="text-content-subtle text-[0.625rem]">{timeAgo(run.finished_at || run.created_at)}</span>
                {run.error && (run.status === 'error' || run.status === 'error_pod_kept') && (
                  <span className="text-content-subtle text-[0.625rem] truncate max-w-[16rem]" title={run.error}>
                    — {run.error}
                  </span>
                )}
                <span className="ml-auto text-content-muted text-[0.6875rem] tabular-nums">
                  {run.gpu ? `${run.gpu} · ` : ''}{run.cost_estimate != null ? `$${run.cost_estimate}` : ''}
                </span>
                {run.checkpoint_ready && (
                  <a href={checkpointHref(run)}
                    className="px-2 py-1 rounded-lg border border-emerald-400/40 bg-emerald-500/10 text-emerald-200 text-xs font-semibold no-underline">
                    ⬇ LoRA
                  </a>
                )}
                {run.status === 'error' && (
                  <button type="button" onClick={() => retry(run)}
                    disabled={isTrainingRecipeReplayBlocked(run) || !!retrying[runRetryKey(run)]}
                    title={isTrainingRecipeReplayBlocked(run)
                      ? 'Disabled: this legacy/incompatible Z-Image recipe cannot be replayed safely; start a fresh run'
                      : run.source === 'local'
                        ? 'Relaunch this run locally with the same settings'
                        : 'Relaunch this run with the same settings on a fresh pod'}
                    className="px-2 py-1 rounded-lg border border-primary/40 bg-primary/15 text-white text-xs font-semibold disabled:opacity-50">
                    {retrying[runRetryKey(run)] ? '↻ Retrying…' : '↻ Retry'}
                  </button>
                )}
                {run.source === 'cloud' && run.status === 'done' && run.checkpoint_ready && (
                  <button type="button" onClick={() => continueRun(run)}
                    disabled={isTrainingRecipeReplayBlocked(run) || !!continuing[run.run_id]}
                    title={isTrainingRecipeReplayBlocked(run)
                      ? 'Disabled: this legacy/incompatible Z-Image checkpoint cannot be continued safely; start a fresh run'
                      : "Resume training from this run's last checkpoint for more steps, on a fresh pod"}
                    className="px-2 py-1 rounded-lg border border-sky-400/40 bg-sky-500/10 text-sky-200 text-xs font-semibold disabled:opacity-50">
                    {continuing[run.run_id] ? '▶ Continuing…' : '▶ Continue (+1000)'}
                  </button>
                )}
                {run.share_key && (
                  <button type="button" onClick={() => shareConfig(run)}
                    title="Download this run's full settings as a paste-safe text file (recipe / help thread)"
                    className="px-2 py-1 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-xs font-semibold">
                    ⎘ Share config
                  </button>
                )}
                {settingsLine(run) && (
                  <span className="w-full text-content-subtle text-[0.625rem]"
                    title="The effective ai-toolkit settings this launch used">
                    ⚙ {settingsLine(run)}
                  </span>
                )}
                <RecipeWarning run={run} />
              </div>
            ))}
          </div>
          )}
        </div>
      )}
    </section>
  );
}
