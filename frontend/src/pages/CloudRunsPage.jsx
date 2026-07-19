import { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import TrainingProgress from '../components/dataset/TrainingProgress';
import ContinueDialog from '../components/dataset/ContinueDialog';
import RunLineageTree from '../components/dataset/RunLineageTree';
import { BaseModelChip, DatasetVersionChip, RunIdChip } from '../components/dataset/RunIdentityBadges';
import { HelpBadge } from '../help/HelpMode';
import { requestHelpTip } from '../help/helpTips';
import { runIdentityOf, runRowDomId } from '../utils/runIdentity';
import {
  canStopLocalRun,
  formatDuration,
  groupRunsByDataset,
  isTrainingRecipeReplayBlocked,
  retryRequest,
  runBaseModelLabel,
  runDurationSeconds,
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
// Per-dataset collapse of the Recent GROUPS (a JSON map dataset_id -> 1),
// persisted like the section collapse above so the fold survives reloads.
const GROUPS_COLLAPSED_KEY = 'cloudRunsGroupsCollapsed';

const STATUS_STYLE = {
  done: 'text-emerald-300 border-emerald-400/40 bg-emerald-500/10',
  error: 'text-rose-300 border-rose-400/40 bg-rose-500/10',
  error_pod_kept: 'text-amber-200 border-amber-400/40 bg-amber-500/10',
  stopped: 'text-content-muted border-border bg-surface',
};
const statusStyle = (s) =>
  STATUS_STYLE[s] || 'text-sky-300 border-sky-400/40 bg-sky-500/10';

// Outcome words on the history cards (raw pipeline statuses read like logs).
// Active phases (preparing/training/syncing…) pass through untranslated.
const STATUS_LABEL = {
  done: 'done',
  error: 'failed',
  error_pod_kept: 'failed · pod kept',
  stopped: 'stopped',
};

// Left accent bar of a history card — the strongest at-a-glance status signal.
const CARD_ACCENT = {
  done: 'border-l-emerald-400/70',
  error: 'border-l-rose-400/70',
  error_pod_kept: 'border-l-amber-400/70',
  stopped: 'border-l-border-strong',
};
const cardAccent = (s) => CARD_ACCENT[s] || 'border-l-border';

/** Strong status pill (dot + word) — rank-1 information on every card. */
function StatusBadge({ status }) {
  if (!status) return null;
  return (
    <span className={'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 '
      + `text-[0.625rem] font-semibold uppercase tracking-wide ${statusStyle(status)}`}>
      <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" />
      {STATUS_LABEL[status] || status}
    </span>
  );
}

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

// Short family names that fit the 5rem fallback thumbnail tile.
const FAMILY_SHORT = { zimage: 'Z-Image', krea: 'Krea', sdxl: 'SDXL', flux: 'FLUX', flux2klein: 'Klein' };

/** Card thumbnail: the LAST sample the run generated (backend stamps
 * `preview_url` when one exists on disk). Fallback: a quiet family tile —
 * runs that never sampled (crashed early, purged staging) stay scannable. */
function RunThumb({ run, broken, onBroken }) {
  if (run.preview_url && !broken) {
    return (
      <a href={run.preview_url} target="_blank" rel="noreferrer"
        title="Last sample this run generated (open full size)"
        className="relative block h-16 w-16 sm:h-20 sm:w-20 shrink-0 overflow-hidden rounded-lg border border-border hover:border-indigo-400">
        <img src={run.preview_url} loading="lazy" onError={onBroken}
          alt={`Last training sample of ${run.dataset_name || run.run_name || 'this run'}`}
          className="h-full w-full object-cover" />
      </a>
    );
  }
  return (
    <div aria-hidden
      className="flex h-16 w-16 sm:h-20 sm:w-20 shrink-0 flex-col items-center justify-center gap-1 rounded-lg border border-border bg-app/60 text-content-subtle">
      <span className="text-base opacity-50">🖼</span>
      <span className="px-1 text-center text-[0.5625rem] uppercase tracking-wide leading-tight">
        {FAMILY_SHORT[run.train_type] || 'LoRA'}
      </span>
    </div>
  );
}

/** error_pod_kept billing warning — INSIDE the concerned card (it used to be
 * an orphan full-width banner above the whole history). */
function PodKeptNote() {
  return (
    <div role="alert"
      className="w-full rounded-md border border-amber-400/40 bg-amber-500/10 px-2.5 py-2 text-amber-200 text-[0.6875rem] leading-relaxed">
      <span className="font-semibold">⚠ Pod kept for manual checkpoint recovery</span> — it keeps
      billing until reaped. Download its LoRA, then it is cleaned up automatically after the
      recovery window.
    </div>
  );
}

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
   predate the snapshot feature. Steps and variant are NOT repeated here —
   they are promoted to the card's metrics row. */
function settingsLine(run) {
  const s = run.settings;
  if (!s) return null;
  return [
    s.rank ? `rank ${s.rank}${s.alpha ? `/${s.alpha}` : ''}` : null,
    Array.isArray(s.resolution) ? `${s.resolution.join('+')} px` : null,
    s.save_every ? `save ${s.save_every}` : null,
    s.optimizer && s.optimizer !== 'adamw8bit' ? s.optimizer : null,
    s.lr_scheduler || null,
    s.dropout ? `dropout ${s.dropout}` : null,
    s.timestep_type || null,
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
  const location = useLocation();
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
  // Per-dataset group folds inside Recent — same persistence pattern.
  const [groupsCollapsed, setGroupsCollapsed] = useState(() => {
    try {
      const m = JSON.parse(localStorage.getItem(GROUPS_COLLAPSED_KEY) || '{}');
      return m && typeof m === 'object' ? m : {};
    } catch { return {}; }
  });
  useEffect(() => {
    try { localStorage.setItem(GROUPS_COLLAPSED_KEY, JSON.stringify(groupsCollapsed)); } catch { /* ignore — private mode */ }
  }, [groupsCollapsed]);
  const toggleGroup = (datasetId) => setGroupsCollapsed((m) => {
    const key = String(datasetId);
    const next = { ...m };
    if (next[key]) delete next[key];
    else next[key] = 1;
    return next;
  });
  // Thumbnails whose image 404'd/broke since load — fall back to the family
  // tile instead of a broken-image glyph. Keyed by the run's share_key.
  const [brokenThumbs, setBrokenThumbs] = useState({});

  // 🌳 Lineage: which run cards have their genealogy tree expanded, and the
  // fetched tree per record id (loaded lazily on first expand; refetched only
  // if forced). Keyed by record_id — the universal run node key.
  const [lineageOpen, setLineageOpen] = useState({});   // record_id -> bool
  const [lineageData, setLineageData] = useState({});    // record_id -> {tree|error|loading}
  const loadLineage = useCallback(async (recordId) => {
    setLineageData((m) => ({ ...m, [recordId]: { loading: true } }));
    try {
      const r = await fetch(`/api/dataset/train/runs/${recordId}/lineage`, { credentials: 'include' });
      if (!r.ok) throw new Error('unavailable');
      const tree = await r.json();
      setLineageData((m) => ({ ...m, [recordId]: { tree } }));
    } catch {
      setLineageData((m) => ({ ...m, [recordId]: { error: 'Could not load this run’s lineage.' } }));
    }
  }, []);
  const toggleLineage = useCallback((recordId) => {
    setLineageOpen((m) => {
      const next = { ...m, [recordId]: !m[recordId] };
      if (next[recordId] && !lineageData[recordId]) loadLineage(recordId);
      return next;
    });
  }, [lineageData, loadLineage]);
  // Jump from a tree node to that run's card (same page): scroll + brief flash,
  // reusing the deep-link highlight the Checkpoints panel already uses.
  const jumpToRun = useCallback((node) => {
    const id = runRowDomId(node.source, node.source === 'cloud' ? node.run_id : node.record_id);
    if (!id) return;
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.classList.add('lds-run-flash');
    setTimeout(() => el.classList.remove('lds-run-flash'), 2200);
  }, []);

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

  // Nudge, once, that a finished run can be continued — resuming from an earlier,
  // less-cooked epoch is the flagship of the Continue dialog and easy to miss.
  useEffect(() => {
    const runs = [...(data?.actives || []), ...(data?.recent || [])];
    if (runs.some((r) => r.status === 'done' && r.checkpoint_ready)) {
      requestHelpTip('continue-any-epoch');
    }
  }, [data]);

  // Deep-link from the Checkpoints panel's "View in Runs ↗": /cloud#run-cloud-49
  // scrolls to and briefly highlights that run's card. Runs after data arrives
  // (the cards must exist). A card hidden by the Recent fold or its dataset
  // group fold is expanded first, then found on the re-render. Flashes ONCE
  // per navigation (location.key) — not again on every 5 s poll.
  const flashedRef = useRef(null);
  useEffect(() => { flashedRef.current = null; }, [location.key]);
  useEffect(() => {
    const id = (location.hash || '').replace(/^#/, '');
    if (!id || !data || flashedRef.current === id) return undefined;
    const el = document.getElementById(id);
    if (!el) {
      const run = (data.recent || []).find((r) => {
        const ident = runIdentityOf(r);
        return ident && runRowDomId(ident.source, ident.id) === id;
      });
      if (run) {
        setRecentCollapsed(false);
        setGroupsCollapsed((m) => {
          const key = String(run.dataset_id);
          if (!m[key]) return m;
          const next = { ...m };
          delete next[key];
          return next;
        });
      }
      return undefined;
    }
    flashedRef.current = id;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.classList.add('lds-run-flash');
    const to = setTimeout(() => el.classList.remove('lds-run-flash'), 2200);
    return () => clearTimeout(to);
  }, [location.hash, location.key, data, recentCollapsed, groupsCollapsed]);

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
  const [continueRunTarget, setContinueRunTarget] = useState(null);   // run being continued | null
  const continueRun = (run) => {
    if (isTrainingRecipeReplayBlocked(run)) {
      toast.error('This checkpoint uses an incompatible legacy Z-Image recipe and cannot be continued safely.');
      return;
    }
    setContinueRunTarget(run);
  };
  const submitContinue = async (payload) => {
    const run = continueRunTarget;
    setContinueRunTarget(null);
    if (!run || !payload) return;
    setContinuing((m) => ({ ...m, [run.run_id]: true }));
    try {
      const d = await postJson('/api/dataset/train/cloud/continue',
        { run_id: run.run_id, extra_steps: payload.extraSteps,
          from_step: payload.fromStep, overrides: payload.overrides });
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

  /* One HISTORY card. Visual hierarchy: rank 1 = thumbnail + identity chip +
     name + a strong status pill; rank 2 = the metrics that matter (duration,
     steps, saves, GPU, cost); rank 3 = the de-emphasized settings line. Every
     per-run warning (Z-Image legacy recipe, kept pod billing) renders INSIDE
     its card. Primary actions are filled buttons, Share config stays ghost. */
  const renderRunCard = (run, i) => {
    const ident = runIdentityOf(run);
    const key = run.run_id ? `c${run.run_id}` : `l${run.record_id || `${run.dataset_id}-${run.created_at || i}`}`;
    const variantLabel = trainingRunVariantLabel(run.train_type, run.variant);
    const baseLabel = runBaseModelLabel(run);
    const duration = formatDuration(runDurationSeconds(run));
    const line = settingsLine(run);
    const thumbKey = run.share_key || key;
    return (
      <div key={key} id={ident ? runRowDomId(ident.source, ident.id) : undefined}
        className={`flex gap-2.5 sm:gap-3 rounded-lg border border-border border-l-2 bg-app/40 p-2.5 ${cardAccent(run.status)}`}>
        <RunThumb run={run} broken={!!brokenThumbs[thumbKey]}
          onBroken={() => setBrokenThumbs((m) => ({ ...m, [thumbKey]: true }))} />
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            {ident ? (
              <RunIdChip source={ident.source} id={ident.id} />
            ) : (
              <span aria-hidden title={run.source === 'cloud' ? 'Cloud run (vast.ai)' : 'Local run'}>
                {run.source === 'cloud' ? '☁️' : '💻'}
              </span>
            )}
            <button type="button" onClick={() => openDataset(run.dataset_id)}
              title="Open this dataset"
              className="max-w-full truncate text-content text-sm font-semibold hover:underline">
              {run.dataset_name || run.run_name || `Dataset #${run.dataset_id}`}
            </button>
            <StatusBadge status={run.status} />
            <AutoRetryBadges run={run} />
            <span className="ml-auto whitespace-nowrap text-content-subtle text-[0.625rem]">
              {timeAgo(run.finished_at || run.created_at)}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.6875rem] text-content-muted">
            <span className="text-[0.625rem] uppercase tracking-wide">
              {famLabel(run.train_type)}{variantLabel ? ` · ${variantLabel}` : ''}
            </span>
            {/* Official bases are already spelled by the family·variant above;
                only a CUSTOM base adds new info here (which checkpoint file). */}
            {baseLabel?.custom && <BaseModelChip label={baseLabel} />}
            <DatasetVersionChip version={run.version} />
            {run.resumed_from != null && (
              <button type="button"
                onClick={() => run.record_id != null && toggleLineage(run.record_id)}
                title="This run resumed from an earlier checkpoint — open its lineage"
                className="rounded border border-border px-1 py-0.5 text-content-subtle text-[0.5625rem] hover:text-content">
                ↳ from step {run.resumed_from}
              </button>
            )}
            {duration && (
              <span className="tabular-nums" title="Wall-clock run duration (launch → finish)">
                ⏱ {duration}
              </span>
            )}
            {run.steps ? <span className="tabular-nums">{run.steps} steps</span> : null}
            {run.source === 'cloud' && run.saves > 0 && (
              <span className="tabular-nums" title="Checkpoints this run saved (synced locally)">
                💾 {run.saves} save{run.saves > 1 ? 's' : ''}
              </span>
            )}
            {run.gpu && <span>{run.gpu}</span>}
            {run.cost_estimate != null && (
              <span className="tabular-nums" title="Estimated cost (price/h × run time)">
                ${run.cost_estimate}
              </span>
            )}
          </div>
          {run.error && (run.status === 'error' || run.status === 'error_pod_kept') && (
            <p className="m-0 truncate text-rose-300/90 text-[0.6875rem]" title={run.error}>
              {run.error}
            </p>
          )}
          {line && (
            <p className="m-0 truncate text-content-subtle text-[0.625rem]"
              title="The effective ai-toolkit settings this launch used">
              ⚙ {line}
            </p>
          )}
          <RecipeWarning run={run} />
          {run.status === 'error_pod_kept' && <PodKeptNote />}
          <div className="mt-0.5 flex flex-wrap items-center gap-2">
            {run.status === 'error' && (
              <button type="button" onClick={() => retry(run)}
                disabled={isTrainingRecipeReplayBlocked(run) || !!retrying[runRetryKey(run)]}
                title={isTrainingRecipeReplayBlocked(run)
                  ? 'Disabled: this legacy/incompatible Z-Image recipe cannot be replayed safely; start a fresh run'
                  : run.source === 'local'
                    ? 'Relaunch this run locally with the same settings'
                    : 'Relaunch this run with the same settings on a fresh pod'}
                className="px-3 py-1.5 rounded-lg bg-primary/90 hover:bg-primary text-white text-xs font-semibold disabled:opacity-40">
                {retrying[runRetryKey(run)] ? '↻ Retrying…' : '↻ Retry'}
              </button>
            )}
            {run.source === 'cloud' && run.status === 'done' && run.checkpoint_ready && (
              <button type="button" onClick={() => continueRun(run)}
                disabled={isTrainingRecipeReplayBlocked(run) || !!continuing[run.run_id]}
                title={isTrainingRecipeReplayBlocked(run)
                  ? 'Disabled: this legacy/incompatible Z-Image checkpoint cannot be continued safely; start a fresh run'
                  : "Resume from any of this run's checkpoints for more steps, on a fresh pod"}
                className="px-3 py-1.5 rounded-lg bg-sky-600/80 hover:bg-sky-600 text-white text-xs font-semibold disabled:opacity-40">
                {continuing[run.run_id] ? '▶ Continuing…' : '▶ Continue…'}
              </button>
            )}
            {run.checkpoint_ready && (
              <a href={checkpointHref(run)}
                title="Download this run's LoRA checkpoint"
                className="px-3 py-1.5 rounded-lg bg-emerald-600/80 hover:bg-emerald-600 text-white text-xs font-semibold no-underline">
                ⬇ LoRA
              </a>
            )}
            {run.lineage && run.record_id != null && (
              <button type="button" onClick={() => toggleLineage(run.record_id)}
                aria-expanded={!!lineageOpen[run.record_id]}
                title="Show this run's lineage — the runs it continued from or that branched off it"
                className="rounded-lg border border-transparent px-2 py-1 text-content-muted hover:border-border hover:text-content text-xs font-medium">
                {lineageOpen[run.record_id] ? '🌳 Hide lineage' : '🌳 Lineage'}
              </button>
            )}
            {run.share_key && (
              <button type="button" onClick={() => shareConfig(run)}
                title="Download this run's full settings as a paste-safe text file (recipe / help thread)"
                className="ml-auto rounded-lg border border-transparent px-2 py-1 text-content-muted hover:border-border hover:text-content text-xs font-medium">
                ⎘ Share config
              </button>
            )}
          </div>
          {run.lineage && run.record_id != null && lineageOpen[run.record_id] && (
            <RunLineageTree
              tree={lineageData[run.record_id]?.tree}
              loading={lineageData[run.record_id]?.loading}
              error={lineageData[run.record_id]?.error}
              onSelect={jumpToRun} />
          )}
        </div>
      </div>
    );
  };

  return (
    <section className="flex flex-col gap-5">
      <header className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="m-0 flex items-center gap-2 text-content text-xl font-bold">
            <span><span aria-hidden>🏋️</span> Training runs</span>
            <HelpBadge topic="page-cloud" />
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
          <div id={runRowDomId('local', data.local_active.record_id)}
            className="flex flex-col gap-2 rounded-xl border border-violet-500/30 bg-violet-500/5 p-3">
            <div className="flex flex-wrap items-center gap-2">
              {data.local_active.record_id != null
                ? <RunIdChip source="local" id={data.local_active.record_id} />
                : <span aria-hidden>💻</span>}
              <button type="button" onClick={() => openDataset(data.local_active.current.dataset_id)}
                title="Open this dataset"
                className="text-content font-semibold text-sm hover:underline">
                {data.local_active.current.name || `Dataset #${data.local_active.current.dataset_id}`}
              </button>
              <span className="rounded border border-violet-400/40 bg-violet-500/10 px-1.5 py-0.5 text-violet-200 text-[0.625rem] uppercase">
                local · training
              </span>
              {/* A live local run with no custom base IS the family's official
                  base — coerce the absent value so it spells out, not blanks. */}
              <BaseModelChip label={runBaseModelLabel({
                base_model: data.local_active.current.base_model || '',
                train_type: data.local_active.current.train_type,
                variant: data.local_active.current.variant,
              })} />
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
            <div key={run.run_id} id={runRowDomId('cloud', run.run_id)}
              className="flex flex-col gap-2 rounded-xl border border-sky-500/30 bg-sky-500/5 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <RunIdChip source="cloud" id={run.run_id} />
                <button type="button" onClick={() => openDataset(run.dataset_id)}
                  title="Open this dataset"
                  className="text-content font-semibold text-sm hover:underline">
                  {run.dataset_name || run.run_name || `Dataset #${run.dataset_id}`}
                </button>
                <span className="rounded border border-border bg-surface px-1.5 py-0.5 text-content-muted text-[0.625rem] uppercase">
                  {famLabel(run.train_type)}
                </span>
                {/* No variant shown on the active card, so spell the base in
                    full here — official ("Z-Image Turbo") and custom alike. */}
                <BaseModelChip label={runBaseModelLabel(run)} />
                <DatasetVersionChip version={run.version} />
                <StatusBadge status={run.status} />
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

      {/* Recent history — one card per run, grouped by dataset. The pod-kept
          billing warning lives INSIDE the concerned card (PodKeptNote), no
          longer as an orphan full-width banner here. */}
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
            {/* the fold must not hide an active billing warning entirely */}
            {recentCollapsed && recent.some((r) => r.status === 'error_pod_kept') && (
              <span className="text-amber-300 text-[0.6875rem]">
                ⚠ a kept pod is still billing — expand for details
              </span>
            )}
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
          <div className="flex flex-col gap-3">
            {groupRunsByDataset(recent).map((group, gi) => {
              const gkey = String(group.datasetId);
              const collapsed = !!groupsCollapsed[gkey];
              const head = group.runs[0];
              const name = head.dataset_name || head.run_name || `Dataset #${group.datasetId}`;
              return (
                <section key={`g${gi}-${gkey}`}
                  className="flex flex-col rounded-xl border border-border bg-surface">
                  {/* discreet group header: the dataset these consecutive runs share */}
                  <div className="flex items-center gap-2 px-3 py-2">
                    <button type="button" onClick={() => toggleGroup(group.datasetId)}
                      aria-expanded={!collapsed}
                      title={collapsed ? 'Show the runs of this dataset' : 'Fold the runs of this dataset'}
                      className="flex min-w-0 items-center gap-1.5 text-content-muted hover:text-content text-xs">
                      <span aria-hidden className="text-[0.625rem] leading-none">{collapsed ? '▸' : '▾'}</span>
                      <span className="truncate font-semibold text-content">{name}</span>
                      <span className="whitespace-nowrap text-content-subtle">
                        · {group.runs.length} run{group.runs.length > 1 ? 's' : ''}
                      </span>
                    </button>
                    {collapsed && group.runs.some((r) => r.status === 'error_pod_kept') && (
                      <span className="whitespace-nowrap text-amber-300 text-[0.625rem]">⚠ kept pod billing</span>
                    )}
                    <button type="button" onClick={() => openDataset(group.datasetId)}
                      className="ml-auto whitespace-nowrap rounded-lg px-2 py-0.5 text-content-muted hover:text-content text-[0.6875rem]">
                      Open dataset ↗
                    </button>
                  </div>
                  {!collapsed && (
                    <div className="flex flex-col gap-2 px-2 pb-2">
                      {group.runs.map((run, i) => renderRunCard(run, i))}
                    </div>
                  )}
                </section>
              );
            })}
          </div>
          )}
        </div>
      )}

      {continueRunTarget && (
        <ContinueDialog
          context={`${famLabel(continueRunTarget.train_type)}${
            trainingRunVariantLabel(continueRunTarget.train_type, continueRunTarget.variant)
              ? ` · ${trainingRunVariantLabel(continueRunTarget.train_type, continueRunTarget.variant)}` : ''}`}
          where="cloud"
          checkpoints={((continueRunTarget.resume_steps?.length
            ? continueRunTarget.resume_steps
            : [continueRunTarget.steps]).filter(Boolean)).map((step) => ({ step }))}
          busy={!!continuing[continueRunTarget.run_id]}
          onResolve={submitContinue} />
      )}
    </section>
  );
}
