import { useCallback, useEffect, useRef, useState } from 'react'
import { Cloud, Dumbbell, Eraser, FlaskConical, HardDrive, Monitor, Save } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router'
import { apiFetch, postJson } from '../../api/fetchClient.js'
import { useToast } from '../common/Toast.jsx'
import { requestHelpTip } from '../../help/helpTips.js'
import TrainingProgress from '../dataset/TrainingProgress.jsx'
import ContinueDialog from '../dataset/ContinueDialog.jsx'
import RunLineageTree from '../dataset/RunLineageTree.jsx'
import { BaseModelChip, DatasetVersionChip, RunIdChip } from '../dataset/RunIdentityBadges.jsx'
import { runIdentityOf, runRowDomId } from '../../utils/runIdentity.js'
import { runOpensGraph } from './runGraph.js'
import { canStopLocalRun, formatDuration, groupRunsByDataset, isTrainingRecipeReplayBlocked, retryRequest, runBaseModelLabel, runDurationSeconds, runRetryKey, trainingRunVariantLabel } from '../../utils/trainingRuns.js'
import { postWithConfirmations, RETRY_CONFIRMABLE_REFUSALS } from '../../utils/trainingRefusals.js'
import { podBootFailureView, uploadStallFailureView } from '../../utils/launchProgress.js'
import { denseContinueBlocker, isFullTransformerRun } from '../../utils/trainingMode.js'
import { runStagingCleanup } from '../../utils/stagingCleanup.js'
import { StatusBadge, timeAgo, famLabel, cardAccent, RunThumb, PodKeptNote, FullArtifactStatus, AutoRetryBadges, RecipeWarning, settingsLine, checkpointHref } from './RunHistoryAtoms.jsx'
import useRunsHubContinue from './useRunsHubContinue.js'
import { studioRunTarget } from '../../utils/studioRunNavigation.js'
import RunError from './RunError.jsx'
export { StatusBadge, timeAgo, famLabel, FullArtifactStatus, AutoRetryBadges, RecipeWarning, checkpointHref } from './RunHistoryAtoms.jsx'

const POLL_MS = 5000
const RECENT_COLLAPSED_KEY = 'cloudRunsRecentCollapsed'
const GROUPS_COLLAPSED_KEY = 'cloudRunsGroupsCollapsed'

// Public main's persisted history and local supervision. Optional execution
// lanes contribute controls and transports; no cloud request is implicit here.
export function RunsHub({ endpoint = '/api/dataset/train/runs', render = null, continuation = null }) {
  const toast = useToast()
  const navigate = useNavigate()
  const location = useLocation()
  const [data, setData] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const alive = useRef(true)
  const requestVersion = useRef(0)
  const [stoppingLocal, setStoppingLocal] = useState(false);
  // Recent-history depth. The 5 s poll stays light by default (15); "Load older
  // runs" bumps this on demand (backend caps the history at 100), so a long
  // history is opt-in rather than paid on every tick.
  const [historyLimit, setHistoryLimit] = useState(15);
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
    const version = ++requestVersion.current
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 15000)
    try {
      if (!/^\/api\/[a-zA-Z0-9_/-]+$/.test(endpoint)) throw new Error('Invalid training history endpoint.')
      const next = await apiFetch(endpoint + '?limit=' + historyLimit, { background: true, signal: controller.signal })
      if (!next || !Array.isArray(next.recent)) throw new Error('Invalid training history response.')
      if (alive.current && version === requestVersion.current) { setData(next); setLoadError(null) }
    } catch (error) { if (alive.current && version === requestVersion.current) setLoadError(error.message || 'Training history is unavailable.') }
    finally { clearTimeout(timeout) }
  }, [endpoint, historyLimit])
  useEffect(() => {
    alive.current = true
    let current = true, timer
    const tick = async () => { await poll(); if (current) timer = setTimeout(tick, POLL_MS) }
    tick()
    return () => { current = false; alive.current = false; requestVersion.current += 1; clearTimeout(timer) }
  }, [poll])
  // Nudge, once, that a finished run can be continued — resuming from an earlier,
  // less-cooked epoch is the flagship of the Continue dialog and easy to miss.
  useEffect(() => {
    const runs = [...(data?.actives || []), ...(data?.recent || [])];
    if (runs.some((r) => !isFullTransformerRun(r) && r.status === 'done' && r.checkpoint_ready)) {
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

  // Keep every Runs surface on the same Studio entry point: the dataset route
  // preselects this run's dataset without having to detour through the library.
  const openTestStudio = (id, family) => {
    const target = studioRunTarget(id, family);
    if (target) navigate(target);
  };

const stopLocal = async () => {
    const local = data?.local_active;
    if (!canStopLocalRun(local) || stoppingLocalRef.current) return;
    const who = local.current.name || `dataset #${local.current.dataset_id}`;
    if (!window.confirm(`Stop the local run for “${who}”?\n\n`
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
  const [retrying, setRetrying] = useState({})
const retry = async (run) => {
    if (run?.source !== 'local') return;
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
      const d = await postWithConfirmations(
        (body) => postJson(req.url, body), req.body,
        'Retry anyway (force)', RETRY_CONFIRMABLE_REFUSALS);
      if (!d) return;                              // declined at a confirm prompt
      toast.success(isLocal
        ? 'Run relaunched locally — watch it under In progress…'
        : 'Run relaunched — provisioning a fresh pod…');
      poll();
    } catch (e) {
      toast.error(e?.message
        ? `Could not retry this run: ${e.message}`
        : 'Could not retry this run. Please try again.');
    } finally {
      setRetrying((m) => ({ ...m, [key]: false }));
    }
  };
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
  const resume = useRunsHubContinue({ data, poll, cloud: continuation })
  const host = { data, poll, loadError, ...resume, recentCollapsed, setRecentCollapsed,
    groupsCollapsed, toggleGroup, historyLimit, setHistoryLimit, brokenThumbs, setBrokenThumbs,
    lineageOpen, lineageData, setLineageData, toggleLineage, jumpToRun, openDataset, openTestStudio,
    stopLocal, stoppingLocal, retry, retrying, shareConfig }
  return render ? render(host) : <RunsHubContent host={host} />
}

export function RunsHubContent({ host, cloud = null }) {
  const { data, loadError, recentCollapsed, setRecentCollapsed, groupsCollapsed, toggleGroup,
    historyLimit, setHistoryLimit, brokenThumbs, setBrokenThumbs, lineageOpen, lineageData,
    setLineageData, toggleLineage, jumpToRun, openDataset, openTestStudio, stopLocal,
    stoppingLocal, shareConfig, continuing = {}, canContinueRun, continueRun, continueFromCheckpoint, dialog } = host
  const { stagingSizes = {}, recheckFullDelivery, recheckingDelivery = {}, hubPresence = {},
    fetchFullModel, purgeRun, purgingRun = {} } = cloud || {}
  const retry = run => run.source === 'local' ? host.retry(run) : cloud?.retry?.(run)
  const retrying = { ...host.retrying, ...cloud?.retrying }
  const actives = data?.actives || []
  const recent = data?.recent || []
const renderRunCard = (run, i) => {
    const fullModel = isFullTransformerRun(run);
    const ident = runIdentityOf(run);
    const key = run.run_id ? `c${run.run_id}` : `l${run.record_id || `${run.dataset_id}-${run.created_at || i}`}`;
    const variantLabel = trainingRunVariantLabel(run.train_type, run.variant);
    const baseLabel = runBaseModelLabel(run);
    const duration = formatDuration(runDurationSeconds(run));
    const line = settingsLine(run);
    const thumbKey = run.share_key || key;
    const cleanup = runStagingCleanup(run, stagingSizes);
    return (
      <div key={key} id={ident ? runRowDomId(ident.source, ident.id) : undefined}
        className={`flex gap-2.5 sm:gap-3 rounded-lg border border-border border-l-2 bg-app/40 p-2.5 ${cardAccent(run.status)}`}>
        <RunThumb run={run} broken={!!brokenThumbs[thumbKey]}
          onBroken={() => setBrokenThumbs((m) => ({ ...m, [thumbKey]: true }))} />
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            {ident ? (
              <RunIdChip source={run.source === 'cloud' ? 'cloud' : 'local'}
                recordId={run.record_id} cloudId={run.source === 'cloud' ? run.run_id : null} />
            ) : (
              <span aria-hidden title={run.source === 'cloud' ? 'Cloud run (vast.ai)' : 'Local run'}>
                {run.source === 'cloud' ? <Cloud aria-hidden="true" className="h-3.5 w-3.5" /> : <Monitor aria-hidden="true" className="h-3.5 w-3.5" />}
              </span>
            )}
            <button type="button" onClick={() => openDataset(run.dataset_id)}
              title="Open this dataset"
              className="max-w-full truncate text-content text-sm font-semibold hover:underline">
              {run.dataset_name || run.run_name || `Dataset #${run.dataset_id}`}
            </button>
            <StatusBadge status={run.status} />
            {fullModel && (
              <span className="rounded border border-sky-400/40 bg-sky-500/10 px-1.5 py-0.5 text-sky-100 text-[0.625rem] font-semibold uppercase">
                full model · experimental
              </span>
            )}
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
            {!fullModel && run.resumed_from != null && (
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
            {!fullModel && run.source === 'cloud' && run.saves > 0 && (
              <span className="tabular-nums" title="Checkpoints this run saved (synced locally)">
                <Save aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />{run.saves} save{run.saves > 1 ? 's' : ''}
              </span>
            )}
            {/* What this run still costs in DISK — the figure a targeted cleanup
                needs. Absent when its staging is already gone (nothing to show,
                and no 🧹 either). */}
            {cleanup.size && (
              <span className="tabular-nums text-content-subtle"
                title="Disk this run's staging folder still holds (dataset copy, samples, logs)">
                <HardDrive aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />{cleanup.size} on disk
              </span>
            )}
            {run.gpu && <span>{run.gpu}</span>}
            {run.cost_estimate != null && (
              <span className="tabular-nums" title="Estimated cost (price/h × run time)">
                ${run.cost_estimate}
              </span>
            )}
          </div>
          {/* NOT truncate: these messages carry their explanation on the SECOND line
              ("Cannot access gated repo … ask for access"), so collapsing them to one
              line kept the useless "403 Client Error (Request ID…)" and hid the part
              that names what to fix. The full text was only in title=, which never
              shows on a phone — where this was reported. Newlines are real, hence
              whitespace-pre-line; clamped so a stack trace cannot take over the page. */}
          {run.error && (run.status === 'error' || run.status === 'error_pod_kept') && (
            <RunError error={run.error} dedicatedToken={run.source === 'cloud' && fullModel} />
          )}
          {/* The boot timeout said in full: the raw error names the timer but
              never what became of the machine that was rented, which is the one
              thing worth knowing before relaunching. */}
          {(() => {
            // Same shape for both launch teardowns: what the machine did, and
            // what became of it. They are mutually exclusive (each matches its
            // own error string), so the first one that answers is rendered.
            const failure = podBootFailureView(run) || uploadStallFailureView(run);
            return failure && (
              <div className="rounded border border-amber-400/40 bg-amber-500/10 px-2 py-1.5 text-amber-100 text-[0.6875rem] leading-snug">
                <div className="font-semibold">{failure.title}</div>
                <p className="m-0 mt-0.5 break-words text-amber-200/90">{failure.message}</p>
              </div>
            );
          })()}
          {fullModel && (
            <FullArtifactStatus run={run} onRecheck={recheckFullDelivery}
              rechecking={!!recheckingDelivery[run.run_id]}
              presence={hubPresence[run.run_id] || null}
              onFetch={fetchFullModel} fetching={!!run.dense_fetch_active} />
          )}
          {line && (
            <p className="m-0 truncate text-content-subtle text-[0.625rem]"
              title="The effective ai-toolkit settings this launch used">
              ⚙ {line}
            </p>
          )}
          <RecipeWarning run={run} />
          {run.status === 'error_pod_kept' && <PodKeptNote fullModel={fullModel} />}
          <div className="mt-0.5 flex flex-wrap items-center gap-2">
            {run.status === 'error' && (run.source === 'local' || typeof cloud?.retry === 'function') && (
              <button type="button" onClick={() => retry(run)}
                disabled={isTrainingRecipeReplayBlocked(run) || !!retrying[runRetryKey(run)]}
                title={isTrainingRecipeReplayBlocked(run)
                  ? 'Disabled: this legacy/incompatible Z-Image recipe cannot be replayed safely; start a fresh run'
                  : run.source === 'local'
                    ? 'Relaunch this run locally with the same settings'
                    : 'Relaunch this run with the same settings on a fresh pod'}
                className="px-3 py-1.5 rounded-lg bg-primary/90 hover:bg-primary text-gray-950 text-xs font-semibold disabled:opacity-40">
                {retrying[runRetryKey(run)] ? '↻ Retrying…' : '↻ Retry'}
              </button>
            )}
            {/* A LoRA is continued from a checkpoint on this disk; a full model
                is continued from EITHER its Hugging Face copy or the copy on
                this computer, which is what `resume_steps` carries for a dense
                run. The dialog prices both roads before the click, because a
                26 GB upload bills a rented GPU for every minute it takes. */}
            {typeof continueRun === 'function' && canContinueRun?.(run) && (fullModel
              ? (run.resume_steps || []).length > 0
              : (run.status === 'done' && run.checkpoint_ready)) && (
              <button type="button" onClick={() => continueRun(run)}
                disabled={isTrainingRecipeReplayBlocked(run) || !!continuing[run.run_id]
                  || !!(fullModel && denseContinueBlocker(run, hubPresence[run.run_id]))}
                title={isTrainingRecipeReplayBlocked(run)
                  ? 'Disabled: this legacy/incompatible Z-Image checkpoint cannot be continued safely; start a fresh run'
                  : fullModel
                    ? (denseContinueBlocker(run, hubPresence[run.run_id])
                      || "Resume this full model on a fresh pod — pick how its 26 GB gets there")
                    : "Resume from any of this run's checkpoints for more steps, on a fresh pod"}
                className="px-3 py-1.5 rounded-lg bg-sky-600/80 hover:bg-sky-600 text-white text-xs font-semibold disabled:opacity-40">
                {continuing[run.run_id] ? '▶ Continuing…' : '▶ Continue…'}
              </button>
            )}
            {!fullModel && run.checkpoint_ready && (
              <a href={checkpointHref(run)}
                title="Download this run's LoRA checkpoint"
                className="px-3 py-1.5 rounded-lg bg-emerald-600/80 hover:bg-emerald-600 text-white text-xs font-semibold no-underline">
                ⬇ LoRA
              </a>
            )}
            {!fullModel && run.dataset_id != null && (
              <button type="button" onClick={() => openTestStudio(run.dataset_id, run.train_type)}
                title="Open Test Studio with this run's dataset selected"
                className="rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-2 py-1 text-indigo-100 hover:bg-indigo-500/20 text-xs font-semibold">
                <FlaskConical aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Test in Studio
              </button>
            )}
            {/* The graph opens for ANY run with saved checkpoints (a single run
                already shows its epochs), every local run included (runGraph.js),
                and labels as Lineage once it has a parent or a branch. */}
            {runOpensGraph(run, { fullModel }) && (
              <button type="button" onClick={() => toggleLineage(run.record_id)}
                aria-expanded={!!lineageOpen[run.record_id]}
                title={run.lineage
                  ? "Show this run's lineage — the runs it continued from or that branched off it"
                  : "Show this run's checkpoints as a graph — import / generate / download / continue from any of them"}
                className={'rounded-lg border px-2 py-1 text-xs font-semibold transition-colors '
                  + (lineageOpen[run.record_id]
                    ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-100 '
                    : 'border-indigo-400/40 bg-indigo-500/10 text-indigo-200 hover:bg-indigo-500/20 ')}>
                {lineageOpen[run.record_id]
                  ? (run.lineage ? '🌳 Hide lineage' : '◉ Hide graph')
                  : (run.lineage ? '🌳 Lineage' : '◉ Graph')}
              </button>
            )}
            {run.share_key && (
              <button type="button" onClick={() => shareConfig(run)}
                title="Download this run's full settings as a paste-safe text file (recipe / help thread)"
                className="ml-auto rounded-lg border border-transparent px-2 py-1 text-content-muted hover:border-border hover:text-content text-xs font-medium">
                ⎘ Share config
              </button>
            )}
            {/* Per-run cleanup, so a long history no longer forces the all-or-
                nothing purge. Only shown when there IS something to move and the
                run is not spared (active pod, kept pod) — the same rule the
                global 🧹 applies, read from runStagingCleanup. */}
            {cleanup.available && typeof purgeRun === 'function' && (
              <button type="button" onClick={() => purgeRun(run)}
                disabled={!!purgingRun[run.run_id]}
                title={cleanup.title}
                className={`rounded-lg border border-red-500/30 bg-red-500/10 px-2 py-1 text-red-200 hover:bg-red-500/20 text-xs font-semibold disabled:opacity-40 ${run.share_key ? '' : 'ml-auto'}`}>
                <Eraser aria-hidden="true" className="mr-1 inline h-3 w-3 align-[-1px]" />{purgingRun[run.run_id] ? 'Cleaning…' : `Clean ${cleanup.size}`}
              </button>
            )}
          </div>
          {runOpensGraph(run, { fullModel }) && lineageOpen[run.record_id] && (
            <RunLineageTree
              tree={lineageData[run.record_id]?.tree}
              loading={lineageData[run.record_id]?.loading}
              error={lineageData[run.record_id]?.error}
              onSelect={jumpToRun}
              continueSource={run.source}
              onContinueCheckpoint={canContinueRun?.(run) ? continueFromCheckpoint : undefined}
              refetchTree={async () => {
                const r = await fetch(`/api/dataset/train/runs/${run.record_id}/lineage`, { credentials: 'include' });
                if (!r.ok) throw new Error('unavailable');
                const tree = await r.json();
                setLineageData((m) => ({ ...m, [run.record_id]: { tree } }));
                return tree;
              }} />
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
            <Dumbbell aria-hidden="true" className="h-4 w-4" /> Training runs
          </h1>
          {cloud?.headerExtra}
        </div>
        <p className="m-0 text-content-muted text-sm">Training progress and saved runs, with the settings each launch used.</p>
      </header>
      {loadError && <p role="alert" className="text-sm text-rose-300">{loadError}</p>}
      {cloud?.notice}
      {cloud?.summary}
      <div className="flex flex-col gap-3">
        <h2 className="m-0 text-content-muted text-xs font-semibold uppercase tracking-wide">In progress</h2>
        {data?.local_active?.current && (
          <div id={runRowDomId('local', data.local_active.record_id)}
            className="flex flex-col gap-2 rounded-xl border border-violet-500/30 bg-violet-500/5 p-3">
            <div className="flex flex-wrap items-center gap-2">
              {data.local_active.record_id != null
                ? <RunIdChip source="local" recordId={data.local_active.record_id} />
                : <Monitor aria-hidden="true" className="h-3.5 w-3.5" />}
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
                {data.local_active.current.dataset_id != null && (
                  <button type="button" onClick={() => openTestStudio(data.local_active.current.dataset_id, data.local_active.current.train_type)}
                    title="Open Test Studio with this run's dataset selected"
                    className="px-2 py-1 rounded-lg text-indigo-200 hover:bg-indigo-500/10 hover:text-indigo-100 text-xs font-semibold">
                    <FlaskConical aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Test in Studio
                  </button>
                )}
              </span>
            </div>
            <RecipeWarning run={{ ...data.local_active, ...data.local_active.current }} />
            <TrainingProgress datasetId={data.local_active.current.dataset_id}
              base={data.local_active.current.base_model}
              trainType={data.local_active.current.train_type}
              variant={data.local_active.current.variant} />
          </div>
        )}

        {!data && !loadError && <p className="text-sm text-content-subtle">Loading…</p>}
        {data && !data.local_active && !actives.length && <p className="text-sm text-content-subtle">No run in progress. Launch one from a dataset’s training panel.</p>}
        {cloud?.activeContent || actives.map(renderRunCard)}
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
            {!recentCollapsed && cloud?.historyAction}

          </div>
          {!recentCollapsed && (
          <div className="flex flex-col gap-3">
            {groupRunsByDataset(recent).map((group, gi) => {
              const gkey = String(group.datasetId);
              const collapsed = !!groupsCollapsed[gkey];
              const head = group.runs[0];
              const name = head.dataset_name || head.run_name || `Dataset #${group.datasetId}`;
              const testRun = group.runs.find((run) => !isFullTransformerRun(run));
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
                    {testRun && group.datasetId != null && (
                      <button type="button" onClick={() => openTestStudio(group.datasetId, testRun.train_type)}
                        title="Open Test Studio with this run's dataset selected"
                        className="whitespace-nowrap rounded-lg px-2 py-0.5 text-indigo-200 hover:bg-indigo-500/10 hover:text-indigo-100 text-[0.6875rem] font-semibold">
                        <FlaskConical aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Test in Studio
                      </button>
                    )}
                  </div>
                  {!collapsed && (
                    <div className="flex flex-col gap-2 px-2 pb-2">
                      {group.runs.map((run, i) => renderRunCard(run, i))}
                    </div>
                  )}
                </section>
              );
            })}
            {recent.length >= historyLimit && historyLimit < 100 && (
              <button type="button"
                onClick={() => setHistoryLimit((n) => Math.min(n + 25, 100))}
                title="The list keeps only the most recent runs to stay light; load older ones on demand."
                className="self-center mt-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-content-muted hover:text-content text-xs font-semibold">
                Load older runs
              </button>
            )}
          </div>
          )}
        </div>
      )}


      {dialog && <ContinueDialog {...dialog} context={famLabel(dialog.target?.train_type)} />}
      {cloud?.dialogs}
    </section>
  )
}
