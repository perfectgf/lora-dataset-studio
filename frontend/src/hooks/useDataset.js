/**
 * useDataset — Face Dataset Maker data hook.
 * Loads the dataset list + the open dataset payload, polls while generation
 * jobs are pending, and exposes all mutations (create/ref/generate/import/
 * classify/caption/status/caption-edit/crop/regenerate/export).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { getCsrfToken, fetchWithCsrfRetry, CSRF_EXPIRED_MESSAGE, putJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { useJobs } from '../context/JobsContext';
import { serializeWatermarkRegions } from '../utils/watermarkRegions';
import { summarizeScrapeImport } from '../utils/smallImageRescue';
import { trainingRunSelection } from '../utils/checkpointBrowser';
import {
  normalizeTrainingMode,
  trainingModeSettingsPayload,
} from '../utils/trainingMode.js';
import { refreshDatasetIfActive } from '../utils/datasetRefresh';
import { ENGINE_LABELS } from '../components/dataset/engineSelection.js';
import { retryRequestForReferenceEdit } from '../components/dataset/referenceEdit.js';
import { classifyResultMessage } from '../components/dataset/classifyFramingGate.js';
import { captionResultSuffix, captionSkippedSuffix } from '../utils/captionEngines.js';

function post(url, body, isForm) {
  // Routes through the shared fetchWithCsrfRetry: a token that aged out mid-session
  // (WTF_CSRF_TIME_LIMIT) is refreshed and the request replayed once, exactly like
  // apiFetch — so a long-lived dataset page no longer starts failing every mutation
  // with a cryptic HTML 400 until a hard refresh.
  return fetchWithCsrfRetry(url, {
    method: 'POST',
    headers: isForm
      ? { 'X-CSRFToken': getCsrfToken() }
      : { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken() },
    body: isForm ? body : JSON.stringify(body || {}),
  });
}

/**
 * GET + parse, swallowing every failure to null. For polling a background-job
 * status where a transient blip should just mean "try again next tick", never a
 * toast — apiFetch's global error handling is deliberately bypassed here.
 */
async function getJsonSilent(url) {
  try {
    const r = await fetch(url, { credentials: 'include' });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

/**
 * POST + defensive JSON parse (I1 + I4). Never throws: returns the parsed
 * payload on success, `{ok:false, error}` on HTTP / network / non-JSON
 * failures, so every caller can surface a toast instead of failing silently
 * or crashing on `.json()` of an HTML error page.
 * Exported for reuse by the dataset-adjacent hooks (useLoraTestStudio).
 */
export async function postJson(url, body, isForm) {
  try {
    const r = await post(url, body, isForm);
    let d = null;
    let parsed = false;
    try { d = await r.json(); parsed = true; } catch { /* non-JSON body (proxy page, empty) */ }
    // Preserve any structured fields the error body carries (e.g. `studio_missing`,
    // `klein_missing`) so callers can render an itemized banner, not just a toast.
    if (!r.ok) {
      // A 400 that STILL isn't our JSON envelope after the shared retry = a CSRF
      // token that aged out mid-session → actionable message, not "Server error (400)".
      const fallback = (!parsed && r.status === 400)
        ? CSRF_EXPIRED_MESSAGE : `Server error (${r.status})`;
      return { ...(d || {}), ok: false, error: (d && d.error) || fallback };
    }
    return d || { ok: true };
  } catch (e) {
    return { ok: false, error: e.message || 'Network error' };
  }
}

export function faceScoringErrorMessage(scoringError) {
  const { kind, detail } = scoringError || {};
  if (kind === 'unavailable') {
    return 'Face scoring is not installed — run the Quality tools step in Setup.';
  }
  if (kind === 'subject_not_photographic') return detail || 'Face scoring is unavailable for this dataset.';
  if (kind === 'busy') {
    return 'Face scoring is already running. Wait for the current image to finish, then try again.';
  }
  if (kind === 'ref_unusable') return detail
    ? `The reference photo is not usable for scoring: ${detail}`
    : 'The reference photo is not usable for scoring.';
  return detail ? `Face scoring failed: ${detail}` : 'Face scoring failed.';
}

/**
 * Compose the 🧽 Clean summary toast from the server's counts — PURE (no React,
 * no toast) so the honest-message logic is testable on its own.
 * Response shape: {cropped, inpainted, inpainted_klein, needs_review, failed, skipped, error}.
 *
 * The old code fired TWO toasts at once: a "Nothing to clean" SUCCESS (it only
 * looked at cropped/inpainted/needs_review/failed) AND a separate "N skipped"
 * WARNING — so a run that skipped 64 images for inpainting showed a green
 * "Nothing to clean" next to the amber warning. Now: one honest toast.
 *   - nothing detected at all              -> "Nothing to clean" (success)
 *   - anything skipped (needs the inpaint  -> single warning summary
 *     install)
 *   - otherwise                            -> single success summary
 * `error` is a separate concern (why an attempted inpaint failed) and is
 * surfaced by its own toast.error at the call site.
 */
export function summarizeClean(d) {
  const cropped = d.cropped || 0;
  // LaMa and Klein inpaints tally together — both "repainted the mark" from the
  // user's point of view (the batch method toggle picks which engine ran).
  const inpainted = (d.inpainted || 0) + (d.inpainted_klein || 0);
  const skipped = d.skipped || 0;
  const needsReview = d.needs_review || 0;
  const failed = d.failed || 0;
  if (!cropped && !inpainted && !skipped && !needsReview && !failed) {
    return { severity: 'success', message: 'Nothing to clean' };
  }
  const parts = [];
  if (cropped) parts.push(`${cropped} cropped`);
  if (inpainted) parts.push(`${inpainted} inpainted`);
  if (skipped) parts.push(`${skipped} waiting for inpainting (⬇ install it)`);
  if (needsReview) parts.push(`${needsReview} need manual review`);
  if (failed) parts.push(`${failed} failed`);
  return { severity: skipped ? 'warning' : 'success', message: parts.join(' · ') };
}

export function useDataset() {
  const toast = useToast();
  const [datasets, setDatasets] = useState([]);
  // Persist the open dataset so a page reload returns to its workspace, not the list.
  const [currentId, setCurrentIdState] = useState(() => {
    try { const v = localStorage.getItem('datasetCurrentId'); return v ? Number(v) : null; }
    catch { return null; }
  });
  // State updates render asynchronously, but request freshness cannot wait for a
  // render. Every navigation path goes through this setter so refresh(A) can see
  // a switch to B immediately and discard A's eventual response.
  const currentIdRef = useRef(currentId);
  const setCurrentId = useCallback((next) => {
    const resolved = typeof next === 'function' ? next(currentIdRef.current) : next;
    currentIdRef.current = resolved;
    setCurrentIdState(resolved);
  }, []);
  const [data, setData] = useState(null);
  // Local request locks are Dataset-scoped. A slow request in A may keep A
  // protected after navigation, but it must not disable unrelated Dataset B;
  // token ownership also prevents A's late finally from unlocking a newer B run.
  const [busyRuns, setBusyRuns] = useState(() => new Map());
  const busyRunsRef = useRef(new Map());
  const busy = busyRuns.has(String(currentId));
  // Tracks an in-flight captioning pass by Dataset + opaque run token. A slow
  // response from A must neither show Captioning on B after navigation nor clear
  // a newer run that already started on B.
  const [captioningRuns, setCaptioningRuns] = useState(() => new Map());
  const captioningRunsRef = useRef(new Map());
  const beginCaptioningRun = useCallback((datasetId) => {
    const run = { datasetId, token: Symbol('caption-run') };
    captioningRunsRef.current.set(String(datasetId), run.token);
    setCaptioningRuns(new Map(captioningRunsRef.current));
    return run;
  }, []);
  const finishCaptioningRun = useCallback((run) => {
    const key = String(run.datasetId);
    if (captioningRunsRef.current.get(key) !== run.token) return;
    captioningRunsRef.current.delete(key);
    setCaptioningRuns(new Map(captioningRunsRef.current));
  }, []);
  const [localActivityRuns, setLocalActivityRuns] = useState(() => new Map());
  const localActivityRunsRef = useRef(new Map());
  const beginLocalActivityRun = useCallback((kind, datasetId) => {
    const run = { kind, datasetId, token: Symbol(`${kind}-run`) };
    localActivityRunsRef.current.set(`${kind}:${datasetId}`, run.token);
    setLocalActivityRuns(new Map(localActivityRunsRef.current));
    return run;
  }, []);
  const finishLocalActivityRun = useCallback((run) => {
    const key = `${run.kind}:${run.datasetId}`;
    if (localActivityRunsRef.current.get(key) !== run.token) return;
    localActivityRunsRef.current.delete(key);
    setLocalActivityRuns(new Map(localActivityRunsRef.current));
  }, []);
  // WHO wrote the captions of the last pass: {datasetId, captioned, engines}. The
  // default 'auto' backend chains JoyCaption and the Ollama vision model, which write
  // in visibly different styles, and the app used to report only a count — so
  // "these captions read nothing like yesterday's" had no answer anywhere. The
  // dataset id rides along so the line can never describe another dataset's run; it
  // is in-session only (deliberately NOT persisted — see the follow-up note in
  // utils/captionEngines.js).
  const [lastCaptionRun, setLastCaptionRun] = useState(null);
  // Per-image cache-bust versions (M1): only the cropped image reloads,
  // plus a separate version counter for the reference photo.
  const [nonces, setNonces] = useState({});
  // A horizontal mirror rewrites one image in place. Keep its busy state scoped
  // to that image so other tiles remain usable, and keep a synchronous ref guard
  // so a rapid double-click cannot enqueue the same rewrite twice before React
  // has rendered the disabled button.
  const [mirroringIds, setMirroringIds] = useState(() => new Set());
  const mirroringRef = useRef(new Set());
  // Per-image re-caption in flight (Identity-leak panel's targeted 🔄): keep the busy
  // state scoped to the offending row so the rest of the panel stays usable, with a
  // synchronous ref guard against a double-click enqueuing the same image twice.
  const [recaptioningIds, setRecaptioningIds] = useState(() => new Set());
  const recaptioningRef = useRef(new Set());
  // Face scoring is GPU-heavy: allow one tile at a time and use a synchronous
  // ref guard so rapid clicks cannot queue additional InsightFace work.
  const [scoringFaceIds, setScoringFaceIds] = useState(() => new Set());
  const scoringFaceRef = useRef(new Set());
  const [refNonce, setRefNonce] = useState(0);
  const pollRef = useRef(null);
  // A retry must replay the exact request, including File objects temporarily
  // attached to the modal. Keep those bytes in memory only for this browser
  // session: persisting them in storage would be surprising and can leak space.
  const referenceEditRetryRef = useRef(new Map());
  // A ref retains transient File objects; this state tick makes availability
  // reactive when a queued edit cannot be refreshed safely.
  const [, bumpReferenceEditRetryRevision] = useState(0);

  const fetchList = useCallback(async () => {
    try {
      const r = await fetch('/api/dataset/list', { credentials: 'include' });
      if (r.ok) setDatasets((await r.json()).datasets || []);
    } catch { /* transient network error — keep the last list */ }
  }, []);

  const refresh = useCallback(async (id) => {
    const dsId = id ?? currentIdRef.current;
    return refreshDatasetIfActive({
      datasetId: dsId,
      getActiveDatasetId: () => currentIdRef.current,
      request: (requestedId) => fetch(`/api/dataset/${requestedId}`, { credentials: 'include' }),
      commitData: setData,
      // Only an ACTIVE dataset's definitive 404 ejects back to the list.
      // Transient errors and stale responses keep the current workspace (M4).
      clearActiveDataset: () => setCurrentId(null),
    });
  }, [setCurrentId]);

  useEffect(() => { fetchList(); }, [fetchList]);

  // Persist the open dataset id + restore its workspace on mount/reload.
  useEffect(() => {
    try {
      if (currentId) localStorage.setItem('datasetCurrentId', String(currentId));
      else localStorage.removeItem('datasetCurrentId');
    } catch { /* ignore */ }
  }, [currentId]);
  useEffect(() => { if (currentId) refresh(currentId); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  // Navbar title = home: closes the open workspace even when already on /datasets
  // (same-route NavLink clicks don't remount the page).
  useEffect(() => {
    const goHome = () => setCurrentId(null);
    window.addEventListener('lds:home', goHome);
    return () => window.removeEventListener('lds:home', goHome);
  }, []);

  // Mirror in-flight dataset generations into the global JobsContext so the
  // floating jobs dock shows (and can cancel) them like other generations.
  // Depend on the STABLE upsert/remove callbacks (not the whole context value).
  const { upsert: gUpsert, remove: gRemove } = useJobs();
  const syncedRef = useRef(new Set());
  useEffect(() => {
    const inflight = (data?.images || []).filter(
      (i) => i.status === 'pending' && !i.filename && i.job_id);
    const ids = new Set();
    for (const img of inflight) {
      ids.add(img.job_id);
      gUpsert({
        jobId: img.job_id, type: 'image', status: 'processing',
        label: `Dataset · ${img.variation_label || 'face'}`,
        prompt: img.variation_label || '',
      });
    }
    for (const old of syncedRef.current) if (!ids.has(old)) gRemove(old);
    syncedRef.current = ids;
  }, [data, gUpsert, gRemove]);
  // Retract on unmount (leaving the page) — polling stops, so don't strand them.
  useEffect(() => () => {
    for (const id of syncedRef.current) gRemove(id);
    syncedRef.current = new Set();
  }, [gRemove]);

  // Poll while generation jobs are still pending (no filename yet).
  useEffect(() => {
    const pending = (data?.images || []).some((i) => i.status === 'pending' && !i.filename);
    if (pending && currentId) {
      pollRef.current = setInterval(() => refresh(currentId), 4000);
      return () => clearInterval(pollRef.current);
    }
    return undefined;
  }, [data, currentId, refresh]);

  // Poll every 2s while a captioning pass is running so captions appear live.
  useEffect(() => {
    if (!captioningRuns.has(String(currentId)) || !currentId) return undefined;
    const id = setInterval(() => refresh(currentId), 2000);
    return () => clearInterval(id);
  }, [captioningRuns, currentId, refresh]);

  // Same poller for a watermark scan, and it is not decoration: `hasActivity`
  // below only starts once a refresh has ALREADY seen activity ≠ null, and
  // findWatermarks does not refresh until the pass ends. So in the tab that
  // launched the scan the "Scanning… N/M" counter never moved and a ⏹ Stop
  // button in the banner would never appear at all.
  useEffect(() => {
    if (!localActivityRuns.has(`watermark:${currentId}`) || !currentId) return undefined;
    const id = setInterval(() => refresh(currentId), 2000);
    return () => clearInterval(id);
  }, [localActivityRuns, currentId, refresh]);

  // Persistence layer: a server-side batch (watermark detect/clean, caption,
  // re-caption, analyze faces, classify) advertises itself in the payload's
  // `activity` field. Whenever it's non-null — INCLUDING after a page reload that
  // dropped the local captioning/analyzing/watermarking flags — poll the dataset
  // every ~3.5s to track progress and detect the end. Keyed on the boolean
  // `hasActivity` (not the activity object, whose identity changes each fetch) so
  // the interval isn't torn down and rebuilt on every poll; it stops the moment
  // `activity` clears (the following refresh brings the final state; the completion
  // toast can't be restored — accepted, only the visual state is).
  const currentActivity = (data && String(data.id) === String(currentId))
    ? (data.activity || null)
    : null;
  const hasActivity = !!currentActivity;
  useEffect(() => {
    if (!hasActivity || !currentId) return undefined;
    const id = setInterval(() => refresh(currentId), 3500);
    return () => clearInterval(id);
  }, [hasActivity, currentId, refresh]);

  const open = useCallback(async (id) => { setCurrentId(id); await refresh(id); }, [refresh]);

  const create = useCallback(async (name, trigger, kind, conceptDesc, trainType, fidelity) => {
    const d = await postJson('/api/dataset/create',
      { name, trigger_word: trigger, ...(kind ? { kind } : {}),
        ...(trainType ? { train_type: trainType } : {}),
        ...(fidelity ? { fidelity } : {}),
        ...(kind === 'concept' && conceptDesc ? { concept_desc: conceptDesc } : {}) });
    if (d.ok) { await fetchList(); await open(d.id); toast.success('Dataset created'); }
    else toast.error(d.error || 'Unexpected error');
  }, [fetchList, open, toast]);

  // Face-only <-> full-body fidelity (character datasets). Future captions ban
  // permanent body marks too; composition target and import default follow.
  const setDatasetFidelity = useCallback(async (fidelity) => {
    const d = await postJson(`/api/dataset/${currentId}/fidelity`, { fidelity });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success(fidelity === 'body'
      ? 'Body fidelity ON — re-caption to apply to existing captions'
      : 'Back to face-only fidelity');
    await refresh();
  }, [currentId, refresh, toast]);

  // Change the target model family later (from the TrainingPanel selector) so the
  // grouped menu re-sorts. Return the server result so TrainingPanel can keep its
  // family/preset controls locked until persistence and the dependent refreshes
  // are complete.
  const setDatasetTrainType = useCallback(async (trainType) => {
    if (!currentId) return { ok: false, error: 'No dataset selected' };
    const d = await postJson(`/api/dataset/${currentId}/train-type`, { train_type: trainType });
    if (d.ok) await Promise.all([fetchList(), refresh(currentId)]);
    return d;
  }, [currentId, fetchList, refresh]);

  // Edit name / trigger / (concept) description / KIND after creation. A trigger change
  // needs no re-caption (prepended at export) but DOES rename what the dataset already
  // produced on disk, since the trigger is the naming key — the reply's trigger_rename
  // says how many files moved, or that a name clash blocked it. The dataset NAME is
  // display-only and never touches disk. A concept-desc change resets the avoid-list → the
  // toast nudges a re-caption (same contract as fidelity). A kind change flips the
  // caption strategy and the visible panels (server refuses with 409 while work is
  // in progress → the !ok branch surfaces the message) and nudges a re-caption.
  // Refreshes both views. prompt_suffix / prompt_suffixes (creative direction) ride
  // along: applied at generation time only, '' / {} clears, absent leaves untouched.
  const updateSettings = useCallback(async ({
    name, trigger_word, concept_desc, kind, prompt_suffix, prompt_suffixes, subject_type,
  }, opts = {}) => {
    if (!currentId) return { ok: false };
    const d = await postJson(`/api/dataset/${currentId}/settings`,
      { name, trigger_word, concept_desc, kind, prompt_suffix, prompt_suffixes, subject_type });
    if (!d.ok) { toast.error(d.error || 'Could not save settings'); return d; }
    // quiet: the generation panel persists suffix edits silently right before a
    // batch (the "Generating…" state is the feedback); the modal stays verbose.
    if (!opts.quiet) {
      const renamed = d.trigger_rename;
      if (renamed && !renamed.ok) {
        // The new trigger already owns files on disk, so nothing was moved rather
        // than half of it — say so, because the old artefacts keep the old name.
        toast.warning('Trigger word saved, but the artefacts it already produced could '
          + 'not be renamed: another dataset already uses that name on disk. They keep '
          + 'the old name.');
      } else if (renamed && renamed.files > 0) {
        toast.success(`Settings saved — ${renamed.files} file${renamed.files > 1 ? 's' : ''} `
          + 'renamed to follow the new trigger word (LoRAs, run folder, export)');
      } else if (d.kind_changed) {
        toast.success(`Kind changed to ${d.kind} — re-caption to apply the new caption style to existing captions`);
      } else {
        toast.success(d.concept_desc_changed
          ? 'Saved — concept changed; re-caption to apply it to existing captions'
          : 'Settings saved');
      }
    }
    await refresh();
    fetchList();
    return d;
  }, [currentId, refresh, fetchList, toast]);

  const deleteDataset = useCallback(async (id) => {
    const d = await postJson(`/api/dataset/${id}/delete`);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success('Dataset deleted');
    if (currentIdRef.current === id) { setCurrentId(null); setData(null); }
    await fetchList();
  }, [fetchList, setCurrentId, toast]);

  // Run a GPU-bound action exclusively (I2): re-entrancy guard + busy flag.
  // A second call while one is in flight is dropped instead of double-firing.
  const wrap = useCallback(async (fn, datasetId = currentIdRef.current) => {
    const key = String(datasetId);
    if (busyRunsRef.current.has(key)) return undefined;
    const token = Symbol('dataset-request');
    busyRunsRef.current.set(key, token);
    setBusyRuns(new Map(busyRunsRef.current));
    try { return await fn(); }
    finally {
      if (busyRunsRef.current.get(key) === token) {
        busyRunsRef.current.delete(key);
        setBusyRuns(new Map(busyRunsRef.current));
      }
    }
  }, []);

  const setRef = useCallback((file, { autoCrop = false } = {}) => wrap(async () => {
    const fd = new FormData(); fd.append('file', file);
    // Auto head-crop is OPT-IN (vision pass, pauses ComfyUI). Default: instant
    // centered crop, then the user adjusts with ✂ Crop (reads the full original).
    if (autoCrop) fd.append('crop', '1');
    const d = await postJson(`/api/dataset/${currentId}/ref`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    // GUARD-RAIL: the backend head-crop can silently fall back to a centered crop
    // (e.g. vision model not pulled). Surface its reason instead of a plain success.
    if (d.warning) toast.warning(d.warning);
    else toast.success(autoCrop ? 'Reference set (auto head-crop)' : 'Reference set — adjust with ✂ Crop if needed');
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  // Références ADDITIONNELLES (Nano Banana multi-références). Pas de fenêtre GPU
  // côté backend (normalisation WEBP simple) mais on garde wrap() pour l'anti-
  // double-clic pendant l'upload.
  const addExtraRef = useCallback((file) => wrap(async () => {
    const fd = new FormData(); fd.append('file', file);
    const d = await postJson(`/api/dataset/${currentId}/ref/extra`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success('Extra reference added');
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  const removeExtraRef = useCallback(async (filename) => {
    const d = await postJson(`/api/dataset/${currentId}/ref/extra/delete`, { filename });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
  }, [currentId, refresh, toast]);

  // `batches`: [{generator, variations}] — one entry per selected engine, the
  // shots already shared between them by engineSelection.js (API engines first,
  // the GPU-bound Klein one last). The server re-validates every entry and
  // refuses the whole run rather than dispatching it half-way.
  // `extraLoras`: optional generation-LoRA preset fragment(s) for this run (Idea
  // by @waltm) — an already-gated `{ generation_lora_preset?,
  // krea_generation_lora_preset? }` object from the two payload builders. One run
  // can carry both: each engine resolves its OWN key. An absent key means "no
  // preset" for that engine.
  const generate = useCallback((batches, multiplier, kleinModel, loraStrength, extraLoras) => wrap(async () => {
    const d = await postJson(`/api/dataset/${currentId}/generate`,
      { engine_batches: batches, multiplier, klein_model: kleinModel,
        lora_strength: loraStrength, ...(extraLoras || {}) });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    // Name the engines on a multi-engine run: "36 queued" alone doesn't say
    // which engine got what, and that is the whole point of running several.
    const per = d.per_engine || {};
    const detail = Object.keys(per).length > 1
      ? ` · ${Object.entries(per).map(([e, n]) => `${ENGINE_LABELS[e] || e} ${n}`).join(' · ')}` : '';
    toast.success(`${d.created} variation(s) queued${detail}`);
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  const importFiles = useCallback((files, { crop = true } = {}) => wrap(async () => {
    const fd = new FormData(); [...files].forEach((f) => fd.append('files', f));
    if (!crop) fd.append('crop', '0');   // keep the original framing (no square head-crop)
    const d = await postJson(`/api/dataset/${currentId}/import`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    const dup = d.duplicates || 0;
    const small = d.small || 0;
    toast.success(`${d.imported} imported${dup ? ` · ${dup} duplicate(s) skipped` : ''}`);
    // No numbers here on purpose: the input budget is a setting now, and a
    // copy of it in a toast is exactly how a hint goes stale.
    if (d.failed) toast.warning(`${d.failed} image${d.failed === 1 ? '' : 's'} not imported — use JPEG, PNG, WebP or BMP within the image size budget (Settings ▸ Captioning & quality ▸ Image size budget); resize a larger file, or raise the budget.`);
    if (dup && !d.imported) toast.warning('All files were already in the dataset (perceptual duplicates).');
    if (small) toast.warning(`${small} image(s) are under 768 px — training only downscales, they will stay soft.`);
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  // Concept only : télécharge les images scannées SÉLECTIONNÉES ({url,title}[])
  // directement dans le dataset (route /scrape-import). Le serveur borne chaque
  // requête (SCRAPE_IMPORT_MAX = 60, téléchargement synchrone) — on découpe donc la
  // sélection en lots envoyés EN SÉQUENCE, avec un toast de progression par lot :
  // « Select all » sur un gros scan s'importe en un clic au lieu d'un rejet 400.
  // La dédup perceptuelle est côté dataset, donc les doublons inter-lots sont
  // attrapés. Retourne {ok} pour que le panneau vide sa sélection sur succès.
  const scrapeImport = useCallback((items, { rescueSmall = false } = {}) => wrap(async () => {
    const BATCH = 60;                       // = svc.SCRAPE_IMPORT_MAX côté serveur
    let imported = 0;
    let rescueQueued = 0;
    let rescueFailed = 0;
    const skipped = {};
    for (let i = 0; i < items.length; i += BATCH) {
      if (items.length > BATCH) {
        toast.info(`Importing ${i + 1}–${Math.min(i + BATCH, items.length)} of ${items.length}…`);
      }
      const d = await postJson(`/api/dataset/${currentId}/scrape-import`,
        { items: items.slice(i, i + BATCH), rescue_small: !!rescueSmall });
      if (!d.ok) {
        toast.error(d.error || 'Unexpected error');
        if (imported || rescueQueued || rescueFailed) {
          const partial = summarizeScrapeImport({ imported, rescueQueued, rescueFailed, skipped });
          toast.warning(`${partial.message} before the failure.`);
          await refresh();
        }
        return d;
      }
      imported += d.imported || 0;
      rescueQueued += d.rescue_queued || 0;
      rescueFailed += d.rescue_failed || 0;
      for (const [k, v] of Object.entries(d.skipped || {})) skipped[k] = (skipped[k] || 0) + v;
    }
    const summary = summarizeScrapeImport({ imported, rescueQueued, rescueFailed, skipped });
    toast[summary.severity](summary.message);
    await refresh();
    return { ok: true, imported, rescue_queued: rescueQueued, rescue_failed: rescueFailed, skipped };
  }), [wrap, currentId, refresh, toast]);

  // Resolve a Klein rescue pair in ONE transaction so a network failure can
  // never leave both versions kept (or both half-updated). The candidate id is
  // the stable pair handle; the server finds its original through parent_image_id.
  const resolveSmallImageRescue = useCallback(async (candidateId, choice) => {
    const d = await postJson(
      `/api/dataset/${currentId}/small-image-rescue/${candidateId}/resolve`,
      { choice },
    );
    if (!d.ok) {
      toast.error(d.error || 'Could not save the rescue choice');
      return d;
    }
    const labels = {
      original: 'Original kept · Klein result rejected',
      klein: 'Klein result kept · original rejected',
      reject: 'Both versions rejected',
    };
    toast.success(labels[choice] || 'Rescue choice saved');
    await refresh();
    return d;
  }, [currentId, refresh, toast]);

  // Manually improve an existing dataset image with Klein. The backend always
  // creates a separate candidate row: the source pixels and their current
  // keep/reject state remain untouched until the user reviews the new version.
  const improveImage = useCallback(async (imageId, { silent = false, refreshAfter = true,
    engine } = {}) => {
    // `engine` is the button that was pressed in the lightbox ('klein' |
    // 'seedvr2'). Absent = the improve.engine setting decides, which is what
    // every single-✨ surface does.
    const d = await postJson(`/api/dataset/image/${imageId}/improve`,
      engine ? { engine } : {});
    if (!d.ok) {
      if (!silent) toast.error(d.error || 'Could not start image improvement');
      return d;
    }
    if (!silent) toast.success('Improvement started — the original stays intact while a separate 2 MP candidate is generated for validation.');
    if (refreshAfter) await refresh();
    return d;
  }, [refresh, toast]);

  // Re-run the ✨ Upscale & improve pass on a tile that IS an improvement. The
  // generic regenerate is closed to those rows (it would restart from the dataset
  // reference and make an unrelated variation); this replaces the result in place,
  // from the same parent image, with the improve settings as they are NOW — which
  // is the point: those knobs are editable in Settings.
  const reimproveImage = useCallback(async (imageId) => {
    const d = await postJson(`/api/dataset/image/${imageId}/reimprove`, {});
    if (!d.ok) {
      toast.error(d.error || 'Could not re-run the improvement');
      return d;
    }
    toast.success('Re-improving from the source image with your current improve settings');
    await refresh();
    return d;
  }, [refresh, toast]);

  // Bulk ✨ Upscale & improve: ONE call that starts a SERVER job. The batch
  // used to be a browser loop, so a selection bigger than the backend's fan-out cap
  // was mostly refused, ⏹ Stop could not reach it, and closing the tab killed it.
  // Progress now rides on `activity` (kind 'improve') and survives a reload.
  const improveBatch = useCallback(async (imageIds, engine) => {
    const ids = (imageIds || []).map((v) => Number(v)).filter(Number.isInteger);
    if (!ids.length) return { ok: false, error: 'nothing selected' };
    // `engine` is the button that was pressed ('klein' | 'seedvr2'). Absent = the
    // improve.engine setting, which is what the single-tile ✨ and re-improve use.
    const d = await postJson(`/api/dataset/${currentId}/improve/batch`,
      engine ? { image_ids: ids, engine } : { image_ids: ids });
    if (!d.ok) toast.error(d.error || 'Could not start the improvement batch');
    await refresh();
    return d;
  }, [currentId, refresh, toast]);

  // `expected` = how many images the caller counted as classifiable. It turns the
  // silent outcome into a diagnosis: the server answers ok/classified=0 when the
  // vision backend never replied (Ollama down), and 0 on its own reads as success.
  // Errors carry their `detail` too — "GPU busy" alone doesn't say training is running.
  const classify = useCallback((expected = 0) => wrap(async () => {
    const want = Number.isFinite(Number(expected)) ? Number(expected) : 0;
    // The route answers only when the whole pass is done, so nothing would refetch
    // the payload while it runs and its server-side `activity` (done/total) would
    // surface only on a manual reload. One seeded refresh flips `hasActivity`, and
    // the generic 3.5 s activity poll then drives the live progress from there.
    const seed = setTimeout(() => { refresh(currentId); }, 1200);
    try {
      const d = await postJson(`/api/dataset/${currentId}/classify`);
      if (!d.ok) {
        toast.error([d.error, d.detail].filter(Boolean).join(' — ') || 'Unexpected error');
        return;
      }
      const msg = classifyResultMessage(d.classified, want);
      (toast[msg.tone] || toast.success)(msg.text);
      await refresh();
    } finally {
      clearTimeout(seed);
    }
  }), [wrap, currentId, refresh, toast]);

  const caption = useCallback((mode) => wrap(async () => {
    const run = beginCaptioningRun(currentId);
    try {
      const d = await postJson(`/api/dataset/${run.datasetId}/caption`, mode ? { mode } : {});
      if (!d.ok) {
        toast.error([d.error, d.detail].filter(Boolean).join(' — ') || 'Unexpected error');
        return;
      }
      setLastCaptionRun({ datasetId: run.datasetId, captioned: d.captioned, engines: d.engines });
      if (d.stopped) toast.info(`Stopped — ${d.captioned} captioned before you stopped; the rest stays uncaptioned.`);
      else toast.success(`${d.captioned} captioned${captionResultSuffix(d.engines)}${captionSkippedSuffix(d)}`);
      await refresh(run.datasetId);
    } finally {
      finishCaptioningRun(run);
    }
  }, currentId), [wrap, currentId, refresh, toast,
                   beginCaptioningRun, finishCaptioningRun]);

  // Re-caption FORCÉ : ré-écrit TOUTES les captions des gardées (après changement de
  // prompt). Handler séparé de `caption` car onClick passe l'event en argument — un
  // `force` positionnel sur `caption` serait toujours truthy.
  const recaption = useCallback((mode) => wrap(async () => {
    const run = beginCaptioningRun(currentId);
    try {
      const d = await postJson(`/api/dataset/${run.datasetId}/caption`, { force: true, ...(mode ? { mode } : {}) });
      if (!d.ok) {
        toast.error([d.error, d.detail].filter(Boolean).join(' — ') || 'Unexpected error');
        return;
      }
      setLastCaptionRun({ datasetId: run.datasetId, captioned: d.captioned, engines: d.engines });
      if (d.stopped) toast.info(`Stopped — ${d.captioned} re-captioned before you stopped; the rest keeps its previous caption.`);
      else toast.success(`${d.captioned} re-captioned${captionResultSuffix(d.engines)}${captionSkippedSuffix(d)}`);
      await refresh(run.datasetId);
    } finally {
      finishCaptioningRun(run);
    }
  }, currentId), [wrap, currentId, refresh, toast,
                   beginCaptioningRun, finishCaptioningRun]);

  // Graceful Stop for a running captioning batch: the server flips a flag the worker
  // checks between images, so the current image finishes and the rest is left as-is.
  // The button shows "Stopping…" (driven by activity.cancelling) until the pass ends.
  const cancelCaption = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/caption/cancel`, {});
    if (d.ok) {
      toast.info('Stopping after the current image…');
      await refresh();   // pull activity.cancelling so the button flips immediately
    } else {
      // 409 = the batch already finished on its own between the poll and the click.
      toast.error(d.error || 'Nothing to stop');
    }
  }, [currentId, refresh, toast]);

  // Re-caption ciblé : ré-écrit la caption d'un sous-ensemble d'images gardées (une
  // ligne fuyante, ou « toutes les fuyantes ») avec le MÊME moteur/mode/contexte que le
  // lot. Volontairement HORS `wrap` : le spinner reste sur la/les ligne(s) concernée(s)
  // (recaptioningIds), le reste du panneau reste utilisable. Sérialisé côté serveur par
  // la fenêtre vision GPU (503 si un autre passage tourne). Retourne le résultat parsé.
  const recaptionImages = useCallback(async (ids, mode) => {
    const list = (ids || []).map((v) => Number(v)).filter((v) => Number.isInteger(v));
    const fresh = list.filter((id) => !recaptioningRef.current.has(id));
    if (!fresh.length) return { ok: false, error: 'nothing to re-caption' };
    fresh.forEach((id) => recaptioningRef.current.add(id));
    setRecaptioningIds((prev) => {
      const next = new Set(prev);
      fresh.forEach((id) => next.add(id));
      return next;
    });
    try {
      const d = await postJson(`/api/dataset/${currentId}/caption`,
        { image_ids: fresh, ...(mode ? { mode } : {}) });
      if (!d.ok) {
        // GPU-busy (a batch/training pass holds the vision window) carries a detail.
        toast.error(d.detail ? `${d.error} — ${d.detail}` : (d.error || 'Could not re-caption'));
        return d;
      }
      setLastCaptionRun({ datasetId: currentId, captioned: d.captioned, engines: d.engines });
      toast.success(`${d.captioned} re-captioned${captionResultSuffix(d.engines)}${captionSkippedSuffix(d)}`);
      await refresh();  // re-pulls captions + the live leak flags (scan is server-side)
      return d;
    } finally {
      fresh.forEach((id) => recaptioningRef.current.delete(id));
      setRecaptioningIds((prev) => {
        const next = new Set(prev);
        fresh.forEach((id) => next.delete(id));
        return next;
      });
    }
  }, [currentId, refresh, toast]);

  // Analyse de ressemblance faciale (InsightFace antelopev2, CPU — ~1-2 min, pas de
  // pause ComfyUI). Persiste face_score/face_state -> badges sur la grille.
  const analyzeFaces = useCallback(() => wrap(async () => {
    const run = beginLocalActivityRun('analyze', currentId);
    try {
      const d = await postJson(`/api/dataset/${run.datasetId}/analyze-faces`);
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      // Un scorer cassé disait « 0 analyzed » en VERT : le backend remonte
      // maintenant scoring_error {kind, detail} — dire POURQUOI.
      if (d.scoring_error) {
        toast.error(faceScoringErrorMessage(d.scoring_error));
        return;
      }
      const grey = (d.states?.too_small || 0) + (d.states?.no_face || 0)
        + (d.states?.extreme_pose || 0) + (d.states?.low_det || 0);
      toast.success(`${d.analyzed} analyzed · ${d.states?.scorable || 0} scored, ${grey} not scorable`);
      await refresh(run.datasetId);
    } finally {
      finishLocalActivityRun(run);
    }
  }, currentId), [wrap, currentId, refresh, toast,
                   beginLocalActivityRun, finishLocalActivityRun]);

  // Score one image without launching the dataset-wide scan. Its busy state stays
  // on this tile so independent curation actions remain available elsewhere.
  const scoreFace = useCallback(async (imageId) => {
    if (data?.face_scoring_blocked) {
      const error = data.face_scoring_blocked;
      toast.error(error);
      return { ok: false, error };
    }
    if (scoringFaceRef.current.size > 0) {
      return { ok: false, error: faceScoringErrorMessage({ kind: 'busy' }) };
    }
    scoringFaceRef.current.add(imageId);
    setScoringFaceIds((prev) => {
      const next = new Set(prev);
      next.add(imageId);
      return next;
    });
    try {
      const d = await postJson(`/api/dataset/image/${imageId}/analyze-face`);
      if (!d.ok) {
        toast.error(d.scoring_error ? faceScoringErrorMessage(d.scoring_error)
          : (d.error || 'Unexpected error'));
        return d;
      }
      if (d.scoring_error) {
        toast.error(faceScoringErrorMessage(d.scoring_error));
        return d;
      }
      await refresh();
      return d;
    } finally {
      scoringFaceRef.current.delete(imageId);
      setScoringFaceIds((prev) => {
        const next = new Set(prev);
        next.delete(imageId);
        return next;
      });
    }
  }, [data, refresh, toast]);

  // Watermark scan. WHICH detector runs follows Settings ▸ Captioning & quality ▸
  // Watermark detection; the server answers with the route it actually took, so a
  // pinned detector that could not run says so instead of quietly changing nothing.
  // Marks kept images with an overlaid watermark → 🚩 badges + a "Clean (N)"
  // button. Deletes nothing. { includeDismissed: true } re-examines the images
  // ruled false positives — the only way to re-judge them under a new detector.
  const findWatermarks = useCallback((options) => wrap(async () => {
    const includeDismissed = !!(options && options.includeDismissed);
    const run = beginLocalActivityRun('watermark', currentId);
    try {
      const d = await postJson(`/api/dataset/${run.datasetId}/watermarks/detect`,
        includeDismissed ? { include_dismissed: true } : undefined);
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      const engine = d.backend === 'detector' ? 'watermark detector' : 'vision model';
      const head = d.stopped ? 'Stopped —' : '';
      toast.success(`${head} ${d.detected || 0} watermark(s) found · ${d.none || 0} clean `
        + `(of ${d.checked || 0}, ${engine})`.trim());
      // A silent fallback is the failure mode this setting exists to remove: say
      // what ran and where to install what was asked for. Its own toast, because
      // it is a different fact from the count and must not be skimmed past.
      if (d.backend_note) toast.info(d.backend_note);
      // Flagged with no position: only the detector cascade produces this, and
      // 🧽 Clean cannot route on it. Named here rather than discovered later.
      if (d.unlocated) {
        toast.info(`${d.unlocated} flagged without a position — open 🔍 Review flagged `
          + 'and draw the zone, or Clean will leave them untouched.');
      }
      await refresh(run.datasetId);
    } finally {
      finishLocalActivityRun(run);
    }
  }, currentId), [wrap, currentId, refresh, toast,
                   beginLocalActivityRun, finishLocalActivityRun]);

  // Graceful Stop for a running watermark scan — same contract as the captioning
  // Stop: the worker checks a flag between images, so the current image finishes,
  // every verdict already written is KEPT, and a later 🧽 Find picks up the rest
  // (detect re-examines every kept row on each pass).
  const cancelWatermarkScan = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/watermarks/detect/cancel`, {});
    if (d.ok) {
      toast.info('Stopping after the current image… what is already flagged is kept.');
      await refresh();   // pull activity.cancelling so the button flips immediately
    } else {
      // 409 = the scan already finished on its own between the poll and the click.
      toast.error(d.error || 'Nothing to stop');
    }
  }, [currentId, refresh, toast]);

  // Clean the detected watermarks: border marks are CROPPED, small off-center ones
  // INPAINTED (LaMa), the rest flagged for manual review. The backend resolves the
  // configured Auto/GPU/CPU device and reserves ComfyUI only for an actual GPU pass.
  const cleanWatermarks = useCallback((method) => wrap(async () => {
    const run = beginLocalActivityRun('watermark', currentId);
    // Capture the ids whose file may change IN PLACE so we can cache-bust their
    // thumbnails (same filename → the browser would otherwise show the stale image).
    const detectedIds = (data?.images || [])
      .filter((i) => i.watermark_state === 'detected').map((i) => i.id);
    try {
      const d = await postJson(`/api/dataset/${run.datasetId}/watermarks/clean`,
        method ? { method } : undefined);
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      // A LaMa inpaint that was attempted and failed surfaces WHY (never silent).
      if (d.error) {
        toast.error(d.error.kind === 'unavailable'
          ? 'Watermark inpainting is not installed — use ⬇ Install inpainting next to the watermark tools.'
          : `Watermark inpainting failed: ${d.error.detail}`);
      }
      // ONE honest summary toast (no more "Nothing to clean" alongside "N skipped").
      const { severity, message } = summarizeClean(d);
      toast[severity](message);
      if (detectedIds.length) {
        setNonces((m) => {
          const next = { ...m };
          detectedIds.forEach((id) => { next[id] = (next[id] || 0) + 1; });
          return next;
        });
      }
      await refresh(run.datasetId);
    } finally {
      finishLocalActivityRun(run);
    }
  }, currentId), [wrap, currentId, data, refresh, toast,
                   beginLocalActivityRun, finishLocalActivityRun]);

  // Review mode (per-image watermark control). These deliberately do NOT use `wrap`
  // (no global busy flag) nor fire a toast: the review lightbox drives them one image
  // at a time and renders the outcome inline, then advances. They RETURN the parsed
  // result so the caller can show per-image success/failure and tally the recap.

  // Clean ONE (or a few) detected image(s) by id — same crop/LaMa/review routing as
  // cleanWatermarks, scoped to a subset. Cache-busts the touched thumbnails (crop/
  // inpaint edit the file IN PLACE, same filename) so the cleaned pixels show.
  const cleanWatermarkImages = useCallback(async (ids, method, allowCrop) => {
    const list = (ids || []).filter((v) => v != null);
    if (!list.length) return { ok: true, cropped: 0, inpainted: 0, inpainted_klein: 0, needs_review: 0, failed: 0, skipped: 0 };
    // allowCrop is the review lightbox's per-image crop-vs-inpaint override; forwarded
    // only when set (undefined → the backend uses the persisted preference, like the batch).
    const d = await postJson(`/api/dataset/${currentId}/watermarks/clean`,
      { image_ids: list, ...(method ? { method } : {}),
        ...(typeof allowCrop === 'boolean' ? { allow_crop: allowCrop } : {}) });
    if (d.ok) {
      setNonces((m) => {
        const next = { ...m };
        list.forEach((id) => { next[id] = (next[id] || 0) + 1; });
        return next;
      });
    }
    await refresh();
    return d;
  }, [currentId, refresh]);

  // Undo a Clean on ONE image: restore the preserved original in place and re-flag it
  // 'detected' so it can be re-cleaned (e.g. with the other engine). Cache-busts the
  // touched thumbnail (same filename, pixels change back) so the restored original shows.
  const restoreWatermarkImage = useCallback(async (imageId) => {
    const d = await postJson(`/api/dataset/${currentId}/image/${imageId}/watermark-restore`, {});
    if (d.ok) {
      setNonces((m) => ({ ...m, [imageId]: (m[imageId] || 0) + 1 }));
    }
    await refresh();
    return d;
  }, [currentId, refresh]);

  // Mark flagged image(s) as NOT a watermark (false positive) — badge clears and
  // future 🧽 Find passes skip them.
  const dismissWatermarks = useCallback(async (ids) => {
    const list = (ids || []).filter((v) => v != null);
    if (!list.length) return { ok: true, dismissed: 0 };
    const d = await postJson(`/api/dataset/${currentId}/watermarks/dismiss`, { image_ids: list });
    await refresh();
    return d;
  }, [currentId, refresh]);

  const saveWatermarkRegions = useCallback(async (imageId, regionsOrNull) => {
    const regions = serializeWatermarkRegions(regionsOrNull);
    const d = await putJson(
      `/api/dataset/${currentId}/image/${imageId}/watermark-regions`,
      { regions },
    );
    await refresh(currentId);
    return d;
  }, [currentId, refresh]);

  const setStatus = useCallback(async (imageId, status) => {
    const d = await postJson(`/api/dataset/image/${imageId}/status`, { status });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
  }, [refresh, toast]);

  /* Returns {ok, error} — the expanded caption editor keeps the text the user
     just typed only if it can tell a refusal from a success. `silent` is for
     that caller and that caller only: it renders the refusal next to the
     textarea, so a toast would say the same thing twice. The inline grid edit
     and 🗑 Caption own no surface of their own and keep the toast. */
  const setCaption = useCallback(async (imageId, captionText, shortText, { silent = false } = {}) => {
    // shortText undefined → only the long caption is sent (inline grid edit); the expanded
    // editor passes a string (possibly '') to also set the short variant.
    const body = shortText === undefined
      ? { caption: captionText }
      : { caption: captionText, caption_short: shortText };
    const d = await postJson(`/api/dataset/image/${imageId}/caption`, body);
    if (!d.ok) {
      const error = d.error || 'Unexpected error';
      if (!silent) toast.error(error);
      return { ok: false, error };
    }
    await refresh();
    return { ok: true };
  }, [refresh, toast]);

  const mirrorImage = useCallback(async (imageId) => {
    if (mirroringRef.current.has(imageId)) return false;
    mirroringRef.current.add(imageId);
    setMirroringIds((previous) => new Set(previous).add(imageId));
    try {
      const d = await postJson(`/api/dataset/image/${imageId}/mirror`, {});
      if (!d.ok) {
        toast.error(d.error || 'Could not mirror the image');
        return false;
      }
      await refresh();
      // The filename does not change, so force only this tile/lightbox/crop
      // editor to request the rewritten pixels instead of its cached response.
      setNonces((m) => ({ ...m, [imageId]: (m[imageId] || 0) + 1 }));
      toast.success('Image mirrored horizontally');
      return true;
    } finally {
      mirroringRef.current.delete(imageId);
      setMirroringIds((previous) => {
        const next = new Set(previous);
        next.delete(imageId);
        return next;
      });
    }
  }, [refresh, toast]);

  // 🔄 Quarter turns (idea by 1Tomber, GitHub #17). Same busy set as the mirror
  // on purpose: both rewrite the SAME file, so one running edit must grey out
  // the other rather than letting two of them race on one image.
  const rotateImage = useCallback(async (imageId, degrees) => {
    if (mirroringRef.current.has(imageId)) return false;
    mirroringRef.current.add(imageId);
    setMirroringIds((previous) => new Set(previous).add(imageId));
    try {
      const d = await postJson(`/api/dataset/image/${imageId}/rotate`, { degrees });
      if (!d.ok) {
        toast.error(d.error || 'Could not rotate the image');
        return false;
      }
      await refresh();
      // The filename does not change, so force only this tile/lightbox/crop
      // editor to request the rewritten pixels instead of its cached response.
      setNonces((m) => ({ ...m, [imageId]: (m[imageId] || 0) + 1 }));
      toast.success(degrees === 180 ? 'Image turned upside down'
        : `Image rotated 90° ${degrees === 90 ? 'right' : 'left'}`);
      return true;
    } finally {
      mirroringRef.current.delete(imageId);
      setMirroringIds((previous) => {
        const next = new Set(previous);
        next.delete(imageId);
        return next;
      });
    }
  }, [refresh, toast]);

  const crop = useCallback(async (imageId, box) => {
    const d = await postJson(`/api/dataset/image/${imageId}/crop`, box);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
    // Bump only this image's version — the rest of the grid keeps its cache (M1).
    setNonces((m) => ({ ...m, [imageId]: (m[imageId] || 0) + 1 }));
  }, [refresh, toast]);

  const cropRef = useCallback(async (box) => {
    const d = await postJson(`/api/dataset/${currentId}/ref/crop`, box);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
    setRefNonce((n) => n + 1);
  }, [currentId, refresh, toast]);

  // Crop ONE extra reference (identified by filename — extras have no numeric id).
  const cropExtraRef = useCallback(async (filename, box) => {
    const d = await postJson(`/api/dataset/${currentId}/ref/extra/crop`, { filename, ...box });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
    setRefNonce((n) => n + 1);
  }, [currentId, refresh, toast]);

  // Reset to the automatic head-crop (re-run on the kept original, no re-upload).
  const recropRefAuto = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/ref/recrop-auto`, {});
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    if (d.warning) toast.warning(d.warning); else toast.success('Reset to auto crop');
    await refresh();
    setRefNonce((n) => n + 1);
  }, [currentId, refresh, toast]);

  // ✦ Edit the reference. STARTS a server-side background job and returns at once
  // (202) — the edit is slow AND paid, so it must NOT ride the client's fetch (a
  // backgrounded mobile tab would kill it and lose the paid result). The candidate
  // is rediscovered through the payload's `reference_edit`; refresh() here starts
  // the activity poll that tracks it. Returns false (with a toast) on a start
  // error; true once the job is queued.
  const editReference = useCallback(async (
    prompt, engineOrEngines, files = [], retryBatchId = null,
  ) => {
    const engines = [...new Set(
      (Array.isArray(engineOrEngines) ? engineOrEngines : [engineOrEngines])
        .filter(Boolean))];
    if (!engines.length) {
      toast.error('Select at least one edit engine');
      return false;
    }
    const retryRequest = { prompt, engines, files: Array.from(files || []) };
    const fd = new FormData();
    fd.append('prompt', retryRequest.prompt);
    retryRequest.engines.forEach((engine) => fd.append('engines', engine));
    // Preserve the old one-engine request contract for older servers and direct
    // clients. A real multi-engine batch uses only the repeated engines field.
    if (retryRequest.engines.length === 1) fd.append('engine', retryRequest.engines[0]);
    retryRequest.files.forEach((f) => fd.append('ref', f));
    if (retryBatchId) fd.append('retry_batch_id', retryBatchId);
    const d = await postJson(`/api/dataset/${currentId}/ref/edit`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return false; }
    // The server remembers the prompt and engine for display/recovery, but not
    // request-scoped File objects. This snapshot is therefore the only honest
    // way to offer an exact Retry without changing backend storage semantics.
    const refreshed = await refresh();
    const confirmedRetry = retryRequestForReferenceEdit(
      { ...retryRequest, batchId: d.batch_id },
      refreshed?.data?.reference_edit,
    );
    if (refreshed?.status !== 'applied' || !confirmedRetry) {
      // The request was accepted, but stale status makes another Retry unsafe:
      // remove the in-memory files and force the modal to disable it until refresh.
      referenceEditRetryRef.current.delete(String(currentId));
      bumpReferenceEditRetryRevision((revision) => revision + 1);
      toast.warning('Edit queued, but its status could not be refreshed. Refresh the page before trying another edit.');
      return false;
    }
    referenceEditRetryRef.current.set(String(currentId), confirmedRetry);
    bumpReferenceEditRetryRevision((revision) => revision + 1);
    return true;
  }, [currentId, refresh, toast]);

  const retryReferenceEdit = useCallback(async () => {
    const retryRequest = retryRequestForReferenceEdit(
      referenceEditRetryRef.current.get(String(currentId)),
      data?.reference_edit,
    );
    if (!retryRequest) return false;
    return editReference(
      retryRequest.prompt, retryRequest.engines, retryRequest.files,
      retryRequest.batchId,
    );
  }, [currentId, data, editReference]);

  // Keep the ready candidate: the server atomically swaps the reference (old files
  // removed only after the new ones are on disk) and deletes the candidate.
  const keepEditedReference = useCallback(async (engine = null, batchId = null) => {
    const payload = {};
    if (engine) payload.engine = engine;
    if (batchId) payload.batch_id = batchId;
    const d = await postJson(`/api/dataset/${currentId}/ref/edit/keep`,
      payload);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return false; }
    referenceEditRetryRef.current.delete(String(currentId));
    bumpReferenceEditRetryRevision((revision) => revision + 1);
    toast.success('Reference updated');
    await refresh();
    setRefNonce((n) => n + 1);
    return true;
  }, [currentId, refresh, toast]);

  // Discard a pending edit (running=abandon or ready) — deletes the candidate.
  const discardEditedReference = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/ref/edit/discard`, {});
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return false; }
    referenceEditRetryRef.current.delete(String(currentId));
    bumpReferenceEditRetryRevision((revision) => revision + 1);
    await refresh();
    return true;
  }, [currentId, refresh, toast]);

  const deleteImage = useCallback(async (imageId) => {
    const d = await postJson(`/api/dataset/image/${imageId}/delete`);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
  }, [refresh, toast]);

  // Bulk find/replace across the kept images' captions. mode 'tag' = whole-tag
  // comma-separated replacement (booru); 'text' = plain substring.
  const replaceCaptions = useCallback(async (find, replace, mode = 'text') => {
    const d = await postJson(`/api/dataset/${currentId}/captions/replace`,
      { find, replace, mode });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return 0; }
    toast.success(`${d.changed} caption(s) updated`);
    await refresh();
    return d.changed;
  }, [currentId, refresh, toast]);

  // Write kohya-style same-stem .txt captions next to the kept images in the
  // dataset folder (same text as the export ZIP) — for external tools that read
  // the folder directly instead of downloading the ZIP.
  const writeCaptionFiles = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/captions/write-files`);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success(`${d.written} caption file(s) written`
      + (d.skipped_uncaptioned ? ` · ${d.skipped_uncaptioned} uncaptioned skipped` : ''));
  }, [currentId, toast]);

  // Open the dataset folder (images + .txt sidecars) in the OS file explorer —
  // same server-resolved open-folder route as the training panel's 📂 buttons.
  const openDatasetFolder = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/train/open-folder`, { target: 'dataset' });
    if (!d.ok) toast.error(d.error || 'Unexpected error');
  }, [currentId, toast]);

  // Multi-select curation: one request for the whole selection (grid checkboxes
  // + auto-triage). action: keep|reject|pending|delete|clear_caption.
  const batchImages = useCallback(async (ids, action, { silent = false } = {}) => {
    if (!ids || !ids.length) return 0;
    const d = await postJson(`/api/dataset/${currentId}/images/batch`, { ids, action });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return null; }
    await refresh();
    if (!silent) toast.success(action === 'delete'
      ? `${d.affected} ${d.affected === 1 ? 'image' : 'images'} deleted`
      : `${d.affected} image(s) updated`);
    return d.affected;
  }, [currentId, refresh, toast]);

  const cancelPending = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/cancel`);
    if (d.ok && d.recovery_error) {
      toast.error(
        `${d.recovery_error} generation(s) was preserved because LDS found an invalid ` +
        'ComfyUI recovery record. Do not delete the cards; check the server logs before retrying.'
      );
    } else if (d.ok && d.restart_required) {
      toast.error(
        `${d.restart_required} generation(s) has an unknown ComfyUI submission. ` +
        'Restart ComfyUI, then confirm the restart; LDS kept the card so recovery stays safe.'
      );
      const confirmed = globalThis.confirm?.(
        'ComfyUI must be restarted before LDS can safely clear this generation.\n\n' +
        'Click OK only if you have now restarted ComfyUI and it is responding.'
      );
      if (confirmed) {
        const recovered = await postJson(
          `/api/dataset/${currentId}/confirm-comfyui-restart`,
          { confirmed_comfyui_restart: true },
        );
        if (recovered.ok) {
          toast.success(`${recovered.cancelled} paused generation(s) recovered`);
        } else {
          toast.error(recovered.error || 'ComfyUI restart recovery failed');
        }
      }
    } else if (d.ok && d.retry_pending) {
      toast.error(
        `${d.retry_pending} generation(s) still await exact ComfyUI recovery. ` +
        'The cards were preserved — wait for ComfyUI to respond, then press Stop again.'
      );
    } else if (d.ok) {
      toast.success(`${d.cancelled} generation(s) cancelled`);
    } else {
      toast.error(d.error || 'Unexpected error');
    }
    await refresh();
  }, [currentId, refresh, toast]);

  // Re-roll one generated variation with a fresh seed (F2). Works on finished
  // AND failed tiles — it is the recovery path for failures. `prompt` (optional)
  // is the user-edited core prompt from the tile's ✏️ bubble; omitted → the
  // server reuses the row's / label's prompt (plain 🔄 and reject→regenerate).
  // A normal retry intentionally does NOT send the workspace generator. Each
  // row owns its provenance (API/Klein/Krea) and the server reuses it; otherwise
  // retrying a failed Krea card after selecting Nano Banana would silently bill
  // the API and change its rendering lane.
  /* Returns {ok, error}, and `silent` is for the ✏️ edit-prompt bubble: it shows
     the refusal itself, right under the prompt it was about, instead of losing a
     hand-written rewrite to a toast that names no field. */
  const regenerate = useCallback(async (imageId, loraStrength, prompt, { silent = false } = {}) => {
    const d = await postJson(`/api/dataset/image/${imageId}/regenerate`,
      { lora_strength: loraStrength, ...(prompt ? { prompt } : {}) });
    if (d.ok) { toast.success('Regeneration started'); await refresh(); return { ok: true }; }
    const error = d.error || 'Unexpected error';
    if (!silent) toast.error(error);
    return { ok: false, error };
  }, [refresh, toast]);

  const purgeUnused = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/purge`);
    if (d.ok) { toast.success(`${d.purged} image(s) deleted`); await refresh(); }
    else toast.error(d.error || 'Unexpected error');
  }, [currentId, refresh, toast]);

  const train = useCallback(async (opts = {}) => {
    const d = await postJson(`/api/dataset/${currentId}/train`,
      { base_model: opts.baseModel || '', variant: opts.variant || 'turbo',
        train_type: opts.trainType || 'zimage',
        training_mode: normalizeTrainingMode(opts.trainingMode),
        allow_caption_mismatch: !!opts.allowCaptionMismatch,
        // Images sans caption : plus un mur — confirm « train anyway » dans
        // TrainingPanel (marqueur UNCAPTIONED:), même flux que le mismatch.
        allow_uncaptioned: !!opts.allowUncaptioned,
        // Style caption quality (trigger-only / identical captions) may be
        // explicitly confirmed after the server explains the risk.
        allow_caption_quality: !!opts.allowCaptionQuality,
        // Custom-weights arch sniff non concluant → confirm « train anyway »
        // (marqueur CUSTOM_WEIGHTS_UNVERIFIED:), même flux confirmable.
        allow_unverified_weights: !!opts.allowUnverifiedWeights,
        // « Continue anyway » du panneau de préparation : lève le garde-fou plancher
        // d'images (garde-fou qualité) — jamais une impossibilité physique.
        allow_not_ready: !!opts.allowNotReady,
        // Overrides SDXL uniquement (le backend refuse 400 hors SDXL) — envoyés
        // seulement pour SDXL pour ne pas déclencher ce refus sur les autres.
        ...(opts.trainType === 'sdxl'
          ? { vae_path: opts.vaePath || '', te_path: opts.tePath || '' } : {}),
        // Masked training (background at 10 %) — a persisted DATASET setting now,
        // so the key is only sent when the caller has an explicit value. Omitted =
        // the server reads the dataset (a browser that never loaded the settings
        // must not overwrite a stored OFF with an optimistic default).
        ...(typeof opts.masked === 'boolean' ? { masked: opts.masked } : {}),
        // Cible de steps absolue (plafond choisi dans TrainingPanel) — omise si
        // vide → le backend calcule la valeur adaptative (recommended_steps).
        ...(opts.steps ? { steps: opts.steps } : {}),
        // fresh : écarte le run existant (archivé) → repart de zéro au lieu de
        // reprendre le dernier checkpoint (choix Resume/Fresh du TrainingPanel).
        ...(opts.fresh ? { fresh: true } : {}) });
    // L'entraînement tourne en CLI headless (pas l'UI ai-toolkit) → on N'OUVRE PAS
    // localhost:8675 (lien mort). La progression se suit ici (checkpoints + statut).
    if (d.ok) toast.success(`Training started (${d.steps || '?'} steps) — ComfyUI paused, follow the checkpoints here`);
    // Les refus confirmables (mismatch caption↔type, images sans caption) sont
    // gérés par un confirm dans TrainingPanel — pas un toast d'erreur.
    else if (!String(d.error || '').includes('MISMATCH_CAPTION')
             && !String(d.error || '').includes('UNCAPTIONED')
             && !String(d.error || '').includes('CAPTION_QUALITY')) {
      toast.error(d.error || 'Unexpected error');
    }
    return d;
  }, [currentId, toast]);

  // Bases entraînables + base/variante choisies + statut de conversion.
  const trainBaseInfo = useCallback(async () => {
    const r = await fetch(`/api/dataset/${currentId}/train/base-info`, { credentials: 'include' });
    return r.ok ? await r.json() : null;
  }, [currentId]);

  // Persists the adapter-vs-dense recipe independently from the other advanced
  // settings. The server returns the canonical exact enum; callers use null as a
  // rollback signal so a failed save never leaves the control lying.
  const setDatasetTrainingMode = useCallback(async (trainingMode, selection = {}) => {
    if (!currentId) {
      toast.error('No dataset selected');
      return null;
    }
    const payload = trainingModeSettingsPayload(trainingMode, selection);
    let d;
    try {
      d = await postJson(`/api/dataset/${currentId}/train/settings`, payload);
    } catch (error) {
      toast.error(error?.message || 'Could not save the training mode');
      return null;
    }
    if (!d.ok) {
      toast.error(d.error || 'Could not save the training mode');
      return null;
    }
    const saved = {
      trainingMode: normalizeTrainingMode(d.training_mode || payload.training_mode),
      trainType: d.train_type ?? selection.trainType,
      baseModel: Object.prototype.hasOwnProperty.call(d, 'base_model')
        ? d.base_model
        : selection.baseModel,
      variant: d.variant ?? selection.variant,
      slider: d.slider ?? null,
    };
    // A family change must refresh the library grouping and the live dataset,
    // just like setDatasetTrainType. Refresh is best-effort AFTER the atomic
    // commit: a failed list poll must never make the caller roll back a save that
    // the server already accepted.
    if (selection.trainType !== undefined) {
      try {
        await Promise.all([fetchList(), refresh(currentId)]);
      } catch {
        toast.warning('Training recipe saved, but the dataset list could not be refreshed yet.');
      }
    }
    return saved;
  }, [currentId, fetchList, refresh, toast]);

  // Persiste un patch de réglages avancés ai-toolkit (rank / resolution /
  // save_every). Renvoie les réglages effectifs, ou null en cas d'échec.
  const setTrainSettings = useCallback(async (patch) => {
    const d = await postJson(`/api/dataset/${currentId}/train/settings`, patch);
    if (d.ok) return d.train_settings;
    toast.error(d.error || 'Could not save the setting');
    return null;
  }, [currentId, toast]);

  // Lance la conversion d'un merge ComfyUI -> diffusers (thread arrière-plan).
  const prepareBase = useCallback(async (baseModel) => {
    const d = await postJson(`/api/dataset/${currentId}/train/prepare-base`, { base_model: baseModel });
    if (d.ok) toast.success(d.status === 'done' ? 'Base already ready' : 'Base conversion started…');
    else toast.error(d.error || 'Unexpected error');
    return d;
  }, [currentId, toast]);

  const stopTraining = useCallback(async () => {
    const d = await postJson('/api/dataset/train/stop');
    // Say what happened (the run stopped) and what survived, not just the side
    // effect — mirrors the Runs hub toast for the same endpoint.
    if (d.ok) toast.success('Training stopped — checkpoints already saved are kept; ComfyUI is re-enabled.');
    else toast.error(d.error || 'Unexpected error');
  }, [toast]);

  // baseModel/variant ciblent le run de la base SÉLECTIONNÉE (undefined → base
  // persistée). Pas de window.open : l'entraînement est headless (CLI), l'ancien
  // lien localhost:8675 était mort (« Ce site est inaccessible »).
  const continueTraining = useCallback(async (extraSteps = 1000, baseModel, variant, trainType, opts = {}) => {
    const body = {
      extra_steps: extraSteps,
      ...trainingRunSelection(baseModel, trainType, variant),
      ...(typeof opts.masked === 'boolean' ? { masked: opts.masked } : {}),
      allow_caption_mismatch: !!opts.allowCaptionMismatch,
      allow_uncaptioned: !!opts.allowUncaptioned,
      allow_unverified_weights: !!opts.allowUnverifiedWeights,
      allow_caption_quality: !!opts.allowCaptionQuality,
      allow_not_ready: !!opts.allowNotReady,
      // fromStep = resume from a chosen (possibly earlier) checkpoint; overrides =
      // safe-subset settings (cadence / preview prompts). Both optional.
      ...(opts.fromStep != null ? { from_step: opts.fromStep } : {}),
      ...(opts.overrides ? { overrides: opts.overrides } : {}),
      resume_mode: opts.resumeMode || 'weights_only',
      ...(opts.stateBundleId ? { state_bundle_id: opts.stateBundleId } : {}),
    };
    const d = await postJson(`/api/dataset/${currentId}/train/continue`, body);
    if (d.ok) toast.success(`Resumed from step ${d.resumed_from} → ${d.target_steps} — ComfyUI paused`);
    // CUSTOM_WEIGHTS_UNVERIFIED is an interactive refusal: TrainingPanel owns
    // the explicit confirm + retry, so do not emit a premature error toast.
    // `opts.quiet` says the CALLER shows the refusal itself — the ▶ Continue
    // dialog now stays open and renders it inside, and one sentence printed
    // twice a centimetre apart reads as a bug.
    else if (!opts.quiet
             && !String(d.error || '').includes('CUSTOM_WEIGHTS_UNVERIFIED: ')
             && !String(d.error || '').includes('CAPTION_QUALITY: ')
             && !String(d.error || '').includes('MISMATCH_CAPTION: ')
             && !String(d.error || '').includes('UNCAPTIONED: ')) {
      toast.error(d.error || 'Unexpected error');
    }
    return d;
  }, [currentId, toast]);

  // ☁ The CLOUD lane of the same ▶ Continue gesture: the chosen LOCAL checkpoint is
  // seeded onto a FRESH pod (the backend's resume_ckpt_path seam) instead of resuming
  // on this machine. Same payload as continueTraining — one dialog, two lanes — and
  // the same interactive-refusal contract, so TrainingPanel's confirm+retry helper
  // drives either lane without a second code path.
  const continueTrainingInCloud = useCallback(async (extraSteps = 1000, baseModel, variant, trainType, opts = {}) => {
    const body = {
      extra_steps: extraSteps,
      ...trainingRunSelection(baseModel, trainType, variant),
      ...(typeof opts.masked === 'boolean' ? { masked: opts.masked } : {}),
      allow_caption_mismatch: !!opts.allowCaptionMismatch,
      allow_uncaptioned: !!opts.allowUncaptioned,
      allow_unverified_weights: !!opts.allowUnverifiedWeights,
      allow_caption_quality: !!opts.allowCaptionQuality,
      allow_not_ready: !!opts.allowNotReady,
      allow_parallel_run: !!opts.allowParallelRun,
      ...(opts.fromStep != null ? { from_step: opts.fromStep } : {}),
      ...(opts.overrides ? { overrides: opts.overrides } : {}),
      resume_mode: opts.resumeMode || 'weights_only',
      ...(opts.stateBundleId ? { state_bundle_id: opts.stateBundleId } : {}),
      ...(opts.gpuName ? { gpu_name: opts.gpuName } : {}),
    };
    const d = await postJson(`/api/dataset/${currentId}/train/cloud/continue-local`, body);
    if (d.ok) toast.success(`Cloud run started from step ${d.resumed_from} → ${d.target_steps}`);
    // `opts.quiet`: same contract as continueTraining above — the caller owns
    // the refusal message (the ▶ Continue dialog stays open and shows it).
    else if (!opts.quiet
             && !String(d.error || '').includes('CUSTOM_WEIGHTS_UNVERIFIED: ')
             && !String(d.error || '').includes('CAPTION_QUALITY: ')
             && !String(d.error || '').includes('MISMATCH_CAPTION: ')
             && !String(d.error || '').includes('UNCAPTIONED: ')) {
      toast.error(d.error || 'Unexpected error');
    }
    return d;
  }, [currentId, toast]);

  // trainType = famille sélectionnée dans le menu LORA TYPE (Z-Image / SDXL / Krea).
  // Transmise à l'API pour que checkpoints + liste « IN COMFYUI » suivent le menu et
  // pas le train_type persisté du dataset (sinon LoRA Krea affichés sur la page Z-Image).
  const listCheckpoints = useCallback(async (baseModel, trainType, variant) => {
    const p = new URLSearchParams(trainingRunSelection(baseModel, trainType, variant));
    const qs = p.toString() ? `?${p.toString()}` : '';
    const r = await fetch(`/api/dataset/${currentId}/train/checkpoints${qs}`, { credentials: 'include' });
    return r.ok ? await r.json() : { checkpoints: [], imported: [] };
  }, [currentId]);

  const importCheckpoint = useCallback(async (filename, baseModel, trainType, variant) => {
    const body = { filename, ...trainingRunSelection(baseModel, trainType, variant) };
    try {
      const d = await postJson(`/api/dataset/${currentId}/train/import`, body);
      // d.note is set when the import was renamed to avoid overwriting a
      // DIFFERENT LoRA already at that name — surface it instead of the plain
      // success line so the rename is never silent.
      if (d.ok) toast.success(d.note || `LoRA imported: ${d.dest}`); else toast.error(d.error || 'Unexpected error');
    } catch (e) {
      // postJson THROWS on non-2xx and only fires a global toast for
      // 401/429/5xx — a 400/404/409 here used to be a silent no-op (the
      // button "did nothing", user-observed from a phone).
      toast.error(e.message || 'Import failed');
    }
  }, [currentId, toast]);

  // Supprime un checkpoint du dossier loras de la famille dans ComfyUI (libère de l'espace).
  const deleteCheckpoint = useCallback(async (filename, trainType, variant) => {
    const body = { filename, ...trainingRunSelection(undefined, trainType, variant) };
    const d = await postJson(`/api/dataset/${currentId}/train/checkpoint/delete`, body);
    if (d.ok) toast.success(`Checkpoint deleted: ${d.removed}`); else toast.error(d.error || 'Unexpected error');
    return d;
  }, [currentId, toast]);

  const exportZipFor = useCallback((datasetId) => {
    if (Number.isInteger(datasetId) && datasetId > 0) {
      window.open(`/api/dataset/${datasetId}/export`, '_blank');
    }
  }, []);
  const exportZip = useCallback(() => exportZipFor(currentId), [currentId, exportZipFor]);

  // Full portable backup (images + captions + settings) — distinct from the
  // training-format export. Restore creates a NEW dataset and opens it.
  const exportBackupFor = useCallback((datasetId) => {
    if (Number.isInteger(datasetId) && datasetId > 0) {
      window.open(`/api/dataset/${datasetId}/backup`, '_blank');
    }
  }, []);
  const exportBackup = useCallback(() => exportBackupFor(currentId), [currentId, exportBackupFor]);

  // === Back up / restore EVERYTHING (whole library + config) ===============
  // A big library is potentially gigabytes, so both run as background jobs the
  // server owns; we poll a compact status snapshot and surface an honest final
  // report. Progress phrasing/summaries live in utils/fullBackup (pure/tested).
  const [backupJob, setBackupJob] = useState(null);     // {state,done,total,current,result,error}
  const [restoreJob, setRestoreJob] = useState(null);   // full-restore progress
  const backupTimer = useRef(null);
  const restoreTimer = useRef(null);
  useEffect(() => () => {
    clearTimeout(backupTimer.current);
    clearTimeout(restoreTimer.current);
  }, []);

  const pollBackup = useCallback(() => {
    clearTimeout(backupTimer.current);
    backupTimer.current = setTimeout(async () => {
      const st = await getJsonSilent('/api/backup/full/status');
      if (!st) { pollBackup(); return; }                 // transient blip → keep polling
      setBackupJob(st);
      if (st.state === 'running') pollBackup();
      else if (st.state === 'error') toast.error(st.error || 'Backup failed');
    }, 900);
  }, [toast]);

  const backupEverything = useCallback(async (includeLoras = false) => {
    const d = await postJson('/api/backup/full/start', { include_loras: !!includeLoras });
    if (!d.ok) { toast.error(d.error || 'Could not start the backup'); return; }
    setBackupJob({ state: 'running', done: 0, total: 0, current: null });
    pollBackup();
  }, [pollBackup, toast]);

  const downloadBackup = useCallback((name) => {
    if (name) window.open(`/api/backup/full/download?name=${encodeURIComponent(name)}`, '_blank');
  }, []);
  const openBackupsFolder = useCallback(async () => {
    const d = await postJson('/api/backup/full/open-folder', {});
    if (!d.ok) toast.error(d.error || 'Could not open the backups folder');
  }, [toast]);
  const dismissBackup = useCallback(() => {
    clearTimeout(backupTimer.current); setBackupJob(null);
  }, []);

  const pollRestore = useCallback(() => {
    clearTimeout(restoreTimer.current);
    restoreTimer.current = setTimeout(async () => {
      const st = await getJsonSilent('/api/backup/full/restore/status');
      if (!st) { pollRestore(); return; }
      setRestoreJob(st);
      if (st.state === 'running') { pollRestore(); return; }
      if (st.state === 'done') await fetchList();
      else if (st.state === 'error') toast.error(st.error || 'Restore failed');
    }, 900);
  }, [fetchList, toast]);

  // Restore ANY backup zip. The server auto-detects a single-dataset backup
  // (imported inline, then opened) vs a master "Back up everything" archive (a
  // background job with its own progress + honest final report) — so the one
  // "Import backup" button in the library accepts both.
  const importBackup = useCallback(async (file) => {
    const fd = new FormData(); fd.append('file', file);
    const d = await postJson('/api/backup/full/restore', fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    if (d.kind === 'single') {
      toast.success(`Dataset “${d.name}” restored`);
      await fetchList();
      await open(d.id);
      return;
    }
    setRestoreJob({ state: 'running', done: 0, total: 0, current: null });
    pollRestore();
  }, [fetchList, open, pollRestore, toast]);
  const dismissRestore = useCallback(() => {
    clearTimeout(restoreTimer.current); setRestoreJob(null);
  }, []);

  // Merge an EXISTING training dataset (ZIP of images + kohya-style same-stem
  // .txt captions) into the open dataset — distinct from importBackup (which
  // restores this app's own backup format as a NEW dataset).
  const importDatasetZip = useCallback((file) => wrap(async () => {
    const fd = new FormData(); fd.append('file', file);
    const d = await postJson(`/api/dataset/${currentId}/import-zip`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    const parts = [`${d.imported} imported`];
    if (d.captions) parts.push(`${d.captions} caption(s) attached`);
    // The caption-elsewhere round trip: images already here are duplicates by
    // design, so "N duplicates skipped" alone read as a failure. Say what the
    // trip actually brought back.
    if (d.captions_applied) parts.push(`${d.captions_applied} caption(s) applied to images already here`);
    if (d.duplicates) parts.push(`${d.duplicates} duplicate(s) skipped`);
    if (d.captions_kept) parts.push(`${d.captions_kept} kept the caption written here`);
    if (d.failed) parts.push(`${d.failed} unreadable`);
    toast.success(parts.join(' · '));
    if (d.small) toast.warning(`${d.small} image(s) under 768 px — they will stay soft in training.`);
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  // Same merge from a FOLDER on this machine's disk (kohya images + same-stem
  // .txt captions) — the path is a server-side path pasted as text, not a
  // browser file pick (a browser can't hand the server a folder path).
  /* {ok, error}, so the in-app folder browser can keep the tree where the user
     left it when the path is refused. `silent` for that browser only — it draws
     the refusal above its own "Use this folder" button.
     wrap() DROPS the call when another dataset job holds the busy flag and
     returns undefined for it; that is a refusal too, and it says so rather than
     reaching the browser as a mute "no answer from the server". */
  const importDatasetFolder = useCallback(async (path, { silent = false } = {}) => {
    const out = await wrap(async () => {
      const d = await postJson(`/api/dataset/${currentId}/import-folder`, { path });
      if (!d.ok) {
        const error = d.error || 'Unexpected error';
        if (!silent) toast.error(error);
        return { ok: false, error };
      }
      const parts = [`${d.imported} imported`];
      if (d.captions) parts.push(`${d.captions} caption(s) attached`);
      // The caption-elsewhere round trip: images already here are duplicates by
      // design, so "N duplicates skipped" alone read as a failure. Say what the
      // trip actually brought back.
      if (d.captions_applied) parts.push(`${d.captions_applied} caption(s) applied to images already here`);
      if (d.duplicates) parts.push(`${d.duplicates} duplicate(s) skipped`);
      if (d.captions_kept) parts.push(`${d.captions_kept} kept the caption written here`);
      if (d.failed) parts.push(`${d.failed} unreadable`);
      toast.success(parts.join(' · '));
      if (d.small) toast.warning(`${d.small} image(s) under 768 px — they will stay soft in training.`);
      await refresh();
      return { ok: true };
    });
    return out ?? { ok: false, error: 'Another dataset job is running — wait for it to finish, then try again.' };
  }, [wrap, currentId, refresh, toast]);

  // Restoration layer: fold the server-side `activity` into the visual flags so a
  // reloaded page (which lost the local captioning/analyzing/watermarking state)
  // still shows the concerned button's spinner and disables concurrent actions —
  // exactly as if the click had just happened. The local flags stay authoritative
  // for the user who actually clicked (their fetch flow is untouched); this only
  // ADDS the server truth on top. `busy` OR'd with any activity re-disables every
  // concurrent action and shows the amber "in progress" banner after a reload.
  const activity = currentActivity;
  const actKind = activity?.kind || null;
  const captioningLive = captioningRuns.has(String(currentId))
    || actKind === 'caption' || actKind === 'recaption';
  const analyzingLive = localActivityRuns.has(`analyze:${currentId}`)
    || actKind === 'analyze_faces';
  const watermarkingLive = localActivityRuns.has(`watermark:${currentId}`)
    || actKind === 'watermark_detect' || actKind === 'watermark_clean';
  const busyLive = busy || !!activity;
  const canRetryReferenceEdit = Boolean(retryRequestForReferenceEdit(
    referenceEditRetryRef.current.get(String(currentId)),
    data?.reference_edit,
  ));


  return { datasets, currentId, data, busy: busyLive, localBusy: busy, captioning: captioningLive,
           lastCaptionRun,
           analyzing: analyzingLive, watermarking: watermarkingLive, activity,
           nonces, mirroringIds, refNonce, scoringFaceIds, recaptioningIds, create, open,
           deleteDataset, updateSettings, setCurrentId, setRef, addExtraRef, removeExtraRef,
           generate, importFiles, scrapeImport, resolveSmallImageRescue, improveImage, reimproveImage, improveBatch, classify, caption, recaption, recaptionImages,
           setStatus, setCaption, mirrorImage, rotateImage, crop, cropRef, cropExtraRef, recropRefAuto, editReference, retryReferenceEdit, canRetryReferenceEdit, keepEditedReference, discardEditedReference, setDatasetTrainType, setDatasetFidelity, deleteImage, batchImages, replaceCaptions, writeCaptionFiles, openDatasetFolder, cancelPending, cancelCaption, regenerate, analyzeFaces, scoreFace,
           findWatermarks, cancelWatermarkScan, cleanWatermarks, cleanWatermarkImages, restoreWatermarkImage, dismissWatermarks, saveWatermarkRegions,
           purgeUnused, exportZip, exportBackup, exportZipFor, exportBackupFor, importBackup, importDatasetZip, importDatasetFolder,
           backupEverything, backupJob, downloadBackup, openBackupsFolder, dismissBackup, restoreJob, dismissRestore,
           refresh, train, stopTraining, continueTraining, continueTrainingInCloud,
           listCheckpoints, importCheckpoint, deleteCheckpoint,
           trainBaseInfo, setTrainSettings, setDatasetTrainingMode, prepareBase };
}
