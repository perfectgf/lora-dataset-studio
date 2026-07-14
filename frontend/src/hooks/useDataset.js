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

/**
 * Compose the 🧽 Clean summary toast from the server's counts — PURE (no React,
 * no toast) so the honest-message logic is testable on its own.
 * Response shape: {cropped, inpainted, needs_review, failed, skipped, error}.
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
  const inpainted = d.inpainted || 0;
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
  const [currentId, setCurrentId] = useState(() => {
    try { const v = localStorage.getItem('datasetCurrentId'); return v ? Number(v) : null; }
    catch { return null; }
  });
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  // Tracks an in-flight captioning pass so the UI can poll progressively.
  const [captioning, setCaptioning] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [watermarking, setWatermarking] = useState(false);
  // Per-image cache-bust versions (M1): only the cropped image reloads,
  // plus a separate version counter for the reference photo.
  const [nonces, setNonces] = useState({});
  const [refNonce, setRefNonce] = useState(0);
  const pollRef = useRef(null);
  const busyRef = useRef(false); // re-entrancy guard for GPU-bound actions (I2)

  const fetchList = useCallback(async () => {
    try {
      const r = await fetch('/api/dataset/list', { credentials: 'include' });
      if (r.ok) setDatasets((await r.json()).datasets || []);
    } catch { /* transient network error — keep the last list */ }
  }, []);

  const refresh = useCallback(async (id) => {
    const dsId = id ?? currentId;
    if (!dsId) return;
    try {
      const r = await fetch(`/api/dataset/${dsId}`, { credentials: 'include' });
      if (r.ok) setData(await r.json());
      // Only a definitive 404 ejects back to the list (dataset gone). Transient
      // failures (500, gateway hiccup) keep the open workspace untouched (M4).
      else if (r.status === 404) { setData(null); setCurrentId(null); }
    } catch { /* network blip — keep current workspace, the poll will retry */ }
  }, [currentId]);

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
    if (!captioning || !currentId) return undefined;
    const id = setInterval(() => refresh(currentId), 2000);
    return () => clearInterval(id);
  }, [captioning, currentId, refresh]);

  // Persistence layer: a server-side batch (watermark detect/clean, caption,
  // re-caption, analyze faces, classify) advertises itself in the payload's
  // `activity` field. Whenever it's non-null — INCLUDING after a page reload that
  // dropped the local captioning/analyzing/watermarking flags — poll the dataset
  // every ~3.5s to track progress and detect the end. Keyed on the boolean
  // `hasActivity` (not the activity object, whose identity changes each fetch) so
  // the interval isn't torn down and rebuilt on every poll; it stops the moment
  // `activity` clears (the following refresh brings the final state; the completion
  // toast can't be restored — accepted, only the visual state is).
  const hasActivity = !!data?.activity;
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
  // grouped menu re-sorts. Refreshes the list; silent on failure (non-critical).
  const setDatasetTrainType = useCallback(async (trainType) => {
    if (!currentId) return;
    const d = await postJson(`/api/dataset/${currentId}/train-type`, { train_type: trainType });
    if (d.ok) fetchList();
  }, [currentId, fetchList]);

  // Edit name / trigger / (concept) description after creation. Trigger change is
  // safe (prepended at export); a concept-desc change resets the avoid-list → the
  // toast nudges a re-caption (same contract as fidelity). Refreshes both views.
  const updateSettings = useCallback(async ({ name, trigger_word, concept_desc }) => {
    if (!currentId) return { ok: false };
    const d = await postJson(`/api/dataset/${currentId}/settings`,
      { name, trigger_word, concept_desc });
    if (!d.ok) { toast.error(d.error || 'Could not save settings'); return d; }
    toast.success(d.concept_desc_changed
      ? 'Saved — concept changed; re-caption to apply it to existing captions'
      : 'Settings saved');
    await refresh();
    fetchList();
    return d;
  }, [currentId, refresh, fetchList, toast]);

  const deleteDataset = useCallback(async (id) => {
    const d = await postJson(`/api/dataset/${id}/delete`);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success('Dataset deleted');
    if (currentId === id) { setCurrentId(null); setData(null); }
    await fetchList();
  }, [currentId, fetchList, toast]);

  // Run a GPU-bound action exclusively (I2): re-entrancy guard + busy flag.
  // A second call while one is in flight is dropped instead of double-firing.
  const wrap = useCallback(async (fn) => {
    if (busyRef.current) return undefined;
    busyRef.current = true;
    setBusy(true);
    try { return await fn(); }
    finally { busyRef.current = false; setBusy(false); }
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

  const generate = useCallback((variations, multiplier, kleinModel, loraStrength, generator) => wrap(async () => {
    const d = await postJson(`/api/dataset/${currentId}/generate`,
      { variations, multiplier, klein_model: kleinModel, lora_strength: loraStrength,
        generator: generator || 'klein' });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success(`${d.created} variation(s) queued`);
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
  const scrapeImport = useCallback((items) => wrap(async () => {
    const BATCH = 60;                       // = svc.SCRAPE_IMPORT_MAX côté serveur
    let imported = 0;
    const skipped = {};
    for (let i = 0; i < items.length; i += BATCH) {
      if (items.length > BATCH) {
        toast.info(`Importing ${i + 1}–${Math.min(i + BATCH, items.length)} of ${items.length}…`);
      }
      const d = await postJson(`/api/dataset/${currentId}/scrape-import`,
        { items: items.slice(i, i + BATCH) });
      if (!d.ok) {
        toast.error(d.error || 'Unexpected error');
        if (imported) {
          toast.warning(`${imported} image(s) were imported before the failure.`);
          await refresh();
        }
        return d;
      }
      imported += d.imported || 0;
      for (const [k, v] of Object.entries(d.skipped || {})) skipped[k] = (skipped[k] || 0) + v;
    }
    const skips = Object.entries(skipped).filter(([, v]) => v > 0).map(([k, v]) => `${v} ${k}`).join(', ');
    toast.success(`${imported} imported${skips ? ` · skipped ${skips}` : ''}`);
    await refresh();
    return { ok: true, imported, skipped };
  }), [wrap, currentId, refresh, toast]);

  const classify = useCallback(() => wrap(async () => {
    const d = await postJson(`/api/dataset/${currentId}/classify`);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success(`${d.classified} classified`);
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  const caption = useCallback((mode) => wrap(async () => {
    setCaptioning(true);
    try {
      const d = await postJson(`/api/dataset/${currentId}/caption`, mode ? { mode } : {});
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      toast.success(`${d.captioned} captioned`);
      await refresh();
    } finally {
      setCaptioning(false);
    }
  }), [wrap, currentId, refresh, toast]);

  // Re-caption FORCÉ : ré-écrit TOUTES les captions des gardées (après changement de
  // prompt). Handler séparé de `caption` car onClick passe l'event en argument — un
  // `force` positionnel sur `caption` serait toujours truthy.
  const recaption = useCallback((mode) => wrap(async () => {
    setCaptioning(true);
    try {
      const d = await postJson(`/api/dataset/${currentId}/caption`, { force: true, ...(mode ? { mode } : {}) });
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      toast.success(`${d.captioned} re-captioned`);
      await refresh();
    } finally {
      setCaptioning(false);
    }
  }), [wrap, currentId, refresh, toast]);

  // Analyse de ressemblance faciale (InsightFace antelopev2, CPU — ~1-2 min, pas de
  // pause ComfyUI). Persiste face_score/face_state -> badges sur la grille.
  const analyzeFaces = useCallback(() => wrap(async () => {
    setAnalyzing(true);
    try {
      const d = await postJson(`/api/dataset/${currentId}/analyze-faces`);
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      // Un scorer cassé disait « 0 analyzed » en VERT : le backend remonte
      // maintenant scoring_error {kind, detail} — dire POURQUOI.
      if (d.scoring_error) {
        const { kind, detail } = d.scoring_error;
        toast.error(kind === 'unavailable'
          ? 'Face scoring is not installed — run the Quality tools step in Setup.'
          : kind === 'ref_unusable'
            ? `The reference photo is not usable for scoring: ${detail}`
            : `Face scoring failed: ${detail}`);
        return;
      }
      const grey = (d.states?.too_small || 0) + (d.states?.no_face || 0)
        + (d.states?.extreme_pose || 0) + (d.states?.low_det || 0);
      toast.success(`${d.analyzed} analyzed · ${d.states?.scorable || 0} scored, ${grey} not scorable`);
      await refresh();
    } finally {
      setAnalyzing(false);
    }
  }), [wrap, currentId, refresh, toast]);

  // Watermark scan (Qwen3-VL, GPU window). Marks kept images with an overlaid
  // watermark → 🚩 badges + a "Clean (N)" button. Deletes nothing.
  const findWatermarks = useCallback(() => wrap(async () => {
    setWatermarking(true);
    try {
      const d = await postJson(`/api/dataset/${currentId}/watermarks/detect`);
      if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
      toast.success(`${d.detected || 0} watermark(s) found · ${d.none || 0} clean (of ${d.checked || 0})`);
      await refresh();
    } finally {
      setWatermarking(false);
    }
  }), [wrap, currentId, refresh, toast]);

  // Clean the detected watermarks: border marks are CROPPED, small off-center ones
  // INPAINTED (LaMa), the rest flagged for manual review. CPU only (no ComfyUI pause).
  const cleanWatermarks = useCallback(() => wrap(async () => {
    setWatermarking(true);
    // Capture the ids whose file may change IN PLACE so we can cache-bust their
    // thumbnails (same filename → the browser would otherwise show the stale image).
    const detectedIds = (data?.images || [])
      .filter((i) => i.watermark_state === 'detected').map((i) => i.id);
    try {
      const d = await postJson(`/api/dataset/${currentId}/watermarks/clean`);
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
      await refresh();
    } finally {
      setWatermarking(false);
    }
  }), [wrap, currentId, data, refresh, toast]);

  // Review mode (per-image watermark control). These deliberately do NOT use `wrap`
  // (no global busy flag) nor fire a toast: the review lightbox drives them one image
  // at a time and renders the outcome inline, then advances. They RETURN the parsed
  // result so the caller can show per-image success/failure and tally the recap.

  // Clean ONE (or a few) detected image(s) by id — same crop/LaMa/review routing as
  // cleanWatermarks, scoped to a subset. Cache-busts the touched thumbnails (crop/
  // inpaint edit the file IN PLACE, same filename) so the cleaned pixels show.
  const cleanWatermarkImages = useCallback(async (ids) => {
    const list = (ids || []).filter((v) => v != null);
    if (!list.length) return { ok: true, cropped: 0, inpainted: 0, needs_review: 0, failed: 0, skipped: 0 };
    const d = await postJson(`/api/dataset/${currentId}/watermarks/clean`, { image_ids: list });
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

  const setCaption = useCallback(async (imageId, captionText) => {
    const d = await postJson(`/api/dataset/image/${imageId}/caption`, { caption: captionText });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    await refresh();
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

  // Reset to the automatic head-crop (re-run on the kept original, no re-upload).
  const recropRefAuto = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/ref/recrop-auto`, {});
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    if (d.warning) toast.warning(d.warning); else toast.success('Reset to auto crop');
    await refresh();
    setRefNonce((n) => n + 1);
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
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return 0; }
    if (!silent) toast.success(`${d.affected} image(s) updated`);
    await refresh();
    return d.affected;
  }, [currentId, refresh, toast]);

  const cancelPending = useCallback(async () => {
    const d = await postJson(`/api/dataset/${currentId}/cancel`);
    if (d.ok) toast.success(`${d.cancelled} generation(s) cancelled`);
    else toast.error(d.error || 'Unexpected error');
    await refresh();
  }, [currentId, refresh, toast]);

  // Re-roll one generated variation with a fresh seed (F2). Works on finished
  // AND failed tiles — it is the recovery path for failures. `prompt` (optional)
  // is the user-edited core prompt from the tile's ✏️ bubble; omitted → the
  // server reuses the row's / label's prompt (plain 🔄 and reject→regenerate).
  // The generator CURRENTLY selected in the workspace (persisted by
  // VariationCatalog) is sent along so the regenerate follows the user's
  // selection instead of being pinned to the engine that made the tile;
  // the Klein model pick rides too for an API→Klein switch. Missing keys =
  // server keeps the legacy reuse-the-row's-engine behaviour.
  const regenerate = useCallback(async (imageId, loraStrength, prompt) => {
    let engine = null; let kleinModel = null;
    try {
      engine = localStorage.getItem('datasetGenerator') || null;
      kleinModel = localStorage.getItem('editPage_flux2KleinModel_v1') || null;
    } catch { /* private mode — legacy behaviour */ }
    const d = await postJson(`/api/dataset/image/${imageId}/regenerate`,
      { lora_strength: loraStrength, ...(prompt ? { prompt } : {}),
        ...(engine ? { engine } : {}), ...(kleinModel ? { klein_model: kleinModel } : {}) });
    if (d.ok) { toast.success('Regeneration started'); await refresh(); }
    else toast.error(d.error || 'Unexpected error');
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
        allow_caption_mismatch: !!opts.allowCaptionMismatch,
        // Images sans caption : plus un mur — confirm « train anyway » dans
        // TrainingPanel (marqueur UNCAPTIONED:), même flux que le mismatch.
        allow_uncaptioned: !!opts.allowUncaptioned,
        // Custom-weights arch sniff non concluant → confirm « train anyway »
        // (marqueur CUSTOM_WEIGHTS_UNVERIFIED:), même flux confirmable.
        allow_unverified_weights: !!opts.allowUnverifiedWeights,
        // Overrides SDXL uniquement (le backend refuse 400 hors SDXL) — envoyés
        // seulement pour SDXL pour ne pas déclencher ce refus sur les autres.
        ...(opts.trainType === 'sdxl'
          ? { vae_path: opts.vaePath || '', te_path: opts.tePath || '' } : {}),
        // Masked training (fond à 10 %) — défaut ON, toggle dans TrainingPanel.
        masked: opts.masked !== false,
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
             && !String(d.error || '').includes('UNCAPTIONED')) {
      toast.error(d.error || 'Unexpected error');
    }
    return d;
  }, [currentId, toast]);

  // Bases entraînables + base/variante choisies + statut de conversion.
  const trainBaseInfo = useCallback(async () => {
    const r = await fetch(`/api/dataset/${currentId}/train/base-info`, { credentials: 'include' });
    return r.ok ? await r.json() : null;
  }, [currentId]);

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
    if (d.ok) toast.success('ComfyUI re-enabled'); else toast.error(d.error || 'Unexpected error');
  }, [toast]);

  // baseModel/variant ciblent le run de la base SÉLECTIONNÉE (undefined → base
  // persistée). Pas de window.open : l'entraînement est headless (CLI), l'ancien
  // lien localhost:8675 était mort (« Ce site est inaccessible »).
  const continueTraining = useCallback(async (extraSteps = 1000, baseModel, variant) => {
    const body = { extra_steps: extraSteps };
    if (baseModel !== undefined && baseModel !== null) body.base_model = baseModel;
    if (variant) body.variant = variant;
    const d = await postJson(`/api/dataset/${currentId}/train/continue`, body);
    if (d.ok) toast.success(`Resumed from step ${d.resumed_from} → ${d.target_steps} — ComfyUI paused`);
    else toast.error(d.error || 'Unexpected error');
    return d;
  }, [currentId, toast]);

  // trainType = famille sélectionnée dans le menu LORA TYPE (Z-Image / SDXL / Krea).
  // Transmise à l'API pour que checkpoints + liste « IN COMFYUI » suivent le menu et
  // pas le train_type persisté du dataset (sinon LoRA Krea affichés sur la page Z-Image).
  const listCheckpoints = useCallback(async (baseModel, trainType) => {
    const p = new URLSearchParams();
    if (baseModel !== undefined && baseModel !== null) p.set('base_model', baseModel);
    if (trainType) p.set('train_type', trainType);
    const qs = p.toString() ? `?${p.toString()}` : '';
    const r = await fetch(`/api/dataset/${currentId}/train/checkpoints${qs}`, { credentials: 'include' });
    return r.ok ? await r.json() : { checkpoints: [], imported: [] };
  }, [currentId]);

  const importCheckpoint = useCallback(async (filename, baseModel, trainType) => {
    const body = { filename };
    if (baseModel !== undefined && baseModel !== null) body.base_model = baseModel;
    if (trainType) body.train_type = trainType;
    try {
      const d = await postJson(`/api/dataset/${currentId}/train/import`, body);
      if (d.ok) toast.success(`LoRA imported: ${d.dest}`); else toast.error(d.error || 'Unexpected error');
    } catch (e) {
      // postJson THROWS on non-2xx and only fires a global toast for
      // 401/429/5xx — a 400/404/409 here used to be a silent no-op (the
      // button "did nothing", user-observed from a phone).
      toast.error(e.message || 'Import failed');
    }
  }, [currentId, toast]);

  // Supprime un checkpoint du dossier loras de la famille dans ComfyUI (libère de l'espace).
  const deleteCheckpoint = useCallback(async (filename, trainType) => {
    const body = { filename };
    if (trainType) body.train_type = trainType;
    const d = await postJson(`/api/dataset/${currentId}/train/checkpoint/delete`, body);
    if (d.ok) toast.success(`Checkpoint deleted: ${d.removed}`); else toast.error(d.error || 'Unexpected error');
    return d;
  }, [currentId, toast]);

  const exportZip = useCallback(() => {
    if (currentId) window.open(`/api/dataset/${currentId}/export`, '_blank');
  }, [currentId]);

  // Full portable backup (images + captions + settings) — distinct from the
  // training-format export. Restore creates a NEW dataset and opens it.
  const exportBackup = useCallback(() => {
    if (currentId) window.open(`/api/dataset/${currentId}/backup`, '_blank');
  }, [currentId]);

  const importBackup = useCallback(async (file) => {
    const fd = new FormData(); fd.append('file', file);
    const d = await postJson('/api/dataset/backup/import', fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    toast.success(`Dataset « ${d.name} » restored`);
    await fetchList();
    await open(d.id);
  }, [fetchList, open, toast]);

  // Merge an EXISTING training dataset (ZIP of images + kohya-style same-stem
  // .txt captions) into the open dataset — distinct from importBackup (which
  // restores this app's own backup format as a NEW dataset).
  const importDatasetZip = useCallback((file) => wrap(async () => {
    const fd = new FormData(); fd.append('file', file);
    const d = await postJson(`/api/dataset/${currentId}/import-zip`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    const parts = [`${d.imported} imported`];
    if (d.captions) parts.push(`${d.captions} caption(s) attached`);
    if (d.duplicates) parts.push(`${d.duplicates} duplicate(s) skipped`);
    if (d.failed) parts.push(`${d.failed} unreadable`);
    toast.success(parts.join(' · '));
    if (d.small) toast.warning(`${d.small} image(s) under 768 px — they will stay soft in training.`);
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  // Same merge from a FOLDER on this machine's disk (kohya images + same-stem
  // .txt captions) — the path is a server-side path pasted as text, not a
  // browser file pick (a browser can't hand the server a folder path).
  const importDatasetFolder = useCallback((path) => wrap(async () => {
    const d = await postJson(`/api/dataset/${currentId}/import-folder`, { path });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    const parts = [`${d.imported} imported`];
    if (d.captions) parts.push(`${d.captions} caption(s) attached`);
    if (d.duplicates) parts.push(`${d.duplicates} duplicate(s) skipped`);
    if (d.failed) parts.push(`${d.failed} unreadable`);
    toast.success(parts.join(' · '));
    if (d.small) toast.warning(`${d.small} image(s) under 768 px — they will stay soft in training.`);
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  // Restoration layer: fold the server-side `activity` into the visual flags so a
  // reloaded page (which lost the local captioning/analyzing/watermarking state)
  // still shows the concerned button's spinner and disables concurrent actions —
  // exactly as if the click had just happened. The local flags stay authoritative
  // for the user who actually clicked (their fetch flow is untouched); this only
  // ADDS the server truth on top. `busy` OR'd with any activity re-disables every
  // concurrent action and shows the amber "in progress" banner after a reload.
  const activity = data?.activity || null;
  const actKind = activity?.kind || null;
  const captioningLive = captioning || actKind === 'caption' || actKind === 'recaption';
  const analyzingLive = analyzing || actKind === 'analyze_faces';
  const watermarkingLive = watermarking
    || actKind === 'watermark_detect' || actKind === 'watermark_clean';
  const busyLive = busy || !!activity;

  return { datasets, currentId, data, busy: busyLive, captioning: captioningLive,
           analyzing: analyzingLive, watermarking: watermarkingLive, activity,
           nonces, refNonce, create, open,
           deleteDataset, updateSettings, setCurrentId, setRef, addExtraRef, removeExtraRef,
           generate, importFiles, scrapeImport, classify, caption, recaption,
           setStatus, setCaption, crop, cropRef, recropRefAuto, setDatasetTrainType, setDatasetFidelity, deleteImage, batchImages, replaceCaptions, writeCaptionFiles, openDatasetFolder, cancelPending, regenerate, analyzeFaces,
           findWatermarks, cleanWatermarks, cleanWatermarkImages, dismissWatermarks, saveWatermarkRegions,
           purgeUnused, exportZip, exportBackup, importBackup, importDatasetZip, importDatasetFolder, refresh, train, stopTraining, continueTraining,
           listCheckpoints, importCheckpoint, deleteCheckpoint,
           trainBaseInfo, setTrainSettings, prepareBase };
}
