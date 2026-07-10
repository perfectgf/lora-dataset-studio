/**
 * useDataset — Face Dataset Maker data hook.
 * Loads the dataset list + the open dataset payload, polls while generation
 * jobs are pending, and exposes all mutations (create/ref/generate/import/
 * classify/caption/status/caption-edit/crop/regenerate/export).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { getCsrfToken } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { useJobs } from '../context/JobsContext';

function post(url, body, isForm) {
  return fetch(url, {
    method: 'POST',
    credentials: 'include',
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
    try { d = await r.json(); } catch { /* non-JSON body (proxy page, empty) */ }
    if (!r.ok) return { ok: false, error: (d && d.error) || `Server error (${r.status})` };
    return d || { ok: true };
  } catch (e) {
    return { ok: false, error: e.message || 'Network error' };
  }
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

  const open = useCallback(async (id) => { setCurrentId(id); await refresh(id); }, [refresh]);

  const create = useCallback(async (name, trigger, kind, conceptDesc, trainType) => {
    const d = await postJson('/api/dataset/create',
      { name, trigger_word: trigger, ...(kind ? { kind } : {}),
        ...(trainType ? { train_type: trainType } : {}),
        ...(kind === 'concept' && conceptDesc ? { concept_desc: conceptDesc } : {}) });
    if (d.ok) { await fetchList(); await open(d.id); toast.success('Dataset created'); }
    else toast.error(d.error || 'Unexpected error');
  }, [fetchList, open, toast]);

  // Change the target model family later (from the TrainingPanel selector) so the
  // grouped menu re-sorts. Refreshes the list; silent on failure (non-critical).
  const setDatasetTrainType = useCallback(async (trainType) => {
    if (!currentId) return;
    const d = await postJson(`/api/dataset/${currentId}/train-type`, { train_type: trainType });
    if (d.ok) fetchList();
  }, [currentId, fetchList]);

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

  const setRef = useCallback((file) => wrap(async () => {
    const fd = new FormData(); fd.append('file', file);
    const d = await postJson(`/api/dataset/${currentId}/ref`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    // GUARD-RAIL: the backend head-crop can silently fall back to a centered crop
    // (e.g. vision model not pulled). Surface its reason instead of a plain success.
    if (d.warning) toast.warning(d.warning);
    else toast.success('Reference set');
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

  const importFiles = useCallback((files) => wrap(async () => {
    const fd = new FormData(); [...files].forEach((f) => fd.append('files', f));
    const d = await postJson(`/api/dataset/${currentId}/import`, fd, true);
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return; }
    const dup = d.duplicates || 0;
    toast.success(`${d.imported} imported${dup ? ` · ${dup} duplicate(s) skipped` : ''}`);
    if (dup && !d.imported) toast.warning('All files were already in the dataset (perceptual duplicates).');
    await refresh();
  }), [wrap, currentId, refresh, toast]);

  // Concept only : télécharge les images scannées SÉLECTIONNÉES ({url,title}[])
  // directement dans le dataset (route /scrape-import). Retourne la réponse pour que
  // le panneau vide sa sélection sur succès ; toast détaillé (imported + skipped).
  const scrapeImport = useCallback((items) => wrap(async () => {
    const d = await postJson(`/api/dataset/${currentId}/scrape-import`, { items });
    if (!d.ok) { toast.error(d.error || 'Unexpected error'); return d; }
    const s = d.skipped || {};
    const skips = Object.entries(s).filter(([, v]) => v > 0).map(([k, v]) => `${v} ${k}`).join(', ');
    toast.success(`${d.imported} imported${skips ? ` · skipped ${skips}` : ''}`);
    await refresh();
    return d;
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
      const grey = (d.states?.too_small || 0) + (d.states?.no_face || 0)
        + (d.states?.extreme_pose || 0) + (d.states?.low_det || 0);
      toast.success(`${d.analyzed} analyzed · ${d.states?.scorable || 0} scored, ${grey} not scorable`);
      await refresh();
    } finally {
      setAnalyzing(false);
    }
  }), [wrap, currentId, refresh, toast]);

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
  // AND failed tiles — it is the recovery path for failures.
  const regenerate = useCallback(async (imageId, loraStrength) => {
    const d = await postJson(`/api/dataset/image/${imageId}/regenerate`,
      { lora_strength: loraStrength });
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
        // Masked training (fond à 10 %) — défaut ON, toggle dans TrainingPanel.
        masked: opts.masked !== false,
        // Cible de steps absolue (plafond choisi dans TrainingPanel) — omise si
        // vide → le backend calcule la valeur adaptative (recommended_steps).
        ...(opts.steps ? { steps: opts.steps } : {}) });
    // L'entraînement tourne en CLI headless (pas l'UI ai-toolkit) → on N'OUVRE PAS
    // localhost:8675 (lien mort). La progression se suit ici (checkpoints + statut).
    if (d.ok) toast.success(`Training started (${d.steps || '?'} steps) — ComfyUI paused, follow the checkpoints here`);
    // Le mismatch caption↔type est géré par un confirm dans TrainingPanel (pas un toast d'erreur).
    else if (!String(d.error || '').includes('MISMATCH_CAPTION')) toast.error(d.error || 'Unexpected error');
    return d;
  }, [currentId, toast]);

  // Bases entraînables + base/variante choisies + statut de conversion.
  const trainBaseInfo = useCallback(async () => {
    const r = await fetch(`/api/dataset/${currentId}/train/base-info`, { credentials: 'include' });
    return r.ok ? await r.json() : null;
  }, [currentId]);

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
    const d = await postJson(`/api/dataset/${currentId}/train/import`, body);
    if (d.ok) toast.success(`LoRA imported: ${d.dest}`); else toast.error(d.error || 'Unexpected error');
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

  return { datasets, currentId, data, busy, captioning, nonces, refNonce, create, open,
           deleteDataset, setCurrentId, setRef, addExtraRef, removeExtraRef,
           generate, importFiles, scrapeImport, classify, caption, recaption,
           setStatus, setCaption, crop, cropRef, recropRefAuto, setDatasetTrainType, deleteImage, batchImages, replaceCaptions, cancelPending, regenerate, analyzing, analyzeFaces,
           purgeUnused, exportZip, refresh, train, stopTraining, continueTraining,
           listCheckpoints, importCheckpoint, deleteCheckpoint,
           trainBaseInfo, prepareBase };
}
