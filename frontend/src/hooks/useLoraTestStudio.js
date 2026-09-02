/**
 * useLoraTestStudio — data hook of the « Studio de test de LoRA ».
 *
 * Polls /api/dataset/<id>/lora-test/status (3 s while cells are pending, same
 * rhythm as the dataset fan-out) and exposes the mutations: launch run, rate
 * a cell 👍/👎, cancel the run, persist the best settings.
 */
import { useCallback, useEffect, useState } from 'react';
import { useToast } from '../components/common/Toast';
import { postJson } from './useDataset';
import { del } from '../api/fetchClient';

export function useLoraTestStudio(datasetId, family = null) {
  const toast = useToast();
  const [data, setData] = useState(null);
  const [launching, setLaunching] = useState(false);

  const [confirmingComfyuiRestart, setConfirmingComfyuiRestart] = useState(false);
  const refresh = useCallback(async () => {
    if (!datasetId) return;
    try {
      // `family` scope la pipeline (ZIT/SDXL/Krea) ; absent → défaut résolu côté serveur.
      const qs = family ? `?family=${encodeURIComponent(family)}` : '';
      const r = await fetch(`/api/dataset/${datasetId}/lora-test/status${qs}`, { credentials: 'include' });
      if (r.ok) setData(await r.json());
    } catch { /* transient network error — the poll retries */ }
  }, [datasetId, family]);

  // Vide la grille DÈS que le dataset change : sinon on continue d'afficher les
  // cellules du LoRA précédent tant que le refetch n'a pas répondu (et si le
  // fetch échoue, ça reste bloqué sur l'autre LoRA — ex. eva6938 dans le studio
  // d'un autre dataset).
  useEffect(() => { setData(null); }, [datasetId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll while generations are in flight (pending cells fill the grid live).
  useEffect(() => {
    if (!data?.pending) return undefined;
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [data, refresh]);

  const launch = useCallback(async (checkpoints, strengths, seed, prompt, zModels, aspects, cfgs, stepsList, steps2List, count = 1, genSettings = {}) => {
    setLaunching(true);
    try {
      // `genSettings` = réglages GLOBAUX snake_case remontés par StudioGenerationSettings
      // (resolution_tier, negative, sampler, scheduler, weight_dtype,
      // detail_amount, permanent_loras) — déjà gatés PAR FAMILLE côté
      // serveur ; les champs vides sont absents (le backend garde alors ses défauts).
      const d = await postJson(`/api/dataset/${datasetId}/lora-test/run`,
        { checkpoints, strengths, seed, prompt, z_models: zModels, aspects, cfgs, steps: stepsList, steps2: steps2List, count, family, ...genSettings });
      if (d.ok) toast.success(`${d.created} generation(s) queued (seed ${d.seed}${d.count > 1 ? ` ×${d.count}` : ''})`);
      else toast.error(d.error || 'Unexpected error');
      await refresh();
      return d;
    } finally {
      setLaunching(false);
    }
  }, [datasetId, refresh, toast, family]);

  const rate = useCallback(async (imageId, rating) => {
    const d = await postJson(`/api/dataset/lora-test/image/${imageId}/rate`, { rating });
    if (!d.ok) toast.error(d.error || 'Unexpected error');
    await refresh();
  }, [refresh, toast]);

  // Scoring facial objectif (« best epoch » auto) : InsightFace CPU côté serveur,
  // puis refresh → le payload porte face_ranking + face_score par cellule.
  const [scoring, setScoring] = useState(false);
  const scoreFaces = useCallback(async () => {
    setScoring(true);
    try {
      const d = await postJson(`/api/dataset/${datasetId}/lora-test/score-faces`,
        family ? { family } : {});
      // Un scorer cassé disait « done — 0/14 » en VERT (user-reported) : le
      // backend remonte maintenant scoring_error {kind, detail} — dire POURQUOI.
      if (!d.ok) toast.error(d.error || 'Scoring failed');
      else if (d.scoring_error) {
        const { kind, detail } = d.scoring_error;
        toast.error(kind === 'unavailable'
          ? 'Face scoring is not installed — run the Quality tools step in Setup.'
          : kind === 'ref_unusable'
            ? `The reference photo is not usable for scoring: ${detail}`
            : `Face scoring failed: ${detail}`);
      } else if (!d.total) {
        toast.info('Nothing to score yet — run a test with several checkpoints (same seed) first.');
      } else {
        toast.success(`Face scoring done — ${d.scored}/${d.total} cell(s) scored`);
      }
      await refresh();
      return d;
    } finally {
      setScoring(false);
    }
  }, [datasetId, family, refresh, toast]);

  const cancel = useCallback(async () => {
    const d = await postJson(`/api/dataset/${datasetId}/lora-test/cancel`);
    if (d.ok) toast.success(`${d.cancelled} generation(s) stopped — resumable`);
    else toast.error(d.error || 'Unexpected error');
    await refresh();
  }, [datasetId, refresh, toast]);

  const resume = useCallback(async () => {
    const d = await postJson(`/api/dataset/${datasetId}/lora-test/resume`);
    if (d.ok) toast.success(`${d.resumed} cell(s) restarted with their settings`);
    else toast.error(d.error || 'Unexpected error');
    await refresh();
    return d;
  }, [datasetId, refresh, toast]);

  // Unknown ComfyUI submissions are never retried automatically: the endpoint
  // probes the restarted process, clears only that paused cell, then refreshes
  // so the normal Resume button is the next explicit action.
  const confirmComfyuiRestart = useCallback(async () => {
    if (confirmingComfyuiRestart) return undefined;
    setConfirmingComfyuiRestart(true);
    try {
      const d = await postJson(`/api/dataset/${datasetId}/lora-test/confirm-comfyui-restart`, {
        confirmed_comfyui_restart: true,
      });
      if (d.ok) toast.success('ComfyUI restart confirmed — the paused cell is ready to resume.');
      else toast.error(d.error || 'Could not confirm the ComfyUI restart');
      await refresh();
      return d;
    } finally {
      setConfirmingComfyuiRestart(false);
    }
  }, [datasetId, confirmingComfyuiRestart, refresh, toast]);

  // Persiste la config gagnante COMPLÈTE (pas juste checkpoint+strength) : on
  // passe la cellule entière pour garder modèle/cfg/steps/format.
  const setBest = useCallback(async (cell) => {
    const d = await postJson(`/api/dataset/${datasetId}/lora-test/best`, {
      checkpoint: cell.checkpoint, strength: cell.strength,
      z_model: cell.z_model ?? null, cfg: cell.cfg ?? null,
      steps: cell.steps ?? null, steps2: cell.steps2 ?? null, aspect: cell.aspect ?? null,
    });
    if (d.ok) toast.success('★ Best setting saved');
    else toast.error(d.error || 'Unexpected error');
    await refresh();
    return d;
  }, [datasetId, refresh, toast]);

  // Supprime le réglage mémorisé via del() du client partagé (même récupération
  // CSRF que toute mutation). `fam` cible une famille précise (les autres
  // gardent leur best) ; absent → famille courante du hook.
  const clearBest = useCallback(async (fam) => {
    const f = fam || family;
    const qs = f ? `?family=${encodeURIComponent(f)}` : '';
    let d;
    try {
      d = await del(`/api/dataset/${datasetId}/lora-test/best${qs}`);
      if (d.ok) toast.success('Saved setting removed'); else toast.error(d.error || 'Error');
    } catch (e) {
      d = { ok: false, error: e.message };
      toast.error(e.message || 'Error');
    }
    await refresh();
    return d;
  }, [datasetId, refresh, toast, family]);

  // Supprime un prompt récent + ses cellules/images de test — sur TOUS les
  // datasets du user (la liste des prompts récents est désormais GLOBALE).
  const deletePrompt = useCallback(async (prompt) => {
    const d = await postJson('/api/studio/recent-prompts/delete', { prompt });
    if (d.ok) toast.success(`Prompt deleted (${d.deleted} image(s))`); else toast.error(d.error || 'Error');
    await refresh();
    return d;
  }, [refresh, toast]);

  return { data, refresh, launch, rate, cancel, resume, confirmComfyuiRestart, confirmingComfyuiRestart, setBest, clearBest, deletePrompt, launching, scoreFaces, scoring };
}
