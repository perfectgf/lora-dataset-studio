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
import { getCsrfToken } from '../api/fetchClient';

export function useLoraTestStudio(datasetId, family = null) {
  const toast = useToast();
  const [data, setData] = useState(null);
  const [launching, setLaunching] = useState(false);

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
      // (resolution_tier, negative, sampler, scheduler, weight_dtype, rebalance(+_strength),
      // enhancer(+_strength), detail_amount, permanent_loras) — déjà gatés PAR FAMILLE côté
      // serveur ; les champs vides sont absents (le backend garde alors ses défauts).
      const d = await postJson(`/api/dataset/${datasetId}/lora-test/run`,
        { checkpoints, strengths, seed, prompt, z_models: zModels, aspects, cfgs, steps: stepsList, steps2: steps2List, count, family, ...genSettings });
      if (d.ok) toast.success(`${d.created} generation(s) queued (seed ${d.seed}${d.count > 1 ? ` ×${d.count}` : ''})`);
      else toast.error(d.error);
      await refresh();
      return d;
    } finally {
      setLaunching(false);
    }
  }, [datasetId, refresh, toast, family]);

  const rate = useCallback(async (imageId, rating) => {
    const d = await postJson(`/api/dataset/lora-test/image/${imageId}/rate`, { rating });
    if (!d.ok) toast.error(d.error);
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
      if (d.ok) toast.success(`Face scoring done — ${d.scored}/${d.total} cell(s) scored`);
      else toast.error(d.error || 'Scoring failed');
      await refresh();
      return d;
    } finally {
      setScoring(false);
    }
  }, [datasetId, family, refresh, toast]);

  const cancel = useCallback(async () => {
    const d = await postJson(`/api/dataset/${datasetId}/lora-test/cancel`);
    if (d.ok) toast.success(`${d.cancelled} generation(s) stopped — resumable`);
    else toast.error(d.error);
    await refresh();
  }, [datasetId, refresh, toast]);

  const resume = useCallback(async () => {
    const d = await postJson(`/api/dataset/${datasetId}/lora-test/resume`);
    if (d.ok) toast.success(`${d.resumed} cell(s) restarted with their settings`);
    else toast.error(d.error);
    await refresh();
    return d;
  }, [datasetId, refresh, toast]);

  // Persiste la config gagnante COMPLÈTE (pas juste checkpoint+strength) : on
  // passe la cellule entière pour garder modèle/cfg/steps/format.
  const setBest = useCallback(async (cell) => {
    const d = await postJson(`/api/dataset/${datasetId}/lora-test/best`, {
      checkpoint: cell.checkpoint, strength: cell.strength,
      z_model: cell.z_model ?? null, cfg: cell.cfg ?? null,
      steps: cell.steps ?? null, steps2: cell.steps2 ?? null, aspect: cell.aspect ?? null,
    });
    if (d.ok) toast.success('★ Best setting saved');
    else toast.error(d.error);
    await refresh();
    return d;
  }, [datasetId, refresh, toast]);

  // Supprime le réglage mémorisé (DELETE — pas géré par postJson). `fam` cible une
  // famille précise (les autres gardent leur best) ; absent → famille courante du hook.
  const clearBest = useCallback(async (fam) => {
    const f = fam || family;
    const qs = f ? `?family=${encodeURIComponent(f)}` : '';
    const res = await fetch(`/api/dataset/${datasetId}/lora-test/best${qs}`, {
      method: 'DELETE',
      headers: { 'X-CSRFToken': getCsrfToken() },
      credentials: 'include',
    });
    const d = await res.json().catch(() => ({}));
    if (d.ok) toast.success('Saved setting removed'); else toast.error(d.error || 'Error');
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

  return { data, refresh, launch, rate, cancel, resume, setBest, clearBest, deletePrompt, launching, scoreFaces, scoring };
}
