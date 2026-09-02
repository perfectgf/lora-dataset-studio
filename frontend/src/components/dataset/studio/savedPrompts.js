/* 📚 Les règles pures de l'historique des prompts de test — normalisation et
 * recherche. Elles vivent ici, comme celles du lot dans promptBatch.js, pour la
 * même raison : un module pur est réellement testable, et les DEUX surfaces de
 * lancement (Studio du dataset, « Generate from the board ») partagent la même
 * règle au lieu de la réécrire chacune de leur côté.
 */

/** Une entrée d'historique, quelle que soit la forme rendue par l'API.
 *
 * Rétro-compat : avant restart Flask, `/api/studio/recent-prompts` renvoie des
 * STRINGS ; après, des objets `{prompt, thumbnail, thumb_dataset_id,
 * thumb_rating, count}`. Tout ce qui lit l'historique passe par ici pour que le
 * cas « string » ne soit traité qu'à un seul endroit.
 */
export function normalizeSavedPrompt(item) {
  const raw = typeof item === 'string' ? { prompt: item } : (item || {});
  return {
    prompt: raw.prompt || '',
    thumbnail: raw.thumbnail || null,
    thumbDatasetId: raw.thumb_dataset_id ?? null,
    liked: raw.thumb_rating === 1,
    count: Number(raw.count) || 0,
  };
}

/** Recherche texte sur l'historique — TOUS les mots doivent être présents.
 *
 * Pourquoi « tous les mots » et pas la sous-chaîne brute : les prompts de test
 * sont longs (médiane ~500 caractères) et se ressemblent par le DÉBUT. Chercher
 * « bathroom mirror » doit trouver la prise de vue voulue même si les deux mots
 * sont à 200 caractères l'un de l'autre — ce qu'une sous-chaîne unique refuse.
 * Requête vide = la liste entière, dans son ordre (récent → ancien).
 */
export function filterSavedPrompts(items, query) {
  const list = Array.isArray(items) ? items : [];
  const words = String(query || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return list;
  return list.filter((item) => {
    const text = normalizeSavedPrompt(item).prompt.toLowerCase();
    return words.every((w) => text.includes(w));
  });
}
