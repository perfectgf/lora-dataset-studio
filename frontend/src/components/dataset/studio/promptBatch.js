/**
 * 📝 Lot de prompts — rejouer PLUSIEURS entrées de l'historique en un lancement.
 *
 * L'historique des prompts est le même composant sur les deux surfaces de
 * génération (le Studio de test du dataset et le panneau « Generate from the
 * board » du canvas), parce que les deux montent RunSetupPanel. Ces quatre
 * fonctions sont donc la totalité de la règle du lot, à un seul endroit : ce que
 * le compteur annonce, ce que le bouton dit, et ce que le POST emporte.
 *
 * Le lot est un AXE côté serveur, pas N lancements : un second POST serait
 * refusé (« a test run is already in progress ») et le GPU est sérialisé de
 * toute façon. Une passe par prompt coché, mêmes checkpoints, mêmes réglages,
 * même seed — c'est ce qui rend deux prompts comparables.
 *
 * Rien de coché ⇒ chaque fonction rend EXACTEMENT l'ancien comportement (pas de
 * clé `prompts` dans le corps, pas de multiplicateur, libellé inchangé).
 */

/** Les textes de l'historique, quelle que soit sa forme (rétro-compat : l'API
 *  renvoyait des strings avant redémarrage de Flask, des objets après). */
export function promptTexts(recentPrompts) {
  return (recentPrompts || [])
    .map((p) => (typeof p === 'string' ? p : p && p.prompt))
    .filter((p) => typeof p === 'string' && p !== '');
}

/** Le lot RÉELLEMENT lançable : un prompt supprimé de l'historique le quitte
 *  tout seul. On ne lance jamais sur une ligne que l'écran ne montre plus. */
export function visibleBatch(batch, recentPrompts) {
  const available = promptTexts(recentPrompts);
  return (batch || []).filter((p) => available.includes(p));
}

/** 🌐 Le lot peut aussi recevoir des prompts qui ne sont PAS (encore) dans
 *  l'historique — ceux cochés dans le navigateur Civitai. Ils s'ajoutent
 *  APRÈS les prompts d'historique cochés, sans doublon : un prompt Civitai déjà
 *  rejoué une fois est dans l'historique, et le cocher des deux côtés doit
 *  compter UNE passe, pas deux. Après le lancement il entre dans l'historique
 *  comme les autres. */
export function mergeBatches(historyPicked, extraPicked) {
  // Le dédoublonnage se fait sur la chaîne AJUSTÉE, alors qu'on renvoie
  // l'ORIGINALE : c'est ce que fait le moteur (`_prompt_axis` strippe puis
  // dédoublonne), et deux règles différentes ici et là-bas donneraient un
  // compteur qui promet une cellule que le run ne rendra pas. Renvoyer la
  // chaîne d'origine garde un lot d'une seule source octet pour octet.
  const seen = new Set();
  const out = [];
  for (const p of [...(historyPicked || []), ...(extraPicked || [])]) {
    if (typeof p !== 'string') continue;
    const key = p.trim();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(p);
  }
  return out;
}

/** Le corps du lancement. La clé `prompts` voyage dans le même objet que les
 *  réglages globaux — les deux hooks (Test Studio et canvas) étalent cet objet
 *  dans leur POST, donc le lot atteint les deux routes sans changer une seule
 *  signature. Lot vide ⇒ l'objet est rendu tel quel, pas une copie enrichie :
 *  le corps envoyé reste celui d'avant. */
export function launchSettings(genSettings, picked) {
  const list = picked || [];
  return list.length ? { ...(genSettings || {}), prompts: [...list] } : genSettings;
}

/** Ce que le bouton dit. Il garde le verbe de SA surface (« Run test » ici,
 *  « Deploy 2 checkpoints, then generate » sur le board) et y ajoute ce que le
 *  lot change — jamais un bouton qui lance neuf images en disant « lancer ». */
export function launchText(baseLabel, picked) {
  const n = (picked || []).length;
  return n > 1 ? `${baseLabel || 'Run test'} · ${n} prompts` : (baseLabel ?? null);
}
