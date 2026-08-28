// react-frontend/src/components/dataset/studio/enhanceGate.js
/**
 * Décide si le bouton « ✨ Enhance » est utilisable, et POURQUOI il ne l'est pas.
 *
 * Extrait du JSX pour être testé sous `node --test`. La règle : sur une install
 * SANS Ollama le bouton est désactivé avec la raison exacte en infobulle — jamais
 * un appel qui part et échoue en silence. Les trois états viennent de
 * /api/capabilities (`caps.ollama`), déjà publié pour les surfaces Bank/Settings.
 */

/** Raison de blocage (string, anglais) ou null si Enhance est utilisable.
 * `customModel` = le modèle choisi dans la ⚙️ ('' = défaut) : quand il est posé,
 * l'état « modèle par défaut pas téléchargé » ne bloque plus — l'appel ne s'en
 * sert pas, et le serveur vérifie le modèle choisi lui-même (409 qui le nomme). */
export function enhanceBlocker(ollama, { capsLoading = false, customModel = '' } = {}) {
  if (capsLoading) return 'Checking local tools…';
  const o = ollama || {};
  if (!o.installed && !o.reachable) {
    return 'Enhance needs Ollama — install it from Settings › Local tools.';
  }
  if (!o.reachable) {
    return 'Ollama is installed but not running — start it from Settings › Local tools.';
  }
  if (!customModel && !o.vision_model_ready) {
    return `Ollama model "${o.vision_model || 'unset'}" is not downloaded yet `
      + '— pull it from Settings › Local tools.';
  }
  return null;
}
