// 🔤 Case « Trigger word » — UNE préférence de navigateur, partagée par toutes
// les surfaces qui lancent (panneau du Test Studio, page Compare, panneau du
// canvas) : décocher ici vaut partout. Défaut = INJECTER (comportement
// historique), et la clé n'est écrite QUE décochée pour que l'état par défaut
// ne laisse aucune trace en stockage. Module pur exprès : les panneaux ne
// touchent pas au stockage eux-mêmes (le contrat du lot de prompts interdit
// toute persistance dans RunSetupPanel — voir prompt-batch-contract).
const KEY = 'studioInjectTrigger';

export function readInjectTrigger() {
  try { return window.localStorage.getItem(KEY) !== '0'; } catch { return true; }
}

export function writeInjectTrigger(v) {
  try {
    if (v) window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, '0');
  } catch { /* stockage indisponible → préférence de session seulement */ }
}
