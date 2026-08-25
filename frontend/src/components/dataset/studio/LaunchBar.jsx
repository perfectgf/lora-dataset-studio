// Bouton « 🚀 Lancer le test ». Désactivé tant que !canLaunch (calculé par RunSetupPanel).
// Extrait behavior-preserving de LoraTestStudio.jsx (bouton de lancement).
//
// `label`/`title` : le ◉ LoRA Canvas fait dire au bouton CE QU'IL VA FAIRE quand
// ce n'est pas juste « lancer » (« Deploy 2 checkpoints, then generate ») et
// POURQUOI quand il ne peut pas (familles mélangées). Absents → le libellé
// historique, à l'identique.
export default function LaunchBar({ canLaunch, onLaunch, label = null, title = null }) {
  return (
    <button type="button" disabled={!canLaunch} onClick={onLaunch} title={title || undefined}
      className="ml-auto min-w-0 px-3 py-1.5 rounded-lg bg-gradient-primary text-gray-950 text-sm font-semibold disabled:opacity-40">
      <span aria-hidden>🚀</span> <span className="break-words">{label || 'Run test'}</span>
    </button>
  );
}
