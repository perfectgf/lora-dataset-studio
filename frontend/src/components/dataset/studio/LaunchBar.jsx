// Bouton « 🚀 Lancer le test ». Désactivé tant que !canLaunch (calculé par RunSetupPanel).
// Extrait behavior-preserving de LoraTestStudio.jsx (bouton de lancement).
export default function LaunchBar({ canLaunch, launching, onLaunch }) {
  return (
    <button type="button" disabled={!canLaunch} onClick={onLaunch}
      className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
      <span aria-hidden>🚀</span> Run test
    </button>
  );
}
