import { useI18n } from '../../../i18n/I18nContext';

// Bouton « 🚀 Lancer le test ». Désactivé tant que !canLaunch (calculé par RunSetupPanel).
// Extrait behavior-preserving de LoraTestStudio.jsx (bouton de lancement).
export default function LaunchBar({ canLaunch, launching, onLaunch }) {
  const { t } = useI18n();
  return (
    <button type="button" disabled={!canLaunch} onClick={onLaunch}
      className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
      <span aria-hidden>🚀</span> {launching ? t('studio.actions.launching') : t('studio.actions.runTest')}
    </button>
  );
}
