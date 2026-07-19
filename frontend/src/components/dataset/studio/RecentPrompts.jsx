// Vignettes des prompts récents (clic pour recharger) — rétro-compat string vs objet.
// Extrait behavior-preserving de LoraTestStudio.jsx (bloc « Prompts récents »),
// + bouton 🗑 par preset (supprime le prompt et ses cellules/images de test).
import { useI18n } from '../../../i18n/I18nContext';

export default function RecentPrompts({ items, datasetId, selectedPrompt, onPick, onDelete }) {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-1">
      <span className="text-content-subtle text-[0.5625rem] uppercase">{t('studio.recent.title')}</span>
      <div className="flex gap-1.5 flex-wrap">
        {items.map((item) => {
          // rétro-compat : avant restart Flask, l'API renvoie des strings ;
          // après, des objets {prompt, thumbnail, thumb_rating, count}.
          const pr = typeof item === 'string' ? { prompt: item } : item;
          const sel = selectedPrompt === pr.prompt;
          // Conteneur = la « carte » (porte la bordure) ; deux boutons frères à
          // l'intérieur (PAS imbriqués) : recharger (principal) + supprimer (🗑).
          return (
            <div key={pr.prompt}
              className={`flex items-stretch rounded-lg border text-[0.625rem] max-w-[260px] overflow-hidden ${
                sel ? 'border-purple-400/60 bg-purple-500/20' : 'border-border bg-surface'}`}>
              <button type="button" onClick={() => onPick(pr.prompt)} title={pr.prompt}
                className={`flex items-center gap-1.5 p-1 text-left min-w-0 ${
                  sel ? 'text-purple-200' : 'text-content-muted'}`}>
                {pr.thumbnail
                  ? <img src={`/api/dataset/${pr.thumb_dataset_id ?? datasetId}/img/${encodeURIComponent(pr.thumbnail)}`}
                      alt="" loading="lazy"
                      className="w-8 h-10 object-cover rounded shrink-0" />
                  : <span className="w-8 h-10 rounded bg-app/60 shrink-0 flex items-center justify-center text-content-subtle">?</span>}
                <span className="flex flex-col items-start min-w-0">
                  <span className="truncate max-w-[150px]">{pr.prompt}</span>
                  {pr.count ? <span className="text-content-subtle">{t('studio.results.imageCount', { count: pr.count })}{pr.thumb_rating === 1 ? ' · 👍' : ''}</span> : null}
                </span>
              </button>
              {onDelete && (
                <button type="button"
                  onClick={() => {
                    if (window.confirm(t('studio.recent.deleteConfirm', { count: pr.count || 0 }))) onDelete(pr.prompt);
                  }}
                  title={t('studio.recent.deleteTitle')}
                  aria-label={t('studio.recent.deleteLabel')}
                  className="shrink-0 px-1.5 flex items-center border-l border-border text-red-300/70 hover:text-red-300 hover:bg-red-500/15">
                  🗑
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
