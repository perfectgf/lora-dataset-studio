// react-frontend/src/components/dataset/studio/StudioPreflightBanner.jsx
/**
 * Bandeau « le pipeline de test ne peut pas tourner » — affiché quand le lancement
 * d'une grille Studio renvoie un 409 `studio_missing` (P0-a). Même esprit que le
 * message Klein « place X ici », mais itemisé : chaque fichier modèle manquant avec
 * son chemin relatif attendu (models/vae/…) et chaque custom node absent du ComfyUI
 * cible. Sans ça, un utilisateur frais lançait une grille dont chaque tuile échouait
 * en silence. Dismissable (le prochain lancement le réémet si le manque persiste).
 *
 * `missing` = { family, files: [{path, kind}], nodes: [class_type],
 *   node_packs: [{class_type, pack, url, search}] } | null. `node_packs` names the
 *   ComfyUI-Manager pack (+ search term & link) for each recognised missing node, so
 *   the user knows WHAT to install instead of reverse-mapping a raw class_type.
 * `archMismatch` = { family, detected, checkpoint } | null — a selected checkpoint
 * whose REAL architecture (read from its header) isn't this Studio's family, so
 * ComfyUI would silently drop it and every tile would render as if the LoRA were
 * off. A distinct, higher-priority stop than a missing asset.
 */
const FAMILY_LABELS = { zimage: 'Z-Image', sdxl: 'SDXL', krea: 'Krea 2 Turbo',
  flux: 'FLUX.1', flux2klein: 'FLUX.2 Klein' };
import { useI18n } from '../../../i18n/I18nContext';

export default function StudioPreflightBanner({ missing, archMismatch, onDismiss }) {
  const { t } = useI18n();
  if (archMismatch) {
    const fam = FAMILY_LABELS[archMismatch.family] || archMismatch.family || 'this';
    const det = FAMILY_LABELS[archMismatch.detected] || archMismatch.detected || 'a different';
    const name = (archMismatch.checkpoint || '').replace(/\\/g, '/').split('/').pop();
    return (
      <div role="alert"
        className="rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2.5 text-sm text-amber-200 flex items-start gap-2">
        <span aria-hidden className="text-base leading-none">⚠</span>
        <p className="m-0">{t('studio.preflight.mismatch', { name, detected: det, family: fam })}</p>
        {onDismiss && (
          <button type="button" onClick={onDismiss} aria-label={t('common.close')}
            className="ml-auto px-1.5 leading-none text-amber-200/70 hover:text-amber-100">×</button>
        )}
      </div>
    );
  }
  if (!missing) return null;
  const files = missing.files || [];
  const nodes = missing.nodes || [];
  const nodePacks = missing.node_packs || [];
  const packFor = (ct) => nodePacks.find((p) => p.class_type === ct);
  if (!files.length && !nodes.length) return null;
  const fam = FAMILY_LABELS[missing.family] || missing.family || 'This';

  return (
    <div role="alert"
      className="rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2.5 text-sm text-red-200 flex flex-col gap-2">
      <div className="flex items-start gap-2">
        <span aria-hidden className="text-base leading-none">⚠</span>
        <p className="m-0 font-semibold">{t('studio.preflight.missing', { family: fam })}</p>
        {onDismiss && (
          <button type="button" onClick={onDismiss} aria-label={t('common.close')}
            className="ml-auto px-1.5 leading-none text-red-200/70 hover:text-red-100">×</button>
        )}
      </div>

      {files.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-red-200/80 text-[0.6875rem] uppercase tracking-wide">
            {t('studio.preflight.files', { count: files.length })}
          </span>
          <ul className="m-0 flex flex-col gap-0.5">
            {files.map((f) => (
              <li key={f.path} className="flex items-baseline gap-2">
                <code className="text-red-100 text-[0.6875rem] break-all">{f.path}</code>
                <span className="text-red-200/60 text-[0.625rem] whitespace-nowrap">({f.kind})</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {nodes.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-red-200/80 text-[0.6875rem] uppercase tracking-wide">
            {t('studio.preflight.nodes', { count: nodes.length })}
          </span>
          <ul className="m-0 flex flex-col gap-1">
            {nodes.map((n) => {
              const p = packFor(n);
              return (
                <li key={n} className="flex flex-col gap-0.5">
                  <code className="self-start px-1.5 py-0.5 rounded border border-red-400/40 bg-red-500/10 text-red-100 text-[0.6875rem]">
                    {n}
                  </code>
                  {p && (
                    <span className="text-red-200/70 text-[0.625rem]">
                      Install <b className="font-semibold">{p.pack}</b> via ComfyUI-Manager
                      {p.search ? <> (search “{p.search}”)</> : null} —{' '}
                      <a href={p.url} target="_blank" rel="noreferrer"
                        className="underline break-all hover:text-red-100">{p.url}</a>
                      , then restart ComfyUI.
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
