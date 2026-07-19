import { Link } from 'react-router-dom';
import { useI18n } from '../../i18n/I18nContext';

/* Vertical, numbered progress checklist — the workspace's left sidebar. Same step
   data as the old horizontal GuidedStepper (from useGuidedFlow): every step is a
   button that scrolls to its section; an unavailable step (e.g. Studio without
   ComfyUI) links to Settings with its hint. Status is glyph + text, never
   color-only, so it stays legible without relying on hue (a11y). */
export default function GuidedChecklist({ steps, currentId, onJump }) {
  const { t } = useI18n();
  return (
    <nav aria-label={t('workspace.guided.progress')}
      className="rounded-lg border border-border bg-surface p-2">
      <p className="px-1.5 pb-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
        {t('workspace.guided.progress')}
      </p>
      <ol className="flex flex-col gap-0.5">
        {steps.map((s, i) => {
          const glyph = s.done ? '✓' : s.busy ? '⏳' : s.id === currentId ? '◉' : '○';
          const tone = s.unavailable ? 'text-content-subtle opacity-60'
            : s.done ? 'text-emerald-400'
            : s.id === currentId ? 'text-content font-semibold'
            : 'text-content-muted';
          const cls = `flex items-center gap-1.5 w-full px-1.5 py-1 rounded text-[0.8125rem] text-left hover:bg-surface-raised transition-colors ${tone}`;
          const body = (
            <>
              <span aria-hidden className="w-4 shrink-0 text-center">{glyph}</span>
              <span aria-hidden className="shrink-0 tabular-nums text-content-subtle">{i + 1}.</span>
              <span className="truncate">{t(`workspace.guided.labels.${s.id}`)}</span>
              {s.optional && <span className="shrink-0 text-[0.6875rem] text-content-subtle">({t('workspace.guided.optional')})</span>}
              {s.subtitle && (
                <span className="ml-auto pl-1 shrink-0 max-w-[6.5rem] truncate text-[0.6875rem] text-content-subtle">
                  {s.subtitle}
                </span>
              )}
            </>
          );
          return (
            <li key={s.id}>
              {s.unavailable ? (
                <Link to="/settings" title={s.hint} className={cls}>
                  {body}
                  <span aria-hidden className="shrink-0 pl-0.5">⚙</span>
                  <span className="sr-only"> — {s.hint}</span>
                </Link>
              ) : (
                <button type="button" onClick={() => onJump(s)}
                  aria-current={s.id === currentId ? 'step' : undefined}
                  title={s.subtitle || s.label} className={cls}>
                  {body}
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
