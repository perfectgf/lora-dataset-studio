import { useEffect, useState } from 'react';
import { useI18n } from '../../i18n/I18nContext';

const LS_KEY = 'guidedCardCollapsed';

export default function NextStepCard({ step, trainMode, busy, totalImages, onAction, actionLabel }) {
  const { t } = useI18n();
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem(LS_KEY) === '1');
  const forceOpen = totalImages === 0;
  const open = forceOpen || !collapsed;
  useEffect(() => { if (!forceOpen) localStorage.setItem(LS_KEY, collapsed ? '1' : '0'); },
    [collapsed, forceOpen]);
  if (!step) return null;
  const key = step.id === 'finish' ? (trainMode ? 'finish_train' : 'finish_export') : step.id;

  return (
    <div role="status" className="rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-3 py-2">
      <div className="flex items-center gap-2">
        <span aria-hidden>💡</span>
        <span className="text-content text-sm font-semibold">
          {t('workspace.guided.nextStep')}: {t(`workspace.guided.labels.${step.id}`)}
        </span>
        {!forceOpen && (
          <button type="button" onClick={() => setCollapsed((v) => !v)} aria-expanded={open}
            className="ml-auto text-content-subtle hover:text-content px-1"
            title={open ? t('workspace.guided.collapseTitle') : t('workspace.guided.expandTitle')}>
            <span aria-hidden>{open ? '▾' : '▸'}</span>
            <span className="sr-only">
              {open ? t('workspace.guided.collapse') : t('workspace.guided.expand')}
            </span>
          </button>
        )}
      </div>
      {open && (
        <div className="mt-1 flex items-center gap-3 flex-wrap">
          <p className="text-content-muted text-sm m-0">{t(`workspace.guided.copy.${key}`)}</p>
          <button type="button" onClick={onAction} disabled={busy}
            className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {actionLabel}
          </button>
        </div>
      )}
    </div>
  );
}
