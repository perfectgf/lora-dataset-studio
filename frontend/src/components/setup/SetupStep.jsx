import { useI18n } from '../../i18n/I18nContext'

const STATUS_META = {
  ready: { glyph: '✓', cls: 'text-emerald-400' },
  partial: { glyph: '◐', cls: 'text-amber-400' },
  available: { glyph: '○', cls: 'text-content-subtle' },
  skipped: { glyph: '–', cls: 'text-content-subtle' },
}

export default function SetupStep({ step, index, effectiveStatus, onSkip, onUnskip, children }) {
  const { t } = useI18n()
  const meta = STATUS_META[effectiveStatus] || STATUS_META.available
  const collapsed = effectiveStatus === 'ready' || effectiveStatus === 'skipped'
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-content">
            {index}. {step.title}
            {step.recommended && (
              <span className="ml-2 rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                {t('setup.recommended')}
              </span>
            )}
          </h2>
          <p className="mt-1 text-xs text-content-muted">
            {t('setup.unlocks', { items: step.unlocks.join(' · ') })}
          </p>
        </div>
        <span className={`inline-flex shrink-0 items-center gap-1 text-xs font-medium ${meta.cls}`}>
          <span aria-hidden="true">{meta.glyph}</span>{t(`setup.status.${effectiveStatus || 'available'}`)}
        </span>
      </div>
      {!collapsed && <div className="mt-4 space-y-4">{children}</div>}
      {!step.recommended && effectiveStatus !== 'ready' && (
        <div className="mt-3 text-right">
          {effectiveStatus === 'skipped'
            ? <button type="button" onClick={onUnskip} className="text-xs text-primary underline">{t('setup.setUp')}</button>
            : <button type="button" onClick={onSkip} className="text-xs text-content-subtle underline">{t('setup.skipForNow')}</button>}
        </div>
      )}
    </section>
  )
}
