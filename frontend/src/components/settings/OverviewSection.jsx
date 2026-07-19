import { Link } from 'react-router-dom'
import { deriveCapabilitySummary } from '../../hooks/useSetupSteps'
import { useI18n } from '../../i18n/I18nContext'

const FIX_LINKS = [
  { to: '/settings/engines', label: 'imageEngines', hint: 'imageEnginesHint' },
  { to: '/settings/local-tools', label: 'localTools', hint: 'localToolsHint' },
  { to: '/settings/training', label: 'training', hint: 'trainingHint' },
  { to: '/setup', label: 'setupWizard', hint: 'setupWizardHint' },
]

const CAPABILITY_KEYS = {
  'Nano Banana (Gemini)': 'nanoBanana',
  'ChatGPT (gpt-image-2)': 'chatgpt',
  'Klein (local)': 'klein',
  Captioning: 'captioning',
  'Auto-framing & head-crop': 'framing',
  'Face-similarity scoring': 'faceScoring',
  'Person masks': 'masks',
  'Watermark inpainting': 'watermark',
  'LoRA training': 'loraTraining',
  'Test Studio': 'testStudio',
}

/* The health map in full: the sidebar LEDs summarized as tiles, plus where to
   go to fix what's off. Status is glyph + text, never color alone. */
export default function OverviewSection({ caps }) {
  const { t } = useI18n()
  const summary = deriveCapabilitySummary(caps)
  const ready = summary.filter((s) => s.ok).length
  return (
    <div className="space-y-6">
      {!caps.configured && (
        <div role="status" className="rounded-xl border border-primary/40 bg-primary/10 p-4 text-sm text-content">
          <p className="font-medium">{t('settings.overview.setupTitle')}</p>
          <p className="mt-1 text-content-muted">
            {t('settings.overview.setupBefore')}{' '}
            <Link to="/setup" className="font-medium text-sky-300 underline hover:text-sky-200">
              {t('settings.overview.setupLink')}
            </Link>
            {' '}{t('settings.overview.setupAfter')}
          </p>
        </div>
      )}

      <section className="rounded-xl border border-border bg-surface p-5">
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="text-base font-semibold text-content">{t('settings.overview.capabilities')}</h2>
          <span className="font-mono text-xs text-content-subtle">
            {t('settings.overview.readyCount', { ready, total: summary.length })}
          </span>
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {summary.map((s) => (
            <div key={s.label}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-sm">
              <span aria-hidden className={s.ok ? 'text-emerald-400' : 'text-content-subtle'}>{s.ok ? '✓' : '✗'}</span>
              <span className={s.ok ? 'text-content' : 'text-content-muted'}>
                {CAPABILITY_KEYS[s.label]
                  ? t(`settings.overview.capability.${CAPABILITY_KEYS[s.label]}`)
                  : s.label}
              </span>
              <span className="sr-only">({s.ok ? t('common.ready') : t('common.notAvailable')})</span>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-surface p-5">
        <h2 className="text-base font-semibold text-content">{t('settings.overview.whereToFix')}</h2>
        <ul className="mt-3 divide-y divide-border">
          {FIX_LINKS.map((l) => (
            <li key={l.to}>
              <Link to={l.to}
                className="group flex items-baseline justify-between gap-3 py-2.5 no-underline">
                <span className="text-sm font-medium text-content group-hover:underline">
                  {t(`settings.overview.${l.label}`)}
                </span>
                <span className="text-right text-xs text-content-subtle">
                  {t(`settings.overview.${l.hint}`)}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
