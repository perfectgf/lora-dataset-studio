import { useEffect, useState } from 'react'
import { INPUT_CLASS, Card, SecretField } from './primitives'
import { useI18n } from '../../i18n/I18nContext'

// Keep in sync with backend TRAIN_TYPES (face_dataset_service.py) — 'flux' had
// been forgotten here when the FLUX.1 family landed (fixed alongside flux2klein).
const FAMILY_OPTIONS = ['zimage', 'sdxl', 'krea', 'flux', 'flux2klein']

/* First-time walkthrough for renting cloud GPUs — collapsed by default so the
   card stays compact for users who already have a key. */
function VastKeyGuide() {
  const { t } = useI18n()
  const link = 'font-medium text-sky-300 underline hover:text-sky-200'
  return (
    <details className="mb-2 rounded-lg border border-border bg-surface px-3 py-2 open:pb-3">
      <summary className="cursor-pointer select-none text-xs font-medium text-content">
        <span aria-hidden>📖</span> {t('settings.training.vastGuide.summary')}
      </summary>
      <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-xs text-content-muted">
        <li>
          {t('settings.training.vastGuide.step1Before')}{' '}
          <a href="https://cloud.vast.ai/" target="_blank" rel="noreferrer" className={link}>cloud.vast.ai</a>
          {' '}{t('settings.training.vastGuide.step1After')}
        </li>
        <li>
          {t('settings.training.vastGuide.step2Before')}{' '}
          <a href="https://cloud.vast.ai/billing/" target="_blank" rel="noreferrer" className={link}>Billing</a>
          {' '}{t('settings.training.vastGuide.step2Middle')}{' '}
          <strong>Add Credit</strong>
          {t('settings.training.vastGuide.step2After')}
        </li>
        <li>
          {t('settings.training.vastGuide.step3Before')}{' '}
          <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noreferrer" className={link}>Keys</a>
          {' '}{t('settings.training.vastGuide.step3After')}
        </li>
        <li>
          {t('settings.training.vastGuide.step4Before')}{' '}
          <strong>{t('common.test')}</strong>
          {t('settings.training.vastGuide.step4After')}
        </li>
      </ol>
    </details>
  )
}

const vastSecret = (t) => ({
  key: 'VAST_API_KEY', label: t('settings.training.vastKey'), testTarget: 'vast',
  help: t('settings.training.vastKeyHelp'),
  guide: <VastKeyGuide />,
})

function CloudOfferFilter({ id, label, help, checked, onChange }) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
      <div>
        <p id={`${id}-label`} className="text-sm font-medium text-content">{label}</p>
        <p id={`${id}-help`} className="mt-0.5 text-xs text-content-muted">{help}</p>
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={`${id}-label`}
        aria-describedby={`${id}-help`}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${checked ? 'bg-emerald-500' : 'bg-surface ring-1 ring-inset ring-border-strong'}`}
      >
        <span
          aria-hidden
          className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`}
        />
      </button>
    </div>
  )
}

/* Cloud training limits: concurrency cap, offer price ceiling, monthly budget
   and the stall watchdog timeout. Fetches the cloud status ONCE on mount for
   the "Spent this month" info line — no poll, this page is not a dashboard. */
function CloudTrainingCard({ config, setField }) {
  const { t } = useI18n()
  const [spend, setSpend] = useState(null)
  const verifiedOnly = config.cloud?.verified_only ?? true
  const secureCloudOnly = config.cloud?.secure_cloud_only ?? false
  useEffect(() => {
    let alive = true
    // Raw fetch (not apiFetch): this info line is best-effort — a transient
    // 500 must not fire the global error toast over a cosmetic detail.
    fetch('/api/dataset/train/cloud/status', { credentials: 'include' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d && typeof d.month_spend === 'number') setSpend(d.month_spend) })
      .catch(() => { /* info line is best-effort */ })
    return () => { alive = false }
  }, [])
  return (
    <Card title={t('settings.training.cloudTitle')} help={t('settings.training.cloudHelp')}>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label htmlFor="cloud-max-concurrent-runs" className="block text-sm font-medium text-content">
            {t('settings.training.maxRuns')}
          </label>
          <input
            id="cloud-max-concurrent-runs"
            type="number"
            min="1"
            max="10"
            step="1"
            value={config.cloud?.max_concurrent_runs ?? 1}
            onChange={(e) => setField('cloud', 'max_concurrent_runs', parseInt(e.target.value) || 1)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-max-price-per-hour" className="block text-sm font-medium text-content">
            {t('settings.training.maxPrice')}
          </label>
          <input
            id="cloud-max-price-per-hour"
            type="number"
            min="0.1"
            max="5"
            step="0.05"
            value={config.cloud?.max_price_per_hour ?? 0.8}
            onChange={(e) => setField('cloud', 'max_price_per_hour', Math.max(0.1, parseFloat(e.target.value) || 0.1))}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-monthly-budget" className="block text-sm font-medium text-content">
            {t('settings.training.monthlyBudget')}
          </label>
          <input
            id="cloud-monthly-budget"
            type="number"
            min="0"
            step="1"
            value={config.cloud?.monthly_budget_usd ?? 0}
            onChange={(e) => setField('cloud', 'monthly_budget_usd', parseFloat(e.target.value) || 0)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-stall-timeout" className="block text-sm font-medium text-content">
            {t('settings.training.stallTimeout')}
          </label>
          <input
            id="cloud-stall-timeout"
            type="number"
            min="5"
            max="240"
            step="1"
            value={config.cloud?.stall_timeout_minutes ?? 30}
            onChange={(e) => setField('cloud', 'stall_timeout_minutes', parseInt(e.target.value) || 30)}
            className={INPUT_CLASS}
          />
        </div>
        <div>
          <label htmlFor="cloud-min-reliability" className="block text-sm font-medium text-content">
            {t('settings.training.minReliability')}
          </label>
          <input
            id="cloud-min-reliability"
            type="number"
            min="0.9"
            max="0.999"
            step="0.005"
            value={config.cloud?.min_reliability ?? 0.98}
            onChange={(e) => setField('cloud', 'min_reliability', Math.min(0.999, Math.max(0.9, parseFloat(e.target.value) || 0.98)))}
            className={INPUT_CLASS}
          />
          <p className="mt-1 text-[0.6875rem] text-content-subtle">
            {t('settings.training.minReliabilityHelp')}
          </p>
        </div>
      </div>
      <div className="space-y-2">
        <p className="text-sm font-medium text-content">{t('settings.training.offerFilters')}</p>
        <div className="grid gap-2 lg:grid-cols-2">
          <CloudOfferFilter
            id="cloud-verified-only"
            label={t('settings.training.verifiedOnly')}
            help={t('settings.training.verifiedOnlyHelp')}
            checked={verifiedOnly}
            onChange={(value) => setField('cloud', 'verified_only', value)}
          />
          <CloudOfferFilter
            id="cloud-secure-cloud-only"
            label={t('settings.training.secureOnly')}
            help={t('settings.training.secureOnlyHelp')}
            checked={secureCloudOnly}
            onChange={(value) => setField('cloud', 'secure_cloud_only', value)}
          />
        </div>
      </div>
      {spend != null && (
        <p className="text-xs text-content-muted">
          {t('settings.training.spentMonth', { amount: spend.toFixed(2) })}
        </p>
      )}
    </Card>
  )
}

export default function TrainingSection(props) {
  const { t } = useI18n()
  const { config, setField } = props
  return (
    <div className="space-y-6">
      <Card title={t('settings.training.defaultsTitle')} help={t('settings.training.defaultsHelp')}>
        <div>
          <label htmlFor="training-default-family" className="block text-sm font-medium text-content">
            {t('settings.training.defaultFamily')}
          </label>
          <select
            id="training-default-family"
            value={config.training.default_family}
            onChange={(e) => setField('training', 'default_family', e.target.value)}
            className={INPUT_CLASS}
          >
            {FAMILY_OPTIONS.map((f) => (
              <option key={f} value={f}>{t(`settings.training.familyLabels.${f}`)}</option>
            ))}
          </select>
        </div>
      </Card>

      <Card title={t('settings.training.cloudGpuTitle')} help={t('settings.training.cloudGpuHelp')}>
        <SecretField field={vastSecret(t)} {...props} />
      </Card>

      <CloudTrainingCard config={config} setField={setField} />
    </div>
  )
}
