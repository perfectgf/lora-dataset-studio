import { useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { useToast } from './Toast'
import { formatDiagnostic } from './diagnosticFormat'
import { useI18n } from '../../i18n/I18nContext'

export { formatDiagnostic }

export default function DiagnosticReport() {
  const toast = useToast()
  const { t } = useI18n()
  const [busy, setBusy] = useState(false)
  const copy = async () => {
    setBusy(true)
    try {
      const d = await apiFetch('/api/diagnostic')
      await navigator.clipboard.writeText(formatDiagnostic(d))
      toast.success(t('guide.diagnostic.copied'))
    } catch (err) {
      toast.error(t('guide.diagnostic.failed', { error: err.message }))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-sm font-medium text-content">{t('guide.diagnostic.title')}</p>
      <p className="mt-1 text-xs text-content-muted">
        {t('guide.diagnostic.description')}
      </p>
      <button type="button" onClick={copy} disabled={busy}
        className="mt-3 rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
        {busy ? t('guide.diagnostic.building') : `📋 ${t('guide.diagnostic.copy')}`}
      </button>
    </div>
  )
}
