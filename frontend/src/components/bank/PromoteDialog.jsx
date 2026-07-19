import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { useI18n } from '../../i18n/I18nContext'

/** ⬆ Promote: copy the selection into a dataset through the normal import
 * path (webp normalization + perceptual dedup vs the dataset). With images
 * selected in the grid, THOSE are promoted; otherwise every KEPT image not
 * yet promoted. The bank keeps its files — promotion copies. */
export default function PromoteDialog({ bankId, keepCount, selectedIds, onClose, onStarted }) {
  const toast = useToast()
  const { t } = useI18n()
  const [datasets, setDatasets] = useState(null)
  const [datasetId, setDatasetId] = useState('')
  const [busy, setBusy] = useState(false)
  const useSelection = selectedIds.length > 0

  useEffect(() => {
    apiFetch('/api/dataset/list')
      .then((d) => setDatasets(d.datasets || []))
      .catch(() => setDatasets([]))
  }, [])

  const promote = async () => {
    if (busy || !datasetId) return
    setBusy(true)
    try {
      await postJson(`/api/bank/${bankId}/promote`, {
        dataset_id: Number(datasetId),
        image_ids: useSelection ? selectedIds : [],
      })
      toast.success(t('bank.promote.started'))
      onStarted?.()
    } catch (e) {
      toast.error(e?.message || t('bank.promote.startFailed'))
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label={t('bank.promote.title')}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-surface p-5 shadow-2xl space-y-4">
        <h2 className="text-base font-bold text-content">⬆ {t('bank.promote.title')}</h2>
        <p className="text-sm text-content-muted">
          {t(useSelection ? 'bank.promote.selectionDescription' : 'bank.promote.keptDescription', {
            count: useSelection ? selectedIds.length : keepCount,
          })}
        </p>
        <div>
          <label htmlFor="promote-dataset" className="block text-sm font-medium text-content">
            {t('bank.promote.target')}
          </label>
          <select id="promote-dataset" value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content">
            <option value="">{datasets == null ? t('common.loading') : t('bank.promote.chooseDataset')}</option>
            {(datasets || []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({t(`datasets.kind.${{
                  character: 'characterShort', concept: 'conceptShort', style: 'styleShort',
                }[d.kind] || 'characterShort'}`)}, {t('bank.counts.images', { count: d.images_total })})
              </option>
            ))}
          </select>
          {datasets != null && datasets.length === 0 && (
            <p className="mt-1 text-xs text-amber-300">
              {t('bank.promote.noDatasets')}
            </p>
          )}
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            {t('common.cancel')}
          </button>
          <button type="button" onClick={promote} disabled={busy || !datasetId}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
            {busy ? t('bank.promote.starting') : t('bank.promote.action')}
          </button>
        </div>
      </div>
    </div>
  )
}
