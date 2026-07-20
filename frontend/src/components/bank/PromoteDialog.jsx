import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'

/** ⬆ Promote: copy the selection into a dataset through the normal import
 * path (webp normalization + perceptual dedup vs the dataset). With images
 * selected in the grid, THOSE are promoted; otherwise every KEPT image not
 * yet promoted. The bank keeps its files — promotion copies. */
export default function PromoteDialog({ bankId, selectedIds, onClose, onStarted }) {
  const toast = useToast()
  const [datasets, setDatasets] = useState(null)
  const [datasetId, setDatasetId] = useState('')
  const [promotable, setPromotable] = useState(null)
  const [busy, setBusy] = useState(false)
  const useSelection = selectedIds.length > 0

  useEffect(() => {
    apiFetch('/api/dataset/list')
      .then((d) => setDatasets(d.datasets || []))
      .catch(() => setDatasets([]))
  }, [])

  // The kept-but-not-yet-on-THIS-dataset count is per-target (an image promoted
  // to another dataset still counts), so it can only be known once a target is
  // chosen. Fetch it then, so the copy line reflects what the server will do.
  useEffect(() => {
    if (useSelection || !datasetId) { setPromotable(null); return }
    let live = true
    setPromotable(null)
    apiFetch(`/api/bank/${bankId}/promotable?dataset_id=${Number(datasetId)}`)
      .then((d) => { if (live) setPromotable(d.count) })
      .catch(() => { if (live) setPromotable(null) })
    return () => { live = false }
  }, [bankId, datasetId, useSelection])

  const promote = async () => {
    if (busy || !datasetId) return
    setBusy(true)
    try {
      await postJson(`/api/bank/${bankId}/promote`, {
        dataset_id: Number(datasetId),
        image_ids: useSelection ? selectedIds : [],
      })
      toast.success('Promotion started — follow the progress bar.')
      onStarted?.()
    } catch (e) {
      toast.error(e?.message || 'Promotion failed to start.')
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Promote to dataset"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="w-full max-w-md rounded-xl border border-border bg-surface-overlay p-5 shadow-2xl space-y-4">
        <h2 className="text-base font-bold text-content">⬆ Promote to dataset</h2>
        <p className="text-sm text-content-muted">
          {useSelection
            ? `The ${selectedIds.length} selected image(s) will be COPIED into the dataset`
            : !datasetId
              ? `Kept image(s) not yet in the chosen dataset will be COPIED into it`
              : promotable == null
                ? `The kept image(s) not yet in this dataset will be COPIED into it`
                : `The ${promotable} kept image(s) not yet in this dataset will be COPIED into it`}
          {' '}— normalized to webp, near-duplicates already in the dataset skipped. The bank and
          its source folder are left as they are.
        </p>
        <div>
          <label htmlFor="promote-dataset" className="block text-sm font-medium text-content">
            Target dataset
          </label>
          <select id="promote-dataset" value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content">
            <option value="">{datasets == null ? 'Loading…' : 'Choose a dataset…'}</option>
            {(datasets || []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.kind}, {d.images_total} image{d.images_total === 1 ? '' : 's'})
              </option>
            ))}
          </select>
          {datasets != null && datasets.length === 0 && (
            <p className="mt-1 text-xs text-amber-300">
              No dataset yet — create one on the Datasets page first, then promote.
            </p>
          )}
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="button" onClick={promote} disabled={busy || !datasetId}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
            {busy ? 'Starting…' : 'Promote'}
          </button>
        </div>
      </div>
    </div>
  )
}
