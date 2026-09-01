import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router'
import { apiFetch } from '../api/fetchClient'
import { videoDatasetUrl } from '../components/videobank/videoBankApi'
import VideoDatasetWorkspace from '../components/videobank/VideoDatasetWorkspace'

/** 🎬 One video training set, on its own page.
 *
 * ADDRESSABLE ON PURPOSE. The library card used to expand an accordion, which
 * meant the set had no address: you could not link to it, a reload lost it, and
 * the back button went back to whatever was before the library. The image lane
 * has never had that problem, and the fix is the same one it uses — a route.
 *
 * The payload carries the dataset AND its clips in one call
 * (`video_dataset_payload`), so there is one fetch here and every refresh after
 * a write goes through the same one. A dataset is tens to hundreds of rows, not
 * a bank's thousands, which is why this page pages nothing.
 */
export default function VideoDatasetPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async ({ background = false } = {}) => {
    try {
      setPayload(await apiFetch(videoDatasetUrl(id), { background }))
      setError(null)
    } catch (e) {
      setError(e?.message || 'This video dataset could not be loaded.')
    }
  }, [id])

  // The first load is foreground (it may fail, and the failure needs saying);
  // every refresh after a write is background, so a caption save does not flash
  // the global loading chrome over the grid.
  useEffect(() => { load() }, [load])
  const refresh = useCallback(() => load({ background: true }), [load])

  const back = () => navigate('/datasets')

  if (error) {
    return (
      <div className="flex flex-col items-start gap-3 p-4">
        <p className="text-sm text-content-muted">{error}</p>
        <button type="button" onClick={back}
          className="min-h-10 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised hover:text-content lg:min-h-0">
          ← Back to Datasets
        </button>
      </div>
    )
  }
  if (!payload) return <p className="p-4 text-sm text-content-muted">Loading…</p>

  return (
    <div className="mx-auto max-w-6xl p-4">
      <VideoDatasetWorkspace ds={payload} items={payload.items || []}
        refresh={refresh} onBack={back} />
    </div>
  )
}
