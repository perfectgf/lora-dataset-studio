import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router'
import { apiFetch } from '../api/fetchClient'
import { videoDatasetUrl } from '../components/videobank/videoBankApi'
import VideoDatasetWorkspace from '../components/videobank/VideoDatasetWorkspace'
import { shouldEjectOnLoadError, staleNote } from './videoDatasetLoad'

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
  // Set when a background refresh failed and the screen is keeping the last
  // good payload; cleared by the next successful load.
  const [stale, setStale] = useState(null)

  // `payload` is read inside `load` to decide whether a failure may take the
  // screen; a ref keeps `load` (and therefore the effect below) from re-running
  // on every payload change, which would refetch in a loop.
  const hasPayload = useRef(false)

  const load = useCallback(async ({ background = false } = {}) => {
    try {
      setPayload(await apiFetch(videoDatasetUrl(id), { background }))
      hasPayload.current = true
      setError(null)
    } catch (e) {
      // A TRANSIENT failure must never take the workspace away. This function
      // is also the refresh after every write, so a one-second network blip, a
      // container restart or a locked SQLite used to replace the whole page
      // with an error paragraph — losing the unsaved caption drafts of every
      // other clip, the selection, the filter, the sort, the open section, and
      // unmounting the training block while its run carried on server-side.
      //
      // The image lane settled this and wrote it down (useDataset.js: "Only an
      // ACTIVE dataset's definitive 404 ejects back to the list. Transient
      // errors and stale responses keep the current workspace"). This page
      // claims to mirror that lane, so it owes the same rule — and the rule is
      // a tested value in videoDatasetLoad.js, not a condition typed here.
      if (shouldEjectOnLoadError(e, hasPayload.current)) {
        setError(e?.message || 'This video dataset could not be loaded.')
        return false
      }
      // Keep what is on screen, but SAY it. apiFetch stays silent in background
      // mode by design, and the offline banner only knows about network
      // failures — a server that answered 500 counts as reachable to it. A
      // caption saved and then not refreshed would otherwise show its previous
      // text with nothing on screen to explain why.
      setStale(staleNote(e))
      return false
    }
    setStale(null)
    return true
  }, [id])

  // The first load is foreground (it may fail, and the failure needs saying);
  // every refresh after a write is background, so a caption save does not flash
  // the global loading chrome over the grid.
  useEffect(() => { hasPayload.current = false; load() }, [load])
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
      {stale && (
        <p role="status"
          className="mb-3 flex flex-wrap items-center gap-2 rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-100">
          <span>{stale}</span>
          <button type="button" onClick={refresh}
            className="min-h-10 rounded border border-amber-400/60 px-2 py-0.5 font-semibold hover:bg-amber-500/20 lg:min-h-0">
            Retry
          </button>
        </p>
      )}
      <VideoDatasetWorkspace ds={payload} items={payload.items || []}
        refresh={refresh} onBack={back} />
    </div>
  )
}
