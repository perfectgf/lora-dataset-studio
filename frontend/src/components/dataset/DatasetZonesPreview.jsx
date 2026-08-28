/* The dataset launch windows' result strip — the twin of the bank's
 * BankZonesPreview, off the dataset's own /text/preview and
 * /watermark/preview endpoints (`kind` picks which).
 *
 * The dataset scan is a synchronous route, but the pass commits per image, so
 * POLLING while `live` is what lets the flagged pages fill in AS THEY ARE
 * FOUND instead of all at once when the run returns — which is what the
 * window's own empty line promises ("Pages appear here as text is found"),
 * and what the maintainer reported from a phone when the strip sat empty for
 * a whole 106-page scan. One last read when `live` flips back (the effect
 * re-runs on it), so the strip always ends on the final state.
 */
import { useEffect, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import TextZonesGallery from '../shared/TextZonesGallery.jsx'
import { datasetThumbUrl } from '../../utils/datasetThumbUrl.js'

const PREVIEW_LIMIT = 12
const POLL_MS = 2500

export default function DatasetZonesPreview({ datasetId, kind = 'text', live = false, emptyLine = null }) {
  const [data, setData] = useState(null)
  useEffect(() => {
    let on = true
    let timer
    const tick = async () => {
      try {
        const d = await apiFetch(`/api/dataset/${datasetId}/${kind}/preview?limit=${PREVIEW_LIMIT}`)
        if (on) setData(d)
      } catch { /* keep the last strip rather than flashing it away mid-poll */ }
      if (on && live) timer = setTimeout(tick, POLL_MS)
    }
    tick()
    return () => { on = false; clearTimeout(timer) }
  }, [datasetId, kind, live])

  if (!data) return null
  const items = (data.items || []).filter((it) => it.filename).map((it) => {
    const url = `/api/dataset/${datasetId}/img/${encodeURIComponent(it.filename)}`
    return {
      id: it.id,
      src: datasetThumbUrl(url, 384),
      href: url,
      regions: it.regions || [],
    }
  })
  return (
    <TextZonesGallery items={items} total={data.total || 0} live={live}
      emptyLine={emptyLine}
      reviewHint="Zones off? Open 🔍 Review flagged to fix them by hand" />
  )
}
