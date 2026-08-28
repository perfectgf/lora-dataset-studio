/* 🔤 The bank launch window's result strip.
 *
 * The bank scan runs as a background job, so the strip POLLS its preview
 * endpoint while a job is live — the flagged pages appear as the scan finds
 * them, which is the whole point of a sample run: judge, adjust the dials,
 * re-run, without leaving the window. One last read when `live` flips back
 * (the effect re-runs on it), so the strip always ends on the final state.
 *
 * The endpoint returns the pages oldest-id first — the SAME deterministic
 * order the sample reads — so "the first pages scanned" and "the first pages
 * shown here" are the same pages.
 */
import { useEffect, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import TextZonesGallery from '../shared/TextZonesGallery.jsx'

const PREVIEW_LIMIT = 12
const POLL_MS = 2500

export default function BankTextScanPreview({ bankId, live = false }) {
  const [data, setData] = useState(null)
  useEffect(() => {
    let on = true
    let timer
    const tick = async () => {
      try {
        const d = await apiFetch(`/api/bank/${bankId}/text/preview?limit=${PREVIEW_LIMIT}`)
        if (on) setData(d)
      } catch { /* keep the last strip rather than flashing it away mid-poll */ }
      if (on && live) timer = setTimeout(tick, POLL_MS)
    }
    tick()
    return () => { on = false; clearTimeout(timer) }
  }, [bankId, live])

  if (!data) return null
  const items = (data.items || []).map((it) => ({
    id: it.id,
    src: `/api/bank/${bankId}/thumb/${it.id}`,
    href: `/api/bank/${bankId}/file/${it.id}`,
    regions: it.regions || [],
  }))
  return (
    <TextZonesGallery items={items} total={data.total || 0} live={live}
      reviewHint="Zones off? Open ▶ Review (flagged) to fix them by hand" />
  )
}
