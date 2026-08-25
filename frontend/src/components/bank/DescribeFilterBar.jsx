import { useState } from 'react'
import { MessagesSquare } from 'lucide-react';
import { postJson } from '../../api/fetchClient'
import { toFilterPatch, describeSummary, headline } from './bankDescribe.js'

/* Say what you want; the app sets ITS OWN filters and you read the result.
 *
 * The model never picks images and nothing is applied without this component
 * handing the patch to the same `setF` a chip click uses. That is the whole
 * safety of the feature: a bad reading costs one glance at controls you can edit,
 * not a silent selection you have to trust.
 *
 * The counts stay where they already are — on the chips below, measured by the
 * server. This surface deliberately prints no number of its own. */
export default function DescribeFilterBar({ bankId, onApply, onFenceBlocked }) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)

  const run = async () => {
    const request = text.trim()
    if (!request || busy) return
    setBusy(true); setError(null)
    try {
      const out = await postJson(`/api/bank/${bankId}/describe-filter`, { request })
      setRes(out)
      // Applied even when the reading is partial: the chips are the honest place
      // to see how far it got, and refusing to move them would hide a correct
      // half-reading behind a blank grid.
      if (out && !out.refused) onApply?.(toFilterPatch(out))
    } catch (e) {
      // The one refusal that carries its own remedy travels by CODE, not by
      // string-matching a sentence: a model held outside the app can be unloaded.
      if (e?.code === 'ollama_fence_blocked' && onFenceBlocked) onFenceBlocked(e)
      setError(e?.message || String(e))
    } finally { setBusy(false) }
  }

  const s = describeSummary(res)
  return (
    <div className="rounded-lg border border-border bg-surface-raised p-2">
      {/* flex-wrap and a full-width field: at 400 px the label, the input and the
          button cannot share a row.
          ⚠️ The field carries a MINIMUM width, not `min-w-0`. With min-w-0 a flex
          child accepts any width, so `flex-wrap` never fires: instead of moving
          to its own line the input is squeezed to a few pixels beside the label
          — which is exactly what it did in the rail, where the whole row has
          17rem to share. Wrapping is the behaviour this row was written for; the
          floor is what makes it happen. */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 text-xs text-content-muted"><MessagesSquare aria-hidden="true" className="h-3.5 w-3.5" /> Describe the set you want</span>
        <input
          className="w-full min-w-[11rem] flex-1 rounded-md border border-border bg-surface px-2 py-1 text-sm"
          placeholder="an amateur photo set, least polished first"
          value={text} maxLength={400} disabled={busy}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') run() }} />
        <button type="button" onClick={run} disabled={busy || !text.trim()}
          className="min-h-10 lg:min-h-0 rounded-md border border-border px-2 py-1 text-xs font-medium
            text-content-muted hover:bg-surface-raised hover:text-content
            disabled:opacity-50">
          {busy ? 'Reading…' : 'Set the filters'}
        </button>
      </div>

      {error && <p className="mt-1 text-xs text-rose-300">{error}</p>}

      {res && (
        <div className="mt-1 space-y-0.5 text-xs">
          <p className={s.refused ? 'text-amber-300' : 'text-content-muted'}>{headline(res)}</p>
          {/* Three registers, never merged: what it did, what it could not, and
              what it reached for that this bank does not hold. A single blended
              paragraph is how a half-read request starts reading as a success. */}
          {!!s.understood.length && (
            <p className="text-content-muted">Read as: {s.understood.join(' · ')}</p>
          )}
          {!!s.unsupported.length && (
            <p className="text-amber-300">Not expressible here: {s.unsupported.join(' · ')}</p>
          )}
          {!!s.dropped.length && (
            <p className="text-content-muted">Ignored: {s.dropped.join(' · ')}</p>
          )}
        </div>
      )}
    </div>
  )
}
