import { useEffect, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import { videoDatasetSamplesUrl } from './videoBankApi'
import { samplesOfStep } from './videoLineage'
import { stepLabel } from './videoCloudStatus'

/** 🎬 The samples ai-toolkit rendered at ONE step, played one at a time.
 *
 * The page's rule stands here too: ONE `<video>` mounted, ever (the grid holds
 * none — Chrome caps media players, silently). The other prompts of the step
 * are a strip of stills; picking one swaps the source. Escape closes. */
export default function VideoSampleLightbox({ datasetId, target, onClose }) {
  const [samples, setSamples] = useState(null)
  const [err, setErr] = useState(null)
  const [idx, setIdx] = useState(0)
  const node = target?.node
  const pill = target?.pill
  const runId = node?.source === 'local' ? null : node?.run_id

  useEffect(() => {
    if (!target) return undefined
    let alive = true
    setSamples(null); setErr(null); setIdx(0)
    apiFetch(videoDatasetSamplesUrl(datasetId, runId), { background: true })
      .then((d) => { if (alive) setSamples(samplesOfStep(d?.samples, pill?.step)) })
      .catch((e) => { if (alive) setErr(e?.message || 'Could not list the samples.') })
    return () => { alive = false }
  }, [datasetId, runId, pill?.step, target])

  useEffect(() => {
    if (!target) return undefined
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.()
      else if (e.key === 'ArrowRight') setIdx((i) => (samples?.length ? (i + 1) % samples.length : i))
      else if (e.key === 'ArrowLeft') setIdx((i) => (samples?.length ? (i - 1 + samples.length) % samples.length : i))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [target, samples, onClose])

  if (!target) return null
  const current = samples?.[idx] || null
  const label = stepLabel({ step: pill?.step, final: !!pill?.final, files: pill?.files })
  return (
    <div className="fixed inset-0 z-[9000] flex items-center justify-center bg-black/80 p-3"
      data-probe-chrome="sample-lightbox" data-probe-layer role="dialog"
      aria-label={`Samples of ${label}`} onClick={onClose}>
      <div className="flex max-h-full w-full max-w-3xl flex-col gap-2 rounded-xl border border-border bg-surface-overlay p-3"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-semibold text-content">{label}</span>
          {samples?.length > 1 && (
            <span className="text-content-muted">sample {idx + 1} / {samples.length} — prompt {current?.prompt_idx ?? idx}</span>
          )}
          <button type="button" onClick={onClose} aria-label="Close"
            className="ml-auto min-h-10 rounded border border-border px-2 text-content-subtle hover:text-content lg:min-h-0">✕</button>
        </div>
        {err && <p className="m-0 text-xs text-amber-200">{err}</p>}
        {!err && samples === null && <p className="m-0 text-xs text-content-subtle">Reading the samples…</p>}
        {samples && !samples.length && (
          <p className="m-0 text-xs text-content-muted">No sample was rendered at this step.</p>
        )}
        {current && (current.kind === 'video' ? (
          <video key={current.url} src={current.url} poster={current.poster_url} controls autoPlay loop
            className="max-h-[70vh] w-full rounded-lg bg-black" />
        ) : (
          // An ANIMATION (Wan 2.2 writes animated WebP) plays inside an <img> —
          // the browser loops it; there are no controls to give. An image
          // sample (a stills set) is just shown.
          <img key={current.url} src={current.url} alt={`Sample at ${label}`}
            className="max-h-[70vh] w-full rounded-lg object-contain" />
        ))}
        {current?.kind === 'animation' && (
          <p className="m-0 text-[0.625rem] text-content-subtle">Animated WebP — loops on its own, no scrub bar.</p>
        )}
        {samples?.length > 1 && (
          <div className="flex flex-wrap gap-1.5">
            {samples.map((s, i) => (
              <button key={s.filename} type="button" onClick={() => setIdx(i)}
                aria-pressed={i === idx} title={s.filename}
                className={'h-14 w-20 overflow-hidden rounded border '
                  + (i === idx ? 'border-indigo-400' : 'border-border hover:border-content-subtle')}>
                <img src={s.poster_url} alt="" className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
