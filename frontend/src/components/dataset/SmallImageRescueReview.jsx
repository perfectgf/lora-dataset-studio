import { useMemo, useState } from 'react';
import { buildSmallImageRescuePairs } from '../../utils/smallImageRescue';

function imageUrl(datasetId, image, nonce = 0) {
  if (!image?.filename) return null;
  const suffix = nonce ? `?v=${nonce}` : '';
  return `/api/dataset/${datasetId}/img/${encodeURIComponent(image.filename)}${suffix}`;
}

function ImagePane({ datasetId, image, nonce, label, tone, fallback, onPreview }) {
  const url = imageUrl(datasetId, image, nonce);
  return (
    <div className="min-w-0 overflow-hidden rounded-lg border border-border bg-app/50">
      <div className="flex min-h-8 items-center justify-between gap-1 border-b border-border px-2 py-1">
        <span className={`truncate text-[0.6875rem] font-semibold ${tone}`}>{label}</span>
        <span className="shrink-0 text-[0.625rem] text-content-subtle">
          {image?.status || 'pending'}
        </span>
      </div>
      <div className="relative aspect-square bg-black">
        {url ? (
          <button type="button" onClick={() => onPreview?.(image)}
            aria-label={`Open ${label} in full-screen preview`}
            title={`Open ${label} full size`}
            className="group relative block h-full w-full cursor-zoom-in focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-300">
            <img src={url} alt={label} loading="lazy"
              className="h-full w-full select-none object-contain" />
            <span aria-hidden
              className="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[0.6875rem] text-white opacity-80 group-hover:opacity-100">
              ⛶
            </span>
          </button>
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 p-3 text-center">
            {image?.status === 'failed' ? (
              <span aria-hidden className="text-2xl">⚠️</span>
            ) : (
              <span aria-hidden className="h-6 w-6 animate-spin rounded-full border-2 border-indigo-300/30 border-t-indigo-300" />
            )}
            <span className={`max-w-full break-words text-[0.6875rem] leading-relaxed ${
              image?.status === 'failed' ? 'text-rose-300' : 'text-content-subtle'}`}>
              {fallback}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function SmallImageRescueReview({
  images,
  datasetId,
  onResolve,
  onPreview,
  nonces = {},
}) {
  const pairs = useMemo(
    () => buildSmallImageRescuePairs(images).filter((pair) => !pair.resolved),
    [images],
  );
  const [resolvingIds, setResolvingIds] = useState(() => new Set());

  if (!pairs.length) return null;

  const resolve = async (candidateId, choice) => {
    if (resolvingIds.has(candidateId)) return;
    setResolvingIds((current) => new Set(current).add(candidateId));
    try {
      await onResolve(candidateId, choice);
    } finally {
      setResolvingIds((current) => {
        const next = new Set(current);
        next.delete(candidateId);
        return next;
      });
    }
  };

  return (
    <section id="ds-curation-small-image-rescue"
      className="flex min-w-0 scroll-mt-20 flex-col gap-3 rounded-xl border border-indigo-400/40 bg-indigo-500/[0.06] p-3 lg:scroll-mt-24"
      aria-labelledby="small-image-rescue-title">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 id="small-image-rescue-title" className="m-0 text-sm font-semibold text-content">
            ✨ Small-image rescue review
          </h3>
          <p className="m-0 mt-0.5 max-w-3xl text-[0.6875rem] leading-relaxed text-content-subtle">
            Klein is generative: compare identity, textures and small details. Both versions stay out
            of training until you make one atomic choice; the original is never overwritten.
          </p>
        </div>
        <span className="shrink-0 rounded-full border border-indigo-400/40 bg-indigo-500/10 px-2 py-0.5 text-[0.6875rem] font-semibold text-indigo-200">
          {pairs.length} to review
        </span>
      </div>

      <div className="flex min-w-0 flex-col gap-3">
        {pairs.map((pair, index) => {
          const { original, candidate, phase } = pair;
          const resolving = resolvingIds.has(candidate.id);
          const controlsDisabled = resolving;
          const detail = phase === 'queued'
            ? 'Klein is generating the candidate…'
            : phase === 'failed'
              ? (candidate.fail_reason || 'Klein could not generate this candidate.')
              : 'Candidate ready — inspect it at full size before choosing.';
          return (
            <article key={candidate.id} aria-busy={resolving}
              className="min-w-0 rounded-lg border border-border bg-surface p-2.5">
              <div className="mb-2 flex min-w-0 flex-wrap items-center gap-2">
                <span className="text-xs font-semibold text-content">Pair {index + 1}</span>
                <span className={`min-w-0 break-words text-[0.6875rem] ${
                  phase === 'failed' ? 'text-rose-300' : phase === 'queued' ? 'text-indigo-200' : 'text-emerald-300'}`}>
                  {phase === 'failed' ? '⚠ ' : phase === 'queued' ? '⏳ ' : '✓ '}{detail}
                </span>
                {resolving && (
                  <span role="status" className="ml-auto text-[0.6875rem] text-content-subtle">
                    Saving choice…
                  </span>
                )}
              </div>

              <div className="grid min-w-0 grid-cols-2 gap-2">
                <ImagePane datasetId={datasetId} image={original} nonce={nonces[original.id] || 0}
                  label="Original small image" tone="text-content" fallback="Original unavailable"
                  onPreview={onPreview} />
                <ImagePane datasetId={datasetId} image={candidate} nonce={nonces[candidate.id] || 0}
                  label="Klein candidate" tone="text-indigo-200" fallback={detail}
                  onPreview={onPreview} />
              </div>

              <div role="group" aria-label={`Choose the result for rescue pair ${index + 1}`}
                className="mt-2 grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-3">
                <button type="button" disabled={controlsDisabled}
                  onClick={() => resolve(candidate.id, 'original')}
                  className="min-h-9 min-w-0 rounded-lg border border-border bg-surface-raised px-2 py-1.5 text-xs font-semibold text-content hover:bg-white/10 disabled:opacity-40">
                  Keep original
                </button>
                <button type="button" disabled={controlsDisabled || phase !== 'ready'}
                  title={phase === 'ready' ? 'Keep the Klein result and reject the original' : 'Wait for a completed Klein candidate'}
                  onClick={() => resolve(candidate.id, 'klein')}
                  className="min-h-9 min-w-0 rounded-lg border border-indigo-400/50 bg-indigo-500/15 px-2 py-1.5 text-xs font-semibold text-indigo-100 hover:bg-indigo-500/25 disabled:opacity-40">
                  Use Klein
                </button>
                <button type="button" disabled={controlsDisabled}
                  onClick={() => resolve(candidate.id, 'reject')}
                  className="min-h-9 min-w-0 rounded-lg border border-rose-500/40 bg-rose-500/10 px-2 py-1.5 text-xs font-semibold text-rose-300 hover:bg-rose-500/20 disabled:opacity-40">
                  Reject both
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
