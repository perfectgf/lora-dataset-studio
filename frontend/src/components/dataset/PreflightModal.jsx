/** Interactive pre-training preflight. Keeps the aggregate warning message but
 * drills into WHICH captions leak identity (editable in place, saves on blur)
 * and WHICH kept images are near-duplicates (reject one per pair) — so the
 * offenders get fixed right at the confirm, not hunted down in the grid after.
 * Replaces the old blocking window.confirm: onResolve(true) = start anyway,
 * onResolve(false) = cancel. */
import { useEffect, useState } from 'react';

export default function PreflightModal({ report, datasetId, ds, onResolve }) {
  const { warnings = [], leak_images: leaks = [], dup_pairs: dups = [] } = report || {};
  const [rejected, setRejected] = useState({});   // imageId -> true (rejected in place)
  const imgUrl = (fn) => `/api/dataset/${datasetId}/img/${encodeURIComponent(fn)}`;

  // Escape cancels, like dismissing a native confirm.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onResolve(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onResolve]);

  const reject = async (id) => {
    setRejected((m) => ({ ...m, [id]: true }));   // optimistic: mark before the round-trip
    await ds.setStatus(id, 'reject');
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Before training"
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) onResolve(false); }}>
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border border-amber-400/40 bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-amber-300 font-semibold"><span aria-hidden>⚠️</span> Before training</span>
          <button type="button" onClick={() => onResolve(false)}
            className="ml-auto text-content-subtle hover:text-content" aria-label="Cancel">✕</button>
        </div>

        {/* Summary — the aggregate message, kept verbatim. */}
        {warnings.length > 0 && (
          <ul className="m-0 pl-4 flex flex-col gap-1 text-content-muted text-[0.8125rem] list-disc">
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        )}

        {/* WHICH captions leak — edit in place (saves when you click away). */}
        {leaks.length > 0 && (
          <div className="rounded-lg border border-amber-400/30 bg-amber-500/5 p-2.5 flex flex-col gap-2">
            <span className="text-amber-300 text-[0.8125rem] font-semibold">
              Captions describing the identity ({leaks.length}) — remove the face / hair words
            </span>
            {leaks.map((li) => (
              <div key={li.id} className="flex gap-2 items-start">
                <img src={imgUrl(li.filename)} alt={`image ${li.id}`} loading="lazy"
                  className="w-14 h-14 rounded object-cover shrink-0 bg-black" />
                <textarea defaultValue={li.caption} rows={2}
                  aria-label={`Caption of image ${li.id}`}
                  onBlur={(e) => { if (e.target.value !== li.caption) ds.setCaption(li.id, e.target.value); }}
                  className="flex-1 bg-app/60 border border-amber-400/30 rounded px-2 py-1 text-[0.6875rem] text-content resize-y" />
              </div>
            ))}
          </div>
        )}

        {/* WHICH pairs are near-duplicate — reject one of each. */}
        {dups.length > 0 && (
          <div className="rounded-lg border border-amber-400/30 bg-amber-500/5 p-2.5 flex flex-col gap-2">
            <span className="text-amber-300 text-[0.8125rem] font-semibold">
              Near-duplicate pairs ({dups.length}) — reject one of each
            </span>
            {dups.map((p, i) => {
              const resolved = rejected[p.a.id] || rejected[p.b.id];
              return (
                <div key={i} className={`flex items-center gap-3 ${resolved ? 'opacity-60' : ''}`}>
                  {[p.a, p.b].map((im) => (
                    <div key={im.id} className="flex flex-col items-center gap-1">
                      <img src={imgUrl(im.filename)} alt={`image ${im.id}`} loading="lazy"
                        className={`w-20 h-20 rounded object-cover bg-black ${rejected[im.id] ? 'ring-2 ring-red-500 grayscale' : ''}`} />
                      <button type="button" disabled={resolved} onClick={() => reject(im.id)}
                        className="px-2 py-0.5 rounded bg-red-500/15 border border-red-500/40 text-red-300 text-[0.625rem] disabled:opacity-40">
                        {rejected[im.id] ? '✕ rejected' : 'Reject this'}
                      </button>
                    </div>
                  ))}
                  {resolved && <span className="text-emerald-400 text-[0.6875rem]">✓ resolved</span>}
                </div>
              );
            })}
          </div>
        )}

        <div className="flex items-center gap-2 pt-1">
          <button type="button" onClick={() => onResolve(false)}
            className="px-3 py-1.5 rounded-lg bg-surface text-content text-sm">Cancel</button>
          <button type="button" onClick={() => onResolve(true)}
            className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold">
            Start anyway
          </button>
        </div>
      </div>
    </div>
  );
}
