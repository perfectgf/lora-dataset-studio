/** Live composition balance vs the recommended training target (≈25 balanced:
 * 12 face / 6 bust / 6 body / 1 back). Shows the DEFICIT so the user knows
 * exactly which image types are still missing after curation/rejections. */
const TARGET = { face: 12, bust: 6, body: 6, back: 1 };
const LABEL = { face: 'Face', bust: 'Bust', body: 'Body', back: 'Back' };

export default function CompositionBar({ composition }) {
  const c = composition || { face: 0, bust: 0, body: 0, back: 0 };
  const total = (c.face || 0) + (c.bust || 0) + (c.body || 0) + (c.back || 0);
  const missing = Object.keys(TARGET)
    .map((k) => ({ k, n: Math.max(0, TARGET[k] - (c[k] || 0)) }))
    .filter((m) => m.n > 0);

  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase tracking-wide">Composition ({total})</span>
        {['face', 'bust', 'body', 'back'].map((k) => {
          const low = (c[k] || 0) < TARGET[k];
          return (
            <span key={k}
              className={`px-2 py-0.5 rounded-full text-[0.6875rem] border ${low ? 'border-amber-400/50 bg-amber-400/10 text-amber-300' : 'border-green-500/40 bg-green-500/10 text-green-300'}`}>
              {LABEL[k]} {c[k] || 0}<span className="opacity-60">/{TARGET[k]}</span>
            </span>
          );
        })}
      </div>
      {missing.length > 0 ? (
        <p className="m-0 text-amber-300/90 text-[0.6875rem]">
          ⚠ Missing: {missing.map((m) => `${m.n} ${LABEL[m.k].toLowerCase()}`).join(' · ')}
          <span className="text-content-subtle"> — generate or import these types (target ≈25 balanced)</span>
        </p>
      ) : (
        <p className="m-0 text-green-300/80 text-[0.6875rem]">✓ Target composition reached — ready to caption/export</p>
      )}
    </div>
  );
}
