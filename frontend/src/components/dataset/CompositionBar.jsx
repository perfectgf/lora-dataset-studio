import { useI18n } from '../../i18n/I18nContext';

/** Live composition balance vs the recommended training target. Face-only
 * (default): ≈25 balanced — 12 face / 6 bust / 6 body / 1 back. Body-fidelity:
 * the body must be learned too, so the target shifts to 8/8/8/2 (≈26). Shows
 * the DEFICIT so the user knows exactly which image types are still missing. */
const TARGET_FACE = { face: 12, bust: 6, body: 6, back: 1 };
const TARGET_BODY = { face: 8, bust: 8, body: 8, back: 2 };

export default function CompositionBar({ composition, upscaled, bodyFidelity = false }) {
  const { t } = useI18n();
  const label = (key) => t(`workspace.composition.framing.${key}`);
  const TARGET = bodyFidelity ? TARGET_BODY : TARGET_FACE;
  const c = composition || { face: 0, bust: 0, body: 0, back: 0 };
  const u = upscaled || { face: 0, bust: 0, body: 0, back: 0 };
  const total = (c.face || 0) + (c.bust || 0) + (c.body || 0) + (c.back || 0);
  const missing = Object.keys(TARGET)
    .map((k) => ({ k, n: Math.max(0, TARGET[k] - (c[k] || 0)) }))
    .filter((m) => m.n > 0);
  // Buckets whose target is MET but mostly by heavily-upscaled crops rather than
  // native shots: counting toward the ratio hides that the shot is fabricated
  // texture (LANCZOS-enlarged from a small detected/fallback box), which biases
  // training toward that crop's local detail instead of the intended framing mix.
  const upscaleHeavy = Object.keys(TARGET)
    .map((k) => ({ k, n: u[k] || 0, of: c[k] || 0 }))
    .filter((m) => m.n > 0 && m.n >= Math.ceil(m.of / 2));

  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border bg-surface px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-content-muted text-[0.6875rem] uppercase tracking-wide">
          {t('workspace.composition.title')} ({total})
          {bodyFidelity && <span className="text-emerald-400 normal-case"> · {t('workspace.composition.bodyFidelity')}</span>}
        </span>
        {['face', 'bust', 'body', 'back'].map((k) => {
          const low = (c[k] || 0) < TARGET[k];
          return (
            <span key={k}
              className={`px-2 py-0.5 rounded-full text-[0.6875rem] border ${low ? 'border-amber-400/50 bg-amber-400/10 text-amber-300' : 'border-green-500/40 bg-green-500/10 text-green-300'}`}>
              {label(k)} {c[k] || 0}<span className="opacity-60">/{TARGET[k]}</span>
            </span>
          );
        })}
      </div>
      {missing.length > 0 ? (
        <p className="m-0 text-amber-300/90 text-[0.6875rem]">
          ⚠ {t('workspace.composition.missing')}:{' '}
          {missing.map((m) => `${m.n} ${label(m.k)}`).join(' · ')}
          <span className="text-content-subtle"> — {t('workspace.composition.missingHint')}</span>
        </p>
      ) : (
        <p className="m-0 text-green-300/80 text-[0.6875rem]">
          ✓ {t('workspace.composition.reached')}
        </p>
      )}
      {upscaleHeavy.length > 0 && (
        <p className="m-0 text-amber-300/90 text-[0.6875rem]">
          ⚠ {t('workspace.composition.upscaled')}:{' '}
          {upscaleHeavy.map((m) => `${m.n}/${m.of} ${label(m.k)}`).join(' · ')}
          <span className="text-content-subtle"> — {t('workspace.composition.upscaledHint')}</span>
        </p>
      )}
    </div>
  );
}
