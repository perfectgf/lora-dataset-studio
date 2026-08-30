import { targetWarnings, targetBadge } from './videoTargetChoice'

const TONE = {
  danger: 'border-rose-500/60 bg-rose-500/10 text-rose-100',
  warning: 'border-amber-500/60 bg-amber-500/10 text-amber-100',
  info: 'border-sky-500/50 bg-sky-500/10 text-sky-100',
}
const BADGE_TONE = {
  danger: 'bg-rose-500/20 text-rose-200',
  warning: 'bg-amber-500/20 text-amber-200',
  ok: 'bg-emerald-500/20 text-emerald-200',
}

/** 🎬 The target list, with its verdicts ON the rows.
 *
 * A separate component from the dialog around it for one concrete reason: it
 * takes `targets` as a PROP, so a test can mount it with the real catalogue and
 * assert that the licence text and the "no trainer exists" label are actually on
 * screen. Left inside the dialog they would only ever appear after a fetch,
 * which a render test cannot run — and these two labels are precisely the ones
 * that must never silently stop being rendered.
 *
 * The badge sits on the row rather than under the picker because a warning that
 * appears only once you have selected something is a warning you read after
 * deciding.
 */
export default function VideoTargetPicker({ targets, targetKey, onPick }) {
  if (targets == null) {
    return <p className="mt-1 text-sm text-content-muted">Loading targets…</p>
  }
  const selected = targets.find((t) => t.key === targetKey) || null
  return (
    <>
      <div className="mt-1 space-y-1.5">
        {targets.map((t) => {
          const badge = targetBadge(t)
          return (
            <label key={t.key}
              className={`flex min-w-0 cursor-pointer items-start gap-2 rounded-md border p-2 text-sm ${
                t.key === targetKey
                  ? 'border-primary/60 bg-primary/10'
                  : 'border-border bg-surface hover:bg-surface-raised'}`}>
              <input type="radio" name="video-target" value={t.key}
                checked={t.key === targetKey} onChange={() => onPick(t.key)}
                className="mt-0.5 shrink-0 accent-indigo-500" />
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="font-semibold text-content">{t.label}</span>
                  <span className={`rounded px-1.5 py-0.5 text-[0.625rem] font-bold ${BADGE_TONE[badge.tone]}`}>
                    {badge.text}
                  </span>
                </span>
                {t.picker_hint ? (
                  <span className="mt-0.5 block text-xs text-content-muted">
                    {t.picker_hint}
                  </span>
                ) : null}
                <span className="mt-0.5 block text-xs text-content-subtle">
                  {t.fps ? `${t.fps} fps` : 'no fixed frame rate'}
                  {t.keep_audio ? ' · keeps audio' : ' · video only'}
                  {t.size_multiple ? ` · sizes ×${t.size_multiple}` : ''}
                </span>
              </span>
            </label>
          )
        })}
      </div>
      {targetWarnings(selected).map((w) => (
        <p key={w.key} role={w.tone === 'danger' ? 'alert' : undefined}
          className={`mt-2 rounded-md border px-3 py-2 text-xs ${TONE[w.tone]}`}>
          <span aria-hidden>{w.icon}</span> {w.text}
        </p>
      ))}
    </>
  )
}
