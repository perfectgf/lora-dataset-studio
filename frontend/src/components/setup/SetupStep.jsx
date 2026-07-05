const STATUS_META = {
  ready: { glyph: '✓', label: 'Ready', cls: 'text-emerald-400' },
  partial: { glyph: '◐', label: 'Almost there', cls: 'text-amber-400' },
  available: { glyph: '○', label: 'Not set up', cls: 'text-content-subtle' },
  skipped: { glyph: '–', label: 'Skipped', cls: 'text-content-subtle' },
}

export default function SetupStep({ step, index, effectiveStatus, onSkip, onUnskip, children }) {
  const meta = STATUS_META[effectiveStatus] || STATUS_META.available
  const collapsed = effectiveStatus === 'ready' || effectiveStatus === 'skipped'
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-content">
            {index}. {step.title}
            {step.recommended && (
              <span className="ml-2 rounded bg-primary/15 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                Recommended
              </span>
            )}
          </h2>
          <p className="mt-1 text-xs text-content-muted">Unlocks: {step.unlocks.join(' · ')}</p>
        </div>
        <span className={`inline-flex shrink-0 items-center gap-1 text-xs font-medium ${meta.cls}`}>
          <span aria-hidden="true">{meta.glyph}</span>{meta.label}
        </span>
      </div>
      {!collapsed && <div className="mt-4 space-y-4">{children}</div>}
      {!step.recommended && effectiveStatus !== 'ready' && (
        <div className="mt-3 text-right">
          {effectiveStatus === 'skipped'
            ? <button type="button" onClick={onUnskip} className="text-xs text-primary underline">Set this up</button>
            : <button type="button" onClick={onSkip} className="text-xs text-content-subtle underline">Skip for now</button>}
        </div>
      )}
    </section>
  )
}
