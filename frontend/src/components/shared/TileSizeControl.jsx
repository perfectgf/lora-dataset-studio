import { LayoutGrid } from 'lucide-react';
/**
 * Discreet segmented S/M/L control, not a slider (mouse-fragile, no useful
 * granularity for 3 steps). Shared by the workspace image grid (DatasetGrid)
 * and the Datasets library (DatasetListPanel) — callers pass their own
 * context-specific `titles` so the tooltips explain what each step is FOR.
 */
export default function TileSizeControl({ size, onChange, titles, className = '' }) {
  return (
    <div role="group" aria-label="Thumbnail size" className={`flex items-center gap-1 shrink-0 ${className}`}>
      <LayoutGrid aria-hidden="true" className="h-3.5 w-3.5 text-content-subtle" />
      {['S', 'M', 'L'].map((s) => (
        <button key={s} type="button" onClick={() => onChange(s)}
          aria-pressed={size === s} title={titles[s]}
          aria-label={`${titles[s]}${size === s ? ' (active)' : ''}`}
          className={`w-6 h-6 rounded-md border text-[0.6875rem] font-semibold transition-colors ${
            size === s
              ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
              : 'border-border bg-surface text-content-muted hover:bg-surface-raised'}`}>
          {s}
        </button>
      ))}
    </div>
  );
}
