import { Link } from 'react-router-dom';

/* Thin persistent progress bar. Every step is a real button (scrolls to its
   section) — flexible: it never locks anything. Unavailable steps link to
   Settings with their hint. Status is glyph + text, never color-only. */
export default function GuidedStepper({ steps, currentId, onJump }) {
  return (
    <nav aria-label="Dataset progress"
      className="rounded-lg border border-border bg-surface px-2 py-1.5">
      <ol className="flex flex-wrap items-center gap-x-1 gap-y-1 text-[0.75rem]">
        {steps.map((s, i) => {
          const glyph = s.done ? '✓' : s.busy ? '⏳' : s.id === currentId ? '◉' : '○';
          const tone = s.unavailable ? 'text-content-subtle opacity-60'
            : s.done ? 'text-emerald-400'
            : s.id === currentId ? 'text-content font-semibold'
            : 'text-content-muted';
          return (
            <li key={s.id} className="flex items-center gap-1">
              {i > 0 && <span aria-hidden className="text-content-subtle px-0.5">—</span>}
              {s.unavailable ? (
                <Link to="/settings" title={s.hint}
                  className={`px-1.5 py-0.5 rounded hover:bg-surface-raised ${tone}`}>
                  <span aria-hidden>{glyph}</span> {s.label}
                  <span className="sr-only"> — {s.hint}</span>
                  <span aria-hidden> ⚙</span>
                </Link>
              ) : (
                <button type="button" onClick={() => onJump(s)}
                  aria-current={s.id === currentId ? 'step' : undefined}
                  title={s.subtitle || s.label}
                  className={`px-1.5 py-0.5 rounded hover:bg-surface-raised ${tone}`}>
                  <span aria-hidden>{glyph}</span> {s.label}
                  {s.optional && <span className="text-content-subtle"> (opt)</span>}
                  {s.subtitle && <span className="text-content-subtle"> · {s.subtitle}</span>}
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
