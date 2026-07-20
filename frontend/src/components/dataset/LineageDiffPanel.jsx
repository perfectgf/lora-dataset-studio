import { useMemo, useState } from 'react';
import { diffConfigs } from './lineageDetail.js';

/* The Lab compare panel — opens when exactly TWO run cards are checked for
   compare in the ◉ Graph. It turns the same persisted config the inspector
   shows into a two-column diff: settings that DIFFER are highlighted (amber),
   settings that MATCH are dimmed and can be folded away, so "what changed
   between v2 and v3" reads at a glance instead of eyeballing two panels.

   Read-only and fully derived (diffConfigs) — no notes editing here, no backend.
   Two legacy runs that never recorded their settings say so honestly rather
   than showing an empty table.

   Opaque side drawer (bg-surface-overlay) so the graph behind never bleeds. */
export default function LineageDiffPanel({ a, b, onClose }) {
  const [showUnchanged, setShowUnchanged] = useState(false);
  const rows = useMemo(() => diffConfigs(a?.config, b?.config), [a?.config, b?.config]);
  if (!a || !b) return null;

  const changedCount = rows.filter((r) => r.changed).length;
  const unchangedCount = rows.length - changedCount;
  const visible = showUnchanged ? rows : rows.filter((r) => r.changed);

  const cell = (v, changed) =>
    v === null
      ? <span className="italic text-content-subtle">—</span>
      : <span className={changed ? 'font-semibold text-amber-100' : 'text-content'}>{v}</span>;

  return (
    <div className="fixed right-0 top-0 z-50 flex h-full w-96 flex-col overflow-y-auto border-l border-border bg-surface-overlay p-4 shadow-xl">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-content">Compare runs</h3>
        <button type="button" onClick={onClose}
          className="text-content-subtle hover:text-content" aria-label="Close">✕</button>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md border border-border bg-app/50 px-2 py-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Run A</div>
          <div className="font-mono text-content">#{a.record_id}</div>
        </div>
        <div className="rounded-md border border-indigo-400/40 bg-indigo-500/10 px-2 py-1.5">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Run B</div>
          <div className="font-mono text-content">#{b.record_id}</div>
        </div>
      </div>

      <section className="mt-3">
        {rows.length === 0 ? (
          <p className="mt-1 text-xs italic text-content-subtle">
            Neither run recorded its settings, so there's nothing to compare.
          </p>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">
                {changedCount === 0 ? 'No differences' : `${changedCount} change${changedCount > 1 ? 's' : ''}`}
              </div>
              {unchangedCount > 0 && (
                <button type="button"
                  onClick={() => setShowUnchanged((v) => !v)}
                  className="text-[10px] text-content-subtle hover:text-content underline decoration-dotted">
                  {showUnchanged ? 'Hide' : 'Show'} {unchangedCount} unchanged
                </button>
              )}
            </div>
            <table className="mt-1.5 w-full border-collapse text-xs">
              <tbody>
                {visible.map((r) => (
                  <tr key={r.key}
                    className={r.changed ? 'bg-amber-500/10' : 'opacity-60'}>
                    <td className="w-24 py-1 pl-1 pr-2 align-top text-content-subtle">{r.label}</td>
                    <td className="py-1 pr-2 align-top tabular-nums">{cell(r.a, r.changed)}</td>
                    <td className="py-1 pr-1 align-top tabular-nums">{cell(r.b, r.changed)}</td>
                  </tr>
                ))}
                {visible.length === 0 && (
                  <tr>
                    <td colSpan={3} className="py-2 text-center text-content-subtle">
                      These two runs trained with identical settings.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  );
}
