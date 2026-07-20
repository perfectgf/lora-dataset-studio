import { configRows } from './lineageDetail.js';

/* The Lab detail panel — opens on a node click in the ◉ Graph. It turns the
   lineage from a picture into an analysis surface: every setting the run trained
   with (from the run's persisted snapshot, exposed on the node as `config`),
   and a slot for the notes editor. A legacy run that never recorded its settings
   says so honestly rather than showing an empty table.

   Opaque side drawer (bg-surface-overlay, never the see-through bg-surface) so
   the graph behind it never bleeds through. */
export default function LineageDetailPanel({ node, onClose, onNodeChanged }) {
  if (!node) return null;
  const rows = configRows(node.config);
  return (
    <div className="fixed right-0 top-0 z-50 flex h-full w-80 flex-col overflow-y-auto border-l border-border bg-surface-overlay p-4 shadow-xl">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-content">
          Run #{node.record_id}
          {node.train_type ? ` · ${node.train_type}` : ''}
          {node.steps ? ` · ${node.steps.toLocaleString()} steps` : ''}
        </h3>
        <button type="button" onClick={onClose}
          className="text-content-subtle hover:text-content" aria-label="Close">✕</button>
      </div>

      <section className="mt-3">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Config</div>
        {rows.length === 0 ? (
          <p className="mt-1 text-xs italic text-content-subtle">Config not recorded for this run.</p>
        ) : (
          <table className="mt-1 w-full text-xs">
            <tbody>
              {rows.map((r) => (
                <tr key={r.label}>
                  <td className="py-0.5 pr-2 text-content-subtle">{r.label}</td>
                  <td className="py-0.5 text-content tabular-nums">{r.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* NOTES_SLOT — Task 4 renders the notes editor here */}
    </div>
  );
}
