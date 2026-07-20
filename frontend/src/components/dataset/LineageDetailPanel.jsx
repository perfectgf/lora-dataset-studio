import { useEffect, useState } from 'react';
import { configRows } from './lineageDetail.js';
import { isRunDeletable } from '../../utils/runDeletable.js';
import { putJson, del } from '../../api/fetchClient';
import { useToast } from '../common/Toast';

/* The Lab detail panel — opens on a node click in the ◉ Graph. It turns the
   lineage from a picture into an analysis surface: every setting the run trained
   with (from the run's persisted snapshot, exposed on the node as `config`), plus
   freeform notes on the run and on each of its checkpoints. A legacy run that
   never recorded its settings says so honestly rather than showing an empty table.

   Notes save on blur (only when changed) via the run/checkpoint note endpoints;
   each save reports through onNodeChanged so the graph can light the ● badge live.

   Opaque side drawer (bg-surface-overlay, never the see-through bg-surface) so
   the graph behind it never bleeds through. */
export default function LineageDetailPanel({ node, onClose, onNodeChanged, onNodeDeleted }) {
  const toast = useToast();
  const [runNote, setRunNote] = useState('');
  const [ckNotes, setCkNotes] = useState({});   // step -> text
  const [deleting, setDeleting] = useState(false);

  // Reseed local editors whenever a DIFFERENT run opens. Keyed on record_id so a
  // same-node re-render (e.g. the parent echoing our own onNodeChanged) doesn't
  // stomp what the user is typing.
  useEffect(() => {
    setRunNote(node?.note || '');
    const seed = {};
    for (const c of (node?.checkpoints || [])) seed[c.step] = c.note || '';
    setCkNotes(seed);
  }, [node?.record_id]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (!node) return null;
  const rows = configRows(node.config);
  const checkpoints = node.checkpoints || [];
  // Only a GONE run (no checkpoints on disk) may be removed — the graph guards a
  // recoverable run behind the same rule the backend enforces (deleting one is
  // refused with 409). The action clears metadata only; nothing on disk is touched.
  const deletable = isRunDeletable(node);

  const deleteRun = async () => {
    if (!deletable || deleting) return;
    if (!window.confirm(
      `Remove run #${node.record_id} from the graph?\n\n`
      + 'Its checkpoints are already gone from disk. This clears the leftover '
      + 'run entry and its notes — it does not delete any files.')) return;
    setDeleting(true);
    try {
      await del(`/api/dataset/train/runs/${node.record_id}`);
      toast.success('Run removed');
      onNodeDeleted?.(node.record_id);
    } catch (e) {
      toast.error(e?.message || 'Could not remove this run');
    } finally {
      setDeleting(false);
    }
  };

  const saveRunNote = async () => {
    const text = runNote || '';
    if (text === (node.note || '')) return;               // unchanged — no write
    try {
      await putJson(`/api/dataset/train/runs/${node.record_id}/note`, { note: text });
      toast.success('Note saved');
      onNodeChanged?.({ ...node, note: text, has_note: !!text.trim() });
    } catch (e) {
      toast.error(e?.message || 'Could not save note');
    }
  };

  const saveCkNote = async (step) => {
    const text = ckNotes[step] || '';
    const orig = checkpoints.find((c) => c.step === step)?.note || '';
    if (text === orig) return;                            // unchanged — no write
    try {
      await putJson(`/api/dataset/train/runs/${node.record_id}/checkpoints/${step}/note`, { note: text });
      toast.success('Note saved');
      onNodeChanged?.({
        ...node,
        checkpoints: checkpoints.map((c) => (c.step === step ? { ...c, note: text } : c)),
      });
    } catch (e) {
      toast.error(e?.message || 'Could not save note');
    }
  };

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

      <section className="mt-4">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Run note</div>
        <textarea
          value={runNote}
          onChange={(e) => setRunNote(e.target.value)}
          onBlur={saveRunNote}
          rows={3}
          placeholder="e.g. best overall — v3 dataset, ProdigyPlus"
          className="mt-1 w-full resize-y rounded-md border border-border bg-app/60 p-2 text-xs text-content placeholder:text-content-subtle focus:border-indigo-400/60 focus:outline-none" />
      </section>

      {checkpoints.length > 0 && (
        <section className="mt-4">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-content-subtle">Checkpoint notes</div>
          <ul className="mt-1 flex flex-col gap-1.5">
            {checkpoints.map((c) => (
              <li key={c.step} className="flex items-center gap-2">
                <span className="w-12 shrink-0 text-right text-[0.625rem] tabular-nums text-content-muted">
                  {c.step >= 1000 && c.step % 1000 === 0 ? `${c.step / 1000}k` : c.step}
                </span>
                <input
                  type="text"
                  value={ckNotes[c.step] ?? ''}
                  onChange={(e) => setCkNotes((m) => ({ ...m, [c.step]: e.target.value }))}
                  onBlur={() => saveCkNote(c.step)}
                  placeholder="note…"
                  className="min-w-0 flex-1 rounded-md border border-border bg-app/60 px-2 py-1 text-xs text-content placeholder:text-content-subtle focus:border-indigo-400/60 focus:outline-none" />
              </li>
            ))}
          </ul>
        </section>
      )}

      {deletable && (
        <section className="mt-auto pt-4">
          <button type="button" onClick={deleteRun} disabled={deleting}
            className="w-full rounded-md border border-rose-500/40 bg-rose-600/10 px-2 py-1.5 text-xs font-medium text-rose-200 hover:bg-rose-600/20 disabled:opacity-50">
            {deleting ? 'Removing…' : 'Remove this run'}
          </button>
          <p className="mt-1 text-[0.625rem] leading-snug text-content-subtle">
            No checkpoints left on disk. Removes the leftover run entry and its notes — no files are deleted.
          </p>
        </section>
      )}
    </div>
  );
}
