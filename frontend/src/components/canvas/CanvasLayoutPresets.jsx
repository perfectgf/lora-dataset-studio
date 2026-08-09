import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, del, postJson } from '../../api/fetchClient';
import {
  canvasLayoutIsEmpty, canvasLayoutSnapshot, canvasPresetApplied,
  canvasPresetName, canvasPresetSummary, PRESET_NAME_MAX,
} from '../../utils/canvasLayoutPresets';
import { HelpBadge } from '../../help/HelpMode';

/* 💾 Save this arrangement · put one back.
 *
 * A disclosure rather than a dialog: the board's toolbar already wraps on a
 * phone, and this is a control you open twice a week, not a step in a flow. It
 * loads its list the first time it is opened — a board that fires a request for
 * a panel nobody opened would be a request on every single visit to the page.
 *
 * The list is deliberately NOT a set of one-click "apply" rows. Restoring moves
 * every card and every picture on the board, which is the least undoable thing
 * this panel can do, so applying is the row's own button and it says what it
 * will do; deleting a preset sits at the other end of the row behind its own
 * confirmation, for the same reason the pinned images' 🗑 arms itself.
 */
export default function CanvasLayoutPresets({ positions, imageNodes, datasetIds,
  onRestored, toast }) {
  const [open, setOpen] = useState(false);
  const [presets, setPresets] = useState(null);      // null = never loaded
  const [busy, setBusy] = useState('');
  const [name, setName] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [error, setError] = useState('');
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const load = useCallback(async () => {
    try {
      const d = await apiFetch('/api/train/canvas/layouts');
      if (alive.current) setPresets(d?.presets || []);
    } catch (e) {
      if (alive.current) { setPresets([]); setError(e?.message || 'Could not read your layouts'); }
    }
  }, []);

  useEffect(() => { if (open && presets == null) load(); }, [open, presets, load]);

  const snapshot = () => canvasLayoutSnapshot({ positions, imageNodes, datasetIds });

  const save = useCallback(async () => {
    const clean = canvasPresetName(name);
    if (!clean) { setError('Give this layout a name first'); return; }
    const body = snapshot();
    if (canvasLayoutIsEmpty(body)) {
      setError('Nothing on this board has been arranged yet — move a run or pin a picture first');
      return;
    }
    setBusy('save');
    setError('');
    try {
      await postJson('/api/train/canvas/layouts', { name: clean, ...body });
      if (!alive.current) return;
      setName('');
      toast?.success?.(`Layout “${clean}” saved`);
      await load();
    } catch (e) {
      if (alive.current) setError(e?.message || 'Could not save this layout');
    } finally {
      if (alive.current) setBusy('');
    }
  }, [name, positions, imageNodes, datasetIds, load, toast]);   // eslint-disable-line react-hooks/exhaustive-deps

  const apply = useCallback(async (preset) => {
    setBusy(`apply:${preset.id}`);
    setError('');
    try {
      const res = await postJson(`/api/train/canvas/layouts/${preset.id}/apply`, {});
      if (!alive.current) return;
      // The sentence names what could NOT be put back. A preset kept for three
      // weeks routinely points at a run that has been deleted since, and a
      // silent partial restore is ten minutes spent hunting for a missing card.
      toast?.success?.(canvasPresetApplied(res, preset), 8000);
      await onRestored?.();
    } catch (e) {
      if (alive.current) setError(e?.message || 'Could not restore this layout');
    } finally {
      if (alive.current) setBusy('');
    }
  }, [onRestored, toast]);

  const remove = useCallback(async (preset) => {
    setBusy(`del:${preset.id}`);
    try {
      await del(`/api/train/canvas/layouts/${preset.id}`);
      if (!alive.current) return;
      setConfirmDelete(null);
      await load();
    } catch (e) {
      if (alive.current) setError(e?.message || 'Could not delete this layout');
    } finally {
      if (alive.current) setBusy('');
    }
  }, [load]);

  return (
    <details data-testid="canvas-layout-presets" open={open}
      onToggle={(e) => setOpen(e.currentTarget.open)}
      className="relative">
      <summary className="flex h-10 cursor-pointer list-none items-center gap-1 rounded-md border border-border bg-app/60 px-3 text-content-muted text-[0.6875rem] font-semibold hover:text-content lg:h-9">
        <span aria-hidden>💾</span> Layouts
      </summary>
      {/* `top-full` and not just `mt-1`: an absolutely-positioned box with no
          `top` falls back to its STATIC position, which is wherever the
          toolbar's wrapping happened to leave it — measured on a real board,
          that put the panel's first line on top of the gestures sentence
          behind it. Anchored to the summary, it opens under its own button at
          every width. */}
      {/* `bg-surface-overlay`, not `bg-surface`: the latter is a 4 %-alpha tint
          for lifting a card off the page, and a popover painted with it is a
          sheet of glass — the toolbar behind it stays perfectly legible through
          the menu. Same panel token the app's modals use. */}
      <div className="absolute left-0 top-full z-40 mt-1 w-[min(18rem,calc(100vw-2rem))] rounded-lg border border-border bg-surface-overlay p-2 shadow-xl">
        <div className="mb-1.5 flex items-center gap-1">
          <span className="text-content text-[0.6875rem] font-semibold">Save this arrangement</span>
          <HelpBadge topic="canvas-layouts" />
        </div>
        <div className="flex items-center gap-1">
          <input value={name} onChange={(e) => setName(e.target.value)}
            maxLength={PRESET_NAME_MAX}
            placeholder="e.g. likeness review"
            aria-label="Name for this board layout"
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); save(); } }}
            className="min-w-0 flex-1 rounded border border-border bg-app/60 px-2 py-1 text-content text-[0.75rem] focus:border-primary focus:outline-none" />
          <button type="button" onClick={save} disabled={busy === 'save'}
            className="shrink-0 rounded border border-indigo-400/50 bg-indigo-500/15 px-2 py-1 text-indigo-100 text-[0.6875rem] font-semibold disabled:opacity-50">
            {busy === 'save' ? '…' : 'Save'}
          </button>
        </div>
        <p className="mt-1 mb-2 text-content-subtle text-[0.625rem]">
          Where every run card and every pinned picture sits — closed pictures
          included. Saving under a name you already used replaces it.
        </p>

        {error && (
          <p role="alert" className="mb-2 rounded border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-amber-100 text-[0.625rem]">
            {error}
          </p>
        )}

        {presets == null && <p className="text-content-subtle text-[0.625rem]">Loading…</p>}
        {presets != null && presets.length === 0 && (
          <p className="text-content-subtle text-[0.625rem]">
            No saved layout yet. Arrange the board, name it above, and it will be here.
          </p>
        )}
        {presets != null && presets.length > 0 && (
          <ul className="m-0 flex max-h-64 list-none flex-col gap-1 overflow-y-auto p-0">
            {presets.map((p) => (
              <li key={p.id}
                className="flex items-center gap-1 rounded border border-border bg-app/40 px-1.5 py-1">
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-content text-[0.75rem]" title={p.name}>{p.name}</span>
                  <span className="block text-content-subtle text-[0.5625rem]">{canvasPresetSummary(p)}</span>
                </span>
                <button type="button" onClick={() => apply(p)}
                  disabled={busy === `apply:${p.id}`}
                  title={`Put this arrangement back on the board — moves every card and picture it names`}
                  className="shrink-0 rounded border border-border bg-surface px-1.5 py-0.5 text-content-muted text-[0.625rem] font-semibold hover:text-content disabled:opacity-50">
                  {busy === `apply:${p.id}` ? '…' : 'Apply'}
                </button>
                <button type="button"
                  data-testid={`canvas-layout-delete-${p.id}`}
                  onClick={() => (confirmDelete === p.id ? remove(p) : setConfirmDelete(p.id))}
                  onBlur={() => setConfirmDelete((cur) => (cur === p.id ? null : cur))}
                  disabled={busy === `del:${p.id}`}
                  title={confirmDelete === p.id
                    ? `Press again to forget the layout “${p.name}”`
                    : `Forget the layout “${p.name}” — the board itself is not touched`}
                  aria-label={confirmDelete === p.id
                    ? `Confirm forgetting ${p.name}` : `Forget the layout ${p.name}`}
                  className={'shrink-0 rounded border px-1.5 py-0.5 text-[0.625rem] font-semibold '
                    + (confirmDelete === p.id
                      ? 'border-red-300 bg-red-600/90 text-white'
                      : 'border-border bg-surface text-content-muted hover:border-red-400/60 hover:text-content')}>
                  {confirmDelete === p.id ? '🗑!' : '🗑'}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </details>
  );
}
