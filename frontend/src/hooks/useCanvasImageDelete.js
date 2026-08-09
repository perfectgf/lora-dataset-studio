import { useCallback, useEffect, useRef, useState } from 'react';
import { postJson } from '../api/fetchClient';
import { canvasImageDeleteTarget, canvasDeleteError, DELETE_ARM_MS } from '../utils/canvasImageDelete';

/* 🗑 The arm-then-confirm delete of ONE pinned picture, per node.
 *
 * One hook instance per node on purpose: `armed` is a property of the button
 * you are pointing at, and a single shared "armed" flag on the board would mean
 * arming one picture and confirming on the next one — the accident this guard
 * exists to prevent, rebuilt out of the guard itself.
 *
 * The arming disarms itself after DELETE_ARM_MS. A board is left open for
 * hours; a live delete that stays live under the cursor is a trap that fires on
 * the next unrelated click.
 *
 * No toast dependency, same reason as useImageDownload: this runs inside the
 * board AND inside a group member, and the error is drawn on the node itself.
 */
export function useCanvasImageDelete(onDeleted) {
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const timer = useRef(null);
  const alive = useRef(true);

  // Re-armed on every mount, not only on the first: StrictMode's mount /
  // unmount / re-mount would otherwise leave `alive` false for the whole life
  // of the component and every setState below would be skipped — the button
  // would say "…" forever (see useImageDownload for the same trap, found the
  // hard way).
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const disarm = useCallback(() => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    setArmed(false);
  }, []);

  /** One press. Arms if idle, deletes if already armed. Returns what it did, so
   *  a test can assert the two-step without a DOM. */
  const press = useCallback(async (node) => {
    if (busy) return 'busy';
    const target = canvasImageDeleteTarget(node);
    if (!target) { setError('This image cannot be traced back to a run'); return 'refused'; }
    if (!armed) {
      setError('');
      setArmed(true);
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => {
        timer.current = null;
        if (alive.current) setArmed(false);
      }, DELETE_ARM_MS);
      return 'armed';
    }
    disarm();
    setBusy(true);
    try {
      const res = await postJson(target.endpoint, { image_ids: [target.imageId] });
      // The node leaves the board here rather than waiting for a reload: the
      // canvas_image_node row now points at nothing and the server prunes it on
      // the next read, but the picture the user just deleted must not stay on
      // screen until then.
      onDeleted?.(node, res);
      return 'deleted';
    } catch (e) {
      if (alive.current) setError(canvasDeleteError(e));
      return 'failed';
    } finally {
      if (alive.current) setBusy(false);
    }
  }, [armed, busy, disarm, onDeleted]);

  return { press, armed, busy, error, disarm, clearError: () => setError('') };
}
