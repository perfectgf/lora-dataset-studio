/** 🔒 Whether a slider is locked, and where that answer is kept.
 *
 * Sliders in this app are LOCKED by default. On a phone a range input eats the
 * gesture that crosses it: scrolling past a panel drags the dial under the
 * thumb and the change is silent — the number moves, nothing says so, and the
 * next render runs on a setting nobody chose. A padlock costs one tap on the
 * rare occasion somebody means to move it, and nothing the rest of the time.
 *
 * The state lives per slider in localStorage rather than per session: someone
 * who unlocked "LoRA strength" is working on strengths, and re-locking it on
 * every visit would be the app forgetting what they are doing.
 *
 * Pure on purpose — the reading and writing are what break (a private window,
 * storage disabled, a value written by an older build), and they are the two
 * things a component test cannot reach.
 */

/** Locked unless the user said otherwise. An unreadable store answers LOCKED:
 *  the safe end of the guess is the one that cannot change a value by itself. */
export function readLock(key, storage) {
  if (!key) return true;
  try {
    const raw = (storage || globalThis.localStorage)?.getItem(key);
    if (raw === null || raw === undefined) return true;
    return raw === 'true';
  } catch {
    return true;
  }
}

/** Remember it, and never let a full or disabled store break the toggle: the
 *  lock still opens for this page, it simply will not be there next time. */
export function writeLock(key, locked, storage) {
  if (!key) return locked;
  try {
    (storage || globalThis.localStorage)?.setItem(key, String(locked));
  } catch {
    /* storage unavailable — the toggle still works for this page */
  }
  return locked;
}
