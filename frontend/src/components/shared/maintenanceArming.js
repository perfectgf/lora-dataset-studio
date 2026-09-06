/* The second press of 🧹 Free memory and ↻ Restart ComfyUI (2026-09-06).
 *
 * A refusal is not a wall: when the obstacle is a render of LDS's own, the
 * server says so (`can_interrupt` / `can_force`, with the job's id) and the
 * SAME button pressed again within a minute sends the forced form for THAT
 * job. This module owns the arming so it can be exercised for real (found
 * in refutation: a source-regex contract let an unconditional arming through):
 *   • armed ONLY by a refusal that carries the matching flag — never by a
 *     training's, a foreign prompt's or a network refusal;
 *   • a press that follows the refusal within ARM_SETTLE_MS is a double-click
 *     (the refusal comes back in ~3 ms, a double-click takes 150-500 ms), not
 *     a decision: it is not the second press;
 *   • consumed by the next press of that button, and expired after ARM_MS;
 *   • one store for every mount (header and Canvas), keyed by button.
 */
export const ARM_MS = 60000;
export const ARM_SETTLE_MS = 600;
export const OFFER_TOAST_MS = 12000;
const FLAG = { free: 'can_interrupt', restart: 'can_force' };
const store = new Map();

/** Arm `kind` from a refusal body; true when it offered the second press. */
export function armFrom(kind, body, now = Date.now()) {
  const flag = FLAG[kind];
  if (!flag || !body || body[flag] !== true) return false;
  const jobId = typeof body.job_id === 'string' && body.job_id ? body.job_id : null;
  store.set(kind, { since: now, until: now + ARM_MS, jobId });
  return true;
}

/** When the arming of `kind` expires (0 when there is none). */
export function armedUntil(kind) {
  return store.get(kind)?.until ?? 0;
}

/** Whether a press of `kind` now would be the second press. */
export function isArmed(kind, now = Date.now()) {
  const a = store.get(kind);
  return !!a && now >= a.since + ARM_SETTLE_MS && now < a.until;
}

/** Take the arming for this press: `{ jobId }` when it is the second press,
 * null otherwise (nothing armed, a double-click, or an expired offer). The
 * arming is spent either way — a stale intent never outlives a press. */
export function consume(kind, now = Date.now()) {
  const a = store.get(kind);
  if (!a) return null;
  store.delete(kind);
  if (now < a.since + ARM_SETTLE_MS || now >= a.until) return null;
  return { jobId: a.jobId };
}

export function disarm(kind) {
  store.delete(kind);
}

export function resetArmingForTests() {
  store.clear();
}
