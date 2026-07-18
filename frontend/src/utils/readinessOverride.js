/* The training-readiness « Continue anyway » gate. The server preflight is the
   authority (it returns `can_override` and a per-check `bypassable` flag, and it
   re-checks `allow_not_ready` at launch); these helpers only shape the checkbox
   and guarantee an acknowledgement never survives into a DIFFERENT blocker set. */

/** Stable signature of the blocking state. The checkbox resets whenever this
 * changes, so acknowledging a « too few images » blocker never carries over to,
 * say, a later « missing slider prompt » one. Only fail rows and can_override
 * matter — warning counts moving in the background must not reset the ack. */
export function readinessSignature(data) {
  if (!data) return '';
  const fails = (data.checks || [])
    .filter((c) => c && c.status === 'fail')
    .map((c) => `${c.id}:${c.bypassable ? 1 : 0}`)
    .sort()
    .join(',');
  return `${data.verdict || ''}|${data.can_override ? 1 : 0}|${fails}`;
}

/** The value to send to the server: true only when the override is BOTH offered
 * (can_override — every blocker is a bypassable quality guard-rail) and the box
 * is checked. A physical impossibility (can_override false) can never yield a
 * truthy ack, so the button stays gated and the launch never carries the flag. */
export function overrideAck(data, checked) {
  return !!(data && data.can_override && checked);
}
