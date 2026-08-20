// Did the Appearance-in-captions policy actually move?
//
// The four omit/describe toggles only steer the NEXT caption run: images already
// captioned keep the rule they were written under (the same "future captions"
// contract as a kind or concept change). A dataset can therefore end up half under
// each rule — the ordinary Caption button only fills images that have no caption —
// so a policy edit has to nudge for a re-caption, and only a REAL edit should.
//
// Compared by value, not by identity: the popover fills the untouched families from
// APPEARANCE_DEFAULTS before saving, so `{hair: 'omit', ...}` reaching the server as a
// full four-family dict must not read as a change when it matches what was stored.
// No policy at all (null / undefined / {}) is the classic identity lock and canonicalizes
// to '', so lock -> policy and policy -> lock both count.

/** Canonical, order-independent form of a policy. '' = no policy (classic lock). */
export function appearanceKey(policy) {
  if (!policy || typeof policy !== 'object') return '';
  const keys = Object.keys(policy).sort();
  if (!keys.length) return '';
  return keys.map((k) => `${k}:${policy[k]}`).join(',');
}

/** True when saving moved the policy, i.e. future captions will follow a new rule. */
export function appearancePolicyChanged(before, after) {
  return appearanceKey(before) !== appearanceKey(after);
}
