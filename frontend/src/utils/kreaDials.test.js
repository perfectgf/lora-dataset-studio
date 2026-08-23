/* Contract for Krea's two previously unreachable dials.

   The bug behind this file: `krea.ref_boost` and `krea.identity_lora_strength`
   were read by every Krea run and editable from nowhere. Someone whose subject
   came out too loosely like the reference had exactly one lever — and it was
   invisible. The regressions worth pinning are therefore (a) the bounds match
   the server's clamps, so the UI never offers a number that would be silently
   corrected, (b) "not loaded yet" never renders as 0 (Number(null) is 0, and a
   reference pull of 0 means "off"), and (c) a drag becomes ONE settings write,
   not forty. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  KREA_REF_BOOST_MIN, KREA_REF_BOOST_MAX,
  KREA_IDENTITY_STRENGTH_MIN, KREA_IDENTITY_STRENGTH_MAX,
  KREA_GROUNDING_MIN, KREA_GROUNDING_MAX, KREA_GROUNDING_STEP,
  KREA_STEPS_MIN, KREA_STEPS_MAX,
  clampRefBoost, clampIdentityStrength, clampGrounding, clampSteps,
  refBoostDescription, identityStrengthDescription, stepsDescription,
  kreaDialPayload, createDialSaver,
} from './kreaDials.js';

// --- bounds mirror the server -----------------------------------------------

test('the bounds are the ones krea_edit_helper clamps to', () => {
  // _ref_boost: _clamp(..., 0.0, 10.0, 0.25) / _identity_strength: (..., 0.0, 1.5, 1.0)
  assert.deepEqual([KREA_REF_BOOST_MIN, KREA_REF_BOOST_MAX], [0, 10]);
  assert.deepEqual([KREA_IDENTITY_STRENGTH_MIN, KREA_IDENTITY_STRENGTH_MAX], [0, 1.5]);
  assert.equal(clampRefBoost(99), 10);
  assert.equal(clampRefBoost(-3), 0);
  assert.equal(clampIdentityStrength(9), 1.5);
  assert.equal(clampIdentityStrength(-1), 0);
});

// --- absent is not zero ------------------------------------------------------

test('a value still loading falls back to the default, not to 0', () => {
  // Number(null) === 0, and 0 reference pull means "off" — the exact opposite of
  // "we do not know yet". Same for '' from a blanked config field.
  assert.equal(clampRefBoost(null, 0.25), 0.25);
  assert.equal(clampRefBoost(undefined, 0.25), 0.25);
  assert.equal(clampRefBoost('', 0.25), 0.25);
  assert.equal(clampIdentityStrength(null, 1.0), 1.0);
  // ...but an explicit 0 IS a choice and survives.
  assert.equal(clampRefBoost(0, 0.25), 0);
  assert.equal(clampIdentityStrength(0, 1.0), 0);
});

test('the fallback comes from the server, and only then from the mirrored literal', () => {
  // config_defaults said 2.0 → that is what an unset value reads as.
  assert.equal(clampRefBoost(null, 2.0), 2.0);
  assert.equal(clampIdentityStrength(null, 0.8), 0.8);
  // A backend too old to send config_defaults: the literal mirrors the server's
  // own clamp fallback, so we land on the value the graph would have used.
  assert.equal(clampRefBoost(null, undefined), 0.25);
  assert.equal(clampIdentityStrength(null, undefined), 1.0);
  // A nonsense default is not trusted either.
  assert.equal(clampRefBoost(null, 'banana'), 0.25);
});

test('slider float noise is rounded, never accumulated', () => {
  assert.equal(clampRefBoost(0.1 + 0.2), 0.3);
  assert.equal(clampIdentityStrength('1.0500000000000001'), 1.05);
  assert.equal(clampRefBoost('2.5'), 2.5);          // <input type=range> gives strings
});

// --- the number is explained -------------------------------------------------

test('every dial position gets a phrase, and 0 says "off"', () => {
  assert.match(refBoostDescription(0), /off/);
  assert.match(identityStrengthDescription(0), /off/);
  for (const v of [0, 0.25, 1, 3, 10]) {
    assert.ok(refBoostDescription(v).trim().length > 3, `no phrase for ref_boost ${v}`);
  }
  for (const v of [0, 0.5, 1, 1.5]) {
    assert.ok(identityStrengthDescription(v).trim().length > 3, `no phrase for identity ${v}`);
  }
  // The trained weight is named as such rather than left as a bare "1".
  assert.match(identityStrengthDescription(1), /trained/);
});

// --- the two dials that joined them ------------------------------------------

test('grounding and steps mirror the server clamps too', () => {
  // krea_edit_helper: GROUNDING_PX_MIN/MAX = 512/1536, _steps() = _clamp(.., 1, 50).
  assert.deepEqual([KREA_GROUNDING_MIN, KREA_GROUNDING_MAX], [512, 1536]);
  assert.deepEqual([KREA_STEPS_MIN, KREA_STEPS_MAX], [1, 50]);
  assert.equal(clampGrounding(9999), 1536);
  assert.equal(clampGrounding(1), 512);
  assert.equal(clampSteps(999), 50);
  assert.equal(clampSteps(0), 1);
});

test('grounding snaps to the 64px grid the server rounds to', () => {
  // _grounding() is int(round(v / 64) * 64): a UI showing 1000 while the graph
  // receives 1024 is exactly the "my settings are ignored" report we chase.
  assert.equal(clampGrounding(1000) % KREA_GROUNDING_STEP, 0);
  assert.equal(clampGrounding(1000), 1024);
  assert.equal(clampGrounding(1216), 1216);   // already on a stop, untouched
  assert.equal(clampGrounding('768'), 768);   // <input type=range> gives strings
});

test('steps stay integers — a KSampler cannot run 13.5 of them', () => {
  assert.equal(clampSteps('13'), 13);
  assert.equal(clampSteps(13.4), 13);
  assert.equal(Number.isInteger(clampSteps(7.6)), true);
});

test('grounding and steps also fall back to the server default, not to 0', () => {
  assert.equal(clampGrounding(null, 768), 768);
  assert.equal(clampSteps(undefined, 12), 12);
  // Backend too old to send config_defaults: the mirrored literal is the
  // server's OWN fallback (512 / 8), so we land where the graph would.
  assert.equal(clampGrounding(null, undefined), 512);
  assert.equal(clampSteps(null, undefined), 8);
});

test('the two new dials get a phrase as well', () => {
  for (const v of [1, 8, 20, 50]) {
    assert.ok(stepsDescription(v).trim().length > 3, `no phrase for steps ${v}`);
    assert.ok(stepsDescription(v).startsWith(String(v)), 'the phrase must name the value');
  }
  assert.match(stepsDescription(50), /no expected quality gain/);
});

// --- the payload is PARTIAL --------------------------------------------------

test('the save body touches only the keys that moved', () => {
  assert.deepEqual(kreaDialPayload({ ref_boost: 3 }), { config: { krea: { ref_boost: 3 } } });
  // Both dials moved before the debounce fired → one request carrying both, and
  // still nothing else in the krea section (the endpoint deep-merges).
  assert.deepEqual(
    kreaDialPayload({ ref_boost: 3, identity_lora_strength: 1.2 }),
    { config: { krea: { ref_boost: 3, identity_lora_strength: 1.2 } } });
});

// --- a drag is one write -----------------------------------------------------

/** Minimal controllable clock: run() fires whatever is currently scheduled. */
const fakeTimers = () => {
  const jobs = new Map();
  let next = 1;
  return {
    setTimeoutFn: (fn) => { jobs.set(next, fn); return next++; },
    clearTimeoutFn: (id) => { jobs.delete(id); },
    run() { const due = [...jobs.values()]; jobs.clear(); for (const fn of due) fn(); },
    get scheduled() { return jobs.size; },
  };
};

test('forty change events become ONE settings write', () => {
  const clock = fakeTimers();
  const sent = [];
  const saver = createDialSaver((patch) => sent.push(patch), { ...clock });
  for (let i = 0; i <= 40; i += 1) saver.schedule('ref_boost', i / 4);
  assert.equal(sent.length, 0, 'nothing must go out mid-drag');
  clock.run();
  assert.deepEqual(sent, [{ ref_boost: 10 }]);       // only the value it landed on
});

test('two dials moved before the delay elapses share one request', () => {
  const clock = fakeTimers();
  const sent = [];
  const saver = createDialSaver((patch) => sent.push(patch), { ...clock });
  saver.schedule('ref_boost', 2);
  saver.schedule('identity_lora_strength', 1.2);
  saver.schedule('ref_boost', 3);                    // last one wins for its key
  clock.run();
  assert.deepEqual(sent, [{ ref_boost: 3, identity_lora_strength: 1.2 }]);
});

test('flush sends what is pending (leaving the screen mid-drag still saves)', () => {
  const clock = fakeTimers();
  const sent = [];
  const saver = createDialSaver((patch) => sent.push(patch), { ...clock });
  saver.schedule('ref_boost', 4);
  saver.flush();
  assert.deepEqual(sent, [{ ref_boost: 4 }]);
  assert.equal(clock.scheduled, 0, 'the timer must not fire a second time');
  clock.run();
  assert.equal(sent.length, 1);
  // Flushing with nothing pending is a no-op, not an empty PUT.
  saver.flush();
  assert.equal(sent.length, 1);
});

test('cancel drops the pending write', () => {
  const clock = fakeTimers();
  const sent = [];
  const saver = createDialSaver((patch) => sent.push(patch), { ...clock });
  saver.schedule('ref_boost', 4);
  saver.cancel();
  clock.run();
  assert.deepEqual(sent, []);
});
