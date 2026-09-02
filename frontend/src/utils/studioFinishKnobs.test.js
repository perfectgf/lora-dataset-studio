/* Contract for the per-run hi-res fix and finishing blocks of the Studio panel.

   The regression worth pinning is the THIRD state: a control the user never
   touched must send nothing (defer to Settings), not a 1.0 that would switch a
   Settings default off behind their back — and an explicit "off" must send a
   1.0 that wins over a Settings default of 1.5. Both mistakes render, and both
   render something the user did not ask for. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  FINISH_REFERENCE, HIRES_DEFAULTS_FALLBACK, HIRES_SCALE_CHOICES,
  finishPayload, hiresDefaultLabel, hiresIsOn, hiresPayload, normaliseHiresDefaults,
} from './studioFinishKnobs.js';

const OFF = { scale: 1, denoise: 0.5, steps: 0 };
const ON = { scale: 1.5, denoise: 0.5, steps: 0 };

test('an untouched control sends nothing when Settings is off', () => {
  assert.deepEqual(hiresPayload({ scale: '', denoise: 0.5 }, OFF), {});
  assert.deepEqual(hiresPayload({ scale: undefined, denoise: 0.5 }, OFF), {});
});

test('an untouched control defers the scale to Settings but may still set this run\'s rewrite', () => {
  // Settings decides the factor; the slider the user CAN see decides the rewrite.
  assert.deepEqual(hiresPayload({ scale: '', denoise: 0.4 }, ON), { hires_denoise: 0.4 });
  assert.equal('hires_scale' in hiresPayload({ scale: '', denoise: 0.4 }, ON), false);
});

test('an explicit off is sent as 1.0 and wins over a Settings default that is on', () => {
  assert.deepEqual(hiresPayload({ scale: '1', denoise: 0.5 }, ON), { hires_scale: 1 });
  // ...and carries no rewrite: there is no pass for it to describe.
  assert.equal('hires_denoise' in hiresPayload({ scale: '1', denoise: 0.5 }, ON), false);
});

test('an explicit factor is sent with its rewrite, clamped to the ceiling', () => {
  assert.deepEqual(hiresPayload({ scale: '1.5', denoise: 0.4 }, OFF),
    { hires_scale: 1.5, hires_denoise: 0.4 });
  assert.equal(hiresPayload({ scale: '9', denoise: 0.4 }, OFF).hires_scale, 2);
  assert.equal(hiresPayload({ scale: '1.5', denoise: 7 }, OFF).hires_denoise, 1);
  assert.equal(hiresPayload({ scale: '1.5', denoise: -1 }, OFF).hires_denoise, 0.05);
});

test('an unparseable value is treated as untouched, never as off', () => {
  // 'abc' -> deferred: with Settings on, that is the denoise-only shape, and
  // crucially NOT { hires_scale: 1 } — garbage must not disable a default.
  assert.deepEqual(hiresPayload({ scale: 'abc', denoise: 0.5 }, ON), { hires_denoise: 0.5 });
  assert.deepEqual(hiresPayload({ scale: 'abc', denoise: 0.5 }, OFF), {});
});

test('whether the pass runs follows the choice, then Settings', () => {
  assert.equal(hiresIsOn('', OFF), false);
  assert.equal(hiresIsOn('', ON), true);
  assert.equal(hiresIsOn('1', ON), false);
  assert.equal(hiresIsOn('1.5', OFF), true);
});

test('the deferral option names what Settings holds', () => {
  assert.equal(hiresDefaultLabel(OFF), 'Settings default (off)');
  assert.equal(hiresDefaultLabel(ON), 'Settings default (1.5×, rewrite 0.5)');
});

test('missing or broken defaults fall back to the shipped ones (off)', () => {
  assert.deepEqual(normaliseHiresDefaults(undefined), HIRES_DEFAULTS_FALLBACK);
  assert.deepEqual(normaliseHiresDefaults({ scale: 'x', denoise: null }), HIRES_DEFAULTS_FALLBACK);
  assert.equal(normaliseHiresDefaults({ scale: '1.75' }).scale, 1.75);
});

test('finishing sends a key per stage that is on, and nothing for off', () => {
  assert.deepEqual(finishPayload({ sharpen: 0, grain: 0 }), {});
  assert.deepEqual(finishPayload({ sharpen: 0.55, grain: 0 }), { finish_sharpen: 0.55 });
  assert.deepEqual(finishPayload({ sharpen: 0, grain: 0.01 }), { finish_grain: 0.01 });
  assert.deepEqual(finishPayload(FINISH_REFERENCE),
    { finish_sharpen: 0.55, finish_grain: 0.01 });
  assert.deepEqual(finishPayload({ sharpen: 'x', grain: NaN }), {});
});

test('finishing clamps to the server ceilings', () => {
  assert.equal(finishPayload({ sharpen: 99, grain: 0 }).finish_sharpen, 3);
  assert.equal(finishPayload({ sharpen: 0, grain: 99 }).finish_grain, 0.2);
});

test('the offered factors stop at the injector ceiling', () => {
  assert.ok(HIRES_SCALE_CHOICES.every((v) => v > 1 && v <= 2));
});

test('a missing rewrite is omitted, never sent as the clamp floor', () => {
  /* `Number(null)` and `Number('')` are 0 — finite, and 0.05 after the clamp.
     A run whose slider never rendered (Settings off, then a factor picked) must
     not arrive with a 0.05 rewrite it never asked for. */
  for (const missing of [null, undefined, '', []]) {
    const out = hiresPayload({ scale: '1.5', denoise: missing }, OFF);
    assert.deepEqual(out, { hires_scale: 1.5 }, `denoise=${String(missing)}`);
  }
  assert.deepEqual(hiresPayload({ scale: '', denoise: null }, ON), {});
});
