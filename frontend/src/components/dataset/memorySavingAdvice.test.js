import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MEMORY_KEYS,
  memoryAdviceText,
  memoryCardLabel,
  memoryPatchFor,
  memoryIsOverridden,
  memoryRiskLine,
  memoryStateLabel,
} from './memorySavingAdvice.js';

test('a big card is told it can switch the savers off', () => {
  const t = memoryAdviceText({ verdict: 'can_disable', vram_gb: 31.4,
    gpu: 'NVIDIA GeForce RTX 5090', unquantised_vram_gb: 18 });
  assert.match(t, /RTX 5090/);
  assert.match(t, /31\.4 GB detected/);
  assert.match(t, /roughly 18 GB/);
  assert.match(t, /switch them off/);
});

test('a small card is told to leave them on, and the symptom is SLOWNESS', () => {
  // The whole point: on Windows/WDDM there is no clean OOM. A user warned about
  // "running out of memory" would wait for an error that never comes while the
  // run pages to system RAM for hours.
  const t = memoryAdviceText({ verdict: 'keep_on', vram_gb: 12,
    gpu: 'NVIDIA GeForce RTX 3060', unquantised_vram_gb: 30 });
  assert.match(t, /Leave them on/);
  assert.match(t, /slows to a crawl/);
  assert.match(t, /pages to system RAM/);
  assert.doesNotMatch(t, /out of memory|OOM|crash/i);
});

test('an undetectable machine gets a generic line that still names the symptom', () => {
  for (const advice of [undefined, {}, { verdict: 'unknown', unquantised_vram_gb: 18 },
    { verdict: 'can_disable' } /* verdict without a card → still generic */]) {
    const t = memoryAdviceText(advice);
    assert.match(t, /Card not detected/);
    assert.match(t, /slows to a crawl/);
  }
});

test('the card label degrades one half at a time', () => {
  assert.equal(memoryCardLabel({ gpu: 'RTX 4090', vram_gb: 24 }), 'RTX 4090 · 24 GB');
  assert.equal(memoryCardLabel({ gpu: 'RTX 4090' }), 'RTX 4090');
  assert.equal(memoryCardLabel({ vram_gb: 24 }), '24 GB');
  assert.equal(memoryCardLabel(null), '');
});

test('returning a box to the family default clears the key instead of pinning it', () => {
  const defaults = { quantize: true, quantize_te: true, low_vram: true };
  assert.deepEqual(memoryPatchFor('quantize', false, defaults), { quantize: false });
  assert.deepEqual(memoryPatchFor('quantize', true, defaults), { quantize: 'auto' });
  // Anima/SDXL default to OFF — the same lever, mirrored.
  const small = { quantize: false, quantize_te: false, low_vram: false };
  assert.deepEqual(memoryPatchFor('quantize', true, small), { quantize: true });
  assert.deepEqual(memoryPatchFor('quantize', false, small), { quantize: 'auto' });
});

test('a stored false counts as an override (it is the request, not an absence)', () => {
  assert.equal(memoryIsOverridden({ quantize: null, quantize_te: null, low_vram: null }), false);
  assert.equal(memoryIsOverridden({}), false);
  assert.equal(memoryIsOverridden({ quantize: false }), true);
  assert.equal(memoryIsOverridden({ low_vram: true }), true);
});

test('a mixed state never reads as "on"', () => {
  // Two of three off is exactly when the user has left the calibrated recipe and
  // most needs to see it; collapsing that to "on" would hide the change.
  const all = { quantize: true, quantize_te: true, low_vram: true };
  assert.match(memoryStateLabel(all), /^on —/);
  assert.match(memoryStateLabel({ quantize: false, quantize_te: false, low_vram: false }), /^all off/);
  assert.equal(memoryStateLabel({ ...all, quantize: false, low_vram: false }), 'partly off (2 of 3)');
});

test('the three keys stay the ones the backend validates', () => {
  assert.deepEqual(MEMORY_KEYS, ['quantize', 'quantize_te', 'low_vram']);
});

/* --- memoryRiskLine: the sentence that was missing entirely ------------------
   A `quantize: false` chosen on Anima or SDXL (2B — where OFF is the calibrated
   default) travelled, via the single global train_settings blob, onto a 12B DiT
   and produced a Krea/FLUX config with no quantisation and no low-VRAM
   streaming. Nothing anywhere said so. */

test('an intact recipe stays visually silent', () => {
  assert.equal(memoryRiskLine(null, 'Krea 2'), null);
  assert.equal(memoryRiskLine({ disabled: [] }, 'Krea 2'), null);
});

test('a disabled saver on a hungry family names it, the family and the need', () => {
  const t = memoryRiskLine({ disabled: ['quantize', 'low_vram'], verdict: 'keep_on',
    vram_gb: 24, unquantised_vram_gb: 30 }, 'Krea 2');
  assert.match(t, /Quantise base model, Low-VRAM loading/);
  assert.match(t, /Krea 2 needs roughly 30 GB/);
  assert.match(t, /this card reports 24 GB/);
  // Same discipline as memoryAdviceText: the symptom is SLOWNESS, not a clean OOM.
  assert.match(t, /slows to a crawl/);
  assert.doesNotMatch(t, /out of memory/i);
});

test('an undetected card says so rather than guessing', () => {
  const t = memoryRiskLine({ disabled: ['quantize'], verdict: 'unknown',
    vram_gb: null, unquantised_vram_gb: 30 }, 'FLUX.1');
  assert.match(t, /no card was detected here/);
});

test('a card that genuinely covers it is stated, not warned about', () => {
  // Issue #14 was a request to STOP paying a 24 GB tax on a 5090. The line must
  // not turn that legitimate choice back into an alarm.
  const t = memoryRiskLine({ disabled: ['quantize'], verdict: 'can_disable',
    vram_gb: 31.4, gpu: 'NVIDIA GeForce RTX 5090', unquantised_vram_gb: 30 }, 'Krea 2');
  assert.match(t, /RTX 5090 · 31\.4 GB covers/);
  assert.doesNotMatch(t, /slows to a crawl/);
});

test('the family is named even when the server sent no label', () => {
  const t = memoryRiskLine({ disabled: ['quantize'], verdict: 'keep_on',
    vram_gb: 8, unquantised_vram_gb: 30 }, undefined);
  assert.match(t, /this family needs roughly 30 GB/);
});
