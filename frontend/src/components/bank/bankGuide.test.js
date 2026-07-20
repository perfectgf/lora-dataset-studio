import test from 'node:test';
import assert from 'node:assert/strict';
import { nextBankStep, BANK_ZONES } from './bankGuide.js';

test('nextBankStep: nothing scanned -> analyze', () => {
  assert.equal(nextBankStep({ scanned: 0, scored: 0, keep: 0, scoringAvailable: true }), 'analyze');
});
test('nextBankStep: scanned but not scored (scoring available) -> analyze', () => {
  assert.equal(nextBankStep({ scanned: 100, scored: 0, keep: 0, scoringAvailable: true }), 'analyze');
});
test('nextBankStep: scored, nothing kept -> triage', () => {
  assert.equal(nextBankStep({ scanned: 100, scored: 100, keep: 0, scoringAvailable: true }), 'triage');
});
test('nextBankStep: some kept -> promote', () => {
  assert.equal(nextBankStep({ scanned: 100, scored: 100, keep: 12, scoringAvailable: true }), 'promote');
});
test('nextBankStep: scanned, scoring NOT available, nothing kept -> triage (skip Score)', () => {
  assert.equal(nextBankStep({ scanned: 100, scored: 0, keep: 0, scoringAvailable: false }), 'triage');
});
test('accent never lands on curate', () => {
  const ids = [
    { scanned: 0, scored: 0, keep: 0, scoringAvailable: true },
    { scanned: 100, scored: 0, keep: 0, scoringAvailable: true },
    { scanned: 100, scored: 100, keep: 0, scoringAvailable: true },
    { scanned: 100, scored: 100, keep: 5, scoringAvailable: true },
  ].map(nextBankStep);
  assert.ok(!ids.includes('curate'));
});
test('BANK_ZONES ordered analyze->triage->curate->promote', () => {
  assert.deepEqual(BANK_ZONES.map((z) => z.id), ['analyze', 'triage', 'curate', 'promote']);
});
