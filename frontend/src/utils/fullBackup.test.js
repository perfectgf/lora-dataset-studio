import test from 'node:test';
import assert from 'node:assert/strict';
import {
  formatBytes,
  describeProgress,
  progressPercent,
  summarizeBackupResult,
  summarizeRestoreReport,
  isSettled,
} from './fullBackup.js';

test('formatBytes uses GB/MB/KB steps', () => {
  assert.equal(formatBytes(0), '0 KB');
  assert.equal(formatBytes(2048), '2 KB');
  assert.equal(formatBytes(5e6), '5 MB');
  assert.equal(formatBytes(1.2e9), '1.2 GB');
});

test('describeProgress: preparing before the count, then X / N', () => {
  assert.equal(describeProgress(null), '');
  assert.equal(describeProgress({ state: 'done', total: 3, done: 3 }), '');
  assert.equal(describeProgress({ state: 'running', total: 0, done: 0 }), 'Preparing…');
  assert.equal(describeProgress({ state: 'running', total: 12, done: 3 }), 'Backing up 3 / 12 datasets…');
  assert.equal(describeProgress({ state: 'running', total: 12, done: 3 }, 'Restoring'),
    'Restoring 3 / 12 datasets…');
  // done can never exceed total in the caption.
  assert.equal(describeProgress({ state: 'running', total: 5, done: 9 }), 'Backing up 5 / 5 datasets…');
});

test('progressPercent is null until a total is known', () => {
  assert.equal(progressPercent({ total: 0, done: 0 }), null);
  assert.equal(progressPercent({ total: 4, done: 1 }), 25);
  assert.equal(progressPercent({ total: 4, done: 99 }), 100);
});

test('summarizeBackupResult headlines count + size and lists skips', () => {
  const clean = summarizeBackupResult({
    name: 'x.zip', size_bytes: 1.2e9, datasets_total: 3, datasets_backed_up: 3, skipped: [],
  });
  assert.equal(clean.headline, 'Backup ready — 3 datasets, 1.2 GB');
  assert.deepEqual(clean.notes, []);

  const withSkip = summarizeBackupResult({
    size_bytes: 5e6, datasets_backed_up: 1,
    skipped: [{ name: 'Busy', reason: 'a file was locked' }],
  });
  assert.equal(withSkip.headline, 'Backup ready — 1 dataset, 5 MB');
  assert.equal(withSkip.notes[0], '1 dataset skipped:');
  assert.match(withSkip.notes[1], /Busy — a file was locked/);
});

test('summarizeRestoreReport is honest about restored/renamed/skipped/config', () => {
  const r = summarizeRestoreReport({
    datasets_total: 3,
    restored: 2,
    renamed: [{ from: 'Alice', to: 'Alice (restored)' }],
    skipped: [{ entry: 'datasets/9-bad.zip', reason: 'corrupt' }],
    config_restored: true,
  });
  assert.equal(r.headline, 'Restored 2 of 3 datasets');
  assert.ok(r.notes.some((n) => /Settings restored/.test(n)));
  assert.ok(r.notes.some((n) => /Renamed .Alice. →/.test(n)));
  assert.ok(r.notes.some((n) => /datasets\/9-bad\.zip — corrupt/.test(n)));
});

test('summarizeBackupResult surfaces training history and bundled LoRAs', () => {
  const withRuns = summarizeBackupResult({
    size_bytes: 5e6, datasets_backed_up: 2, runs_total: 3, skipped: [],
    loras_included: true, loras_total: 4, loras_bytes: 8e8,
  });
  assert.ok(withRuns.notes.some((n) => /Training history included \(3 runs\)/.test(n)));
  assert.ok(withRuns.notes.some((n) => /4 trained LoRAs bundled \(800 MB\)/.test(n)));

  const optedInEmpty = summarizeBackupResult({
    size_bytes: 1e6, datasets_backed_up: 1, runs_total: 0,
    loras_included: true, loras_total: 0,
  });
  assert.ok(optedInEmpty.notes.some((n) => /No deployed LoRA files/.test(n)));
});

test('summarizeRestoreReport reports restored runs and skipped LoRAs', () => {
  const r = summarizeRestoreReport({
    datasets_total: 2, restored: 2, runs_restored: 2, loras_restored: 1,
    loras_skipped: [{ name: 'lora_bob.safetensors', reason: 'ComfyUI not configured on this machine' }],
  });
  assert.ok(r.notes.some((n) => /Training history restored \(2 runs\)/.test(n)));
  assert.ok(r.notes.some((n) => /1 trained LoRA re-deployed/.test(n)));

  const resync = summarizeRestoreReport({
    datasets_total: 1, restored: 1, runs_restored: 0, runs_resynced: 1,
  });
  assert.ok(resync.notes.some((n) => /1 dataset re-detected as trained from files already/.test(n)));
  assert.ok(r.notes.some((n) => /1 LoRA file not restored:/.test(n)));
  assert.ok(r.notes.some((n) => /lora_bob\.safetensors — ComfyUI not configured/.test(n)));
});

test('isSettled only for done/error', () => {
  assert.equal(isSettled({ state: 'running' }), false);
  assert.equal(isSettled({ state: 'done' }), true);
  assert.equal(isSettled({ state: 'error' }), true);
  assert.equal(isSettled(null), false);
});
