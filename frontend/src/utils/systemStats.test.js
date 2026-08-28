import assert from 'node:assert/strict';
import test from 'node:test';
import {
  HEADER_MACHINE_LOAD_PREF_KEY, MACHINE_LOAD_PREF_KEY, formatGb, loadTone,
  machineLoadSummary, readMachineLoadPref, shouldPoll, systemStatsSegments,
  tempTone, writeMachineLoadPref,
} from './systemStats.js';

const memoryStore = () => {
  const data = {};
  return {
    getItem: (key) => (key in data ? data[key] : null),
    setItem: (key, value) => { data[key] = String(value); },
  };
};

test('a full machine reads as one line, work first, memory second', () => {
  const segments = systemStatsSegments({
    cpu_percent: 34, gpu_percent: 87,
    vram_used_gb: 21.3, vram_total_gb: 24,
    ram_used_gb: 45.2, ram_total_gb: 63.9,
  });
  assert.deepEqual(segments.map((s) => `${s.label} ${s.text}`),
    ['CPU 34%', 'GPU 87%', 'VRAM 21/24G', 'RAM 45/64G']);
});

test('a machine that reports its GPU temperature shows it, in degrees, last', () => {
  const segments = systemStatsSegments({
    cpu_percent: 34, gpu_percent: 87,
    vram_used_gb: 21.3, vram_total_gb: 24,
    ram_used_gb: 45.2, ram_total_gb: 63.9,
    gpu_temp_c: 47,
  });
  assert.deepEqual(segments.map((s) => `${s.label} ${s.text}`),
    ['CPU 34%', 'GPU 87%', 'VRAM 21/24G', 'RAM 45/64G', 'Temp 47°']);
});

test('a GPU that cannot report its temperature costs one segment, not a 0°', () => {
  // Same omission rule as the GPU itself: [N/A] upstream means the key is
  // absent, and an absent key must never be drawn as a freezing card.
  const segments = systemStatsSegments({
    gpu_percent: 87, vram_used_gb: 21.3, vram_total_gb: 24,
  });
  assert.ok(!segments.some((s) => s.key === 'temp'));
});

test('a machine with no GPU draws two numbers, not four zeros', () => {
  // The whole reason the API omits fields: "GPU 0%" reads as an idle card.
  const segments = systemStatsSegments({
    cpu_percent: 12, ram_used_gb: 6.1, ram_total_gb: 16,
  });
  assert.deepEqual(segments.map((s) => s.key), ['cpu', 'ram']);
  assert.equal(segments.find((s) => s.key === 'ram').text, '6.1/16G');
});

test('a machine that answered nothing draws nothing at all', () => {
  assert.deepEqual(systemStatsSegments({}), []);
  assert.deepEqual(systemStatsSegments(null), []);
  assert.deepEqual(systemStatsSegments(undefined), []);
});

test('a total of zero is unknown, not a division by zero', () => {
  const segments = systemStatsSegments({ vram_used_gb: 0, vram_total_gb: 0 });
  assert.deepEqual(segments, []);
});

test('the tone steps at 50% and 80%, not only once a resource is in trouble', () => {
  assert.equal(loadTone(0.10), 'calm');
  assert.equal(loadTone(0.49), 'calm');
  assert.equal(loadTone(0.50), 'warm');
  assert.equal(loadTone(0.79), 'warm');
  assert.equal(loadTone(0.80), 'hot');
  assert.equal(loadTone(1), 'hot');
  // An unmeasured value must never paint a warning.
  assert.equal(loadTone(null), 'calm');
  assert.equal(loadTone(NaN), 'calm');
});

test('the tone of each segment comes from its own fraction, not the raw number', () => {
  const [cpu, gpu, vram, ram] = systemStatsSegments({
    cpu_percent: 95, gpu_percent: 5,
    vram_used_gb: 23.5, vram_total_gb: 24,
    ram_used_gb: 8, ram_total_gb: 64,
  });
  assert.equal(cpu.tone, 'hot');      // 95%
  assert.equal(gpu.tone, 'calm');     // 5%
  assert.equal(vram.tone, 'hot');     // 23.5/24 = 0.979
  assert.equal(ram.tone, 'calm');     // 8/64 = 0.125
});

test('temperature warns on the throttle band, not on half of anything', () => {
  // 50 °C is a card at rest; the load fractions cannot serve heat. Amber from
  // 70°, rose from 85° — the band where NVIDIA cards defend themselves.
  assert.equal(tempTone(47), 'calm');
  assert.equal(tempTone(69), 'calm');
  assert.equal(tempTone(70), 'warm');
  assert.equal(tempTone(84), 'warm');
  assert.equal(tempTone(85), 'hot');
  // An unmeasured temperature must never paint a warning.
  assert.equal(tempTone(null), 'calm');
  assert.equal(tempTone(NaN), 'calm');
  // …and the segment's tone is the same verdict the function gives.
  const [temp] = systemStatsSegments({ gpu_temp_c: 90 }).filter((s) => s.key === 'temp');
  assert.equal(temp.tone, 'hot');
});

test('gigabytes lose their decimal only where it is noise', () => {
  assert.equal(formatGb(21.3), '21');
  assert.equal(formatGb(23.99), '24');
  assert.equal(formatGb(9.6), '9.6');
  assert.equal(formatGb(4), '4');
  assert.equal(formatGb(null), '');
});

test('an out-of-range percentage is clamped instead of drawn as -3% or 140%', () => {
  const [low] = systemStatsSegments({ cpu_percent: -3 });
  const [high] = systemStatsSegments({ cpu_percent: 140 });
  assert.equal(low.text, '0%');
  assert.equal(high.text, '100%');
});

test('a hidden tab never polls, however enabled the widget is', () => {
  assert.equal(shouldPoll({ enabled: true, visibility: 'visible' }), true);
  assert.equal(shouldPoll({ enabled: true, visibility: 'hidden' }), false);
  assert.equal(shouldPoll({ enabled: false, visibility: 'visible' }), false);
  assert.equal(shouldPoll({ enabled: false, visibility: 'hidden' }), false);
});

test('an environment with no Page Visibility API still polls', () => {
  assert.equal(shouldPoll({ enabled: true, visibility: undefined }), true);
});

test('the readout is shown by default and remembers being folded away', () => {
  const store = memoryStore();
  assert.equal(readMachineLoadPref(store), true);
  writeMachineLoadPref(false, store);
  assert.equal(store.getItem(MACHINE_LOAD_PREF_KEY), 'off');
  assert.equal(readMachineLoadPref(store), false);
  writeMachineLoadPref(true, store);
  assert.equal(readMachineLoadPref(store), true);
});

test('storage that throws leaves the readout shown instead of crashing the board', () => {
  const hostile = {
    getItem() { throw new Error('storage disabled'); },
    setItem() { throw new Error('storage disabled'); },
  };
  assert.equal(readMachineLoadPref(hostile), true);
  assert.doesNotThrow(() => writeMachineLoadPref(false, hostile));
});

test('the header readout starts folded and remembers being opened', () => {
  // A header serves every page: the poll is opt-in there, where the Canvas
  // one is opt-out — and the two choices are remembered apart, under keys
  // that must never be renamed.
  const store = memoryStore();
  assert.equal(readMachineLoadPref(store, HEADER_MACHINE_LOAD_PREF_KEY, false), false);
  writeMachineLoadPref(true, store, HEADER_MACHINE_LOAD_PREF_KEY);
  assert.equal(store.getItem(HEADER_MACHINE_LOAD_PREF_KEY), 'on');
  assert.equal(readMachineLoadPref(store, HEADER_MACHINE_LOAD_PREF_KEY, false), true);
  // …without touching the Canvas choice.
  assert.equal(readMachineLoadPref(store), true);
  assert.notEqual(HEADER_MACHINE_LOAD_PREF_KEY, MACHINE_LOAD_PREF_KEY);
});

test('storage that throws leaves the header readout folded, its own default', () => {
  const hostile = { getItem() { throw new Error('storage disabled'); } };
  assert.equal(readMachineLoadPref(hostile, HEADER_MACHINE_LOAD_PREF_KEY, false), false);
});

test('the tooltip never promises a GPU the machine does not have', () => {
  const cpuOnly = systemStatsSegments({ cpu_percent: 20, ram_used_gb: 4, ram_total_gb: 16 });
  const summary = machineLoadSummary(cpuOnly);
  assert.equal(summary, 'Machine load: CPU 20% · RAM 4/16G');
  assert.ok(!summary.includes('GPU'));
  assert.equal(machineLoadSummary([]), 'Machine load — nothing measurable here');
});
