import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildStartPayload, isLiveRunning, paceLine, pauseLine, sceneCount, streamUrlFor,
} from './liveStudioApi.js';

test('scenes are counted between --- lines, blank blocks ignored', () => {
  assert.equal(sceneCount('a\n---\nb\n  ---  \n\n---\nc'), 3);
  assert.equal(sceneCount(''), 0);
  assert.equal(sceneCount(null), 0);
});

test('the start payload says auto as 0, leaves empty options out, carries the LoRA when picked', () => {
  const base = { scenes: 's', subject: 'Jessy', megapixels: '0.3', aspect: 'square', frames: '124',
    fps: 'auto', turbo: true, steps: 4 };
  const p = buildStartPayload(base);
  assert.deepEqual(p, { scenes: 's', subject: 'Jessy', megapixels: 0.3, aspect: 'square',
    frames: 124, fps: 0, turbo: true, steps: 4 });
  assert.equal(buildStartPayload({ ...base, fps: 18 }).fps, 18);
  assert.equal(buildStartPayload({ ...base, fps: '' }).fps, 0);
  assert.equal('lora' in p, false);
  const withLora = buildStartPayload({ ...base, lora: 'h3/lds/x.safetensors', loraStrength: '1.3' });
  assert.equal(withLora.lora, 'h3/lds/x.safetensors');
  assert.equal(withLora.lora_strength, 1.3);
  assert.equal(buildStartPayload({ ...base, turbo: false }).turbo, false);
  assert.equal('seed' in buildStartPayload({ ...base, seed: '' }), false);
  assert.equal(buildStartPayload({ ...base, seed: '7' }).seed, 7);
});

test('the stream address is the playlist on the app origin — what VLC opens', () => {
  assert.equal(streamUrlFor({ playlist: '/api/video-studio/live/ab12/stream.m3u8' }, 'http://192.0.2.10:5050'),
    'http://192.0.2.10:5050/api/video-studio/live/ab12/stream.m3u8');
  assert.equal(streamUrlFor(null, 'http://x'), null);
  assert.equal(streamUrlFor({ playlist: '/p' }, ''), null);
});

test('running means starting, running or stopping — never idle or stopped', () => {
  for (const s of ['starting', 'running', 'stopping']) assert.equal(isLiveRunning({ state: s }), true);
  for (const s of ['idle', 'stopped']) assert.equal(isLiveRunning({ state: s }), false);
  assert.equal(isLiveRunning(null), false);
});

test('the pace line says measuring, keeping up or behind with the numbers that decide it', () => {
  assert.equal(paceLine(null), '');
  assert.equal(paceLine({ state: 'idle' }), '');
  assert.match(paceLine({ state: 'running', pace: 'measuring', produced: 0, measured: 1, play_fps: null }),
    /Measuring the pace — 1 clip measured, playback starts after 1 more\./);
  const up = paceLine({ state: 'running', pace: 'keeping_up', produced: 5, play_fps: 12,
    play_seconds: 10.33, render_seconds: 8.1, sustain_fps: 15.3, margin_seconds: 2.23 });
  assert.match(up, /Playing at 12 fps \(motion at 50 % speed\)/);
  assert.match(up, /plays for 10\.33 s and renders in 8\.1 s — the card sustains 15\.3 fps/);
  assert.match(up, /Keeping up, 2\.23 s of buffer gained per clip\./);
  const behind = paceLine({ state: 'running', pace: 'behind', produced: 5, play_fps: 24,
    play_seconds: 5.17, render_seconds: 8.1, sustain_fps: 15.3, margin_seconds: -2.93, runway_clips: 3 });
  assert.match(behind, /Behind by 2\.93 s per clip — 3 clips of buffer left\./);
  const empty = paceLine({ state: 'running', pace: 'behind', produced: 5, play_fps: 24,
    play_seconds: 5.17, render_seconds: 8.1, sustain_fps: 15.3, margin_seconds: -2.93, runway_clips: null });
  assert.match(empty, /the player waits between clips\. Pick a lower rate, lengthen the clips or drop the resolution\./);
  // At the floor there is no lower rate to pick — and shorter clips never help.
  const floor = paceLine({ state: 'running', pace: 'behind', produced: 5, play_fps: 6,
    play_seconds: 9.33, render_seconds: 36, sustain_fps: 1.6, margin_seconds: -26.67, runway_clips: null });
  assert.match(floor, /This is the lowest rate: lengthen the clips or drop the resolution\./);
  assert.doesNotMatch(floor, /lower rate/);
  assert.doesNotMatch(paceLine({ state: 'running', pace: 'behind', produced: 1, play_fps: 12,
    play_seconds: 10, render_seconds: 20, sustain_fps: 6, margin_seconds: -10 }), /shorten/);
  assert.equal(paceLine({ state: 'stopped', produced: 12 }), 'Channel stopped — 12 clips streamed.');
});

test('the pause line speaks only while the channel runs and the producer waits for the player', () => {
  assert.equal(pauseLine(null), '');
  assert.equal(pauseLine({ state: 'running', paused_for_viewer: false }), '');
  assert.equal(pauseLine({ state: 'stopped', paused_for_viewer: true }), '');
  assert.match(pauseLine({ state: 'running', paused_for_viewer: true }), /paused until the player catches up/);
});
