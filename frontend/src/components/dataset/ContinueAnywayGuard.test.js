import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const readiness = fs.readFileSync(new URL('./TrainingReadiness.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const workspace = fs.readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');

test('the readiness pastille shows the checkbox ONLY when the server offers the override', () => {
  // Gated on can_override — a physical impossibility never renders the option.
  assert.match(readiness, /data\.can_override\s*&&\s*\(/);
  assert.match(readiness, /Continue anyway — train with these issues unresolved/);
  // The honest per-blocker risk line comes from the server.
  assert.match(readiness, /data\.override_hint/);
});

test('the checkbox resets whenever the blocking state changes and reports the ack up', () => {
  assert.match(readiness, /readinessSignature\(data\)/);
  assert.match(readiness, /useEffect\(\(\)\s*=>\s*\{\s*setAck\(false\);\s*\},\s*\[sig\]\)/);
  assert.match(readiness, /onOverrideChange\?\.\(overrideAck\(data, ack\)\)/);
});

test('the workspace lifts the ack from the pastille to the training panel', () => {
  assert.match(workspace, /onOverrideChange=\{setNotReadyAck\}/);
  assert.match(workspace, /allowNotReady=\{notReadyAck\}/);
});

test('the Train button relaxes the image-floor gate only when acknowledged', () => {
  // belowFloor folds allowNotReady in, so a physical 0-image case (ack forced
  // false server-side) still disables the button.
  assert.match(panel, /const belowFloor = keptCount < trainMinFloor && !allowNotReady/);
  assert.match(panel, /disabled=\{!status\.installed \|\| belowFloor/);
});

test('the preflight gate lets a bypassable blocker through only with the ack', () => {
  assert.match(panel, /if \(!\(d\.can_override && allowNotReady\)\) \{ toast\.error\(d\.blockers\.join/);
});

test('every launch lane carries allow_not_ready when acknowledged', () => {
  // local launch opts, cloud/enqueue/schedule bodies
  assert.match(panel, /allowNotReady \}/);                     // ds.train opts
  assert.match(panel, /allowNotReady \? \{ allow_not_ready: true \} : \{\}/); // cloud + queue bodies
  assert.match(hook, /allow_not_ready: !!opts\.allowNotReady/);
});
