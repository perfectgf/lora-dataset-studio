/* 📷 The camera vocabulary exists TWICE — once in Python, once here — and this
   test is the reason that is allowed.

   The frontend cannot simply fetch it: the dial is DRAWN from the degrees, so
   the axes have to exist before any request resolves, and a picker that renders
   empty until the network answers is a worse screen than a duplicated table.
   The duplication is therefore deliberate, and its whole risk is drift.

   Drift here is invisible, which is what makes it expensive. If the two sides
   disagree about a TOKEN, the picker shows one sentence and the model receives
   another; the run succeeds, the pictures look fine, and every one of them is
   at an angle nobody asked for — filed under the label of the angle that was.
   If they disagree about an ID, the frontend sends a pose the server refuses,
   or worse, stores a label against a row rendered from something else.

   So this reads the Python source as TEXT and compares it, field by field, with
   the module the picker imports. Reading the source rather than importing it
   keeps `node --test` free of a Python runtime — the same trick
   canvasImprove.test.js uses to pin its route. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import {
  AZIMUTHS, DISTANCES, ELEVATIONS, MAX_VIEWS, REFERENCE_POSE, TRIGGER,
} from './cameraAngles.js';

const PY = path.join(process.cwd(), '..', 'backend', 'app', 'services', 'camera_angles.py');
const source = fs.readFileSync(PY, 'utf8');

/** Every `{'id': ..., 'token': ..., 'label': ...}` row of one Python tuple. */
function pythonRows(name) {
  const block = source.slice(source.indexOf(`${name} = (`));
  const end = block.indexOf('\n)');
  assert.ok(end > 0, `${name} tuple not found in camera_angles.py`);
  return [...block.slice(0, end).matchAll(
    /\{'id':\s*'([^']+)',[^}]*?'token':\s*'([^']+)',\s*'label':\s*'([^']+)'/g)]
    .map(([, id, token, label]) => ({ id, token, label }));
}

const PAIRS = [
  ['AZIMUTHS', AZIMUTHS],
  ['ELEVATIONS', ELEVATIONS],
  ['DISTANCES', DISTANCES],
];

for (const [name, js] of PAIRS) {
  test(`${name}: the two sides agree, in order`, () => {
    const py = pythonRows(name);
    assert.equal(py.length, js.length,
      `${name} has ${py.length} entries in Python and ${js.length} here`);
    py.forEach((row, i) => {
      assert.equal(js[i].id, row.id, `${name}[${i}] id`);
      // The token is the part the MODEL reads. A mismatch here is the silent
      // wrong-angle bug this whole file exists for.
      assert.equal(js[i].token, row.token, `${name}[${i}] token for '${row.id}'`);
      assert.equal(js[i].label, row.label, `${name}[${i}] label for '${row.id}'`);
    });
  });
}

test('the trigger, the ceiling and the reference pose agree', () => {
  assert.ok(source.includes(`TRIGGER = '${TRIGGER}'`),
    `camera_angles.py must define TRIGGER = '${TRIGGER}'`);
  // The ceiling is the vocabulary itself on both sides — Python spells it
  // `= POSE_COUNT`, and POSE_COUNT is pinned to 8*4*3 a few lines up.
  assert.ok(source.includes('MAX_VIEWS_PER_RUN = POSE_COUNT'),
    'camera_angles.py must define MAX_VIEWS_PER_RUN = POSE_COUNT');
  assert.equal(MAX_VIEWS, 96, 'the JS ceiling must equal the vocabulary size');
  const [az, el, di] = REFERENCE_POSE.split('/');
  assert.ok(source.includes(`REFERENCE_POSE = ('${az}', '${el}', '${di}')`),
    `camera_angles.py must define REFERENCE_POSE = ('${az}', '${el}', '${di}')`);
});

test('the pose id separator is the same on both sides', () => {
  // pose_id builds 'azimuth/elevation/distance'; the frontend parses on '/'.
  assert.ok(source.includes("f'{azimuth}/{elevation}/{distance}'"),
    'camera_angles.pose_id must join with "/"');
});

test('the derivation kind the tile reads is the one the row stores', () => {
  // BOTH tables, one word: the gallery's and the dataset's rows must file a
  // camera view under the same kind, or one surface's badge lies about the
  // other's rows the day an image is promoted between them.
  for (const mod of ['lora_test_studio.py', 'face_dataset_service.py']) {
    const src = fs.readFileSync(
      path.join(process.cwd(), '..', 'backend', 'app', 'services', mod), 'utf8');
    assert.ok(src.includes("CAMERA_ANGLE = 'camera_angle'"),
      `${mod} must store derivation_kind = camera_angle`);
  }
});

test('the caption phrase never leaks a prompt token', () => {
  // The phrase goes into TRAINING captions; `<sks>` or the LoRA's shot tokens
  // there would teach the trigger to expect them.
  const py = fs.readFileSync(
    path.join(process.cwd(), '..', 'backend', 'app', 'services', 'camera_angles.py'),
    'utf8');
  const block = py.slice(py.indexOf('def pose_caption_phrase'),
    py.indexOf('def pose_label'));
  assert.ok(block.length > 100, 'pose_caption_phrase not found');
  assert.ok(!block.includes('<sks>'), 'the caption phrase must not carry the trigger');
  assert.ok(!/'[^']*shot[^']*'/.test(block.replace(/#[^\n]*/g, '')),
    'the caption phrase must not reuse the LoRA\'s "shot" tokens');
});
