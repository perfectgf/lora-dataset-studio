import test from 'node:test';
import assert from 'node:assert/strict';

import {
  previewError, previewRunning, previewStartLabel, previewStatusLabel,
  previewStoppedNotice,
} from './faceMaskProgress.js';

/* Three ways a face-mask pass can end, and the panel has to tell them apart —
   because only one of them is the user's doing and only one of them means the
   pass cannot work.

     finished     — the preview is drawn;
     stopped      — the user clicked Stop, and what was analyzed is kept;
     interrupted  — nobody asked: the pass ran out of its time budget, or the
                    child died. It was WORKING. Wording it as a failure would
                    tell a user their setup is broken when the honest answer is
                    "it needed longer, and here is where to carry on from".

   The third state is new, and it is the one an incident produced: a 138-image
   set was killed by a fixed 900 s watchdog and reported as a failure that threw
   the whole pass away. */

const interrupted = (note) => ({
  phase: 'detecting', done: 97, total: 138, error: null, finished: true,
  stopping: false, stopped: false, interrupted: true, note,
});

test('an interrupted pass is not worded as a failure', () => {
  const job = interrupted('Face detection ran out of its 3660s budget for 138 images.');
  assert.equal(previewStatusLabel(job), 'Interrupted before the end.');
  // No red alert line: `error` is what the panel renders as "⚠️ something is
  // wrong", and nothing is wrong with a pass that simply needed more time.
  assert.equal(previewError(job), '');
  assert.equal(previewRunning(job), false);
});

test('the interruption notice says what happened and what was kept', () => {
  const note = 'Face detection ran out of its 3660s budget for 138 images.';
  const line = previewStoppedNotice(interrupted(note), { done: 97, total: 138 });
  assert.ok(line.includes(note), 'the reason must reach the user, not just the log');
  assert.ok(line.includes('97 of 138'), 'the credit is the reason to start again');
  assert.ok(/continues from there/.test(line));
});

test('an interruption before any image says so instead of promising a resume', () => {
  const line = previewStoppedNotice(interrupted('It gave up during the model load.'), null);
  assert.ok(/nothing to keep/.test(line));
  assert.ok(!/of undefined/.test(line));
});

test('the start button offers the resume the interruption banked', () => {
  assert.equal(previewStartLabel({ done: 97, total: 138 }, false),
               '▶ Resume — 97 of 138 already analyzed');
});

test('a plain stop keeps its own wording', () => {
  const job = { phase: 'detecting', done: 2, total: 5, error: null, finished: true,
                stopping: false, stopped: true, interrupted: false, note: null };
  assert.equal(previewStatusLabel(job), 'Stopped.');
  assert.match(previewStoppedNotice(job, { done: 2, total: 5 }), /^Stopped\./);
});

test('a real failure still reads as one', () => {
  const job = { phase: 'starting', done: 0, total: 5, finished: true,
                error: "No module named 'insightface'", stopping: false,
                stopped: false, interrupted: false, note: null };
  assert.equal(previewStatusLabel(job), "No module named 'insightface'");
  assert.equal(previewStoppedNotice(job, null), '');
});
