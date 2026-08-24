import test from 'node:test';
import assert from 'node:assert/strict';
import {
  countUnclassified, classifyBlockedReason, classifyFramingState, classifyResultMessage,
} from './classifyFramingGate.js';

const READY = { installed: true, reachable: true, vision_model_ready: true };

test('counts exactly what the server pass acts on: has a file AND no framing', () => {
  const images = [
    { id: 1, source: 'import', framing: null, filename: 'a.png' },
    { id: 2, source: 'import', framing: null, status: 'reject', filename: 'b.png' },
    { id: 3, source: 'import', framing: 'body', filename: 'c.png' },
    { id: 4, source: 'generated', framing: null, filename: 'd.png' },  // cropped generated
    { id: 5, source: 'import', framing: '', filename: 'e.png' },
    { id: 6, source: 'generated', framing: null },                    // no file yet
  ];
  assert.equal(countUnclassified(images), 4);
  assert.equal(countUnclassified([]), 0);
  assert.equal(countUnclassified(undefined), 0);
});

test('a dataset with nothing to classify shows NO button (empty, or fully classified)', () => {
  assert.equal(classifyFramingState({ images: [], ollama: READY }).visible, false);
  assert.equal(classifyFramingState({
    images: [
      { source: 'import', framing: 'face', filename: 'a.png' },
      { source: 'generated', framing: 'bust', filename: 'b.png' },
    ],
    ollama: READY,
  }).visible, false);
});

test('a cropped generated shot with no framing brings the button back', () => {
  const s = classifyFramingState({
    images: [
      { source: 'generated', framing: 'body', filename: 'keep.png' },
      { source: 'generated', framing: null, filename: 'cropped.png' },
    ],
    ollama: READY,
  });
  assert.equal(s.visible, true);
  assert.equal(s.count, 1);
  assert.match(s.label, /Classify framing \(1\)/);
});

test('the label carries the count — a bare verb does not say what it will do', () => {
  const s = classifyFramingState({
    images: [
      { source: 'import', framing: null, filename: 'a.png' },
      { source: 'import', framing: null, filename: 'b.png' },
    ],
    ollama: READY,
  });
  assert.equal(s.visible, true);
  assert.equal(s.count, 2);
  assert.match(s.label, /Classify framing \(2\)/);
  assert.equal(s.disabled, false);
  assert.equal(s.blocked, false);
});

test('each Ollama state names its OWN fix (never a dead button)', () => {
  assert.match(classifyBlockedReason({ installed: false, reachable: false }), /not installed/i);
  assert.match(classifyBlockedReason({ installed: true, reachable: false }), /not running/i);
  assert.match(classifyBlockedReason({ installed: true, reachable: true, vision_model_ready: false }),
    /vision model is not pulled/i);
  assert.equal(classifyBlockedReason(READY), null);
  // A REMOTE Ollama (url pointing elsewhere) is reachable without a local binary.
  assert.equal(classifyBlockedReason({ installed: false, reachable: true, vision_model_ready: true }), null);
});

test('a blocked pass is disabled AND the reason is the tooltip', () => {
  const s = classifyFramingState({
    images: [{ source: 'import', framing: null, filename: 'a.png' }], ollama: {},
  });
  assert.equal(s.blocked, true);
  assert.equal(s.disabled, true);
  assert.equal(s.title, s.blockedReason);
});

test('capabilities not loaded yet is not "missing" — no false accusation', () => {
  const s = classifyFramingState({
    images: [{ source: 'import', framing: null, filename: 'a.png' }], ollama: {}, capsLoading: true,
  });
  assert.equal(s.blocked, false);
  assert.equal(s.blockedReason, null);
  assert.equal(s.disabled, true);   // idle until we actually know
});

test('a running pass shows its persistent progress and stays on screen', () => {
  const s = classifyFramingState({
    images: [{ source: 'import', framing: null, filename: 'a.png' }], ollama: READY,
    activity: { kind: 'classify', done: 12, total: 1222 },
  });
  assert.equal(s.running, true);
  assert.equal(s.visible, true);
  assert.equal(s.disabled, true);
  assert.match(s.label, /12\/1222/);
});

test('another dataset activity does not masquerade as a framing pass', () => {
  const s = classifyFramingState({
    images: [{ source: 'import', framing: null, filename: 'a.png' }], ollama: READY,
    activity: { kind: 'caption', done: 3, total: 9 },
  });
  assert.equal(s.running, false);
  assert.equal(s.disabled, false);
});

test('0 classified while there was work is reported as a failure, not a success', () => {
  const m = classifyResultMessage(0, 1222);
  assert.equal(m.tone, 'error');
  assert.match(m.text, /Ollama/);
  // Partial pass: honest count, and it says a retry can pick up the rest.
  assert.equal(classifyResultMessage(900, 1222).tone, 'info');
  assert.match(classifyResultMessage(900, 1222).text, /900\/1222/);
  // Full pass.
  assert.equal(classifyResultMessage(1222, 1222).tone, 'success');
  // Nothing asked, nothing done → not an error.
  assert.equal(classifyResultMessage(0, 0).tone, 'success');
  // Server queued nothing — not Ollama.
  assert.equal(classifyResultMessage(0, 12, { attempted: 0 }).tone, 'success');
  // Looked at the files, none were on disk.
  const disk = classifyResultMessage(0, 12, { attempted: 12, unanswered: 0 });
  assert.equal(disk.tone, 'error');
  assert.match(disk.text, /not found on disk/i);
});
