import test from 'node:test';
import assert from 'node:assert/strict';
import { failureView, GENERIC_CAUSES, MODULE_CAUSES } from './trainingFailure.js';

const EXCERPT = { kind: 'traceback', text: 'GatedRepoError: 401', headline: 'GatedRepoError: 401' };

const GATED_401 = {
  status: 401,
  repo: 'krea/Krea-2-Turbo',
  url: 'https://huggingface.co/krea/Krea-2-Turbo',
  title: 'Hugging Face saw no valid token — this is not a licence problem',
  message: 'The download of krea/Krea-2-Turbo was refused as NOT AUTHENTICATED (HTTP 401).',
};

test('a gated-repo verdict is surfaced as the cause', () => {
  const view = failureView({ rc: 1, excerpt: EXCERPT, hf_gated: GATED_401 });
  assert.equal(view.hfGated.status, 401);
  assert.ok(view.hfGated.message.includes('NOT AUTHENTICATED'));
});

test('a proven cause replaces the generic guesswork list', () => {
  // Reported by SurpassHR (GitHub): the generic list says "the base model needs
  // a Hugging Face token" for EVERY crash, which is noise once we know exactly.
  const withGate = failureView({ rc: 1, excerpt: EXCERPT, hf_gated: GATED_401 });
  assert.equal(withGate.causes, '');
  assert.equal(withGate.checkToolkit, false);
  const without = failureView({ rc: 1, excerpt: EXCERPT });
  assert.equal(without.causes, GENERIC_CAUSES);
  assert.equal(without.hfGated, null);
});

test('a payload predating the verdict still renders (no crash, no invention)', () => {
  const view = failureView({ rc: 1, log_tail: 'boom' });
  assert.equal(view.hfGated, null);
  assert.equal(view.causes, GENERIC_CAUSES);
});

test('an empty gated payload is ignored rather than shown blank', () => {
  const view = failureView({ rc: 1, excerpt: EXCERPT, hf_gated: { status: 401 } });
  assert.equal(view.hfGated, null);
  assert.equal(view.causes, GENERIC_CAUSES);
});

// --- which Python ran ai-toolkit (GitHub #19, strouder) ------------------------

const TORCH_EXCERPT = {
  kind: 'traceback',
  text: "Traceback (most recent call last):\n    import torch\nModuleNotFoundError: No module named 'torch'",
  headline: "ModuleNotFoundError: No module named 'torch'",
};

const INTERPRETER = {
  python: '~\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe',
  module: 'torch',
  windows_store: true,
  alternative: '',
  title: 'The Python configured for ai-toolkit cannot import torch',
  message: 'ai-toolkit is set to run with ~\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe, and that interpreter cannot `import torch`.',
};

test('the interpreter verdict is surfaced, path and all', () => {
  const view = failureView({ rc: 1, excerpt: TORCH_EXCERPT, interpreter: INTERPRETER });
  assert.ok(view.interpreter.python.includes('WindowsApps'));
  assert.equal(view.causes, '');            // a proven cause replaces the guesses
});

test('a torch failure NEVER suggests a Hugging Face token', () => {
  // The false lead of #19: it sent troubleshooting the wrong way, then made the
  // real gated-model message ambiguous when it finally appeared.
  const withVerdict = failureView({ rc: 1, excerpt: TORCH_EXCERPT, interpreter: INTERPRETER });
  assert.equal(withVerdict.causes, '');
  assert.equal(withVerdict.checkToolkit, false); // the verdict already has its own link
  // …and even without the backend verdict, the log alone is enough to drop it.
  const bare = failureView({ rc: 1, excerpt: TORCH_EXCERPT });
  assert.equal(bare.interpreter, null);
  assert.equal(bare.causes, MODULE_CAUSES);
  assert.equal(bare.checkToolkit, true);
  // Hugging Face appears only to be RULED OUT, never as somewhere to go.
  assert.ok(!/token/i.test(bare.causes));
  assert.ok(/not a model or hugging face problem/i.test(bare.causes));
});

test('a non-module crash keeps the generic list untouched', () => {
  const view = failureView({
    rc: 1,
    excerpt: { kind: 'error', text: 'RuntimeError: CUDA out of memory', headline: 'RuntimeError' },
  });
  assert.equal(view.causes, GENERIC_CAUSES);
  assert.equal(view.interpreter, null);
  assert.equal(view.checkToolkit, false);
});

test('an empty interpreter payload is ignored rather than shown blank', () => {
  const view = failureView({ rc: 1, excerpt: EXCERPT, interpreter: { python: 'x' } });
  assert.equal(view.interpreter, null);
  assert.equal(view.causes, GENERIC_CAUSES);
});
