/* Contract test for wiring that is invisible in a screenshot.
 *
 * The fence surfaces look IDENTICAL whether or not they are guarded — the
 * notice only ever appears when another tool happens to be holding the model,
 * which is exactly the state nobody reproduces while refactoring a button. A
 * rewrite that dropped `runGuarded` and went back to a plain toast would
 * therefore pass every visual check and silently restore the dead end.
 *
 * So the wiring is asserted from the source text. `node --test` cannot parse
 * JSX, hence reading the files rather than rendering them.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), 'utf8');

const enhance = read('EnhancePromptButton.jsx');
const describe = read('DescribeImageModal.jsx');
const hook = read('../../../hooks/useOllamaFence.js');

test('✨ Enhance runs through the fence guard and can show the way out', () => {
  assert.match(enhance, /useOllamaFence/);
  assert.match(enhance, /runGuarded\(enhance\)/);
  assert.match(enhance, /<OllamaFenceNotice[\s\S]*onUnload=\{unloadAndRetry\}/);
  assert.match(enhance, /onStop=\{stopWaiting\}/);
});

test('🔎 Describe replays the SAME file instead of asking for it again', () => {
  assert.match(describe, /runGuarded\(\(run\) => send\(file, run\)\)/);
  // And not after the window was closed: the modal stays mounted while
  // closed, so the vigil is stopped by hand when `open` goes false.
  assert.match(describe, /useEffect\(\(\) => \{ if \(!open\) stopWaiting\(\); \}, \[open, stopWaiting\]\)/);
  // The raw-Response path has to carry the code by hand — apiFetch is not
  // involved here, so nothing else would recognise the refusal.
  assert.match(describe, /body\.code === OLLAMA_FENCE_CODE/);
  assert.match(describe, /fenced\.code = OLLAMA_FENCE_CODE/);
  assert.match(describe, /<OllamaFenceNotice/);
});

test('the guard polls the fence state and resumes without a click', () => {
  assert.match(hook, /'\/api\/system\/ollama-fence'/);
  assert.match(hook, /background: true/);
  // The poll's whole purpose: a free runner replays the action, no click.
  // The replay carries the vigil it was started under, so a click made
  // while it runs supersedes it (RUN in tests/ollama-fence-hook-replay.test.mjs).
  assert.match(hook, /if \(free\) \{\s*\n\s*stopTimer\(\);\s*\n\s*const mine = vigilRef\.current;/);
  assert.match(hook, /await replay\(mine\)/);
  assert.match(hook, /AUTO_RETRY_CAP_MS/);
});

test('a model freed then immediately taken again puts the vigil back on watch', () => {
  // replay() reports whether the fence took it again; both callers must
  // reschedule, or the notice would say "waiting" with nothing watching.
  const reschedules = hook.match(
    /if \(await replay\(mine\) && aliveRef\.current && mine === vigilRef\.current\)/g) || [];
  assert.equal(reschedules.length, 2);
});

test('a replay that fails for another reason is never swallowed', () => {
  // The first attempt fails inside the caller's catch; the replay does not.
  assert.match(hook, /onErrorRef\.current\?\.\(e\)/);
  assert.match(enhance, /onError: \(e\) => toast\.error/);
});

test('the eviction is never sent without the explicit consent flag', () => {
  assert.match(hook, /'\/api\/system\/ollama-fence\/unload', \{ confirmed_unload_external: true \}/);
  // And it is reachable only from unloadAndRetry — never from the poll, the
  // replay, or a catch block.
  const unloadCalls = hook.match(/ollama-fence\/unload/g) || [];
  assert.equal(unloadCalls.length, 1);
});

test('every action is handed the run handle, and every surface asks it before writing', () => {
  // The guard cannot stop a request in flight; the reply comes back to the
  // action, which writes it. So the action is run WITH the handle, on both
  // paths (the click and the replay), and each surface that writes into a
  // field asks `keepAnswer(run, …)` first — the handle is RUN in
  // tests/ollama-fence-hook-replay.test.mjs, the helper in
  // src/utils/ollamaFence.test.js; this pins that the guard sits on its own
  // line between the reply and the write (a commented-out one does not).
  assert.match(hook, /await action\(runOf\(mine\)\)/);
  assert.match(hook, /await action\(runOf\(vigil\)\)/);
  assert.match(enhance, /const enhance = async \(run\) =>[\s\S]*?\n[ \t]*if \(!keepAnswer\(run, \(\) => toast\.info\(SUPERSEDED_ANSWER_NOTICE\)\)\) return;[\s\S]*?onResult\(d\.prompt\)/);
  assert.match(describe, /async function send\(file, run\)[\s\S]*?\n[ \t]*if \(!keepAnswer\(run\)\) return;[\s\S]*?onResult\(body\.prompt\)/);
  const bar = read('../../bank/DescribeFilterBar.jsx');
  assert.match(bar, /runGuarded\(async \(run\) =>[\s\S]*?\n[ \t]*if \(!keepAnswer\(run\)\) return\n\s*setRes\(out\)/);
});
