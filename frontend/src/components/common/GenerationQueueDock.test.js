/**
 * ⏳ The queue dock's ONE control that is not about an LDS job.
 *
 * Every other button here re-orders or cancels work LDS owns. "Run anyway"
 * accepts a cost on the user's own card instead — so what this file pins is not
 * that the button exists, it is that the button cannot be pressed by accident,
 * cannot be pressed by the app, and never lies about what it does.
 *
 * `node --test` renders nothing (it parses no JSX), so these read the source as
 * text — the same shape as ContinueDialog.test.js and StopButtonWording.test.js.
 * They prove the wiring is WRITTEN; the responsive probe is what proves it fits.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const dock = fs.readFileSync(new URL('./GenerationQueueDock.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('../../utils/queuePanel.js', import.meta.url), 'utf8');

test('the offer is the server’s to make — the dock never invents one', () => {
  // The dock reads a validated action; it does not decide on its own that a
  // hold has lasted long enough to be worth sharing a card over.
  assert.match(dock, /const action = pausedAction\(listing\)/);
  assert.match(panel, /export function pausedAction/);
  assert.match(dock, /\{action && <ShareGpuOffer/);
});

test('sharing takes two clicks, and the second one is the one that means it', () => {
  assert.match(dock, /const \[confirming, setConfirming\] = useState\(false\)/);
  // First click reveals the cost...
  assert.match(dock, /onClick=\{\(\) => setConfirming\(true\)\}/);
  // ...the second sends it, and "Keep waiting" is offered as a real answer.
  assert.match(dock, /onClick=\{\(\) => \{ setConfirming\(false\); onShare\?\.\(\) \}\}/);
  assert.match(dock, /Keep waiting/);
});

test('the consent flag travels with the request, and nothing else calls it', () => {
  assert.match(dock, /postJson\('\/api\/system\/ollama-fence\/share',\s*\n?\s*\{ confirmed_share_gpu: true \}\)/);
  // One call site only: not a retry, not a fallback, not a poll.
  assert.equal((dock.match(/ollama-fence\/share/g) || []).length, 1);
});

test('a failed share says so instead of leaving the queue looking answered', () => {
  assert.match(dock, /toast\.error\(e\?\.message \|\| 'The GPU could not be shared\.'\)/);
  // And the panel re-polls either way: the hold is the server's to clear.
  assert.match(dock, /finally \{\s*\n\s*if \(aliveRef\.current\) setSharing\(false\)\s*\n\s*await poll\(\)/);
});

test('the share buttons stay finger-sized on a phone', () => {
  // 40px below lg, unchanged on a desktop — the probe's target rule, applied at
  // the source rather than exempted.
  assert.match(dock, /min-h-10[^'`]*lg:min-h-0/);
});
