// ↗ Smooth asks for the rate before it runs — the contract, read as text.
//
// node --test renders nothing, so this pins the wiring the way the other
// studio contracts do: the button opens the window (it no longer posts), the
// window posts the factor the user picked, the choices are whole factors of
// the source, and the window carries the probe markers and the help topic.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), 'utf8');
const studio = read('./VideoTestStudio.jsx');
const history = read('./VideoClipHistory.jsx');
const dialog = read('./SmoothDialog.jsx');
const api = read('./videoStudioApi.js');

test('the Smooth button opens the window; the window posts the factor', () => {
  assert.match(studio, /onVfi=\{setVfiClip\}/, 'the card hands the clip to the window, it posts nothing');
  assert.match(studio, /postJson\(clipVfiUrl\(clip\.id\), \{ multiplier \}\)/, 'the factor travels in the body');
  assert.doesNotMatch(studio, /postJson\(clipVfiUrl\(clip\.id\), \{\}\)/, 'no more silent ×2');
  assert.match(studio, /\{vfiClip && \(\s*<SmoothDialog clip=\{vfiClip\}/);
  assert.match(history, /pick the rate \(×2, ×3 or ×4 of its own\)/, 'the tooltip no longer promises 48');
});

test('the choices are whole factors of the source rate, nothing else', () => {
  assert.match(api, /export const SMOOTH_MULTIPLIERS = \[2, 3, 4\];/);
  assert.match(dialog, /smoothTargets\(clip\)/);
  assert.match(dialog, /role="radiogroup" aria-label="Playback rate"/, 'a segmented control: 2-5 choices');
  assert.match(dialog, /Whole factors only/);
});

test('the window is measured by the probe, reachable from help, finger-sized on a phone', () => {
  assert.match(dialog, /data-probe-chrome="smooth-dialog" data-probe-layer/);
  assert.match(dialog, /<HelpBadge topic="video-smooth-rate" \/>/);
  assert.match(dialog, /createPortal\(/, 'portalled like every Studio modal');
  assert.ok((dialog.match(/min-h-10/g) || []).length >= 3, 'the three segments and the two buttons');
  assert.match(dialog, /e\.key === 'Escape'/, 'Escape closes it');
});
