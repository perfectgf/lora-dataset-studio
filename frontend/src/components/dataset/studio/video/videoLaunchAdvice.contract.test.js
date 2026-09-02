// ⏱ The launch advice reaches the screen, and says what the SERVER decided.
//
// `video_test_studio.launch_advice` answers only when it can tell — the flag is
// missing from the running ComfyUI's argv, that ComfyUI is new enough to know
// it, and the machine's RAM is under the floor. The studio must render THAT
// answer through `launchAdviceLines` (tested on its own in videoStudioApi.test.js),
// never a flag or a RAM figure of its own: a notice that spelled `--fast-disk`
// would go stale the day the server names a second flag, and one that
// invented the RAM would contradict the machine ComfyUI runs on.
//
// This reads the JSX as text (node --test renders nothing), so it proves the
// wiring exists, never that the box looks right.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, 'VideoTestStudio.jsx'), 'utf8')
  .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')   // JSX comments say what the code must do, not do it
  .replace(/\/\/[^\n]*/g, '');

test('the studio phrases the advice through the shared helper and gates the notice on it', () => {
  assert.match(src, /const launchAdvice = launchAdviceLines\(options\?\.launch_advice\)/,
    'the sentences come from launchAdviceLines, fed the server payload');
  assert.match(src, /\{launchAdvice && \(/, 'no advice from the server, no notice');
  assert.match(src, /data-testid="video-launch-advice"/);
});

test('the notice prints the helper sentences and the payload figures, never its own', () => {
  const notice = src.slice(src.indexOf('data-testid="video-launch-advice"'));
  const block = notice.slice(0, notice.indexOf('</div>'));
  assert.match(block, /\{launchAdvice\.title\}/, 'the headline is the helper\'s');
  assert.match(block, /\{launchAdvice\.action\}/, 'and so is the change to make');
  assert.match(block, /options\.launch_advice\.ram_total_gb/, 'the RAM figure comes from the server');
  assert.match(block, /options\.launch_advice\.weights_gb/, 'and the weight of the set');
  assert.match(block, /the machine running ComfyUI has/, 'the RAM is ComfyUI\'s host, which may not be this machine');
  assert.doesNotMatch(block, /--fast-disk|--disable-dynamic-vram/, 'no flag spelled out in the JSX');
  assert.doesNotMatch(block, /\b4[0-9] GB\b/, 'no RAM figure spelled out in the JSX');
  assert.doesNotMatch(block, /Setup screen/, 'no pointer to a Start button that is hidden while ComfyUI runs');
});
