// ⏭ Continue + the batch prompt mode — the contract, read as text.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(join(here, p), 'utf8');
const studio = read('./VideoTestStudio.jsx');
const history = read('./VideoClipHistory.jsx');

test('every finished clip offers ⏭ Continue, and a continuation names its parent on the card', () => {
  assert.match(history, /clip\.status === 'done' && onContinue && \(/);
  assert.match(history, /Use the last frame as the next start frame/);
  assert.match(history, /clip\.nr_of \|\| clip\.vfi_of \|\| clip\.continues_of/);
  assert.match(history, /'continues'/);
});

test('the studio stages the last frame, marks the launch, and says what the render will be', () => {
  assert.match(studio, /postJson\(clipLastFrameUrl\(clip\.id\), \{\}\)/);
  assert.match(studio, /key: `continue:\$\{clip\.id\}`/);
  assert.match(studio, /continues: clip\.id/);
  assert.match(studio, /setMode\('i2v'\)/, 'a continuation is image-to-video by definition');
  assert.match(studio, /onContinue=\{continueFrom\}/);
  assert.match(studio, /the render lands joined behind it/);
});

test('the batch prompt pair appears with two frames, and per-picture writing happens before queueing', () => {
  assert.match(studio, /data-testid="video-prompt-mode"/);
  assert.match(studio, /mode === 'i2v' && sources\.length > 1 && \(/);
  assert.match(studio, /role="radiogroup" aria-label="Prompt for the batch"/);
  // The writer is now a resolver over ONE batched reply (see
  // videoMotionLength.contract): the loop and its fallbacks are unchanged,
  // only WHERE the writing happens moved — one vision window, not N.
  assert.match(studio, /await writePromptsFor\(launches, prompt\)/);
  assert.match(studio, /await perImagePrompts\(launches, prompt,/);
  assert.match(studio, /enhance: enhanceOn && !perPicture/, 'no second rewrite of a prompt written per picture');
  assert.match(studio, /min-h-10 rounded-md px-2 py-1 text-xs font-semibold lg:min-h-0/, 'finger-sized segments');
});
