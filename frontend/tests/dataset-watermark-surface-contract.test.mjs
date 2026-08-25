/**
 * The three curation affordances this wave added, pinned on the source.
 *
 * `node --test` cannot mount this JSX, and every one of these is a one-line
 * regression that no unit test would notice:
 *   - the bulk-reject button silently switching its count back to
 *     `watermarkDetected` (which is the "5 930 announced, 0 rejected" defect);
 *   - the ⏹ Stop disappearing from the watermark banner;
 *   - the poller that makes that Stop visible at all being dropped as redundant
 *     with `hasActivity` — it is not: `hasActivity` only starts once a refresh
 *     has already seen an activity, and the scan's own call does not refresh
 *     until it ends.
 */
import test from 'node:test';
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict';

const read = readSource
const workspace = read('src/components/dataset/DatasetWorkspace.jsx');
const hook = read('src/hooks/useDataset.js');
const settings = read('src/components/settings/CaptioningSection.jsx');

test('the bulk-reject button counts what it will really reject', () => {
  assert.match(workspace, /id="ds-curation-reject-flagged"/);
  // Rendered on, labelled with and acting on the SAME derived number.
  assert.match(workspace, /\{flagged\.rejectable > 0 && \(/);
  assert.match(workspace, /Reject all flagged \(\{flagged\.rejectable\}\)/);
  assert.match(workspace, /ds\.batchImages\(flagged\.rejectableIds, 'reject'\)/);
  assert.doesNotMatch(workspace, /batchImages\(\s*images\.filter/);
});

test('it confirms first, and reports the SERVER\'s number with the way back', () => {
  assert.match(workspace, /window\.confirm\(rejectFlaggedConfirmText\(flagged\)\)/);
  assert.match(workspace, /const affected = await ds\.batchImages/);
  assert.match(workspace, /Show ▸ Rejected in the grid, then ✓ Keep/);
});

test('the flagged pile is summarised once, by the shared pure module', () => {
  assert.match(workspace, /from '\.\/watermarkFlagged\.js'/);
  assert.match(workspace, /const flagged = summarizeFlagged\(images\);/);
  assert.match(workspace, /watermarkRejectable: rejectableFlagged\(navImages\)\.length,/);
});

test('a running watermark scan can be stopped from its banner', () => {
  assert.match(workspace, /act\?\.kind === 'watermark_detect' && \(/);
  assert.match(workspace, /onClick=\{ds\.cancelWatermarkScan\} disabled=\{!!act\?\.cancelling\}/);
  assert.match(hook, /watermarks\/detect\/cancel/);
});

test('the scan is polled while it runs, or the Stop would never appear', () => {
  assert.match(hook, /if \(!localActivityRuns\.has\(`watermark:\$\{currentId\}`\) \|\| !currentId\) return undefined;/);
  assert.match(hook, /\}, \[localActivityRuns, currentId, refresh\]\);/);
});

test('the dismissed pile is re-judgeable, or changing engine changes nothing', () => {
  assert.match(workspace, /ds\.findWatermarks\(\{ includeDismissed: true \}\)/);
  assert.match(hook, /include_dismissed: true/);
});

test('a fallback is spoken, never silent', () => {
  assert.match(hook, /if \(d\.backend_note\) toast\.info\(d\.backend_note\);/);
  assert.match(hook, /d\.unlocated/);
});

test('the engine choice is a setting, in the captioning-backend shape', () => {
  assert.match(settings, /const WATERMARK_BACKEND_OPTIONS = \[/);
  for (const id of ['auto', 'detector', 'vision']) {
    assert.match(settings, new RegExp(`id: '${id}'`));
  }
  assert.match(settings, /id="wmdet-backend"/);
  assert.match(settings, /setField\('watermark_detect', 'backend', e\.target\.value\)/);
});
