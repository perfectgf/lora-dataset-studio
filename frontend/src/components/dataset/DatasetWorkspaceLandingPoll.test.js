import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Regression guard for the "view re-centers on the reference photo every ~5 s
// during a generation" bug.
//
// Mechanism: while a batch is pending, useDataset polls the dataset every 4 s and
// commits a freshly parsed JSON object (new identity each time). The workspace's
// panel-landing effect scrolls the active panel's anchor into view. If that effect
// depends on the raw `d` object, every poll re-runs it and re-fires scrollIntoView,
// yanking the user back to `ds-add-reference` (or `ds-add-generate`) whenever a
// `?panel=` is active — the exact reported symptom. Keying the effect on the stable
// dataset id (which does not change between polls) makes it land once per real
// navigation instead of once per poll.

const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
const datasetRefresh = readFileSync(new URL('../../utils/datasetRefresh.js', import.meta.url), 'utf8');

// Isolate the panel-landing effect: the one that smooth-scrolls the target anchor.
function landingEffect(src) {
  const anchor = src.indexOf("target.scrollIntoView({ behavior: 'smooth', block: 'start' })");
  assert.notEqual(anchor, -1, 'landing effect scrollIntoView not found');
  const start = src.lastIndexOf('useEffect(', anchor);
  // End at the effect's dependency-array close (`]);`), which is the first such
  // token after the scroll call — not the mid-body `focus({ ... });`.
  const depsClose = src.indexOf(']);', anchor);
  assert.ok(start !== -1 && depsClose !== -1 && depsClose > start, 'could not bound the landing effect');
  return src.slice(start, depsClose + 3);
}

test('the poll hands back a fresh dataset object every 4 s (why id-keying matters)', () => {
  // 4 s generation poll.
  assert.match(hook, /setInterval\(\(\) => refresh\(currentId\), 4000\)/);
  // Each refresh commits a newly parsed payload → a new object identity per poll.
  assert.match(datasetRefresh, /const payload = await response\.json\(\);[\s\S]*commitData\(payload\);/);
});

test('landing effect keys on the stable dataset id, not the polled object', () => {
  // A stable id is derived once so the effect can depend on identity, not the object.
  assert.match(workspace, /const datasetId = d\?\.id \?\? null;/);

  const effect = landingEffect(workspace);

  // Guard and dependency array both use the stable id.
  assert.match(effect, /if \(datasetId == null \|\| !panel \|\| workspaceLocation\.pending\) return undefined;/);
  assert.match(effect, /\}, \[datasetId, section, panel, workspaceLocation\.pending, landingRequest,/);

  // The buggy form — depending on the raw `d` object — must never come back: it
  // re-fires scrollIntoView on every poll.
  assert.doesNotMatch(effect, /\}, \[d, section, panel, workspaceLocation\.pending, landingRequest,/);
  // The guard must not reference the polled object either (that would drag `d` back
  // into the deps via exhaustive-deps).
  assert.doesNotMatch(effect, /if \(!d \|\|/);
});
