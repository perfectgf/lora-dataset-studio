import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const hook = readFileSync(new URL('./useDataset.js', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('../components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8');
const grid = readFileSync(new URL('../components/dataset/DatasetGrid.jsx', import.meta.url), 'utf8');
const item = readFileSync(new URL('../components/dataset/DatasetGridItem.jsx', import.meta.url), 'utf8');

function actionSource() {
  const start = hook.indexOf('const scoreFace = useCallback');
  const end = hook.indexOf('// Watermark scan', start);
  return hook.slice(start, end);
}

test('scoreFace permits only one in-flight image with a synchronous guard', () => {
  const action = actionSource();
  assert.match(hook, /const \[scoringFaceIds, setScoringFaceIds\] = useState\(\(\) => new Set\(\)\)/);
  assert.match(hook, /const scoringFaceRef = useRef\(new Set\(\)\)/);
  assert.ok(action.includes('`/api/dataset/image/${imageId}/analyze-face`'));
  assert.doesNotMatch(action, /\/analyze-faces/);
  assert.match(action, /if \(data\?\.face_scoring_blocked\)/);
  assert.match(action, /scoringFaceRef\.current\.size > 0/);
  assert.match(action, /setScoringFaceIds\(\(prev\) => \{[\s\S]*next\.add\(imageId\)/);
  assert.match(action, /if \(d\.scoring_error\)[\s\S]*await refresh\(\)/);
  assert.match(action, /finally[\s\S]*scoringFaceRef\.current\.delete\(imageId\)[\s\S]*next\.delete\(imageId\)/);
});

test('scoreFace locks all score controls while keeping the active tile busy', () => {
  assert.match(workspace, /onScoreFace=\{ds\.scoreFace\} scoringFaceIds=\{ds\.scoringFaceIds\}/);
  assert.match(grid, /onMirror, onRegenerate, onScoreFace, scoringFaceIds/);
  assert.match(grid, /onScoreFace=\{onScoreFace\} scoreFaceBusy=\{Boolean\(scoringFaceIds\?\.has\(img\.id\)\)\}\s+faceScoringBusy=\{Boolean\(scoringFaceIds\?\.size\)\}/);
  assert.match(grid, /faceScoringBlocked=\{faceScoringBlocked\}/);
  assert.match(item, /onScoreFace, scoreFaceBusy = false, faceScoringBusy = false, faceScoringBlocked = null/);
  const actionAt = item.indexOf("{url && onScoreFace && ['keep', 'pending'].includes(img.status) && (");
  const regenerateAt = item.indexOf('{canRegenerate && (', actionAt);
  assert.ok(actionAt >= 0 && actionAt < regenerateAt, 'score button sits left of regenerate');
  assert.match(item, /e\.stopPropagation\(\); onScoreFace\(img\.id\)/);
  // Scoring is a curation write: the server's vision window is fail-closed and
  // refuses in words while ComfyUI has work, so the UI no longer greys it out
  // for queued generations — only for a pass that owns the rows.
  assert.match(item, /disabled=\{curationRefused \|\| faceScoringBusy \|\| !!faceScoringBlocked \|\| scoreFaceBusy\}/);
  assert.match(item, /aria-busy=\{scoreFaceBusy\}[\s\S]*scoreFaceBusy \? 'animate-pulse' : ''/);
  assert.match(item, /title=\{scoreFaceTitle\} aria-label=\{scoreFaceTitle\}/);
});

test('the single-image action reuses the bulk scorer error language', () => {
  assert.match(hook, /export function faceScoringErrorMessage\(scoringError\)/);
  assert.match(hook, /kind === 'unavailable'/);
  assert.match(hook, /kind === 'subject_not_photographic'/);
  assert.match(hook, /kind === 'ref_unusable'/);
  const action = actionSource();
  assert.match(action, /faceScoringErrorMessage\(d\.scoring_error\)/);
  const bulkStart = hook.indexOf('const analyzeFaces = useCallback');
  const bulkEnd = hook.indexOf('// Score one image', bulkStart);
  assert.match(hook.slice(bulkStart, bulkEnd), /faceScoringErrorMessage\(d\.scoring_error\)/);
});
