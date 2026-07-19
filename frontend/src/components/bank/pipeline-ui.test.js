import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const dialog = fs.readFileSync(new URL('./LaunchAllDialog.jsx', import.meta.url), 'utf8');
const report = fs.readFileSync(new URL('./PipelineReport.jsx', import.meta.url), 'utf8');
const ws = fs.readFileSync(new URL('./BankWorkspace.jsx', import.meta.url), 'utf8');

test('the launch dialog posts the three config keys the backend expects', () => {
  assert.match(dialog, /steps:\s*\[\.\.\.steps\]/);
  assert.match(dialog, /reject_flags:\s*autoRejectOn\s*\?\s*\[\.\.\.rejectFlags\]\s*:\s*\[\]/);
  assert.match(dialog, /resolve_dups:\s*autoRejectOn\s*&&\s*resolveDups/);
});

test('captioning is OFF by default; auto-reject defaults to blur+uniform and keep-best dedup', () => {
  // Default checked set never includes caption.
  const m = dialog.match(/useState\(\(\)\s*=>\s*new Set\(\s*\[([^\]]*)\]/);
  assert.ok(m, 'found the default step set');
  assert.doesNotMatch(m[1], /caption/);
  assert.match(m[1], /'scan'/);
  assert.match(m[1], /'auto_reject'/);
  assert.match(dialog, /new Set\(\['blur',\s*'uniform'\]\)/);
  assert.match(dialog, /useState\(true\)/);            // resolveDups defaults on
});

test('a heavy pass whose tool is not ready is auto-unchecked and flagged "will skip"', () => {
  assert.match(dialog, /score:\s*!!caps\?\.bank_scoring/);
  assert.match(dialog, /watermark:\s*!!visionReady/);
  assert.match(dialog, /faces:\s*!!caps\?\.face_scoring/);
  assert.match(dialog, /\.filter\(\(k\)\s*=>\s*ready\[k\]\)/);   // default set intersects readiness
  assert.match(dialog, /will skip/);
});

test('the progress bar understands the pipeline kind (step X/N + per-step chips)', () => {
  assert.match(ws, /kind === 'pipeline' \? activity\.pipeline/);
  assert.match(ws, /step \$\{\(pipe\.index \?\? 0\) \+ 1\}\/\$\{pipe\.total_steps\}/);
  assert.match(ws, /pipe\.results\.map/);
});

test('the report renders per-step status and is fed from the persisted payload field', () => {
  assert.match(report, /STATUS_STYLE/);
  assert.match(report, /skipped/);
  assert.match(report, /cancelled/);
  assert.match(report, /error/);
  // The workspace shows it only when idle, from the persisted field.
  assert.match(ws, /payload\.pipeline_report/);
  assert.match(ws, /<PipelineReport/);
});

test('launching posts to the pipeline endpoint', () => {
  assert.match(ws, /\/api\/bank\/\$\{bankId\}\/pipeline/);
});
