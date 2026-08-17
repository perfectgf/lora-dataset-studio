import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const dialog = readFileSync(new URL('./RegionTouchupDialog.jsx', import.meta.url), 'utf8');
const brush = readFileSync(new URL('./InpaintBrushEditor.jsx', import.meta.url), 'utf8');

test('touch-up paints a mask instead of drawing boxes', () => {
  assert.match(dialog, /import InpaintBrushEditor, \{ maskPngFromCanvas \}/);
  assert.match(dialog, /<InpaintBrushEditor/);
  assert.doesNotMatch(dialog, /WatermarkRegionEditor/);
  assert.doesNotMatch(dialog, /\+ Add zone/);
  assert.match(brush, /maskPngFromCanvas/);
  assert.match(brush, /destination-out/);
  assert.match(brush, /cssBrushCursor/);
  assert.match(brush, /cursor: disabled \? 'default' : cursorCss/);
  assert.doesNotMatch(brush, /setCursor/);
  assert.match(brush, /max-h-\[100cqh\]/);
  assert.doesNotMatch(brush, /70vh/);
});

test('touch-up parks the controls beside a portrait and under a landscape', () => {
  assert.match(dialog, /decideActionPlacement/);
  assert.match(dialog, /rememberImageRatio/);
  assert.match(dialog, /readImageRatio/);
  assert.match(dialog, /placement === 'rail'/);
  assert.match(dialog, /w-\[17rem\]/);
  assert.match(dialog, /locked: applying/);
  assert.match(dialog, /onNaturalSize/);
});

test('touch-up is Klein-only and names the model', () => {
  assert.match(dialog, /import KleinModelSetting from '\.\.\/shared\/KleinModelSetting'/);
  assert.match(dialog, /<KleinModelSetting datasetId=\{datasetId\}/);
  assert.match(dialog, /caps\?\.watermark_klein !== false/);
  assert.match(dialog, /localEngineUnavailableReason\('klein', caps\)/);
  assert.doesNotMatch(dialog, /setMethod\('lama'\)/);
  assert.doesNotMatch(dialog, /watermark_inpaint/);
});

test('touch-up sends the painted mask plus a prompt, and can reset', () => {
  assert.match(dialog, /onApply\(mask, prompt\.trim\(\)\)/);
  assert.match(dialog, /maskPngFromCanvas/);
  assert.match(dialog, /↩ Reset/);
  assert.match(dialog, /onRestore\(\)/);
  assert.match(dialog, /image\?\.has_region_touchup/);
});

test('touch-up offers removal chips and does not persist regions', () => {
  assert.match(dialog, /remove necklace/);
  assert.match(dialog, /remove earrings/);
  assert.match(dialog, /remove jewelry/);
  assert.match(dialog, /remove skin blemish/);
  assert.match(dialog, /remove makeup/);
  assert.doesNotMatch(dialog, /watermark-regions/);
});
