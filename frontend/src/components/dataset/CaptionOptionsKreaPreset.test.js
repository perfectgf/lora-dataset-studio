import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const popover = readFileSync(new URL('./CaptionOptionsPopover.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');

test('the Krea 2 Ollama suggestion is gated by the dataset train type', () => {
  assert.match(
    workspace,
    /<CaptionOptionsPopover datasetId=\{d\.id\} trainType=\{d\.train_type\} kind=\{d\.kind\}/,
  );
  assert.match(
    popover,
    /CaptionOptionsPopover\(\{ datasetId, trainType, kind, onClose, onSaved \}\)/,
  );
  assert.match(popover, /\{trainType === 'krea' && \(/);
  assert.doesNotMatch(popover, /\{trainType !== 'sdxl' && \(/);
});

test('an installed Krea companion selects Ollama and the exact 4B model', () => {
  assert.match(
    popover,
    /export const KREA_2_OLLAMA_MODEL = 'qwen3-vl:4b-instruct';/,
  );
  assert.match(
    popover,
    /const kreaModelInstalled = models\.includes\(KREA_2_OLLAMA_MODEL\);/,
  );
  assert.match(
    popover,
    /if \(kreaModelInstalled\) \{\s*setBackend\('ollama'\);\s*setOllamaModel\(KREA_2_OLLAMA_MODEL\);\s*return;/,
  );
});

test('an absent companion offers Pull & use and selects Ollama only after success', () => {
  assert.match(popover, /: 'Pull & use'}/);
  assert.match(popover, /startPullByName\(KREA_2_OLLAMA_MODEL, true\);/);
  assert.match(popover, /postJson\('\/api\/ollama\/pull', \{ model: name \}\)/);
  assert.match(popover, /apiFetch\('\/api\/ollama\/pull', \{ background: true \}\)/);

  const success = popover.slice(
    popover.indexOf("if (s.state === 'success')"),
    popover.indexOf("} else if (s.state === 'error')"),
  );
  assert.match(success, /setOllamaModel\(s\.model\);/);
  assert.match(success, /if \(selectOllamaBackend\) setBackend\('ollama'\);/);
});

test('pull polling is sequential and ignores stale or unmounted responses', () => {
  const polling = popover.slice(
    popover.indexOf('const poll = () =>'),
    popover.indexOf('const startPullByName ='),
  );
  const success = polling.slice(
    polling.indexOf("if (s.state === 'success')"),
    polling.indexOf("} else if (s.state === 'error')"),
  );

  assert.doesNotMatch(polling, /setInterval/);
  assert.match(polling, /const generation = \+\+pollGenerationRef\.current;/);
  assert.match(
    polling,
    /mountedRef\.current && pollGenerationRef\.current === generation/,
  );
  assert.match(polling, /s = await apiFetch\('\/api\/ollama\/pull'/);
  assert.match(
    polling,
    /if \(s\.state === 'running'\) \{[\s\S]*setTimeout\(checkPull, 1200\);[\s\S]*return;/,
  );

  assert.match(
    popover,
    /mountedRef\.current = false;[\s\S]*pollGenerationRef\.current \+= 1;[\s\S]*clearTimeout\(pollRef\.current\);/,
  );
  assert.match(
    success,
    /await refreshModels\(isCurrent\);\s*if \(!isCurrent\(\)\) return;/,
  );
});

test('a pull status GET failure becomes retryable instead of staying running', () => {
  const polling = popover.slice(
    popover.indexOf('const poll = () =>'),
    popover.indexOf('const startPullByName ='),
  );
  const failure = polling.slice(
    polling.indexOf('catch (error)'),
    polling.indexOf('if (!isCurrent()) return;', polling.indexOf('catch (error)')) + 32,
  );

  assert.match(polling, /catch \(error\) \{[\s\S]*state: 'error'/);
  assert.match(polling, /selectOllamaAfterPullRef\.current = false;/);
  assert.match(polling, /toast\.error\(message\);/);
  assert.doesNotMatch(failure, /state: 'running'/);
});

test('the arbitrary model picker and generic pull field remain available', () => {
  assert.match(
    popover,
    /const modelChoices = ollamaModel && !models\.includes\(ollamaModel\)\s*\? \[ollamaModel, \.\.\.models\] : models;/,
  );
  assert.match(popover, /\{modelChoices\.map\(\(m\) => <option key=\{m\} value=\{m\}>\{m\}<\/option>\)\}/);
  assert.match(popover, /<input value=\{pullName\} onChange=\{\(e\) => setPullName\(e\.target\.value\)\}/);
  assert.match(popover, /const startPull = \(\) => startPullByName\(pullName\);/);
  assert.match(popover, /huihui_ai\/qwen3-vl-abliterated:8b-instruct/);
});
