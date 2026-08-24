import assert from 'node:assert/strict';
import { readSource } from './support/readSource.mjs'
import test from 'node:test';

const read = readSource
const singleHook = read('src/hooks/useLoraTestStudio.js');
const comparisonHook = read('src/hooks/useStudioRun.js');
const singleStudio = read('src/components/dataset/studio/RunSetupPanel.jsx');
const comparisonStudio = read('src/components/dataset/studio/ComparisonStudio.jsx');

test('both Studio hooks confirm a restarted ComfyUI, then refresh status for Resume', () => {
  assert.match(singleHook, /lora-test\/confirm-comfyui-restart/);
  assert.match(comparisonHook, /run\/\$\{runId\}\/confirm-comfyui-restart/);
  for (const source of [singleHook, comparisonHook]) {
    assert.match(source, /confirmed_comfyui_restart: true/,
      'the server must receive explicit restart authority');
    assert.match(source, /const confirmComfyuiRestart = useCallback/,
      'the explicit recovery action must be exposed from the hook');
    assert.match(source, /await refresh\(\);/,
      'successful confirmation must refresh the payload so Resume appears');
  }
});

test('the restart-control is visible only for the backend’s unknown-submit recovery state', () => {
  for (const source of [singleStudio, comparisonStudio]) {
    assert.match(source, /comfyui_recovery\?\.requires_comfyui_restart_confirmation/,
      'the control must be gated by the backend recovery contract');
    assert.match(source, /J’ai redémarré ComfyUI/,
      'the button must make the restart claim explicit');
    assert.match(source, /confirmComfyuiRestart/,
      'the explicit gesture must invoke the confirmation hook');
  }
});
