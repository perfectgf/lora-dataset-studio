import test from 'node:test'
import assert from 'node:assert/strict'
import {
  isEnginePart, shortLoraName, checkpointStep, groupTrained, splitDeployed,
} from './videoLoraGroups.js'

test('engine parts are recognised by what the graph grafts, candidates are not', () => {
  for (const f of ['h3/minimax_h3_fl2v_lightx2v_turbo_4step_v1.0.safetensors',
    'h3/camera_motion_h3_lora_v1_3000_pruned_h3keys.safetensors',
    'h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors']) {
    assert.equal(isEnginePart(f), true, f)
  }
  for (const f of ['h3/lds/lds174_video_Harbour___stills.safetensors',
    'h3/MiniMax_H3_Combat_LoRA.safetensors', 'h3/breastplayjiggle_h3_v1.safetensors']) {
    assert.equal(isEnginePart(f), false, f)
  }
})

test('a checkpoint filename reads as a name, a run and a step', () => {
  assert.equal(shortLoraName('h3/lds/lds174_video_Harbour___stills_000001000.safetensors'), 'Harbour — stills')
  assert.equal(shortLoraName('lds171_video_absmoke-b.safetensors'), 'absmoke-b')
  assert.equal(shortLoraName('MiniMax_H3_Combat_LoRA.safetensors'), 'MiniMax H3 Combat LoRA')
  assert.equal(checkpointStep('lds174_video_Harbour___stills_000001000.safetensors'), 1000)
  assert.equal(checkpointStep('lds174_video_Harbour___stills.safetensors'), null)
})

test('trained checkpoints group into one row per run, final first, newest run first', () => {
  const groups = groupTrained([
    { run_id: 171, dataset_id: 3, filename: 'lds171_video_absmoke-b_000000050.safetensors', label: 'x' },
    { run_id: 174, dataset_id: 1, filename: 'lds174_video_Harbour___stills_000001000.safetensors', label: 'a', deployed_as: null },
    { run_id: 174, dataset_id: 1, filename: 'lds174_video_Harbour___stills.safetensors', label: 'b', deployed_as: 'h3/lds/lds174_video_Harbour___stills.safetensors' },
    { run_id: 171, dataset_id: 3, filename: 'lds171_video_absmoke-b.safetensors', label: 'y' },
  ])
  assert.deepEqual(groups.map((g) => g.run_id), [174, 171])
  assert.equal(groups[0].name, 'Harbour — stills')
  assert.deepEqual(groups[0].checkpoints.map((c) => [c.final, c.step]), [[true, null], [false, 1000]])
  assert.equal(groups[0].checkpoints[0].deployed_as, 'h3/lds/lds174_video_Harbour___stills.safetensors')
  assert.deepEqual(groups[1].checkpoints.map((c) => c.step), [null, 50])
})

test('the folder list splits into candidates and engine parts, nothing dropped', () => {
  const { candidates, parts } = splitDeployed([
    { filename: 'h3/lds174_video_Harbour___stills.safetensors', label: 'j' },
    { filename: 'h3/minimax_h3_fl2v_lightx2v_turbo_8step.safetensors', label: 't' },
    { filename: 'h3/VBVR_H3_attn_only.safetensors', label: 'v' },
  ])
  assert.deepEqual(candidates.map((d) => d.label), ['j', 'v'])
  assert.deepEqual(parts.map((d) => d.label), ['t'])
})
