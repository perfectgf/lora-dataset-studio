import test from 'node:test';
import assert from 'node:assert/strict';

import {
  canRestoreImproveSettings, matchPresetName, parseExtraLoras,
  restoreImproveMessage, restoreImprovePatch,
} from './improveSettingsRestore.js';

const SHIPPED = 'add detailed texture, add sharp details';
const PRESETS = [
  { name: 'Detail', loras: [{ file: 'klein/d.safetensors', strength: 0.7 }] },
  { name: 'Duo', loras: [{ file: 'klein/a.safetensors', strength: 0.4 },
    { file: 'klein/b.safetensors', strength: 0.3 }] },
];

// --- the gate ----------------------------------------------------------------

test('offered only on a Klein improve row that recorded an instruction', () => {
  assert.equal(canRestoreImproveSettings(
    { derivation_kind: 'canvas_image_improve', prompt: 'add grain' }), true);
  // A plain render is not an improve result.
  assert.equal(canRestoreImproveSettings({ derivation_kind: null, prompt: 'x' }), false);
  // A SeedVR2 restoration stored the sentinel sentence, not an instruction.
  assert.equal(canRestoreImproveSettings({ derivation_kind: 'canvas_image_improve',
    prompt: 'SeedVR2 upscale (no prompt — restoration pass)' }), false);
  // A legacy row with nothing recorded has nothing to restore.
  assert.equal(canRestoreImproveSettings({ derivation_kind: 'canvas_image_improve',
    prompt: '' }), false);
  assert.equal(canRestoreImproveSettings(null), false);
});

// --- reading the rows --------------------------------------------------------

test('extra_loras parses leniently — user-database content never crashes the button', () => {
  assert.deepEqual(parseExtraLoras('[{"filename":"klein/d.safetensors","strength":0.7}]'),
    [{ filename: 'klein/d.safetensors', strength: 0.7 }]);
  for (const junk of [null, undefined, '', '{not json', '"a string"', '[{"nope":1}]']) {
    assert.deepEqual(parseExtraLoras(junk), []);
  }
});

test('a preset matches on files, strengths AND order — order is the chain', () => {
  const rows = [{ filename: 'klein/a.safetensors', strength: 0.4 },
    { filename: 'klein/b.safetensors', strength: 0.3 }];
  assert.equal(matchPresetName(rows, PRESETS), 'Duo');
  assert.equal(matchPresetName([...rows].reverse(), PRESETS), null);
  assert.equal(matchPresetName(
    [{ filename: 'klein/a.safetensors', strength: 0.5 }], PRESETS), null);
  assert.equal(matchPresetName([], PRESETS), null);
});

// --- the patch ---------------------------------------------------------------

test('a custom instruction restores as itself, with the toggle on', () => {
  const { patch, report } = restoreImprovePatch({
    img: { prompt: 'keep it a drawing', extra_loras: null },
    shipped: SHIPPED, presets: PRESETS,
  });
  assert.deepEqual(patch, { config: {
    identity_prompts: { klein_improve: 'keep it a drawing', klein_improve_enabled: true },
    klein: { improve_lora_preset: '' },
  } });
  assert.equal(report.hadLoras, false);
  assert.match(restoreImproveMessage(report), /LoRA preset set to None/);
});

test('a prompt equal to the shipped default is stored as EMPTY — the follow-the-default contract', () => {
  const { patch, report } = restoreImprovePatch({
    img: { prompt: `  ${SHIPPED}  ` }, shipped: SHIPPED, presets: [],
  });
  assert.equal(patch.config.identity_prompts.klein_improve, '');
  assert.equal(report.followsDefault, true);
  assert.match(restoreImproveMessage(report), /built-in default/);
});

test('matched rows restore the preset BY NAME', () => {
  const { patch, report } = restoreImprovePatch({
    img: { prompt: 'add grain',
      extra_loras: '[{"filename":"klein/d.safetensors","strength":0.7}]' },
    shipped: SHIPPED, presets: PRESETS,
  });
  assert.equal(patch.config.klein.improve_lora_preset, 'Detail');
  assert.match(restoreImproveMessage(report), /“Detail”/);
});

test('a recorded profile restores every knob, and the model pin (null = auto)', () => {
  const { patch, report } = restoreImprovePatch({
    img: {
      prompt: 'add grain',
      extra_loras: '[{"filename":"klein/d.safetensors","strength":0.7}]',
      improve_profile: { engine: 'klein', klein_model: null,
        consistency_strength: 0.8, steps: 6, base_lora_strength: 0.5,
        megapixels: 4, lora_preset: 'Detail' },
    },
    shipped: SHIPPED, presets: PRESETS,
  });
  assert.deepEqual(patch.config.klein, {
    improve_lora_preset: 'Detail',
    improve_consistency_strength: 0.8,
    improve_steps: 6,
    improve_base_lora_strength: 0.5,
    improve_megapixels: 4,
    unet: '',                                  // ran on auto → the auto pin
  });
  assert.equal(report.knobs, true);
  assert.match(restoreImproveMessage(report), /strength, steps, output size and model/);
});

test('the recorded preset NAME wins over content matching when it still exists', () => {
  // The preset's strength was tweaked since the render: the rows no longer
  // match by content, but the name still names the user's intent.
  const { patch } = restoreImprovePatch({
    img: { prompt: 'add grain',
      extra_loras: '[{"filename":"klein/d.safetensors","strength":0.9}]',
      improve_profile: { lora_preset: 'Detail', steps: 4 } },
    shipped: SHIPPED, presets: PRESETS,
  });
  assert.equal(patch.config.klein.improve_lora_preset, 'Detail');
});

test('junk profile values degrade knob by knob, never to a default silently written', () => {
  const { patch } = restoreImprovePatch({
    img: { prompt: 'add grain', extra_loras: null,
      improve_profile: { steps: 'many', megapixels: 4, klein_model: 'klein/x.safetensors' } },
    shipped: SHIPPED, presets: PRESETS,
  });
  assert.equal(patch.config.klein.improve_steps, undefined);
  assert.equal(patch.config.klein.improve_megapixels, 4);
  assert.equal(patch.config.klein.unet, 'klein/x.safetensors');
});

test('a row from before the profile column restores less, and the toast says so', () => {
  const { patch, report } = restoreImprovePatch({
    img: { prompt: 'add grain', extra_loras: null },
    shipped: SHIPPED, presets: PRESETS,
  });
  assert.deepEqual(patch.config.klein, { improve_lora_preset: '' });
  assert.equal(report.knobs, false);
  assert.match(restoreImproveMessage(report), /were not recorded on this image/);
});

test('unmatched rows leave the preset knob ALONE and say so out loud', () => {
  const { patch, report } = restoreImprovePatch({
    img: { prompt: 'add grain',
      extra_loras: '[{"filename":"klein/gone.safetensors","strength":1}]' },
    shipped: SHIPPED, presets: PRESETS,
  });
  // Writing '' would claim "no preset" about a pass that ran one.
  assert.equal(patch.config.klein, undefined);
  assert.equal(report.unmatchedLoras, true);
  assert.match(restoreImproveMessage(report), /match none of your presets/);
  assert.match(restoreImproveMessage(report), /left unchanged/);
});
