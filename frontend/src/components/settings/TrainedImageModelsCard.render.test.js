import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { transformSync } from 'esbuild';
import { createElement, render } from '../../../tests/support/mountJsx.mjs';

const { Card } = await import('./primitives.jsx');
const { default: ModelFilePicker } = await import('./ModelFilePicker.jsx');
const { default: InstallRunner } = await import('../setup/InstallRunner.jsx');

// Render the real component and children after its data fetch, with only its three local hook
// values supplied by the test. Effects stay inert: this never contacts ComfyUI or starts installs.
const source = readFileSync(new URL('./TrainedImageModelsCard.jsx', import.meta.url), 'utf8')
  .replace(/^import .*\r?\n/gm, '')
  .replace('export default function TrainedImageModelsCard', 'function TrainedImageModelsCard');
const compiled = transformSync(source, { loader: 'jsx', jsxFactory: 'createElement' }).code;

function renderLoaded(families, config = {}) {
  const values = [families, false, ''];
  let next = 0;
  const context = vm.createContext({
    createElement, Card, ModelFilePicker, InstallRunner,
    useState: () => [values[next++], () => {}],
    useCallback: (fn) => fn, useEffect: () => {},
    apiFetch: () => { throw new Error('Render tests must not fetch models.'); },
  });
  vm.runInContext(`${compiled}\nglobalThis.Component = TrainedImageModelsCard;`, context);
  return render(context.Component, { config, setField: () => {} });
}

const family = (id, label) => ({
  family: id, label, ready: false, models_ready: false, nodes_checked: false,
  missing_nodes: [], assets: {}, install_actions: [`${id}-model`],
  slots: [{ key: 'diffusion_model', label: 'Diffusion model', filename: `${id}.safetensors`,
    kind: 'diffusion_models', files: [], action: `${id}-model`,
    source_url: 'https://huggingface.co/' }],
});

test('each newly supported family presents its own missing models and installer', () => {
  const html = renderLoaded([
    family('flux', 'FLUX.1'), family('anima', 'Anima'), family('qwenimage21', 'Qwen-Image 2.1'),
    family('moss', 'MOSS-Image'),
  ]);
  for (const label of ['FLUX.1', 'Anima', 'Qwen-Image 2.1', 'MOSS-Image']) {
    assert.ok(html.includes(`aria-label="${label} Diffusion model"`));
  }
  assert.equal((html.match(/>Install Diffusion model</g) || []).length, 4);
  assert.equal((html.match(/Model files needed/g) || []).length, 4);
  assert.ok(html.includes('Model files alone do not confirm readiness.'));
});

test('a missing custom pin shows how to repair its selection instead of offering an unrelated download', () => {
  const html = renderLoaded([{ ...family('anima', 'Anima'), install_actions: [],
    pin_warnings: [{ slot: 'diffusion_model', message: 'Choose another file or clear this selection.' }],
  }], { studio_models: { anima: { diffusion_model: 'custom/anima.safetensors' } } });
  assert.match(html, /Check file selection/);
  assert.match(html, /Choose another file or clear this selection/);
  assert.match(html, /Use automatic detection/);
  assert.match(html, /value="custom\/anima\.safetensors"/);
  assert.doesNotMatch(html, />Install Diffusion model</);
});

test('Z-Image shows the files its resolvers found and installs the missing ones, without file fields', () => {
  const slot = (key, label, filename) => ({ key, label, filename, kind: 'vae', files: [],
    action: `studio_zimage_${key}`, source_url: 'https://huggingface.co/Comfy-Org/Z-Image-Turbo',
    configurable: false });
  const html = renderLoaded([{ ...family('zimage', 'Z-Image'),
    assets: { diffusion_model: '', vae: 'ae.safetensors' },
    install_actions: ['studio_zimage_diffusion_model'],
    slots: [slot('diffusion_model', 'Turbo diffusion model', 'z_image_turbo_bf16.safetensors'),
      slot('vae', 'VAE', 'ae.safetensors')],
  }], { studio_models: { zimage: { vae: 'ignored.safetensors' } } });
  assert.doesNotMatch(html, /aria-label="Z-Image /);
  assert.doesNotMatch(html, /Use automatic detection/);
  assert.match(html, /Not found · z_image_turbo_bf16\.safetensors/);
  assert.match(html, /Found: ae\.safetensors/);
  assert.equal((html.match(/>Install Turbo diffusion model</g) || []).length, 1);
  assert.doesNotMatch(html, />Install VAE</);
});

test('an invalid configuration displays its error even when the resolver has no assets', () => {
  const html = renderLoaded([{ ...family('qwenimage21', 'Qwen-Image 2.1'),
    install_actions: [], config_error: 'Select a model inside your configured model folders.',
  }]);
  assert.match(html, /Select a model inside your configured model folders/);
  assert.match(html, /Check file selection/);
  assert.doesNotMatch(html, />Install Diffusion model</);
});
