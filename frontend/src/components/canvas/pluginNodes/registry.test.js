import test from 'node:test';
import assert from 'node:assert/strict';
import { PLUGIN_NODE_TYPES, nodeKey, knownType } from './registry.js';

test('external-lora is a registered type with the full contract', () => {
  const t = PLUGIN_NODE_TYPES['external-lora'];
  assert.ok(t, 'type registered');
  for (const k of ['type', 'Card', 'AddFlow', 'payload', 'edge']) {
    assert.ok(t[k], `type declares ${k}`);
  }
  assert.equal(t.type, 'external-lora');
  assert.equal(t.edge.gradient, 'lds-edge-external');
});

test('nodeKey is stable and separator/case-insensitive for external LoRAs', () => {
  assert.equal(nodeKey('external-lora', { filename: 'Krea\\Foo.safetensors' }),
    nodeKey('external-lora', { filename: 'krea/foo.safetensors' }));
  assert.ok(nodeKey('external-lora', { filename: 'a.safetensors' }).startsWith('ext:'));
});

test('unknown types are reported unknown', () => {
  assert.equal(knownType('external-lora'), true);
  assert.equal(knownType('teleporter'), false);
});

test('external-lora payload returns only the checked nodes, in the shape genSettings expects', () => {
  const nodes = [
    { filename: 'a.safetensors', strength: 0.5 },
    { filename: 'b.safetensors', strength: 1 },
    { filename: 'c.safetensors', strength: 2 },
  ];
  const checked = new Set(['a.safetensors', 'c.safetensors']);
  const payload = PLUGIN_NODE_TYPES['external-lora'].payload(nodes, checked);
  assert.deepEqual(payload, {
    external_loras: [
      { filename: 'a.safetensors', strength: 0.5 },
      { filename: 'c.safetensors', strength: 2 },
    ],
  });
});
