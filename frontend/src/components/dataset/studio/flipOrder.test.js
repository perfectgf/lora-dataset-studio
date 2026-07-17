import test from 'node:test';
import assert from 'node:assert/strict';
import { flipOrder } from './flipOrder.js';

// Clé « studio solo » : variante (z_model/aspect/cfg/steps) → checkpoint → seed →
// STRENGTH en dernier (les strengths d'un même rendu doivent être adjacentes).
const soloKey = (c) => [
  c.z_model_label || c.z_model || '', c.aspect || '', c.cfg ?? 0,
  c.steps ?? 0, c.steps2 ?? 0, c.label || '', c.seed ?? 0, c.strength ?? 0,
];

const done = (over) => ({ status: 'done', filename: `${Math.random()}.png`, ...over });

test('drops cells that have no openable image (pending / failed / no filename)', () => {
  const cells = [
    done({ id: 1, label: 'ep8', seed: 5, strength: 0.5 }),
    { id: 2, status: 'pending', label: 'ep8', seed: 5, strength: 0.7 },
    { id: 3, status: 'failed', filename: 'x.png', label: 'ep8', seed: 5, strength: 0.9 },
    { id: 4, status: 'done', filename: null, label: 'ep8', seed: 5, strength: 1.1 },
  ];
  const ids = flipOrder(cells, soloKey).map((c) => c.id);
  assert.deepEqual(ids, [1]);
});

test('strength variants of the SAME render are adjacent, ascending', () => {
  // Deux rendus (seed 5 et seed 9) sur le même checkpoint, chacun en 3 strengths,
  // fournis dans le désordre. On attend chaque rendu regroupé, strengths croissantes.
  const cells = [
    done({ id: 'a', label: 'ep8', seed: 5, strength: 0.9 }),
    done({ id: 'b', label: 'ep8', seed: 9, strength: 0.5 }),
    done({ id: 'c', label: 'ep8', seed: 5, strength: 0.5 }),
    done({ id: 'd', label: 'ep8', seed: 9, strength: 0.9 }),
    done({ id: 'e', label: 'ep8', seed: 5, strength: 0.7 }),
    done({ id: 'f', label: 'ep8', seed: 9, strength: 0.7 }),
  ];
  const ids = flipOrder(cells, soloKey).map((c) => c.id);
  // seed 5 : 0.5, 0.7, 0.9 puis seed 9 : 0.5, 0.7, 0.9
  assert.deepEqual(ids, ['c', 'e', 'a', 'b', 'f', 'd']);
});

test('different checkpoints stay grouped (checkpoint before seed before strength)', () => {
  const cells = [
    done({ id: 1, label: 'ep8', seed: 5, strength: 1.0 }),
    done({ id: 2, label: 'ep4', seed: 5, strength: 1.0 }),
    done({ id: 3, label: 'ep8', seed: 5, strength: 0.6 }),
    done({ id: 4, label: 'ep4', seed: 5, strength: 0.6 }),
  ];
  const ids = flipOrder(cells, soloKey).map((c) => c.id);
  // ep4 (0.6, 1.0) puis ep8 (0.6, 1.0) — tri numérique sur le label
  assert.deepEqual(ids, [4, 2, 3, 1]);
});

test('comparison key groups strengths of the same LoRA+seed adjacently', () => {
  const compKey = (c) => [c.dataset_id ?? 0, c.aspect || '', c.seed ?? 0, c.strength ?? 0];
  const cells = [
    done({ id: 1, dataset_id: 42, seed: 7, strength: 1.0 }),
    done({ id: 2, dataset_id: 11, seed: 7, strength: 0.5 }),
    done({ id: 3, dataset_id: 42, seed: 7, strength: 0.5 }),
    done({ id: 4, dataset_id: 11, seed: 7, strength: 1.0 }),
  ];
  const ids = flipOrder(cells, compKey).map((c) => c.id);
  // dataset 11 (0.5, 1.0) puis dataset 42 (0.5, 1.0)
  assert.deepEqual(ids, [2, 4, 3, 1]);
});

test('empty / nullish input is safe', () => {
  assert.deepEqual(flipOrder(null, soloKey), []);
  assert.deepEqual(flipOrder(undefined, soloKey), []);
  assert.deepEqual(flipOrder([], soloKey), []);
});
