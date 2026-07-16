import test from 'node:test';
import assert from 'node:assert/strict';
import {
  describeZImageRecipe,
  isLongZImageTurboRun,
  normalizeZImageVariant,
  ZIMAGE_TURBO_LONG_RUN_STEPS,
} from './zimageTrainingRecipe.js';

test('unknown persisted Z-Image variants fall back to the safe Turbo recipe', () => {
  assert.equal(normalizeZImageVariant('4b'), 'turbo');
  assert.equal(normalizeZImageVariant('deturbo'), 'deturbo');
});

test('official Z-Image variants expose their effective base and adapter state', () => {
  assert.deepEqual(
    describeZImageRecipe({ variant: 'turbo' }),
    {
      variant: 'turbo',
      variantLabel: 'Turbo (distilled)',
      baseLabel: 'Tongyi-MAI/Z-Image-Turbo',
      adapterActive: true,
      adapterLabel: 'Turbo training adapter v2',
      inferenceHint: '8-step inference',
      officialBase: true,
      customVerificationRequired: false,
    },
  );
  assert.equal(describeZImageRecipe({ variant: 'base' }).baseLabel, 'Tongyi-MAI/Z-Image');
  assert.equal(describeZImageRecipe({ variant: 'deturbo' }).baseLabel, 'ostris/Z-Image-De-Turbo');
  assert.equal(describeZImageRecipe({ variant: 'deturbo' }).adapterActive, false);
});

test('a custom base keeps its truthful label while Turbo still declares its adapter', () => {
  const recipe = describeZImageRecipe({
    variant: 'turbo',
    base: 'my_merge.safetensors',
    baseLabel: 'custom: my_merge.safetensors',
  });
  assert.equal(recipe.baseLabel, 'custom: my_merge.safetensors');
  assert.equal(recipe.officialBase, false);
  assert.equal(recipe.customVerificationRequired, true);
  assert.equal(recipe.adapterActive, true);
});

test('long-run warning only applies above the Turbo threshold', () => {
  assert.equal(ZIMAGE_TURBO_LONG_RUN_STEPS, 2000);
  assert.equal(isLongZImageTurboRun({ variant: 'turbo', steps: ZIMAGE_TURBO_LONG_RUN_STEPS }), false);
  assert.equal(isLongZImageTurboRun({ variant: 'turbo', steps: ZIMAGE_TURBO_LONG_RUN_STEPS + 1 }), true);
  assert.equal(isLongZImageTurboRun({ variant: 'base', steps: 12000 }), false);
  assert.equal(isLongZImageTurboRun({ variant: 'turbo', steps: null }), false);
});
