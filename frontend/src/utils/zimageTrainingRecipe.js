const ZIMAGE_OFFICIAL_RECIPES = Object.freeze({
  turbo: Object.freeze({
    variant: 'turbo',
    variantLabel: 'Turbo (distilled)',
    baseLabel: 'Tongyi-MAI/Z-Image-Turbo',
    adapterActive: true,
    adapterLabel: 'Turbo training adapter v2',
    inferenceHint: '8-step inference',
  }),
  base: Object.freeze({
    variant: 'base',
    variantLabel: 'Base (non-distilled)',
    baseLabel: 'Tongyi-MAI/Z-Image',
    adapterActive: false,
    adapterLabel: 'No training adapter',
    inferenceHint: '28–50 steps · CFG 3–5',
  }),
  deturbo: Object.freeze({
    variant: 'deturbo',
    variantLabel: 'De-Turbo',
    baseLabel: 'ostris/Z-Image-De-Turbo',
    adapterActive: false,
    adapterLabel: 'No training adapter',
    inferenceHint: '20–30 steps · CFG 2–3',
  }),
});

export const ZIMAGE_TURBO_LONG_RUN_STEPS = 2000;

/**
 * Z-Image is the one family where the variant selects a different official
 * repository (and, for Turbo, a required training adapter). Keep the fallback
 * deliberately conservative: an unknown/stale persisted value must never turn
 * into a hidden non-Turbo recipe.
 */
export function normalizeZImageVariant(variant) {
  return Object.hasOwn(ZIMAGE_OFFICIAL_RECIPES, variant) ? variant : 'turbo';
}

/**
 * Describe what the next Z-Image run will actually use. A selected local or
 * converted base remains labelled as such: the UI must not claim it is one of
 * the official repositories merely because a variant was selected.
 */
export function describeZImageRecipe({ variant, base = '', baseLabel = '', customBase = false } = {}) {
  const safeVariant = normalizeZImageVariant(variant);
  const official = ZIMAGE_OFFICIAL_RECIPES[safeVariant];
  const usesCustomBase = customBase || Boolean(String(base || '').trim());

  return {
    ...official,
    baseLabel: usesCustomBase
      ? (String(baseLabel || '').trim() || 'Selected custom/local Z-Image base')
      : official.baseLabel,
    officialBase: !usesCustomBase,
    customVerificationRequired: usesCustomBase,
  };
}

export function isLongZImageTurboRun({ variant, steps } = {}) {
  const parsed = Number(steps);
  return normalizeZImageVariant(variant) === 'turbo'
    && Number.isFinite(parsed)
    && parsed > ZIMAGE_TURBO_LONG_RUN_STEPS;
}
