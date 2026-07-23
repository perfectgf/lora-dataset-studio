export function defaultCheckpointBase(bases) {
  const choices = Array.isArray(bases) ? bases : [];
  const official = choices.find((item) => item?.value === '');
  return official ? '' : (choices[0]?.value || '');
}

const CHECKPOINT_VARIANTS = Object.freeze({
  zimage: Object.freeze([
    Object.freeze({ value: 'turbo', label: 'Turbo · adapter v2' }),
    Object.freeze({ value: 'base', label: 'Base · non-distilled' }),
    Object.freeze({ value: 'deturbo', label: 'De-Turbo · no adapter' }),
  ]),
  krea: Object.freeze([
    Object.freeze({ value: 'base', label: 'Raw' }),
    Object.freeze({ value: 'turbo', label: 'Turbo · adapter' }),
  ]),
  flux2klein: Object.freeze([
    Object.freeze({ value: '4b', label: '4B' }),
    Object.freeze({ value: '9b', label: '9B' }),
  ]),
});

export function checkpointVariantOptions(trainType) {
  return CHECKPOINT_VARIANTS[trainType] || [{ value: 'turbo', label: 'Default' }];
}

export function defaultCheckpointVariant(trainType) {
  return checkpointVariantOptions(trainType)[0].value;
}

export function normalizeCheckpointVariant(trainType, variant) {
  const choices = checkpointVariantOptions(trainType);
  return choices.some((choice) => choice.value === variant)
    ? variant
    : choices[0].value;
}

export function checkpointVariantLabel(trainType, variant) {
  const normalized = normalizeCheckpointVariant(trainType, variant);
  return checkpointVariantOptions(trainType).find((choice) => choice.value === normalized)?.label || normalized;
}

export function checkpointSelectionMatchesTraining(
  checkpointType, checkpointBase, checkpointVariant,
  trainType, trainBase, trainVariant,
) {
  return checkpointType === trainType
    && checkpointBase === trainBase
    && checkpointVariant === trainVariant;
}

/** Canonical family+base+variant payload shared by checkpoint list/import/open/
 * continue/cleanup calls. Empty official bases are intentionally preserved. */
export function trainingRunSelection(baseModel, trainType, variant) {
  return {
    ...(baseModel !== undefined && baseModel !== null ? { base_model: baseModel } : {}),
    ...(trainType ? { train_type: trainType } : {}),
    ...(variant ? { variant } : {}),
  };
}

/** Cloud launches must always carry the selected base, including the empty
 * official sentinel. A non-empty local/custom selection then reaches the
 * authoritative server guard instead of silently falling back to official. */
export function cloudTrainingLaunchPayload({
  baseModel = '', variant, trainType, masked = true, steps, gpuName,
} = {}) {
  return {
    base_model: baseModel,
    variant,
    train_type: trainType,
    masked,
    ...(steps ? { steps } : {}),
    ...(gpuName ? { gpu_name: gpuName } : {}),
  };
}

export function trainFamilyLabel(type) {
  if (type === 'sdxl') return 'SDXL';
  if (type === 'krea') return 'Krea 2';
  if (type === 'flux') return 'FLUX.1';
  if (type === 'flux2klein') return 'FLUX.2 Klein';
  if (type === 'anima') return 'Anima';
  return 'Z-Image';
}

export function loraFolderLabel(type) {
  if (type === 'sdxl') return 'loras/sdxl';
  if (type === 'krea') return 'loras/krea';
  if (type === 'flux') return 'loras/flux';
  if (type === 'flux2klein') return 'loras/flux2klein';
  if (type === 'anima') return 'loras/anima';
  return 'loras/z image';
}
