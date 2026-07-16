const DATASET_KINDS = new Set(['character', 'concept', 'style'])
const VARIANT_SCOPED_FAMILIES = new Set(['zimage', 'krea', 'flux2klein'])

/** Return the optional dataset scope carried by a preset.
 *
 * `dataset_kind` is the current API field. `kind` is accepted only when it is
 * an actual dataset kind so the JSON envelope (`kind: "training-preset"`) can
 * never be mistaken for preset metadata.
 */
export function trainingPresetDatasetKind(preset) {
  if (DATASET_KINDS.has(preset?.dataset_kind)) return preset.dataset_kind
  if (DATASET_KINDS.has(preset?.kind)) return preset.kind
  return null
}

/** A preset is family-exact and, when scoped, dataset-kind-exact.
 *
 * Old database presets did not store a dataset kind. They remain compatible
 * with the matching model family; every new built-in carries the explicit
 * scope and is therefore filtered strictly.
 */
export function isTrainingPresetCompatible(preset, { trainType, datasetKind, variant } = {}) {
  if (!preset || String(preset.train_type || '') !== String(trainType || '')) return false
  const presetKind = trainingPresetDatasetKind(preset)
  if (presetKind && presetKind !== datasetKind) return false
  const variants = Array.isArray(preset.variants)
    ? preset.variants.map(String).filter(Boolean)
    : []
  return !variants.length || !variant || variants.includes(String(variant))
}

export function filterTrainingPresets(presets, context) {
  return (Array.isArray(presets) ? presets : [])
    .filter((preset) => isTrainingPresetCompatible(preset, context))
}

/** Keep a selection only while it is present in the currently visible scope. */
export function compatibleTrainingPresetSelection(selection, presets, context) {
  const wanted = String(selection ?? '')
  if (!wanted) return ''
  return filterTrainingPresets(presets, context)
    .some((preset) => String(preset.id) === wanted) ? wanted : ''
}

/** Build the only payload the UI may send to the apply endpoint.
 *
 * Returning null is the last client-side mismatch guard: the caller performs
 * no request in that case. The server repeats this validation atomically.
 */
export function trainingPresetApplyPayload(preset, context = {}) {
  if (!isTrainingPresetCompatible(preset, context) || preset?.id == null) return null
  return {
    preset_id: preset.id,
    train_type: context.trainType,
    variant: context.variant,
  }
}

/** Scope metadata for “Save current”. Single-recipe families deliberately omit
 * `variant`; sending their UI placeholder (`turbo`) would be invalid API data. */
export function trainingPresetSnapshotScope({ trainType, datasetKind, variant } = {}) {
  return {
    train_type: trainType,
    dataset_kind: datasetKind,
    ...(VARIANT_SCOPED_FAMILIES.has(trainType) && variant ? { variant } : {}),
  }
}
