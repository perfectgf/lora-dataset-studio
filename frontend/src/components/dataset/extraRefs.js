/** Which file the ✂ editor must open for an extra reference.
 *
 * The backend sends `ref_extra_crop_sources` aligned index-by-index with
 * `ref_extra_filenames`: the kept full-frame ORIGINAL when there is one, else the
 * extra itself. Kept as a pure helper (no JSX) so it is directly testable.
 *
 * Returns null when the filename isn't one of this dataset's extras — the caller
 * then opens nothing rather than guessing a path. Falls back to the extra itself
 * when the payload predates the crop-sources field (older tab, cached response):
 * cropping still works, it just can't widen past the current frame. */
export function extraRefCropSource(extraRefs, cropSources, filename) {
  const list = Array.isArray(extraRefs) ? extraRefs : [];
  const i = list.indexOf(filename);
  if (i < 0) return null;
  const src = Array.isArray(cropSources) ? cropSources[i] : null;
  return typeof src === 'string' && src ? src : filename;
}
