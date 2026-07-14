export const MIN_WATERMARK_REGION_SIZE = 0.005;
export const MAX_WATERMARK_REGIONS = 32;

export function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function normalizeRegion(region) {
  const left = clamp(Math.min(region[0], region[2]));
  const top = clamp(Math.min(region[1], region[3]));
  const right = clamp(Math.max(region[0], region[2]));
  const bottom = clamp(Math.max(region[1], region[3]));
  return [left, top, right, bottom];
}

function minimumSpan(start, end, minSize) {
  if (end - start >= minSize) return [start, end];
  const center = (start + end) / 2;
  let nextStart = clamp(center - minSize / 2, 0, 1 - minSize);
  let nextEnd = Math.min(1, nextStart + minSize);
  // Keep the actual binary span at or above the server minimum. This is not
  // coordinate rounding; it only compensates for one sub-ULP subtraction loss.
  if (nextEnd - nextStart < minSize) {
    if (nextEnd < 1) nextEnd = Math.min(1, nextEnd + Number.EPSILON);
    else nextStart = Math.max(0, nextStart - Number.EPSILON);
  }
  return [nextStart, nextEnd];
}

function clampLowerEdge(value, upper, minSize) {
  let next = clamp(value, 0, upper - minSize);
  if (upper - next < minSize) next = Math.max(0, next - Number.EPSILON);
  return next;
}

function clampUpperEdge(value, lower, minSize) {
  let next = clamp(value, lower + minSize, 1);
  if (next - lower < minSize) next = Math.min(1, next + Number.EPSILON);
  return next;
}

export function ensureMinimumRegion(region, minSize = MIN_WATERMARK_REGION_SIZE) {
  const size = clamp(minSize);
  const [left, top, right, bottom] = normalizeRegion(region);
  const [nextLeft, nextRight] = minimumSpan(left, right, size);
  const [nextTop, nextBottom] = minimumSpan(top, bottom, size);
  return [nextLeft, nextTop, nextRight, nextBottom];
}

export function regionFromPoints(start, end, minSize = MIN_WATERMARK_REGION_SIZE) {
  return ensureMinimumRegion([start[0], start[1], end[0], end[1]], minSize);
}

export function pointToNormalized(point, rect) {
  if (!rect || rect.width <= 0 || rect.height <= 0) {
    throw new RangeError('Rendered image bounds must have positive width and height');
  }
  return [
    clamp((point.clientX - rect.left) / rect.width),
    clamp((point.clientY - rect.top) / rect.height),
  ];
}

export function moveRegion(region, dx, dy) {
  const [left, top, right, bottom] = normalizeRegion(region);
  const width = right - left;
  const height = bottom - top;
  const nextLeft = clamp(left + dx, 0, 1 - width);
  const nextTop = clamp(top + dy, 0, 1 - height);
  return [nextLeft, nextTop, nextLeft + width, nextTop + height];
}

export function resizeRegion(
  region,
  corner,
  dx,
  dy,
  minSize = MIN_WATERMARK_REGION_SIZE,
) {
  const size = clamp(minSize);
  let [left, top, right, bottom] = ensureMinimumRegion(region, size);

  if (corner === 'nw' || corner === 'sw') {
    left = clampLowerEdge(left + dx, right, size);
  } else if (corner === 'ne' || corner === 'se') {
    right = clampUpperEdge(right + dx, left, size);
  } else {
    throw new RangeError(`Unknown resize corner: ${corner}`);
  }

  if (corner === 'nw' || corner === 'ne') {
    top = clampLowerEdge(top + dy, bottom, size);
  } else {
    bottom = clampUpperEdge(bottom + dy, top, size);
  }

  return [left, top, right, bottom];
}

export function replaceRegion(regions, index, region) {
  return regions.map((item, itemIndex) => (
    itemIndex === index ? [...region] : [...item]
  ));
}

export function removeRegion(regions, index) {
  return regions.filter((_, itemIndex) => itemIndex !== index).map((region) => [...region]);
}

export function serializeWatermarkRegions(regionsOrNull) {
  if (regionsOrNull === null) return null;
  return regionsOrNull.map((region) => region.map((value) => (
    Math.round((value + Number.EPSILON) * 10_000) / 10_000
  )));
}
