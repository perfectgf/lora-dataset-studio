/* 🔤 The zones gallery's JSX-free half.
 *
 * Both Find-text launch windows (bank and dataset) show the flagged pages with
 * their zones drawn on top. The zones arrive as normalized [x1,y1,x2,y2]
 * regions — the SAME shape the mask editor edits (utils/watermarkRegions.js) —
 * and are drawn as percentage-positioned boxes over an aspect-preserving
 * thumbnail, so one bad row must never take the whole strip down. That
 * defensive reading is logic, so it lives here where node --test can reach it.
 */

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** The drawable zones of one page: finite, normalized, ordered corners, and
 *  never zero-area (a degenerate box would render as an invisible sliver and
 *  read as "no zone here" on a page the scan DID flag). */
export function galleryZones(regions) {
  const out = [];
  for (const region of Array.isArray(regions) ? regions : []) {
    if (!Array.isArray(region) || region.length < 4) continue;
    const nums = region.slice(0, 4).map(Number);
    if (nums.some((v) => !Number.isFinite(v))) continue;
    const left = clamp01(Math.min(nums[0], nums[2]));
    const top = clamp01(Math.min(nums[1], nums[3]));
    const right = clamp01(Math.max(nums[0], nums[2]));
    const bottom = clamp01(Math.max(nums[1], nums[3]));
    if (right - left <= 0 || bottom - top <= 0) continue;
    out.push([left, top, right, bottom]);
  }
  return out;
}

/** CSS for one zone box, as percentages of the rendered thumbnail. */
export function zoneStyle([left, top, right, bottom]) {
  const pct = (v) => `${(v * 100).toFixed(2)}%`;
  return {
    left: pct(left),
    top: pct(top),
    width: pct(right - left),
    height: pct(bottom - top),
  };
}

/** The strip's one-line header. `shown` is what the strip renders, `total`
 *  what the scan flagged — saying both is what keeps "12 tiles" from reading
 *  as "12 flagged" on a bank where 300 pages carry text. */
export function galleryHeadline(shown, total) {
  const t = Number(total) || 0;
  const s = Number(shown) || 0;
  let line = ` — ${t} page${t === 1 ? '' : 's'} flagged`;
  if (t > s) line += `, first ${s} shown`;
  return line;
}
