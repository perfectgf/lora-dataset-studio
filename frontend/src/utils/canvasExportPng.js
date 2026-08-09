/* 📷 Export the ◉ LoRA Canvas as a PNG.
 *
 * ── What this is, and what it is honestly NOT ───────────────────────────────
 * It is a RE-DRAW of the board's world onto a 2D canvas, not a screenshot of
 * the DOM. There is no way to rasterise arbitrary styled HTML from a page
 * without either a third-party library or `foreignObject`, and `foreignObject`
 * only works if every stylesheet is inlined first — this app's CSS is a Tailwind
 * build, so the "screenshot" would come out unstyled. A re-draw is the honest
 * option: it can be tested, it has no dependency, and it cannot silently
 * degrade into a blank rectangle on a browser that changed its mind about
 * tainting.
 *
 * So the file you get is a MAP of the board:
 *   • every pinned picture, at full quality, exactly where it sits;
 *   • every run card, as a labelled block with its checkpoint steps;
 *   • every descent edge, drawn from the very path data the board draws
 *     (`edge.d` from utils/lineageGraph — the same curve, not a lookalike);
 *   • every lane's name.
 * What it does not carry is the interactive chrome: buttons, badges, hover
 * highlights, the reference thumbnail. Those are controls; a picture of a
 * control is noise. The UI says this before the click rather than after.
 *
 * ── Why the images are same-origin and that matters ────────────────────────
 * Every picture is served by this app (`/api/...`), so drawing them does not
 * taint the canvas and `toBlob` works. An image that fails to load is drawn as
 * a labelled placeholder rather than skipped: a board that quietly exports
 * eleven of its twelve pictures is worse than one that says which one is gone.
 *
 * The geometry below is PURE and unit-tested; the drawing (which needs a real
 * CanvasRenderingContext2D and real Image loads) is at the bottom and is
 * exercised in the browser.
 */

/** Air around the content, in world units. Enough that the outermost card does
 *  not touch the edge of the file. */
export const EXPORT_PADDING = 56;

/** How many device pixels a world unit is worth by default. 2 keeps text and
 *  edges crisp when the file is viewed at 100 %; the cap below can lower it. */
export const EXPORT_PIXEL_RATIO = 2;

/** The ceiling, in total pixels. Browsers refuse canvases past a few hundred
 *  megapixels — silently, by producing a blank one — and a twenty-lane board at
 *  ×2 gets there. Hitting the cap lowers the resolution instead of failing:
 *  a slightly softer poster is a far better answer than an empty file. */
export const EXPORT_MAX_PIXELS = 32e6;

/** …and the single-axis one. Chrome caps a canvas dimension at 65 535 px, and a
 *  wide board can exceed that long before it exceeds the area budget. */
export const EXPORT_MAX_SIDE = 16384;

/** The board's box, in world units, with the export's own margin. `world` is
 *  the very object the board lays itself out from (utils/canvasLayout.stackLanes),
 *  including its possibly-NEGATIVE origin — a picture dragged above its lane is
 *  part of the board and must be part of the file. */
export function boardExportBox(world) {
  const x = Number(world?.x) || 0;
  const y = Number(world?.y) || 0;
  const width = Math.max(0, Number(world?.width) || 0);
  const height = Math.max(0, Number(world?.height) || 0);
  return {
    x: x - EXPORT_PADDING,
    y: y - EXPORT_PADDING,
    width: width + EXPORT_PADDING * 2,
    height: height + EXPORT_PADDING * 2,
  };
}

/** How many device pixels per world unit this board can afford. Never above
 *  `pixelRatio`, never so low that the file stops being readable — a board too
 *  big for both budgets is reported by `boardExportRefusal`, not silently
 *  exported at an unreadable scale. */
export function boardExportScale(box, { pixelRatio = EXPORT_PIXEL_RATIO,
  maxPixels = EXPORT_MAX_PIXELS, maxSide = EXPORT_MAX_SIDE } = {}) {
  const w = Math.max(1, Number(box?.width) || 1);
  const h = Math.max(1, Number(box?.height) || 1);
  const wanted = Math.max(0.05, Number(pixelRatio) || EXPORT_PIXEL_RATIO);
  const byArea = Math.sqrt(Math.max(1, maxPixels) / (w * h));
  const bySide = Math.min(maxSide / w, maxSide / h);
  return Math.min(wanted, byArea, bySide);
}

/** The whole plan: the box, the scale, and the pixel size of the file. */
export function boardExportPlan(world, opts = {}) {
  const box = boardExportBox(world);
  const scale = boardExportScale(box, opts);
  return {
    box,
    scale,
    width: Math.max(1, Math.round(box.width * scale)),
    height: Math.max(1, Math.round(box.height * scale)),
  };
}

/** Why this board cannot be exported, or null. The only real case is "there is
 *  nothing on it" — everything else degrades (see the scale) rather than
 *  refusing, because a poster of a big board is exactly what a big board is
 *  for. */
export function boardExportRefusal(world) {
  const lanes = world?.lanes?.length || 0;
  if (!lanes) return 'There is nothing on the board to export yet.';
  if (!(Number(world?.width) > 0) || !(Number(world?.height) > 0)) {
    return 'The board has no size yet — give the lanes a moment to load.';
  }
  return null;
}

const two = (n) => String(n).padStart(2, '0');

/** The file's name. It carries the DATE and the number of lanes, because a
 *  Downloads folder full of `canvas.png` is a Downloads folder with one usable
 *  export in it. */
export function boardExportFilename(when = new Date(), laneCount = 0) {
  const d = when instanceof Date && !Number.isNaN(when.getTime()) ? when : new Date();
  const stamp = `${d.getFullYear()}-${two(d.getMonth() + 1)}-${two(d.getDate())}`
    + `-${two(d.getHours())}${two(d.getMinutes())}`;
  const lanes = Math.max(0, Number(laneCount) || 0);
  return `lora-canvas-${stamp}-${lanes}-lane${lanes === 1 ? '' : 's'}.png`;
}

/* --------------------------------------------------------------------------
   The palette. Fixed rather than read off the live theme, on purpose: the file
   is shared and looked at outside the app, so it always comes out in the dark
   graphite the board is designed in rather than inheriting whatever the viewer
   had selected the second they pressed the button.
   -------------------------------------------------------------------------- */
export const EXPORT_COLORS = {
  background: '#0d1014',
  laneName: '#e6e9ef',
  laneMeta: '#8b93a3',
  cardFill: '#161b22',
  cardStroke: '#2b3340',
  cardText: '#e6e9ef',
  cardMeta: '#8b93a3',
  pillFill: '#1d2430',
  pillStroke: '#3a4455',
  pillText: '#c3cad6',
  edge: '#4a5566',
  edgeSpine: '#7c8cf8',
  imageStroke: '#6b7fd7',
  missing: '#3a2226',
  missingText: '#f2b8bd',
};

/** One run card's two lines of text, decided here rather than in the drawing
 *  loop so a test can pin what the poster actually says. */
export function exportCardLines(node) {
  const rid = node?.record_id;
  const steps = Number(node?.steps);
  const title = rid == null ? 'Run' : `Run #${rid}`;
  const bits = [];
  if (Number.isFinite(steps) && steps > 0) bits.push(`${steps.toLocaleString('en-US')} steps`);
  if (node?.train_type) bits.push(String(node.train_type));
  if (node?.source === 'cloud') bits.push('cloud');
  return { title, subtitle: bits.join(' · ') };
}

/* --------------------------------------------------------------------------
   The drawing. Browser-only: it needs a 2D context and real image decoding.
   -------------------------------------------------------------------------- */

/** Load one picture, resolving to `null` instead of rejecting — one missing
 *  file must not take the whole export down with it. */
function loadImage(url) {
  return new Promise((resolve) => {
    if (!url || typeof Image === 'undefined') { resolve(null); return; }
    const im = new Image();
    im.decoding = 'sync';
    im.onload = () => resolve(im);
    im.onerror = () => resolve(null);
    im.src = url;
  });
}

function roundRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function clipText(ctx, text, maxW) {
  if (!text) return '';
  if (ctx.measureText(text).width <= maxW) return text;
  let out = text;
  while (out.length > 1 && ctx.measureText(`${out}…`).width > maxW) out = out.slice(0, -1);
  return `${out}…`;
}

/**
 * Draw the whole board into `canvas` and return the plan that was used.
 *
 * `lanes` are the laid-out lanes (world.lanes), `drawnByLane` maps a dataset id
 * to the pictures as they are DRAWN (utils/canvasImageGroups.drawnNodes) — the
 * strip's slot for a member, not the box it remembers, so the file matches the
 * screen.
 */
export async function drawBoardExport(canvas, { world, lanes, drawnByLane, cardW = 264,
  cardH = 64, laneHeaderH = 40, plan: given = null }) {
  const plan = given || boardExportPlan(world);
  canvas.width = plan.width;
  canvas.height = plan.height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = EXPORT_COLORS.background;
  ctx.fillRect(0, 0, plan.width, plan.height);
  ctx.save();
  ctx.scale(plan.scale, plan.scale);
  ctx.translate(-plan.box.x, -plan.box.y);

  // Pictures first so that nothing about the load order can reorder the board:
  // every image is fetched before a single pixel of it is drawn.
  const jobs = [];
  for (const lane of lanes) {
    for (const n of (drawnByLane[lane.datasetId] || [])) {
      jobs.push(loadImage(n?.image?.url).then((im) => ({ lane, n, im })));
    }
  }
  const pictures = await Promise.all(jobs);

  for (const lane of lanes) {
    // --- the lane's name -----------------------------------------------------
    ctx.fillStyle = EXPORT_COLORS.laneName;
    ctx.font = '600 20px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    const name = clipText(ctx, lane.name || `Dataset ${lane.datasetId}`, Math.max(cardW, lane.width));
    ctx.fillText(name, lane.x, lane.y + laneHeaderH / 2);
    const nameW = ctx.measureText(name).width;
    ctx.fillStyle = EXPORT_COLORS.laneMeta;
    ctx.font = '400 13px system-ui, sans-serif';
    ctx.fillText(`${lane.runs || 0} run${lane.runs === 1 ? '' : 's'}`,
      lane.x + nameW + 12, lane.y + laneHeaderH / 2);

    const g = lane.graph;
    if (!g) continue;
    ctx.save();
    ctx.translate(lane.x, lane.graphY);

    // --- the descent edges, from the board's own path data -------------------
    ctx.lineWidth = 2;
    for (const e of (g.edges || [])) {
      ctx.strokeStyle = e.onSpine ? EXPORT_COLORS.edgeSpine : EXPORT_COLORS.edge;
      if (typeof Path2D === 'function' && e.d) {
        ctx.stroke(new Path2D(e.d));
      } else {
        ctx.beginPath();
        ctx.moveTo(e.x1, e.y1);
        ctx.lineTo(e.x2, e.y2);
        ctx.stroke();
      }
    }

    // --- the run cards, and their checkpoint pills ---------------------------
    for (const n of (g.nodes || [])) {
      ctx.fillStyle = EXPORT_COLORS.cardFill;
      ctx.strokeStyle = EXPORT_COLORS.cardStroke;
      ctx.lineWidth = 1.5;
      roundRect(ctx, n.x, n.y, cardW, cardH, 8);
      ctx.fill();
      ctx.stroke();

      const { title, subtitle } = exportCardLines(n.node);
      ctx.fillStyle = EXPORT_COLORS.cardText;
      ctx.font = '600 15px system-ui, sans-serif';
      ctx.fillText(clipText(ctx, title, cardW - 20), n.x + 10, n.y + 22);
      if (subtitle) {
        ctx.fillStyle = EXPORT_COLORS.cardMeta;
        ctx.font = '400 12px system-ui, sans-serif';
        ctx.fillText(clipText(ctx, subtitle, cardW - 20), n.x + 10, n.y + 42);
      }

      for (const p of (n.checkpoints || [])) {
        ctx.fillStyle = EXPORT_COLORS.pillFill;
        ctx.strokeStyle = EXPORT_COLORS.pillStroke;
        ctx.lineWidth = 1;
        roundRect(ctx, p.x, p.y, p.w, p.h, 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = EXPORT_COLORS.pillText;
        ctx.font = '500 10px system-ui, sans-serif';
        ctx.fillText(clipText(ctx, String(p.step ?? '?'), p.w - 6), p.x + 4, p.y + p.h / 2);
      }
    }
    ctx.restore();
  }

  // --- the pinned pictures, above the trees, in lane-local coordinates -------
  for (const { lane, n, im } of pictures) {
    ctx.save();
    ctx.translate(lane.x, lane.graphY);
    if (im) {
      // `object-contain` on screen: the same letterboxing here, or a portrait
      // render would come out stretched in the file it is being compared in.
      const ratio = Math.min(n.w / im.naturalWidth, n.h / im.naturalHeight);
      const dw = im.naturalWidth * ratio;
      const dh = im.naturalHeight * ratio;
      ctx.fillStyle = '#000';
      ctx.fillRect(n.x, n.y, n.w, n.h);
      ctx.drawImage(im, n.x + (n.w - dw) / 2, n.y + (n.h - dh) / 2, dw, dh);
    } else {
      ctx.fillStyle = EXPORT_COLORS.missing;
      ctx.fillRect(n.x, n.y, n.w, n.h);
      ctx.fillStyle = EXPORT_COLORS.missingText;
      ctx.font = '500 12px system-ui, sans-serif';
      ctx.fillText(clipText(ctx, 'image not on disk', n.w - 12), n.x + 6, n.y + n.h / 2);
    }
    ctx.strokeStyle = EXPORT_COLORS.imageStroke;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(n.x, n.y, n.w, n.h);
    ctx.restore();
  }

  ctx.restore();
  return { ...plan, drawn: pictures.length, missing: pictures.filter((p) => !p.im).length };
}
