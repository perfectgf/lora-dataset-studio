/* 📌 Pin ALL the images a run just produced onto the ◉ LoRA Canvas, in one go.

   A finished generation says "5 images ready" and names the checkpoints they
   joined. Getting them onto the board meant opening each of those galleries and
   pinning the pictures one by one — for a lot spanning four runs, four panels
   and five gestures to see the thing the run was fired to let you see.

   This module is the whole decision layer of the one-click version. Pure, no
   JSX and no DOM, because `node --test` cannot parse JSX and the part that
   decides whether the feature is usable is geometry, not markup.

   ⚠️ EVERYTHING HERE IS IN BOARD UNITS, never in screen pixels. The board is
   zoomable (a real board is often read at 24 %); a rule written in pixels would
   be right at one zoom level and wrong at every other one.

   ── The placement, and why this one ──────────────────────────────────────────

   The requirement is not "tidy", it is "nothing may ever be on top of anything
   else" — a lot dropped onto a board that already carries cards, checkpoint
   pills and hand-placed pictures. Three candidates were on the table: a spiral
   around each source, a free-space search from the source's own spot, and a
   band under the tree. The first two share one flaw: they place INTO the tree,
   so they can only work by searching around obstacles, and the result reads as
   scatter — you cannot tell at a glance which picture came from which run, which
   is the only reason to put them on the board at all.

   So: a CONTACT-SHEET BAND below the lane, one COLUMN per source checkpoint,
   each column starting at the x of its own source and sliding right until it
   finds a free column.

     • no overlap is structural, not searched for: the band starts below the
       lowest thing already on the lane, and inside the band each column is
       reserved before it is filled. There is no case where a tile is placed
       "hopefully";
     • a source's images are a vertical stack under that source, so a lot from
       four runs reads as four groups without a legend — and the connector each
       node already draws to its own pill (utils/canvasImageNodes.imageNodeEdges)
       confirms it;
     • it is trivially deterministic — no search order to depend on.

   The cost, stated: a picture is BELOW the tree rather than beside its card, so
   on a deep lineage it is further from its pill than a single hand-pin would
   be. That is the trade the requirement asks for; a pin dropped by hand still
   lands beside its card (defaultImageSpot), which is unchanged.

   ── When there is no room ────────────────────────────────────────────────────

   Structurally, there always is: the band grows down and right, outside
   everything, and the lane grows with it (imageNodeExtent). So the only bound
   is a deliberate one — PIN_BATCH_MAX — and it is REPORTED, never silent. A lot
   over the cap places the cap and says how many it left out and where to get
   them. Nothing is ever stacked quietly. */

import { CARD_W } from './lineageGraph.js';
import {
  layoutBoxes, layoutImageNodes, nextGroupId, occupiedBox,
} from './canvasImageGroups.js';
import { groupBarMaxHeight } from './canvasNodeChrome.js';
import {
  IMG_DEFAULT, IMG_MAX, IMG_MIN, imageNodeExtent, slideBelow, slideRight,
  spotBesideCard,
} from './canvasImageNodes.js';

/* How many pictures one click may put down. Not a technical limit — the band
   would happily hold ten times that — but an editorial one: past a few dozen
   tiles the board stops being a comparison and becomes a folder, and undoing a
   click that changed a hundred rows is a worse offer than being told to pick.
   Over the cap, the surplus is named, not dropped in silence. */
export const PIN_BATCH_MAX = 40;

// Air between two tiles, and between the tree and the band under it.
const TILE_GAP = 24;
const BAND_GAP = 56;
// How tall one column may grow before the next image starts a new one. Six
// tiles is roughly one screen at a readable zoom; beyond that a column stops
// being scannable.
const COLUMN_ROWS = 6;

/**
 * How big the tiles of a lot are.
 *
 * A lot is looked at as a WHOLE — that is why it is pinned in one gesture — so
 * the size follows the count: a pair of renders at full size, thirty as a
 * contact sheet you can actually take in. Every value stays inside the bounds
 * the node itself enforces (IMG_MIN/IMG_MAX), and each tile is still resizable
 * afterwards like any other.
 */
export function batchTileSize(count) {
  const n = Number(count) || 0;
  const size = n <= 4 ? IMG_DEFAULT
    : n <= 12 ? 240
      : n <= 24 ? 192
        : 160;
  return Math.min(IMG_MAX, Math.max(IMG_MIN, size));
}

const rectsOverlap = (a, b) => (
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
);

/**
 * Grouping turns several independently placed squares into a WIDE rendered
 * strip whose title bar also occupies space above it. The source-column placer
 * cannot see that future footprint: after its sixth row wraps, a new strip's
 * anchor may sit exactly under the right half of an earlier strip.
 *
 * Keep every automatic group at its preferred x, then move only a conflicting
 * anchor DOWN until its real `occupiedBox` is free. Down is deliberate: the
 * batch band already sits below the lineage tree, so it preserves checkpoint
 * order and cannot cross back into the cards. Existing nodes and fresh singles
 * are reserved too. Only fresh rows are changed; no persisted/manual group is
 * ever reflowed by a later Pin all gesture.
 */
function separateFreshGroupFootprints(nodes, rows, groupIds, graph = null) {
  const targets = groupIds instanceof Set ? groupIds : new Set(groupIds || []);
  if (!targets.size) return rows;
  const freshIds = new Set((rows || []).map((row) => Number(row.imageId)));
  const existing = (nodes || []).filter((node) => node?.visible !== false
    && !freshIds.has(Number(node?.imageId)));
  // Cards/pills are not image nodes and cannot be reconstructed here. The
  // caller therefore passes the lane graph it actually placed against; this is
  // load-bearing for remembered positions, which can each be free while the
  // WIDE strip they are about to form reaches back across a card.
  const reserved = [
    ...boardObstacles(graph, []),
    ...layoutImageNodes(existing).map(occupiedBox),
  ];
  const layout = layoutImageNodes(rows || []);

  // Singles keep the exact spots the placer/remembered geometry chose. Groups
  // move around them, never the other way round.
  for (const row of layout) {
    if (row.kind !== 'group' || !targets.has(row.groupId)) reserved.push(occupiedBox(row));
  }

  const groups = layout.filter((row) => row.kind === 'group' && targets.has(row.groupId))
    .sort((a, b) => a.y - b.y || a.x - b.x
      || (a.members[0]?.node.imageId ?? 0) - (b.members[0]?.node.imageId ?? 0));
  const rowById = new Map((rows || []).map((row) => [Number(row.imageId), row]));
  for (const group of groups) {
    const anchor = rowById.get(Number(group.members[0]?.node.imageId));
    if (!anchor) continue;
    let footprint = occupiedBox(group);
    while (true) {
      const hits = reserved.filter((box) => rectsOverlap(footprint, box));
      if (!hits.length) break;
      const nextTop = Math.max(...hits.map((box) => box.y + box.h)) + TILE_GAP;
      const dy = nextTop - footprint.y;
      anchor.y += dy;
      footprint = { ...footprint, y: footprint.y + dy };
    }
    reserved.push(footprint);
  }
  return rows;
}

/**
 * Everything already occupying the lane, as plain rectangles.
 *
 * A run card and its wrapped checkpoint pills are ONE rectangle: `cellH` is the
 * cell height the layout computed (card + pill block) and the pills never wrap
 * past CARD_W, so the cell box contains them. Publishing two overlapping
 * obstacles for one card would be harmless but would make the tests lie about
 * what they check.
 *
 * CLOSED pictures are not obstacles: they are not on the board.
 */
export function boardObstacles(graph, imageNodes) {
  const out = [];
  for (const n of (graph?.nodes || [])) {
    out.push({ x: n.x, y: n.y, w: CARD_W, h: n.cellH });
  }
  for (const n of (imageNodes || [])) {
    if (n?.visible === false) continue;
    out.push({ x: n.x, y: n.y, w: n.w, h: n.h });
  }
  return out;
}

/** The source checkpoint of an image, as a stable key. One BAND COLUMN each —
 *  which is a question about where a picture came from on the tree, not about
 *  which click produced it. */
const sourceKey = (img) => `${img?.record_id ?? '?'}:${img?.step ?? '?'}`;

/**
 * WHICH CHECKPOINT made a picture — the identity two grids must never share.
 *
 * `record_id` names the LoRA (the training run, and so the dataset it was
 * trained on) and `step` names the epoch inside it. Together they are ONE
 * CHECKPOINT, and one checkpoint is one grid. Two LoRAs never share a strip,
 * two datasets never share one, and two epochs of the same LoRA do not share
 * one either. Everything that checkpoint ever rendered goes in that strip: a
 * second launch fired at the same epoch JOINS the grid its predecessor
 * started, and so does a second prompt.
 *
 * This is the same identity the BAND already places by (`sourceKey` above),
 * which is the point — a grid now fills exactly one band column instead of
 * straddling several.
 *
 * ── What this key used to be, and why it isn't ───────────────────────────────
 *
 * Until 2026-08-21 it was `run_id + prompt + record_id`, with `step`
 * deliberately excluded so that the epochs of one LoRA formed a single
 * epoch-comparison strip. Both halves of that are reversed here, on report:
 *
 *   • keyed by LAUNCH, two batches fired at the same checkpoint could never
 *     meet. The ordinary case is one picture per checkpoint per launch — and a
 *     grid of one is not a grid — so a board generated on over several
 *     clicks filled with loose tiles that never coalesced. That is the bug
 *     this key exists to fix: generations of one epoch must accumulate.
 *   • the epoch comparison did not actually live in the strip's MEMBERSHIP. It
 *     lives in the PLACEMENT, which has always been one column per checkpoint
 *     in training order (see the band, below), and in `byTrainingOrder`. Epochs
 *     read side by side as adjacent columns; what the shared strip added was a
 *     grid whose members came from different checkpoints, which is precisely
 *     what "separate the epochs" asks it not to do.
 *
 * `run_id` and `prompt` are therefore no longer read at all, which also
 * retires the pre-backfill special case: an image with no run id is not a
 * different kind of image any more, it is just a picture from a checkpoint.
 *
 * A picture missing either half of the identity has NO key and stays a loose
 * tile, rather than being fused into somebody else's grid on a guess.
 */
export function imageBatchKey(value) {
  const image = value?.image || value;
  if (image?.record_id == null || image?.step == null) return null;
  return `ckpt:${String(image.record_id)}:${String(image.step)}`;
}

/**
 * TRAINING order: the step that made the picture, ascending. The one order a
 * LoRA's strip is allowed to have — reading epoch 500 next to epoch 2000 is
 * the entire reason the pictures are side by side — and inside one epoch the
 * tie falls through to the image id, the order the pictures were made.
 *
 * It was `${record_id}:${step}` compared as TEXT, which sorts step 1000 before
 * step 500 and put a four-checkpoint lot on the board shuffled. Steps are
 * numbers here, and a missing one sorts last rather than colliding with 0.
 */
export function byTrainingOrder(a, b) {
  const imageA = a?.image || a;
  const imageB = b?.image || b;
  const rawA = Number(imageA?.step);
  const rawB = Number(imageB?.step);
  const stepA = Number.isFinite(rawA) ? rawA : Infinity;
  const stepB = Number.isFinite(rawB) ? rawB : Infinity;
  if (stepA !== stepB) return stepA - stepB;
  const recA = Number(imageA?.record_id) || 0;
  const recB = Number(imageB?.record_id) || 0;
  if (recA !== recB) return recA - recB;
  return (Number(a?.id ?? a?.imageId) || 0) - (Number(b?.id ?? b?.imageId) || 0);
}

/** Turn freshly pinned images into (or append them to) one strip per CHECKPOINT
 * — LoRA + epoch, `imageBatchKey`.
 *
 * This is the ONE grouping path. A picture pinned alone from a gallery and a
 * whole lot dropped by 📌 Pin all run through the same function and get the
 * same answer, because they are the same question: this picture came from that
 * checkpoint, so it belongs in that checkpoint's grid. Pin all used to have its
 * own grouper that only ever looked at the lot it was placing, so a second
 * batch fired at the same checkpoint parked a rival grid beside the first
 * instead of joining it — the whole reason there is now one function.
 *
 * A grid is joined whether it is an existing homogeneous strip or a single
 * loose tile from the same checkpoint waiting to become one. Manual mixed
 * groups are never reused. The undo snapshot covers both the new images and
 * any existing member whose membership is rewritten. */
export function groupPinnedBatchBySource({ nodes = [], placed = [], graph = null } = {}) {
  const before = new Map((nodes || []).filter((n) => n?.imageId != null)
    .map((n) => [Number(n.imageId), { ...n }]));
  const working = new Map([...before].map(([id, n]) => [id, { ...n }]));
  const fresh = [...(placed || [])].filter((p) => p?.imageId != null)
    .sort(byTrainingOrder);

  for (const p of fresh) {
    const id = Number(p.imageId);
    const old = before.get(id);
    working.set(id, {
      imageId: id, x: p.x, y: p.y, w: p.w, h: p.h, visible: true,
      groupId: old?.groupId ?? null, groupPos: old?.groupPos ?? null,
      image: p.image || old?.image,
    });
  }

  const affected = new Map();
  const undo = new Map();
  const gridGroupIds = new Set();
  const remember = (node) => {
    const id = Number(node.imageId);
    if (!undo.has(id)) {
      const old = before.get(id);
      undo.set(id, old ? { ...old } : {
        ...node, visible: false, groupId: null, groupPos: null,
      });
    }
    affected.set(id, node);
  };

  const freshByBatch = new Map();
  for (const p of fresh) {
    const node = working.get(Number(p.imageId));
    const key = imageBatchKey(node);
    if (!key) continue;
    if (!freshByBatch.has(key)) freshByBatch.set(key, []);
    freshByBatch.get(key).push(node);
  }

  for (const [key, additions] of [...freshByBatch].sort(([a], [b]) => a.localeCompare(b))) {
    const originalVisible = [...before.values()].filter((n) => n.visible !== false);
    const groupIds = [...new Set(originalVisible.map((n) => n.groupId).filter(Boolean))].sort();
    let members = null;
    let groupId = null;
    for (const gid of groupIds) {
      const list = originalVisible.filter((n) => n.groupId === gid)
        .sort((a, b) => (a.groupPos ?? 0) - (b.groupPos ?? 0) || a.imageId - b.imageId);
      if (list.length >= 2 && list.every((n) => imageBatchKey(n) === key)) {
        members = [...list, ...additions.filter((n) => !list.some((m) => m.imageId === n.imageId))];
        groupId = gid;
        break;
      }
    }
    if (!members) {
      const anchor = originalVisible
        .filter((n) => !n.groupId && imageBatchKey(n) === key)
        .sort(byTrainingOrder)[0];
      members = [...(anchor ? [anchor] : []), ...additions]
        .filter((n, i, all) => all.findIndex((m) => m.imageId === n.imageId) === i);
      if (members.length < 2) continue;
      groupId = nextGroupId([...working.values()], members[0].imageId);
    }
    // Every grid this function forms is an automatic Canvas grid, so every one
    // of them takes part in the footprint reflow below. Manual mixed groups
    // still do not (the homogeneous-group check above refuses to reuse those).
    // The whole current membership is in `affected`, so moving its anchor
    // remains one reversible/undoable write.
    gridGroupIds.add(groupId);
    // ALWAYS training order, even when the picture that just arrived belongs
    // earlier than everything already in the strip: pinning epoch 500 after
    // epoch 2000 must not put 500 on the right-hand end. Nothing is lost by
    // re-sorting — an automatic strip's order is never hand-set (dragging a
    // member around INSIDE its strip is a no-op, see canvasImageGroups).
    members.sort(byTrainingOrder);
    members.forEach((member, pos) => {
      const updated = { ...working.get(member.imageId), groupId, groupPos: pos };
      working.set(member.imageId, updated);
      remember(updated);
    });
  }

  for (const p of fresh) remember(working.get(Number(p.imageId)));
  const byId = (a, b) => a.imageId - b.imageId;
  const rows = [...affected.values()].sort(byId);
  separateFreshGroupFootprints(nodes, rows, gridGroupIds, graph);
  return { rows, undoRows: [...undo.values()].sort(byId) };
}

/**
 * Where a source's column WANTS to start, horizontally: the x of the card that
 * produced it. An image whose run is not on the board (deleted, filtered off)
 * anchors past the right edge instead of being refused — losing the picture
 * would be a worse answer than losing the alignment.
 */
function anchorX(graph, recordId) {
  const nodes = graph?.nodes || [];
  const card = nodes.find((n) => n.node?.record_id === recordId);
  if (card) return card.x;
  let maxX = 0;
  for (const n of nodes) maxX = Math.max(maxX, n.x + CARD_W);
  return maxX + TILE_GAP;
}

/**
 * Split a lot into what still has to be pinned and what is already on the board.
 *
 * An image the user can already see is NOT part of the lot: re-placing it would
 * move a picture they positioned themselves, and adding a second node for it
 * would be a duplicate of the same file. A CLOSED one is part of the lot again —
 * closing is "not now", not "never".
 */
export function pinBatchPending(images, laneMap) {
  const map = laneMap || {};
  const pending = [];
  let already = 0;
  for (const img of (Array.isArray(images) ? images : [])) {
    if (img?.id == null) continue;
    if (map[img.id]?.visible) { already += 1; continue; }
    pending.push(img);
  }
  return { pending, already };
}

/**
 * The same question across the WHOLE board: of the lot a run produced, how much
 * is not on the board yet?
 *
 * This is what the button counts, and it is why the button can disappear: a lot
 * whose every picture is already pinned has nothing to offer, and a button that
 * stays lit doing nothing is worse than no button at all.
 *
 * `candidates` are {id, datasetId} (utils/canvasRunResults.runPinCandidates);
 * `lanes` is the board's {datasetId: {imageId: node}} map.
 */
export function pinBatchPendingAcrossLanes(candidates, lanes) {
  const byLane = lanes || {};
  const pending = [];
  let already = 0;
  for (const c of (Array.isArray(candidates) ? candidates : [])) {
    if (c?.id == null) continue;
    if (byLane[c.datasetId]?.[c.id]?.visible) { already += 1; continue; }
    pending.push(c);
  }
  return { pending, already };
}

/**
 * Place a lot of images on one lane.
 *
 * `graph`      — the lane's graph AS DISPLAYED (applyPlacement's output), so the
 *                obstacles are where the user sees them, not where the automatic
 *                tree would have put them.
 * `existing`   — the lane's visible image nodes.
 * `images`     — the lot, each {id, record_id, step, …}. Already-visible ones
 *                must have been filtered out by pinBatchPending first.
 * `remembered` — the lane's full node map, so a picture that was CLOSED can come
 *                back where it was closed (the promise of canvasImageNodes) —
 *                but only when that spot is free. "Nothing overlaps" outranks
 *                "everything remembers": a remembered spot that would land on
 *                something goes into the band like a fresh one.
 *
 * Returns {size, placed, skipped}. `skipped` is never empty in silence — the
 * caller announces it (pinBatchAnnouncement).
 */
export function placeImageBatch({ graph, existing, images, remembered, max,
                                  beside = false } = {}) {
  const cap = Number.isFinite(max) ? Math.max(0, max) : PIN_BATCH_MAX;
  const all = (Array.isArray(images) ? images : []).filter((i) => i?.id != null);

  // Deterministic order FIRST, cap SECOND: which images are refused must not
  // depend on the order the API happened to list the run's cells in. Training
  // order, so an over-cap lot keeps the EARLY epochs and drops the tail —
  // "which checkpoint starts working" is the question the band is read for.
  const ordered = [...all].sort(byTrainingOrder);
  const taking = ordered.slice(0, cap);
  const skipped = ordered.slice(cap).map((image) => ({ image, reason: 'over-cap' }));

  const size = batchTileSize(taking.length);
  const occupied = boardObstacles(graph, existing);
  const placed = [];

  // Pass 1 — the pictures that were closed and remember a free spot. Done first
  // so the band, computed next, is placed below them too.
  const bandBound = new Set();
  for (const image of taking) {
    const known = (remembered || {})[image.id];
    if (!known || known.visible) continue;
    const box = { x: known.x, y: known.y, w: known.w, h: known.h };
    if (![box.x, box.y, box.w, box.h].every((v) => Number.isFinite(v))) continue;
    if (occupied.some((o) => rectsOverlap(box, o))) continue;
    occupied.push(box);
    placed.push({ imageId: image.id, ...box, image });
    bandBound.add(image.id);
  }

  const band = taking.filter((i) => !bandBound.has(i.id));
  if (band.length) {
    /* The band starts clear of EVERYTHING already on the lane. That single
       measurement is what makes "no overlap" structural rather than searched
       for — the only question is on which axis.

       BELOW, for a fresh 📌 batch: the pictures read as a contact sheet hung
       under the lineage that produced them, which is what they are.

       BESIDE, for ✦ Tidy up: a lane stacks under the one above it, so anything
       a tidied lane reaches DOWN is room the next dataset must be pushed away
       by. Measured on a 150-unit tree, one pinned picture cost 518 units of
       reserved board and two cost 862 — under every lane that had ever pinned
       anything, pressed or not. Sideways costs nothing: the board pans that
       way and no lane owns the space to its right. */
    let bandTop = 0;
    let bandLeft = 0;
    for (const o of occupied) {
      if (beside) bandLeft = Math.max(bandLeft, o.x + o.w);
      else bandTop = Math.max(bandTop, o.y + o.h);
    }
    if (beside) bandLeft += BAND_GAP;
    // A picture that knows its checkpoint will become part of a Canvas group
    // immediately after this placement — either with the rest of its lot, or by
    // joining the grid that checkpoint already has on the board. Reserve the
    // TALLEST possible group bar now: it is drawn above the strip, and without
    // this allowance the first grid can cover the lineage card the band was
    // placed below. Reserving for a bar that never appears costs air; not
    // reserving for one that does costs an overlap, so this errs on air.
    const futureGroupBar = band.some((image) => imageBatchKey(image) != null)
      ? groupBarMaxHeight(size) : 0;
    bandTop += BAND_GAP + futureGroupBar;

    const colW = size + TILE_GAP;
    // The same reservation between source rows prevents the bar of grid N+1
    // from climbing into grid N before the final cross-column reflow above.
    const rowH = size + TILE_GAP + futureGroupBar;
    /* How tall a column may get before the next one starts. Six for a 📌 batch,
       which is what a contact sheet under a lineage should read like.
       For ✦ Tidy up it is however many rows fit BESIDE the tree, because a
       seventh row there would make the lane taller than its own content and
       push the dataset below it away by that much. One row minimum: a picture
       taller than the whole tree still has to land somewhere. */
    const rowsPerColumn = beside
      ? Math.max(1, Math.floor((Number(graph?.height) || 0) / rowH))
      : COLUMN_ROWS;

    // One group per source checkpoint, ordered by where that source sits on the
    // board (left to right, then top to bottom) so the band reads in the same
    // order as the tree above it.
    const groups = new Map();
    for (const image of band) {
      const key = sourceKey(image);
      if (!groups.has(key)) {
        groups.set(key, { key, first: image, ax: anchorX(graph, image.record_id), images: [] });
      }
      groups.get(key).images.push(image);
    }
    // Columns read left to right in TRAINING order when two checkpoints anchor
    // at the same place — a tie broken by the alphabet put step 1000 left of
    // step 500.
    const ordering = [...groups.values()].sort((a, b) => (a.ax - b.ax)
      || byTrainingOrder(a.first, b.first));

    // Columns are RESERVED before they are filled: a group takes the first free
    // column at or right of the one its own source sits over. Two runs at the
    // same depth therefore get neighbouring columns instead of the same one.
    const usedColumns = new Set();
    const takeColumn = (preferred) => {
      let c = Math.max(0, preferred);
      while (usedColumns.has(c)) c += 1;
      usedColumns.add(c);
      return c;
    };

    for (const group of ordering) {
      const preferred = Math.max(0, Math.round(group.ax / colW));
      let col = takeColumn(preferred);
      let row = 0;
      for (const image of group.images.sort(byTrainingOrder)) {
        if (row >= rowsPerColumn) { col = takeColumn(col + 1); row = 0; }
        const box = { x: bandLeft + col * colW, y: bandTop + row * rowH,
          w: size, h: size };
        occupied.push(box);
        placed.push({ imageId: image.id, ...box, image });
        row += 1;
      }
    }
  }

  placed.sort(byTrainingOrder);
  return { size, placed, skipped };
}

/**
 * ✦ Tidy up, for the side-by-side STRIPS of one lane: bring each one back
 * beside the run that made it, WITHOUT taking it apart.
 *
 * ── Why strips move at all now ───────────────────────────────────────────────
 * They used to be left exactly where they were, and the argument was sound
 * while a picture could not leave its lane: a strip is something the user
 * assembled on purpose, and re-flowing its members one by one would not tidy it,
 * it would dismantle it. Free placement changes the stakes, not the argument.
 * A strip can now be parked thousands of units above or left of everything, and
 * "the button that rebuilds the board" leaving it out there would mean a whole
 * assembled comparison with no way back short of finding it by hand at 10 %
 * zoom. So the strip comes back — as ONE object, which is the thing the old
 * argument was actually protecting.
 *
 * ── How ──────────────────────────────────────────────────────────────────────
 * Only the ANCHOR's row is written. The strip is DERIVED from the anchor's spot
 * (utils/canvasImageGroups.layoutImageNodes), so moving the anchor moves the
 * band, and every member keeps the geometry it gets back the day it is dragged
 * out. The group fields are deliberately NOT in the returned rows either: a
 * write that only mentions geometry can never dissolve a group, so a half-failed
 * tidy leaves a strip moved, never broken.
 *
 * Placed through the SAME rule a hand-dropped pin uses — beside its own card,
 * sliding down past whatever is already there (canvasImageNodes.spotBesideCard
 * / slideBelow) — measured on `occupiedBox`, so the strip's drag BAR is
 * reserved too and the contact-sheet band placed afterwards cannot land on it.
 *
 * `taken` are extra boxes already spoken for; the lane's own run cards are added
 * here rather than asked for, so a caller cannot forget them and drop a strip on
 * a card. Returns { rows, boxes }: the rows to persist, and the footprints the
 * STRIPS now occupy — exactly the shape `placeImageBatch` wants as `existing`,
 * so the contact-sheet band placed next cannot disagree about what is free.
 * (The cards are not in `boxes`: `placeImageBatch` reads those off the graph
 * itself, and publishing them twice would only make its tests lie.)
 */
export function tidyGroupRows({ graph, layout, taken } = {}) {
  const strips = (layout || []).filter((r) => r?.kind === 'group' && r.members?.length);
  const boxes = [];
  if (!strips.length) return { rows: [], boxes };
  const cards = graph?.nodes || [];
  const busy = [...(taken || [])];
  for (const n of cards) busy.push({ x: n.x, y: n.y, w: CARD_W, h: n.cellH });

  // Deterministic order: where the source card sits on the tree, top to bottom
  // then left to right, tie-broken by the anchor's image id. Two strips must
  // never swap places just because the API listed their rows the other way
  // round — the same reason applyPlacement sorts its arrivals.
  const cardOf = (row) => cards.find(
    (c) => c.node?.record_id === row.members[0].node.image?.record_id);
  const ordered = [...strips].sort((a, b) => {
    const ca = cardOf(a);
    const cb = cardOf(b);
    return ((ca?.y ?? Infinity) - (cb?.y ?? Infinity))
      || ((ca?.x ?? Infinity) - (cb?.x ?? Infinity))
      || (a.members[0].node.imageId - b.members[0].node.imageId);
  });

  const rows = [];
  for (const strip of ordered) {
    const anchor = strip.members[0].node;
    const footprint = occupiedBox(strip);
    // What occupiedBox reserved ABOVE the pictures for the group's drag bar.
    const bar = footprint.h - strip.h;
    const at = spotBesideCard(graph, anchor.image?.record_id);
    // Sideways, not down: see canvasImageNodes.slideRight — every unit a tidied
    // lane reaches below its tree is a unit the next dataset gets pushed away
    // by, in advance and for good.
    const spot = slideRight({ ...at, w: footprint.w, h: footprint.h }, busy);
    const landed = { x: spot.x, y: spot.y, w: footprint.w, h: footprint.h };
    busy.push(landed);
    boxes.push(landed);
    rows.push({ imageId: anchor.imageId, x: spot.x, y: spot.y + bar,
      w: anchor.w, h: anchor.h });
  }
  return { rows, boxes };
}

/**
 * ✦ Tidy up for a WHOLE lane: the strips first, then the loose pictures in the
 * contact-sheet band below them. Returns `{ rows, boxes }` — the geometry to
 * persist, and every footprint the tidied lane occupies.
 *
 * Extracted so there is exactly ONE answer to "where does this lane's content
 * land when the board is rebuilt". It has two callers that must never disagree:
 * the button itself, and the LANE STACK, which has to reserve that much room
 * before the button is ever pressed (see `tidyLaneReach`).
 *
 * ⚠️ POSITION-INDEPENDENT, and that is load-bearing. Nothing here reads where a
 * picture currently sits — only the lane's tree, the strips' membership and the
 * pictures' SIZES. So dragging a render anywhere on the board cannot change
 * what this returns, which is what lets the stack reserve it without the lane
 * below jumping under the hand still dragging.
 */
export function tidyLaneRows({ graph, nodes } = {}) {
  const visible = (Array.isArray(nodes) ? nodes : []).filter((n) => n?.visible !== false);
  const strips = tidyGroupRows({ graph, layout: layoutImageNodes(visible) });
  const rows = [...strips.rows];
  const boxes = [...strips.boxes];
  const loose = visible.filter((n) => !n.groupId);
  if (loose.length) {
    const res = placeImageBatch({
      graph,
      beside: true,   // ✦ Tidy up spends width, never the next dataset's room
      // Nothing may land ON one of those strips either — nor on the BAR above
      // one, which is the group's only grip and carries its ✕. These are the
      // footprints the strips ended up on, so the two passes cannot disagree.
      existing: strips.boxes,
      images: loose.map((n) => ({ id: n.imageId,
        record_id: n.image?.record_id, step: n.image?.step })),
      max: loose.length,
    });
    for (const p of res.placed) {
      rows.push({ imageId: p.imageId, x: p.x, y: p.y, w: p.w, h: p.h });
      boxes.push({ x: p.x, y: p.y, w: p.w, h: p.h });
    }
  }
  return { rows, boxes };
}

/**
 * How far down a lane's content reaches once it is tidied — the room the STACK
 * has to leave under the tree.
 *
 * ── The bug this exists for ──────────────────────────────────────────────────
 * A lane's stacking height is its tree and nothing else, so that a picture
 * dragged below its dataset stops shoving every lane underneath it down the
 * board. Right for a DRAG, wrong for ✦ Tidy up: the tidy layout puts the strips
 * and the contact-sheet band BELOW the tree, often thousands of units below it,
 * and a stack that reserves only the tree starts the next dataset straight
 * through them. On a loaded board that reads as strips piled on strips and on
 * other datasets' run cards — nobody dragged anything, so nobody asked for it.
 *
 * Reserving the tidy reach instead of the CURRENT reach keeps both promises: a
 * drag still moves nothing (this number does not depend on where anything sits),
 * and the layout the button produces has room to exist. Resizing a picture does
 * move it, which is correct — a bigger strip genuinely needs more room.
 */
export function tidyLaneReach({ graph, nodes } = {}) {
  const { boxes } = tidyLaneRows({ graph, nodes });
  let bottom = 0;
  for (const b of boxes) bottom = Math.max(bottom, (Number(b?.y) || 0) + (Number(b?.h) || 0));
  return bottom;
}

/**
 * The entries the LANE STACK is built from (utils/canvasLayout.stackLanes) —
 * one per placed dataset, carrying the two extents that function keeps apart.
 *
 * Lives here, out of the component, for one reason: this is where the promise
 * "a drag moves no lane" is either kept or quietly lost, and `node --test`
 * cannot parse JSX. The wiring below is the whole fix, so the wiring is what
 * the test drives.
 *
 * ── The two lists, and why they must NOT be the same one ─────────────────────
 * `layoutByLane` is what the lane DRAWS this frame: the picture under the hand
 * sits where the hand is, at the size it is being given, pulled out of its
 * strip if it is on its way out. The REACH (minX/minY/maxX/maxY) is measured on
 * it, and must be — ✦ Fit, 📷 Export and the pan clamp all have to reach a
 * picture while it is still moving.
 *
 * `restingByLane` is the lane's COMMITTED rows — the board as it would reload.
 * The stacking height is measured on it, and only on it. `tidyLaneReach` was
 * written to be position-independent, and it is; but it is not, and cannot be,
 * gesture-independent, because ✦ Tidy up's layout depends on how many strips a
 * lane has, how many members each one holds, and how many loose pictures go in
 * the contact-sheet band — and a drag changes all three, live. Measured on a
 * three-member strip whose anchor is dragged out: the reserve went 576 → 1143
 * mid-gesture, shoving the next dataset 567 units down the board while the
 * hand was still moving. Pulling the drag OUT of this input is what freezes it.
 *
 * ⚠️ The lane still settles ONCE on release, and that is not the bug: a picture
 * that has genuinely left its strip is a membership change, so the lane really
 * does need a different amount of room for the button to have somewhere to put
 * it. What nothing may do is move a lane while the gesture is still in flight.
 *
 * ── …and the same rule for the RUN CARDS ────────────────────────────────────
 * `stackPlaced` carries each lane's AUTOMATIC tree — the layout the lane has
 * before anybody moved anything — and the stacking height is read from there.
 *
 * Freezing it for the duration of the gesture was not enough, and the second
 * report says why: "dragging a dataset node down makes the ones below come down
 * too, which creates space for nothing". A card dropped 800 units low left its
 * lane permanently that much taller, so the next dataset sat 800 units lower
 * FOREVER, with dead board between them. Measuring the stack on the arrangement
 * at all is what puts one lane's layout in charge of another lane's position.
 *
 * So: lanes sit where the automatic layout says, and moving a card moves
 * nothing but that card. The cost is stated rather than hidden — a lane
 * arranged taller than its automatic tree can overlap the lane below. That is
 * the same bargain pinned pictures already make, and ✦ Tidy up (which clears
 * the arrangement) is the way back.
 *
 * The moved card still has to be REACHED — ✦ Fit and 📷 Export must frame it —
 * so the live graph's height feeds `maxY` instead, where it grows the board's
 * box without ever advancing the stack. Omit `stackPlaced` and this behaves
 * exactly as it always did: the live graph is used for both.
 */
export function laneStackEntries({ placed, layoutByLane, restingByLane,
                                   stackPlaced } = {}) {
  const stackGraphs = new Map(
    (Array.isArray(stackPlaced) ? stackPlaced : [])
      .map((e) => [e?.datasetId, e?.graph]));
  return (Array.isArray(placed) ? placed : []).map((e) => {
    const ext = imageNodeExtent(layoutBoxes(layoutByLane?.[e.datasetId] || []));
    const tidy = tidyLaneReach({ graph: e.graph, nodes: restingByLane?.[e.datasetId] || [] });
    const stackGraph = stackGraphs.has(e.datasetId)
      ? stackGraphs.get(e.datasetId) : e.graph;
    const liveHeight = e.graph?.height || 0;
    return {
      ...e,
      width: e.graph?.width || 0,
      height: Math.max(stackGraph?.height || 0, tidy),
      minX: ext.minX,
      minY: ext.minY,
      maxX: ext.width,
      maxY: Math.max(ext.height, liveHeight),
    };
  });
}

/** The button's own words. It must say HOW MANY it is about to put down —
 *  "Pin all" alone gives no idea whether one click adds two pictures or thirty. */
export function pinBatchLabel(count) {
  const n = Number(count) || 0;
  if (n <= 0) return '';
  if (n === 1) return '📌 Pin this image to the board';
  return `📌 Pin all ${n} to the board`;
}

/**
 * What is said out loud once the click has happened (aria-live, and on screen).
 *
 * A bulk action that reports nothing is a bulk action you have to go and audit.
 * The refused images are named WITH the way to get them — a count on its own
 * ("7 left out") is a dead end.
 */
export function pinBatchAnnouncement(result) {
  const placed = result?.placed?.length || 0;
  const left = result?.skipped?.length || 0;
  if (!placed && !left) return 'Nothing to pin — these images are already on the board.';
  const head = placed === 1 ? '1 image pinned' : `${placed} images pinned`;
  if (!left) return `${head} to the board.`;
  return `${head} to the board — ${left} left out (${PIN_BATCH_MAX} at a time). `
    + 'Pin the rest from the checkpoint gallery.';
}
