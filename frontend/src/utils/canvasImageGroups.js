/* 🖼🖼 Grouped pinned images on the ◉ LoRA Canvas — the arithmetic.

   Asked for: "drop one image node onto another and they become ONE node, with
   the pictures side by side and NO border between them; no limit on how many I
   add; and to take one out I just drag it outside the node."

   ── What a group IS (the decision everything else hangs off) ────────────────
   A group is NOT a new kind of node holding N pictures. It is a MEMBERSHIP
   carried by pictures that stay ordinary nodes: one row each, exactly as
   before, plus `groupId` and `groupPos`. Three reasons, in order of weight:

     • the promise already made. "Close an image and re-open it from its gallery
       and it comes back where and how big you left it" is a PER-IMAGE promise
       held by a per-image row (canvas_image_node, `visible: false` + geometry).
       A container node would have to keep a shadow copy of every member's
       geometry to keep that promise — the same data, in a second place, free to
       disagree. Here the remembered box IS the stored box, grouped or not;
       joining a group never touches it.
     • taking one out is then a two-field write (groupId := null), not the
       splitting of a container.
     • each picture keeps its OWN actions — 🔍, ✕, and whatever the chrome gains
       next. A container that swallowed its members would have to re-invent
       every one of them, per member, and get "which one am I closing?" right
       from scratch.

   The strip is therefore DERIVED, never stored: `layoutImageNodes` turns the
   flat list into singles and groups, and the component only draws the numbers.
   That is what makes all of this testable without a DOM — `node --test` cannot
   parse JSX, and the parts that age badly are exactly these: which node a drop
   lands on, what order the strip ends up in, what a leaver gets back, and what
   the ones staying behind keep.

   Coordinates are LANE-LOCAL world units, like everything else on this board.

   ⚠️ Both new fields are ADDITIVE, nullable, and never rename anything: a lane
   loaded from a database that predates them reads `null` everywhere and draws
   exactly the board it drew before. */

import { clampImageBox } from './canvasImageNodes.js';

// Explicit extension: imported by `node --test`, which does not resolve
// extensionless specifiers, as well as by Vite.
import { groupBarMaxHeight } from './canvasNodeChrome.js';

/** Sort key inside a strip: the stored position, then the image id so a tie
 *  (two rows written by different gestures in the same millisecond, or a hand-
 *  edited database) still draws the same order on every load. */
const memberOrder = (a, b) => (a.groupPos ?? 0) - (b.groupPos ?? 0)
  || a.imageId - b.imageId;

/** The members of one group, in strip order. */
export function groupMembers(nodes, groupId) {
  if (!groupId) return [];
  return (nodes || []).filter((n) => n.groupId === groupId).sort(memberOrder);
}

/** Every group present in a list, as {groupId: members[]} — only the real ones.
 *  A single row left carrying a group id is not a group; see layoutImageNodes. */
function groupIndex(nodes) {
  const out = new Map();
  for (const n of (nodes || [])) {
    if (!n.groupId) continue;
    if (!out.has(n.groupId)) out.set(n.groupId, []);
    out.get(n.groupId).push(n);
  }
  for (const [, list] of out) list.sort(memberOrder);
  return out;
}

/**
 * A group id that is free on this lane.
 *
 * Derived from the anchor's image id so it is deterministic (a test can name
 * it) and readable in a database dump, with a suffix when that id is already
 * taken — which happens for real: an image can anchor a group, be dragged out
 * of it, and later anchor a second one while the first still exists.
 *
 * The id is OPAQUE. It means nothing after the fact; in particular the anchor
 * leaving does not rename it.
 */
export function nextGroupId(nodes, anchorImageId) {
  const taken = new Set((nodes || []).map((n) => n.groupId).filter(Boolean));
  const base = `g${anchorImageId}`;
  if (!taken.has(base)) return base;
  for (let i = 2; i < 1000; i += 1) {
    if (!taken.has(`${base}-${i}`)) return `${base}-${i}`;
  }
  return `${base}-${Date.now()}`;
}

/** A member's aspect, from the generated image's real format when available.
 *  Older API payloads do not carry that format, so the remembered node box is
 *  kept as a backwards-compatible fallback. */
const aspectOf = (n) => {
  const match = String(n?.image?.aspect ?? '').match(
    /^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$/,
  );
  if (match) {
    const imageW = Number(match[1]);
    const imageH = Number(match[2]);
    if (Number.isFinite(imageW) && Number.isFinite(imageH) && imageW > 0 && imageH > 0) {
      return imageW / imageH;
    }
  }
  const w = Number(n?.w);
  const h = Number(n?.h);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return 1;
  return w / h;
};

const inBox = (box, p) => !!box && p.x >= box.x && p.x <= box.x + box.w
  && p.y >= box.y && p.y <= box.y + box.h;

/**
 * Turn the flat list of visible nodes into what the lane draws: `single`
 * renderables (unchanged) and `group` renderables (a strip).
 *
 * The strip:
 *   • sits at its ANCHOR's stored x/y — the anchor is the member that was
 *     dropped ON, and the thing you dropped onto must not move under you;
 *   • is as tall as the anchor, every member scaled to that height so the band
 *     is continuous. Letterboxing members to a common box would put visible
 *     dead space between two pictures, which is the border the request is
 *     explicitly about;
 *   • is laid edge to edge, gap zero. The separation between two members is
 *     drawn (on hover) by lighting ONE of them, never by a rule between them.
 *
 * Width grows without limit, on purpose — see the module note in the component.
 */
export function layoutImageNodes(nodes) {
  const groups = groupIndex(nodes);
  const out = [];
  const done = new Set();
  for (const n of (nodes || [])) {
    if (done.has(n.imageId)) continue;
    const members = n.groupId ? (groups.get(n.groupId) || []) : [];
    // A lone row still carrying a group id is a plain picture. It happens after
    // the second-to-last member is taken out, and after a database has been
    // pruned of an image that no longer exists.
    if (members.length < 2) {
      out.push({ kind: 'single', key: `img:${n.imageId}`, node: n,
        x: n.x, y: n.y, w: n.w, h: n.h });
      done.add(n.imageId);
      continue;
    }
    const anchor = members[0];
    const h = Math.max(1, Number(anchor.h) || 1);
    let x = anchor.x;
    const laid = members.map((m) => {
      const w = Math.max(1, Math.round(h * aspectOf(m)));
      const tile = { key: `img:${m.imageId}`, node: m, x, y: anchor.y, w, h };
      x += w;
      done.add(m.imageId);
      return tile;
    });
    out.push({ kind: 'group', key: `grp:${n.groupId}`, groupId: n.groupId,
      x: anchor.x, y: anchor.y, w: x - anchor.x, h, members: laid });
  }
  return out;
}

/** Flatten a layout back to plain boxes, for the lane extent (✦ Fit must not
 *  crop a strip off the board) and for anything that only needs geometry. */
export function layoutBoxes(layout) {
  return (layout || []).map(occupiedBox);
}

/**
 * The board space a layout row really OCCUPIES — which, for a group, is taller
 * than the strip: its drag bar is drawn ABOVE the box, and that bar carries the
 * group's only grip plus its ✕ and its Export grid.
 *
 * A placer that ignored this would drop a picture squarely on top of a group's
 * bar and leave it unmovable and undeletable — measured, not feared: with a
 * picture pinned flush above a two-image strip, 5 of 11 points sampled along
 * the bar hand the pointer to the picture instead. `groupBarMaxHeight` reserves
 * the bar at its zoom-out maximum, because a gap that is enough at 100 % is not
 * enough at 40 %.
 *
 * Only about PLACEMENT: what is drawn, and where, is unchanged.
 */
export function occupiedBox(row) {
  if (!row) return { x: 0, y: 0, w: 0, h: 0 };
  if (row.kind !== 'group') return { x: row.x, y: row.y, w: row.w, h: row.h };
  const bar = groupBarMaxHeight(row.h);
  return { x: row.x, y: row.y - bar, w: row.w, h: row.h + bar };
}

/** Every picture with the box it is actually DRAWN in — a member's slot in its
 *  strip, not the box it remembers. What the links back to the source
 *  checkpoints have to be computed from: an edge pointing at a member's stored
 *  x/y would point somewhere the picture is not. */
export function drawnNodes(layout) {
  const out = [];
  for (const r of (layout || [])) {
    if (r.kind === 'single') { out.push({ ...r.node, x: r.x, y: r.y, w: r.w, h: r.h }); continue; }
    for (const m of r.members) out.push({ ...m.node, x: m.x, y: m.y, w: m.w, h: m.h });
  }
  return out;
}

/**
 * The nodes the LINKS back to the source checkpoints should be drawn from —
 * which is NOT one per picture once pictures are grouped.
 *
 * A lone picture answers for itself. A STRIP answers as one object: it is one
 * thing to the eye and to every gesture (one grip, one ✕, one Export), so eight
 * lines fanning out of eight points along its band to the pills behind it is
 * eight times the ink for one fact. All of them therefore leave the strip at the
 * SAME point — its left edge, mid-height, where the anchor's own line already
 * left — so a strip reads as a single object with threads, not as a comb.
 *
 * But one line per DISTINCT SOURCE, not one line per strip. A strip is routinely
 * a whole generation run's contact sheet, so its pictures come from several
 * checkpoints of the same card (📌 Pin all groups a lot by run, across its
 * epochs). Collapsing to the anchor alone would draw one line to ONE pill and
 * quietly attribute the other seven pictures to it — a provenance board must
 * never invent an edge, and that includes the edge it leaves out. A strip whose
 * pictures really do share one checkpoint collapses to exactly one line, which
 * is the ordinary case; a mixed one says so, in as many lines as it has truths.
 *
 * The picture chosen to carry each line is the first in strip order, so the
 * result is stable and its React key cannot collide.
 */
export function edgeAnchors(layout) {
  const out = [];
  for (const r of (layout || [])) {
    if (r.kind === 'single') {
      out.push({ ...r.node, x: r.x, y: r.y, w: r.w, h: r.h });
      continue;
    }
    const seen = new Set();
    for (const m of r.members) {
      const key = `${m.node.image?.record_id ?? '?'}:${m.node.image?.step ?? '?'}`;
      if (seen.has(key)) continue;
      seen.add(key);
      // Every line leaves the STRIP, not the member's own slot.
      out.push({ ...m.node, x: r.x, y: r.y, w: r.w, h: r.h });
    }
  }
  return out;
}

/** The strip a given image is currently drawn in, or null. */
export function groupBoxOf(layout, imageId) {
  for (const r of (layout || [])) {
    if (r.kind !== 'group') continue;
    if (r.members.some((m) => m.node.imageId === imageId)) {
      return { x: r.x, y: r.y, w: r.w, h: r.h, groupId: r.groupId,
        count: r.members.length };
    }
  }
  return null;
}

/**
 * THE gesture rule: has a member been dragged far enough to leave its group?
 *
 * Not a tuned pixel threshold — the geometry of the group itself. "Drag it out
 * of the node" was the request and it is also the only rule that explains
 * itself while you are doing it: while the pointer is over the strip the
 * picture is still in it; the moment it is off the strip, it is out. Dragging
 * a member around INSIDE the strip therefore does nothing, and moving the whole
 * group is a different grip entirely (its title bar), so the two gestures can
 * never be confused with each other.
 */
export function shouldExtract(groupBox, point) {
  if (!groupBox) return false;
  return !inBox(groupBox, point || { x: 0, y: 0 });
}

/**
 * What a drop at `point` would merge into: the target picture, which side of it
 * the dragged one would land on, and how big the group would become.
 *
 * `point` is the CENTRE of the picture being dragged, not the pointer and not
 * the overlap of the two boxes. Overlap is ambiguous and near-constant on a
 * board where pictures sit close together — two nodes an inch apart overlap the
 * moment either moves — whereas the centre is unambiguous, is what "superposer"
 * means to the eye, and is also exactly what the highlight shown during the
 * drag is drawn from, so the feedback cannot promise a merge the drop refuses.
 *
 * Returns null when the drop lands on bare board — which is the ordinary move.
 */
export function mergeTargetAt(layout, draggedImageId, point) {
  for (const r of (layout || [])) {
    if (r.kind === 'single') {
      if (r.node.imageId === draggedImageId) continue;
      if (!inBox(r, point)) continue;
      const side = point.x < r.x + r.w / 2 ? 'before' : 'after';
      return { targetImageId: r.node.imageId, groupId: null, count: 2, side,
        caret: side === 'before' ? r.x : r.x + r.w,
        box: { x: r.x, y: r.y, w: r.w, h: r.h } };
    }
    if (!inBox(r, point)) continue;
    const tile = r.members.find((m) => point.x < m.x + m.w) || r.members[r.members.length - 1];
    if (tile.node.imageId === draggedImageId) continue;
    const inside = r.members.some((m) => m.node.imageId === draggedImageId);
    const side = point.x < tile.x + tile.w / 2 ? 'before' : 'after';
    return { targetImageId: tile.node.imageId, groupId: r.groupId,
      count: r.members.length + (inside ? 0 : 1), side,
      // The slot the picture would take, inside the strip — not the strip's
      // edge. "Where exactly does it land?" is half of what the feedback owes.
      caret: side === 'before' ? tile.x : tile.x + tile.w,
      box: { x: r.x, y: r.y, w: r.w, h: r.h } };
  }
  return null;
}

/** One row as the page persists it. `image` is added by the caller. */
const row = (n, over = {}) => ({
  imageId: n.imageId, x: n.x, y: n.y, w: n.w, h: n.h,
  groupId: n.groupId ?? null, groupPos: n.groupPos ?? null, ...over,
});

/**
 * Take one member out of its group and hand back the rows to write.
 *
 * The LEAVER gets its own remembered size back (that box was never touched
 * while it was in the strip — see the module note) and lands wherever it was
 * dropped. It is the thing the gesture was about; it is allowed to change.
 *
 * The ones STAYING BEHIND must not move a pixel. So when the anchor is the one
 * leaving, the strip's position and height are handed to the new anchor, its
 * own width scaled by the same factor so it keeps its shape and draws exactly
 * as wide as it did a moment ago. And when only one member would be left, the
 * group dissolves: a "group" of one is just a picture, and leaving the id on it
 * would only be a trap for the next reader.
 *
 * Returns [] when the image is in no group — nothing to write.
 */
export function extractFromGroup(nodes, imageId, dropPoint) {
  const leaver = (nodes || []).find((n) => n.imageId === imageId);
  if (!leaver?.groupId) return [];
  const members = groupMembers(nodes, leaver.groupId);
  if (members.length < 2) return [];
  const anchor = members[0];
  const rest = members.filter((m) => m.imageId !== imageId);
  const rows = [row(leaver, {
    ...clampImageBox({ x: dropPoint?.x ?? leaver.x, y: dropPoint?.y ?? leaver.y,
      w: leaver.w, h: leaver.h }),
    groupId: null, groupPos: null,
  })];
  const dissolving = rest.length < 2;
  rest.forEach((m, i) => {
    let box = { x: m.x, y: m.y, w: m.w, h: m.h };
    if (i === 0 && m.imageId !== anchor.imageId) {
      // The strip lost its anchor. Inherit its spot AND its height, scaling the
      // new anchor's own width by the same factor: same place, same size on
      // screen, nothing jumps.
      const k = (Number(anchor.h) || 1) / (Number(m.h) || 1);
      box = { x: anchor.x, y: anchor.y, w: m.w * k, h: anchor.h };
    } else if (dissolving && i === 0) {
      // The survivor keeps exactly what it drew inside the strip.
      const h = Math.max(1, Number(anchor.h) || 1);
      box = { x: anchor.x, y: anchor.y, w: Math.max(1, Math.round(h * aspectOf(m))), h };
    }
    rows.push(row(m, {
      ...clampImageBox(box),
      groupId: dissolving ? null : m.groupId,
      groupPos: dissolving ? null : i,
    }));
  });
  return rows;
}

/**
 * Merge `draggedId` into the group `targetId` belongs to (creating it when the
 * target is a lone picture), on the given side of the target.
 *
 * The dragged node's own x/y/w/h are DELIBERATELY left alone: they are what it
 * gets back the day it is taken out again, and what "re-open it where I closed
 * it" reads. Only its two group fields change.
 *
 * A node already in another group leaves that one first, through the very same
 * extraction rules — including the dissolution of a group it would leave with a
 * single member.
 */
export function mergeIntoGroup(nodes, draggedId, targetId, side = 'after') {
  const list = nodes || [];
  const dragged = list.find((n) => n.imageId === draggedId);
  const target = list.find((n) => n.imageId === targetId);
  if (!dragged || !target || draggedId === targetId) return [];

  // Leave the old group first, in place (the drop point is the node's own spot:
  // it is about to be re-parented anyway, and this way a failed second half
  // cannot strand it somewhere it never was).
  const out = new Map();
  let working = list;
  if (dragged.groupId && dragged.groupId !== target.groupId) {
    for (const r of extractFromGroup(list, draggedId, { x: dragged.x, y: dragged.y })) {
      out.set(r.imageId, r);
    }
    working = list.map((n) => (out.has(n.imageId) ? { ...n, ...out.get(n.imageId) } : n));
  }

  const freshTarget = working.find((n) => n.imageId === targetId);
  const groupId = freshTarget.groupId || nextGroupId(working, targetId);
  const members = freshTarget.groupId
    ? groupMembers(working, groupId).filter((m) => m.imageId !== draggedId)
    : [freshTarget];

  const at = members.findIndex((m) => m.imageId === targetId);
  const index = Math.max(0, side === 'before' ? at : at + 1);
  const ordered = [...members];
  ordered.splice(index, 0, working.find((n) => n.imageId === draggedId));

  ordered.forEach((m, i) => {
    const prev = out.get(m.imageId);
    out.set(m.imageId, { ...(prev || row(m)), groupId, groupPos: i });
  });
  return [...out.values()];
}
