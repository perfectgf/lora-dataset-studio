/* 🔌 EXTERNAL-LORA PROVENANCE — which board images an external plugin node
   contributed to. Sibling of canvasBlendEdges.js and drawn in the SAME
   board-level layer: a 🔌 node lives in WORLD coordinates already (it is
   positioned absolutely in the pan/zoom layer, not inside a lane), so only
   the image side needs the lane offset. Matching is by normalized filename:
   the node stores the picker string, the cell stores the validated name —
   separators and case may differ, identity may not. JSX-free for node --test. */

import { edgePath } from './lineageGraph.js';
import { PLUGIN_NODE_TYPES, normalizeLoraName, nodeKey } from '../components/canvas/pluginNodes/registry.js';

/** The world-space X where an edge should touch a node's box, per the
    type's declared anchor side (`registry.js`'s `edge.side`) — 'left'
    anchors at the box's left edge, anything else (including no config)
    anchors at the right, the historical default. */
function anchorX(box, side) {
  return side === 'left' ? box.x : box.x + box.w;
}

/** External entries recorded on an image, or [] — never a throw. */
export function externalMembersOf(image) {
  const raw = image?.extra_loras;
  if (!raw) return [];
  let list = raw;
  if (typeof raw === 'string') {
    try { list = JSON.parse(raw); } catch { return []; }
  }
  if (!Array.isArray(list)) return [];
  return list.filter((e) => e && typeof e === 'object' && e.external);
}

/** One edge per (placed 🔌 node → image generated with it), world coords. */
export function externalEdgesFor(imageNodes, lanes, extNodes, boxByKey) {
  const byName = new Map((extNodes || [])
    .map((n) => [normalizeLoraName(n.filename), n]).filter(([k]) => k));
  const edges = [];
  for (const n of (imageNodes || [])) {
    const members = externalMembersOf(n.image);
    if (!members.length) continue;
    const lane = (lanes || []).find((l) => Number(l.datasetId) === Number(n.datasetId));
    const x2 = (lane?.x || 0) + n.x;
    const y2 = (lane?.graphY || 0) + n.y + n.h / 2;
    for (const m of members) {
      const node = byName.get(normalizeLoraName(m.filename));
      if (!node) continue;                       // node removed from the board
      const key = nodeKey('external-lora', node);
      const box = boxByKey?.get?.(key);
      if (!box) continue;                        // not measured yet this frame
      const x1 = anchorX(box, PLUGIN_NODE_TYPES['external-lora']?.edge?.side);
      const y1 = box.y + box.h / 2;
      edges.push({ parentId: key, childId: `img:${n.imageId}`,
        x1, y1, x2, y2, d: edgePath(x1, y1, x2, y2), external: true });
    }
  }
  return edges;
}
