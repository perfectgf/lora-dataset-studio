import { groupBarHeight } from '../../utils/canvasNodeChrome';
import CanvasImageNode from './CanvasImageNode';

/* 🖼🖼 Several pinned images fused into ONE node: a continuous strip, side by
   side, with nothing drawn between them.

   Drop one pinned picture onto another and they merge; drop a third, a tenth,
   there is no limit. Drag one off the strip and it is a node of its own again.

   ── What this component is NOT ──────────────────────────────────────────────
   It is not a container that owns its pictures. Each member is still the very
   same CanvasImageNode, with its own 🔍, its own ✕ and whatever the chrome
   gains next — only its frame, its resize corner and its permanent header are
   dropped, and its box comes from the strip instead of from itself (see
   utils/canvasImageGroups). A group that swallowed its members would have had
   to re-invent every one of their actions and answer "which one am I closing?"
   from scratch. Here the answer is the ordinary one: the picture you are
   pointing at lights up and shows its own buttons.

   ── The two gestures, and why they cannot be confused ───────────────────────
   Dragging a picture and dragging the node it is in are the same physical
   gesture, so they are given different GRIPS:

     • the title bar moves the whole strip. It is the only thing that does, it
       is visible at rest, it says how many pictures are here, and it carries
       the group's ✕. It is drawn by CanvasGroupBar, in a LAYER above every
       picture — living inside this component, it was a plain sibling with no
       z-index and any picture placed over that strip took its clicks, which
       left the group neither movable nor closable;
     • dragging a picture INSIDE the strip means "take this one out", and it
       only takes effect once the pointer is off the strip — which is exactly
       the way it was asked for ("je la drague en dehors du node"), and the only
       rule that explains itself while you are performing it. Let go while still
       over the strip and nothing happened.

   The bar's height is counter-scaled by the board zoom (canvasNodeChrome.
   groupBarHeight): a grip four pixels tall at 24 % would not make the gesture
   awkward, it would make the group immovable — the same bug the ✕ already had
   once on a phone. That counter-scale is also why the bar can be TWICE as tall
   at 40 % as at 100 %, and why the placers reserve its worst case rather than
   its current one (utils/canvasImageGroups.occupiedBox).

   ⚠️ Width grows without limit, on purpose. Ten pictures side by side is ten
   times as wide, and this is a board that zooms and pans, so ✦ Fit is the
   answer rather than a silent wrap onto a second row — a strip that quietly
   stopped being a strip at some threshold nobody can see would be worse than a
   wide one. */

export default function CanvasImageGroup({ group, datasetId, laneName, boardScale = 1,
  onClose, onOpen, onDelete, dropHint = null, blendNotes = null }) {
  const count = group.members.length;
  const anchorId = group.members[0]?.node.imageId;
  // The drag-out hint below is CHROME, not content: it must stay readable at
  // 24 % exactly as at 100 %, which is the one thing groupBarHeight computes.
  // Recomputed here rather than passed down, so the hint and the bar's own
  // label keep ONE source of truth for that counter-scale — the bar moved out
  // of this file and took the binding with it, which is how the hint came to
  // reference a `barH` nothing declared any more.
  const barH = groupBarHeight(boardScale, group.h);

  return (
    <div
      data-canvas-group=""
      data-dataset-id={datasetId}
      data-group-id={group.groupId}
      // The anchor is what the frame's pointer handlers actually move and
      // resize: the strip sits at its box, so moving the strip IS moving the
      // anchor. One gesture machinery, not two.
      data-anchor-id={anchorId}
      role="group"
      aria-label={`Group of ${count} pinned images from ${laneName || 'this dataset'}`}
      style={{ position: 'absolute', left: group.x, top: group.y,
        width: group.w, height: group.h }}
      className="lds-canvas-group rounded-md border border-indigo-400/40 bg-surface-overlay shadow-lg">

      {/* The pictures. Edge to edge, gap zero — the strip is one band. */}
      {group.members.map((m) => (
        <CanvasImageNode key={m.node.imageId} node={m.node} datasetId={datasetId}
          laneName={laneName} variant="member"
          box={{ x: m.x - group.x, y: m.y - group.y, w: m.w, h: m.h }}
          onClose={onClose} onOpen={onOpen} onDelete={onDelete} boardScale={boardScale}
          blendNote={blendNotes?.get(m.node.imageId) || null} />
      ))}

      {/* Resizing the strip resizes it as a WHOLE, keeping its shape: a member
          has no width of its own to drag (it is its aspect ratio at the strip's
          height), so a per-member handle would be a control with nothing behind
          it. Same 28 units and the same counter-scale as everywhere else. */}
      <span data-canvas-group-resize="" aria-hidden
        title="Drag to resize the whole group"
        style={{ position: 'absolute', right: 0, bottom: 0, width: 28, height: 28,
          transform: `scale(${Math.max(1, 1 / Math.max(boardScale, 0.01))})`,
          transformOrigin: 'bottom right' }}
        className="cursor-nwse-resize touch-none rounded-tl-md border-l border-t border-indigo-400/40 bg-app/80 text-content-subtle after:absolute after:bottom-1 after:right-1 after:text-[0.625rem] after:content-['◢']" />

      {/* ⤢ While a picture is being dragged out, say so — and say it INSIDE the
          strip, where the finger is. Without it, letting go one pixel too early
          just looks like the drag failed. */}
      {dropHint === 'leaving' && (
        <div data-testid="canvas-group-drop-hint"
          style={{ position: 'absolute', left: 0, top: 0, width: group.w, height: group.h }}
          className="pointer-events-none flex items-end justify-center rounded-md bg-indigo-500/10 ring-2 ring-inset ring-indigo-300/50">
          <span style={{ fontSize: Math.max(9, barH * 0.42), marginBottom: barH * 0.3 }}
            className="rounded bg-app/90 px-1.5 py-0.5 font-semibold text-indigo-100">
            Drag it off the group to take it out
          </span>
        </div>
      )}
    </div>
  );
}
