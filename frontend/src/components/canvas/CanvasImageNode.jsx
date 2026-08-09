import { nudgeImageNode } from '../../utils/canvasImageNodes';
import { CONTROL_UNITS, chromeScale, clusterUnits } from '../../utils/canvasNodeChrome';
import { imageFactsLine } from '../../utils/generatedImageFacts';
import { useImageDownload } from '../../hooks/useImageDownload';
import { useCanvasImageDelete } from '../../hooks/useCanvasImageDelete';
import { canvasDeleteButtonState } from '../../utils/canvasImageDelete';

/* 🖼 One generated image, pinned ON the board.

   The board compares checkpoints and the thing being compared is the picture —
   so a picture that can only be seen one at a time, in a modal, cannot be
   compared at all. Pinned, it becomes a node like any other: draggable,
   resizable, joined by an edge to the checkpoint pill that produced it (that
   edge is drawn by the lane, with the SAME connector a continuation uses — see
   utils/canvasImageNodes.imageNodeEdges).

   Closing is a ✕ on the node and it does NOT forget anything: the geometry is
   kept and re-opening the same image from its gallery puts it back exactly
   here, at exactly this size. That is the feature; the storage side of it is
   `visible: false` on the canvas_image_node row.

   ⚠️ Touch. Dragging a node and panning the board are the same physical
   gesture, and this file inherits the board's answer to that (LineageCanvas):
   with a mouse the press hit-tests onto the node and moves it; with a finger
   the board pans until a LONG PRESS picks the node up. The resize corner is the
   exception — it is hit-tested first on any pointer type, because a finger that
   lands on a 28-px corner handle can only mean one thing.

   ⚠️ …and the size of those controls is NOT a board size. Reported from a phone:
   "the cross does not close the preview". The handler was right — the board has
   always refused to start a gesture from a node's own button — but the button
   itself was ~16 board units on a board read at 65 %, so about ten pixels under
   a finger, with the 🔍 immediately beside it. The controls are therefore
   counter-scaled by the board zoom (utils/canvasNodeChrome.chromeScale) so they
   keep a constant size on screen, and they are laid out as one row along an
   EDGE rather than two glyphs crammed into a 12-px header row.

   ⚠️ …and that row is one LINE, in the bottom-right corner. It wrapped into two
   columns for a while, on the reasoning that a wider row spends more of the cap
   that keeps the controls off the picture, so every target shrinks. True — and
   wrong about what a control must never do. Four controls in two columns is not
   a cluster in a corner, it is a 2×2 block: 70 % of the tile wide and two rows
   tall, it landed in the middle of the picture it decorates and covered the
   "step N · strength X" label sitting beside it, so it hid both of the things
   the board exists to compare. One line along the bottom edge costs each target
   some size at extreme zoom-out (the number is pinned in canvasNodeChrome.test)
   and gives the picture back at every zoom.

   Bottom-right, not top-right, and the gallery already settled that: its 📌 sits
   bottom-right precisely because the top corners belong to what LABELS a tile
   (the 👍/👎 verdict there, the step/strength label here). Actions and labels do
   not share a corner. The row keeps clear of the resize corner by reserving it,
   which is why chromeScale is told about it.

   What ⬇ downloads keeps its lineage in its NAME — dataset, run, step, seed —
   because this board is the only place that knows all four and a file called
   out_00042_.png in Downloads knows none of them.

   ⌨ Keyboard. The node itself is focusable: arrows move it, Shift+arrows move
   it faster, +/− resize it, Esc closes it. Moving and resizing by mouse alone
   would put the whole feature out of reach of anyone who does not use one. The
   arithmetic is nudgeImageNode(), unit-tested; this file only routes keys. */

export default function CanvasImageNode({ node, datasetId, laneName, onGeometry,
  onClose, onOpen, onDelete, boardScale = 1, variant = 'node', box = null,
  blendNote = null }) {
  const img = node.image || {};
  const stepLabel = img.step == null ? 'step unknown' : `step ${img.step}`;
  // The gallery payload publishes the value persisted on LoraTestImage as
  // `strength`.  It is nullable on legacy rows: absence stays absent instead
  // of silently becoming 1.0 (or any other plausible-but-invented value).
  const rawStrength = img.strength;
  const strengthLabel = rawStrength != null && String(rawStrength).trim() !== ''
    && Number.isFinite(Number(rawStrength))
    ? `strength ${String(rawStrength)}`
    : null;
  const imageLabel = strengthLabel ? `${stepLabel} · ${strengthLabel}` : stepLabel;
  const facts = imageFactsLine(img);
  /* 🖼🖼 A MEMBER of a group draws the same picture with the same actions, in a
     box the strip decides (utils/canvasImageGroups.layoutImageNodes) instead of
     its own. Three things go away and nothing is added:

       • the frame and the rounding — that IS the request ("side by side with no
         border"), and a strip of framed tiles is a contact sheet, not one node;
       • the resize corner — a member has no size of its own to drag; the strip
         is resized as a whole, from its own corner;
       • the permanent header and control cluster — at rest the strip is just
         pictures. They come back on hover/focus of THIS picture and only this
         one, which is also how you can tell which ✕ you are about to press.

     Deliberately the SAME component and the same control cluster rather than a
     second set of buttons drawn by the group: anything the chrome gains next
     lands in a group for free. */
  const member = variant === 'member';
  const geom = box || node;
  // The controls are drawn in SCREEN space: counter-scaled by the board zoom so
  // a finger finds them at 24 % exactly as it does at 100 %.
  // geom.w, not node.w: a member's drawn width is the strip's tile, not the
  // box it carries for the day it leaves.
  const dl = useImageDownload();
  // 🗑 One arm-then-confirm delete per node — never one shared by the board, or
  // arming here and confirming there would be possible (see the hook).
  const rm = useCanvasImageDelete(onDelete);
  // The row's width budget is the row that is actually drawn: 🔍 ✕ ⬇, plus 🗑
  // when a host wired it. Asking for four when three are rendered would shrink
  // the three for nothing.
  const controlCount = 3 + (onDelete ? 1 : 0);
  const rowUnits = clusterUnits(controlCount);
  // A member has no resize corner; a node of its own has one, on this very
  // edge, so the row must be told to leave room for it (both are drawn at the
  // same counter-scale, so the reservation is in the same unscaled units).
  const corner = member ? 0 : CONTROL_UNITS;
  const k = chromeScale(boardScale, geom.w, rowUnits, corner);
  // ⚠️ maxWidth, not a guess: it is the number chromeScale capped k against.
  // Without it flex would happily draw a wider row inside that budget and every
  // target would silently lose size at low zoom.
  const chrome = { transform: `scale(${k})`, transformOrigin: 'bottom right',
    maxWidth: rowUnits, right: corner * k };
  const rmState = canvasDeleteButtonState({ armed: rm.armed, busy: rm.busy, label: imageLabel });
  // Revealed on hover/focus for a member, always on for a node of its own.
  const reveal = member
    ? ' opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100'
    : '';

  const onKeyDown = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); onClose?.(node); return; }
    // A member has no geometry of its own to nudge — the strip decides its box.
    // Swallowing the arrows anyway would silently break scrolling the page.
    if (member) return;
    const next = nudgeImageNode(node, e.key, e.shiftKey);
    if (!next) return;                    // never swallow a key we do not handle
    e.preventDefault();
    e.stopPropagation();
    onGeometry?.(node, next);
  };

  return (
    <div
      // The hit-test handles. The frame's pointer handlers read the dataset and
      // the image off these rather than every node carrying its own listeners —
      // the same arrangement the run cards use (data-canvas-node).
      data-canvas-image=""
      data-dataset-id={datasetId}
      data-image-id={node.imageId}
      role="group"
      tabIndex={0}
      onKeyDown={onKeyDown}
      aria-label={member
        ? `Pinned image from ${laneName || 'this dataset'}, ${imageLabel}, inside a group. `
          + 'Drag it off the group to take it out. Escape closes it.'
        : `Pinned image from ${laneName || 'this dataset'}, ${imageLabel}. `
          + 'Arrow keys move it, plus and minus resize it, Escape closes it.'}
      title={`${imageLabel} · ${facts}`}
      style={{ position: 'absolute', left: geom.x, top: geom.y,
        width: geom.w, height: geom.h }}
      className={'lds-canvas-image group flex flex-col overflow-hidden bg-surface-overlay '
        + (member
          // No frame and no rounding BETWEEN pictures — that is the request. The
          // inset ring on hover is what replaces it: it lights the picture you
          // are pointing at without ever drawing a rule between two of them.
          ? 'hover:ring-2 hover:ring-inset hover:ring-indigo-300/70 '
          : 'rounded-lg border border-indigo-400/40 shadow-lg ')
        + 'focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-300'}>
      {/* ⚠️ A member's label is an OVERLAY, not a row. Left in the flex flow it
          still reserved its height while invisible, the picture below it was
          then shorter than its tile, and `object-contain` answered with a dark
          band down each side — which is a border between two images, drawn by
          the very code that exists to remove it. */}
      {/* Le libelle d'un membre flotte SUR l'image : une pastille, pas un
          bandeau. Pleine largeur et quasi opaque (bg-app/80), il masquait le
          haut de la photo des qu'on la survolait — un controle qui cache ce
          qu'il decrit. Meme recette que les badges de la bibliotheque et de
          la banque : bg-black/50 + backdrop-blur, texte blanc. */}
      <header className={(member
        ? 'pointer-events-none absolute left-1 top-1 z-10 flex max-w-[calc(100%-0.5rem)] items-center gap-1 rounded border border-white/15 bg-black/50 px-1.5 py-px backdrop-blur-sm'
        : 'flex shrink-0 items-center gap-1 border-b border-border bg-app/70 px-1.5 py-0.5')
        + reveal}>
        <span className={'min-w-0 flex-1 truncate text-[0.5625rem] font-semibold tabular-nums '
          + (member ? 'text-white' : 'text-content-muted')}>
          {imageLabel}
        </span>
      </header>
      {/* The controls, as ONE row along the bottom edge, drawn at a constant
          SCREEN size. Out of the header's flow on purpose: counter-scaling
          something inside a 12-px row would either clip it or push the label
          off — and the label is at the OTHER end of the node precisely so that
          neither ever hides the other. Each target is 28 units square with air
          between them — two glyphs a pixel apart is how a miss on ✕ opened 🔍
          instead. `flex-nowrap` is load-bearing: a wrap here is the 2×2 block
          this layout exists to undo. */}
      <div style={chrome}
        data-testid="canvas-image-controls"
        // gap-0.5/p-0.5 and not gap-1/p-1: every unit of padding is a unit the
        // buttons do not get, because the cap chromeScale applies is spent on
        // the row's total width.
        className={'absolute bottom-0 z-10 flex flex-nowrap items-center justify-end'
          + ' gap-0.5 p-0.5' + reveal}>
        {/* Opens the full record — every setting, the prompt, the copy buttons.
            The node is the picture; the facts stay one click away rather than
            being crammed onto a thumbnail. */}
        <button type="button" onClick={(e) => { e.stopPropagation(); onOpen?.(node); }}
          title="Open this image full-screen with all its settings"
          aria-label={`Open ${imageLabel} full-screen`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-white/15 bg-black/50 text-white backdrop-blur-sm transition-colors text-[0.75rem] leading-none hover:bg-black/70">🔍</button>
        {/* ✕ closes the node and REMEMBERS where it was. Re-pinning the same
            image from its gallery brings it back here, this size. */}
        <button type="button" onClick={(e) => { e.stopPropagation(); onClose?.(node); }}
          data-testid="canvas-image-close"
          title="Close this image — re-opening it from its gallery puts it back here, at this size"
          aria-label={`Close the pinned image at ${imageLabel}`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-white/15 bg-black/50 text-white backdrop-blur-sm transition-colors text-[0.875rem] leading-none hover:border-red-400/60 hover:bg-red-500/70">✕</button>
        {/* ⬇ Keep this picture. Third in the row, after the two controls a hand
            already knows the position of.
            The file lands under a name that still says where it came from —
            dataset, run, step, seed (services/gallery_download.py). That name
            is the ONLY carrier of the lineage: there is no sidecar, and a
            board's whole value is lost the moment two checkpoints' renders
            become two files called out_00042_.png. */}
        <button type="button" disabled={dl.busy}
          onClick={(e) => { e.stopPropagation(); dl.download(node.imageId); }}
          data-testid="canvas-image-download"
          title="Download this image — the file name keeps its dataset, run, step and seed"
          aria-label={`Download the image at ${imageLabel}`}
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-white/15 bg-black/50 text-white backdrop-blur-sm transition-colors text-[0.75rem] leading-none hover:bg-black/70 disabled:opacity-50">
          {dl.busy ? '…' : '⬇'}
        </button>
        {/* 🗑 Delete the PICTURE, not the node.
            LAST in the row, furthest from ✕, and the two are told apart by
            colour AND by an arming step rather than by position alone: ✕ and 🗑
            one tap apart on a 28-px cluster is how a board gets cleaned up by
            accident. First press arms (the glyph gains a !, the button turns
            red), second press deletes, and it disarms itself after a few
            seconds — a live delete must not sit under the cursor of a board
            left open for an hour. Only rendered when a host wired it, so the
            surfaces that have no way to refresh afterwards do not offer it. */}
        {onDelete && (
          <button type="button" disabled={rmState.disabled}
            onClick={(e) => { e.stopPropagation(); rm.press(node); }}
            onBlur={rm.disarm}
            data-testid="canvas-image-delete"
            data-armed={rm.armed ? 'true' : 'false'}
            title={rmState.title}
            aria-label={rmState.aria}
            className={'flex h-7 w-7 shrink-0 items-center justify-center rounded-md border '
              + 'backdrop-blur-sm transition-colors text-[0.75rem] leading-none disabled:opacity-50 '
              + (rm.armed
                ? 'border-red-300 bg-red-600/90 text-white'
                : 'border-white/15 bg-black/50 text-white hover:border-red-400/60 hover:bg-red-500/70')}>
            {rmState.glyph}
          </button>
        )}
      </div>
      {/* A refused delete says so on the node, in the same strip a refused
          download uses — one place on a thumbnail for "that did not work". */}
      {rm.error && (
        <div role="alert" data-testid="canvas-image-delete-error"
          onClick={(e) => { e.stopPropagation(); rm.clearError(); }}
          style={{ transform: `scale(${k})`, transformOrigin: 'bottom left' }}
          className="absolute bottom-0 left-0 z-20 max-w-full cursor-pointer rounded-tr-md bg-red-900/90 px-1 py-0.5 text-[0.5rem] leading-tight text-red-50">
          {rm.error}
        </div>
      )}
      {/* A refusal has to be READABLE, and the node is small — so it is a strip
          across the bottom of the picture, counter-scaled like the buttons are,
          rather than a tooltip nobody hovers on a phone. It happens: the board
          lists rows whose file a resume or a trash sweep took off the disk. */}
      {dl.error && (
        <div role="alert" data-testid="canvas-image-download-error"
          onClick={(e) => { e.stopPropagation(); dl.clearError(); }}
          style={{ transform: `scale(${k})`, transformOrigin: 'bottom left' }}
          className="absolute bottom-0 left-0 z-10 max-w-full cursor-pointer rounded-tr-md bg-red-900/90 px-1 py-0.5 text-[0.5rem] leading-tight text-red-50">
          {dl.error}
        </div>
      )}
      {/* 🧬 A blended picture whose sources are not all on the board SAYS SO.
          The violet edges show the provenance we could place; this badge is the
          other half of the same honesty — without it, "two edges" and "three
          sources" are indistinguishable, and the board silently under-reports
          where a picture came from. Nothing is drawn when every source is
          placed: a badge that always speaks is noise. */}
      {blendNote && (
        <span data-testid="canvas-blend-note"
          style={{ transform: `scale(${Math.max(1, 1 / Math.max(boardScale, 0.01))})`,
            transformOrigin: 'bottom left' }}
          title={`🧬 Blended image — ${blendNote}. Only the sources still on the `
            + 'board can be linked to it.'}
          // max-w-[55%] and not max-w-full: this badge is permanent and it lives
          // on the same edge as the control row, at the other end. Full width it
          // would sit UNDER the buttons on every blended picture.
          className="pointer-events-none absolute bottom-0 left-0 z-10 max-w-[55%] truncate rounded-tr-md border-r border-t border-purple-400/50 bg-black/60 px-1 py-px text-[0.5rem] font-semibold leading-tight text-purple-200 backdrop-blur-sm">
          <span aria-hidden>🧬</span> {blendNote}
        </span>
      )}
      <div className="relative min-h-0 flex-1 bg-black/30">
        <img src={img.url} alt={`Generated at ${imageLabel}`} draggable={false}
          className="h-full w-full select-none object-contain" />
      </div>
      {/* The resize corner. 28 px on purpose — a hairline handle is a desktop-only
          affordance, and this board is used on a phone. Hit-tested BEFORE the
          drag/pan decision, so a finger landing here always resizes.
          A group MEMBER has none: its width is its aspect ratio at the strip's
          height, so there is nothing about it to drag. The strip has one. */}
      {!member && (
      <span data-canvas-image-resize="" aria-hidden
        title="Drag to resize"
        style={{ position: 'absolute', right: 0, bottom: 0, width: 28, height: 28,
          transform: `scale(${k})`, transformOrigin: 'bottom right' }}
        className="cursor-nwse-resize touch-none rounded-tl-md border-l border-t border-indigo-400/40 bg-app/80 text-content-subtle after:absolute after:bottom-1 after:right-1 after:text-[0.625rem] after:content-['◢']" />
      )}
    </div>
  );
}
