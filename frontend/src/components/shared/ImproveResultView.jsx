/* 🔍 The improve RESULT, close enough to judge.

   The modal showed the finished render at whatever size the box gave it and
   stopped there — which is fine for "did it work?" and useless for the question
   people actually open it with: what did Klein do to the skin, the hair, the
   hands. The whole point of an upscale is detail that is not visible at fit,
   and the only way to see it was to close the dialog and reopen the picture in
   the viewer that CAN zoom.

   So this borrows that viewer's engine rather than growing a second one:
   `useImageZoomPan` is the same wheel, pinch, drag and double-tap the
   generated-image lightbox runs on, over the same geometry (utils/imageZoomPan),
   with the same ceiling — a picture never magnifies past its own pixels.

   ⚠️ THE PILL IS A SIBLING OF THE FRAME, NOT A CHILD, and that is not a layout
   preference. The frame calls `setPointerCapture` on itself at pointerdown
   (useImageZoomPan), and a captured pointer retargets the pointerup AND the
   compatibility click to the capturing element — so a button INSIDE the frame
   never receives a mouse click at all. Measured in Chromium: a tap works, a
   click does not, which is exactly the desktop path the wheel was asked for.
   The lightbox has always drawn its own reset pill outside its pane; this is
   the same shape, and it is load-bearing.

   Deliberately NOT wired to Escape. In the lightbox, Escape peels the zoom
   before it closes the viewer because there is a stack of layers to walk back
   through; here Escape has exactly one meaning, "close this dialog", and the
   render it would abandon is already saved and already in the feed. The ways
   back out are the pill and the double-tap, both of which say so. */
import { useRef } from 'react';
import useImageZoomPan from '../../hooks/useImageZoomPan';

/** ⤾ The way back, and how far in you are. Its own component so a render test
 *  can execute the zoomed branch — inside the view it only appears after a
 *  gesture, and `node --test` fires none. 40 px tall below `lg` like every
 *  other control of this dialog. */
export function ZoomResetPill({ scale = 1, onReset }) {
  return (
    <button
      type="button"
      data-testid="improve-result-zoom-reset"
      onClick={(e) => { e.stopPropagation(); onReset?.(); }}
      title="Back to the whole picture (or double-tap it)"
      aria-label="Reset the zoom"
      className="absolute right-2 top-2 z-10 flex min-h-10 items-center rounded-full bg-white/10
                 px-3 text-[0.75rem] font-semibold leading-none text-white hover:bg-white/20 lg:min-h-0 lg:h-9"
    >
      <span aria-hidden className="mr-1">⤾</span>{Math.round(scale * 100)}%
    </button>
  );
}

export default function ImproveResultView({ url, alt = 'Improved result' }) {
  const frameRef = useRef(null);
  const imgRef = useRef(null);
  /* The gestures live on the FRAME, not on the picture: magnified past the box
     on one axis there are still bars of empty frame on the other, and a pan
     that dies the moment the thumb crosses onto one is a pan that fights you.
     `resetKey` is the URL — a second improve run from the same dialog must not
     inherit the first one's 3× view, panned to a corner that no longer exists. */
  const zoom = useImageZoomPan({ imgRef, frameRef, active: !!url, resetKey: url });
  if (!url) return null;
  return (
    /* Fills the dialog's body instead of claiming a fixed 62vh. A fixed height
       inside a SCROLLABLE body was the worst of both: on a phone held sideways
       the bottom of the result sat under the fold, and `touch-none` had taken
       away the drag that would have scrolled to it. The modal stops scrolling
       in this phase and the picture takes the room — nothing is below a fold
       that no longer exists. */
    <div className="relative flex min-h-0 w-full flex-1 items-center justify-center">
      <div
        ref={frameRef}
        data-testid="improve-result-frame"
        /* touch-none: without it the browser claims the pinch and zooms the
           whole PAGE over a dialog that was already zooming the picture. */
        className="absolute inset-0 flex touch-none select-none items-center justify-center overflow-hidden rounded-lg"
        title="Scroll to zoom, pinch on a touchscreen — double-tap to fit again"
        {...zoom.handlers}
      >
        <img
          ref={imgRef}
          src={url}
          alt={alt}
          draggable={false}
          style={zoom.style}
          className="max-h-full max-w-full rounded-lg border border-white/15 object-contain"
        />
      </div>
      {/* Outside the frame — see the header. Only while it can do something. */}
      {zoom.zoomed && <ZoomResetPill scale={zoom.view.scale} onReset={zoom.reset} />}
    </div>
  );
}
