/* WHERE THE FACTS PANEL SITS — the one decision, in one place.
 *
 * The generated-image lightbox has two shapes: image over facts (stacked), or
 * image beside facts (split). The shape is chosen by ONE breakpoint, and the
 * breakpoint used to be `lg` (1024 px).
 *
 * ⚠️ WHY THAT WAS WRONG, measured rather than argued. A tablet held in
 * LANDSCAPE reports ~900-960 CSS px, not the ~1850 physical px on its spec
 * sheet — the device pixel ratio eats the difference. So the widest, most
 * horizontal screen most people open this on landed one notch UNDER the split
 * and got the phone layout: at 923x520 the picture was drawn 179x262 — 9 % of
 * the viewport — with a full-width band of chips, settings and buttons pushed
 * below the fold. The layout was not wrong at any width; it started too late.
 *
 * `md` (768 px) is the honest floor. Below it a bounded reading column plus an
 * image column is a lie: at 640 px the panel would take half the screen and the
 * picture would be a stamp beside it, which is the same failure mirrored. So
 * under `md` the stack stays, and it stays deliberately — the image takes the
 * height it can and the facts scroll under it.
 *
 * ⚠️ AND WIDTH ALONE IS NOT THE QUESTION — ORIENTATION IS THE OTHER HALF. A
 * 900x2000 portrait tablet clears 768 px and is still the wrong shape for a
 * rail: measured, splitting it drew the picture 556x813 where the stack drew it
 * 832x1216. The scarce axis on a portrait screen is WIDTH, and a side column
 * spends exactly that. So the split asks for both: at least `md` wide AND
 * `landscape` (a plain `(orientation: landscape)` media query, i.e. wider than
 * tall). Every case then falls out right without naming any device — a tablet
 * held sideways splits, the same tablet stood up stacks, a desktop window
 * dragged taller than it is wide stacks, a phone never splits.
 *
 * THE PANEL WIDTH GROWS WITH THE SCREEN, between two hard bounds:
 *  - never under MIN_PANEL_PX, or "✨ Upscale & improve" and the Klein model
 *    picker wrap into unreadable stubs;
 *  - never over MAX_PANEL_PX, because the whole point of the column is a
 *    PARAGRAPH's reading width — a prompt set 900 px wide is the wall of text
 *    this component was written to end.
 * Between them the image gets every pixel that is left, which is the axis the
 * user is actually looking at.
 *
 * These are Tailwind class strings rather than computed styles on purpose: the
 * shape must be right on the FIRST paint, before any measurement, and a
 * resize must not run JavaScript. They live here, not inline in the JSX, so
 * `node --test` can assert the breakpoint and the bounds without a DOM.
 */

/** Viewport width at which the panel moves beside the image (Tailwind `md`). */
export const SPLIT_MIN_WIDTH_PX = 768;

/** Narrowest the facts column is ever drawn, in px (`w-[20rem]`). */
export const MIN_PANEL_PX = 320;

/** Widest the facts column is ever drawn, in px (`xl:w-[27rem]`). */
export const MAX_PANEL_PX = 432;

/** Every responsive utility that turns the stack into a split carries this
 *  prefix — and it is ONE string so the two halves can never disagree about
 *  when they change shape (a panel that goes to the side while the shell is
 *  still a column is a blank screen with a stripe). */
export const SPLIT_VARIANT = 'md:landscape:';

/** The overlay: a column, and a row once it is wide enough AND landscape. */
export const SHELL_CLASS = 'fixed inset-0 z-[9997] flex flex-col bg-black/95 md:landscape:flex-row';

/** The image half. `min-h-0`/`min-w-0` so a tall picture cannot push the panel
 *  out of the viewport instead of shrinking itself. */
export const IMAGE_PANE_CLASS = 'flex min-h-0 min-w-0 flex-1 items-center justify-center p-3';

/** The picture. `object-contain` keeps the aspect; the height cap is what makes
 *  it fill the column rather than sit small and centred at the top. */
export const IMAGE_CLASS = 'max-h-full max-w-full select-none rounded-lg object-contain shadow-2xl';

/* ── AND THE SECOND STATE: THE FACTS PUT AWAY ────────────────────────────────
 *
 * Everything above is about where the facts SIT. It says nothing about the case
 * where you do not want them at all — which, measured, is most of what a phone
 * is for here. At 412x780 with the panel open the picture is drawn 388x290:
 * 35 % of the screen, because the panel takes 45 vh under it and the pane keeps
 * 12 px of padding around what is left. At 904x750 held sideways it is the same
 * 35 %, the rail spending 320 of the 904 px instead. A viewer whose picture
 * never gets past a third of the screen is a viewer you cannot judge a render
 * in, and judging renders is the entire job.
 *
 * So the panel folds away, and when it does the padding goes with it: 12 px a
 * side is a considered frame around a picture in a layout, and 24 px of a
 * 412-px screen when the picture IS the layout. The rounding and the drop
 * shadow go too — they draw the edge of a card, and there is no card left, just
 * a picture on black.
 *
 * ⚠️ Two helper FUNCTIONS were the obvious way to keep the pane and the picture
 * from disagreeing, and they were wrong. This module must stay nothing but
 * constants: tests/modal-opacity-contract.test.mjs follows a component's class
 * module only when every binding it imports is SCREAMING_SNAKE_CASE — which is
 * what tells a class module apart from a component or a hook, and what stops
 * some unrelated `bg-app` three files away from vouching for a see-through
 * form. Adding `imageClass` to the import made the whole module invisible to
 * it, and this viewer was reported as a dialog with no opaque surface while the
 * panel's `bg-app` sat right here. The guard was right.
 *
 * So the pair is chosen at the call site instead — from ONE derived boolean, so
 * a bare pane can still never end up around a framed picture. */

/** The image half with the facts put away — same box, no frame around it. */
export const IMAGE_PANE_CLASS_BARE = 'flex min-h-0 min-w-0 flex-1 items-center justify-center p-0';

/** The picture with nothing beside it: no card edge, no shadow. */
export const IMAGE_CLASS_BARE = 'max-h-full max-w-full select-none object-contain';

/** The facts column. Stacked: a bounded band under the image with its own
 *  scroll. Split: a full-height bordered column of bounded width. */
export const FACTS_PANEL_CLASS = 'max-h-[45vh] w-full shrink-0 overflow-y-auto '
  + 'border-t border-white/10 bg-app px-3 py-2.5 '
  + 'md:landscape:max-h-none md:landscape:h-full md:landscape:w-[20rem] '
  + 'md:landscape:border-l md:landscape:border-t-0 md:landscape:px-4 md:landscape:py-10 '
  + 'lg:landscape:w-[24rem] xl:landscape:w-[27rem]';
