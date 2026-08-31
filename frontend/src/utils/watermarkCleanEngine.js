/* What the two 🧽 CLEAN engines do, in ONE place — because the sentence is read on
 * three surfaces (the bank's Level 3 panel, the dataset's Clean bar, the review
 * lightbox) and hand-copied wording is how they drift apart.
 *
 * WHY IT HAD TO BE REWRITTEN (2026-08-31). Klein stopped being an inpaint that day.
 * It used to crop a padded square around each detected box, repaint inside it and
 * paste only that footprint back, which is what let these tooltips promise "only the
 * mark changes". The maintainer measured the alternative on their own images and
 * switched the lane: the WHOLE photo now goes to Klein with the instruction "remove
 * watermark", and the whole photo comes back re-rendered. That REACHES what a box
 * cannot — a tiled mark with no "outside", a zone the detector drew in the wrong
 * place — and it costs the byte-exactness the old sentence sold. Both halves belong
 * in the tooltip: a user who reads only the first half will think a bad render is a
 * bug rather than the trade they were offered.
 *
 * LaMa did not change, and is still described as what it is.
 */
import { sentPromptLine } from './kleinCleanOptions.js'

/** The full tooltip on the Klein engine button. Same words on every surface.
 *  It names the LIMIT as well as the reach on purpose. The zones are erased
 *  before the render — without that step Klein does not delete a logo it can
 *  still see, it REDRAWS it (a round logo came back as a moon in the sky), and
 *  the watermark detector scores that image at zero zones, so nothing but a
 *  human catches it. With the erase it comes back clean; what can still survive
 *  is a mark the scan never found, because nothing erased that one. A tooltip
 *  that promised "it clears them" would send people to #help; one that does not
 *  tell them to look would be worse. */
export const KLEIN_CLEAN_TITLE =
  'Klein: the zones the scan found are erased first, then Flux.2 re-renders the WHOLE '
  + 'photo through ComfyUI under one instruction — remove the watermarks — so it also '
  + 'clears marks the scan missed, including one tiled across the frame or sitting ON '
  + 'the subject. What can survive is a distinct mark nobody found. Every pixel is '
  + 'regenerated, so details shift outside the marks too: look at the result, and '
  + '↩ Restore original brings your file back.';

/** The tooltip, with the instruction THIS install will actually send appended.
 *
 *  It takes the capabilities object rather than a prompt string because that is where
 *  the resolved value lives (`caps.watermark_clean_prompt`, published by the backend
 *  after its own clamping) — so every surface quotes what the pass will really send,
 *  and a caps refresh updates all of them at once with no per-screen state. Before the
 *  prompt was editable the constant above was the whole truth; now it is the half that
 *  does not depend on the install, and a tooltip that kept naming a prompt the user had
 *  changed would be worse than one that named none. */
export const kleinCleanTitle = (caps) =>
  `${KLEIN_CLEAN_TITLE} ${sentPromptLine(caps?.watermark_clean_prompt)}`

/** The half-sentence appended after "Engine: Klein" in a launch window, where the
 *  engine name is already on screen and only the difference is worth the room. */
export const KLEIN_CLEAN_SHORT =
  ' — erases the zones it found, then re-renders the whole photo, which also clears '
  + 'what the scan missed (a tiled mark included); every pixel is regenerated, and '
  + '↩ Restore original brings the file back.';

/** The same trade, short enough for a level card / bar blurb next to LaMa's. */
export const CLEAN_ENGINES_BLURB =
  'LaMa repaints just the marked zones; Klein erases them and re-renders the whole '
  + 'photo, which also clears a tiled or repeated mark the scan missed.';
