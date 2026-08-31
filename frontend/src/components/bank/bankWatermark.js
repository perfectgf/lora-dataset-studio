/** 🚩 Bank watermark cleaning — the state of the two escalation levels (pure,
 * JSX-free so `node --test` exercises it directly).
 *
 * The feature is an ESCALATION, and the UI has to show where each level stands:
 * cheap and lossless first (crop off a border mark — invents no pixel), then the
 * expensive one (repaint what's left) on what the first level could not handle.
 * Both are launched by hand, so at every moment the panel must answer three
 * questions without the user guessing: how many images does THIS level still
 * have to work on, is the button live right now, and if not, WHY (and what to do
 * about it).
 *
 * Everything here derives from the /watermark/levels payload:
 *   flagged      — still carrying a watermark (both levels' pool)
 *   croppable    — of those, the ones the router would crop → level 2's share
 *   inpaintable  — the rest → level 3's share
 *   cropped / inpainted / dismissed — already handled, by which route
 *   needs_rescan — flagged by an older build with no box: nothing can route them
 */

const num = (v) => (Number.isFinite(v) ? v : 0);

/** Normalized counters — a missing/half-loaded payload reads as all-zero rather
 * than rendering NaNs. */
export function levelCounts(levels) {
  const l = levels || {};
  return {
    scanned: num(l.scanned),
    unscanned: num(l.unscanned),
    flagged: num(l.flagged),
    croppable: num(l.croppable),
    inpaintable: num(l.inpaintable),
    cropped: num(l.cropped),
    inpainted: num(l.inpainted),
    dismissed: num(l.dismissed),
    needsRescan: num(l.needs_rescan),
    // Flagged images whose zones the user drew by hand in ▶ Review, and how many
    // of those were emptied on purpose. Both change what the LEVELS will do, so
    // neither may stay invisible.
    handMasked: num(l.hand_masked),
    emptyMasks: num(l.empty_masks),
    // 🔤 Find text — the OCR pass's own tallies (its flagged rows are already
    // inside `flagged` above; these only drive its card's label and progress).
    textScanned: num(l.text?.scanned),
    textFound: num(l.text?.found),
    textUnscanned: num(l.text?.unscanned),
  };
}

/** 🔤 Find text — the OTHER detection on the same rung as 🚩 Find. Same ladder,
 * different instrument: it reads burned-in text (bubbles, subtitles, captions,
 * sound effects) with the RapidOCR engine the video lane ships and folds the
 * zones into the mask channel 🧽 Inpaint repaints. CPU-only by construction, so
 * the one install it can be missing is the `video_text` extra. */
export function textLevelState(levels, { live = false, ocrReady = false } = {}) {
  const c = levelCounts(levels);
  const reason = live
    ? 'A pass is already running on this bank — wait for it to finish.'
    : !ocrReady
      // The SAME extra the Video bank's Safe zone pass uses — one install
      // serves both, and naming the card is what makes the fix one click.
      ? 'Install “Burned-in text” (Setup ▸ Quality tools) to read text — '
        + 'a small CPU-only package, no GPU, works offline.'
      : null;
  return {
    done: c.textScanned,
    remaining: c.textUnscanned,
    disabled: reason !== null,
    reason,
    label: c.textUnscanned > 0 && c.textScanned > 0
      ? `🔤 Read the remaining ${c.textUnscanned}`
      : c.textScanned > 0 ? '🔤 Scan again' : '🔤 Find text',
  };
}

/** Step 1 — FIND. Detection is the first rung of the same ladder, not a pass
 * that lives elsewhere: the two cleaning levels are dead without it (they route
 * on the box it stores), and splitting them across the page is what made the
 * feature read as two unrelated things.
 *
 * TWO routes can drive it. The dedicated detector extra, when installed, does not
 * need Ollama at all — so the "pull the vision model" refusal must NOT fire on a
 * machine that has the detector, or the fast route would be locked behind a
 * dependency it removed. */
export function findLevelState(levels, {
  live = false, visionReady = false, detectorReady = false,
} = {}) {
  const c = levelCounts(levels);
  const reason = live
    ? 'A pass is already running on this bank — wait for it to finish.'
    : !visionReady && !detectorReady
      // Settings ▸ Local tools, NOT ▸ Captioning & quality: that section holds the
      // ENGINE selector, while the Ollama vision model field lives in Local tools.
      // Sending someone to the wrong tab to fix a blocked pass is worse than silence.
      ? 'Pull the vision model (Settings ▸ Local tools), or install the '
        + 'watermark detector (Setup ▸ Quality tools), to scan for watermarks.'
      : null;
  return {
    done: c.scanned,
    // What is LEFT TO SCAN — not what was flagged. Detection resumes where a
    // stop left it, but a per-run progress bar restarting at 0 made that look
    // like a fresh start over already-analysed images. Naming the remainder is
    // what makes "it picked up where it stopped" visible.
    remaining: c.unscanned,
    disabled: reason !== null,
    reason,
    label: c.unscanned > 0 && c.scanned > 0
      ? `🚩 Scan the remaining ${c.unscanned}`
      : c.scanned > 0 ? '🚩 Scan again' : '🚩 Find watermarks',
  };
}

/** Level 2 — auto-crop. Always available (CPU/PIL), so the only reasons it can
 * be off are "a pass is already running" and "nothing to crop".
 *
 * `binWaiting` is how many flagged images sit in the BIN — a pile this payload
 * cannot see (its pool has always been `status != 'reject'`). It exists because
 * the level now opens a window where the bin CAN be chosen: a button disabled on
 * "nothing to crop" would lock that choice away from exactly the user who needs
 * it, and a capability nobody can reach reads as one that was never shipped. */
export function cropLevelState(levels, { live = false, binWaiting = 0 } = {}) {
  const c = levelCounts(levels);
  const reason = live
    ? 'A pass is already running on this bank — wait for it to finish.'
    : (c.croppable === 0 && binWaiting > 0)
      ? null
      : c.croppable === 0
      ? (c.flagged > 0
        ? (c.handMasked >= c.flagged
          // A hand-drawn mask can hold several zones and zones on the subject —
          // a border crop expresses neither, so this level skips those images.
          // Saying "not in a border" there would be a lie.
          ? 'The flagged images all carry a hand-edited mask — 🧽 Inpaint repaints '
            + 'the zones you drew.'
          : 'Nothing to crop: the remaining marks are not in a border, or cropping '
            + 'would shrink the image too much. Use level 3.')
        : 'Nothing flagged — run 🚩 Find watermarks first.')
      : null;
  return {
    done: c.cropped,
    remaining: c.croppable,
    disabled: reason !== null,
    reason,
    // The bin is NAMED rather than folded into the total: "✂ Auto-crop (0)" on a
    // live button is a contradiction, and adding binWaiting to croppable would
    // add an unrouted figure to a routed one.
    label: c.croppable === 0 && binWaiting > 0
      ? `✂ Auto-crop (${binWaiting} in the bin)`
      : `✂ Auto-crop (${c.croppable})`,
    // …and the line under the button follows the label. Leaving the default
    // "0 image(s) waiting" under a button offering 2 is the card contradicting
    // itself, which is exactly what the counts on this page exist to end.
    note: binWaiting > 0 && c.croppable === 0 ? binNote(binWaiting) : '',
  };
}


/** The line under a level whose only remaining work sits in the bin. */
function binNote(n) {
  return `${n} image(s) waiting, all of them unkept — pick “✕ Unkept only” in `
    + 'the window.';
}

/** Level 3 — repaint. `method` is the engine toggle: 'lama' (or 'auto') repaints
 * small off-centre marks and leaves marks ON the subject flagged; 'klein' erases the
 * detected zones and hands the WHOLE photo to Flux.2 with the instruction to remove the
 * marks, which reaches those too — generatively, so it also clears what the scan missed
 * (a tiled mark) and can leave standing a mark nobody detected. It re-renders everything
 * else with them; utils/watermarkCleanEngine says all of that on screen, with the
 * measurements behind it.
 * An engine that isn't installed disables the button with the install path spelled
 * out — never a silent failure mid-pass. */
export function inpaintLevelState(levels, {
  live = false, method = 'auto', lamaReady = false, kleinReady = false,
  kleinReason = null, binWaiting = 0,
} = {}) {
  const c = levelCounts(levels);
  const wantsKlein = method === 'klein';
  const engineReady = wantsKlein ? kleinReady : lamaReady;
  // `kleinReason` is the ONE shared sentence (utils/localEngineReason) naming the
  // actual gap — a stopped ComfyUI, a named missing weight, a present-but-broken
  // one, a widget value this install does not have. The catch-all below is only
  // the fallback for a caller that has no capabilities payload to reason from;
  // "needs ComfyUI running and the Klein models" sends a user with a corrupted
  // 9.5 GB file to re-check the two things that were already fine.
  const reason = live
    ? 'A pass is already running on this bank — wait for it to finish.'
    : !engineReady
      ? (wantsKlein
        ? (kleinReason
          || 'Klein inpainting needs ComfyUI running and the Klein models (Setup ▸ ComfyUI).')
        : 'LaMa inpainting is not installed yet (Setup ▸ Quality tools).')
      : (c.flagged === 0 && binWaiting > 0)
        ? null                      // see cropLevelState: the bin is reachable now
        : c.flagged === 0
          ? (c.cropped > 0
            ? 'Nothing left — every flagged image has been handled.'
            : 'Nothing flagged — run 🚩 Find watermarks first.')
          : null;
  return {
    done: c.inpainted,
    remaining: c.flagged,
    disabled: reason !== null,
    reason,
    label: c.flagged === 0 && binWaiting > 0
      ? `🧽 Inpaint (${binWaiting} in the bin)`
      : `🧽 Inpaint (${c.flagged})`,
    note: binWaiting > 0 && c.flagged === 0 && engineReady ? binNote(binWaiting) : '',
  };
}

/** True once anything has been cleaned — the Undo affordance's condition. */
export function hasCleanedImages(levels) {
  const c = levelCounts(levels);
  return c.cropped + c.inpainted > 0;
}

/** The one-line status above the two buttons. Says what is left AND what has
 * been done, because "0 flagged" alone can't be told apart from "never ran". */
export function progressSummary(levels) {
  const c = levelCounts(levels);
  if (c.scanned === 0) return 'Not scanned yet — run 🚩 Find watermarks to flag the marked images.';
  // A PARTIAL scan must say so first. Reporting "N still flagged" while
  // thousands of images were never looked at reads as a finished pass, and it
  // is the same misreading that makes a resumed scan look like it started over.
  const partial = c.unscanned > 0
    ? `${c.scanned} of ${c.scanned + c.unscanned} scanned — ${c.unscanned} still to look at. `
    : '';
  const done = [];
  if (c.cropped) done.push(`${c.cropped} cropped`);
  if (c.inpainted) done.push(`${c.inpainted} repainted`);
  if (c.dismissed) done.push(`${c.dismissed} dismissed`);
  const left = c.flagged === 0
    ? 'nothing left flagged'
    : `${c.flagged} still flagged (${c.croppable} croppable, ${c.inpaintable} to repaint)`;
  return partial + (done.length ? `${done.join(', ')} — ${left}.` : `${left}.`);
}

/** What the hand-edited masks change, or null. An emptied mask repaints NOTHING:
 * that image would otherwise sit in the flagged count forever, looking like a
 * pass that keeps failing on it. */
export function maskNote(levels) {
  const c = levelCounts(levels);
  if (!c.handMasked) return null;
  const drawn = `${c.handMasked} flagged image(s) carry a mask you edited — 🧽 Inpaint `
    + 'repaints those zones, ✂ Auto-crop skips them.';
  if (!c.emptyMasks) return drawn;
  return `${drawn} ${c.emptyMasks} of them has an EMPTY mask and will be cleaned by `
    + 'neither level — draw a zone in ▶ Review, or dismiss the image.';
}

/** WHO decided these verdicts, and who will decide the next ones — one sentence,
 * or null when there is nothing scanned and nothing to say.
 *
 * This is not trivia. The two routes are different instruments: the detector
 * returns a SCORE compared against a threshold the user can move, the vision
 * model returns a sentence it wrote. A user looking at a flag and asking "why?"
 * gets a different answer, and a different remedy, depending on which one ruled
 * — so a panel that hides the source hides the only actionable half. */
export function sourceNote(levels) {
  const l = levels || {};
  const s = l.sources || {};
  const detector = num(s.detector);
  const vision = num(s.vision);
  const unknown = num(s.unknown);
  const next = l.next_source === 'detector' ? 'detector' : 'vision';
  const thr = Number.isFinite(l.threshold) ? l.threshold : null;
  const nextLabel = next === 'detector'
    ? `the watermark detector${thr === null ? '' : ` (flags at a score of ${thr})`}`
    : 'the vision model';
  if (detector + vision + unknown === 0) return `Nothing scanned yet — ${nextLabel} will do it.`;
  const done = [];
  if (detector) done.push(`${detector} by the detector`);
  if (vision) done.push(`${vision} by the vision model`);
  // Never attributed to either route: these rows were scanned before the app
  // recorded which one ruled. Saying "by the vision model" would be a guess.
  if (unknown) done.push(`${unknown} before the source was recorded`);
  return `Judged ${done.join(', ')}. A new run uses ${nextLabel}.`;
}

/** Rows a previous build flagged without keeping the box: they are invisible to
 * both levels until a scan re-adopts them, so say it instead of losing them. */
export function rescanNote(levels) {
  const c = levelCounts(levels);
  if (!c.needsRescan) return null;
  return `${c.needsRescan} image(s) were flagged before the marks' positions were `
    + 'recorded — run 🚩 Find watermarks again to make them cleanable.';
}
