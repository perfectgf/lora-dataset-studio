/* THE ONE THING THE CAPTIONERS ARE KNOWN TO DO BADLY, said where it is decided.
 *
 * Plain .js (no JSX) so `node --test` executes all of it.
 *
 * WHAT WAS MEASURED, AND BY WHOM
 * ------------------------------
 * This project's maintainer captioned one explicit test image with three
 * abliterated qwen3-vl builds (30b-a3b, 8b-instruct, 8b) and with JoyCaption.
 * All three qwen builds missed a person who was present in the image and each
 * described a generic act that was not the one in the picture; JoyCaption read the
 * same image correctly. Refusal was NOT the failure mode — the models answered
 * confidently, at length, about something else.
 *
 * That is the whole evidence base: ONE image, ONE session, THREE named builds. The
 * sentences below say exactly that and stop. They do not say "vision models
 * hallucinate", they do not say "Ollama is unreliable", and they must never be
 * edited into either — over-claiming about a third party's model would be the same
 * failure of honesty this feature exists to correct, pointed outward.
 *
 * WHY IT BELONGS IN THE CAPTION WINDOW
 * ------------------------------------
 * The default backend is 'auto', a CHAIN: JoyCaption writes what it can, the Ollama
 * vision model covers the rest. The measured failure is in the second half, and the
 * chain is invisible — the two halves land in the same column, in the same font.
 * The launch window is the last moment where the choice is still free and costs
 * nothing; afterwards the only remedy is re-reading captions one by one.
 *
 * WHEN IT STAYS QUIET, which is most of the time and is the point:
 *   · below CAPTION_NSFW_SHARE_MIN of the measured images — a bank that is
 *     incidentally spicy is not a bank this measurement is about, and a notice
 *     that fires everywhere is a notice nobody reads;
 *   · below CAPTION_NSFW_MEASURED_MIN measured images — "3 of 8" is not a share,
 *     it is noise, and quoting it would dress a coin flip up as a finding;
 *   · when ✨ Score has never run on the pile: the share is then UNKNOWN, not low.
 *     Silence is the honest reading — an unmeasured image is not a SFW image, and
 *     the app does not guess which. The cost is real and named here rather than
 *     hidden: on a never-scored bank this notice cannot fire at all.
 *
 * THE DENOMINATOR IS THE MEASURED ROWS, never the pile, for that same reason: a
 * share over the pile would silently count "never scored" as "not NSFW" and
 * understate itself on exactly the banks that need the sentence.
 */

/** At or above this share of the MEASURED images, the pass is running mostly on
 *  the material the failure was measured on.
 *
 *  A quarter, and not a majority: the failure is per image, not per bank. At 25%
 *  a 400-image run puts ~100 captions in the zone where a caption that describes
 *  something else would be indistinguishable from a good one — far past the point
 *  where a user who spot-checks a handful would happen to catch it. Going higher
 *  (a "majority") would stay silent on banks where a quarter of the work is
 *  affected; going lower would fire on ordinary portrait banks and turn the notice
 *  into furniture. */
export const CAPTION_NSFW_SHARE_MIN = 0.25;

/** …and never on fewer measured images than this. A percentage of 8 rows is not a
 *  measurement of anything. */
export const CAPTION_NSFW_MEASURED_MIN = 20;

/** How many images of this scope ✨ Score measured, and how many it flagged NSFW.
 *  null when the server has not sent the figures (a build that predates them, or a
 *  payload that has not landed) — which is not the same as zero and never renders
 *  as one. */
export function captionNsfwCounts(payload, scopeId, piles) {
  const p = payload?.pass_scopes?.caption;
  if (!p || !p.nsfw || !p.nsfw_measured) return null;
  const list = Array.isArray(piles) && piles.length ? piles : ['keep', 'pending'];
  let flagged = 0;
  let measured = 0;
  for (const pile of list) {
    flagged += Number(p.nsfw[pile]) || 0;
    measured += Number(p.nsfw_measured[pile]) || 0;
  }
  return { flagged, measured, share: measured ? flagged / measured : 0 };
}

/** Does this run reach the half that was measured going wrong?
 *  'joycaption' does not; '', 'auto' and 'ollama' do, in two different ways. */
function ollamaHalf(engineId) {
  const id = (engineId || '').trim().toLowerCase();
  if (id === 'joycaption') return 'none';
  if (id === 'ollama') return 'only';
  return 'chain';   // '' (Settings) and 'auto' — the default, and a chain
}

/** The notice, or null for "say nothing".
 *
 *  Shape: { tone, heading, paragraphs: [string] }. `tone` is 'warn' when the run
 *  reaches the measured half and 'info' when the user has already picked the way
 *  around it — the second one is a confirmation, not an alarm, and it exists
 *  because a protection nobody can see is a protection nobody trusts.
 */
export function captionNsfwNotice({ payload, scopeId, piles, engineId,
                                   providerLabel = 'Ollama' } = {}) {
  const counts = captionNsfwCounts(payload, scopeId, piles);
  if (!counts) return null;
  if (counts.measured < CAPTION_NSFW_MEASURED_MIN) return null;
  if (counts.share < CAPTION_NSFW_SHARE_MIN) return null;
  const pct = Math.round(counts.share * 100);
  const scale = `${pct}% of the ${counts.measured} scored images in this scope `
    + 'are flagged NSFW.';
  // What the measurement was, stated once and identically in every branch, so the
  // claim can never drift between them. Named builds, named count, named tester.
  const measured =
    'On one explicit test image, three abliterated qwen3-vl builds (30b-a3b, '
    + '8b-instruct, 8b) each missed a person who was present and each described a '
    + 'generic act that was not the one in the picture, while JoyCaption read the '
    + 'same image correctly.';
  const limits =
    'That is one image, captioned once by this project\'s maintainer, on those '
    + 'three builds — not a benchmark, and nothing is claimed about other vision '
    + 'models. It is worth saying because the failure is silent: an invented '
    + 'caption reads exactly like an accurate one.';
  const half = ollamaHalf(engineId);
  if (half === 'none') {
    return {
      tone: 'info',
      heading: 'Mostly NSFW — and you already picked JoyCaption',
      paragraphs: [
        scale,
        `JoyCaption writes this whole run: the ${providerLabel} vision model, the half that `
        + 'was measured going wrong on explicit images here, is not called at all.',
        measured,
        limits,
      ],
    };
  }
  if (half === 'only') {
    return {
      tone: 'warn',
      heading: `Mostly NSFW — and this run is entirely the ${providerLabel} vision model`,
      paragraphs: [
        scale,
        `You picked the local LLM, so every caption in this run comes from the ${providerLabel} `
        + 'vision model — the half that was measured going wrong on explicit '
        + 'images here.',
        measured,
        limits,
        'Every caption records who wrote it, so you can read them back knowing '
        + `where they came from. Picking JoyCaption above runs without ${providerLabel}.`,
      ],
    };
  }
  return {
    tone: 'warn',
    heading: 'Mostly NSFW — check who wrote what afterwards',
    paragraphs: [
      scale,
      `Auto runs JoyCaption first, then the ${providerLabel} vision model on whatever `
      + 'JoyCaption did not caption. That second half is the one that was measured '
      + 'going wrong on explicit images here.',
      measured,
      limits,
      `Every caption records who wrote it, so you can read the ${providerLabel}-written ones `
      + 'first — or pick JoyCaption above to run without that second half.',
    ],
  };
}
