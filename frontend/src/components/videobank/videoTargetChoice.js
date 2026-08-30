/** 🎬 Choosing what to promote a bank INTO — the target, the length, the size.
 *
 * TWO FIELDS OF THE CATALOGUE ARE THE POINT OF THIS FILE, and neither is a
 * footnote:
 *
 *   `training_verified` — whether a LoRA trainer for this target is known to
 *     exist. The app can know a model's geometry perfectly and still have no way
 *     to train it. ONE target out of the four clears this bar. Someone who picks
 *     an unverified one and finds out afterwards has spent a week of cutting,
 *     captioning and GPU time on a dataset nothing can read.
 *
 *   `licence_note` — MiniMax H3's Community Licence grants rights only inside an
 *     "Applicable Territory" that EXCLUDES the EU, the UK, South Korea and the
 *     USA, and the restriction reaches the OUTPUTS. Keeping the training private
 *     is not a way around it. That belongs where the choice is made, not in a
 *     doc nobody opens.
 *
 * Both are therefore surfaced BY THIS MODULE, at the picker, and pinned by tests
 * — a target that quietly loses its warning is a regression, not a tidy-up.
 *
 * THE LENGTH SELECTOR OFFERS FRAME COUNTS FROM THE CATALOGUE, NEVER A FREE FIELD
 * IN SECONDS. The frame rule is a property of each model's VAE, not of video:
 * 29 frames is legal for Wan and illegal for LTX; MiniMax wants f % 17 == 5. A
 * seconds field would produce an illegal count on every keystroke, and the
 * trainers do not refuse it — they floor it in latent space, silently.
 *
 * PURE: no JSX, no fetch.
 */

/** One option of the length selector: the frame count, and what it means in
 * seconds AT THE TARGET'S OWN RATE. Seconds are shown, never entered. */
export function frameOptions(target) {
  if (!target || !Array.isArray(target.frame_choices)) return []
  return target.frame_choices.map((frames) => ({
    frames,
    seconds: clipSeconds(target, frames),
    label: frameOptionLabel(target, frames),
  }))
}

/** "(frames - 1) / fps", because N frames span N-1 intervals. Not cosmetic: it
 * decides how much source a cut needs, and it is why both Wan variants land on
 * exactly 5.00 s at their own rate (81 @ 16, 121 @ 24). Null with no fps. */
export function clipSeconds(target, frames) {
  const fps = target && target.fps
  if (!fps || !frames) return null
  return (frames - 1) / fps
}

export function frameOptionLabel(target, frames) {
  const s = clipSeconds(target, frames)
  return s == null ? `${frames} frames` : `${frames} frames — ${s.toFixed(2)}s`
}

/** The length pre-selected when a target is picked. Null when the catalogue
 * declares none (the "Generic / other" escape hatch). */
export function defaultFrames(target) {
  return (target && target.frame_default) || null
}

/** Does this target hand us a menu, or must the user type a count?
 *
 * "Generic / other" imposes NOTHING — no fps, no rule, no lengths — so it has an
 * empty `frame_choices`. That must render as "we have no verified lengths for
 * this target", never as a silent fallback to Wan's menu, and never as "any
 * length is fine". */
export function needsManualFrames(target) {
  return !!target && (target.frame_choices || []).length === 0
}

/** The size choices. `null` width/height means "keep the source's size", which
 * is the honest default: the catalogue's recommended sizes mirror the models'
 * inference CLIs and are NOT training constraints. */
export function sizeOptions(target) {
  const out = [{
    key: 'source',
    label: 'Source size (no resize)',
    width: null,
    height: null,
  }]
  // A target that caps the canvas AREA makes "no resize" the one path that can
  // smuggle an out-of-spec size through — the backend refuses it at launch, and
  // this hint is what keeps that refusal from being a surprise.
  if (target && target.max_pixels) {
    out[0].hint = `This target caps the canvas at ${target.max_pixels.toLocaleString()} px — sources larger than that must be rescaled (pick a size below).`
  }
  // Sizes the app VERIFIED survive the trainer's re-bucketing lead the list,
  // said in plain words; the model's own stated sizes follow, with the cost of
  // picking one named instead of discovered after the encode. Membership only,
  // no arithmetic here — the backend derives exact_sizes and a test holds it.
  const exact = ((target && target.exact_sizes) || []).map(([w, h]) => `${w}x${h}`)
  for (const pair of (target && target.exact_sizes) || []) {
    const [w, h] = pair
    out.push({ key: `${w}x${h}`, label: `${w} × ${h} — trains exactly as cut`,
               width: w, height: h })
  }
  for (const pair of (target && target.recommended_sizes) || []) {
    const [w, h] = pair
    if (exact.includes(`${w}x${h}`)) continue
    out.push({
      key: `${w}x${h}`,
      label: exact.length
        ? `${w} × ${h} — model's stated size (rescaled a little in training)`
        : `${w} × ${h}`,
      width: w, height: h,
    })
  }
  return out
}

/** One honest sentence about the dataset's SIZE, the number the research says
 * matters most and the promote window never mentioned: on one recipe, 1,500
 * steps won at 53 clips and the same recipe needed 5,000 at 176 — strong LoRAs
 * live around 50-200 clips, and a dozen proves plumbing, not a look. */
export function datasetScaleNote(count) {
  const n = Number(count)
  if (!Number.isFinite(n) || n <= 0) return null
  if (n < 10) {
    return { tone: 'warning',
             text: `${n} clip${n > 1 ? 's' : ''} — enough to prove the pipeline, not to teach a look.` }
  }
  if (n < 50) {
    return { tone: 'info',
             text: `${n} clips — fine for a test; strong LoRAs are typically trained on 50-200 clips.` }
  }
  if (n <= 200) {
    return { tone: 'good', text: `${n} clips — in the measured 50-200 sweet spot.` }
  }
  return { tone: 'info',
           text: `${n} clips — above the measured range; curation beats volume from here.` }
}

/** Everything the picker must SHOW about a target, in one object.
 *
 * `warnings` is ordered by what costs the most to discover late: a licence that
 * grants nothing in your country outranks a trainer that does not exist, which
 * outranks anything else. */
export function targetWarnings(target) {
  if (!target) return []
  const out = []
  if (target.licence_note) {
    out.push({ key: 'licence', tone: 'danger', icon: '⚖', text: target.licence_note })
  }
  if (!target.training_verified) {
    out.push({
      key: 'unverified',
      tone: 'warning',
      icon: '⚠',
      text: 'No LoRA trainer is known to exist for this target. You can cut a '
        + 'dataset for it, but nothing is known to train on it yet.',
    })
  }
  if (target.keep_audio) {
    out.push({
      key: 'audio',
      tone: 'info',
      icon: '🔊',
      text: 'This target trains audio and video together — the soundtrack is kept, '
        + 'and captions should describe it.',
    })
  }
  return out
}

/** The one-line badge next to a target's name in the list. Short on purpose:
 * the full sentence is in the warnings below the picker. */
export function targetBadge(target) {
  if (!target) return null
  if (target.licence_note) return { tone: 'danger', text: 'Licence limits' }
  if (!target.training_verified) return { tone: 'warning', text: 'Not trainable yet' }
  return { tone: 'ok', text: 'Trainable' }
}

/** A client-side refusal, or null. Duplicating the server's checks is NOT the
 * goal — the server refuses authoritatively. This exists so the dialog can grey
 * its own button out and say why, instead of round-tripping a 400. */
export function promoteProblem({ name, target, frames }) {
  if (!(name || '').trim()) return 'Name the dataset first.'
  if (!target) return 'Pick a target model first.'
  if (!frames) {
    return needsManualFrames(target)
      ? `${target.label} declares no clip lengths — type a frame count.`
      : 'Pick a clip length.'
  }
  if (!Number.isInteger(frames) || frames <= 0) {
    return 'A clip length is a whole number of frames.'
  }
  return null
}

/** The POST body for /video-bank/<id>/promote.
 *
 * `ids` is OMITTED when the selection is empty, because an empty list means
 * EVERY KEPT CLIP on the server — the caller must have decided that on purpose,
 * and it always goes through this function so the two cannot drift.
 *
 * `width`/`height` ride together or not at all: the server treats "both present"
 * as a resize and anything else as "keep the source's size", so sending a lone
 * width would silently be ignored. */
export function promotePayload({ name, targetKey, frames, size, ids, edgeInsetS,
                                 maxPerSource, triggerWord }) {
  const body = {
    name: (name || '').trim(),
    target_profile: targetKey,
    frames,
  }
  if (size && size.width && size.height) {
    body.width = size.width
    body.height = size.height
  }
  if (Array.isArray(ids) && ids.length > 0) body.ids = ids
  // Omitted when it is zero, not sent as 0. The server's default IS zero, and a
  // body that always carries the key invites a later reader to give it a
  // non-zero default "since it is always sent anyway" — which would silently
  // change what every existing recipe exports.
  const inset = Number(edgeInsetS)
  if (Number.isFinite(inset) && inset > 0) body.edge_inset_s = inset
  // Same rule as the inset: omitted when there is no cap, never sent as 0 —
  // which the server would refuse anyway, and which reads as "cap of zero".
  const cap = Number(maxPerSource)
  if (Number.isFinite(cap) && cap >= 1) body.max_per_source = Math.trunc(cap)
  // Omitted when blank: no trigger is a legitimate choice (a style set), and
  // the server treats absence and empty the same on purpose.
  const trig = (triggerWord || '').trim()
  if (trig) body.trigger_word = trig
  return body
}

// ── 🎚 Capping one source's share ─────────────────────────────────────────────
// Found by this project's own first end-to-end test: it promoted "the first 50
// clips that pass", which meant id order, which meant three videos
// over-represented in a 50-clip dataset. The folder on disk looks exactly like a
// diverse one, so the imbalance is invisible without being told.
//
// Above this share of one source, the result is lopsided enough to be worth a
// sentence. 0.5 rather than the 0.6 that was actually measured: the point is to
// speak before it gets that bad.
const LOPSIDED_SHARE = 0.5

/** Why this cap cannot be used, or '' when it can. */
export function capProblem(value) {
  if (value === '' || value === null || value === undefined) return ''
  const cap = Number(value)
  if (!Number.isFinite(cap)) return 'Enter a number of clips.'
  if (cap !== Math.trunc(cap)) {
    return 'A cap is a whole number of clips.'
  }
  if (cap < 1) return 'A cap of at least 1 — leave it empty for no cap.'
  return ''
}

/** What the cap DOES, in the words that matter: which clips survive it.
 *
 * It is not a sample. Each source keeps its EARLIEST clips, in detector order —
 * stable and explainable, and deliberately not random, so promoting the same
 * bank twice gives the same dataset. A user who assumes "a representative pick"
 * gets something other than what they think, and the difference is invisible. */
export function capHint() {
  return 'Caps how many clips ONE source may contribute. Each source keeps its '
    + 'earliest clips, so promoting the same bank twice gives the same dataset. '
    + 'Sources with fewer clips than the cap keep all of theirs.'
}

/** What the finished promotion says about its own balance, or '' when there is
 * nothing worth saying.
 *
 * Deliberately AFTER the fact rather than as a pre-flight suggestion: the
 * composition is computed on the selection the server resolved, and it comes
 * back with the response. Guessing it in the dialog would mean re-deriving the
 * selection in the browser, and a suggestion built on a different view of the pool
 * than the server's is worse than none. */
export function capBalanceNote(composition, appliedCap) {
  if (appliedCap) return ''                     // the user already made the call
  const sources = Number(composition?.sources) || 0
  const share = Number(composition?.top_source_share) || 0
  // One source is not an imbalance, it is the whole bank — there is nothing to
  // spread it across and the advice would be impossible to follow.
  if (sources < 2 || share <= LOPSIDED_SHARE) return ''
  return `${Math.round(share * 100)}% of this set comes from a single source. `
    + 'Set “Max clips per source” to spread it across the other files.'
}

// ── ✂ Trimming the edges of every clip ────────────────────────────────────────
// A shot boundary is where a cut just happened, so the frames around both ends
// are disproportionately dissolves and fades — the embedding pass already
// refuses to look at them, and it matters far more for what gets TRAINED on.
// The researched figure is ~0.25 s per end. The cap mirrors the server's
// (video_bank_service.MAX_EDGE_INSET_S): not a claim about how long a transition
// is, but a guard against a typo emptying a dataset.
const MAX_EDGE_INSET_S = 5

/** Why this inset cannot be used, or '' when it can. Checked here as well as on
 * the server so the reason sits next to the field instead of arriving as a red
 * banner after a round trip. */
export function insetProblem(value) {
  if (value === '' || value === null || value === undefined) return ''
  const inset = Number(value)
  if (!Number.isFinite(inset)) return 'Enter a number of seconds.'
  if (inset < 0) {
    return 'A negative trim would extend every clip into the shot next to it.'
  }
  if (inset > MAX_EDGE_INSET_S) {
    return `That is more than any transition is — the cap is ${MAX_EDGE_INSET_S}s per end.`
  }
  return ''
}

/** What the setting is about to do, in the units the user did NOT type: they
 * enter a per-end trim, and what decides whether a clip still fits is twice it.
 * That doubling is exactly where a surprising drop count comes from. */
export function insetHint(value) {
  const inset = Number(value)
  if (!Number.isFinite(inset) || inset <= 0) return ''
  // Trailing zeros stripped: "0.50s" reads like a precision we do not have.
  const total = Number((inset * 2).toFixed(3))
  return `Trims ${inset}s off each end, so every clip is ${total}s `
    + 'shorter. Clips that no longer supply the frame count are dropped rather '
    + 'than exported short.'
}

/** The line the result shows about what the trim COST, or '' when it cost
 * nothing. Kept apart from the "too short" count on purpose: a clip that was
 * never long enough is not fixed by lowering this. */
export function insetOutcome(composition) {
  const inset = Number(composition?.edge_inset_s) || 0
  const dropped = Number(composition?.inset_would_drop) || 0
  if (!inset || !dropped) return ''
  return `${dropped} clip${dropped === 1 ? '' : 's'} will be dropped by the `
    + `${inset}s edge trim — they fit the frame count, but not once trimmed. `
    + 'Lower the trim to keep them.'
}

/** What the confirm button is about to do, spelled out. "Promote" alone hides
 * whether it takes the selection or the whole bank, and those differ by an order
 * of magnitude in GPU minutes. */
export function promoteScopeLabel(selectedCount, keepCount) {
  if (selectedCount > 0) {
    return `${selectedCount} selected clip${selectedCount === 1 ? '' : 's'}`
  }
  return `all ${keepCount} kept clip${keepCount === 1 ? '' : 's'}`
}
