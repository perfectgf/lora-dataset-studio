// 🔎 Find scenes — the wording that keeps a SIMILARITY RANKING over shots from
// reading as a filter, kept out of the JSX so `node --test` can exercise it.
//
// The image lane already solved most of this problem and its reasoning is not
// repeated here — it is IMPORTED (see bankTextSearch.js for the measurements
// behind the spread bands, the readiness hints and the negation trap). Two lanes
// with two copies of the same calibrated sentences is how one of them quietly
// stops being true.
//
// WHAT IS GENUINELY DIFFERENT ABOUT VIDEO, and therefore lives here:
//
//  1. A result is a SHOT, and a shot is a span of time. The backend embeds
//     several frames of it and scores it by the best one, so a result carries a
//     SECOND — the moment that actually matched. Showing the shot without the
//     second hands the user a thirty-second clip and tells them the answer is
//     somewhere inside it, which is barely better than not answering.
//  2. That second is ABSOLUTE, in the source file's own timeline, because the
//     player streams the source and addresses it with a media fragment. Treating
//     it as an offset from the shot's start lands the playhead in the wrong shot.
//  3. "Not embedded" is a much more likely state here than in the image lane. A
//     bank of rushes is triaged long before anyone thinks to search it, so the
//     honest default is to explain the pass rather than to return an empty grid.
import {
  spreadLabel, readinessHint, pendingLabel, suggestPushDown,
} from '../bank/bankTextSearch.js'

// Re-exported so the video components have ONE import and cannot accidentally
// grow a second, drifting copy of a sentence that was measured once.
export { spreadLabel, readinessHint, pendingLabel, suggestPushDown }

/** Why a search cannot be run here, or '' when it can.
 *
 * Three different states that a single "unavailable" would flatten into one
 * useless sentence: there is nothing in the bank yet, the shots exist but have
 * no vectors (run the pass), or this install cannot run CLIP at all (nothing in
 * this bank will fix that). Offering the pass in the third case sends the user
 * round a loop — the pass 503s for the same reason the search would.
 */
export function searchUnavailableReason(counts, status) {
  const c = counts || {}
  if (status && status.available === false) {
    return status.reason || 'Searching by words is unavailable on this install.'
  }
  if (!Number(c.clips)) return 'This bank has no shots yet — find the shots first.'
  if (!Number(c.embedded)) {
    return 'Run 🔎 Find scenes first — it looks at a few frames of every shot so '
      + 'a typed word can reach them.'
  }
  return ''
}

/** The one-line summary above the grid, announced to screen readers.
 *
 * Never claims a match: it says "closest", gives the range the ranking spans,
 * and names the shots that could not be searched at all. */
export function summarize(result) {
  if (!result) return ''
  const shown = (result.clip_ids || []).length
  if (!shown) {
    return `No searchable shot to rank for “${result.query}”. ${unsearchableNote(result)}`.trim()
  }
  const r = result.score_range || {}
  const spread = spreadLabel(r, result.pool_median)
  const parts = [
    `${shown} closest shot${shown === 1 ? '' : 's'} of ${result.pool} for `
      + `“${result.query}”, best first.`,
    // Raw cosines, shown as what they are and never as a percentage: on this
    // model even a perfect match tops out around 0.2, so "22%" reads as failure.
    `Similarity ${fmt(r.top)} down to ${fmt(r.bottom)}${spread ? ` — ${spread}` : ''}.`,
    'Search brings the likeliest shots to the front; it does not select them. '
      + 'Every shot scores something against every phrase.',
  ]
  // WHICH HALVES RAN. A visual-only ranking and one that also read the captions
  // answer different questions, and a user who cannot tell them apart cannot
  // tell "not in this bank" from "not visible in any single frame".
  if (result.hybrid) {
    parts.splice(2, 0, `Ranked on what the frames look like AND on the words in `
      + `${result.captioned} caption(s).`)
  }
  const note = unsearchableNote(result)
  if (note) parts.push(note)
  return parts.join(' ')
}

function fmt(v) {
  const n = Number(v)
  return Number.isFinite(n) ? n.toFixed(2) : '—'
}

/** The load-bearing warning. A shot with no vectors cannot be found by ANY
 * phrase — staying silent about it lets the user conclude the scene is not in
 * the bank, which is how a search quietly becomes a lie. */
export function unsearchableNote(result) {
  const missing = Number(result?.unembedded) || 0
  if (missing <= 0) return ''
  return `${missing} shot(s) here have not been looked at yet and could NOT be `
    + 'searched — run 🔎 Find scenes to include them.'
}

/** "1:23.4" — the position a human can find on a scrub bar. One decimal because
 * a shot can be two seconds long and whole seconds would point at a third of it.
 * Minutes are NOT wrapped into hours: this is an offset inside one rush, and
 * "61:11" is easier to scrub to than "1:01:11". */
export function formatTimestamp(seconds) {
  const s = Number(seconds)
  if (seconds == null || !Number.isFinite(s) || s < 0) return '—'
  const m = Math.floor(s / 60)
  const rest = s - m * 60
  return `${m}:${rest < 10 ? '0' : ''}${rest.toFixed(1)}`
}

/** Where in the shot the match was found, in words. '' for a label we do not
 * know — leaking an internal key into the UI is worse than saying nothing. */
export function frameLabelPhrase(label) {
  return {
    start: 'near the start',
    key: 'at its sharpest frame',
    end: 'near the end',
  }[String(label || '')] || ''
}

/** "matched at 0:12.5, near the end" — the second to seek to, plus the reminder
 * that ONE moment of the shot matched, not the whole span. That distinction is
 * the entire reason several frames per shot are embedded, and hiding it would
 * oversell every result. */
export function matchLine(hit) {
  const t = Number(hit?.frame_s)
  if (!hit || !Number.isFinite(t)) return ''
  const where = frameLabelPhrase(hit.frame_label)
  return `matched at ${formatTimestamp(t)}${where ? `, ${where}` : ''}`
}

/** The media fragment that opens the player ON the matched second.
 *
 * The second is ABSOLUTE in the source file, and so is the fragment — the
 * player streams the whole rush and range-requests the span. A matched second
 * outside the shot means the bounds were re-cut after embedding; falling back to
 * the shot's own start is the honest answer, because seeking outside it would
 * show a frame from a neighbouring shot and label it the match. */
export function seekFragment(clip, seconds) {
  return `#t=${round3(playFromSecond(clip, seconds))},${round3(Number(clip?.end_s) || 0)}`
}

/** The second the player should OPEN at for this shot: the matched one when it
 * really falls inside the shot, the shot's own start otherwise. Separate from
 * the fragment because the lightbox builds its own src and needs the number, and
 * one clamp in two places is one clamp too many. */
export function playFromSecond(clip, seconds) {
  const start = Number(clip?.start_s) || 0
  const end = Number(clip?.end_s) || 0
  const t = Number(seconds)
  return (Number.isFinite(t) && t >= start && t <= end) ? t : start
}

function round3(v) {
  return Math.round(Number(v) * 1000) / 1000
}

/** What CLIP genuinely cannot do, shown in the panel rather than buried in a
 * doc: a user who trusts a wrong answer here silently builds a bad dataset.
 * These are measured weaknesses of this exact checkpoint, not a disclaimer.
 *
 * The negation entry is the expensive one. On a photo of a helmeted astronaut,
 * "an astronaut without a helmet" scored HIGHER (0.217) than "an astronaut with
 * a helmet" (0.212): CLIP does not weigh "without", it ignores it. The results
 * come back full and confident carrying exactly what was asked to be gone, with
 * nothing anywhere to reveal it. Hence `-term`, which subtracts instead of
 * speaking. */
export const VIDEO_CLIP_LIMITS = [
  'Negation — “without a hat” returns hats. Type “-hat” instead: it subtracts '
    + 'the unwanted thing from the score rather than saying the word.',
  'Counting — “two people” barely outranks one person; expect “one vs several” '
    + 'at best.',
  'Spatial relations — “to the left of” carries almost no meaning.',
  'Sound and motion — only frames are looked at, so “a door slamming” or '
    + '“panning left” describe nothing it can see.',
]

/** What this bank's search can actually reach, in one line.
 *
 * The two halves find different things and neither is a superset: CLIP ranks
 * what a moment LOOKS like and cannot see an action, because an action is a fact
 * about time that no single frame carries; a caption carries the action and
 * nothing its writer did not name. Saying which are running is what lets someone
 * read an empty result correctly. */
export function searchBasisNote(counts) {
  const c = counts || {}
  const embedded = Number(c.embedded) || 0
  const captioned = Number(c.captioned) || 0
  const clips = Number(c.clips) || 0
  if (!embedded) return ''
  if (!captioned) {
    return 'Searching what the frames LOOK like. An action that no single frame '
      + 'shows — “turns and walks away” — needs captions; run 🗣 Describe shots.'
  }
  if (captioned < clips) {
    return `Searching what the frames look like, plus the words in ${captioned} of `
      + `${clips} captions — the rest can only be found by how they look.`
  }
  return 'Searching what the frames look like AND what happens in them — the '
    + 'captions carry the action.'
}

/** Why a shot moved up, when a caption is part of the reason. '' otherwise. */
export function captionMatchNote(hit) {
  const share = Number(hit?.caption_hit)
  if (!Number.isFinite(share) || share <= 0) return ''
  return share >= 1 ? 'every word of your search is in its caption'
    : 'some of your search words are in its caption'
}

/** What state this shot's caption is in, and what to do about it. */
export function captionStateNote(clip) {
  const state = clip?.caption_state
  if (state === 'edited') return 'You wrote this caption — the pass will not overwrite it.'
  if (state === 'error') return 'The model could not caption this shot. Run the pass again, or write one.'
  if ((clip?.caption || '').trim()) return ''
  return 'No caption yet. It is what this clip trains on, and what a word search reads.'
}

/** Which checkpoint writes this bank's captions, and whether it is here yet.
 *
 * Named rather than assumed. Two checkpoints do not produce comparable captions
 * — one may describe plainly what another talks around — so a bank captioned
 * across a settings change is only readable if the model is visible somewhere.
 * And a checkpoint this machine does not have yet means gigabytes over the
 * user's connection from a button that looks like every other pass, so that is
 * said BEFORE the click rather than discovered at 0/470. */
export function captionModelNote(info) {
  const model = (info?.model || '').trim()
  if (!model) return ''
  if (info.cached === false) {
    return `Captions will be written by ${model}, which is not on this machine `
      + 'yet — the first run downloads it before captioning anything.'
  }
  if (info.is_default === false) {
    return `Captions will be written by ${model} (set in your config, not the default).`
  }
  return `Captions are written by ${model}.`
}

/** "Plain — also names explicit content", for the style currently chosen.
 *
 * The key alone ("plain") is not a choice anyone can weigh, and the server ships
 * the label and the hint precisely so the two sides cannot describe the same
 * option differently. '' for a key the server does not offer. */
export function captionStyleLabel(styles, key) {
  const found = (styles || []).find((s) => s.key === key)
  if (!found) return ''
  return found.hint ? `${found.label} — ${found.hint}` : found.label
}

/** The promotion's load-bearing warning. An empty sidecar is not a neutral
 * default: ai-toolkit trains it as an EMPTY PROMPT and says nothing anywhere. */
export function uncaptionedWarning(composition) {
  const missing = Number(composition?.uncaptioned) || 0
  if (missing <= 0) return ''
  return `${missing} clip(s) have no caption and will ship with an EMPTY prompt `
    + '— the trainer accepts that silently. Run 🗣 Describe shots first, or accept it.'
}

/** Over the target's PUBLISHED word budget — WAN caps its own prompts at
 * 200 (T2V) / 100 (I2V) words and its encoder truncates past that in silence.
 * '' for targets whose vendor published no number: an invented budget would be
 * a figure somebody plans around. */
export function overBudgetWarning(composition) {
  const over = Number(composition?.over_caption_budget) || 0
  const budget = Number(composition?.caption_word_budget) || 0
  if (over <= 0 || budget <= 0) return ''
  return `${over} caption(s) run past this model's ${budget}-word prompt budget `
    + `(longest: ${composition?.caption_words_max} words) — the trainer will cut `
    + 'them off mid-sentence without saying so. Shorten them, or accept the cut.'
}

/** Past the encoder's TOKEN window — the count that decides, where words only
 * warn. Measured by the caption pass with umT5's own tokenizer when it found
 * one, estimated (1.4 tokens per word, rounded up) when it did not; the reserve
 * for the trigger and the measured lines is already inside the count. Fires
 * only for what STILL overruns after the export served its short forms. */
export function overTokenBudgetWarning(composition) {
  const over = Number(composition?.over_token_budget) || 0
  const budget = Number(composition?.caption_token_budget) || 0
  if (over <= 0 || budget <= 0) return ''
  const blocked = Number(composition?.short_blocked) || 0
  return `${over} prompt(s) still run past this model's ${budget}-token encoder window `
    + `(longest: up to ${composition?.caption_tokens_max} tokens) — the encoder drops `
    + 'the tail without saying so. Shorten those captions, or accept the cut.'
    + (blocked > 0
      ? ` ${blocked} of them wrote an unfinished short form (cut by the `
        + 'generation cap), so their full paragraph ships instead.'
      : '')
}

/** How many prompts the export wrote in their SHORT form — the caption's own
 * Subject / Motion / Setting / Style lines — because the paragraph would not
 * fit the encoder window. Said, so a sidecar shorter than its caption is no
 * mystery. */
/** What slicing ADDED, said after the build like every other composition fact.
 * Silent when the option was off — a zero here would advertise a feature by
 * reporting that it did nothing. */
export function sliceGainNote(composition) {
  const extra = Number(composition?.slice_extra_clips) || 0
  if (extra <= 0) return ''
  return `${extra} extra clip(s) came from slicing long shots — each carries its `
    + 'shot’s caption, so check the ones whose action changes mid-shot.'
}

export function servedShortNote(composition) {
  const n = Number(composition?.served_short) || 0
  if (n <= 0) return ''
  return `${n} prompt(s) were written in their short form (Subject / Motion / Setting / Style) `
    + "because the full paragraph would not fit this model's encoder window."
}

export function limitsSentence() {
  return 'Best at subjects, settings, styles and framing, in the frames it '
    + 'looked at. It cannot count, cannot hear, and ignores “without” — so '
    + 'describe what IS on screen, and type “-word” for what should be pushed '
    + 'down.'
}
