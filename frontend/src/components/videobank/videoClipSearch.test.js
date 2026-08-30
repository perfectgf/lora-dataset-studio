import test from 'node:test'
import assert from 'node:assert/strict'

import {
  searchUnavailableReason, summarize, unsearchableNote, formatTimestamp,
  frameLabelPhrase, matchLine, seekFragment, playFromSecond,
  VIDEO_CLIP_LIMITS, limitsSentence,
  searchBasisNote, captionMatchNote, captionStateNote, uncaptionedWarning,
  captionModelNote, captionStyleLabel, overBudgetWarning,
} from './videoClipSearch.js'

// ---- what stops a search before it starts ------------------------------------

test('a bank whose shots were never embedded says which pass to run', () => {
  // Not "no results": the shots are there, they simply cannot be reached by any
  // phrase. Those two sentences send the user to two different places.
  const why = searchUnavailableReason({ clips: 40, embedded: 0 }, { available: true })
  assert.match(why, /Find scenes/i)
})

test('a partly embedded bank is searchable and says nothing', () => {
  assert.equal(searchUnavailableReason({ clips: 40, embedded: 12 },
    { available: true }), '')
})

test('an install that cannot run CLIP says so instead of offering the pass', () => {
  // Telling someone to run a pass that cannot start is a loop, not an answer.
  const why = searchUnavailableReason({ clips: 40, embedded: 0 },
    { available: false, reason: 'text search needs torch + open_clip' })
  assert.match(why, /torch/)
  assert.doesNotMatch(why, /Find scenes/i)
})

test('an empty bank is not scolded about embeddings', () => {
  assert.match(searchUnavailableReason({ clips: 0, embedded: 0 },
    { available: true }), /no shots/i)
})

// ---- the honest summary -------------------------------------------------------

test('the summary never claims a match, only a ranking', () => {
  const line = summarize({
    query: 'a red car', pool: 120, unembedded: 0, clip_ids: [1, 2, 3],
    score_range: { top: 0.24, bottom: 0.19 }, pool_median: 0.12,
  })
  assert.match(line, /3 closest/)
  assert.match(line, /120/)
  // The load-bearing disclaimer: every shot scores something against every
  // phrase, so a full-looking result list is not evidence of anything.
  assert.match(line, /does not select|not a filter/i)
})

test('shots that could not be searched are named in the summary', () => {
  const line = summarize({
    query: 'a red car', pool: 12, unembedded: 28, clip_ids: [1],
    score_range: { top: 0.2, bottom: 0.2 }, pool_median: 0.1,
  })
  assert.match(line, /28/)
})

test('an empty result set still explains itself', () => {
  const line = summarize({ query: 'a red car', pool: 0, unembedded: 40, clip_ids: [] })
  assert.match(line, /a red car/)
  assert.match(line, /40/)
})

test('nothing at all is an empty string, not the word undefined', () => {
  assert.equal(summarize(null), '')
  assert.equal(unsearchableNote({ unembedded: 0 }), '')
})

// ---- pointing at the right second ---------------------------------------------

test('a timestamp is minutes and seconds, not raw float seconds', () => {
  // "83.4" is not a position anyone can find in a player.
  assert.equal(formatTimestamp(83.4), '1:23.4')
  assert.equal(formatTimestamp(4), '0:04.0')
  assert.equal(formatTimestamp(3671.2), '61:11.2')
  assert.equal(formatTimestamp(null), '—')
  assert.equal(formatTimestamp('nope'), '—')
})

test('the frame that matched is described in words, not by its internal label', () => {
  assert.match(frameLabelPhrase('start'), /start/i)
  assert.match(frameLabelPhrase('end'), /end/i)
  assert.match(frameLabelPhrase('key'), /sharp/i)
  // An unknown label must degrade to nothing rather than leak a key name.
  assert.equal(frameLabelPhrase('wat'), '')
})

test('the match line gives the second AND says it is one moment of the shot', () => {
  // The whole reason several frames are embedded: the phrase may describe two
  // seconds of a thirty-second shot, and pretending otherwise oversells it.
  const line = matchLine({ frame_s: 12.5, frame_label: 'end', score: 0.21 })
  assert.match(line, /0:12\.5/)
  assert.match(line, /end/i)
})

test('a match line for a shot with no timestamp does not invent one', () => {
  assert.equal(matchLine(null), '')
  assert.equal(matchLine({ score: 0.2 }), '')
})

// ---- seeking the player to the matched second ---------------------------------

test('the media fragment seeks to the matched second inside the shot', () => {
  // The player streams the SOURCE file, so the offset is absolute — using the
  // second as if it were relative to the shot lands in the wrong place.
  assert.equal(seekFragment({ start_s: 10, end_s: 20 }, 14.5), '#t=14.5,20')
})

test('a matched second outside the shot falls back to the shot itself', () => {
  // Bounds can be re-cut after embedding. Seeking outside the span would show a
  // frame from a neighbouring shot and call it the match.
  assert.equal(seekFragment({ start_s: 10, end_s: 20 }, 55), '#t=10,20')
  assert.equal(seekFragment({ start_s: 10, end_s: 20 }, null), '#t=10,20')
})

test('the player opens on the matched second, and never outside the shot', () => {
  assert.equal(playFromSecond({ start_s: 10, end_s: 20 }, 14.5), 14.5)
  assert.equal(playFromSecond({ start_s: 10, end_s: 20 }, 9.9), 10)
  assert.equal(playFromSecond({ start_s: 10, end_s: 20 }, undefined), 10)
})

// ---- what CLIP genuinely cannot do --------------------------------------------

test('the negation trap is stated, because it fails invisibly', () => {
  // Measured on this exact checkpoint: "without a helmet" scored HIGHER on a
  // helmeted astronaut than "with a helmet". The results come back full and
  // confident, carrying exactly what was asked to be gone.
  const all = VIDEO_CLIP_LIMITS.join(' ')
  assert.match(all, /without/i)
  assert.match(limitsSentence(), /-/)
})

// ---- hybrid search (wave 5) ---------------------------------------------------

test('the summary says what the ranking actually leaned on', () => {
  // "Searched 340 shots" hides the difference between a visual-only ranking and
  // one that also read the captions — and they answer different questions.
  const clipOnly = summarize({ query: 'a red car', pool: 40, unembedded: 0,
    clip_ids: [1], score_range: { top: 0.2, bottom: 0.2 }, pool_median: 0.1,
    hybrid: false })
  assert.doesNotMatch(clipOnly, /caption/i)

  const hybrid = summarize({ query: 'a red car', pool: 40, unembedded: 0,
    clip_ids: [1], score_range: { top: 0.2, bottom: 0.2 }, pool_median: 0.1,
    hybrid: true, captioned: 31 })
  assert.match(hybrid, /caption/i)
  assert.match(hybrid, /31/)
})

test('the readiness line distinguishes what CAN be found from what cannot', () => {
  // CLIP finds what is visible; captions find what HAPPENS. A user who does not
  // know which half is running cannot tell "not in the bank" from "not visible
  // in a single frame".
  assert.match(searchBasisNote({ clips: 40, embedded: 40, captioned: 0 }),
    /looks? like|visible/i)
  const both = searchBasisNote({ clips: 40, embedded: 40, captioned: 40 })
  assert.match(both, /happen|action/i)
})

test('a partly captioned bank says so instead of implying the whole of it', () => {
  const note = searchBasisNote({ clips: 40, embedded: 40, captioned: 12 })
  assert.match(note, /12/)
  assert.match(note, /40/)
})

test('a caption match is explained on the result, never left as a number', () => {
  assert.equal(captionMatchNote({ caption_hit: 0 }), '')
  assert.equal(captionMatchNote({}), '')
  assert.match(captionMatchNote({ caption_hit: 1 }), /caption/i)
  assert.match(captionMatchNote({ caption_hit: 0.5 }), /caption/i)
})

// ---- caption editing ----------------------------------------------------------

test('a caption state is described in words the user can act on', () => {
  assert.match(captionStateNote({ caption_state: 'edited' }), /you|yours|edited/i)
  assert.match(captionStateNote({ caption_state: 'error' }), /fail|could not/i)
  assert.equal(captionStateNote({ caption_state: 'ok', caption: 'x' }), '')
  assert.match(captionStateNote({}), /no caption/i)
})

test('the promotion warns about clips that would ship with an empty prompt', () => {
  // ai-toolkit trains an empty sidecar as an empty prompt and says nothing.
  const note = uncaptionedWarning({ captioned: 8, uncaptioned: 12 })
  assert.match(note, /12/)
  assert.match(note, /empty|no caption|without/i)
  assert.equal(uncaptionedWarning({ captioned: 20, uncaptioned: 0 }), '')
})

// ---- which model wrote the captions (wave 5b) --------------------------------

test('the caption pass names the model it will use', () => {
  // Two checkpoints do not write comparable captions, so "Describe shots" alone
  // leaves nobody able to say what wrote theirs.
  const note = captionModelNote({ model: 'Qwen/Qwen3-VL-4B-Instruct',
    cached: true, is_default: true })
  assert.match(note, /Qwen3-VL-4B-Instruct/)
})

test('a model this machine does not have warns BEFORE the click', () => {
  // Gigabytes over someone's connection, from a button that otherwise looks
  // like every other pass.
  const note = captionModelNote({ model: 'someone/other-vlm', cached: false,
    is_default: false })
  assert.match(note, /download/i)
  assert.match(note, /someone\/other-vlm/)
})

test('a non-default model is flagged as a choice someone made', () => {
  const note = captionModelNote({ model: 'someone/other-vlm', cached: true,
    is_default: false })
  assert.doesNotMatch(note, /download/i)
  assert.match(note, /someone\/other-vlm/)
})

test('no info means no invented sentence', () => {
  assert.equal(captionModelNote(null), '')
  assert.equal(captionModelNote({}), '')
})

test('the caption style is named in words a user can weigh', () => {
  // "plain" alone is not a choice anybody can make. The label has to say what
  // changes, without the panel becoming explicit itself.
  const styles = [
    { key: 'standard', label: 'Standard', hint: 'Describes the action.' },
    { key: 'plain', label: 'Plain', hint: 'Also names explicit content.' },
  ]
  assert.match(captionStyleLabel(styles, 'plain'), /Plain/)
  assert.match(captionStyleLabel(styles, 'plain'), /explicit/i)
  assert.equal(captionStyleLabel(styles, 'nope'), '')
  assert.equal(captionStyleLabel(null, 'plain'), '')
})

test('overBudgetWarning fires only past a PUBLISHED budget, and says the worst case', () => {
  assert.equal(overBudgetWarning({ over_caption_budget: 0, caption_word_budget: 200 }), '')
  assert.equal(overBudgetWarning({ over_caption_budget: 3, caption_word_budget: 0 }), '')
  assert.equal(overBudgetWarning(null), '')
  const note = overBudgetWarning({
    over_caption_budget: 3, caption_word_budget: 200, caption_words_max: 241 })
  assert.match(note, /3 caption\(s\)/)
  assert.match(note, /200-word/)
  assert.match(note, /241 words/)
  assert.match(note, /without saying so/)
})
