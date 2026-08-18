/** 🎬 Quality flags in the grid — pure helpers (no JSX, so `node --test` runs them).
 *
 * The backend stores RAW scores and derives flags at read time against the cuts
 * in force, so everything here is presentation: counting, filtering, describing.
 * Two decisions worth stating:
 *
 * "Unmeasured" is a state of its own, never "clean". A clip with no flags
 * because nothing was measured and a clip with no flags because it is fine are
 * different facts — collapsing them makes a half-scanned bank look healthy.
 *
 * And the dry-run sentence changes TONE when a cut would flag most of the bank.
 * The failure mode is documented: a public pipeline once kept 47 clips out of
 * 1493 with one mis-set threshold and discovered it after the fact. A preview
 * that reports "400 clips flagged" in the same cheerful voice as "4 clips
 * flagged" does not prevent that.
 */

export const FLAG_LABELS = {
  brief: 'Very short',
  still: 'Barely moves',
  agitated: 'Too much motion',
  black: 'Black moment',
  freeze: 'Frozen stretch',
  soft: 'No sharp frames',
  soft_start: 'Soft first frame',
  silent: 'Mostly silent',
  quiet: 'Very quiet',
  // The two verdicts that come from their OWN pass rather than from the metrics
  // decode. "Same as another shot" names the finding from the user's side — the
  // group's representative is deliberately NOT flagged, so a chip here always
  // means "you already have this one".
  duplicate: 'Same as another shot',
  watermark: 'Watermark',
  // Same wording the image bank puts on a still, because it is the same head on
  // the same scale — a user who learned "Low aesthetic" on the Bank must not
  // have to learn a second name for the same finding on a shot.
  low_aesthetic: 'Low aesthetic',
  // 🔳 The safe zone's three findings, named apart because they have three
  // different remedies: crop it, drop it, or stop trying. A single "bad framing"
  // chip could only ever suggest the last one.
  letterboxed: 'Letterbox bars',
  burned_text: 'Burned-in text',
  small_safe_zone: 'Little usable frame',
  unmeasured: 'Not measured yet',
}

/** {flagName: count, flagged: N, unmeasured: N}. `flagged` counts CLIPS, so a
 * clip carrying two flags is one clip — the same rule as the backend dry run,
 * and for the same reason: the total must not overstate the damage. */
export function flagCounts(clips) {
  const counts = { flagged: 0, unmeasured: 0 }
  for (const clip of clips) {
    // "Unmeasured" stays a state of its own — but it no longer swallows the
    // clip's flags. It used to `continue` here, which was correct while every
    // flag came out of the metrics pass. The duration cut reads the bounds
    // instead, so an unmeasured clip CAN be flagged, and skipping it left a
    // "Very short" chip in the grid next to a chip counter reading zero.
    if (!clip.metrics) counts.unmeasured += 1
    const flags = clip.flags || []
    if (flags.length) counts.flagged += 1
    for (const f of flags) counts[f] = (counts[f] || 0) + 1
  }
  return counts
}

/** The flag chips to offer, most-hit first — [{flag, label, count}].
 *
 * Built from what the grid HOLDS rather than from a bank-wide count, and the
 * caption below says so (see `flagFilterNote`). A chip that silently searched
 * only the loaded page while reading like a bank-wide total is how a user
 * concludes their bank has twelve duplicates when it has ninety.
 *
 * 'unmeasured' rides along as a pseudo-flag: "nothing was measured" is the state
 * people most need to select, and it is the one no verdict can express. */
export function flagChips(clips) {
  const counts = flagCounts(clips)
  return Object.keys(FLAG_LABELS)
    .filter((flag) => counts[flag] > 0
      || (flag === 'unmeasured' && counts.unmeasured > 0))
    .map((flag) => ({ flag, label: FLAG_LABELS[flag], count: counts[flag] || 0 }))
    .sort((a, b) => b.count - a.count || a.flag.localeCompare(b.flag))
}

/** The sentence under the flag chips, or '' when there is nothing to warn about.
 *
 * The chips count the LOADED page, not the bank. With everything loaded that is
 * the whole truth and the note stays out of the way; with a page of 120 out of
 * 900 it is a quarter of an answer, and saying so is the difference between a
 * filter and a wrong number. */
export function flagFilterNote(loaded, total) {
  const have = Number(loaded) || 0
  const all = Number(total) || 0
  if (all <= have) return ''
  return `Counted over the ${have} shots loaded, not all ${all} — load more to `
    + 'count the rest.'
}

/** The clips a flag chip selects. `'unmeasured'` is a pseudo-flag over the
 * metrics field; null/undefined means no filter. */
export function filterByFlag(clips, flag) {
  if (!flag) return clips
  if (flag === 'unmeasured') return clips.filter((c) => !c.metrics)
  return clips.filter((c) => (c.flags || []).includes(flag))
}

/** The threshold panel renders from this table. A cut the backend supports but
 * this table omits would be configurable only by hand-editing config.json —
 * invisibly — so the table IS the panel's contract, and a test pins the keys
 * against the backend's. `direction` says which side of the value gets flagged,
 * because "0.001" alone does not tell a user whether raising it is stricter. */
export function thresholdFields() {
  return [
    // First, and the only cut here that fires with nothing measured — shot
    // detection runs with a deliberately low frame minimum so real flash cuts
    // are not hidden, which is right and leaves the grid full of half-second
    // shots to scroll past.
    { key: 'min_duration_s', flag: 'brief', direction: 'below',
      label: 'Minimum length (s)',
      hint: 'Flags shots shorter than this, in seconds. Works straight after '
        + 'detection — no measuring pass needed. Shots too short for your '
        + 'target profile are refused at promotion anyway; this is how you see '
        + 'and sort them BEFORE spending triage time on them.' },
    { key: 'motion_floor', flag: 'still', direction: 'below',
      label: 'Motion floor',
      hint: 'Flags clips whose average motion falls below this.' },
    { key: 'motion_ceiling', flag: 'agitated', direction: 'above',
      label: 'Motion ceiling',
      hint: 'Flags clips whose busiest moments exceed this.' },
    { key: 'luma_floor', flag: 'black', direction: 'below',
      label: 'Darkest moment',
      hint: 'Flags clips whose darkest frame falls below this brightness.' },
    { key: 'freeze_max', flag: 'freeze', direction: 'above',
      label: 'Frozen share',
      hint: 'Flags clips where more than this share of frames do not move.' },
    { key: 'sharpness_floor', flag: 'soft', direction: 'below',
      label: 'Sharpness floor',
      hint: 'Flags clips whose sharpest stretch stays below this.' },
    // Supported by the backend since the quality wave and named nowhere until
    // now, which made it settable only by hand-editing config.json — the exact
    // failure the comment above this table warns about.
    { key: 'first_frame_floor', flag: 'soft_start', direction: 'below',
      label: 'First-frame sharpness',
      hint: 'Flags clips whose FIRST frame is soft. Mostly matters for '
        + 'image-to-video targets, where that frame is the conditioning image.' },
    // Audio. Only meaningful for the targets that keep a track (LTX, MiniMax H3
    // — Wan datasets have no audio by design), and a clip with no track is never
    // flagged by either.
    { key: 'silence_max', flag: 'silent', direction: 'above',
      label: 'Silent share',
      hint: 'Flags clips where more than this share of the sound is silence. '
        + '0.5 means half the clip. Clips measured before sound was looked at '
        + 'carry no reading and are never flagged — re-measure to include them.' },
    { key: 'audio_floor', flag: 'quiet', direction: 'below',
      label: 'Loudness floor (dBFS)',
      hint: 'Flags clips quieter than this overall. dBFS is negative: -40 is '
        + 'very quiet, -12 is a healthy level. A clip with no sound track is '
        + 'never flagged — that is not a defect, it is the file.' },
    // The one cut here that arrives with a number, because it is not a property
    // of your footage but of a classifier — see config.py. It reads what the
    // 🔖 Watermarks pass stored; a shot that pass never judged is never flagged.
    { key: 'watermark_max', flag: 'watermark', direction: 'above',
      label: 'Watermark score',
      hint: 'Flags clips whose watermark score exceeds this, after the '
        + '🔖 Watermarks pass has run. The scores sit close to 1, so 0.94 is the '
        + 'measured cut, not 0.5 — lower it to catch faint marks and hand-check '
        + 'a few clean shots. Shots that pass has not judged are never flagged.' },
    // The look score. Empty by default like the footage cuts, because the LAION
    // references were chosen to filter a web crawl and deliberately-shot rushes
    // sit well above them — so they belong in this hint, not in a default that
    // would decide for the user what is beautiful.
    { key: 'aesthetic_floor', flag: 'low_aesthetic', direction: 'below',
      label: 'Aesthetic floor',
      hint: 'Flags shots the LAION aesthetic head rates below this — the same '
        + '~1–10 rating, and the same model, the Bank’s ✨ Score puts on a '
        + 'still. LAION reference: 4 casual, 4.75 strict, both set for filtering '
        + 'a web crawl, so start by previewing 4 against your own bank. Shots '
        + 'rate after 🔎 Find scenes has run; an unrated shot is never flagged.' },
    // 🔳 The safe zone. All three read what that pass measured on three frames
    // of each shot, and all three ship empty: bands and burned-in text are
    // properties of somebody's footage rather than of a classifier, so the
    // published figures belong in these hints and in no default.
    { key: 'bars_max', flag: 'letterboxed', direction: 'above',
      label: 'Letterbox share',
      hint: 'Flags shots where more than this share of the frame is a flat band '
        + '— letterbox, pillarbox, or a vertical video padded into a wide one. '
        + 'The same number the Bank puts on a still, where 0.04 was the measured '
        + 'cut. Bands survive a training crop, so they are worth seeing; a '
        + '2.35:1 film honestly carries about 0.12 and is not a defect. Needs '
        + 'the 🔳 Safe zone pass; a shot it has not measured is never flagged.' },
    { key: 'text_coverage_max', flag: 'burned_text', direction: 'above',
      label: 'Burned-in text share',
      hint: 'Flags shots where subtitles, chyrons or a text watermark cover more '
        + 'than this share of the frame. Only text that HOLDS STILL across the '
        + 'shot counts — a passing shop sign is scene content and is never '
        + 'counted. 0.01 is already a full subtitle line. Needs the 🔳 Safe zone '
        + 'pass AND its text extra from Setup; with the extra missing the pass '
        + 'measures bands only and no shot is ever flagged here.' },
    { key: 'safe_area_min', flag: 'small_safe_zone', direction: 'below',
      label: 'Usable frame floor',
      hint: 'Flags shots where cropping away the bands AND the burned-in text '
        + 'would leave less than this share of the frame. HunyuanVideo 1.5 keeps '
        + 'only clips whose crop leaves 60 % or more; below about 50 % there is '
        + 'not enough picture left to be worth the trouble. Text in the MIDDLE '
        + 'of a frame lands here rather than under the share above — it is '
        + 'small, and there is no crop that removes it. Needs the 🔳 Safe zone '
        + 'pass AND its text extra from Setup: without the extra the pass '
        + 'measures bands only and stores no usable-frame reading at all, so '
        + 'this cut flags nothing rather than clearing every shot.' },
  ]
}

/** Which of the THREE audio states a clip is in: 'ok' | 'none' | 'unmeasured'.
 *
 * They must not collapse. "No track" is a property of the file and there is
 * nothing to do about it; "silent" is a defect in a clip that has a track; and
 * "nobody measured it" is fixed by re-running the pass. A bank measured before
 * the audio metric shipped carries NO audio keys at all — that absence is the
 * signature of an earlier pass, and reading it as "no sound" would tell the user
 * their footage is mute when nothing has listened to it yet. */
export function audioState(clip) {
  const m = clip?.metrics
  if (!m || !m.audio_state) return 'unmeasured'
  return m.audio_state === 'ok' ? 'ok' : m.audio_state
}

/** What to DO about this clip's audio state, or '' when there is nothing to say.
 * A state with no remedy attached is trivia. */
export function audioNote(clip) {
  const state = audioState(clip)
  if (state === 'unmeasured') {
    return 'Sound was not measured on this clip — re-measure the bank to include it.'
  }
  if (state === 'none') return 'This file carries no sound track.'
  if (state === 'unreadable') return 'This clip’s sound track could not be read.'
  return ''
}

/** "-14.2 dBFS · 42% silent" — the two audio numbers, in units people read.
 * Empty for anything not measured: a blank is honest, a "0.00" is not. */
export function audioSummary(metrics) {
  if (!metrics || metrics.audio_state !== 'ok') return ''
  const parts = []
  if (Number.isFinite(Number(metrics.rms_dbfs))) {
    parts.push(`${Number(metrics.rms_dbfs).toFixed(1)} dBFS`)
  }
  if (Number.isFinite(Number(metrics.silence_ratio))) {
    parts.push(`${Math.round(Number(metrics.silence_ratio) * 100)}% silent`)
  }
  return parts.join(' · ')
}

/** One sentence for the dry-run result. Real numbers, per-rule detail, and a
 * warning tone once the cut would take most of the bank. */
export function cutSummary(dryRun, totalClips) {
  const flagged = dryRun?.total_flagged || 0
  if (!flagged) return 'These cuts would remove nothing — no clips flagged.'
  const parts = Object.entries(dryRun)
    .filter(([k, v]) => k !== 'total_flagged' && v > 0)
    .map(([k, v]) => `${k}: ${v}`)
    .join(', ')
  const head = `${flagged} of ${totalClips} clips would be flagged (${parts}).`
  if (flagged > totalClips / 2) {
    return `⚠ ${head} That is most of the bank — check the thresholds before applying.`
  }
  return head
}

/** A draft copy of the saved cuts, for the panel to edit. The grid keeps
 * flagging against the SAVED cuts while the user types; nothing reaches config
 * until Apply — editing live would re-flag the bank on every keystroke,
 * including through values the user is merely passing through. */
export function draftThresholds(saved) {
  const draft = {}
  for (const f of thresholdFields()) {
    const v = saved ? saved[f.key] : null
    draft[f.key] = (v === undefined || v === null) ? null : Number(v)
  }
  return draft
}

/** One edit, immutably. An EMPTY input disables the cut (null) — zero is a real
 * threshold and the two must never be confused. Garbage keeps the previous
 * value: mid-typing states ("0.", "-") must not wipe a number out. */
export function editThreshold(draft, key, raw) {
  const next = { ...draft }
  const text = String(raw ?? '').trim()
  if (text === '') {
    next[key] = null
    return next
  }
  const value = Number(text)
  if (Number.isFinite(value)) next[key] = value
  return next
}

/** The dry-run/apply payload: only the backend's known keys, only active cuts.
 * Anything else a component stuffed into the draft object stays behind. */
export function payloadFromDraft(draft) {
  const payload = {}
  for (const f of thresholdFields()) {
    if (draft[f.key] !== null && draft[f.key] !== undefined) {
      payload[f.key] = draft[f.key]
    }
  }
  return payload
}
