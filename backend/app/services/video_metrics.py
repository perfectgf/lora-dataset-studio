"""One decode per clip, every metric out of it — and how per-frame numbers become
a per-clip verdict.

WHY ONE PASS. Decoding is roughly 85 % of this lane's cost, measured on a real
4.5-hour corpus once shot detection moved to the GPU. Motion, exposure, sharpness
and freeze detection are each nearly free once a frame is in hand, so writing them
as four passes would multiply the only expensive part by four. Everything below
consumes ONE list of per-frame readings.

WHY THE AGGREGATION IS NOT UNIFORM, WHICH IS THE SUBTLE PART. The model trains on
EVERY frame, so the useful question is rarely "what is the average?":

  exposure  → MIN. A half-second fade in the middle ruins the sample. An average
              of 0.87 hides a stretch of 0.02 completely.
  sharpness → p90. "Does real sharpness exist in this clip?" Legitimate motion
              blur drags a mean down, so a threshold on a mean rejects exactly the
              clips with the most interesting movement. p90 rather than the p75
              a first draft used, and the arithmetic decides it: a clip that is
              sharp for a fifth of its length is perfectly usable, and p75 sits
              inside the blurred four fifths and calls it soft. p90 finds it,
              while still ignoring a single fluke frame that a max would trust.
  motion    → mean AND a high percentile. "Does anything move at all?" and "is it
              thrashing?" are different questions and cannot share a number.
  freeze    → the SHARE of near-still frames. A frozen second inside a lively shot
              leaves the mean perfectly healthy; only the share reveals it.
  aesthetic → MEAN, and it is the one entry here that does NOT follow the rule
              above. See `aesthetic_of` for the argument: it is the only quantity
              in this list that barely moves inside a shot, so its per-frame
              readings are noise around one value rather than moments the model
              learns separately.

WHY RAW SCORES ARE STORED AND VERDICTS ARE NOT. Same philosophy as the image
bank: retuning a threshold then re-sorts the bank with no rescan. It matters more
here than there, because the published thresholds DO NOT TRANSFER — the floor a
public pipeline uses lands at the 7th percentile of this machine's own test bank.
A cut belongs to the bank being worked on, not to a constant.
"""

import math

# A frame counts as "still" below this normalised motion magnitude. Not a quality
# threshold — a near-zero test, used only to measure the SHARE of frozen frames.
# The value is a floor on numerical noise, not a judgement about movement.
_STILL_EPSILON = 1e-5

# An audio window counts as "silent" at or below this level, in dBFS. Like
# _STILL_EPSILON this is a near-zero test and NOT a quality cut: -60 dBFS is four
# thousandths of full scale, which is below the noise floor of any real recording
# and above the exact zero that only a synthesised mute produces. Room tone,
# handling noise and a distant conversation all sit far above it. What counts as
# "too quiet" is a judgement, and it lives in the `audio_floor` threshold where
# the user can move it.
_SILENCE_DBFS = -60.0

# Where the dBFS scale stops. log10(0) is -inf, which is not a number JSON can
# carry and not a value anything downstream can compare. A digital-silence clip
# reports the floor, which is honest (it IS at or below it) and comparable.
_DBFS_FLOOR = -120.0

# Sentinel for "this pass did not look at the audio", as distinct from None,
# which means "there is no audio track". The two must not collapse: a summary
# written before this metric existed carries NO audio keys at all, and a reader
# has to be able to tell that from a measured absence — one is fixed by a
# re-measure, the other is a property of the file. See summarise().
UNMEASURED = object()

# Exposure band a frame must sit in to represent its clip (thumbnail, embedding).
# Outside it, violent local contrast comes from a flash or a dissolve edge, not
# from real detail. Deliberately loose — this guards against degenerate frames,
# it does not judge the clip (luma_min does that).
_LUMA_SANE_LOW = 0.06
_LUMA_SANE_HIGH = 0.97

# The clip-level sharpness score, named once. Other passes rank shots by it (the
# dedup pass keeps the sharpest member of a near-duplicate pile), and a second
# copy of the string is how one of them would keep reading a key nobody writes.
SHARPNESS_KEY = 'sharpness_p90'

# Keys OTHER passes write into the same metrics_json blob. They describe the same
# clip and are read by the same `verdicts()`, but they are produced by passes with
# their own cost, their own cancel and their own reasons to fail — so the metrics
# scan, which rewrites the blob wholesale, must carry them across rather than
# erase a dedup or watermark verdict every time somebody re-measures a bank.
ADVISORY_KEYS = ('duplicate_group', 'duplicate_of',
                 'watermark_score', 'watermark_state',
                 # 🎨 The look score. Produced from the search vectors, so a
                 # re-measure (which decodes frames and knows nothing about
                 # CLIP) must carry it across rather than silently drop it and
                 # send the whole bank back through the aesthetic pass.
                 'aesthetic_score',
                 # 🔳 The safe zone. Bands, burned-in text and what is left of
                 # the frame — three decoded frames and a CPU OCR per shot, the
                 # most expensive advisory pass in the lane, so dropping it on a
                 # re-measure would be the costliest silence of the five. Pinned
                 # against video_safe_zone.OWNED_KEYS by a test: a key that pass
                 # writes and this list does not carry is erased by the next
                 # quality scan with nothing anywhere to see.
                 'safe_zone_state', 'safe_zone_frames', 'safe_bands',
                 'bars_ratio', 'text_coverage', 'safe_rect', 'safe_area',
                 # 🩻 The defect sweep. Produced by ONE ffmpeg pass over the
                 # whole source file, so re-measuring a single clip cannot
                 # reproduce them — dropping them here would send the entire
                 # bank back through a decode of every file it holds, which is
                 # the most expensive silence of the six. Pinned against
                 # video_defect_sweep.OWNED_KEYS by a test.
                 'defect_state', 'defect_frames', 'dup_frame_ratio',
                 'block_score', 'blur_score',
                 # 🤖 The AI check. Sixteen contiguous frames decoded, cropped
                 # and pushed through a vision transformer per shot — 0.83 s
                 # each, measured, so a bank of a few thousand is tens of
                 # minutes. Dropping it on a re-measure would be the most
                 # expensive silence in this list after the defect sweep's, and
                 # unlike that one it would also lose the STATE that says why a
                 # shot has no score. Pinned against
                 # video_ai_check.OWNED_KEYS by a test.
                 'ai_check_state', 'ai_check_frames', 'motion_irregularity')


def merge_advisory(previous, summary):
    """`summary` with the advisory verdicts of `previous` carried over.

    The metrics pass measures ONE clip's frames and knows nothing about its
    neighbours or its watermarks; re-running it must not be a way to silently
    lose the two verdicts that took a separate pass to produce. The bounds are
    unchanged by a re-measure — a re-CUT clears the whole blob, which is a
    different gesture and the correct one."""
    out = dict(summary)
    for key in ADVISORY_KEYS:
        if previous and key in previous:
            out[key] = previous[key]
    return out


def percentile(values, p):
    """Linear-interpolation percentile. None for an empty list — never 0.0, which
    would be a measurement rather than the absence of one."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = p * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def to_dbfs(amplitude):
    """A 0..1 amplitude as dBFS, floored rather than infinite. 1.0 is 0 dBFS."""
    a = float(amplitude or 0.0)
    if a <= 0:
        return _DBFS_FLOOR
    return max(_DBFS_FLOOR, 20.0 * math.log10(a))


def summarise_audio(windows):
    """Collapse per-window RMS readings into the audio numbers stored on a clip.

    `windows` is a list of RMS amplitudes in 0..1, one per short window of the
    clip — or None when the file HAS NO AUDIO TRACK, or [] when it has one that
    could not be decoded. Those three inputs produce three different states and
    that is the whole point of this function:

      'ok'         → measured; silence_ratio and rms_dbfs are real numbers.
      'none'       → no track. Not silence: a Wan dataset has no audio by design
                     (video_targets.py forces `-an`), and flagging all of it as
                     silent would be an app-wide false alarm.
      'unreadable' → a track we could not read. Not silence either: the same
                     distinction summarise() already makes between an unreadable
                     clip and a black one, and for the same reason — collapsing
                     them filters a file for the wrong reason.

    TWO NUMBERS FOR TWO QUESTIONS, exactly like motion. `rms_dbfs` is the level
    over the whole clip and answers "is this quiet"; `silence_ratio` is the SHARE
    of silent windows and answers "is part of it missing". A dropout in the middle
    of a loud clip leaves the level perfectly healthy — only the share reveals it,
    which is the same argument that put `freeze_ratio` next to `motion_mean`.
    """
    if windows is None:
        return {'audio_state': 'none', 'silence_ratio': None, 'rms_dbfs': None}
    if not windows:
        return {'audio_state': 'unreadable', 'silence_ratio': None,
                'rms_dbfs': None}
    silent = sum(1 for w in windows if to_dbfs(w) <= _SILENCE_DBFS)
    # RMS over the clip is the root of the MEAN OF SQUARES, not the mean of the
    # per-window RMS values: averaging amplitudes under-reports a clip whose
    # energy is concentrated in part of it, which is precisely the shape of the
    # material this measures.
    mean_square = sum(float(w) ** 2 for w in windows) / len(windows)
    return {
        'audio_state': 'ok',
        'silence_ratio': silent / len(windows),
        'rms_dbfs': round(to_dbfs(math.sqrt(mean_square)), 2),
    }


def aesthetic_of(frame_scores):
    """Collapse a shot's per-frame LAION aesthetic ratings (~1..10) into ONE
    number — the score `video_aesthetic` stores and `aesthetic_floor` cuts on.

    MEAN, and it is the only aggregation in this module NOT chosen by "which
    moment does the model also learn". Exposure takes the MIN because half a
    second of black IS in the training data; sharpness takes p90 because
    legitimate motion blur must not sink a clip that is sharp somewhere. Look is
    a different KIND of quantity: framing, lighting, palette and subject barely
    move inside one shot — a cut is precisely where they change, which is why
    shots exist at all. Three readings of one shot are therefore three noisy
    measurements of ONE value, and the mean is what reduces that noise.

    THE HONEST LIMIT, and why the two obvious alternatives are worse HERE. The
    public curation pipelines this head comes from sample ~1 fps; this lane
    embeds three frames chosen by POSITION and by SHARPNESS, and that biased
    sample is what makes the choice:

      * MIN would mostly report an EDGE frame. 'start' and 'end' sit
        EDGE_MARGIN_S (0.25 s) inside the bounds — close enough to a cut that
        dissolves and half-faded frames land there disproportionately, which is
        the very reason that margin exists. A min would answer "how ugly is the
        worst boundary" rather than "how does this shot look".
      * MAX would collapse onto the 'key' frame, which IS the metrics pass's
        ambassador — the sharpest sanely-exposed frame of the shot. That would
        quietly turn this into a second reading of sharpness_p90 instead of an
        independent signal.

    None for an empty list, never 0.0: nothing was measured, and a 0 on a 1..10
    scale is the strongest possible claim that a shot is hideous. Same rule as
    `percentile` above, and the same reason.
    """
    values = [float(s) for s in (frame_scores or []) if s is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarise(frames, fps, audio=UNMEASURED):
    """Collapse per-frame readings into the numbers stored on a clip.

    `frames` is a list of {'luma', 'sharp', 'motion'} in decode order. An empty
    list is `unreadable`, NOT a clip of zeros: collapsing the two would make a
    file we could not open look like a perfectly black, perfectly still clip, and
    it would then be filtered out for the wrong reason.
    """
    if not frames:
        # No audio claim either: if the segment could not be decoded, saying "no
        # track" about it would be a claim about a file we could not open.
        return {'metrics_state': 'unreadable', 'motion_mean': None,
                'motion_p95': None, 'luma_min': None, 'luma_mean': None,
                'sharpness_p90': None, 'freeze_ratio': None,
                'sharpest_frame_s': None, 'first_frame_sharpness': None}

    lumas = [f['luma'] for f in frames]
    sharps = [f['sharp'] for f in frames]
    motions = [f['motion'] for f in frames]
    # The ambassador frame: sharpest AMONG frames with sane exposure. The score
    # uses p90 so one lucky frame cannot vouch for the clip, but the frame CHOICE
    # is an argmax — and an overexposed flash or a dissolve-to-black edge carries
    # huge local contrast while being useless to look at or to embed. When every
    # frame violates the constraint (a clip that is all flash), the plain argmax
    # returns: a bad thumbnail beats no thumbnail, and the flags tell the story.
    candidates = [i for i in range(len(frames))
                  if _LUMA_SANE_LOW <= lumas[i] <= _LUMA_SANE_HIGH]
    pool = candidates or range(len(frames))
    sharpest = max(pool, key=sharps.__getitem__)

    out = {
        'metrics_state': 'ok',
        # Two numbers for two questions — see the module docstring.
        'motion_mean': sum(motions) / len(motions),
        'motion_p95': percentile(motions, 0.95),
        # The WORST moment, because that is the one the model also learns.
        'luma_min': min(lumas),
        'luma_mean': sum(lumas) / len(lumas),
        # "Is there real sharpness anywhere", not "is it sharp on average".
        'sharpness_p90': percentile(sharps, 0.90),
        'freeze_ratio': sum(1 for m in motions if m <= _STILL_EPSILON) / len(motions),
        # Frame 0 measured on its own: for image-to-video targets it IS the
        # conditioning image, and nobody chooses it — it is whatever the cut
        # starts on. A gorgeous clip with a blurred first frame is a bad i2v
        # clip; the number was already computed, storing it is free.
        'first_frame_sharpness': sharps[0],
        # Free, since every frame was measured anyway, and a better thumbnail than
        # the middle frame — a shot boundary is where a cut just happened, so the
        # middle is a guess while this is a measurement.
        'sharpest_frame_s': sharpest / float(fps) if fps else None,
    }
    # The audio keys are ADDED only when the audio was looked at. Their absence
    # is the on-disk signature of a summary written by a pass that predates this
    # metric, and every reader treats it as "not measured" rather than as
    # "silent" — which is what keeps a new cut from retro-flagging a bank that
    # was measured last week. An explicit re-measure is what fills them in.
    if audio is not UNMEASURED:
        out.update(summarise_audio(audio))
    return out


# Every cut `verdicts()` below honours, in panel order. THE canonical list, and
# it exists because there were three copies of it — the config reader, the
# dry-run route's allow-list, and the frontend's panel table — and they had
# already drifted: `first_frame_floor` was supported here from wave 2 and named
# by none of the other two, so the `soft_start` flag it feeds could not fire in
# the app at all. A cut that exists only in this file is not a feature.
# Anything added here must gain a row in videoMetricsFilter.thresholdFields()
# and a label in FLAG_LABELS; a test pins the two lists against each other.
THRESHOLD_KEYS = ('min_duration_s', 'motion_floor', 'motion_ceiling',
                  'luma_floor', 'freeze_max', 'sharpness_floor',
                  'first_frame_floor', 'silence_max', 'audio_floor',
                  'watermark_max', 'aesthetic_floor',
                  'bars_max', 'text_coverage_max', 'safe_area_min',
                  'dup_frames_max', 'block_max', 'blur_max',
                  'motion_irregularity_floor')


def verdicts(scores, thresholds, duration_s=None):
    """The flags a clip carries RIGHT NOW, given the cuts in force. Computed at
    read time from the raw scores, so moving a cut re-sorts the bank instantly.

    An unmeasured score never produces a flag. Absence of measurement must not
    read as a defect — that is how a scan that failed quietly becomes a bank that
    appears to have filtered half its clips.

    TWO SOURCES, DELIBERATELY SEPARATE. `scores` is what the metrics pass wrote
    into metrics_json; `duration_s` comes off the clip ROW, derived from the
    bounds the detector set. It is passed in rather than looked up here because
    this module never touches the database — and it is a separate argument rather
    than a key smuggled into `scores` because the two have different lifetimes:
    re-cutting a clip changes its duration and INVALIDATES its scores.
    """
    flags = set()
    scores = scores or {}

    # Duration first, and it is the only rule here that can fire on a clip
    # nobody has measured — see `min_duration_s` in THRESHOLD_KEYS.
    min_duration = thresholds.get('min_duration_s')
    if (duration_s is not None and min_duration is not None
            and duration_s < min_duration):
        # NOT `too_short`, which the promotion already uses for its own refusal.
        # The two are different claims about the same clip and must not share a
        # word: `too_short` says the TARGET PROFILE cannot be fed by this clip —
        # an arithmetic fact about frames and fps that no setting in this panel
        # moves — while `brief` says the USER decided shots this short are not
        # worth their triage time. Collapsing them would suggest that lowering
        # this field buys a clip its way into a dataset. It does not.
        flags.add('brief')

    motion = scores.get('motion_mean')
    floor = thresholds.get('motion_floor')
    if motion is not None and floor is not None and motion < floor:
        flags.add('still')

    luma = scores.get('luma_min')
    luma_floor = thresholds.get('luma_floor')
    if luma is not None and luma_floor is not None and luma < luma_floor:
        flags.add('black')

    freeze = scores.get('freeze_ratio')
    freeze_max = thresholds.get('freeze_max')
    if freeze is not None and freeze_max is not None and freeze > freeze_max:
        # Deliberately its own flag, not a stillness one: a still clip is useless,
        # while a clip with a frozen stretch can be re-cut around it. Different
        # defects, different remedies.
        flags.add('freeze')

    agitated = scores.get('motion_p95')
    ceiling = thresholds.get('motion_ceiling')
    if agitated is not None and ceiling is not None and agitated > ceiling:
        flags.add('agitated')

    sharp = scores.get('sharpness_p90')
    sharp_floor = thresholds.get('sharpness_floor')
    if sharp is not None and sharp_floor is not None and sharp < sharp_floor:
        flags.add('soft')

    first = scores.get('first_frame_sharpness')
    first_floor = thresholds.get('first_frame_floor')
    if first is not None and first_floor is not None and first < first_floor:
        # Advisory like every flag, and mostly meaningful when the target is
        # image-to-video — the first frame is that lane's conditioning image.
        flags.add('soft_start')

    # Audio. Both read scores that are None on a clip with no track AND absent
    # entirely on a clip measured before this metric existed — the `is not None`
    # guard covers both, which is why no separate audio_state check is needed
    # here and why a cut can be set on a half-measured bank without lying about
    # the half it cannot see.
    silence = scores.get('silence_ratio')
    silence_max = thresholds.get('silence_max')
    if silence is not None and silence_max is not None and silence > silence_max:
        flags.add('silent')

    level = scores.get('rms_dbfs')
    audio_floor = thresholds.get('audio_floor')
    if level is not None and audio_floor is not None and level < audio_floor:
        # Its own flag, not a silent one: a quiet clip can be normalised and a
        # silent one cannot be rescued at all. Same split as freeze vs still.
        flags.add('quiet')

    # ── The two verdicts other passes produced ───────────────────────────────
    # They read the same `scores` blob and obey the same rule as everything
    # above: no measurement, no flag. What differs is only where the number came
    # from — a dedup over the search vectors, and a classifier over one frame.

    # ✂ Near-duplicate. The flag is the ABSENCE of representative status, not the
    # presence of a group: every member of a pile carries `duplicate_group` (the
    # grid says "1 of 3"), and exactly one of them carries `duplicate_of: None`
    # because it is the one being kept. Flagging the whole pile would tell the
    # user to drop all of it, which is the opposite of what the pass found.
    if scores.get('duplicate_of') is not None:
        flags.add('duplicate')

    # 🔖 Watermark. Read at the same read time as every other cut, so moving the
    # threshold re-sorts the bank with no rescan — and a frame the detector could
    # not judge stores a None score and is therefore never flagged, exactly like
    # a clip with no audio track is never 'silent'.
    mark = scores.get('watermark_score')
    mark_max = thresholds.get('watermark_max')
    if mark is not None and mark_max is not None and mark > mark_max:
        flags.add('watermark')

    # 🎨 Look. The LAION aesthetic head over the frame vectors 🔎 Find scenes
    # cached — the SAME head, the same weights and the same 1..10 scale the image
    # bank's ✨ Score pass puts on a still, so the flag wears the image lane's
    # name (`low_aesthetic`) because it means the same thing on both surfaces.
    # A shot the aesthetic pass never scored carries NO key at all and is never
    # flagged: no measurement, no verdict, exactly like every rule above.
    look = scores.get('aesthetic_score')
    look_floor = thresholds.get('aesthetic_floor')
    if look is not None and look_floor is not None and look < look_floor:
        flags.add('low_aesthetic')

    # ── 🔳 The safe zone: three cuts on three different findings ─────────────
    # They are deliberately NOT one. A padded clip, a subtitled clip and a clip
    # with nothing croppable left are three states with three different
    # remedies — crop it, drop it, forget it — and a single number could only
    # ever recommend the last one. Same split as freeze vs still, and as quiet
    # vs silent. All three obey the rule every cut above obeys: no measurement,
    # no flag, so a bank measured without the OCR engine (which stores its bands
    # and NO text keys) can never be flagged for text it was never shown.

    # Container. `bars_ratio` is the same arithmetic the image bank's own
    # `bars_ratio` column carries, so the number a user calibrated on stills
    # means the same thing here.
    bars = scores.get('bars_ratio')
    bars_max = thresholds.get('bars_max')
    if bars is not None and bars_max is not None and bars > bars_max:
        flags.add('letterboxed')

    # Burned-in text — subtitles, chyrons, lower thirds. Only zones that held
    # still ACROSS the shot's frames feed this; a shop sign is scene content and
    # never counted (see video_safe_zone_geometry.structural_text).
    text = scores.get('text_coverage')
    text_max = thresholds.get('text_coverage_max')
    if text is not None and text_max is not None and text > text_max:
        flags.add('burned_text')

    # What is LEFT once both are cut away, as a share of the frame. The one cut
    # here whose published numbers are worth quoting: HunyuanVideo 1.5 discards
    # a clip whose crop keeps under 60 % of the frame. Quoted in the hint, and
    # in no default — that figure was set for a web-scale crawl.
    area = scores.get('safe_area')
    area_min = thresholds.get('safe_area_min')
    if area is not None and area_min is not None and area < area_min:
        flags.add('small_safe_zone')

    # ── 🩻 The defect sweep: three cuts on what a re-encode left behind ──────
    # One ffmpeg pass per SOURCE FILE produced all three (video_defect_sweep),
    # and they are three flags rather than one for the same reason the safe
    # zone's are: three findings with three different remedies — re-cut around
    # the stall, drop the file, or find a better copy of it. All three obey the
    # rule every cut above obeys: no measurement, no flag, so a bank swept
    # before this shipped, or a shot too short to have been sampled, is never
    # flagged for something nothing looked at.

    # Frames delivered twice. NOT the same finding as `freeze`, and the two must
    # not be collapsed: that one reads motion vectors and says nothing MOVED,
    # this one says the same picture ARRIVED twice. A 24-into-30 pulldown
    # produces the second with no trace of the first, and it is the single most
    # common thing wrong with re-uploaded footage.
    dup = scores.get('dup_frame_ratio')
    dup_max = thresholds.get('dup_frames_max')
    if dup is not None and dup_max is not None and dup > dup_max:
        flags.add('dup_frames')

    # The macroblock grid showing through. Never legitimate — no camera, lens or
    # light produces one — so unlike every other cut here the only question is
    # how much of it the user will tolerate.
    block = scores.get('block_score')
    block_max = thresholds.get('block_max')
    if block is not None and block_max is not None and block > block_max:
        flags.add('blocky')

    # Wide edges at FULL SIZE, which is the half `soft` cannot see: that flag
    # reads a Laplacian computed on a 160-pixel-wide analysis copy, and footage
    # upscaled from something smaller is identical to the real thing at 160
    # pixels (measured: three files, 354.35 / 353.69 / 353.72). Deliberately its
    # own flag rather than more evidence for `soft` — a clip can be sharp at
    # analysis size and mush at full size, and only one of the two is worth
    # re-downloading a better copy over.
    blur = scores.get('blur_score')
    blur_max = thresholds.get('blur_max')
    if blur is not None and blur_max is not None and blur > blur_max:
        flags.add('blurry')

    # ── 🤖 The AI check: the one cut here whose polarity is inverted ─────────
    # A LOW score is the suspicious one. `motion_irregularity` measures how
    # erratically a shot's motion changes over two seconds, and the finding the
    # method rests on is that real footage is erratic while a generated clip is
    # smoother than the world — so this is a FLOOR, like `sharpness_floor` and
    # `aesthetic_floor`, and raising it flags more shots. Writing it as a `_max`
    # would flag every handheld shot in the bank and clear every generated one.
    #
    # `maybe_generated`, and the hedge in the name is the honest part rather
    # than politeness: on re-encoded material — which a scraped bank is by
    # construction — the best blind-evaluated detector in the field scores about
    # three in four. Advisory, in no default, and no code path anywhere rejects
    # or deletes on it. A shot the pass could not measure (too short for the
    # window, frames unreadable) carries a STATE and no score, and is therefore
    # never flagged — the same rule as every cut above.
    smooth = scores.get('motion_irregularity')
    smooth_floor = thresholds.get('motion_irregularity_floor')
    if (smooth is not None and smooth_floor is not None
            and smooth < smooth_floor):
        flags.add('maybe_generated')

    return flags


def dry_run(bank_clips, thresholds):
    """How many clips EACH cut would remove, plus how many would be removed in
    total — before anything is committed.

    Each item is either a scores dict, or a ``(scores, duration_s)`` pair when
    the caller knows the bounds — the same two sources `verdicts()` keeps apart,
    made visible at the call site. A bare dict is not a shortcut for "zero
    seconds": it means the duration is unknown, and an unknown duration is never
    brief.

    Never a silent filter, and the count is per RULE rather than a lump sum: a
    public dataset pipeline once kept 47 clips out of 1493 with one mis-set
    threshold and only discovered it afterwards. A single total would have looked
    equally alarming for a filter that was working correctly.

    `total_flagged` counts CLIPS, not flags, so a clip caught by two rules is not
    counted twice — otherwise the preview overstates the damage.
    """
    counts = {}
    flagged_clips = 0
    for item in bank_clips:
        scores, duration_s = item if isinstance(item, tuple) else (item, None)
        flags = verdicts(scores, thresholds, duration_s=duration_s)
        if flags:
            flagged_clips += 1
        for flag in flags:
            counts[flag] = counts.get(flag, 0) + 1
    counts['total_flagged'] = flagged_clips
    return counts
