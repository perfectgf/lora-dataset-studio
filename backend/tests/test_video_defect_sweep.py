"""🩻 The defect sweep — duplicated frames, compression blocks, soft edges.

NO FFMPEG RUNS HERE except in the one test that says so and skips itself without
a binary. Everything else is a pure function over CAPTURED output: the blocks
pasted below were produced by this repo's own chain on a forged file (a clip with
a two-second frozen stretch in the middle, encoded at CRF 16), so the parser is
tested against what ffmpeg really prints rather than against what we remember it
printing — including the two shapes that broke the first draft: a `pts_time` with
no decimal point (`pts_time:2`), and frames that carry a key we added ourselves.

WHY THAT SEPARATION IS THE POINT. The arithmetic decides whether a user's bank
gets filtered; the subprocess decides nothing. Testing them together would mean
every assertion about a percentile needed a video file, a binary and eight
seconds, and the ones about a clip STRADDLING a dropped-frame zone could not be
written at all — you cannot ask ffmpeg for a shot boundary in a specific place.
"""
import json

import pytest

from app.services import video_defect_sweep as sweep
from app.services import video_metrics, video_probe


# --- captured output, from a real run on a forged file -----------------------------
#
# The frozen stretch starts just after pts_time 2: `ALL` lists every frame across
# the boundary, `KEPT` stops at 2 because mpdecimate drops what follows. That gap
# is what every mapping test below leans on.

ALL_TEXT = """frame:48   pts:24576   pts_time:1.92
lds.f=1
frame:49   pts:25088   pts_time:1.96
lds.f=1
frame:50   pts:25600   pts_time:2
lds.f=1
frame:51   pts:26112   pts_time:2.04
lds.f=1
frame:52   pts:26624   pts_time:2.08
lds.f=1
frame:53   pts:27136   pts_time:2.12
lds.f=1
frame:54   pts:27648   pts_time:2.16
lds.f=1
frame:55   pts:28160   pts_time:2.2
lds.f=1
"""

KEPT_TEXT = """frame:48   pts:24576   pts_time:1.92
lds.f=1
frame:49   pts:25088   pts_time:1.96
lds.f=1
frame:50   pts:25600   pts_time:2
lds.f=1
"""

QUALITY_TEXT = """frame:8    pts:24576   pts_time:1.92
lds.f=1
lavfi.block=13.819922
lavfi.blur=4.593876
frame:9    pts:27648   pts_time:2.16
lds.f=1
lavfi.block=15.225635
lavfi.blur=7.552114
"""


# --- the parser ---------------------------------------------------------------------

def test_the_parser_reads_real_ffmpeg_output():
    records = sweep.parse_records(QUALITY_TEXT)

    assert [seconds for seconds, _v in records] == [1.92, 2.16]
    assert records[0][1]['lavfi.block'] == pytest.approx(13.819922)
    assert records[1][1]['lavfi.blur'] == pytest.approx(7.552114)


def test_a_timestamp_with_no_decimal_point_is_still_a_timestamp():
    """ffmpeg prints `pts_time:2`, not `2.0`. A parser that split on '.' — the
    obvious way to find the number — loses exactly one frame per whole second."""
    assert 2.0 in sweep.frame_times(ALL_TEXT)
    assert len(sweep.frame_times(ALL_TEXT)) == 8


def test_a_frame_with_no_timestamp_is_dropped_rather_than_placed():
    """`pts_time:N/A` happens on streams whose timestamps ffmpeg cannot express.
    A reading with no position cannot be attributed to a clip, and giving it 0.0
    would attribute it to whichever shot starts the file."""
    text = ('frame:0    pts:0       pts_time:N/A\n'
            'lavfi.block=99.0\n'
            'frame:1    pts:512     pts_time:0.04\n'
            'lavfi.block=1.0\n')
    records = sweep.parse_records(text)

    assert [seconds for seconds, _v in records] == [0.04]
    assert records[0][1]['lavfi.block'] == 1.0


def test_an_unparseable_line_costs_that_line_and_not_the_file():
    text = ('frame:0    pts:0       pts_time:0\n'
            'lavfi.block=not-a-number\n'
            'some noise ffmpeg decided to print\n'
            'lavfi.blur=4.5\n')
    records = sweep.parse_records(text)

    assert len(records) == 1
    assert 'lavfi.block' not in records[0][1]
    assert records[0][1]['lavfi.blur'] == 4.5


def test_empty_output_parses_to_nothing_rather_than_raising():
    assert sweep.parse_records('') == []
    assert sweep.frame_times(None) == []


# --- mapping timestamps onto clips ---------------------------------------------------

def _summary(start_s, end_s, all_text=ALL_TEXT, kept_text=KEPT_TEXT,
            quality_text=QUALITY_TEXT):
    return sweep.summarise_clip(sweep.frame_times(all_text),
                               sweep.frame_times(kept_text),
                               sweep.parse_records(quality_text),
                               start_s, end_s)


def test_a_clip_entirely_before_the_frozen_stretch_reports_no_duplicates():
    out = _summary(1.90, 2.02)          # frames 1.92, 1.96, 2.00 — all kept

    assert out['defect_state'] == 'ok'
    assert out['dup_frame_ratio'] == 0.0


def test_a_clip_straddling_a_dropped_frame_zone_reports_the_share_it_holds():
    """The case no real file can be asked for: a shot boundary landing INSIDE a
    run of duplicates. Three of this window's six frames survived mpdecimate."""
    out = _summary(1.94, 2.18)          # 1.96 2.00 2.04 2.08 2.12 2.16 -> 2 kept

    assert out['dup_frame_ratio'] == pytest.approx(4 / 6, abs=1e-4)


def test_a_clip_wholly_inside_the_frozen_stretch_is_all_duplicates():
    out = _summary(2.02, 2.22)          # 2.04 .. 2.20, none kept

    assert out['dup_frame_ratio'] == 1.0


def test_the_window_is_half_open_so_a_shared_boundary_is_counted_once():
    """Adjacent shots touch. A closed interval would count the frame on the seam
    into both, inflating both denominators — and this pass, unlike the metrics
    scan, feeds every clip of a file from ONE decode, so the seam is shared."""
    times = sweep.frame_times(ALL_TEXT)

    # Five frames sit in [1.90, 2.10): 1.92 1.96 2.00 2.04 2.08. Split at 2.00
    # the two halves hold 2 and 3 of them — never 3 and 3, which is what a closed
    # interval would report by giving the frame at 2.00 to both.
    assert len(sweep._in_window(times, 1.90, 2.00)) == 2
    assert len(sweep._in_window(times, 2.00, 2.10)) == 3
    assert (len(sweep._in_window(times, 1.90, 2.00))
            + len(sweep._in_window(times, 2.00, 2.10))
            == len(sweep._in_window(times, 1.90, 2.10)))
    assert _summary(1.90, 2.10)['defect_state'] == 'ok'


def test_a_clip_no_frame_landed_in_is_unreadable_and_carries_no_numbers():
    """The rule the whole lane keeps: absence of measurement is a STATE, never a
    zero. A 0.0 here would claim a shot nobody looked at is perfectly clean."""
    out = _summary(9.0, 9.5)

    assert out == {'defect_state': 'unreadable'}
    assert 'dup_frame_ratio' not in out
    assert 'block_score' not in out


def test_a_clip_the_sampler_missed_keeps_its_dup_ratio_and_no_quality_scores():
    """A shot shorter than the sampling interval falls between two sampled
    frames. Its duplicate share is real — that half looks at every frame — and
    its quality scores are ABSENT rather than zero."""
    out = _summary(1.98, 2.10)          # holds frames, holds no quality sample

    assert out['defect_state'] == 'ok'
    assert out['defect_frames'] == 0
    assert 'dup_frame_ratio' in out
    assert 'block_score' not in out and 'blur_score' not in out


# --- the aggregations, on forged values ----------------------------------------------

def _quality(*pairs):
    return [(seconds, {'lavfi.block': block, 'lavfi.blur': blur})
            for seconds, block, blur in pairs]


def test_the_block_score_is_the_worst_tenth_not_the_average():
    """Blocking is never legitimate, so the question is whether it exists
    anywhere — the worst decile answers it, and an average buries a blocky third
    of a shot under two clean thirds."""
    quality = _quality(*[(i / 10, 2.0, 5.0) for i in range(9)],
                      (0.9, 90.0, 5.0))
    out = sweep.summarise_clip([0.0], [0.0], quality, 0.0, 1.0)

    assert out['block_score'] > 10          # the spike is seen
    assert out['block_score'] < 90          # and not trusted on its own


def test_the_blur_score_is_the_sharpest_tenth_so_a_fast_pan_is_not_called_blurry():
    """The asymmetry with block_score, and the reason for it: softness IS
    sometimes a choice. A shot that is sharp at the start and smeared through a
    pan must read SHARP — a p90 would flag exactly the clips with the most
    movement, the false positive `sharpness_p90` chose its own p90 to avoid."""
    quality = _quality((0.0, 2.0, 4.0), (0.1, 2.0, 4.2),
                      *[(i / 10, 2.0, 20.0) for i in range(2, 10)])
    out = sweep.summarise_clip([0.0], [0.0], quality, 0.0, 1.0)

    assert out['blur_score'] < 5            # judged on its sharpest moments
    assert out['blur_score'] == pytest.approx(4.0, abs=0.2)


def test_a_uniformly_soft_clip_is_still_caught_by_the_sharpest_tenth():
    """The other half of the same argument: p10 must not become "never flags".
    Footage upscaled from something smaller is soft in EVERY frame."""
    quality = _quality(*[(i / 10, 2.0, 9.5) for i in range(10)])
    out = sweep.summarise_clip([0.0], [0.0], quality, 0.0, 1.0)

    assert out['blur_score'] == pytest.approx(9.5, abs=0.01)


def test_the_duplicate_share_is_a_share_and_not_a_count():
    """Same shape as freeze_ratio and for the same reason: fifty duplicates in a
    two-minute shot and fifty in a two-second one are not the same defect."""
    out = sweep.summarise_clip([0.0, 0.1, 0.2, 0.3], [0.0, 0.2], [], 0.0, 1.0)

    assert out['dup_frame_ratio'] == 0.5


def test_the_percentile_of_nothing_is_none_and_never_zero():
    assert sweep.percentile([], 0.9) is None
    assert sweep.percentile([4.0], 0.1) == 4.0


def test_this_module_and_video_metrics_agree_on_what_a_percentile_is():
    """Two copies of the arithmetic, deliberately (see the docstring on
    `percentile`), so the thing that must not drift is pinned here."""
    values = [1.0, 3.0, 7.5, 9.0, 12.25, 40.0]
    for p in (0.1, 0.5, 0.9):
        assert sweep.percentile(values, p) == pytest.approx(
            video_metrics.percentile(values, p))


# --- the chain and the sampling ------------------------------------------------------

def test_the_sampler_asks_the_file_for_its_own_rate():
    """A fixed step samples a 60 fps file two and a half times as often as a
    24 fps one — a difference in the measurement, not in the material."""
    assert sweep.framestep_for(25) == 6           # ~4.2 samples a second
    assert sweep.framestep_for(60) == 15          # ~4.0
    assert sweep.framestep_for(12) == 3           # ~4.0
    assert sweep.framestep_for(None) == 6         # falls back to 25, like the scan
    assert sweep.framestep_for(0) == 6
    assert sweep.framestep_for(2) == 1            # never zero: that is every frame


def test_the_chain_tags_frames_before_the_first_print():
    """`metadata=print` emits NOTHING for a frame carrying no keys, so the first
    print in the chain sees nothing unless something put a key on the frame
    first. Measured, not assumed: without this the file that counts frames comes
    back empty and the duplicate share silently becomes unmeasurable."""
    chain = sweep.sweep_chain(6)

    assert chain.index('metadata=mode=add') < chain.index('file=all.txt')


def test_the_chain_counts_every_frame_before_mpdecimate_and_survivors_after():
    chain = sweep.sweep_chain(6)

    assert chain.index('file=all.txt') < chain.index('mpdecimate')
    assert chain.index('mpdecimate') < chain.index('file=kept.txt')


def test_the_expensive_filters_run_on_a_sample_taken_after_mpdecimate():
    """Two things pinned at once, and both were arrived at by measuring:
    `framestep` and not `fps` (which DUPLICATES frames to reach a constant rate,
    handing the detectors back exactly what mpdecimate just removed), and AFTER
    the drop rather than before, so the sample spreads over distinct frames."""
    chain = sweep.sweep_chain(6)

    assert 'framestep=step=6' in chain
    assert 'fps=' not in chain
    assert chain.index('mpdecimate') < chain.index('framestep')
    assert chain.index('framestep') < chain.index('blockdetect')
    assert chain.index('blurdetect') < chain.index('file=quality.txt')


def test_the_metadata_files_are_relative_names():
    """They are passed inside a filter argument, which is split on ':' — a
    Windows absolute path there needs two levels of escaping and is the classic
    way this breaks on one platform only. The pass runs ffmpeg IN a scratch
    directory instead."""
    chain = sweep.sweep_chain(6)

    assert ':\\' not in chain and ':/' not in chain
    assert 'file=all.txt' in chain


# --- the verdicts ---------------------------------------------------------------------

def test_each_defect_feeds_its_own_flag():
    scores = {'dup_frame_ratio': 0.4, 'block_score': 30.0, 'blur_score': 9.0}
    flags = video_metrics.verdicts(
        scores, {'dup_frames_max': 0.1, 'block_max': 20.0, 'blur_max': 7.0})

    assert flags == {'dup_frames', 'blocky', 'blurry'}


def test_moving_a_cut_re_flags_the_same_stored_scores_with_no_rescan():
    """The whole reason raw scores are stored and verdicts are not."""
    scores = {'dup_frame_ratio': 0.2, 'block_score': 14.0, 'blur_score': 6.0}

    assert video_metrics.verdicts(scores, {'block_max': 20.0}) == set()
    assert video_metrics.verdicts(scores, {'block_max': 10.0}) == {'blocky'}
    assert video_metrics.verdicts(scores, {'dup_frames_max': 0.5}) == set()
    assert video_metrics.verdicts(scores, {'dup_frames_max': 0.1}) == {'dup_frames'}


def test_a_shot_the_sweep_never_touched_is_never_flagged():
    """No measurement, no verdict — the rule every cut in this panel obeys. A
    bank swept before this pass existed carries none of these keys, and a cut set
    on it must not retro-flag the half nobody looked at."""
    assert video_metrics.verdicts(
        {'motion_mean': 0.5},
        {'dup_frames_max': 0.0, 'block_max': 0.0, 'blur_max': 0.0}) == set()


def test_a_cut_left_empty_flags_nothing_however_bad_the_score():
    assert video_metrics.verdicts(
        {'dup_frame_ratio': 1.0, 'block_score': 900.0, 'blur_score': 90.0},
        {}) == set()


def test_duplicated_frames_and_a_frozen_stretch_are_different_findings():
    """They must not collapse: `freeze` reads motion vectors and says nothing
    MOVED, `dup_frames` says the same picture ARRIVED twice. A 24-into-30
    pulldown of a moving shot produces the second with no trace of the first."""
    pulldown = {'dup_frame_ratio': 0.2, 'freeze_ratio': 0.0, 'motion_mean': 0.4}
    flags = video_metrics.verdicts(pulldown, {'dup_frames_max': 0.1,
                                              'freeze_max': 0.1})

    assert flags == {'dup_frames'}


def test_blurry_and_soft_are_different_findings_on_the_same_clip():
    """The measured case that earns `blurry` its own flag: footage upscaled from
    480p reads IDENTICALLY to native 1080p on the 160-pixel analysis copy
    `sharpness_p90` uses (354.35 against 353.69, measured), and obviously soft at
    full size. A clip can honestly carry either flag without the other."""
    upscaled = {'sharpness_p90': 353.69, 'blur_score': 7.35}
    cuts = {'sharpness_floor': 100.0, 'blur_max': 7.0}

    assert video_metrics.verdicts(upscaled, cuts) == {'blurry'}

    plain_fog = {'sharpness_p90': 40.0, 'blur_score': 4.5}
    assert video_metrics.verdicts(plain_fog, cuts) == {'soft'}


# --- the contracts that stop the three lists drifting apart --------------------------

def test_every_cut_this_pass_feeds_is_in_the_canonical_list():
    for key in ('dup_frames_max', 'block_max', 'blur_max'):
        assert key in video_metrics.THRESHOLD_KEYS


def test_a_re_measure_carries_this_pass_verdicts_across():
    """The metrics scan rewrites metrics_json wholesale. A key this pass owns
    and ADVISORY_KEYS does not carry is erased by the next quality scan with
    nothing anywhere to see — and re-earning it costs a decode of every FILE in
    the bank, the most expensive silence of the six."""
    for key in sweep.OWNED_KEYS:
        assert key in video_metrics.ADVISORY_KEYS, f'{key} is dropped by a re-measure'

    previous = {'defect_state': 'ok', 'block_score': 12.0, 'blur_score': 4.4,
                'dup_frame_ratio': 0.0, 'defect_frames': 9}
    merged = video_metrics.merge_advisory(previous, {'metrics_state': 'ok'})

    assert merged['block_score'] == 12.0
    assert merged['defect_state'] == 'ok'


def test_the_state_key_is_the_one_this_pass_owns():
    assert sweep.STATE_KEY in sweep.OWNED_KEYS


# --- bits per pixel ------------------------------------------------------------------

def test_bits_per_pixel_is_comparable_across_resolutions():
    """The reason it is shown at all: 5 Mb/s is generous at 480p and starving at
    4K, so the bitrate alone cannot be read across a mixed bank."""
    small = video_probe.bits_per_pixel(1_236_684, 640, 360, 25.0)
    large = video_probe.bits_per_pixel(10_998_064, 1920, 1080, 25.0)

    assert small == pytest.approx(0.2147, abs=1e-3)
    assert large == pytest.approx(0.2121, abs=1e-3)


def test_a_container_that_carries_no_bitrate_reports_nothing_rather_than_zero():
    """MKV and WebM routinely carry no per-stream bitrate — measured, on files
    holding the same footage as an .mp4 that reports one fine. A 0.0 would read
    as a file with nothing in it."""
    assert video_probe.bits_per_pixel(None, 1920, 1080, 25.0) is None
    assert video_probe.bits_per_pixel(0, 1920, 1080, 25.0) is None
    assert video_probe.bits_per_pixel(5_000_000, 1920, 1080, None) is None
    assert video_probe.bits_per_pixel(5_000_000, None, None, 25.0) is None


def test_a_probe_that_cannot_read_the_bitrate_still_reports_the_geometry():
    """The regression this guards: reading `stream.bit_rate` as a plain attribute
    raised on a PyAV build that does not expose it, and the exception reported
    the whole FILE unreadable — a bank losing every geometry it had over a
    display detail."""
    class _Codec:
        name = 'h264'
        # No `profile` and no `bit_rate` anywhere: this fake IS the old PyAV.

    class _Stream:
        # `type` shadows the builtin inside a class body, so nothing here may
        # call it — the codec context is a class of its own above.
        type = 'video'
        average_rate = 30
        duration = None
        time_base = None
        width, height = 1280, 720
        codec_context = _Codec()

    class _Container:
        streams = type('S', (), {'video': [_Stream()]})()
        duration = 5_000_000

        def close(self):
            pass

    probe = video_probe.probe.__globals__
    original = probe['_open']
    probe['_open'] = lambda path: _Container()
    try:
        result = video_probe.probe('/src/a.mkv')
    finally:
        probe['_open'] = original

    assert result['probe_state'] == 'ok'
    assert (result['width'], result['height']) == (1280, 720)
    assert result['bit_rate'] is None and result['profile'] is None


# --- storing, and not clobbering the neighbours --------------------------------------

class _Clip:
    def __init__(self, blob=None):
        self.metrics_json = json.dumps(blob) if blob else None


def test_storing_merges_into_the_blob_rather_than_replacing_it(monkeypatch):
    """metrics_json is shared with five other passes. Replacing it here would
    erase a watermark verdict, a look score and a safe zone with nothing to see —
    the flags would simply stop appearing."""
    monkeypatch.setattr(sweep.db.session, 'commit', lambda: None)
    clip = _Clip({'metrics_state': 'ok', 'sharpness_p90': 400.0,
                  'watermark_score': 0.99})

    sweep._store(clip, {'defect_state': 'ok', 'block_score': 12.0})
    stored = json.loads(clip.metrics_json)

    assert stored['sharpness_p90'] == 400.0
    assert stored['watermark_score'] == 0.99
    assert stored['block_score'] == 12.0


def test_a_re_run_that_measures_less_does_not_leave_last_run_numbers_behind(monkeypatch):
    monkeypatch.setattr(sweep.db.session, 'commit', lambda: None)
    clip = _Clip({'defect_state': 'ok', 'block_score': 12.0, 'blur_score': 4.0,
                  'dup_frame_ratio': 0.1, 'defect_frames': 9})

    sweep._store(clip, {'defect_state': 'unreadable'})
    stored = json.loads(clip.metrics_json)

    assert stored == {'defect_state': 'unreadable'}


def test_a_corrupt_blob_is_never_the_reason_a_bank_loses_its_scores(monkeypatch):
    monkeypatch.setattr(sweep.db.session, 'commit', lambda: None)
    clip = _Clip()
    clip.metrics_json = 'not json at all'

    sweep._store(clip, {'defect_state': 'ok'})

    assert json.loads(clip.metrics_json) == {'defect_state': 'ok'}


def test_the_sweep_never_waits_on_an_undrained_pipe(monkeypatch, tmp_path):
    """A poll loop and a PIPE together are a deadlock, and this pass needs the
    poll loop: a source file can be an hour long and a Stop has to land in a
    quarter second, not at the end of it. With nobody draining, ffmpeg fills the
    OS buffer (~64 KB), blocks on its next write and never exits — so a file that
    merely produced a lot of warnings would burn the whole budget and then be
    reported as broken. Pinned because `capture_output=True` is exactly what a
    tidying refactor would reach for.
    """
    seen = {}

    class _Proc:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(args, **kwargs):
        seen.update(kwargs)
        return _Proc()

    monkeypatch.setattr(sweep.subprocess, 'Popen', fake_popen)
    sweep._run_polled(['ffmpeg'], str(tmp_path), 10.0, None)

    assert seen.get('stdout') is sweep.subprocess.DEVNULL
    assert seen.get('stdin') is sweep.subprocess.DEVNULL
    # stderr is a real file object, never a pipe: a file has no buffer ceiling.
    assert seen.get('stderr') is not sweep.subprocess.PIPE
    assert hasattr(seen.get('stderr'), 'write')


def test_the_stderr_tail_survives_more_than_a_pipe_would_hold(tmp_path):
    """The other half of the same decision: the reason for a file is that the
    output can be big, so the tail has to come back from a big one."""
    noisy = 'x' * 200_000 + '\nthe last words that matter'
    err_path = tmp_path / sweep._STDERR_FILE
    err_path.write_text(noisy, encoding='utf-8')

    with open(err_path, encoding='utf-8') as fh:
        tail = fh.read().strip()[-300:]

    assert 'the last words that matter' in tail
    assert len(tail) == 300


# --- the one test that runs the real binary ------------------------------------------

def _ffmpeg_or_skip():
    from app.services import ffmpeg_tools
    if not ffmpeg_tools.has_ffmpeg():
        pytest.skip('no usable ffmpeg on this machine')
    return ffmpeg_tools.ffmpeg_path()


@pytest.mark.live_ffmpeg
def test_the_sweep_really_separates_a_clean_file_from_a_ruined_one(tmp_path):
    """The integration, once, gated on the binary — everything above proves the
    arithmetic without it.

    Two files are forged with ffmpeg itself: the same three seconds of synthetic
    footage, one encoded at CRF 16 and one at CRF 51 with a frozen second spliced
    into the middle. The assertion is a SEPARATION rather than a value, because
    the values move with the ffmpeg build and with the content — which is exactly
    why none of these cuts ships with a default.
    """
    import subprocess
    binary = _ffmpeg_or_skip()

    def run(args):
        subprocess.run([binary, '-hide_banner', '-loglevel', 'error', '-y'] + args,
                       check=True, capture_output=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    moving = str(tmp_path / 'moving.mp4')
    still = str(tmp_path / 'still.mp4')
    clean = str(tmp_path / 'clean.mp4')
    ruined = str(tmp_path / 'ruined.mp4')
    listing = tmp_path / 'list.txt'

    run(['-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=25:duration=2',
         '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', moving])
    # A still image held for a second: 25 frames mpdecimate must collapse to one.
    run(['-f', 'lavfi', '-i', 'color=c=gray:size=320x180:rate=25:duration=1',
         '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', still])
    listing.write_text(f"file '{moving}'\nfile '{still}'\n", encoding='utf-8')
    run(['-f', 'concat', '-safe', '0', '-i', str(listing),
         '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', clean])
    run(['-f', 'concat', '-safe', '0', '-i', str(listing),
         '-c:v', 'libx264', '-crf', '51', '-pix_fmt', 'yuv420p', ruined])

    def swept(path, start_s, end_s):
        all_times, kept_times, quality = sweep.sweep_file(path, 25.0)
        return sweep.summarise_clip(all_times, kept_times, quality, start_s, end_s)

    # The MOVING first two seconds, where the only difference is the encode.
    good = swept(clean, 0.0, 2.0)
    bad = swept(ruined, 0.0, 2.0)

    assert good['defect_state'] == 'ok' and bad['defect_state'] == 'ok'
    assert good['defect_frames'] > 0
    assert bad['block_score'] > good['block_score'] * 1.5, (
        f'blocking did not separate: clean {good["block_score"]}, '
        f'ruined {bad["block_score"]}')

    # The FROZEN second, where the difference is the duplication.
    moving_window = swept(clean, 0.0, 1.9)
    frozen_window = swept(clean, 2.1, 2.9)

    assert moving_window['dup_frame_ratio'] < 0.1
    assert frozen_window['dup_frame_ratio'] > 0.8, (
        f'the frozen second reported {frozen_window["dup_frame_ratio"]} duplicates')


@pytest.mark.live_ffmpeg
def test_a_file_ffmpeg_cannot_read_raises_something_readable(tmp_path):
    _ffmpeg_or_skip()
    broken = tmp_path / 'broken.mp4'
    broken.write_bytes(b'this is not a video')

    with pytest.raises(RuntimeError) as caught:
        sweep.sweep_file(str(broken), 25.0)

    assert 'ffmpeg' in str(caught.value).lower()


# --- the pass over a bank ------------------------------------------------------------
#
# The ffmpeg SEAM is stubbed here, deliberately: what is under test is the
# plumbing around it — which files get swept, which clips get numbers, what a
# failure costs, what a Stop keeps — and none of that becomes truer for having
# spent eight seconds decoding a real file.

def _bank(app, *, clips_per_source=(2,), probe_state='ok'):
    """A bank with one source per entry of `clips_per_source`. Returns
    (bank_id, [[clip ids of source 0], ...])."""
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        per_source = []
        for index, count in enumerate(clips_per_source):
            src = VideoSource(bank_id=bank.id, relpath=f'{index}.mp4',
                              duration_s=60.0, fps_native=25.0,
                              probe_state=probe_state)
            db.session.add(src)
            db.session.flush()
            ids = []
            for i in range(count):
                clip = VideoClip(bank_id=bank.id, source_id=src.id,
                                 start_s=float(i * 10), end_s=float(i * 10 + 5))
                db.session.add(clip)
                db.session.flush()
                ids.append(clip.id)
            per_source.append(ids)
        db.session.commit()
        return bank.id, per_source


def _summaries(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return {c.id: (json.loads(c.metrics_json) if c.metrics_json else {})
                for c in VideoClip.query.filter_by(bank_id=bank_id).all()}


def _fake_sweep(monkeypatch, calls=None, *, fail_on=None):
    """The ffmpeg seam, stubbed: 25 frames a second across TWO minutes, every
    fifth of them dropped by the (imaginary) mpdecimate, one quality sample a
    second.

    Longer than any clip the helper above lays down, on purpose: a source file
    outlasts its shots, and a fixture that stopped at the last clip would make
    every off-the-end assertion pass for the wrong reason."""
    def fake(path, fps, *, should_stop=None, duration_s=None):
        if calls is not None:
            calls.append(path)
        if fail_on and fail_on in str(path):
            raise RuntimeError('ffmpeg read no frames (exit 1)')
        all_times = [i / 25.0 for i in range(25 * 120)]
        kept = [t for i, t in enumerate(all_times) if i % 5]
        quality = [(float(s), {'lavfi.block': 12.0, 'lavfi.blur': 4.5})
                   for s in range(120)]
        return all_times, kept, quality
    monkeypatch.setattr(sweep, 'sweep_file', fake)


def test_the_pass_writes_numbers_onto_every_clip_of_every_file(app, monkeypatch):
    bank_id, per_source = _bank(app, clips_per_source=(2, 1))
    _fake_sweep(monkeypatch)

    with app.app_context():
        out = sweep.run_defects(bank_id)

    assert out['measured'] == 3
    assert out['files'] == 2
    assert out['error'] is None
    stored = _summaries(app, bank_id)[per_source[0][0]]
    assert stored['defect_state'] == 'ok'
    assert stored['dup_frame_ratio'] == pytest.approx(0.2, abs=0.01)
    assert stored['block_score'] == pytest.approx(12.0)
    assert stored['blur_score'] == pytest.approx(4.5)


def test_one_decode_serves_every_clip_of_a_file(app, monkeypatch):
    """The whole cost argument, pinned. Sweeping per clip would be the
    obvious-looking refactor and it would multiply the only expensive part of
    this pass by the number of shots in a rush."""
    bank_id, _ids = _bank(app, clips_per_source=(7,))
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        out = sweep.run_defects(bank_id)

    assert out['measured'] == 7
    assert len(calls) == 1, f'the file was decoded {len(calls)} times for 7 shots'


def test_a_file_with_no_clips_is_not_decoded_at_all(app, monkeypatch):
    """Minutes of decoding with nowhere to put the answer: this pass writes onto
    clips, and a file nobody has cut yet has none."""
    bank_id, _ids = _bank(app, clips_per_source=(2, 0))
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        sweep.run_defects(bank_id)

    assert len(calls) == 1


def test_an_unreadable_file_is_never_swept(app, monkeypatch):
    bank_id, _ids = _bank(app, clips_per_source=(2,), probe_state='unreadable')
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        out = sweep.run_defects(bank_id)

    assert calls == []
    assert out['measured'] == 0


def test_a_second_run_skips_what_the_first_one_did(app, monkeypatch):
    """The resume contract. A re-run of a swept bank must cost nothing."""
    bank_id, _ids = _bank(app, clips_per_source=(2,))
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        sweep.run_defects(bank_id)
        second = sweep.run_defects(bank_id)

    assert len(calls) == 1
    assert second['measured'] == 0


def test_rescan_sweeps_a_bank_that_is_already_done(app, monkeypatch):
    bank_id, _ids = _bank(app, clips_per_source=(2,))
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        sweep.run_defects(bank_id)
        again = sweep.run_defects(bank_id, rescan=True)

    assert len(calls) == 2
    assert again['measured'] == 2


def test_one_new_shot_puts_its_whole_file_back_in_the_queue(app, monkeypatch):
    """Re-cutting one shot of a forty-shot rush leaves that shot unswept, and the
    only way to measure it is to decode the file it came from. Sweeping the whole
    file is what the one-decode argument buys — and the alternative, leaving the
    new shot permanently unswept, is a silent hole."""
    from app.extensions import db
    from app.models import VideoClip, VideoSource
    bank_id, per_source = _bank(app, clips_per_source=(2,))
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        sweep.run_defects(bank_id)
        src = VideoSource.query.filter_by(bank_id=bank_id).first()
        fresh = VideoClip(bank_id=bank_id, source_id=src.id,
                          start_s=30.0, end_s=35.0)
        db.session.add(fresh)
        db.session.commit()
        fresh_id = fresh.id
        out = sweep.run_defects(bank_id)

    assert len(calls) == 2
    assert out['measured'] == 3
    assert _summaries(app, bank_id)[fresh_id]['defect_state'] == 'ok'


def test_a_broken_file_costs_that_file_and_leaves_its_shots_in_the_queue(app, monkeypatch):
    """A bank is swept in bulk. One file ffmpeg cannot read must not throw away
    the other nineteen — and its shots must keep NO state, because that absence
    is exactly what queues them for the next run."""
    bank_id, per_source = _bank(app, clips_per_source=(1, 1))
    _fake_sweep(monkeypatch, fail_on='0.mp4')

    with app.app_context():
        out = sweep.run_defects(bank_id)

    assert out['measured'] == 1
    assert out['files'] == 1
    assert 'ffmpeg' in out['error']
    stored = _summaries(app, bank_id)
    assert stored[per_source[0][0]] == {}, 'a broken file must not retire its shots'
    assert stored[per_source[1][0]]['defect_state'] == 'ok'


def test_the_pass_keeps_every_measurement_the_other_passes_made(app, monkeypatch):
    """metrics_json is shared with five other passes; this one must add to it."""
    from app.extensions import db
    from app.models import VideoClip
    bank_id, per_source = _bank(app, clips_per_source=(1,))
    _fake_sweep(monkeypatch)

    with app.app_context():
        clip = db.session.get(VideoClip, per_source[0][0])
        clip.metrics_json = json.dumps({'metrics_state': 'ok',
                                        'sharpness_p90': 402.0,
                                        'safe_zone_state': 'ok'})
        db.session.commit()
        sweep.run_defects(bank_id)

    stored = _summaries(app, bank_id)[per_source[0][0]]
    assert stored['sharpness_p90'] == 402.0
    assert stored['safe_zone_state'] == 'ok'
    assert stored['defect_state'] == 'ok'


def test_a_stop_between_files_keeps_everything_already_swept(app, monkeypatch):
    bank_id, per_source = _bank(app, clips_per_source=(1, 1))
    calls = []
    _fake_sweep(monkeypatch, calls)

    with app.app_context():
        out = sweep.run_defects(bank_id, should_stop=lambda: len(calls) >= 1)

    assert len(calls) == 1
    assert out['measured'] == 1
    assert _summaries(app, bank_id)[per_source[1][0]] == {}


def test_the_cuts_ship_with_no_number():
    from app.config import DEFAULTS
    for key in ('dup_frames_max', 'block_max', 'blur_max'):
        assert DEFAULTS['video_bank'][key] is None


def test_the_threshold_reader_hands_the_new_cuts_through(app):
    from app.services import video_bank_service as svc
    with app.app_context():
        reader = svc.metric_thresholds()
    for key in ('dup_frames_max', 'block_max', 'blur_max'):
        assert key in reader


def test_the_route_refuses_with_the_setup_sentence_when_ffmpeg_is_missing(
        app, client, monkeypatch):
    """A 503 naming FFMPEG, not the decode extra — this pass never opens the file
    itself, so an install with `av` and no binary must be told which one is
    missing. A 202 followed by a job that dies is the same news, later."""
    bank_id, _ids = _bank(app, clips_per_source=(1,))
    monkeypatch.setattr(sweep.ffmpeg_tools, 'ffmpeg_ready',
                        lambda force=False: {'ok': False, 'path': None,
                                             'reason': 'no ffmpeg binary found'})

    response = client.post(f'/api/video-bank/{bank_id}/defects', json={})

    assert response.status_code == 503
    assert 'ffmpeg' in response.get_json()['error'].lower()


def test_the_route_starts_the_pass_when_ffmpeg_is_there(app, client, monkeypatch):
    bank_id, _ids = _bank(app, clips_per_source=(1,))
    monkeypatch.setattr(sweep.ffmpeg_tools, 'ffmpeg_ready',
                        lambda force=False: {'ok': True, 'path': '/bin/ffmpeg',
                                             'reason': 'ffmpeg runs'})
    _fake_sweep(monkeypatch)

    response = client.post(f'/api/video-bank/{bank_id}/defects', json={})

    assert response.status_code == 202


def test_an_unknown_bank_is_a_404_not_a_503(app, client, monkeypatch):
    monkeypatch.setattr(sweep.ffmpeg_tools, 'ffmpeg_ready',
                        lambda force=False: {'ok': True, 'path': '/bin/ffmpeg',
                                             'reason': 'ffmpeg runs'})
    assert client.post('/api/video-bank/999999/defects', json={}).status_code == 404


def test_the_two_lanes_treat_a_container_fact_the_same_way():
    """CLAUDE.md's rule: a shared question ships on both surfaces or names why it
    differs. `bits_per_pixel` here and `jpeg_quality` on a still are the SAME kind
    of fact — how hard the file was squeezed, read from the header — and both are
    displayed, never cut on. A flag on one and not the other would teach a user a
    behaviour on one surface that the other contradicts."""
    from app.config import DEFAULTS

    assert 'bits_per_pixel' not in DEFAULTS['video_bank']
    assert 'bpp_min' not in DEFAULTS['video_bank']
    assert not any('bpp' in key or 'bit_rate' in key
                   for key in video_metrics.THRESHOLD_KEYS)
    # The image lane's own container fact carries no cut either.
    assert not any('jpeg' in key for key in (DEFAULTS.get('bank') or {}))
