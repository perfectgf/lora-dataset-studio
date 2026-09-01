"""🎬 Turning a pair of timestamps into one training clip on disk.

Every test here is about a way the export could go wrong QUIETLY. A clip two
frames short, encoded at the source's frame rate, stream-copied from the nearest
keyframe, or missing its caption file does not raise anything — it trains, and it
trains badly. Several of these are documented behaviours of real trainers:

  * musubi-tuner CRASHES on a missing sidecar (FileNotFoundError, no try/except
    on the path); diffusion-pipe DROPS the clip instead, because its
    skip_empty_caption defaults to true.
  * diffusion-pipe and finetrainers read captions with a bare open(), i.e. the
    host's locale encoding. A BOM or a cp1252 byte is mojibake or a crash.
  * ai-toolkit never drops a too-short clip and never warns: it clamps and
    repeats frames, so a 5-frame clip trains as repeated stills.

So the guarantees are pinned as argument-level assertions rather than trusted to
a review of the ffmpeg line. No ffmpeg binary is invoked: this is the builder.
"""
import pytest

from app.services import video_clip_export as ex


def _cmd(**over):
    args = dict(ffmpeg='/usr/bin/ffmpeg', src='/src/a.mp4', dst='/out/clip_0001.mp4',
                start_s=41.2, end_s=50.0, frames=81, fps=16)
    args.update(over)
    return ex.clip_command(**args)


# --- the frame count is the contract ------------------------------------------

def test_the_exact_frame_count_is_requested_of_ffmpeg():
    """The target's VAE accepts 81 frames, not 'about five seconds'. Asking for a
    duration and hoping the count lands right is how a clip ends up at 80 or 82 —
    and Wan's own code never validates it, so nothing downstream will complain."""
    args = _cmd()
    assert args[args.index('-frames:v') + 1] == '81'


def test_the_source_needed_is_intervals_not_frames():
    """81 frames span 80 intervals: 5.00 s at 16 fps, not 5.0625. Requiring the
    larger figure rejects segments that fit perfectly."""
    assert ex.clip_duration_s(81, 16) == pytest.approx(5.0)


def test_a_segment_of_exactly_the_needed_length_is_accepted():
    """The boundary must not be lost to floating point — a detector handing back
    precisely 5.00 s for an 81-frame clip must not have it refused."""
    assert _cmd(start_s=0.0, end_s=5.0)


def test_a_segment_too_short_for_the_requested_length_is_refused():
    """Handed 2 s where 5 are needed, ffmpeg writes a 32-frame clip and exits 0.
    That is the most expensive silent failure in this lane, because the dataset
    looks complete — and ai-toolkit will not flag it either."""
    with pytest.raises(ex.ClipTooShort):
        _cmd(start_s=0.0, end_s=2.0)


# --- re-encoding is not optional ----------------------------------------------

def test_the_command_never_stream_copies():
    """A stream copy can only start on a keyframe. Scraped material routinely has
    a 250-frame GOP (~10 s), so on a 3-second clip the boundary error is the size
    of the clip — it would throw away the frame-exact cut the detector produced."""
    assert 'copy' not in _cmd()


def test_the_output_is_normalised_to_the_targets_frame_rate():
    """Not the source's. A 30 fps rush exported for a 16 fps target must be
    resampled, or the motion plays back accelerated at inference."""
    args = _cmd()
    assert 'fps=16' in args[args.index('-vf') + 1]


def test_the_keyframe_interval_follows_the_frame_rate():
    args = _cmd()
    assert args[args.index('-g') + 1] == '16'


# --- audio is a per-target decision -------------------------------------------

def test_audio_is_dropped_for_a_target_that_does_not_read_it():
    """Wan trainers ignore the audio track, and carrying it inflates every clip
    and the trainer's latent cache for nothing."""
    assert '-an' in _cmd()


def test_audio_is_kept_for_a_joint_audio_video_target():
    """LTX-2.3 and MiniMax H3 train sound and picture together. A blanket -an
    teaches those models to be silent — a degradation with no error message
    anywhere, which is exactly why it cannot be a global flag."""
    args = _cmd(audio={'muxed': True, 'sample_rate': None, 'channels': None})
    assert '-an' not in args
    assert 'aac' in args


def test_a_target_that_pins_its_audio_format_gets_it_imposed():
    """MiniMax H3 trains on 32 kHz stereo. "Keep the audio" is not enough: a
    44.1 kHz mono source would ride through untouched, and whether the trainer
    resamples it is not something the dataset should be gambling on."""
    args = _cmd(audio={'muxed': True, 'sample_rate': 32000, 'channels': 2})
    assert args[args.index('-ar') + 1] == '32000'
    assert args[args.index('-ac') + 1] == '2'


def test_a_target_with_no_pinned_format_does_not_resample():
    """None means "keep the source's". Forcing a rate the model never asked for
    is a lossy conversion bought for nothing."""
    args = _cmd(audio={'muxed': True, 'sample_rate': None, 'channels': None})
    assert '-ar' not in args
    assert '-ac' not in args


def test_the_profile_decides_the_audio_policy():
    """The caller should not have to remember which targets are joint models, nor
    at which sample rate each one wants its sound."""
    common = dict(ffmpeg='/f', src='/a.mp4', dst='/o.mp4', start_s=0.0, end_s=9.0,
                  frames=49)
    assert '-an' in ex.command_for_profile(profile_key='wan22_14b', **common)
    assert '-an' not in ex.command_for_profile(profile_key='ltx23', **common)

    h3 = ex.command_for_profile(profile_key='minimax_h3', start_s=0.0, end_s=9.0,
                                frames=39, ffmpeg='/f', src='/a.mp4', dst='/o.mp4')
    assert h3[h3.index('-ar') + 1] == '32000'


def test_an_unknown_profile_is_refused_rather_than_defaulted():
    """Silently falling back to Wan's geometry for a key we do not know is how a
    dataset ends up cut for the wrong model."""
    with pytest.raises(ValueError):
        ex.command_for_profile(profile_key='nope', ffmpeg='/f', src='/a.mp4',
                               dst='/o.mp4', start_s=0.0, end_s=9.0, frames=49)


# --- seeking and scaling ------------------------------------------------------

def test_seeking_happens_before_the_input():
    """`-ss` after `-i` makes ffmpeg decode the file from zero and throw the result
    away. On a two-hour rush cut into hundreds of clips that is the difference
    between minutes and hours."""
    args = _cmd()
    assert args.index('-ss') < args.index('-i')


def test_no_scale_filter_when_no_size_is_asked_for():
    """A target with a free size keeps the source's, and an identity scale is a
    resample that costs quality for nothing."""
    args = _cmd()
    assert 'scale=' not in args[args.index('-vf') + 1]


def test_a_requested_size_is_scaled_with_a_quality_filter():
    args = _cmd(size=(848, 480))
    filters = args[args.index('-vf') + 1]
    assert 'scale=848:480' in filters
    assert 'lanczos' in filters


def test_resampling_happens_before_scaling():
    """Scaling frames that are about to be dropped is work thrown away — on a
    30 to 16 fps conversion, nearly half of them."""
    filters = _cmd(size=(848, 480))[_cmd(size=(848, 480)).index('-vf') + 1]
    assert filters.index('fps=') < filters.index('scale=')


# --- what lands on disk -------------------------------------------------------

def test_clip_filenames_are_zero_padded_so_they_sort():
    """Trainers walk the folder in filename order; clip_10 sorting before clip_2
    is a reordered dataset."""
    assert ex.clip_filename(1) == 'clip_0001.mp4'
    assert ex.clip_filename(342) == 'clip_0342.mp4'


def test_the_clip_extension_is_lowercase():
    """musubi-tuner matches extensions by exact string against an explicit list —
    a clip named .MP4 is simply never found, with no message."""
    assert ex.clip_filename(1).endswith('.mp4')


def test_the_caption_rides_a_homonym_txt_sidecar():
    assert ex.sidecar_path('/out/clip_0007.mp4') == '/out/clip_0007.txt'


def test_a_sidecar_is_written_even_when_there_is_no_caption(tmp_path):
    """An empty caption is a choice; a MISSING file is a crash. musubi raises
    FileNotFoundError out of a worker with no handler, and diffusion-pipe drops
    the clip silently. Wave 1 has no captioning, so every clip it exports would
    hit that path."""
    clip = tmp_path / 'clip_0001.mp4'
    ex.write_sidecar(str(clip), None)

    assert (tmp_path / 'clip_0001.txt').read_text(encoding='utf-8') == ''


def test_the_sidecar_is_utf8_without_a_bom(tmp_path):
    """diffusion-pipe and finetrainers open captions with the host's locale
    encoding. A BOM shows up as a stray character at the head of the first
    caption; on a non-UTF-8 Windows host a bare accent is a UnicodeDecodeError
    that kills the run."""
    clip = tmp_path / 'clip_0002.mp4'
    ex.write_sidecar(str(clip), 'a café at dusk')

    raw = (tmp_path / 'clip_0002.txt').read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf')
    assert raw.decode('utf-8') == 'a café at dusk'


# --- slicing a long shot into consecutive clips (2026-09-01) -------------------------

def test_slice_spans_cuts_a_long_shot_into_whole_consecutive_clips():
    """A fifteen-second shot promoted at 209 frames used to train on its first
    8.7 s. Sliced, it yields whole clips end to end — never overlapping (no two
    clips of a dataset show the same frames) and never partial."""
    from app.services.video_clip_export import clip_duration_s, slice_spans
    need = clip_duration_s(209, 24)                     # 8.666… s
    spans = slice_spans(0.0, 20.0, 209, 24)
    assert len(spans) == 2
    assert spans[0][0] == 0.0
    assert abs((spans[0][1] - spans[0][0]) - need) < 1e-3
    # End to end: the second starts exactly where the first ended.
    assert spans[1][0] == spans[0][1]
    # The tail too short for a whole clip is left behind, as the encoder would.
    assert spans[-1][1] <= 20.0


def test_slice_spans_refuses_what_the_encoder_would_refuse():
    from app.services.video_clip_export import fits_frames, slice_spans
    assert slice_spans(0.0, 2.0, 209, 24) == []
    assert not fits_frames(2.0, 209, 24)                # the same verdict
    assert slice_spans(0.0, 0.0, 39, 24) == []


def test_the_inset_is_applied_once_at_the_ends_not_between_slices():
    """A shot boundary is where a dissolve lives; the joins BETWEEN slices are
    not shot boundaries, so trimming them would throw away good frames."""
    from app.services.video_clip_export import slice_spans
    spans = slice_spans(10.0, 30.0, 39, 24, inset_s=0.5)
    assert spans[0][0] == 10.5                          # trimmed at the head
    assert spans[-1][1] <= 29.5                         # and at the tail
    for a, b in zip(spans, spans[1:]):                  # nothing trimmed inside
        assert a[1] == b[0]


def test_one_shot_cannot_become_the_whole_dataset():
    """`limit` is the same guard as the per-source cap, one level down: a very
    long take must not outweigh every other shot on its own."""
    from app.services.video_clip_export import slice_spans
    assert len(slice_spans(0.0, 600.0, 39, 24, limit=8)) == 8


def test_the_promote_job_is_actually_told_to_slice():
    """The wiring, pinned because it BROKE: slice_long reached the preflight
    (which counted 152 extra clips) and never the job (which encoded 39), since
    the job is called with positional arguments and the option was appended as
    a keyword to the wrong call. A trial run caught in ten minutes what four
    green test files could not — the option now travels, and this reads the
    call itself."""
    import ast
    import inspect
    from app.services import video_bank_service as svc
    src = inspect.getsource(svc.start_promote)
    tree = ast.parse(src.lstrip())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == '_promote_job']
    assert calls, 'start_promote must build the promote job'
    assert any(kw.arg == 'slice_long' for kw in calls[0].keywords), \
        '_promote_job is called without slice_long — the option would be dead'
