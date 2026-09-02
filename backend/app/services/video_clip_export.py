"""Turning a pair of timestamps into one training clip on disk.

This module is where every quiet way an export can be wrong is made loud. The
failures it guards against share a shape: ffmpeg exits 0, a file appears, the
dataset looks complete, and the training is subtly poisoned. Several of them are
documented behaviours of the trainers this output is written for:

  * musubi-tuner CRASHES if a clip's caption sidecar is missing — FileNotFoundError
    out of a worker future, with no handler on the path.
  * diffusion-pipe DROPS such a clip instead: skip_empty_caption defaults to true,
    whatever its README says.
  * diffusion-pipe and finetrainers read captions with a bare open(), i.e. the
    host's locale encoding.
  * ai-toolkit never drops a too-short clip and never warns: it clamps frame
    indices and re-expands, so a five-frame clip trains as repeated stills.

WHY WE ALWAYS RE-ENCODE. `-c copy` is not an optimisation we skipped, it is
disqualified. A stream copy can only start on a keyframe, and scraped material
routinely carries a 250-frame GOP (~10 s). Cutting 3-second clips that way makes
the boundary error the size of the clip, throwing away the frame-exact cut the
detector just produced. We would have to re-encode anyway to normalise the frame
rate and the resolution.

WHY WE ASK FOR FRAMES, NOT SECONDS. The target's VAE accepts a frame COUNT — 81,
or 121, or something congruent to 5 modulo 17. `-t 5.0` asks for a duration and
leaves the count to rounding. `-frames:v 81` asks for the thing that matters.
And nothing downstream will correct a mistake here: Wan's "should be 4n+1" is a
CLI help string, not an assert.

x264 on the CPU rather than NVENC, deliberately: a 3-second clip at 848x480
encodes in well under a second, so there is nothing to gain, and the GPU has to
stay free for captioning and training.
"""
import os

from . import video_targets

# Quality knobs. CRF 16 is visually lossless for training material; `slow` costs
# a fraction of a second per clip at these sizes and buys real bitrate efficiency,
# which matters because these files are read thousands of times during training.
_CRF = '16'
_PRESET = 'slow'

# Audio, when the target is a joint audio-video model. 160 kbps stereo AAC is
# transparent enough for a training signal and costs ~60 kB on a 3-second clip.
_AUDIO_CODEC = 'aac'
_AUDIO_BITRATE = '160k'

# Floating-point slack when comparing a segment's length against what we need.
# Bounds arrive as PTS seconds, so an exact 5.0 can present as 4.999999999999999 —
# refusing that clip would be arithmetic, not judgement. A quarter of a frame at
# 240 fps is far below any real tolerance question and far above float noise.
_EPSILON = 1.0 / 960


class ClipTooShort(Exception):
    """The segment cannot supply the requested number of frames.

    Raised rather than silently shortening the clip: handed 2 seconds when it
    needs 5, ffmpeg writes a 32-frame file and exits 0. Nothing downstream can
    tell the difference between a short clip and a correct one — ai-toolkit will
    train on it as repeated stills without a word — so this is the only place the
    mistake can be caught.
    """


def clip_duration_s(frames, fps):
    """Seconds of source a clip of `frames` frames needs, at the TARGET's fps.

    (frames - 1) / fps: N frames span N-1 intervals. Using frames/fps asks for one
    frame period more than the cut needs and refuses segments that fit exactly.
    """
    return (frames - 1) / float(fps)


def slice_spans(start_s, end_s, frames, fps, *, inset_s=0.0, limit=None):
    """A long shot as CONSECUTIVE clip-length spans, in order.

    The export takes `frames` frames from a shot's start and leaves the rest, so
    a fifteen-second shot promoted at 209 frames trains on its first 8.7 s only.
    This cuts the same shot into whole clips end to end — never overlapping, so
    no two clips of the dataset show the same frames, and never partial: a tail
    too short for the profile is dropped exactly as the encoder would refuse it.

    `inset_s` is applied ONCE at each end of the shot, not per slice: it exists
    because a shot boundary is where a dissolve lives, and the joins between
    slices are not shot boundaries. `limit` caps how many one shot may yield —
    a single very long take must not become half the dataset on its own.

    Returns [(start, end), ...]; a shot that cannot fill one clip returns [],
    which is the same verdict `fits_frames` gives.
    """
    lo = float(start_s) + float(inset_s or 0.0)
    hi = float(end_s) - float(inset_s or 0.0)
    need = clip_duration_s(frames, fps)
    if need <= 0 or hi - lo + _EPSILON < need:
        return []
    out = []
    cursor = lo
    while hi - cursor + _EPSILON >= need:
        out.append((round(cursor, 6), round(cursor + need, 6)))
        cursor += need
        if limit and len(out) >= int(limit):
            break
    return out


def fits_frames(span_s, frames, fps):
    """Can a segment of `span_s` seconds supply `frames` frames at `fps`?

    The same test `clip_command` performs before raising ClipTooShort, exposed so
    a caller can ask the question WITHOUT building a command — which is what lets
    the promotion tell "this clip was never long enough" apart from "your edge
    inset made it too short". Those read identically and are fixed differently.

    Deliberately the same expression and the same `_EPSILON`, not a second
    implementation of the rule: a predictor that disagrees with the refusal it
    predicts is worse than no explanation at all, and a test pins them together.
    """
    return (float(span_s) + _EPSILON) >= clip_duration_s(frames, fps)


def clip_filename(index):
    """`clip_0001.mp4`. Zero-padded because trainers walk the folder in filename
    order, and clip_10 sorting before clip_2 is a reordered dataset. Lowercase
    extension because musubi-tuner matches extensions by exact string against an
    explicit list — a `.MP4` is never found, and never mentioned."""
    return f'clip_{index:04d}.mp4'


def sidecar_path(clip_path):
    """The homonym `.txt` that carries the caption."""
    return os.path.splitext(clip_path)[0] + '.txt'


def write_sidecar(clip_path, caption):
    """Write the caption file for a clip. ALWAYS — an absent caption is an empty
    file, never a missing one.

    Wave 1 produces no captions at all, so without this every clip it exports
    would land on the missing-sidecar path: a crash in musubi-tuner, a silent drop
    in diffusion-pipe.

    UTF-8 with no BOM and Unix newlines, because two of the four trainers read
    these with a bare open() at the host's locale encoding. A BOM becomes a stray
    character at the head of the first caption; an accent on a cp1252 Windows host
    becomes a UnicodeDecodeError that kills the run.

    THE CAPTION IS WRITTEN VERBATIM, FOR EVERY TARGET — no prefix, no template,
    no per-architecture dialect, and this function deliberately takes no profile.
    MiniMax H3 is the one that invites the opposite belief, and the belief is
    wrong on both counts: its ``"<Picture i>: "`` labels are built by the encoder
    from the KEYFRAME IMAGES it is handed, never parsed out of the caption
    (ai-toolkit ``.../minimax_h3/src/text_encoder.py:65-74``, whose docstring
    states "no chat template, no special tokens"), and its soundtrack is decoded
    from the video file's own audio track into VAE latent rows, with no text
    channel at all (``toolkit/dataloader_mixins.py:718-761``,
    ``minimax_h3.py:733-760``). ai-toolkit's caption path is architecture-blind
    end to end: read the homonym .txt, pass it through a ``clean_caption`` that
    normalises nothing, encode ``p.strip()``. Pinned by
    test_video_sidecar_caption_format.py.
    """
    with open(sidecar_path(clip_path), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(caption or '')


def clip_command(*, ffmpeg, src, dst, start_s, end_s, frames, fps, size=None,
                 audio=None):
    """The argv that cuts ONE clip. Pure — nothing is executed or written here.

    `size` is (width, height) or None. None means "keep the source's size", which
    is what a target with a free size wants; an identity scale would be a resample
    that costs quality for nothing.
    """
    needed = clip_duration_s(frames, fps)
    if (end_s - start_s) + _EPSILON < needed:
        raise ClipTooShort(
            f'{end_s - start_s:.3f}s available, {needed:.3f}s needed for '
            f'{frames} frames at {fps} fps')

    # Resample first, then scale: scaling frames that are about to be dropped is
    # work thrown away, and on a 30->16 fps conversion that is nearly half of them.
    filters = [f'fps={fps}']
    if size:
        filters.append(f'scale={size[0]}:{size[1]}:flags=lanczos')

    args = [
        ffmpeg, '-y',
        # BEFORE -i. After the input, ffmpeg decodes the file from zero and throws
        # the result away; on a two-hour rush cut into hundreds of clips that is
        # the difference between minutes and hours.
        '-ss', f'{start_s:.6f}',
        '-i', src,
        '-frames:v', str(frames),
        '-vf', ','.join(filters),
        '-c:v', 'libx264', '-crf', _CRF, '-preset', _PRESET,
        '-pix_fmt', 'yuv420p',
        # One keyframe per second of output. Trainers seek into these files
        # repeatedly; a long GOP makes every seek decode a chain of frames.
        '-g', str(fps),
    ]
    if audio:
        # Only for joint audio-video targets, and the track is MUXED INTO the clip
        # rather than written beside it: the loader reads it from the video file,
        # so a sidecar .wav is simply invisible.
        args += ['-c:a', _AUDIO_CODEC, '-b:a', _AUDIO_BITRATE]
        # Pinned only when the target actually asks. "Keep the audio" is not
        # enough for a model trained on 32 kHz stereo — a 44.1 kHz mono source
        # would ride through untouched. And forcing a rate a model never asked
        # for is a lossy conversion bought for nothing, hence None = leave alone.
        if audio.get('sample_rate'):
            args += ['-ar', str(audio['sample_rate'])]
        if audio.get('channels'):
            args += ['-ac', str(audio['channels'])]
        # Trimmed to the video's length so the two tracks agree — the trainers
        # derive audio length from frames/fps and mis-trim a longer track.
        args += ['-shortest']
    else:
        args += ['-an']
    args += ['-movflags', '+faststart', dst]
    return args


def command_for_profile(*, ffmpeg, src, dst, start_s, end_s, profile_key, frames,
                        size=None):
    """`clip_command` with the fps AND the audio policy read from the target
    catalogue rather than passed in.

    The form callers should prefer, because it removes the three places a caller
    could get it wrong on their own: handing in the SOURCE's frame rate, stripping
    the audio of a joint audio-video model, and keeping an audio track at a rate
    the model was not trained on.
    """
    profile = video_targets.get(profile_key)
    if profile is None or not profile['fps']:
        raise ValueError(f'{profile_key} declares no fps to encode at')
    return clip_command(ffmpeg=ffmpeg, src=src, dst=dst, start_s=start_s,
                        end_s=end_s, frames=frames, fps=profile['fps'], size=size,
                        audio=profile['audio'])
