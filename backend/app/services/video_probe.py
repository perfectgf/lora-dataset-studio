"""Reading what a source video IS, and pulling one frame to look at.

In-process through PyAV rather than an ffprobe subprocess. That is not a
preference: `imageio-ffmpeg`, the package that makes the video extra
self-sufficient, bundles ffmpeg WITHOUT ffprobe. An ffprobe-based probe would
work on a developer machine — which has a full install on PATH — and fail on
precisely the machine the extra was written for.

Everything here is per-FILE. Per-clip facts live on VideoClip; the split exists
because a probe that fails must fail once per file rather than once per shot.

`fps_native` is the SOURCE's frame rate and is never what a clip is encoded at.
The target profile decides that. Reading this value at encode time is how a 16 fps
target ends up with accelerated motion, so it is stored to be DISPLAYED and to
convert the detector's frame indices — never to encode from.
"""
import os

# How far into a shot the thumbnail frame is taken, as a fraction of its length.
# The middle, not the start: a shot boundary is where a cut just happened, so the
# opening frames routinely carry the tail of a dissolve or a black frame — the two
# things that make a grid of thumbnails useless.
_THUMB_POSITION = 0.5


def _open(path):
    """The single decode seam. Imported lazily so this module stays importable on
    an install with no video extra — the capability probe is what tells the user,
    not an ImportError at startup."""
    import av
    return av.open(path)


def _duration_seconds(container, stream):
    """Seconds, or None. NEVER zero as a stand-in for "unknown": zero reads as a
    file with no content and makes every later length check pass or fail for the
    wrong reason."""
    if container.duration:
        # PyAV reports the container duration in microseconds.
        return container.duration / 1_000_000
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    return None


def probe(path):
    """What this file is. Always returns a dict; never raises.

    A bank is scanned in bulk, so one corrupt file among four hundred must cost
    that file and not the scan. `probe_state` carries the verdict:
      'ok'         — geometry below is usable
      'unreadable' — could not be opened, or holds no video stream
    """
    result = {'duration_s': None, 'fps_native': None, 'width': None,
              'height': None, 'codec': None, 'probe_state': 'unreadable',
              'file_size': None, 'bit_rate': None, 'profile': None}
    try:
        result['file_size'] = os.path.getsize(path)
    except OSError:
        pass
    try:
        container = _open(path)
    except Exception:                       # noqa: BLE001 — any decode error
        return result
    try:
        streams = container.streams.video
        if not streams:
            # An .mp4 that holds only audio opens perfectly and has nothing to cut.
            return result
        stream = streams[0]
        result.update(
            width=stream.width,
            height=stream.height,
            codec=getattr(stream.codec_context, 'name', None),
            # float() of a Fraction, not round(): 29.97 is 30000/1001, and calling
            # it 30 drifts about a second every thousand.
            fps_native=float(stream.average_rate) if stream.average_rate else None,
            duration_s=_duration_seconds(container, stream),
            probe_state='ok',
            # How hard this file was squeezed, and by which flavour of the
            # codec. Free here — the container header already holds both and the
            # stream is open — and they are what turns "this clip looks blocky"
            # into "of course it does, it is 0.02 bits per pixel". Displayed
            # only: the cut lives on `block_score`, which MEASURES the damage
            # these two merely predict.
            #
            # `getattr`, like `codec` above, and NOT a plain attribute read.
            # These two are the newest things this probe asks PyAV for, and an
            # attribute a given build does not expose would raise inside the try
            # below — which reports the whole FILE as unreadable. A bank would
            # then lose every geometry it had over a display detail. The rule the
            # module already states, applied: an unproven absence must never turn
            # a working install red.
            bit_rate=_int_or_none(getattr(stream, 'bit_rate', None)),
            # PyAV answers 'High', 'Main', 'Profile 0'… Stored as the string it
            # gives rather than mapped onto a vocabulary of ours: it is a label
            # to show, and inventing a second spelling for it would only be a
            # thing to keep in sync with ffmpeg.
            profile=_text_or_none(
                getattr(stream.codec_context, 'profile', None)),
        )
        return result
    except Exception:                       # noqa: BLE001
        result['probe_state'] = 'unreadable'
        return result
    finally:
        try:
            container.close()
        except Exception:                   # noqa: BLE001
            pass


def _int_or_none(value):
    """An int, or None — never 0 as a stand-in for "the container did not say".

    MKV and WebM routinely carry no per-stream bitrate at all (measured: PyAV
    reports None for both, while reporting it fine for the same footage in an
    .mp4). A zero there would read as a file with no data in it and would make
    `bits_per_pixel` answer 0.0 — a number that looks measured and is not.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _text_or_none(value):
    """A non-empty string, or None. PyAV answers None for a codec whose profile
    it cannot name, and some builds answer an empty string instead — both mean
    "nothing to show", and a card printing '' with a separator around it is how
    an empty fact becomes a visible glitch."""
    text = str(value).strip() if value is not None else ''
    return text or None


def bits_per_pixel(bit_rate, width, height, fps):
    """Bits per pixel per frame, or None when any ingredient is missing.

    DERIVED at read time and never stored, which is the same rule this lane
    applies to its flags and for the same reason: it is a pure function of four
    values that are already on the row, and a stored copy is a number that can
    go stale against them the day a re-probe corrects the frame rate.

    What it is FOR: a rough, universal answer to "how much was thrown away",
    comparable across resolutions in a way a bitrate is not — 5 Mbit/s is
    generous at 480p and starving at 4K. Roughly, under 0.05 is visibly damaged
    and over 0.15 is comfortable, for ordinary 8-bit H.264. It is shown and never
    cut on: `block_score` measures the damage this only predicts, and a cut on a
    prediction when the measurement is right there would be the app guessing in
    front of the answer.
    """
    try:
        pixels = float(width or 0) * float(height or 0) * float(fps or 0)
        rate = float(bit_rate or 0)
    except (TypeError, ValueError):
        return None
    if pixels <= 0 or rate <= 0:
        return None
    return rate / pixels


def thumbnail_timestamp(start_s, end_s):
    """The instant a shot's thumbnail is grabbed from — its middle.

    Choosing the shot's FIRST frame is the tempting shortcut and the wrong one: a
    boundary is where a cut just happened, so that frame is disproportionately a
    dissolve or a black frame.
    """
    return start_s + (end_s - start_s) * _THUMB_POSITION
