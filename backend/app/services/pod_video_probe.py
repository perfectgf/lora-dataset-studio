"""Can the pod we just rented actually READ the clips we just sent it?

A video dataset arrives on a pod as a flat folder of `.mp4`. Whether the image
that pod booted can decode them is not a given: OpenCV's bundled ffmpeg has no
software AV1 decoder, PyAV is not in every image, and a pod that finds no
`libGL` fails to import cv2 at all. Every one of those produces the same visible
outcome — a job that starts, runs, and yields a LoRA trained on nothing, or a
traceback several minutes into a paid hour.

This module asks the question BEFORE `start_job`, on the pod's own filesystem,
with the pod's own interpreter. That timing is the whole design: run #138 spent
an hour on an upload phase nobody had verified, and the lesson taken from it was
not "upload more carefully" but "observe the thing you are about to pay for".
The probe costs one command and a few seconds of a pod that is already booted;
the failure it prevents costs the run.

IT PROBES THE DECODER THE TRAINER USES, NOT A DECODER
-----------------------------------------------------
ai-toolkit's video loader opens a clip with `cv2.VideoCapture` and falls back to
PyAV for the frames OpenCV could not decode (`toolkit/dataloader_mixins.py`).
So the program below tries cv2 first and PyAV second, in that order, and reports
which one answered. Asking `ffprobe` instead would have been easier and would
have measured a different program than the one that has to work — the classic
green test that controls nothing.

AUDIO IS PART OF THE QUESTION FOR SOME TARGETS
----------------------------------------------
MiniMax H3 is a joint video+audio model and its clips are muxed. A pod that
decodes the picture and silently finds no audio stream trains a video-only LoRA
under an audio target's name, and nothing anywhere says so. The caller passes
`want_audio` from the target profile, and only then is a missing track a
refusal.
"""
import logging

from . import dense_pod_hub as dph

logger = logging.getLogger(__name__)

RESULT_PREFIX = dph.RESULT_PREFIX

# A booted pod answers this in seconds. Bounded anyway: a host wedged on a
# decoder import must not hold a billed instance until the run's max runtime.
DEFAULT_PROBE_BUDGET_SECONDS = 300


class PodProbeUnavailable(RuntimeError):
    """The probe could not be RUN — which is not the same as failing it.

    vast's remote-exec endpoint is constrained to `ls`, `rm` and `du`, so the
    program below never reaches a pod rented there (measured, run #165). That is
    a missing capability of the provider, not a fact about the clips, and the
    caller must not turn it into a failed run: refusing to launch over a check
    that cannot run anywhere would keep the lane down for as long as the
    restriction lasts, which is exactly the opposite of what the check is for.

    Kept separate from `PodDecoderUnusable` so the two never blur: one says the
    pod answered no, this one says nobody asked."""


class PodDecoderUnusable(RuntimeError):
    """This pod cannot read this dataset's clips, and the reason is stated.

    Its own type rather than `PodHubError`: that name reaches a run's error
    field, and telling someone whose video dataset failed that a Hugging Face
    transfer went wrong sends them to the wrong settings page."""


# Free of single quotes on purpose — the whole text is passed to the pod's shell
# inside single quotes (see dense_pod_hub.quote), and escaping is not something
# to be clever about in a command that runs on someone else's machine.
PROBE_PROGRAM = (
    'import json,os,sys\n'
    'folder=sys.argv[1]\n'
    'want_audio=sys.argv[2]=="1"\n'
    'out={"ok":False,"error":None,"clip":None,"frames":0,"decoder":None,'
    '"audio":None,"cv2_error":None}\n'
    'try:\n'
    '    names=sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))\n'
    '    if not names:\n'
    '        raise RuntimeError("no .mp4 reached the pod in "+folder)\n'
    '    path=os.path.join(folder,names[0])\n'
    '    out["clip"]=names[0]\n'
    '    try:\n'
    '        import cv2\n'
    '        cap=cv2.VideoCapture(path)\n'
    '        got=cap.isOpened() and cap.read()[0]\n'
    '        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)\n'
    '        cap.release()\n'
    '        if got:\n'
    '            out["decoder"]="cv2"\n'
    '            out["frames"]=n\n'
    '    except Exception as e:\n'
    '        out["cv2_error"]=str(e)[:200]\n'
    '    if not out["decoder"]:\n'
    '        import av\n'
    '        with av.open(path) as c:\n'
    '            for _f in c.decode(video=0):\n'
    '                out["frames"]=out["frames"]+1\n'
    '                break\n'
    '        if out["frames"]:\n'
    '            out["decoder"]="pyav"\n'
    '    if not out["decoder"]:\n'
    '        raise RuntimeError("neither OpenCV nor PyAV decoded "+names[0])\n'
    '    if want_audio:\n'
    '        import av\n'
    '        with av.open(path) as c:\n'
    '            out["audio"]=len(c.streams.audio)>0\n'
    '        if not out["audio"]:\n'
    '            raise RuntimeError("this target trains on the clip audio and the '
    'pod found no audio stream in "+names[0])\n'
    '    out["ok"]=True\n'
    'except Exception as e:\n'
    '    out["ok"]=False\n'
    '    out["error"]=str(e)[:400]\n'
    'print("' + RESULT_PREFIX + ' "+json.dumps(out))\n'
)


def build_probe_command(pod_dataset_dir, want_audio=False) -> str:
    """The one command the pod runs. `_sized` refuses anything approaching the
    provider's command ceiling, which is what keeps the program from quietly
    growing interpolated arguments instead of argv ones."""
    return dph._sized(' '.join(
        ['python', '-c', dph.quote(PROBE_PROGRAM),
         dph.quote(pod_dataset_dir), dph.quote('1' if want_audio else '0')]))


# The shared pod-side executor: ship what the program needs, run ONE command,
# read ONE result line. Bound to a module name so a test can replace it without
# reaching into `dense_pod_hub`, whose other two callers are unrelated.
_run_program = dph.run_program


def probe_decoder(remote, *, instance_id, pod_dataset_dir, tmp_dir,
                  want_audio=False, budget_seconds=None, on_state=None,
                  should_cancel=None) -> dict:
    """Ask the pod to decode one uploaded clip. Returns the verdict dict
    (`decoder`, `frames`, `clip`, `audio`); raises `PodDecoderUnusable` when the
    pod could not, whatever the reason — a missing decoder, an empty folder, or
    no answer at all."""
    try:
        return _run_program(
            remote, instance_id=instance_id, token=None,
            command=lambda _token_file: build_probe_command(pod_dataset_dir,
                                                            want_audio),
            budget_seconds=int(budget_seconds or DEFAULT_PROBE_BUDGET_SECONDS),
            tmp_dir=tmp_dir, on_state=on_state, should_cancel=should_cancel,
            need_token=False)
    except vast_command_unsupported() as e:
        raise PodProbeUnavailable(str(e)) from e
    except dph.PodHubError as e:
        raise PodDecoderUnusable(str(e)) from e


def vast_command_unsupported():
    """The provider-refused-the-command type, resolved late.

    A function and not a top-level import because `vast_client` reads config at
    import time and this module is imported by the launch path; the lazy shape
    is the one `dense_pod_hub` already uses for the same client."""
    from . import vast_client
    return vast_client.VastCommandUnsupported
