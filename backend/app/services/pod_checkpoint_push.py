"""Send a big local checkpoint to a pod, in slices, and survive an interruption.

WHY THIS EXISTS
---------------
Continuing a full model used to be possible from ONE place: the Hugging Face
copy of the run. Not because sending the local file was a bad idea, but because
the only write seam the pod exposes — ai-toolkit's ``/api/datasets/upload`` —
was driven by ``requests``' ``files=``, which builds the entire multipart body
in memory. An 85 MB LoRA is fine there; a 26 GB checkpoint is an OOM. The
refusal said so honestly, and it named an implementation limit of one route as
if it were a property of the world.

``RemoteAiToolkit.upload_file_slice`` removed the memory cost (the body is
produced as it is sent). This module removes the second and larger objection:
an upload measured in HOURS that restarts from zero when the link hiccups is
not a feature, it is a way to waste an evening and a rented GPU.

HOW A RESUMABLE UPLOAD IS BUILT OUT OF A ROUTE THAT CANNOT RESUME
-----------------------------------------------------------------
The route takes whole files and has no notion of an offset or a Range. So the
file is not sent as one object:

1. it is cut into numbered SLICES (``<name>.p0001`` …), each one its own POST;
2. before sending anything, ONE pod-side program reports which slices are
   already there and how big they are — every slice that already landed at its
   exact expected size is skipped. That is what makes the transfer survive not
   just a dropped connection but an app restart, a reboot, a day later;
3. when every slice is present, ONE pod-side program appends them into the
   destination file **and deletes each slice as it is consumed**, so the peak
   disk cost is the file plus one slice, not the file twice.

Step 3 keeps its own small manifest next to the destination, so an assembly cut
in half resumes at the slice it stopped on instead of starting over.

WHAT IS PROVEN AND WHAT IS NOT
------------------------------
No pod is rented by the test suite. What the tests assert is the slicing
arithmetic, the exact commands, the exact skip decisions on every shape of
partial state, and that a truncated or oversized slice is never accepted. The
transfer itself is proven only in production — the same honesty ``dense_pod_hub``
states about its own two programs.
"""
import logging
import os

from . import dense_pod_hub as hub

logger = logging.getLogger(__name__)

# Where the slices wait, relative to the pod's DATASETS_FOLDER.
POD_DIR_NAME = '_lds_push'

# One slice. Chosen for what an interruption COSTS, not for memory (the body is
# streamed, so a slice of any size is the same handful of megabytes of RAM):
# 2 GiB over a 100 Mbit/s uplink is ~3 minutes of work to redo, and a 26 GB
# checkpoint becomes 13 POSTs rather than hundreds of round-trips.
DEFAULT_SLICE_BYTES = 2 * 1024 * 1024 * 1024

# Per-slice socket timeout. The 300 s that used to cap the whole transfer was
# never a transfer budget: requests' timeout is an inactivity timeout, and the
# part of it that actually bites is waiting for the pod's answer once the body
# is sent — the pod has just written gigabytes to disk. Per SLICE, generous.
SLICE_TIMEOUT_SECONDS = 1800

# The assembly reads and rewrites the whole file on the pod's own disk. Fast
# (local NVMe), but bounded: a degraded host must not hold a paid pod for ever.
DEFAULT_ASSEMBLE_BUDGET_SECONDS = 3600
DEFAULT_PROBE_BUDGET_SECONDS = 600

# How many times one slice is retried before the transfer gives up. Generous on
# purpose: some vast hosts' proxies cut long streams (the download side needs up
# to 400 resumed connections for the same reason), and giving up on a slice
# throws away every slice already on the pod.
SLICE_ATTEMPTS = 5

MANIFEST_SUFFIX = '.ldspush.json'


class PodPushError(RuntimeError):
    """The checkpoint could not be placed on the pod."""


def slice_name(remote_name: str, index: int) -> str:
    """``<name>.p0001``. The route sanitises FILENAMES to [A-Za-z0-9._-], and
    this spelling survives it unchanged — a suffix that got rewritten would
    land every slice under one mangled name and silently overwrite itself."""
    return f'{remote_name}.p{index:04d}'


def plan_slices(total_bytes: int, slice_bytes: int = DEFAULT_SLICE_BYTES) -> list:
    """The cut, as a list of ``{'index', 'offset', 'length'}``.

    An empty file gets ONE empty slice rather than none: zero slices would make
    "everything already landed" and "there is nothing to send" the same state,
    and the assembly would then produce no file at all while reporting success.
    """
    total = max(0, int(total_bytes))
    step = max(1, int(slice_bytes))
    out = []
    offset = 0
    while offset < total:
        out.append({'index': len(out) + 1, 'offset': offset,
                    'length': min(step, total - offset)})
        offset += step
    if not out:
        out.append({'index': 1, 'offset': 0, 'length': 0})
    return out


def pod_paths(datasets_folder: str, job_name: str) -> dict:
    root = str(datasets_folder or '').rstrip('/') or '/workspace/datasets'
    safe = ''.join(c if (c.isalnum() or c in '._-') else '_'
                   for c in str(job_name or 'run'))
    return {'root': root, 'staging': f'{root}/{POD_DIR_NAME}/{safe}'}


# -- the two pod-side programs -------------------------------------------------
# CONSTANTS, exactly like dense_pod_hub's. Every user-influenced value is a
# shell-quoted argv entry read with sys.argv — nothing is interpolated into the
# program text, so a job name carrying quotes cannot become code. Free of single
# quotes, because the whole text travels inside single quotes.

PROBE_PROGRAM = (
    'import json,os,shutil,sys\n'
    'staging,dest=sys.argv[1:3]\n'
    'out={"ok":True,"error":None,"slices":{},"dest_bytes":-1,"free_bytes":-1,'
    '"assembled":0}\n'
    'try:\n'
    '    if os.path.isdir(staging):\n'
    '        for n in os.listdir(staging):\n'
    '            p=os.path.join(staging,n)\n'
    '            if os.path.isfile(p):\n'
    '                out["slices"][n]=os.path.getsize(p)\n'
    '    if os.path.isfile(dest):\n'
    '        out["dest_bytes"]=os.path.getsize(dest)\n'
    '    m=dest+"' + MANIFEST_SUFFIX + '"\n'
    '    if os.path.isfile(m):\n'
    '        out["assembled"]=int(json.load(open(m)).get("done") or 0)\n'
    # Free space where the DESTINATION goes, not where python happens to run:
    # a pod can mount the training folder on a different volume, and a refusal
    # computed against the wrong volume is worse than no refusal.
    '    probe=os.path.dirname(dest) or "/"\n'
    '    while probe and not os.path.isdir(probe):\n'
    '        probe=os.path.dirname(probe)\n'
    '    out["free_bytes"]=shutil.disk_usage(probe or "/").free\n'
    'except Exception as e:\n'
    '    out["ok"]=False\n'
    '    out["error"]=str(e)[:400]\n'
    'print("' + hub.RESULT_PREFIX + ' "+json.dumps(out))\n'
)

ASSEMBLE_PROGRAM = (
    'import json,os,sys\n'
    'staging,dest,prefix=sys.argv[1:4]\n'
    'count=int(sys.argv[4]);total=int(sys.argv[5])\n'
    'out={"ok":False,"error":None,"bytes":0,"appended":0}\n'
    'man=dest+"' + MANIFEST_SUFFIX + '"\n'
    'try:\n'
    '    names=[prefix+".p%04d"%i for i in range(1,count+1)]\n'
    '    done=0\n'
    '    if os.path.isfile(man):\n'
    '        done=int(json.load(open(man)).get("done") or 0)\n'
    '    have=os.path.getsize(dest) if os.path.isfile(dest) else 0\n'
    # The manifest is a CLAIM; the file on disk is the fact. If they disagree
    # the manifest is dropped and assembly restarts, because appending to a
    # file of unknown length is how a checkpoint becomes garbage that still
    # loads.
    '    missing=[n for n in names[done:] '
    'if not os.path.isfile(os.path.join(staging,n))]\n'
    '    if missing or done>count:\n'
    '        done=0\n'
    '        have=0\n'
    '        missing=[n for n in names if not os.path.isfile(os.path.join(staging,n))]\n'
    '        if missing:\n'
    '            raise Exception("missing slices: "+",".join(missing[:6]))\n'
    '    rest=sum(os.path.getsize(os.path.join(staging,n)) for n in names[done:])\n'
    '    if done and have+rest!=total:\n'
    '        done=0\n'
    '        have=0\n'
    '        rest=sum(os.path.getsize(os.path.join(staging,n)) for n in names)\n'
    '    if have+rest!=total:\n'
    '        raise Exception("slice sizes total %d, expected %d"%(have+rest,total))\n'
    '    os.makedirs(os.path.dirname(dest) or ".",exist_ok=True)\n'
    '    fh=open(dest,"ab" if done else "wb")\n'
    '    try:\n'
    '        for i in range(done,count):\n'
    '            p=os.path.join(staging,names[i])\n'
    '            src=open(p,"rb")\n'
    '            try:\n'
    '                while True:\n'
    '                    b=src.read(8388608)\n'
    '                    if not b:\n'
    '                        break\n'
    '                    fh.write(b)\n'
    '            finally:\n'
    '                src.close()\n'
    '            fh.flush()\n'
    '            os.remove(p)\n'
    '            out["appended"]=i+1-done\n'
    '            open(man,"w").write(json.dumps({"done":i+1}))\n'
    '    finally:\n'
    '        fh.close()\n'
    '    got=os.path.getsize(dest)\n'
    '    if got!=total:\n'
    '        raise Exception("assembled %d bytes, expected %d"%(got,total))\n'
    '    out["bytes"]=got\n'
    '    out["ok"]=True\n'
    '    os.remove(man)\n'
    '    try:\n'
    '        os.rmdir(staging)\n'
    '    except Exception:\n'
    '        pass\n'
    'except Exception as e:\n'
    '    out["error"]=str(e)[:400]\n'
    'print("' + hub.RESULT_PREFIX + ' "+json.dumps(out))\n'
)


def build_probe_command(staging_dir, dest_path) -> str:
    return _sized(' '.join(['python', '-c', hub.quote(PROBE_PROGRAM),
                            hub.quote(staging_dir), hub.quote(dest_path)]))


def build_assemble_command(staging_dir, dest_path, prefix, count, total) -> str:
    return _sized(' '.join(['python', '-c', hub.quote(ASSEMBLE_PROGRAM),
                            hub.quote(staging_dir), hub.quote(dest_path),
                            hub.quote(prefix), hub.quote(int(count)),
                            hub.quote(int(total))]))


# vast's command endpoint carries 16384 characters. Same ceiling discipline as
# dense_pod_hub: the programs are constants of a few kilobytes and every value
# rides in argv, so anything approaching this means something started being
# interpolated into the program text. The slice LIST in particular is never
# sent — the program derives the names from a count, which is why a 26 GB
# transfer and a 260 GB one produce the same command length.
MAX_COMMAND_CHARS = 8192


def _sized(command: str) -> str:
    if len(command) > MAX_COMMAND_CHARS:
        raise PodPushError(
            f'the pod command grew to {len(command)} characters (ceiling '
            f'{MAX_COMMAND_CHARS}) — refusing to send it')
    return command


def _program(remote, *, instance_id, command, budget_seconds, tmp_dir, vast,
             on_state=None, should_cancel=None, **kw) -> dict:
    """Run one of the programs above and return its verdict.

    Routed through ``dense_pod_hub.run_program`` rather than a second
    ``execute_command`` caller: shipping nothing, running ONE program and
    reading ONE result line is the same three steps whatever the program does,
    and two implementations of "what does a failure look like" is exactly the
    kind of divergence that shows up once, in production, on a rented pod.
    """
    try:
        return hub.run_program(
            remote, instance_id=instance_id, token=None,
            command=lambda _token_file: command, budget_seconds=budget_seconds,
            tmp_dir=tmp_dir, vast=vast, on_state=on_state,
            should_cancel=should_cancel, need_token=False, **kw)
    except hub.PodHubError as e:
        raise PodPushError(str(e)) from e


def probe(remote, *, instance_id, staging_dir, dest_path, tmp_dir, vast=None,
          budget_seconds=DEFAULT_PROBE_BUDGET_SECONDS, **kw) -> dict:
    """What is already on the pod: slice sizes, destination size, free space."""
    return _program(remote, instance_id=instance_id,
                    command=build_probe_command(staging_dir, dest_path),
                    budget_seconds=budget_seconds, tmp_dir=tmp_dir, vast=vast, **kw)


def disk_shortfall(free_bytes, total_bytes, slice_bytes=DEFAULT_SLICE_BYTES,
                   already_bytes=0) -> dict | None:
    """The pod-disk refusal, with its arithmetic — or None when it fits.

    The peak is the finished file plus ONE slice, not the file twice: the
    assembly deletes each slice as it appends it. Slices already on the pod are
    space we are not about to ask for again.
    """
    free = int(free_bytes or 0)
    if free < 0:
        return None                    # unmeasurable never blocks (dld's rule)
    total = max(0, int(total_bytes))
    need = total + min(int(slice_bytes), total) - max(0, int(already_bytes))
    if need <= free:
        return None
    return {'need_bytes': need, 'free_bytes': free,
            'short_bytes': need - free, 'total_bytes': total,
            'slice_bytes': min(int(slice_bytes), total)}


def push_checkpoint(remote, *, instance_id, local_path, dest_dir, remote_name,
                    datasets_folder, job_name, tmp_dir, vast=None,
                    slice_bytes=DEFAULT_SLICE_BYTES,
                    assemble_budget_seconds=DEFAULT_ASSEMBLE_BUDGET_SECONDS,
                    on_state=None, on_progress=None, should_cancel=None,
                    _remote_kwargs=None) -> dict:
    """Place ``local_path`` on the pod as ``dest_dir/remote_name``.

    RAISES on failure, deliberately, for ``fetch_checkpoint``'s reason: this is
    the seeding step of a resume, and a resume that cannot place its checkpoint
    must fail loudly instead of quietly training a brand-new model from step 0
    on the user's money.

    ``on_progress(bytes_done, bytes_total)`` is called as slices land — the byte
    counter the freeze watchdog reads. It is fed from the SLICE boundary, not
    from the socket: bytes that reached the wire but not the pod's disk are not
    progress, and a watchdog that counts them keeps a dead transfer alive.
    """
    total = os.path.getsize(local_path)
    paths = pod_paths(datasets_folder, job_name)
    staging = paths['staging']
    dest_path = f"{str(dest_dir).rstrip('/')}/{remote_name}"
    plan = plan_slices(total, slice_bytes)

    def note(detail):
        if on_state:
            try:
                on_state(detail)
            except Exception:
                pass

    note('Checking what the pod already has…')
    state = probe(remote, instance_id=instance_id, staging_dir=staging,
                  dest_path=dest_path, tmp_dir=tmp_dir, vast=vast,
                  should_cancel=should_cancel, **(_remote_kwargs or {}))
    landed = state.get('slices') or {}

    # Already whole? Then this is a re-entry after the assembly succeeded and
    # something later failed — do not send 26 GB again to prove it.
    if int(state.get('dest_bytes') or -1) == total and not landed:
        note('The pod already has this checkpoint.')
        if on_progress:
            try:
                on_progress(total, total)
            except Exception:
                pass
        return {'bytes': total, 'sent_bytes': 0, 'slices': len(plan),
                'sent_slices': 0, 'skipped_slices': len(plan), 'reused': True}

    todo = []
    already = 0
    for part in plan:
        name = slice_name(remote_name, part['index'])
        size = landed.get(name)
        # EXACT size or it does not count. A short slice is a cut transfer and
        # a long one belongs to a different file; both would assemble into a
        # checkpoint that is the right length and the wrong bytes.
        if size == part['length']:
            already += part['length']
        else:
            todo.append(part)

    short = disk_shortfall(state.get('free_bytes'), total, slice_bytes, already)
    if short:
        raise PodPushError(
            f"the pod has {_gb(short['free_bytes'])} free and this needs "
            f"{_gb(short['need_bytes'])} — the {_gb(short['total_bytes'])} "
            f"checkpoint plus one {_gb(short['slice_bytes'])} slice while it is "
            f"assembled. Short by {_gb(short['short_bytes'])}. Rent a pod with "
            'more disk, or continue from the Hugging Face copy instead.')

    sent_total = already
    if on_progress:
        try:
            on_progress(sent_total, total)
        except Exception:
            on_progress = None

    for part in todo:
        name = slice_name(remote_name, part['index'])
        base = sent_total
        last_error = None
        for attempt in range(SLICE_ATTEMPTS):
            if should_cancel is not None and should_cancel():
                raise TransferCancelledPush(
                    f'upload cancelled after {_gb(base)} of {_gb(total)} — the '
                    'slices already on the pod are kept, so continuing this run '
                    'again resumes where it stopped')
            note(f'Sending the checkpoint to the pod — slice {part["index"]}'
                 f'/{len(plan)} ({_gb(base)} of {_gb(total)} done)')
            try:
                remote.upload_file_slice(
                    paths['root'], staging, name, local_path,
                    offset=part['offset'], length=part['length'],
                    timeout=SLICE_TIMEOUT_SECONDS,
                    on_progress=None, should_cancel=should_cancel)
                break
            except Exception as e:
                if type(e).__name__ == 'TransferCancelled':
                    raise TransferCancelledPush(str(e)) from e
                last_error = e
                logger.warning('slice %s attempt %s failed: %s',
                               name, attempt + 1, e)
        else:
            raise PodPushError(
                f'slice {part["index"]} of {len(plan)} could not be sent after '
                f'{SLICE_ATTEMPTS} attempts ({last_error}). Everything sent so '
                f'far ({_gb(base)}) stays on the pod — continuing this run '
                'again picks up from there.')
        sent_total = base + part['length']
        if on_progress:
            try:
                on_progress(sent_total, total)
            except Exception:
                on_progress = None

    note('Assembling the checkpoint on the pod…')
    result = _program(
        remote, instance_id=instance_id,
        command=build_assemble_command(staging, dest_path, remote_name,
                                       len(plan), total),
        budget_seconds=assemble_budget_seconds, tmp_dir=tmp_dir, vast=vast,
        should_cancel=should_cancel, **(_remote_kwargs or {}))
    if int(result.get('bytes') or 0) != total:
        raise PodPushError(
            f"the pod assembled {result.get('bytes')} bytes, expected {total}")
    return {'bytes': total, 'sent_bytes': total - already, 'slices': len(plan),
            'sent_slices': len(todo), 'skipped_slices': len(plan) - len(todo),
            'reused': False}


class TransferCancelledPush(PodPushError):
    """The user stopped the transfer. Its own type because the slices already
    on the pod are deliberately KEPT — a cancel that threw away 20 GB of
    progress would make it the most expensive button in the app, which is the
    reasoning ``RemoteAiToolkit.TransferCancelled`` already spells out for the
    download direction."""


def _gb(n) -> str:
    v = float(n or 0)
    if v < 1e9:
        return f'{v / 1e6:.0f} MB'
    return f'{v / 1e9:.1f} GB'
