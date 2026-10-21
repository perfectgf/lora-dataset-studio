"""Run the fp8 export ON THE POD, at the end of a successful dense run.

WHY ON THE POD
--------------
The bf16 master is a ~26 GB file that already lives on the rented machine, next
to a GPU and a fast link to Hugging Face. Quantizing it there costs minutes of a
pod that is about to be destroyed anyway; quantizing it at home costs a 26 GB
download first, on a machine that may not have the RAM for it. The window is
narrow — after ``status == completed`` and before the instance is destroyed —
which is exactly where this module is called from.

HOW THE CODE GETS THERE
-----------------------
There is no shell in the ai-toolkit UI API, so this uses the two seams that do
exist:

* **write** — the pod's ``/api/datasets/upload`` route accepts a direct child
  of DATASETS_FOLDER. ``seed_checkpoint`` ships ``fp8_export.py`` to the
  ``_lds_fp8`` directory, byte for byte — the SAME file the unit tests
  exercise, so there is no second implementation to drift.
* **execute** — vast's asynchronous instance-command endpoint
  (``vast_client.execute_command``), which returns a URL where the combined
  output appears when the command ends.

FAIL-OPEN, ALWAYS
-----------------
Every failure in this module is recorded and swallowed. By the time it runs, the
trainer has already pushed the bf16 master to the private repo: the run is a
SUCCESS with or without an fp8 twin. Turning "the convenience export did not
work" into "the run failed" would throw away hours of paid GPU for a file the
user can regenerate.

WHAT IS PROVEN AND WHAT IS NOT
------------------------------
The quantizer is unit tested against real torch and its output was fed to
ComfyUI's own ``convert_old_quants``. The pod-side execution is NOT smoke tested
— no pod is rented by the test suite — so what is asserted here is the exact
command and payload that would be sent. That distinction is deliberate and is
the reason every step below reports its own state instead of assuming.
"""
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# Where the exporter and its credential land on the pod, relative to the pod's
# DATASETS_FOLDER. A dedicated directory, so nothing here can ever be mistaken
# for a dataset by the trainer.
POD_DIR_NAME = '_lds_fp8'
POD_SCRIPT_NAME = 'fp8_export.py'
POD_TOKEN_NAME = 'hf_token.txt'

# The single line the pod script prints. Parsed rather than scraped: the command
# output also carries pip/torch chatter.
RESULT_PREFIX = 'LDS_FP8_RESULT'
_RESULT_RE = re.compile(re.escape(RESULT_PREFIX) + r'\s+(\{.*\})')

DEFAULT_BUDGET_SECONDS = 1800          # 30 min: measured conversions are minutes
_POLL_SECONDS = 15
_POLL_SLACK_SECONDS = 300              # queueing + upload after the conversion

# What vast will carry. The sibling lane (cloud_quantize) has to fit its WHOLE
# program into vast's 16384-character ask and was refused twice for exceeding it;
# this path stays small for a structural reason — the exporter travels as an
# uploaded FILE and the command only names it. This ceiling exists so that
# structure cannot quietly change: it is a path-and-integers command, and
# anything approaching a kilobyte means something got inlined into it.
MAX_COMMAND_CHARS = 2048


class Fp8DeliveryError(RuntimeError):
    """Internal signal. Never escapes ``run_pod_fp8_export``."""


def script_source() -> str:
    """The exporter's own source text — one implementation, two machines."""
    from lds_sdk.cloud_host.services import fp8_export
    with open(fp8_export.__file__, encoding='utf-8') as fh:
        return fh.read()


def pod_paths(datasets_folder: str) -> dict:
    root = str(datasets_folder or '').rstrip('/') or '/workspace/datasets'
    directory = f'{root}/{POD_DIR_NAME}'
    return {'root': root, 'dir': directory,
            'script': f'{directory}/{POD_SCRIPT_NAME}',
            'token': f'{directory}/{POD_TOKEN_NAME}'}


def build_command(paths: dict, training_folder: str, repo_id: str, *,
                  budget_seconds=DEFAULT_BUDGET_SECONDS, drop_bf16=False,
                  with_token=True) -> str:
    """The exact shell line sent to the pod.

    Single-quoted arguments only: every value here is a server-resolved path or
    an integer, but a repo id is user-influenced (it carries the dataset name)
    and a quoting mistake in a remote command is not a bug anyone finds twice.
    """
    def q(value):
        return "'" + str(value).replace("'", "'\\''") + "'"

    parts = ['python', q(paths['script']),
             '--src-dir', q(training_folder),
             '--repo-id', q(repo_id),
             '--budget-seconds', str(int(budget_seconds))]
    if with_token:
        parts += ['--token-file', q(paths['token'])]
    if drop_bf16:
        parts.append('--drop-bf16')
    command = ' '.join(parts)
    if len(command) > MAX_COMMAND_CHARS:
        # Fail-open, like everything here: the run is already delivered, so a
        # command we do not trust costs an fp8 twin, never the training.
        raise Fp8DeliveryError(
            f'the pod command grew to {len(command)} characters (ceiling '
            f'{MAX_COMMAND_CHARS}) — refusing to send it')
    return command


def parse_result(output) -> dict | None:
    """The exporter's JSON verdict from the command's combined output."""
    match = _RESULT_RE.search(str(output or ''))
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _upload_text(remote, paths, name, text, tmp_dir):
    """Write one text file into the pod's ``_lds_fp8`` directory.

    Reuses ``seed_checkpoint`` — the only write seam the pod exposes — which
    takes a LOCAL file, so the payload is staged next to the run's other
    temporaries and removed immediately.
    """
    local = os.path.join(tmp_dir, name)
    with open(local, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    try:
        remote.seed_checkpoint(paths['root'], paths['dir'], name, local)
    finally:
        try:
            os.remove(local)
        except OSError:
            pass


def run_pod_fp8_export(run, remote, *, instance_id, repo_id, hf_token,
                       keep_bf16=True, budget_seconds=DEFAULT_BUDGET_SECONDS,
                       tmp_dir=None, vast=None, on_state=None,
                       _sleep=time.sleep, _now=time.monotonic,
                       upload=True) -> dict:
    """Ship, execute and collect the fp8 export. NEVER raises.

    Returns ``{'state', 'detail', 'result'}`` where ``state`` is one of
    ``done`` / ``failed`` / ``skipped``. The caller stamps it on the run and
    carries on with the delivery verification either way.

    ``upload=False`` writes the twin on the pod and stops there — it is what a
    run delivering to the local disk wants: the file is then HARVESTED like the
    master, and pushing a second ~10 GB object into a private Hugging Face
    quota to bring it home would be paying twice for it. The bf16 master is
    never dropped in that mode (the CLI only deletes it after a confirmed
    upload), which is also the only safe reading.
    """
    from lds_cloud_training import vast_client
    vast = vast or vast_client
    out = {'state': 'failed', 'detail': '', 'result': None}

    def note(detail):
        out['detail'] = detail
        if on_state:
            try:
                on_state(detail)
            except Exception:
                pass

    if not instance_id:
        out['state'] = 'skipped'
        note('no pod instance to run the fp8 export on')
        return out
    if upload and not repo_id:
        out['state'] = 'skipped'
        note('no Hugging Face delivery repository for the fp8 export')
        return out
    try:
        settings = remote.get_settings() or {}
        paths = pod_paths(settings.get('DATASETS_FOLDER') or '')
        training_folder = settings.get('TRAINING_FOLDER') or ''
        if not training_folder:
            raise Fp8DeliveryError('the pod did not report its training folder')
        staging = tmp_dir or os.path.dirname(os.path.abspath(
            getattr(run, 'checkpoint_local_path', None) or '')) or os.getcwd()
        os.makedirs(staging, exist_ok=True)

        note('Shipping the fp8 exporter to the pod…')
        _upload_text(remote, paths, POD_SCRIPT_NAME, script_source(), staging)
        if hf_token and upload:
            _upload_text(remote, paths, POD_TOKEN_NAME, hf_token, staging)

        command = build_command(paths, training_folder,
                                repo_id if upload else '',
                                budget_seconds=budget_seconds,
                                drop_bf16=upload and not keep_bf16,
                                with_token=bool(hf_token) and upload)
        note('Quantizing the checkpoint to fp8 on the pod…')
        result_url = vast.execute_command(instance_id, command)

        deadline = _now() + budget_seconds + _POLL_SLACK_SECONDS
        output = None
        while _now() < deadline:
            _sleep(_POLL_SECONDS)
            try:
                output = vast.fetch_command_result(result_url)
            except Exception:
                output = None            # transient: keep waiting, not failing
            if output is not None:
                break
        if output is None:
            raise Fp8DeliveryError(
                'the pod did not report an fp8 export result in time')
        parsed = parse_result(output)
        if parsed is None:
            raise Fp8DeliveryError(
                'the fp8 exporter produced no result line on the pod')
        out['result'] = parsed
        if not parsed.get('ok'):
            raise Fp8DeliveryError(str(parsed.get('error') or 'fp8 export failed'))
        if upload and not parsed.get('uploaded'):
            raise Fp8DeliveryError(
                'the fp8 file was written on the pod but never uploaded')
        out['state'] = 'done'
        note(_success_detail(parsed, keep_bf16, upload=upload))
    except Exception as e:
        # The bf16 master is already delivered — this is a missing convenience,
        # not a lost run, and it must read that way everywhere it is shown.
        logger.warning('run %s: fp8 export unavailable (%s)',
                       getattr(run, 'id', '?'), e)
        out['state'] = 'failed'
        note(f'fp8 export unavailable — the bf16 model was delivered ({e})'[:400])
    return out


def _success_detail(parsed, keep_bf16, upload=True) -> str:
    name = os.path.basename(str(parsed.get('path') or 'model_fp8.safetensors'))
    size = parsed.get('bytes_after')
    size_text = f' ({size / 1000 ** 3:.1f} GB)' if isinstance(size, (int, float)) else ''
    if not upload:
        return f'fp8 export {name}{size_text} written on the pod, ready to fetch.'
    tail = ('the bf16 master was kept next to it' if keep_bf16
            else 'the bf16 master was replaced by it')
    return f'fp8 export {name}{size_text} uploaded — {tail}.'
