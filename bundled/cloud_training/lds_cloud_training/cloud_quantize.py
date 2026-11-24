"""Quantize a checkpoint that is ALREADY on Hugging Face, without downloading it.

THE PROBLEM
-----------
A dense run delivers a ~26 GB bf16 master into a private Hugging Face repo. To
get the ~10 GB fp8 file from it locally you first have to pull 26 GB down and
push 10 GB back — an hour of home bandwidth for a conversion that takes under a
minute of CPU. Runs delivered before the automatic pod export shipped are all in
that position, including the ones already trained.

So: rent a cheap machine for a few minutes, let IT do the round trip on a
datacentre link, and download only the 10 GB result. The user's own connection
never sees the master.

WHY THE CHEAPEST GPU, AND WHY THAT IS NOT A CONTRADICTION
---------------------------------------------------------
Quantization is an elementwise cast plus one reduction per tensor: measured at
~1.2 GB/s on an ordinary CPU, i.e. well under a minute of arithmetic for 26 GB.
The job is entirely bound by the network — 26 GB down, ~10 GB up. The GPU is
never touched. vast.ai has no CPU-only instances, so the selection asks for the
smallest card that clears the price filter and instead filters on DOWNLINK
bandwidth and disk, which are the things that actually decide the bill.

HOW THE POD IS DRIVEN, AND HOW ITS CLEANUP IS TRACKED
-------------------------------------------------------
No inbound connection to the pod is needed:

* everything it does is in its ``onstart`` script — the exporter's own source is
  embedded (base64) so THIS runs the same code the unit tests exercise;
* it reports back by uploading a tiny JSON result file to the SAME Hugging Face
  repo, which we already have a token for. We poll the Hub, not the pod.

Cleanup is attempted on every outcome, with unresolved billing kept visible:

1. the monitor requests verified cleanup in a ``finally`` — success, failure or
   exception;
2. a HARD deadline (``cloud.quantize.max_minutes``, default 60) triggers cleanup
   even if nothing was ever reported;
3. recovery uses a durable local receipt and the original provider credential.
   A matching label alone never authorizes deletion. Ambiguous creation or
   cleanup remains visible and blocks another rental until safely resolved.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid

from filelock import FileLock, Timeout

from lds_sdk.cloud_host import config as cfg
from lds_sdk.cloud_host.extensions import db
from lds_sdk.cloud_host.job_queue import queue_manager
from lds_sdk.cloud_host.models import SystemState
from lds_sdk.cloud_host.services import fp8_export
from lds_cloud_training import vast_client

logger = logging.getLogger(__name__)

_STATE_KEY = 'cloud_quantize'
_STATE_TTL = 6 * 3600
_RENTAL_KEY = 'cloud_quantize_rental'
_lock = threading.Lock()
_credentials = {}

# Labels are unique within a durable local intent, never ownership by prefix.
LABEL_PREFIX = 'lds-quantize-'

# The pod reports here, in the repo it is already authenticated against.
RESULT_FILE = '_lds_fp8_result.json'

# Hard ceiling on the whole rental. A 26 GB pull + 10 GB push on a filtered host
# is minutes; an hour means something is wrong and the machine must go.
DEFAULT_MAX_MINUTES = 60
DEFAULT_POLL_SECONDS = 20
# The cast itself is CPU-bound and fast; this bounds it inside the pod so a
# pathological file cannot hold the rental open to the hard deadline.
DEFAULT_EXPORT_BUDGET_SECONDS = 1800
# Smallest card that still clears the filters — the GPU is never used (see note).
MIN_VRAM_GB = 8

# The quote is the sentence the user clicked on. Renting is a fresh search, so
# the machine that gets rented may not be the machine that was priced: a few
# cents of drift is market noise, a different price CLASS is not. Above this
# ceiling the lane says so and rents nothing. The absolute term keeps the
# percentage from being absurd on the cheap end ($0.08/h + 25% = 2 cents).
QUOTE_PRICE_TOLERANCE = 1.25
QUOTE_PRICE_ABSOLUTE = 0.05


class CloudQuantizeError(ValueError):
    """Refusal with a sentence for the user."""


def _cfg() -> dict:
    return ((cfg.get('cloud') or {}).get('quantize') or {})


def _int_cfg(key, default) -> int:
    try:
        return int(_cfg().get(key) or default)
    except (TypeError, ValueError):
        return default


def max_minutes() -> int:
    return max(5, _int_cfg('max_minutes', DEFAULT_MAX_MINUTES))


def status() -> dict:
    return queue_manager._get_system_state(_STATE_KEY, {}) or {}


def _receipt():
    # The general KV reader treats malformed/expired values as absent and may
    # delete them. A rental proof must never inherit those cache semantics.
    row = db.session.get(SystemState, _RENTAL_KEY)
    if row is None:
        return None
    try:
        envelope = json.loads(row.value)
    except (TypeError, ValueError):
        envelope = None
    if not isinstance(envelope, dict) or envelope.get('exp') is not None:
        raise CloudQuantizeError('The local quantization rental proof is damaged; review the provider manually.')
    value = envelope.get('v')
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or not isinstance(value.get('job_id'), str)
            or not re.fullmatch(r'[a-f0-9]{32}', value['job_id'])
            or value.get('label') != LABEL_PREFIX + value['job_id']
            or not isinstance(value.get('fingerprint'), str)
            or not re.fullmatch(r'[a-f0-9]{64}', value['fingerprint'])
            or any(type(value.get(key)) is not bool for key in ('pending', 'released', 'delete_pending'))
            or (value.get('instance_id') is not None and not _instance_id(value['instance_id']))
            or (value['pending'] and (value['released'] or value.get('instance_id') is not None))
            or (value['delete_pending'] and value.get('instance_id') is None)):
        raise CloudQuantizeError('The local quantization rental proof is damaged; review the provider manually.')
    return dict(value)


def _instance_id(value):
    return (type(value) is int and value > 0) or (
        isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*', value) is not None)


def _save_receipt(receipt, **updates):
    value = {**receipt, **updates}
    try:
        queue_manager._set_system_state(_RENTAL_KEY, value, ttl_seconds=None)
    except Exception:
        db.session.rollback()  # discard uncommitted proof before any recovery read
        raise
    return value


def has_unreleased_rental():
    """Local-only disable/admission check, including damaged proof and legacy jobs."""
    try:
        receipt = _receipt()
    except CloudQuantizeError:
        return True
    legacy = status() if not receipt else {}
    return bool((receipt and not receipt['released']) or
                (not receipt and (legacy.get('instance_id') is not None
                                  or legacy.get('status') in ('provisioning', 'running')
                                  or legacy.get('cleanup_pending'))))


def _lease():
    # Held across the worker lifetime, also preventing a second LDS process
    # sharing this data directory from treating that active rental as orphaned.
    directory = Path(cfg.data_dir())
    directory.mkdir(parents=True, exist_ok=True)
    lease = FileLock(str(directory / 'cloud-quantize.lock'), timeout=0, thread_local=False)
    lease.acquire()
    return lease


def _remember(credential):
    _credentials[credential.fingerprint] = credential
    return credential


def _credential_for(receipt):
    credential = _credentials.get(receipt['fingerprint'])
    if credential is None:
        credential = vast_client.capture_credentials()
    if credential.fingerprint != receipt['fingerprint']:
        raise CloudQuantizeError('Restore the original vast.ai credential before recovering this rental.')
    return _remember(credential)


def _cleanup_pending(message):
    current = status()
    queue_manager._set_system_state(_STATE_KEY, {
        **current, 'status': 'error', 'cleanup_pending': True, 'error': message,
    }, ttl_seconds=None)


def _redact_error(error, token=None):
    message = vast_client._scrub(str(error))
    return message.replace(token, '[redacted]') if token else message


def _reserve(credential):
    if has_unreleased_rental():
        raise CloudQuantizeError('A previous quantization rental still needs recovery; no new machine was requested.')
    nonce = uuid.uuid4().hex
    return _save_receipt({'schema_version': 1, 'job_id': nonce, 'label': LABEL_PREFIX + nonce,
                          'fingerprint': credential.fingerprint, 'pending': False,
                          'released': False, 'delete_pending': False, 'instance_id': None,
                          'created_at': time.time()})


def _cleanup_owned(receipt, credential):
    """Caller holds the cross-process lease; never infer ownership from a prefix."""
    if receipt['released']:
        return []
    if receipt['instance_id'] is None and not receipt['pending']:
        _save_receipt(receipt, released=True, released_at=time.time())
        return []                         # the durable CREATE boundary was never crossed
    if credential.fingerprint != receipt['fingerprint']:
        raise CloudQuantizeError('The cleanup account does not match the rental proof.')
    instances = vast_client.list_instances(credential=credential)
    if (not isinstance(instances, list) or any(not isinstance(item, dict)
            or not _instance_id(item.get('instance_id'))
            or (item.get('label') is not None and not isinstance(item.get('label'), str))
            for item in instances)):
        raise CloudQuantizeError('The provider inventory is incomplete; rental cleanup was not attempted.')
    matching = [item for item in instances if item['label'] == receipt['label']]
    known = receipt['instance_id']
    target = [item for item in instances if known is not None and str(item['instance_id']) == str(known)]
    if len(matching) > 1 or len(target) > 1:
        raise CloudQuantizeError('The provider returned ambiguous rental identities; review cleanup manually.')
    if known is None:
        if not matching:
            raise CloudQuantizeError('The CREATE result is still uncertain; no second machine will be requested.')
        target = matching
        known = target[0]['instance_id']
        if sum(str(item['instance_id']) == str(known) for item in instances) != 1:
            raise CloudQuantizeError('The provider returned an ambiguous instance id; cleanup was refused.')
        receipt = _save_receipt(receipt, instance_id=known, pending=False)
    elif not target:
        if matching:
            raise CloudQuantizeError('The rental label now names another instance; cleanup was refused.')
        observed = vast_client.get_instance(known, credential=credential)
        if observed is None:
            if not receipt['delete_pending']:
                raise CloudQuantizeError('Absence alone does not confirm rental cleanup; review the provider manually.')
        elif (not isinstance(observed, dict) or str(observed.get('instance_id')) != str(known)
              or observed.get('label') != receipt['label']):
            raise CloudQuantizeError('The observed instance does not match the local rental proof.')
        # A persisted DELETE intent may be retried with the original account;
        # only the provider's successful/idempotent DELETE response acquits it.
        target = [observed] if observed is not None else []
    if target and target[0]['label'] != receipt['label']:
        raise CloudQuantizeError('The observed instance does not match the local rental proof.')
    # A duplicate id anywhere in the listing is ambiguous, including adoption.
    if sum(str(item['instance_id']) == str(known) for item in instances) > 1:
        raise CloudQuantizeError('The provider returned an ambiguous instance id; cleanup was refused.')
    receipt = _save_receipt(receipt, delete_pending=True)
    if vast_client.destroy_instance(known, credential=credential) is not True:
        raise CloudQuantizeError('The provider did not confirm cleanup; the rental may still be billing.')
    _save_receipt(receipt, released=True, pending=False, released_at=time.time())
    return [known]


# --- planning -------------------------------------------------------------------

def _hf_api(token):
    from lds_sdk.cloud_host.services.hf_publish import _make_api
    return _make_api(token)


def _weight_sibling(info, filename=None):
    """(path, size) of the dense weight file to quantize, from Hub metadata."""
    best = None
    for sibling in (getattr(info, 'siblings', None) or []):
        path = str(getattr(sibling, 'rfilename', '') or '')
        if not path.lower().endswith('.safetensors'):
            continue
        if path.endswith('_fp8.safetensors'):
            continue                    # an export, not a master
        if filename and os.path.basename(path) != os.path.basename(filename):
            continue
        size = getattr(sibling, 'size', None)
        lfs = getattr(sibling, 'lfs', None)
        if size is None and lfs is not None:
            size = getattr(lfs, 'size', None) if not isinstance(lfs, dict) else lfs.get('size')
        size = int(size) if isinstance(size, int) and not isinstance(size, bool) else 0
        if best is None or size > best[1]:
            best = (path, size)
    return best


def _choose(offers):
    """The offer this lane would rent, by the SAME rule as a training launch.

    Delegating to cloud_training is the point: hosts blacklisted for failing to
    boot a training pod fail a quantization pod too, and "cheapest wins" is what
    put a 57 GB machine at the top of the list for an 86 GB job. Quoting with
    one rule and renting with another is also how a price guard turns into a
    permanent refusal, so plan() and the rental share this function.
    """
    from lds_cloud_training import cloud_training
    pool = cloud_training._filter_offers([o for o in (offers or []) if o])
    return cloud_training._best_of(pool) if pool else None


def plan(repo_id, *, filename=None, keep_bf16=True, token=None, _api=None,
         _offers=None) -> dict:
    """What renting a machine to quantize ``repo_id`` would cost and produce.

    Every refusal is raised here, BEFORE anything is rented — including the
    Hugging Face storage one: the fp8 file adds ~10 GB to a repo that may
    already be at the account's ceiling.
    """
    repo = str(repo_id or '').strip()
    if '/' not in repo:
        raise CloudQuantizeError('give the full Hugging Face repository id (owner/name)')
    token = token or cfg.secret('HF_CLOUD_TOKEN') or cfg.secret('HF_TOKEN')
    if not token:
        raise CloudQuantizeError(
            'no Hugging Face token configured — add HF_CLOUD_TOKEN in '
            'Settings ▸ Local tools so the pod can read and write this repository')
    try:
        vast_client.capture_credentials()
    except vast_client.VastError:
        raise CloudQuantizeError('no vast.ai API key configured — add it in Settings ▸ Training')

    api = _api or _hf_api(token)
    try:
        info = api.repo_info(repo_id=repo, repo_type='model', files_metadata=True)
    except Exception as e:
        raise CloudQuantizeError(f'could not read {repo} on Hugging Face: {_redact_error(e, token)}') from e
    found = _weight_sibling(info, filename)
    if not found:
        raise CloudQuantizeError(
            f'{repo} holds no full-precision .safetensors checkpoint to quantize'
            + (f' named {os.path.basename(str(filename))}' if filename else ''))
    weight_path, source_bytes = found
    if any(str(getattr(s, 'rfilename', '')) ==
           fp8_export.fp8_name_for(weight_path) for s in (info.siblings or [])):
        raise CloudQuantizeError(
            f'{fp8_export.fp8_name_for(os.path.basename(weight_path))} is already in '
            f'{repo} — delete it there first if you want to rebuild it.')

    # Disk first: it decides which offers can be rented AT ALL, so it has to be
    # known before the search, and the search has to ask for exactly what the
    # rental will claim.
    disk_gb = _disk_gb_for(source_bytes)
    offers = _offers if _offers is not None else _search_offers(disk_gb)
    offer = _choose(offers)
    minutes = _estimated_minutes(source_bytes, offer)
    price = float((offer or {}).get('dph_total') or 0)
    fp8_typical = fp8_export.typical_fp8_bytes(source_bytes)

    from lds_cloud_training import hf_storage
    forecast = hf_storage.dense_storage_forecast(
        repo.split('/')[0], token, keeps=0 if not keep_bf16 else 0,
        fp8_export=False)
    # The repo already holds the master; the only NEW bytes are the fp8 file.
    forecast['needed_bytes'] = fp8_export.estimate_fp8_bytes(source_bytes)
    if forecast.get('free_bytes') is not None:
        forecast['fits'] = forecast['needed_bytes'] <= forecast['free_bytes']
        forecast['shortfall_bytes'] = (None if forecast['fits'] else
                                       forecast['needed_bytes'] - forecast['free_bytes'])

    return {
        'repo_id': repo,
        'weight_path': weight_path,
        'weight_name': os.path.basename(weight_path),
        'source_bytes': source_bytes,
        'output_name': fp8_export.fp8_name_for(os.path.basename(weight_path)),
        'output_bytes_typical': fp8_typical,
        'keep_bf16': bool(keep_bf16),
        'offer': offer,
        'disk_gb': disk_gb,
        'price_per_hour': price,
        'estimated_minutes': minutes,
        'estimated_cost': round(price * minutes / 60.0, 3),
        'max_minutes': max_minutes(),
        'storage': forecast,
    }


def _search_offers(min_disk_gb=0) -> list:
    c = cfg.get('cloud') or {}
    return vast_client.search_offers(
        min_vram_gb=MIN_VRAM_GB,
        max_dph=float(_cfg().get('max_price_per_hour')
                      or c.get('max_price_per_hour') or 0.80),
        # Downlink is the real cost driver: the whole job is 26 GB in, 10 GB out.
        min_inet_down_mbps=_int_cfg('min_inet_down_mbps',
                                    int(c.get('min_inet_down_mbps') or 200)),
        min_reliability=float(c.get('min_reliability') or 0.98),
        verified_only=bool(c.get('verified_only', True)),
        # The master, its fp8 twin and the Hub cache all land on this disk. An
        # offer that cannot hold them is not a cheaper machine, it is a refused
        # rental (measured: the cheapest offer of a live search had 57 GB free
        # against the 86 GB this job asks for).
        min_disk_gb=int(min_disk_gb or 0))


def _estimated_minutes(source_bytes, offer) -> int:
    """Transfer time, plus a floor for boot and the cast.

    Honest about what is guessed: the downlink is advertised by the host and the
    UPLINK is not published at all, so the return leg is assumed to be half the
    downlink — pessimistic on purpose, since this number becomes a price.
    """
    mbps = float((offer or {}).get('inet_down') or 0) or 200.0
    down_s = (source_bytes * 8) / (mbps * 1e6) if source_bytes else 0
    up_s = (fp8_export.typical_fp8_bytes(source_bytes) * 8) / (mbps / 2 * 1e6) \
        if source_bytes else 0
    return max(6, int((down_s + up_s) / 60) + 6)   # +6 min: boot, cast, push setup


def _disk_gb_for(source_bytes) -> int:
    """The pod holds the master plus its fp8 twin plus the HF cache copy."""
    need = int((source_bytes or 0) / 1e9 * 2.6) + 20
    return max(60, need)


# --- the pod's whole life, in one script ----------------------------------------

def build_onstart(planned: dict, *, export_budget_seconds=None) -> str:
    """The onstart script. Nothing else ever talks to this machine.

    The exporter's SOURCE is embedded base64 rather than fetched: the pod has no
    access to this app, and a second copy of the algorithm pasted into a shell
    string is exactly the drift this whole feature is built to avoid.
    """
    from lds_cloud_training import dense_fp8_delivery
    payload = base64.b64encode(
        dense_fp8_delivery.script_source().encode('utf-8')).decode('ascii')
    budget = int(export_budget_seconds
                 or _int_cfg('export_budget_seconds', DEFAULT_EXPORT_BUDGET_SECONDS))
    drop = '' if planned['keep_bf16'] else ' --drop-bf16'
    # Single-quoted everywhere; the only interpolated values are a repo id, a
    # file name resolved from the Hub, and integers.
    return '\n'.join([
        '#!/bin/bash',
        'set -o pipefail',
        'mkdir -p /workspace/lds && cd /workspace/lds',
        f"echo '{payload}' | base64 -d > fp8_export.py",
        # `huggingface_hub` and nothing else, because that is all the pod
        # imports: the exporter below reads and writes the safetensors format
        # with plain file I/O — deliberately NOT `safe_open`, which memory-maps
        # the whole 26 GB checkpoint — and the two heredocs here import only
        # huggingface_hub. The package rode along from before that change; on a
        # machine billed by the hour an install nothing imports is pure boot.
        'export PIP_CONFIG_FILE=/dev/null',
        'python -m pip --isolated install --disable-pip-version-check --only-binary=:all: -q '
        '--index-url https://lds.perfectgf.fr/models/tools/pypi/simple/ "huggingface_hub>=0.30" || exit 1',
        'python - <<\'PY\'',
        'import os',
        'from huggingface_hub import hf_hub_download',
        f'p = hf_hub_download(repo_id={planned["repo_id"]!r}, '
        f'filename={planned["weight_path"]!r}, local_dir="/workspace/lds/src", '
        'token=os.environ.get("HF_TOKEN"))',
        'open("/workspace/lds/src_path", "w").write(p)',
        'PY',
        'SRC=$(cat /workspace/lds/src_path)',
        f'python fp8_export.py --src "$SRC" --repo-id {planned["repo_id"]!r} '
        f'--budget-seconds {budget}{drop} > /workspace/lds/result.txt 2>&1',
        'python - <<\'PY\'',
        'import json, os, re',
        'from huggingface_hub import HfApi',
        'raw = open("/workspace/lds/result.txt", encoding="utf-8", errors="replace").read()',
        'm = re.search(r"LDS_FP8_RESULT\\s+(\\{.*\\})", raw)',
        'out = json.loads(m.group(1)) if m else {"ok": False, "error": raw[-800:]}',
        'open("/workspace/lds/result.json", "w").write(json.dumps(out))',
        'HfApi(token=os.environ.get("HF_TOKEN")).upload_file('
        'path_or_fileobj="/workspace/lds/result.json", '
        f'path_in_repo={planned.get("result_file", RESULT_FILE)!r}, repo_id={planned["repo_id"]!r}, repo_type="model")',
        'PY',
    ])


# --- running it -----------------------------------------------------------------

def start(app, repo_id, *, filename=None, keep_bf16=True, token=None,
          quoted_price=None, _api=None, _offers=None) -> dict:
    """Rent, convert, upload, then confirm cleanup or retain its recovery proof.

    ``quoted_price`` is the $/h the user actually read on screen before clicking
    (the estimate is a separate, earlier request). It is what the rental is held
    to — see _rent — so a market that moved between the two is reported instead
    of being paid.
    """
    credential = _remember(vast_client.capture_credentials())
    token = token or cfg.secret('HF_CLOUD_TOKEN') or cfg.secret('HF_TOKEN')
    with vast_client.using_credentials(credential):
        planned = plan(repo_id, filename=filename, keep_bf16=keep_bf16, token=token,
                       _api=_api, _offers=_offers)
    if quoted_price:
        try:
            planned['quoted_price_per_hour'] = max(0.0, float(quoted_price))
        except (TypeError, ValueError):
            pass                      # an unreadable quote is simply no quote
    if not planned['offer']:
        raise CloudQuantizeError(
            f'no vast.ai machine with {planned["disk_gb"]} GB of free disk matches '
            'the price cap right now — raise it in Settings ▸ Training and retry')
    reconcile_orphans()
    with _lock:
        try:
            lease = _lease()
        except Timeout as exc:
            raise CloudQuantizeError('A quantization worker already holds this data directory.') from exc
        receipt = None
        try:
            receipt = _reserve(credential)
            planned = {**planned, 'result_file': f'_lds_fp8_result_{receipt["job_id"]}.json'}
            _set('provisioning', planned)

            def _run():
                with app.app_context():
                    _drive(planned, token, _api=_api, _credential=credential, _intent=receipt, _rental_lease=lease)

            threading.Thread(target=_run, daemon=True).start()
        except BaseException:
            # No worker was admitted. A reserved intent cannot have sent CREATE.
            try:
                current = _receipt()
                if (receipt and current and current['job_id'] == receipt['job_id']
                        and not current['pending'] and current['instance_id'] is None):
                    _save_receipt(current, released=True, released_at=time.time())
                    _set('error', planned, error='The quantization worker could not start; no machine was requested.')
            finally:
                lease.release()
            raise
    return planned


def _repriced(planned, offer) -> dict:
    """The plan, re-stated on the machine that was ACTUALLY rented.

    The rental re-searches, so the quoted offer and the rented one need not be
    the same box. Whatever the status panel shows from here on is the real
    price, not the estimate that led to it.
    """
    price = float((offer or {}).get('dph_total') or 0)
    minutes = _estimated_minutes(planned['source_bytes'], offer)
    return {**planned, 'offer': offer, 'price_per_hour': price,
            'gpu_name': (offer or {}).get('gpu_name'),
            'estimated_minutes': minutes,
            'estimated_cost': round(price * minutes / 60.0, 3)}


def _rent(planned, *, label, env, _sleep=None, credential=None):
    """Rent a machine that can hold this job — trying more than one.

    The first cloud quantization died on ``create_instance failed: HTTP 400 {}``
    one second after the click, with no machine and no reason. Two things were
    wrong and both are fixed here: the offer was picked by price alone (the
    cheapest box on the market had 57 GB free against the 86 GB this job asks
    for, which vast refuses outright), and a single refusal ended the job.

    The retry loop, the host blacklist and the bait-price filter are the
    training lane's — imported, not re-implemented.
    """
    from lds_cloud_training import cloud_training
    disk_gb = int(planned.get('disk_gb') or _disk_gb_for(planned['source_bytes']))
    quoted = float(planned.get('quoted_price_per_hour')
                   or planned.get('price_per_hour') or 0)
    onstart = build_onstart(planned)
    image = (_cfg().get('image') or (cfg.get('cloud') or {}).get('image'))

    def _pick(offers):
        if not offers:
            raise CloudQuantizeError(
                'every machine that matched is blacklisted here for failing to '
                'boot recently — nothing was rented; retry in a few minutes')
        ceiling = max(quoted * QUOTE_PRICE_TOLERANCE, quoted + QUOTE_PRICE_ABSOLUTE)
        chosen = _choose([o for o in offers if not quoted
                          or float(o.get('dph_total') or 0) <= ceiling])
        if chosen is None:
            cheapest = min(float(o.get('dph_total') or 0) for o in offers)
            raise CloudQuantizeError(
                f'the ${quoted:.3f}/h machines are gone — the cheapest one left is '
                f'${cheapest:.3f}/h, too far above the estimate you agreed to. '
                'Nothing was rented; ask for a new estimate to see the real price.')
        return chosen

    def _create(offer):
        receipt = _receipt()
        if (not receipt or receipt['released'] or receipt['label'] != label
                or receipt['fingerprint'] != credential.fingerprint or receipt['pending']
                or receipt['instance_id'] is not None):
            raise CloudQuantizeError('A fresh, local rental intent is required before CREATE.')
        receipt = _save_receipt(receipt, pending=True)
        try:
            instance_id = vast_client.create_instance(
                offer['offer_id'], disk_gb=disk_gb, label=label, image=image,
                env=env, onstart=onstart, credential=credential)
        except vast_client.VastCreateUncertain:
            raise
        except vast_client.VastError:
            _save_receipt(receipt, pending=False)    # provider explicitly refused the ask
            raise
        except Exception as exc:
            raise vast_client.VastCreateUncertain('The quantization CREATE result is uncertain.') from exc
        if not _instance_id(instance_id):
            raise vast_client.VastCreateUncertain('The quantization CREATE returned no valid instance id.')
        _save_receipt(receipt, instance_id=instance_id, pending=False)
        return instance_id

    # Only explicit refusals can retry. The shared rental helper propagates
    # VastCreateUncertain, preserving this pending intent without another CREATE.
    return cloud_training.rent_with_fresh_offers(
        search=lambda: _search_offers(disk_gb), create=_create, pick=_pick,
        sleep=_sleep,
        no_offer_message=(
            f'no vast.ai machine with {disk_gb} GB of free disk matches the price '
            'cap right now — raise it in Settings ▸ Training and retry'))


def _drive(planned, token, *, _api=None, _sleep=time.sleep, _now=time.monotonic,
           _credential=None, _intent=None, _rental_lease=None):
    credential = _remember(_credential or vast_client.capture_credentials())
    lease = _rental_lease or _lease()
    receipt = _intent
    instance_id = None
    deadline = _now() + max_minutes() * 60
    try:
        receipt = receipt or _reserve(credential)
        planned = {**planned, 'result_file': f'_lds_fp8_result_{receipt["job_id"]}.json'}
        with vast_client.using_credentials(credential):
            env = {'HF_TOKEN': token or ''}
            instance_id, offer = _rent(planned, label=receipt['label'], env=env,
                                       _sleep=_sleep, credential=credential)
            planned = _repriced(planned, offer)
            _set('running', planned, instance_id=instance_id, label=receipt['label'],
                 started_at=time.time())
            api = _api or _hf_api(token)
            result = None
            while _now() < deadline:
                _sleep(_int_cfg('poll_seconds', DEFAULT_POLL_SECONDS))
                result = _read_result(api, planned['repo_id'], token=token, result_file=planned['result_file'])
                if result is not None:
                    break
            if result is None:
                raise CloudQuantizeError(
                    f'the machine reported nothing within {max_minutes()} minutes; '
                    'no result was confirmed and rental cleanup must be checked')
            if not result.get('ok') or not result.get('uploaded'):
                raise CloudQuantizeError(str(result.get('error')
                                             or 'the quantization failed on the pod'))
            _set('done', planned, instance_id=instance_id, result=result)
    except Exception as e:
        with vast_client.using_credentials(credential):
            message = _redact_error(e, token)
        logger.warning('cloud quantization failed: %s', message[:400])
        _set('error', planned, instance_id=instance_id, error=message[:400])
    finally:
        try:
            current = _receipt()
            if receipt and current and current['job_id'] == receipt['job_id']:
                _cleanup_owned(current, credential)
        except Exception:
            _cleanup_pending('Quantization cleanup is unconfirmed; the rental may still be billing. Restore its original credential and refresh status.')
        finally:
            lease.release()
        if receipt and 'result_file' in planned:
            try:
                _cleanup_result(_api or _hf_api(token), planned['repo_id'], result_file=planned['result_file'])
            except Exception:
                logger.debug('Could not remove this quantization result marker.')


def _read_result(api, repo_id, *, token=None, result_file=RESULT_FILE):
    """The pod's JSON verdict, or None while it has not been uploaded yet."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=result_file,
                               repo_type='model', token=token)
    except Exception:
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            parsed = json.load(fh)
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _cleanup_result(api, repo_id, *, result_file=RESULT_FILE):
    api.delete_file(path_in_repo=result_file, repo_id=repo_id, repo_type='model')


def reconcile_orphans() -> list:
    """Recover only this data directory's durable rental, on its original account."""
    try:
        lease = _lease()
    except Timeout:
        return []
    try:
        receipt = _receipt()
        if not receipt:
            if has_unreleased_rental():
                _cleanup_pending('A legacy quantization has no account-bound rental proof; review its machine manually before another rental.')
            return []
        if receipt['released']:
            return []
        credential = _credential_for(receipt)
        killed = _cleanup_owned(receipt, credential)
        current = status()
        if current.get('cleanup_pending'):
            queue_manager._set_system_state(_STATE_KEY, {**current, 'cleanup_pending': False}, ttl_seconds=None)
        return killed
    except Exception:
        try:
            _cleanup_pending('Quantization recovery needs its intact local proof and original credential; no new rental is allowed.')
        except Exception:
            logger.warning('Could not persist quantization recovery status.')
        return []
    finally:
        lease.release()


def _set(state, planned, **extra):
    queue_manager._set_system_state(_STATE_KEY, {
        'status': state,
        'repo_id': planned['repo_id'],
        'weight_name': planned['weight_name'],
        'output_name': planned['output_name'],
        'source_bytes': planned['source_bytes'],
        'output_bytes_typical': planned['output_bytes_typical'],
        'price_per_hour': planned['price_per_hour'],
        'estimated_cost': planned['estimated_cost'],
        # None until a machine is actually rented — the quote does not know
        # which box it will get (see _repriced).
        'gpu_name': planned.get('gpu_name'),
        'keep_bf16': planned['keep_bf16'],
        **extra,
    }, ttl_seconds=_STATE_TTL)
