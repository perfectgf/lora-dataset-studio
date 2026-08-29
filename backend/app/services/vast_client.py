"""Thin vast.ai REST client (no SDK dependency). All vast-specific HTTP lives
here so an API change touches one file. The API key is read from the secret
store on every call — never cached, so a key pasted in Settings applies
immediately."""
import logging
import re

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)

API_BASE = 'https://console.vast.ai/api/v0'
# Instance LISTING moved to v1 on 2026-07-12 (v0 answers 410 deprecated_endpoint).
# Everything else (bundles, asks, single instance, destroy) still lives on v0.
API_BASE_V1 = 'https://console.vast.ai/api/v1'
_TIMEOUT = 30

# Hard limits vast enforces on the ask itself, quoted from its own refusal:
# "error 400/3471: Invalid args: len(image) > 1024, or len(args) > 16384".
# Learned the expensive way — the cloud quantization lane embeds its whole
# program in `onstart`, exceeded this, and every launch was refused before a
# machine existed. Checked HERE, before the request, so an ask that cannot be
# accepted never becomes a round trip (and never a paid one).
MAX_IMAGE_CHARS = 1024
MAX_ONSTART_CHARS = 16384

# How much of vast's own answer travels inside an exception. Long enough for a
# sentence and a field name, short enough to stay a log line and to fit the
# error column of the runs table.
_ERROR_BODY_CHARS = 400

# Anything below is redacted before a response body is quoted anywhere. A vast
# error can echo the request it refused, and OUR requests carry secrets: the
# pod's HF_TOKEN and the training UI's bearer token both travel in `env`.
_REDACTED = '[redacted]'
# a JSON pair whose NAME says the value is a secret: "HF_TOKEN": "hf_…",
# "AI_TOOLKIT_AUTH": "…", "api_key": "…" — value replaced, name kept readable.
_SECRET_PAIR_RE = re.compile(
    r'("(?:[A-Za-z_]*(?:token|secret|key|auth|password)[A-Za-z_]*)"\s*:\s*")[^"]*(")',
    re.I)
# and the shapes worth killing wherever they appear, pair or not.
_SECRET_VALUE_RES = (
    re.compile(r'\bhf_[A-Za-z0-9_-]{8,}\b'),
    re.compile(r'\bbearer\s+\S+', re.I),
)


class VastError(RuntimeError):
    pass


class VastCommandUnsupported(VastError):
    """vast refused the COMMAND ITSELF, not the work it described.

    Its remote-exec endpoint is documented as constrained: vast-cli's own
    `execute` help lists the available commands as `ls`, `rm` and `du`, and
    answers anything else with HTTP 400 `invalid_args` / "Invalid command
    given." A caller that ships a PROGRAM therefore never gets a verdict here —
    not today's, not any.

    It has its own type because "the answer is no" and "there is no answer" are
    different facts and callers owe them different behaviour. A feature whose
    whole point is the remote program (the fp8 twin, the checkpoint assembly)
    has genuinely failed and must say so. A PRE-FLIGHT has merely lost its
    pre-flight, and refusing to launch over a check that cannot run anywhere
    would take the lane down for the life of the restriction."""


def _scrub(text: str) -> str:
    """Same text minus every secret shape we know how to send."""
    out = str(text or '')
    key = cfg.secret('VAST_API_KEY')
    if key:
        out = out.replace(key, _REDACTED)
    out = _SECRET_PAIR_RE.sub(r'\1' + _REDACTED + r'\2', out)
    for pattern in _SECRET_VALUE_RES:
        out = pattern.sub(_REDACTED, out)
    return out


def _detail(r) -> str:
    """What vast actually said, scrubbed and capped — never the empty ``{}``.

    Every failure sentence in this file used to end at ``HTTP 400 {}`` because
    the body was parsed only on 200: the first cloud quantization attempt died
    that way and the reason had to be reconstructed by hand. The body is the
    diagnosis, so it travels with the exception.
    """
    try:
        text = ' '.join(str(r.text or '').split())
    except Exception:          # a response object that cannot even be read
        return ''
    if not text:
        return ''
    text = _scrub(text)
    return text[:_ERROR_BODY_CHARS] + '…' if len(text) > _ERROR_BODY_CHARS else text


def _failed(r, what: str) -> VastError:
    return VastError(f'{what} failed: HTTP {r.status_code} {_detail(r)}'.rstrip())


def _request(method, path, *, base=API_BASE, **kwargs):
    key = cfg.secret('VAST_API_KEY')
    if not key:
        raise VastError('VAST_API_KEY is not configured')
    headers = {'Authorization': f'Bearer {key}', 'Accept': 'application/json'}
    try:
        return requests.request(method, f'{base}{path}', headers=headers,
                                timeout=_TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise VastError(f'vast.ai request failed: {e}') from e


def search_offers(min_vram_gb: int, max_dph: float, limit: int = 20,
                  min_inet_down_mbps: int = 0, min_reliability: float = 0.95,
                  min_disk_bw_mbps: int = 0, verified_only: bool = True,
                  secure_cloud_only: bool = False, min_disk_gb: int = 0,
                  min_compute_cap: int = 0) -> list:
    """Offers matching the configured trust tier and resource constraints.

    Vast calls its normal host trust flag ``verified`` and exposes Secure
    Cloud as the ``datacenter`` search field.  Omitting either predicate means
    "any tier"; it does not mean the inverse tier.  gpu_ram is expressed in MB
    on the vast side. inet_down
    (Mbps) filters out hosts whose registry pull of the ~7 GB image would eat
    the whole boot budget (observed live: a retry-looping host on 2026-07-12);
    disk_bw (MB/s) filters out hosts too slow to EXTRACT it (a 5090 host froze
    in 'loading' on 2026-07-13 with network fine — the disk was the bound).

    min_disk_gb is the one that decides whether the rental happens AT ALL: an
    ask whose ``disk`` exceeds what the offer has free is refused outright, and
    the cheapest offer on the market is exactly where free disk runs out — a
    live search on 2026-08-04 returned a $0.081/h box with 57 GB against 19
    others averaging 500+, and "cheapest" is what the quantization lane picked.
    Callers MUST pass the same number they will send as ``disk``; filtering for
    less than you ask for is the failure this parameter exists to remove.

    min_compute_cap is the same idea applied to the GPU itself. vast reports
    each offer's compute capability as an integer (750 Turing, 800/860 Ampere,
    900 Hopper, 1200 Blackwell), and a recipe that trains in bf16 needs 800 or
    better — bf16 is not a speed on Turing, it is absent. Left at 0 the
    predicate is not sent at all, so no lane inherits a floor it did not
    choose."""
    body = {
        'gpu_ram': {'gte': int(min_vram_gb) * 1024},
        'reliability': {'gte': float(min_reliability)},
        'rentable': {'eq': True},
        'dph_total': {'lte': float(max_dph)},
        'num_gpus': {'eq': 1},
        'type': 'ondemand',
        'limit': int(limit),
    }
    if verified_only:
        body['verified'] = {'eq': True}
    if secure_cloud_only:
        body['datacenter'] = {'eq': True}
    if min_inet_down_mbps:
        body['inet_down'] = {'gte': int(min_inet_down_mbps)}
    if min_disk_bw_mbps:
        body['disk_bw'] = {'gte': int(min_disk_bw_mbps)}
    if min_disk_gb:
        body['disk_space'] = {'gte': int(min_disk_gb)}
    if min_compute_cap:
        body['compute_cap'] = {'gte': int(min_compute_cap)}
    r = _request('POST', '/bundles/', json=body)
    if r.status_code != 200:
        raise _failed(r, 'offer search')
    offers = (r.json() or {}).get('offers') or []
    if min_disk_gb:
        # Belt and braces: the predicate above is honoured server-side (verified
        # live), but a silently-ignored filter would hand back exactly the
        # unrentable offers we are trying to avoid. An offer that does not
        # publish its free disk is kept — unknown is not "too small".
        offers = [o for o in offers
                  if not o.get('disk_space') or float(o['disk_space']) >= int(min_disk_gb)]
    out = [{
        'offer_id': o.get('id'),
        'gpu_name': o.get('gpu_name'),
        'dph_total': o.get('dph_total'),
        'gpu_ram_gb': round((o.get('gpu_ram') or 0) / 1024.0, 1),
        # Free disk (GB) and advertised downlink (Mbps): what the rental asks
        # for and what the duration estimate is built on. Both are documented
        # offer fields; both are treated as optional by every consumer.
        'disk_space_gb': round(float(o.get('disk_space') or 0), 1),
        'inet_down': o.get('inet_down'),
        # host identity + quality signals for the selection layer (blacklist
        # of hosts that failed to boot, reliability preference within a class).
        # machine_id alone was not enough: it lives in a file on the host
        # (/var/lib/vastai_kaalia/machine_id) and a daemon reinstall mints a new
        # one, so the SAME physical box comes back under a new id — measured on
        # 2026-07-28, where a blacklisted machine returned three minutes later
        # as a different machine_id at the same public address. The address and
        # the owning account are carried through so the selection layer can
        # recognise it. Documented on the offer object, but not guaranteed to be
        # populated for every offer: every consumer treats None as "unknown".
        'machine_id': o.get('machine_id'),
        'host_id': o.get('host_id'),
        'public_ipaddr': o.get('public_ipaddr'),
        'reliability': o.get('reliability2') or o.get('reliability'),
    } for o in offers if o.get('id') is not None]
    out.sort(key=lambda x: x['dph_total'] if x['dph_total'] is not None else 9e9)
    return out


def create_instance(offer_id, disk_gb: int, label: str, template_hash: str | None = None,
                    image: str | None = None, env: dict | None = None,
                    onstart: str | None = None) -> str:
    """Rent the offer. Preferred path: template_hash — the instance inherits the
    official template's env/ports/entrypoint (the raw-image path never published
    the UI port; smoke-tested 2026-07-12). image/env/onstart remain as a
    config-escape-hatch fallback when no template hash is set.

    The two branches are not interchangeable: env and onstart are DROPPED on the
    template branch (the template owns them, and vast refuses an env override
    there with a 400). A caller that drives its pod entirely through onstart —
    the cloud quantization lane, which needs no inbound port at all — therefore
    has to take the raw-image branch, and a raw-image ask carrying env+onstart
    was confirmed accepted (HTTP 200, contract created and immediately
    destroyed) on 2026-08-04.

    ``disk_gb`` must be backed by an offer that HAS that much free disk: vast
    refuses the ask otherwise. See search_offers(min_disk_gb=…)."""
    if image and len(str(image)) > MAX_IMAGE_CHARS:
        raise VastError(f'image reference is {len(str(image))} characters; vast '
                        f'refuses more than {MAX_IMAGE_CHARS} — nothing was sent')
    if onstart and len(onstart) > MAX_ONSTART_CHARS:
        # Caught before the request: vast answers this with a plain 400, which
        # for months read as "the offer is gone" and cost two real launches.
        raise VastError(
            f'onstart script is {len(onstart)} characters; vast refuses more than '
            f'{MAX_ONSTART_CHARS} ("Invalid args") — nothing was sent, so no '
            'machine was rented. Shrink what the script embeds.')
    if template_hash:
        body = {'template_hash_id': template_hash, 'label': label, 'disk': int(disk_gb)}
        # The official template pins an OLD image tag (2026-05-20 — predates the
        # krea2 arch: run #5 died on 'StableDiffusionPipeline expected [...]').
        # vast merges body params over template defaults, so overriding just the
        # image keeps the template's env/ports/entrypoint with current code.
        if image:
            body['image'] = image
    else:
        body = {'image': image, 'label': label, 'disk': int(disk_gb),
                'runtype': 'args', 'env': dict(env or {})}
        if onstart:
            body['onstart'] = onstart
    r = _request('PUT', f'/asks/{offer_id}/', json=body)
    if r.status_code != 200:
        raise _failed(r, 'create_instance')
    try:
        data = r.json() or {}
    except ValueError:
        data = {}
    if not data.get('success'):
        raise _failed(r, 'create_instance')
    return str(data.get('new_contract'))


def execute_command(instance_id, command: str) -> str:
    """Run ONE shell command inside a running instance; return its result URL.

    vast's ``PUT /instances/command/{id}/`` is asynchronous by design: it queues
    the command on the host daemon and answers with a URL where the combined
    output will appear once the command finishes. Nothing about the response
    says the command succeeded — only ``fetch_command_result`` can.

    It CANNOT be grown into a general remote shell, and not by choice: vast
    documents this endpoint as constrained to `ls`, `rm` and `du` (vast-cli
    `execute`, "available commands"), and refuses anything else with
    `invalid_args` — measured on a rented pod, run #165, against a `python -c`
    that had never been run against a live one. Callers that ship a program get
    `VastCommandUnsupported`, which says the capability is absent rather than
    that their work failed.
    """
    if not str(command or '').strip():
        raise VastError('execute_command needs a command')
    r = _request('PUT', f'/instances/command/{instance_id}/',
                 json={'command': command})
    try:
        data = r.json() or {}
    except ValueError:
        data = {}
    if r.status_code == 400 and str(data.get('error') or '') == 'invalid_args':
        # Read from the provider's OWN error code rather than matched against
        # its sentence: the code is the contract, the wording is not.
        raise VastCommandUnsupported(
            'vast will not run this command: its remote-exec endpoint accepts '
            'ls, rm and du only, so a program cannot be run on the pod '
            f'({_scrub(str(data.get("msg") or ""))[:120]})')
    url = str(data.get('result_url') or '') if r.status_code == 200 else ''
    if r.status_code != 200 or not data.get('success') or not url:
        raise _failed(r, 'execute_command')
    return url


def fetch_command_result(result_url: str):
    """The command's output, or None while it has not landed yet.

    The result object is written by the host when the command ends, so a 404 is
    the normal "still running" answer and must not read as a failure.
    """
    try:
        r = requests.get(result_url, timeout=_TIMEOUT)
    except requests.RequestException as e:
        raise VastError(f'vast.ai result fetch failed: {e}') from e
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise _failed(r, 'result fetch')
    return r.text


def _normalize(i: dict) -> dict:
    return {
        'instance_id': str(i.get('id')),
        'actual_status': i.get('actual_status'),
        'public_ipaddr': i.get('public_ipaddr'),
        'ports': i.get('ports'),
        'label': i.get('label'),
        'dph_total': i.get('dph_total'),
        # Free-text progress line the host daemon publishes while the pod boots
        # (image pull / extraction). Optional and UNSPECIFIED in wording — the
        # boot watchdog only ever uses it as "did this string change", so an
        # absent or exotic field costs nothing.
        'status_msg': i.get('status_msg'),
        # per-instance auth token generated by vast — the template's Caddy
        # proxy accepts it as `Authorization: Bearer <jupyter_token>`
        'jupyter_token': i.get('jupyter_token'),
        # The image the pod is REALLY running. Worth carrying even though we
        # asked for one: the default launch path is a vast.ai TEMPLATE published
        # by a third party, so what we asked for and what booted are two
        # different facts, and only this one is the trainer that produced the
        # weights. Optional — an absent field simply records nothing.
        'image_uuid': i.get('image_uuid'),
    }


def list_instances() -> list:
    r = _request('GET', '/instances/', base=API_BASE_V1)
    if r.status_code != 200:
        raise _failed(r, 'list_instances')
    return [_normalize(i) for i in (r.json() or {}).get('instances') or []]


def get_instance(instance_id):
    """Single-instance lookup (v0 show endpoint; body is {'instances': {...}}).
    Falls back to the list scan if the shape ever changes."""
    r = _request('GET', f'/instances/{instance_id}/')
    if r.status_code == 200:
        one = (r.json() or {}).get('instances')
        if isinstance(one, dict) and one.get('id') is not None:
            return _normalize(one)
        if isinstance(one, list):
            for i in one:
                if str(i.get('id')) == str(instance_id):
                    return _normalize(i)
            return None
        if one is None:
            return None            # instance gone (vast answers 200 + null)
    for inst in list_instances():
        if inst['instance_id'] == str(instance_id):
            return inst
    return None


def destroy_instance(instance_id) -> bool:
    """Idempotent: a 404 means the instance is already gone — success.
    Network failure -> False (callers log and retry via reconciliation)."""
    try:
        r = _request('DELETE', f'/instances/{instance_id}/')
    except VastError as e:
        logger.warning('destroy_instance %s: %s', instance_id, e)
        return False
    if r.status_code in (200, 404):
        return True
    logger.warning('destroy_instance %s: HTTP %s %s', instance_id,
                   r.status_code, _detail(r))
    return False


def derive_base_url(instance: dict, container_port: int):
    """Public URL of the pod's UI from the docker-style port mapping.
    Returns None while the mapping isn't published yet (instance booting)."""
    if not instance:
        return None
    ip = instance.get('public_ipaddr')
    ports = instance.get('ports') or {}
    entries = ports.get(f'{container_port}/tcp') or []
    if not ip or not entries:
        return None
    host_port = (entries[0] or {}).get('HostPort')
    return f'http://{ip}:{host_port}' if host_port else None
