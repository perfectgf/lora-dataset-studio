"""Fail-closed ownership fence for LDS-managed local Ollama models.

Ollama is a separate process: a successful ``keep_alive: 0`` request is not a
proof that its CUDA allocation has already disappeared.  LDS therefore records
every local model it asks to run, then verifies ``/api/ps`` is empty before a
ComfyUI workflow may load its own models.  Remote Ollama endpoints do not share
this machine's GPU and are deliberately never probed or unloaded here.
"""
from __future__ import annotations

import errno
import ipaddress
import json
import logging
import os
import threading
import time
from datetime import datetime
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

_lock = threading.RLock()
# Normalised local endpoint -> exact models LDS has admitted in this process.
# ComfyUI probes before every new prompt: a manual Ollama load between cells
# must block rather than overlap with ComfyUI.
_owned_models: dict[str, set[str]] = {}
# Which driver speaks to each endpoint. `_owned_models` is keyed by ENDPOINT and
# ensure_released_for_comfy() walks every key, so after a provider switch this map
# holds both the old Ollama endpoint (with residue from before the switch) and the
# new LM Studio one. Choosing the driver from the GLOBAL setting would then send
# one of them an unload in the other's wire format — a release that answers 200
# and frees nothing, which the fence would read as success. So the driver is
# resolved per endpoint, recorded at admission.
_endpoint_driver: dict[str, str] = {}
# Models LDS is allowed to USE but never to UNLOAD. The two rights were one set,
# and conflating them handed LDS the right to unload a model the user had loaded
# by hand — the one rule this module's own docstring says never moves ("Over-
# adopting is the dangerous direction"). Under LM Studio the user loads the model
# themselves, deliberately, so using it is right and unloading it is not ours to
# decide. A model only ever enters `_owned_models` when LDS saw the endpoint EMPTY
# first, which is the proof that it put it there.
_borrowed_models: dict[str, set[str]] = {}
# Endpoint seen with a pre-existing / otherwise unowned model. It may be a
# user's own Ollama session, so it is never automatically unloaded.
_foreign_local_endpoints: set[str] = set()

# Ownership that survives a restart. `_owned_models` is process memory, and a
# keep-warm lease outlives the process that took it: restarting LDS while its
# OWN model was still resident made LDS read that model as a stranger's and
# fence itself out until the lease expired. The claims below are the missing
# half — written when a model is admitted with a keep-alive, and re-adopted at
# boot only when the runner's own `expires_at` still matches the claim.
#
# Adoption is deliberately hard to earn. Over-adopting is the dangerous
# direction: it would let LDS unload a model another app loaded. So a claim is
# honoured only while it is fresh, and only when the residency Ollama reports
# could NOT be a later load by someone else.
_CLAIM_SLACK_S = 30.0        # request round-trip between our call and expires_at
_CLAIM_MAX_AGE_S = 3600.0    # a claim never speaks for a runner an hour later

# Probe states that prove no model is on this GPU: a daemon that answered with
# an empty runner, and a port nothing is listening on.
_RUNNER_HOLDS_NOTHING = ('empty', 'down')
_claims_loaded = False

# Why the last ComfyUI hand-off was refused. The queue dock reads this through
# job_queue.gpu_hold(), so a fence refusal becomes a NAMED hold with a remedy
# instead of a silently frozen queue — it used to warn once per worker cycle
# (every second) while the dock showed nothing at all. Refreshed by every
# refused hand-off (the worker retries each second while jobs wait), cleared by
# the next successful release, and read with a freshness bound so a stale
# refusal never explains a queue that is no longer held.
_BLOCK_LOG_INTERVAL_S = 60.0
_BLOCK_FRESH_S = 15.0
_last_block: dict | None = None
_last_block_log_at = 0.0
# What _release_endpoint last refused on, for ensure_released_for_comfy to
# publish. A refusal during a caption-batch unload sets it too, but only the
# ComfyUI hand-off path promotes it to _last_block — the dock must never blame
# ComfyUI's queue for a captioning cleanup.
_last_release_failure: dict | None = None
# details.family per endpoint from the last parsed /api/ps. A side channel on
# purpose: _probe's (state, names, expiry) shape is stubbed as-is by conftest
# and the fence tests, so it does not change. KoboldCPP self-identifies here
# ('family': 'koboldcpp'), and naming it in the dock is what turns "the queue
# is frozen" into "the wrong tool sits in the Ollama slot".
_probe_families: dict[str, set] = {}

# A refusal must not be able to hold the queue open-endedly with no answer.
# The fence's own remedy — "unload it there" — is not always available: the
# resident model may be another tool's live work, or another LDS instance in
# the middle of a caption batch, and neither is the user's to evict. So a held
# queue needs a SECOND door, and this is it: share the card knowingly.
#
# Nothing below evicts a foreign model. The absolute rule does not move. What
# moves is that a refusal becomes a BOUNDED, answerable state instead of an
# open wait — which is the whole difference between a guard and a wall.
_BLOCK_CONTINUITY_S = 15.0   # a gap longer than this starts a new block episode
# How long a block must have stood before the surfaces offer to share. A block
# that clears on its own in a few seconds (Ollama's idle unload, the other app
# closing) must never tempt anyone into sharing a card they did not have to.
OFFER_SHARE_AFTER_S = 60.0
# Long enough for a batch someone is watching, short enough that a forgotten
# click cannot quietly disable the fence for the rest of the day.
SHARE_SECONDS = 900.0
_share_until = 0.0
_share_endpoint: str | None = None


def _exception_chain(exc):
    seen, pending = set(), [exc]
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ('__cause__', '__context__', 'reason'):
            nested = getattr(current, attr, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
        pending.extend(arg for arg in getattr(current, 'args', ())
                       if isinstance(arg, BaseException))


def _connection_refused(exc) -> bool:
    return any(
        isinstance(item, ConnectionRefusedError)
        or (isinstance(item, OSError) and item.errno in (errno.ECONNREFUSED, 10061))
        for item in _exception_chain(exc)
    )


def _endpoint_scope(url) -> tuple[str, str | None]:
    """Return ``local``/``remote``/``unknown`` and a stable local endpoint."""
    if not isinstance(url, str) or not url.strip():
        return 'unknown', None
    try:
        parsed = urlsplit(url.strip())
    except Exception:
        return 'unknown', None
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        return 'unknown', None
    # The vision client preserves a configured URL path while this fence speaks
    # directly to /api/ps and /api/generate. Treat a proxy base-path as unknown
    # rather than proving the wrong daemon empty.
    if parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        return 'unknown', None
    host = parsed.hostname.lower().rstrip('.')
    if host == 'localhost':
        local = True
    else:
        try:
            # ``127.evil.example`` is a hostname, not loopback. Never use a
            # string prefix here: this fence must never probe or unload a
            # remote Ollama endpoint merely because its DNS name looks local.
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if not local:
        return 'remote', None
    try:
        port = parsed.port
    except ValueError:
        return 'unknown', None
    authority_host = f'[{host}]' if ':' in host else host
    authority = authority_host if port is None else f'{authority_host}:{port}'
    return 'local', f'{parsed.scheme}://{authority}'


def _keep_alive_seconds(value) -> float | None:
    """Seconds a ``keep_alive`` asks Ollama to hold the model, or ``None``.

    ``None`` means "no bounded claim to record": either the value unloads the
    model straight away (0) or it is a form this fence will not guess at. A
    claim we cannot bound is a claim we refuse to write, which costs one cold
    reload after a restart and never costs someone else's model.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if seconds > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    units = {'s': 1.0, 'm': 60.0, 'h': 3600.0}
    factor = units.get(text[-1:])
    if factor is not None:
        text = text[:-1]
    else:
        factor = 1.0
    try:
        seconds = float(text) * factor
    except ValueError:
        return None
    # A negative keep_alive means "resident until something unloads it". There
    # is no deadline to compare an expires_at against, so no claim is written.
    return seconds if seconds > 0 else None


def _parse_expires_at(value) -> float | None:
    """RFC3339 ``expires_at`` from /api/ps as an epoch float, or ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith('Z'):
        text = f'{text[:-1]}+00:00'
    # Ollama emits nanosecond precision; datetime accepts at most microseconds.
    if '.' in text:
        head, _, tail = text.partition('.')
        digits = ''
        while tail and tail[0].isdigit():
            digits, tail = digits + tail[0], tail[1:]
        text = f'{head}.{digits[:6]}{tail}' if digits else head + tail
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.timestamp()
    return parsed.timestamp()


def _claims_path():
    try:
        from ..config import data_dir
        return data_dir() / 'ollama_fence_claims.json'
    except Exception:
        return None


def _read_claims() -> dict:
    path = _claims_path()
    if path is None:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    claims = data.get('claims')
    return claims if isinstance(claims, dict) else {}


def _write_claims(claims: dict) -> None:
    path = _claims_path()
    if path is None:
        return
    payload = {'version': 1, 'claims': claims}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        os.replace(temporary, path)
    except OSError as exc:
        # Ownership that does not survive a restart is exactly today's
        # behaviour, so a read-only data dir degrades instead of failing.
        logger.debug('ollama GPU fence: could not persist ownership claim (%s)', exc)


def _record_claim(endpoint, model, keep_alive) -> None:
    seconds = _keep_alive_seconds(keep_alive)
    if seconds is None:
        return
    with _lock:
        claims = _read_claims()
        endpoint_claims = claims.get(endpoint)
        if not isinstance(endpoint_claims, dict):
            endpoint_claims = {}
        endpoint_claims[model] = {'deadline': time.time() + seconds}
        claims[endpoint] = endpoint_claims
        _write_claims(claims)


def _drop_claims(endpoint) -> None:
    with _lock:
        claims = _read_claims()
        if claims.pop(endpoint, None) is not None:
            _write_claims(claims)


def _adopt_persisted(endpoint, loaded, expiry) -> set[str]:
    """Models on ``endpoint`` a persisted claim still proves are LDS's own.

    Two independent facts must agree: the claim has not expired (so LDS asked
    for this residency recently enough that it can still be the one on screen),
    and the residency Ollama reports ends no later than the claim does. A model
    another app loaded after our restart carries its own, later ``expires_at``
    and is therefore never adopted.
    """
    if not loaded:
        return set()
    claims = _read_claims().get(endpoint)
    if not isinstance(claims, dict):
        return set()
    now = time.time()
    adopted = set()
    for model in loaded:
        entry = claims.get(model)
        if not isinstance(entry, dict):
            continue
        deadline = entry.get('deadline')
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
            continue
        horizon = float(deadline) + _CLAIM_SLACK_S
        if now > horizon or now > float(deadline) + _CLAIM_MAX_AGE_S:
            continue
        observed = expiry.get(model)
        if observed is not None and observed > horizon:
            continue  # a later load: someone else owns this residency now
        adopted.add(model)
    if adopted:
        logger.info('ollama GPU fence: re-adopted %d model(s) LDS loaded before it restarted',
                    len(adopted))
    return adopted


def mark_before_generate(url, model, keep_alive=None, provider=None) -> str:
    """Return ``local``, ``remote`` or ``blocked`` before a local LLM request.

    A shared Ollama daemon exposes no client owner id. LDS probes before its
    local admission: if another model is already resident, LDS must not
    alter its ``keep_alive`` or load a competing model. The request is blocked
    until that runner is empty (or a dedicated LDS endpoint is configured).
    """
    scope, endpoint = _endpoint_scope(url)
    if scope == 'unknown':
        return 'blocked'
    if scope != 'local':
        return scope
    if not isinstance(model, str) or not model.strip():
        return 'blocked'

    model = model.strip()

    # Pin the driver to the endpoint at admission. Doing it here rather than at
    # release time is what makes a mid-session provider switch safe: the endpoint
    # that already holds an LDS model keeps the wire format it was admitted with,
    # even after the global setting points somewhere else.
    _remember_driver(endpoint, provider or _driver_for(endpoint))

    state, loaded, expiry = _probe(endpoint)
    if state == 'down':
        # Nothing is listening, so there is no residency to fence — and none to
        # own either. The request goes through (it will either fail on its own
        # connection error or start the daemon and be admitted for real on the
        # retry), but NO claim is written: a keep-warm lease is a statement about
        # a model that is loaded. Writing one here for a model that never loaded
        # made LDS adopt — and unload — a same-named model the user started by
        # hand minutes later.
        return 'local'
    with _lock:
        # Re-read process ownership after the probe: concurrent LDS requests may
        # have claimed another model, but they never make an external model safe
        # to change or unload.
        owned = set(_owned_models.get(endpoint, set()))
        if state == 'unknown':
            return 'blocked'
        # LM Studio inverts what a resident model MEANS, and reading it the Ollama
        # way makes the provider unusable. Ollama loads on demand, so a model LDS
        # did not load is evidence of another app competing for the card. LM Studio
        # has no JIT loading by default: the user loads the model in its app, on
        # purpose, precisely so something can use it — and that something is this.
        # Refusing it as a stranger blocked EVERY call, which is how this was found:
        # the first real Enhance against a live LM Studio came back 409.
        # The fence's other half is untouched — ensure_released_for_comfy() still
        # has to prove the card is free before a ComfyUI hand-off, and now it can
        # actually free it, because LM Studio's unload really releases the VRAM.
        borrowed = _driver_for(endpoint) == 'lmstudio' and model in loaded
        if borrowed:
            _borrowed_models.setdefault(endpoint, set()).add(model)
        if state != 'empty' and not loaded.issubset(owned):
            # Before calling a resident model a stranger's, ask the claims LDS
            # wrote down: after a restart this is its own keep-warm lease.
            owned |= _adopt_persisted(endpoint, loaded - owned, expiry)
            if loaded.issubset(owned):
                _owned_models.setdefault(endpoint, set()).update(owned)
        if state == 'empty' or loaded.issubset(owned):
            _owned_models.setdefault(endpoint, set()).add(model)
            _foreign_local_endpoints.discard(endpoint)
            claim = True
        elif borrowed:
            # Borrowing costs the card nothing: LDS loads no second model, it uses
            # one that is already there. So a CO-TENANT is not a reason to refuse —
            # and LM Studio routinely holds one (an embedding model for its own
            # document chat, a chat model beside the VLM). Reading that the Ollama
            # way blocked every Describe and Enhance AND froze the ComfyUI queue,
            # telling the user to unload a model they legitimately need.
            # The endpoint is deliberately NOT marked foreign here: that flag is
            # about the RELEASE path, which stays strict below.
            claim = True
        else:
            _foreign_local_endpoints.add(endpoint)
            claim = False
    if claim:
        # A borrowed model gets no claim: a claim is what lets a restarted LDS
        # re-adopt — and eventually unload — a model as its own. Writing one for
        # somebody else's residency is exactly the over-adoption above.
        if not borrowed:
            _record_claim(endpoint, model, keep_alive)
        return 'local'
    logger.info('ollama GPU fence: preserving a pre-existing local model; LDS inference is blocked')
    return 'blocked'


def _driver_for(endpoint: str) -> str:
    """'ollama' | 'lmstudio' for this endpoint, without guessing on the wire.

    Recorded at admission wins. Otherwise the endpoint is matched against the
    ACTIVE provider's configured URL, and anything still unmatched is Ollama —
    every endpoint this map ever held before there was a second provider was one.
    """
    with _lock:
        known = _endpoint_driver.get(endpoint)
    if known:
        return known
    try:
        from . import vision_llm
        if vision_llm.provider() == vision_llm.LMSTUDIO:
            from . import vision_lmstudio
            # Compare what _endpoint_scope produced against the SAME normalisation.
            # It lower-cases the host, so an oddly-cased URL (Localhost:1234) failed
            # to match its own configured value and the fence spoke Ollama's API to
            # LM Studio — an unload that answers 200 and frees nothing.
            _, mine = _endpoint_scope(vision_lmstudio.base_url())
            if mine and endpoint.rstrip('/') == str(mine).rstrip('/'):
                return 'lmstudio'
    except Exception:                      # noqa: BLE001 - fall back to the old world
        pass
    return 'ollama'


def _remember_driver(endpoint: str, name: str) -> None:
    with _lock:
        _endpoint_driver[endpoint] = name


def _probe(endpoint):
    """Dispatch to the driver that speaks this endpoint's API."""
    if _driver_for(endpoint) == 'lmstudio':
        return _probe_lmstudio(endpoint)
    return _probe_ollama(endpoint)


def _probe_lmstudio(endpoint):
    """LM Studio residency, mapped onto the fence's own four states.

    `unknown` is carried through faithfully rather than flattened into `empty`:
    a server answering only the OpenAI-compatible surface reports no residency at
    all, and reading "cannot tell" as "nothing is loaded" is how a fail-closed
    guard hands ComfyUI a card somebody else is holding.

    LM Studio has no per-model expiry (no TTL by default), so the expiry map is
    empty — which the claim logic already tolerates: a missing value leaves a
    claim to be judged on its own freshness.
    """
    try:
        from . import vision_lmstudio
        state, names, _meta = vision_lmstudio.probe_resident(endpoint)
    except Exception:                      # noqa: BLE001 - same contract as below
        return 'unknown', set(), {}
    with _lock:
        _probe_families.pop(endpoint, None)
    return state, set(names), {}


def _probe_ollama(endpoint):
    """Return (``empty`` | ``down`` | ``models`` | ``unknown``, names, expires_at map).

    ``down`` is "nothing is listening on this port" (the connection was refused),
    as opposed to ``empty`` — "a daemon answered and holds no model". Both mean
    the GPU is free, so every release path treats them alike; they part ways on
    the admission path, where ``empty`` describes a runner that can hold what LDS
    is about to load and ``down`` describes one that cannot hold anything at all.

    ``expires_at`` is epoch seconds when Ollama reported a parseable one. It is
    only ever used to REFUSE an ownership claim, so a missing value is not a
    failure — it simply leaves the claim to be judged on its own freshness.
    """
    try:
        response = requests.get(f'{endpoint}/api/ps', timeout=(3, 5), allow_redirects=False)
        status = getattr(response, 'status_code', None)
        if type(status) is not int or not 200 <= status < 300:
            return 'unknown', set(), {}
        data = response.json()
        models = data.get('models') if isinstance(data, dict) else None
        if not isinstance(models, list):
            return 'unknown', set(), {}
        names, expiry, families = set(), {}, set()
        for item in models:
            if not isinstance(item, dict):
                return 'unknown', set(), {}
            name = item.get('name') or item.get('model')
            if not isinstance(name, str) or not name.strip():
                return 'unknown', set(), {}
            name = name.strip()
            names.add(name)
            parsed = _parse_expires_at(item.get('expires_at'))
            if parsed is not None:
                expiry[name] = parsed
            details = item.get('details')
            family = details.get('family') if isinstance(details, dict) else None
            if isinstance(family, str) and family.strip():
                families.add(family.strip().lower())
        with _lock:
            if names:
                _probe_families[endpoint] = families
            else:
                _probe_families.pop(endpoint, None)
        return ('empty' if not names else 'models'), names, expiry
    except (requests.RequestException, OSError) as exc:
        if _connection_refused(exc):
            return 'down', set(), {}
        return 'unknown', set(), {}
    except Exception:
        return 'unknown', set(), {}


def _post_unload(endpoint, model) -> bool:
    if _driver_for(endpoint) == 'lmstudio':
        try:
            from . import vision_lmstudio
            # Measured: this genuinely frees the VRAM, where Ollama can only be
            # asked to stop keeping the model warm.
            return vision_lmstudio.release(endpoint, model)
        except Exception as exc:           # noqa: BLE001 - reported like the peer below
            logger.warning('LM Studio fence: unload request failed for %s (%s)', model, exc)
            return False
    try:
        response = requests.post(f'{endpoint}/api/generate',
                                 json={'model': model, 'keep_alive': 0},
                                 timeout=(10, 30), allow_redirects=False)
        status = getattr(response, 'status_code', None)
        return type(status) is int and 200 <= status < 300
    except Exception as exc:
        logger.warning('ollama GPU fence: unload request failed for %s (%s)', model, exc)
        return False


def _refused(endpoint, reason, models, message) -> bool:
    """Record why a release was refused and warn about it at most once a minute.

    The worker retries a held queue every second; the old per-attempt warning
    was 60 lines a minute of the same sentence (and the 'unknown' path said
    nothing at all). The recorded failure is what ensure_released_for_comfy
    publishes to the queue dock. Always returns False so refusal sites can
    `return _refused(...)`."""
    global _last_release_failure, _last_block_log_at
    with _lock:
        _last_release_failure = {'reason': reason, 'endpoint': endpoint,
                                 'models': sorted(models)}
    now = time.monotonic()
    if now - _last_block_log_at >= _BLOCK_LOG_INTERVAL_S:
        _last_block_log_at = now
        logger.warning('ollama GPU fence: %s', message)
    return False


def _release_endpoint(endpoint, expected_models) -> bool:
    """Unload only LDS-owned models and prove the runner is empty afterwards."""
    state, loaded, expiry = _probe(endpoint)
    if state == 'unknown':
        return _refused(endpoint, 'unreachable', (),
                        f'{endpoint} does not answer /api/ps the way an Ollama '
                        'daemon does; ComfyUI stays blocked')
    if state in _RUNNER_HOLDS_NOTHING:
        # Empty, or not running at all: either way this GPU is free, and any
        # claim left over from a previous life stops speaking for it.
        with _lock:
            _owned_models.pop(endpoint, None)
            _foreign_local_endpoints.discard(endpoint)
        _drop_claims(endpoint)
        return True

    with _lock:
        foreign = endpoint in _foreign_local_endpoints
        if not foreign and loaded - expected_models:
            # Same restart case as the admission path: a model this process has
            # no memory of may still be one it loaded a moment ago.
            expected_models = set(expected_models) | _adopt_persisted(
                endpoint, loaded - expected_models, expiry)
    if foreign:
        # A later /api/ps empty response is the only way this endpoint becomes
        # safe again. Never infer that a same-named resident model is still ours.
        return _refused(endpoint, 'foreign', loaded,
                        'preserving a pre-existing local model; ComfyUI stays blocked')

    with _lock:
        borrowed = set(_borrowed_models.get(endpoint, set()))
    still_borrowed = loaded & borrowed
    if still_borrowed:
        # LDS may use these, never unload them: the user loaded them. Refusing the
        # hand-off is the honest answer, and the queue dock already offers the two
        # consent-gated doors for it (unload it for me / share the GPU anyway).
        return _refused(endpoint, 'foreign', still_borrowed,
                        'a model loaded outside LDS is holding the card; ComfyUI stays blocked')

    unknown = loaded - expected_models
    if unknown:
        with _lock:
            _foreign_local_endpoints.add(endpoint)
        return _refused(endpoint, 'foreign', loaded,
                        'local runner has an unowned model; ComfyUI stays blocked')
    for model in loaded:
        if not _post_unload(endpoint, model):
            return _refused(endpoint, 'stuck', (model,),
                            f'{model} did not accept the unload request; ComfyUI stays blocked')
    state, remaining, _ = _probe(endpoint)
    if state not in _RUNNER_HOLDS_NOTHING or remaining:
        return _refused(endpoint, 'stuck', remaining,
                        'runner still holds a model after the unload; ComfyUI stays blocked')
    with _lock:
        _owned_models.pop(endpoint, None)
        _borrowed_models.pop(endpoint, None)
        _foreign_local_endpoints.discard(endpoint)
    _drop_claims(endpoint)
    return True


def release_owned_models(*, ollama_url=None, model=None) -> bool | None:
    """Release all tracked local LDS models, or ``None`` for a remote endpoint.

    ``model`` is intentionally only a filter for ownership registration, not an
    instruction to unload an arbitrary user model. Bare ``unload_vision_model``
    calls release every tracked custom model from a caption batch.
    """
    scope, endpoint = _endpoint_scope(ollama_url) if ollama_url is not None else ('all', None)
    if scope == 'remote':
        return None
    with _lock:
        foreign = set(_foreign_local_endpoints)
        if scope == 'local':
            candidates = {endpoint: set(_owned_models.get(endpoint, set()))}
            if endpoint in foreign:
                candidates.setdefault(endpoint, set())
        elif scope == 'all':
            candidates = {key: set(value) for key, value in _owned_models.items()}
            for foreign_endpoint in foreign:
                candidates.setdefault(foreign_endpoint, set())
        else:
            return False
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            return False
        wanted = model.strip()
        candidates = {key: values for key, values in candidates.items()
                      if wanted in values}

    # No local model has been admitted by LDS and no endpoint is known foreign:
    # do not poke or unload a user's manually loaded Ollama model.
    if not candidates:
        return True
    for local_endpoint, expected in candidates.items():
        if not _release_endpoint(local_endpoint, expected):
            return False
    return True


def _configured_local_endpoint() -> tuple[str, str | None]:
    """Resolve the current local runner without importing Vision request code."""
    try:
        # The ACTIVE provider's endpoint, not Ollama's by definition: this feeds
        # ensure_released_for_comfy(), fence_status() and unload_foreign_models(),
        # and reading ollama.url under an LM Studio install would have the fence
        # guarding a daemon nobody uses while ignoring the one holding the card.
        from . import vision_llm
        url = vision_llm.base_url()
    except Exception:
        return 'unknown', None
    return _endpoint_scope(url)


def _note_block(failure: dict) -> None:
    """Publish a refused ComfyUI hand-off for the queue dock to explain."""
    global _last_block
    endpoint = failure.get('endpoint') or ''
    reason = failure.get('reason') or 'unreachable'
    now = time.time()
    with _lock:
        prior = _last_block
        # The worker republishes the SAME refusal every second while jobs wait,
        # so `at` (freshness) cannot answer "how long has this stood still?" —
        # it is always a second old. `since` is the start of the episode, kept
        # across republications and reset only when the block actually lapsed
        # or changed into a different refusal.
        continues = bool(prior
                         and prior.get('reason') == reason
                         and prior.get('endpoint') == endpoint
                         and now - float(prior.get('at') or 0) <= _BLOCK_CONTINUITY_S)
        _last_block = {
            'at': now,
            'since': float(prior['since']) if continues else now,
            'reason': reason,
            'endpoint': endpoint,
            'models': list(failure.get('models') or ()),
            'families': sorted(_probe_families.get(endpoint, set())),
        }


def _clear_block() -> None:
    global _last_block
    with _lock:
        _last_block = None


def last_block(max_age_s: float = _BLOCK_FRESH_S):
    """The last refused ComfyUI hand-off, or None once cleared or stale.

    The freshness bound is what keeps this honest: a held queue re-refuses every
    worker cycle (about a second), so a live block is always fresh, while a
    refusal from a queue that has since emptied or recovered goes silent on its
    own instead of explaining a pause that no longer exists."""
    with _lock:
        blk = _last_block
    if not blk or time.time() - float(blk.get('at') or 0) > max_age_s:
        return None
    return {'reason': blk['reason'], 'endpoint': blk['endpoint'],
            'models': list(blk['models']), 'families': list(blk['families']),
            'held_seconds': max(0.0, time.time() - float(blk.get('since') or blk['at']))}


def _shared_endpoint() -> str | None:
    """The endpoint the user consented to share, while that consent is live."""
    global _share_until, _share_endpoint
    with _lock:
        if _share_endpoint is None:
            return None
        if time.monotonic() < _share_until:
            return _share_endpoint
        _share_until, _share_endpoint = 0.0, None   # expired: back to fencing
    return None


def share_state() -> dict:
    """Whether a consented share is live, and for how much longer."""
    endpoint = _shared_endpoint()
    if endpoint is None:
        return {'sharing': False, 'endpoint': None, 'seconds_left': 0}
    with _lock:
        left = max(0.0, _share_until - time.monotonic())
    return {'sharing': True, 'endpoint': endpoint, 'seconds_left': int(left)}


def share_gpu_with_foreign_model() -> dict:
    """"Run it anyway" — proceed WITH a foreign model resident, ON CONSENT ONLY.

    The other half of `unload_foreign_models`, for the case where evicting is
    not an option: the resident model may be another tool's live work, or
    another LDS instance mid-batch. Nothing here touches it. The fence simply
    stops holding the queue for this endpoint until the consent runs out.

    This is deliberately NOT free of cost, and the surfaces say so: on Windows
    an over-committed card does not raise, it pages silently — a measured 13x
    slowdown on the vision side (see `services/vision_keepalive.py`). Sharing
    is therefore the user's call to make about their own card, never a fallback
    the app takes on its own. Nothing calls this on a timer or on a retry.
    """
    blk = last_block()
    if blk is None:
        return {'ok': False, 'reason': 'not-blocked', 'seconds': 0}
    scope, configured = _configured_local_endpoint()
    target = blk.get('endpoint') or (configured if scope == 'local' else None)
    if not target:
        return {'ok': False, 'reason': 'not-local', 'seconds': 0}
    global _share_until, _share_endpoint
    with _lock:
        _share_until = time.monotonic() + SHARE_SECONDS
        _share_endpoint = target
    # The hold is over as far as the queue is concerned: clear it now rather
    # than let the dock keep explaining a pause the user just answered.
    _clear_block()
    logger.warning('ollama GPU fence: user consented to SHARE the GPU with a model '
                   'LDS does not own for %d s', int(SHARE_SECONDS))
    return {'ok': True, 'reason': 'sharing', 'seconds': int(SHARE_SECONDS),
            'models': list(blk.get('models') or ())}


def stop_sharing() -> None:
    """Hand the card back to the fence before the consent runs out."""
    global _share_until, _share_endpoint
    with _lock:
        _share_until, _share_endpoint = 0.0, None


def ensure_released_for_comfy() -> bool:
    """Prove local Ollama is empty before every new ComfyUI prompt admission.

    The read-only ``/api/ps`` probe is intentionally repeated between cells: a
    user can start Ollama after the previous image. It never unloads ComfyUI
    models; it either verifies the runner is empty, releases a model LDS proved
    it owns, or blocks safely without touching a pre-existing user model.
    A refusal is published via ``last_block`` so the queue dock can say WHY the
    queue is standing still; success clears it.
    """
    with _lock:
        candidates = {key: set(value) for key, value in _owned_models.items()}
        for foreign_endpoint in _foreign_local_endpoints:
            candidates.setdefault(foreign_endpoint, set())

    # Deliberately the ACTIVE provider's endpoint only, not both. Guarding the
    # other one's DEFAULT url would probe a second port on every ComfyUI job of
    # every install that never runs it — a behaviour change for everyone, to cover
    # a case that is already covered: an endpoint LDS has used is in _owned_models
    # (so it is a candidate above, even after a provider switch), and a model LDS
    # never admitted is one the fence refuses to touch by doctrine anyway. What is
    # NOT covered is a second daemon holding the card that LDS has never spoken to
    # — and the honest answer there is that the app cannot know about it.
    scope, endpoint = _configured_local_endpoint()
    if scope == 'local':
        candidates.setdefault(endpoint, set())
    # `unknown` — an unparseable Ollama URL, or one with a proxy base-path — is
    # deliberately NOT a block. LDS cannot address such an endpoint, so
    # `mark_before_generate` refuses every vision call on it and LDS has nothing
    # loaded there to release. Refusing the hand-off anyway made a mistyped
    # CAPTIONING setting stop image GENERATION, with a dock sentence pointing at
    # a URL the user may never have meant to use — a blast radius far larger
    # than the risk being managed. Endpoints LDS did use before the URL changed
    # are still in `candidates` above, and are still released properly.

    # A consented share covers exactly one endpoint, and only until it expires.
    shared = _shared_endpoint()
    if shared is not None:
        candidates.pop(shared, None)

    for local_endpoint, expected in candidates.items():
        if not _release_endpoint(local_endpoint, expected):
            with _lock:
                failure = dict(_last_release_failure or {})
            _note_block(failure or {'reason': 'unreachable',
                                    'endpoint': local_endpoint, 'models': ()})
            return False
    _clear_block()
    return True


FENCE_BLOCKED_MESSAGE = (
    'A local Ollama model is already in use outside LDS. LDS will not change it; '
    'unload it first or configure a dedicated Ollama endpoint for LDS.')


def blocked_message(name: str | None = None) -> str:
    """The same refusal, naming the server the user actually runs.

    Kept alongside the constant rather than replacing it: FENCE_BLOCKED_MESSAGE is
    matched by name in the suite, and the Ollama wording is still exactly right for
    an Ollama install. What was wrong was showing it to someone running LM Studio —
    an accurate refusal about the wrong product is a support ticket, not a fix.
    """
    try:
        from . import vision_llm
        if (name or vision_llm.provider()) == vision_llm.LMSTUDIO:
            return ('A model is already loaded in LM Studio outside LDS. LDS will not '
                    'unload it on its own; unload it in LM Studio, or share the GPU '
                    'from the queue panel to run anyway.')
    except Exception:                      # noqa: BLE001 - the default wording is safe
        pass
    return FENCE_BLOCKED_MESSAGE


def fence_status() -> dict:
    """What the fence would say about the CONFIGURED local endpoint, right now.

    A read-only /api/ps probe, so it is safe to poll: the surfaces that were
    refused use it to notice the moment the other model goes away (Ollama's own
    idle unload does that on its own after a few minutes) and resume by
    themselves. ``blocked`` is the whole answer; ``models`` only names what is
    standing in the way so the user is not told to hunt for it.
    """
    scope, endpoint = _configured_local_endpoint()
    # The banner has to name the server the user actually runs: telling someone on
    # LM Studio that "another tool is using a model in Ollama" sends them to look at
    # a daemon that is not holding anything.
    try:
        from . import vision_llm
        _provider = vision_llm.provider()
    except Exception:                      # noqa: BLE001 - the banner degrades, never fails
        _provider = 'ollama'
    if scope == 'remote':
        return {'applies': False, 'blocked': False, 'scope': 'remote', 'models': [],
                'provider': _provider}
    if scope != 'local':
        return {'applies': False, 'blocked': False, 'scope': scope, 'models': [],
                'provider': _provider}

    state, loaded, expiry = _probe(endpoint)
    if state in ('unknown', 'down'):
        # Not reachable / not answering usefully: that is not this fence's
        # story to tell, and reporting "blocked" here would offer an unload
        # button for a daemon nobody can talk to.
        return {'applies': True, 'blocked': False, 'scope': 'local',
                'reachable': False, 'models': [], 'provider': _provider}
    with _lock:
        owned = set(_owned_models.get(endpoint, set()))
        if state != 'empty' and not loaded.issubset(owned):
            owned |= _adopt_persisted(endpoint, loaded - owned, expiry)
            if loaded and loaded.issubset(owned):
                _owned_models.setdefault(endpoint, set()).update(owned)
                _foreign_local_endpoints.discard(endpoint)
    foreign = sorted(loaded - owned)
    return {'applies': True, 'blocked': bool(foreign), 'scope': 'local',
            'reachable': True, 'models': foreign, 'provider': _provider}


def unload_foreign_models() -> dict:
    """Unload the models blocking the local endpoint — ON EXPLICIT CONSENT ONLY.

    Everything else in this module refuses rather than touch a model another
    tool loaded, and that default does not move: this function exists so a user
    who KNOWS what the other model is can say "take it, I'll reload it" in one
    click instead of leaving the app to go find an Ollama prompt. Nothing calls
    it on a timer, on a retry, or as a fallback — only a route behind a
    confirmed flag does.
    """
    scope, endpoint = _configured_local_endpoint()
    if scope != 'local':
        return {'ok': False, 'reason': 'not-local', 'unloaded': [], 'still_loaded': []}
    state, loaded, _ = _probe(endpoint)
    if state == 'unknown':
        return {'ok': False, 'reason': 'unreachable', 'unloaded': [], 'still_loaded': []}
    if state in _RUNNER_HOLDS_NOTHING:
        with _lock:
            _owned_models.pop(endpoint, None)
            _foreign_local_endpoints.discard(endpoint)
        _drop_claims(endpoint)
        return {'ok': True, 'reason': 'already-free', 'unloaded': [], 'still_loaded': []}

    unloaded = [model for model in sorted(loaded) if _post_unload(endpoint, model)]
    state, remaining, _ = _probe(endpoint)
    if state not in _RUNNER_HOLDS_NOTHING or remaining:
        # Ollama acknowledged but the runner is not empty (a request still in
        # flight holds it). Say so rather than let the caller retry into the
        # same wall.
        return {'ok': False, 'reason': 'still-loaded', 'unloaded': unloaded,
                'still_loaded': sorted(remaining)}
    with _lock:
        _owned_models.pop(endpoint, None)
        _foreign_local_endpoints.discard(endpoint)
    _drop_claims(endpoint)
    logger.info('ollama GPU fence: user consented to unloading %d external model(s)',
                len(unloaded))
    return {'ok': True, 'reason': 'unloaded', 'unloaded': unloaded, 'still_loaded': []}


def reset_for_tests() -> None:
    """Forget process-local bookkeeping only; never call a local service."""
    global _last_block, _last_release_failure, _last_block_log_at
    global _share_until, _share_endpoint
    with _lock:
        _owned_models.clear()
        _borrowed_models.clear()
        # Process state like the rest: a driver pinned by one test leaked into the
        # next, and it OUTRANKS the setting by design, so a stale entry made the
        # fence speak the wrong API for the whole run.
        _endpoint_driver.clear()
        _foreign_local_endpoints.clear()
        _probe_families.clear()
        _last_block = None
        _last_release_failure = None
        _last_block_log_at = 0.0
        _share_until = 0.0
        _share_endpoint = None
