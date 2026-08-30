"""Settings API: config/secrets CRUD + capability probes."""
import os
import sys

from flask import Blueprint, current_app, jsonify, request

from .. import capabilities
from .. import config as cfg
from .. import netguard as _netguard
# The path-redaction helper moved to a shared util so services (run_share) can
# reuse it without a route<-service back-import. Kept under its historical
# private name here for the diagnostic call site below.
from ..utils.redact import redact_user_paths as _redact_user_paths

bp = Blueprint('settings', __name__, url_prefix='/api')


_TEST_TARGETS = {
    'gemini': capabilities.probe_gemini,
    'openai': capabilities.probe_openai,
    'openrouter': capabilities.probe_openrouter,
    'comfyui': capabilities.probe_comfyui,
    # End-to-end (reachable + vision model pulled), NOT reachability alone: the old
    # reachability-only target returned a green check while the Setup/diagnostic model
    # probe said the model wasn't pulled on the SAME machine (issue #7). Shares
    # probe_ollama_model so the Test button, the Setup step and the diagnostic are one
    # source of truth.
    'ollama': capabilities.probe_ollama_connection,
    'lmstudio': capabilities.probe_lmstudio_connection,
    # Folder checks PLUS an `import torch` on the chosen interpreter: a Test that
    # goes green on a Python without torch is the trap of GitHub #19 (strouder).
    'aitoolkit': capabilities.probe_aitoolkit_test,
    'face_scoring': capabilities.probe_face_scoring,
    'masks': capabilities.probe_masks,
    'vast': capabilities.probe_vast,
}


def _secret_presence() -> dict:
    return {name: bool(cfg.secret(name)) for name in cfg.SECRET_KEYS}


def _hf_cloud_secret_check(status: dict) -> dict:
    """Adapt the delivery preflight to the Settings TestResult contract."""
    namespace = status.get('namespace')
    warning = status.get('warning')
    if status.get('ok'):
        detail = warning or (
            'HF_CLOUD_TOKEN verified: krea/Krea-2-Raw is readable; '
            f'delivery namespace: {namespace}.')
    else:
        detail = (
            status.get('error') or 'HF_CLOUD_TOKEN could not be verified.')
    result = {
        'ok': bool(status.get('ok')),
        'detail': detail,
        'code': status.get('code') or (
            'ready' if status.get('ok') else 'invalid'),
        'configured': bool(status.get('configured')),
        'severity': status.get('severity') or (
            'success' if status.get('ok') else 'error'),
        'settings_focus': (
            status.get('settings_focus') or 'HF_CLOUD_TOKEN'),
    }
    if warning:
        result['warning'] = warning
    if namespace:
        result['namespace'] = namespace
    return result


def _lan_ip():
    """This machine's primary LAN IPv4, or None. Uses the standard UDP-connect
    trick: opening a datagram socket toward a public address makes the OS pick the
    outbound interface — no packet is ever sent — and getsockname() then reveals
    that interface's IPv4. Returns None on OSError (no route / offline) or when only
    loopback is available, so callers can fall back to a placeholder."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))       # selects the route; no traffic leaves the host
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if ip.startswith('127.') else ip


def _is_cgnat(ip) -> bool:
    """True for the 100.64.0.0/10 carrier-grade-NAT block that Tailscale draws
    every node's address from — the reliable signature of a tailnet IP."""
    try:
        a, b = (int(x) for x in ip.split('.')[:2])
    except (ValueError, AttributeError):
        return False
    return a == 100 and 64 <= b <= 127


def _tailscale_ip():
    """This host's Tailscale IPv4, or None when Tailscale isn't up. Same
    UDP-connect probe as _lan_ip but aimed at Tailscale's service IP
    (100.100.100.100): when the tunnel is up Tailscale owns the route for
    100.64.0.0/10, so the OS picks the tailscale interface and getsockname()
    reveals its address. With Tailscale down the probe falls through the default
    route to the LAN IP, which is outside 100.64/10 and gets rejected — so this
    is None exactly when there's no tailnet address to offer. A Tailscale URL is
    the phone's bulletproof path: it sidesteps Wi-Fi client-isolation, a shifting
    DHCP LAN IP, and works even off the home network."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('100.100.100.100', 80))   # selects the tailnet route; nothing is sent
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return ip if _is_cgnat(ip) else None


def _settings_payload() -> dict:
    # The shipped default of each editable identity/quality prompt, so the
    # Settings UI can display the REAL default text (and offer "Load default to
    # edit") instead of leaving the field blank behind a generic placeholder.
    # Import-pure module (no Flask); these are code constants, not secrets.
    # `identity_prompt_defaults` stays the HUMAN set (the historical payload key,
    # unchanged for any client reading it); `_by_subject` adds the other four sets
    # so the Settings screen — which edits out of any dataset context — can show
    # the real default next to whichever subject type the user is editing.
    from ..services.face_variations import (identity_prompt_defaults,
                                            identity_prompt_defaults_by_subject)
    return {
        'config': cfg.load_config(), 'secrets': _secret_presence(),
        # The shipped default of every config key, for the SCALAR settings what
        # identity_prompt_defaults is for the prompts: `config` above is already
        # merged, so a number in it is indistinguishable from the default and the
        # UI had no way to offer "Reset to default" on a number, a path or a
        # select. Sent whole and derived from cfg.DEFAULTS (see cfg.defaults) so
        # the frontend never carries its own copy of a default that could go
        # stale. No secret lives here — secrets are in .env, and `secrets` above
        # only reports presence.
        'config_defaults': cfg.defaults(),
        'identity_prompt_defaults': identity_prompt_defaults(),
        'identity_prompt_defaults_by_subject': identity_prompt_defaults_by_subject(),
        # What THIS running process is actually bound to — run.py stamps these
        # before app.run(); a dev/test boot that never went through run.py (or a
        # WSGI launch) leaves them unset, so the Server card just hides the
        # "running vs saved" diff instead of showing a misleading n/a:n/a.
        'runtime': {'host': current_app.config.get('LDS_BOUND_HOST'),
                    'port': current_app.config.get('LDS_BOUND_PORT'),
                    # Docker owns the bind through LDS_HOST/LDS_PORT and the
                    # published host port through LDS_HOST_PORT.  The UI uses
                    # this bit to prevent saving controls that cannot take
                    # effect inside the managed container.
                    'bind_managed': os.environ.get('LDS_BIND_MANAGED', '').strip().lower()
                                    in {'1', 'true', 'yes', 'on'},
                    # LDS_PUBLIC=1: the bind is reachable from the internet, so
                    # netguard forces the token gate on whatever server.require_token
                    # says. The card locks the toggle rather than letting the user
                    # flip a switch that cannot take effect.
                    'public': _netguard.public_bind(),
                    # LAN IPv4 so the Server card can show a real, copyable
                    # http://<ip>:port/ URL instead of a <this-computer> placeholder;
                    # None (offline / loopback-only) -> the UI keeps the placeholder.
                    'lan_ip': _lan_ip(),
                    # Tailscale IPv4 (100.64/10), or None when the tunnel is down.
                    # Offered alongside the LAN URL as the phone's bulletproof path
                    # (survives Wi-Fi client-isolation, a shifting DHCP IP, off-LAN).
                    'tailscale_ip': _tailscale_ip()},
    }


@bp.get('/settings')
def get_settings():
    return jsonify(_settings_payload())


@bp.post('/settings/prompt-preview')
def post_prompt_preview():
    """The COMPOSED prompt one engine would receive for one shot — the ~1000
    characters assembled from the six editable parts, which nothing in the app
    ever showed. Pure text: it calls the same wrappers generation calls, and
    starts no job, touches no GPU and spends nothing.

    POST (not GET) because it carries the editor's UNSAVED `identity_prompts`
    tree: Settings saves on an explicit button, so a preview reading the saved
    config would show the previous text at exactly the moment the user is editing.
    Omit `identity_prompts` to preview what is saved. A malformed body degrades to
    a default preview rather than 400 — this panel is a debugging aid, and a
    broken one that answers 'error' is worth less than one that answers with the
    shipped prompt."""
    from ..services import face_variations as fv
    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        body = {}
    overrides = body.get('identity_prompts')
    return jsonify(fv.compose_preview(
        body.get('engine'), subject_type=body.get('subject_type') or 'human',
        framing=body.get('framing') or 'bust', nsfw=bool(body.get('nsfw')),
        suffix=body.get('suffix') if isinstance(body.get('suffix'), str) else '',
        overrides=overrides if isinstance(overrides, dict) else None))


@bp.put('/settings')
def put_settings():
    """Save config/secrets. Used by both the Setup wizard and the Settings page.

    Config sections this running version recognizes (`cfg.DEFAULTS`) are
    validated as before -- 400 on a non-object section, etc. A section this
    version does NOT recognize is tolerated, not rejected: it is dropped from
    the partial (never validated, never applied) so it cannot fail the Save for
    the sections that ARE valid, and reported back in
    `preserved_unknown_sections`. Because save_config() deep-merges the partial
    onto whatever is already on config.json, a dropped section already on disk
    is left untouched there -- preserved for whichever app version wrote it."""
    body = request.get_json(force=True, silent=True) or {}
    if 'config' in body and not isinstance(body['config'], dict):
        return jsonify({'error': "'config' must be an object"}), 400
    if 'secrets' in body and not isinstance(body['secrets'], dict):
        return jsonify({'error': "'secrets' must be an object"}), 400
    config_partial = body.get('config') or {}
    secrets_partial = body.get('secrets') or {}
    # Validate through config's persistence boundary before saving config.json
    # too, so a rejected combined request changes neither file.  This covers all
    # control/format/Unicode line separators and a pre-existing poisoned .env,
    # not only literal CR/LF.
    try:
        cfg.validate_secrets(secrets_partial)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    # A section this running version doesn't know is NOT a reason to fail the
    # whole Save. The client always echoes back the WHOLE config (GET -> edit ->
    # PUT), so a section written by another app version (a dev branch that ran
    # once, a downgrade, an installer ahead of the app) rides along in every
    # subsequent Save. Rejecting it with 400 took the entire request hostage --
    # every valid setting the user just entered was lost too (hit for
    # 'bank_scoring', then again for 'variations'). Instead: drop it from the
    # partial before validation/merge (neither validated nor applied), and
    # report it so the caller can see what was skipped. save_config() deep-merges
    # the partial onto the file already on disk, so a dropped section that
    # exists there is simply left alone -- preserved for the version that wrote
    # it; one that doesn't exist there is dropped with nothing to preserve.
    unknown = set(config_partial) - set(cfg.DEFAULTS)
    preserved_unknown_sections = sorted(unknown)
    for k in unknown:
        config_partial.pop(k)
    # Each section must stay an object -- _deep_merge only recurses when both
    # sides are dicts, so a non-dict value here would REPLACE the whole section
    # (e.g. {"ollama": "x"} silently overwriting ollama.url + ollama.vision_model).
    for k, v in config_partial.items():
        if not isinstance(v, dict):
            return jsonify({'error': f"config section '{k}' must be an object"}), 400
    # Auto-correct the classic portable-bundle mistake: a base_dir pointing at
    # ...\ComfyUI_windows_portable gets rewritten to the nested ...\ComfyUI that
    # actually holds main.py + models/, so the base/model listers find checkpoints
    # instead of silently scanning an empty ...\<wrapper>\models.
    bd = (config_partial.get('comfyui') or {}).get('base_dir')
    if bd:
        r = capabilities.resolve_comfyui_base(bd)
        if r['valid'] and r['nested']:
            config_partial['comfyui']['base_dir'] = r['resolved']
        # Pointing the app at a directory ANNULS a prior "continue without ComfyUI"
        # skip — the user changed their mind. Clearing the stored flag keeps
        # config.json honest (the derived comfyui.skipped would already read False
        # with base_dir set, but a lingering true would resurface if base_dir is
        # later cleared). Only when the client didn't already send an explicit value.
        if 'setup_skipped' not in config_partial['comfyui']:
            config_partial['comfyui']['setup_skipped'] = False
    # The scoped-ML interpreters (watermark/masks/face_scoring/bank_semantic
    # `.python`) have NO input
    # in Settings — they're written out-of-band by the installers (the watermark
    # "Install inpainting" button auto-provisions a dedicated venv and records its
    # python here). The frontend only ever echoes back what it loaded, blank on a fresh
    # install, so a full-config Save carries `python: ""` for these keys. Left in the
    # partial that blank deep-merges OVER the auto-provisioned path, blanking it — after
    # which probe_watermark_inpaint falls back to the app's own Pillow-12 venv and the
    # feature reads "NOT installed" forever despite a perfect install. Drop the blank so
    # a stale Save can't undo an install. (aitoolkit.python IS user-editable — not here.)
    for _managed in ('watermark', 'masks', 'face_scoring', 'bank_scoring',
                     'bank_semantic', 'watermark_detect', 'shot_detect'):
        node = config_partial.get(_managed)
        if isinstance(node, dict) and 'python' in node and not str(node.get('python') or '').strip():
            node.pop('python')
    hf_cloud_check = None
    hf_cloud_candidate = secrets_partial.get('HF_CLOUD_TOKEN')
    if hf_cloud_candidate:
        from ..services import cloud_training
        hf_cloud_check = _hf_cloud_secret_check(
            cloud_training.full_transformer_token_status(hf_cloud_candidate))
        if not hf_cloud_check['ok']:
            return jsonify({
                'error': hf_cloud_check['detail'],
                'secret_checks': {'HF_CLOUD_TOKEN': hf_cloud_check},
            }), 400
    cfg.save_config(config_partial)
    cfg.set_secrets(secrets_partial)
    # A changed ComfyUI location must take effect NOW: the base/model listers cache
    # their scans for 5 min, so without this the training-base dropdowns keep showing
    # the pre-save (often empty) list right after the user points the app at ComfyUI.
    # (The wizard's _scan_models view refreshes via the frontend's forced
    # /api/capabilities?force=1 call, so no probe(force) is needed here.)
    if 'comfyui' in config_partial:
        from ..utils import comfyui
        comfyui.clear_model_caches()
    payload = _settings_payload()
    if hf_cloud_check is not None:
        payload['secret_checks'] = {'HF_CLOUD_TOKEN': hf_cloud_check}
    if preserved_unknown_sections:
        payload['preserved_unknown_sections'] = preserved_unknown_sections
    return jsonify(payload)


@bp.delete('/settings/secret/<name>')
def delete_secret(name):
    """Clear a saved API key. Explicit deletion — set_secrets ignores blanks so a
    key can never be wiped by just emptying its (write-only) field."""
    if name not in cfg.SECRET_KEYS:
        return jsonify({'error': 'unknown secret'}), 400
    cfg.delete_secrets([name])
    return jsonify(_settings_payload())


@bp.get('/capabilities')
def get_capabilities():
    force = bool(request.args.get('force'))
    return jsonify(capabilities.probe(force=force))


@bp.get('/loras/list')
def loras_list():
    """LoRAs on disk for the generation-LoRA preset picker, shared by the
    Klein and the Krea 2 Edit cards: the whole loras tree (base
    ``models/loras`` + every ``extra_model_paths.yaml`` root, recursive),
    each badged with its architecture — ``{loras: [{name, arch, label, compatible}]}``,
    Klein-compatible first. ``name`` is the exact ComfyUI-relative value a preset row
    stores and the generate path resolves (``comfy_model_paths.list_models('loras')``).
    ``compatible`` is judged against ``?family=`` (default the Klein graph); the
    Krea preset card asks for ``family=krea``.
    ``?force=1`` bypasses the mtime cache (the ↻ rescan button). Degrades to
    ``{loras: []}`` — never an error — when no loras root exists (ComfyUI
    unconfigured) or the scan fails, so the picker falls back to a free-text field
    instead of a blocking empty dropdown."""
    from ..services import klein_lora_picker
    force = bool(request.args.get('force'))
    # The picker of whichever engine is asking. Unknown values fall back to the
    # default inside the scanner, so this stays a plain pass-through.
    family = (request.args.get('family') or '').strip() or klein_lora_picker._KLEIN_FAMILY
    try:
        loras = klein_lora_picker.scan_generation_loras(force=force, family=family)
    except Exception:
        current_app.logger.exception('loras list scan failed')
        loras = []
    return jsonify({'loras': loras})


@bp.get('/comfy/model-files')
def comfy_model_files():
    """The model files on disk for ONE picker slot — what backs the Klein model-file
    fields and Krea 2 Edit's base model / identity LoRA, which used to be free text.

    ``?slot=`` is one of ``comfy_model_picker.SLOTS``; ``?force=1`` bypasses the
    mtime cache (the ↻ rescan button). Answers
    ``{files: [relative loader name], folder: "where to put one"}``.

    Each name is exactly the string the field already stored, so switching those
    fields to a picker needs no alias and an install that typed a name by hand
    finds it selected. ``folder`` is what the empty state SAYS to do — a mute
    empty dropdown is the one outcome this endpoint must never produce.

    An unknown slot and any scan failure both answer 200 with an empty list: the
    picker degrades to a free-text field rather than blocking the panel."""
    from ..services import comfy_model_picker
    slot = (request.args.get('slot') or '').strip()
    force = bool(request.args.get('force'))
    files, folder = comfy_model_picker.list_slot_files(slot, force=force)
    return jsonify({'files': files, 'folder': folder})


@bp.get('/seedvr2/models')
def seedvr2_models_list():
    """The SeedVR2 DiT builds actually PRESENT in this install's SEEDVR2 folder(s),
    plus the catalog of builds the app can talk about.

    ``{installed: [name], catalog: [{file, label, size_gb, vram_gb, recommended,
    installed}], resolved: name|null, vae: name|null,
    vae_choices: [{file, likely_vae}]}``.

    ``vae_choices`` covers the WHOLE folder, each entry flagged with whether its
    name looks like a VAE: the automatic path already handles every install
    where it does, so the pin exists for the one where it does not, and a picker
    that hid those files could not express that install.

    Only installed builds are offered as a pin: the pack's loader nodes download
    an unknown name on first use, so a picker listing everything would turn a
    dropdown into a silent multi-gigabyte download. The catalog is still returned
    so the card can SHOW what else exists (with its size and VRAM guidance) and
    say it has to be placed in the folder — informing is not fetching.

    Degrades to empty lists rather than an error when ComfyUI is unconfigured."""
    from ..services import seedvr2_helper as svr
    try:
        installed = svr.installed_dit_models()
        resolved = svr.resolve_seedvr2_dit()
        vae = svr.resolve_seedvr2_vae()
        vae_choices = svr.vae_choices()
    except Exception:
        current_app.logger.exception('seedvr2 model scan failed')
        installed, resolved, vae, vae_choices = [], None, None, []
    catalog = [{**v, 'installed': v['file'] in installed} for v in svr.DIT_VARIANTS]
    return jsonify({'installed': installed, 'catalog': catalog,
                    'resolved': resolved, 'vae': vae,
                    'vae_choices': vae_choices})


def _interpreter_list(profile):
    """Pythons on this machine that could run one heavy pass, each with a
    per-dependency verdict — the picker behind "use a GPU Python you already
    have". ``?force=1`` re-probes (after the user pip-installed something);
    ``?path=`` adds a hand-typed interpreter to the list. Read-only: nothing is
    ever installed into an environment the app did not build.

    Degrades rather than 500s, so the panel can never break the page — but it
    says WHICH degradation it is. An empty list is also the legitimate verdict
    'nothing to borrow on this machine', and returning that shape for a crash
    left the user with no reason to press '↻ Check again' about a failure that
    retrying might well fix. `detection_failed` separates the two, and
    `default_python` stays filled in: it is the one thing we know regardless."""
    from ..services import scoring_python
    prof = scoring_python.get_profile(profile)
    force = bool(request.args.get('force'))
    if force:
        # "↻ Check again" is what a user clicks right after installing a package
        # by hand. Re-probing only OUR cache would leave the capability probes
        # (10 min TTL) still saying "not installed" — two surfaces disagreeing
        # about the same interpreter is exactly what makes a fix look broken.
        capabilities.clear_import_cache()
    try:
        return jsonify(scoring_python.detect(
            force=force, extra_path=request.args.get('path') or '',
            profile=prof))
    except Exception as e:
        current_app.logger.exception('%s interpreter detection failed', prof.key)
        try:
            selected = (cfg.get(prof.config_key) or '').strip()
        except Exception:      # noqa: BLE001 — a config read that also fails
            selected = ''
        return jsonify({
            'selected': selected,
            'profile': prof.key,
            'default_python': sys.executable,
            'interpreters': [],
            'detection_failed': True,
            'detection_error': str(e)[:300],
        })


def _interpreter_select(profile):
    """Point one feature at an interpreter (``{python: "<path>"}``), or back at
    the app's own (``{python: ""}``). Refuses — 400, with the verdict attached —
    any interpreter that could not be proven able to run the pass, so a bad pick
    can never turn an hour of work into an import error."""
    from ..services import scoring_python
    body = request.get_json(silent=True) or {}
    try:
        result = scoring_python.select(body.get('python') or '', profile=profile)
    except scoring_python.SelectionError as e:
        return jsonify({'error': str(e), 'verdict': e.verdict}), 400
    except Exception as e:
        current_app.logger.exception('interpreter selection failed')
        return jsonify({'error': f'could not save the interpreter: {e}'}), 500
    return jsonify(result)


@bp.get('/scoring-python')
def scoring_python_list():
    """The picker for ✨ Score (``bank_scoring.python``)."""
    return _interpreter_list('scoring')


@bp.post('/scoring-python')
def scoring_python_select():
    """Select the interpreter ✨ Score runs in."""
    return _interpreter_select('scoring')


@bp.get('/semantic-python')
def semantic_python_list():
    """The same picker for the SigLIP 2 semantic engine (``bank_semantic.python``).

    Same detector, a SHORTER dependency list: the semantic worker never imports
    open_clip or timm, so an interpreter Score would refuse can be perfectly
    good here. The pinned SigLIP2 weights live in the app's data folder, not in
    the interpreter, so a borrowed Python needs no download at all."""
    return _interpreter_list('semantic')


@bp.post('/semantic-python')
def semantic_python_select():
    """Select the interpreter the SigLIP 2 index runs in. Execution only — Setup
    ▸ Quality tools still installs into the app-managed environment and never
    into a borrowed one."""
    return _interpreter_select('semantic')


@bp.get('/watermark-python')
def watermark_python_list():
    """The same picker for the 🚩 watermark detector (``watermark_detect.python``).

    The Setup install pins the app-managed environment here — which carries a
    CPU torch on purpose — so until this picker existed a machine with a
    perfectly good CUDA interpreter scanned tens of thousands of images on the
    CPU and nothing ever said so. The detector needs torch + a transformers
    that has both cascade halves (SigLIP classifier, Grounding-DINO locator);
    its pinned weights live under the app's models_root, not in the
    interpreter, so borrowing one downloads nothing."""
    return _interpreter_list('watermark_detect')


@bp.post('/watermark-python')
def watermark_python_select():
    """Select the interpreter the 🚩 watermark scan runs in. Execution only —
    Setup ▸ Quality tools still installs into the app-managed environment and
    never into a borrowed one."""
    return _interpreter_select('watermark_detect')


@bp.post('/settings/test/<target>')
def test_connection(target):
    if target == 'hf_cloud':
        from ..services import cloud_training
        return jsonify(_hf_cloud_secret_check(
            cloud_training.full_transformer_token_preflight()))
    probe_fn = _TEST_TARGETS.get(target)
    if probe_fn is None:
        return jsonify({'error': f"unknown test target '{target}'"}), 404
    return jsonify(probe_fn())


# Update check: compares the latest GitHub release tag to the local version.
# Cached 6 h so the SPA banner can call it freely. Degrades to
# update_available=False with a reason when the feed is unreachable (offline,
# repo private, no release yet) — never an error, never a blocker.
_UPDATE_TTL = 6 * 3600           # GitHub releases feed (packaged builds; rare)
_GIT_CHECK_TTL = 3600            # git commits-behind check — the project moves fast
_update_cache = {'ts': 0.0, 'data': None}
# Auto-detection (nav badge): the git fetch is allowed but CACHED — the SPA
# asks on every load, the network is hit at most once per TTL.
_git_check_cache = {'ts': 0.0, 'data': None}


@bp.get('/update/check')
def update_check():
    import time
    import requests
    from ..version import APP_VERSION
    from ..services import updater
    force = bool(request.args.get('force'))
    auto = bool(request.args.get('auto'))
    docker_runtime = updater.is_docker_runtime()
    # A git checkout: the meaningful signal is commits-behind-origin (the user pushes
    # commits to a branch, not tagged releases — a release-only check reads "up to date"
    # while the tree is many commits behind). The fetch runs on an explicit check
    # (force, always fresh) or an auto check (nav badge — served from a TTL cache so
    # SPA loads don't hammer the network); never from the bare passive path.
    if (force or auto) and not docker_runtime and updater.is_git_checkout():
        now = time.time()
        if auto and not force and _git_check_cache['data'] is not None \
                and (now - _git_check_cache['ts']) < _GIT_CHECK_TTL:
            return jsonify(_git_check_cache['data'])
        gs = updater.git_update_status()
        if gs is not None:
            # Pinokio: the commits-behind answer stays true and useful (its
            # Update tab pulls exactly those), only the in-app apply must go.
            if updater.is_pinokio_runtime():
                gs.update(updater.pinokio_update_payload())
            _git_check_cache.update(ts=now, data=gs)
            return jsonify(gs)
    now = time.time()
    if (_update_cache['data'] is not None and (now - _update_cache['ts']) < _UPDATE_TTL
            and not force):
        return jsonify(_update_cache['data'])
    repo = cfg.get('updates.repo') or 'perfectgf/lora-dataset-studio'
    out = {'ok': True, 'current': APP_VERSION, 'latest': None,
           'update_available': False, 'url': f'https://github.com/{repo}/releases'}
    if docker_runtime:
        out.update(updater.docker_update_payload())
    if updater.is_pinokio_runtime():
        out.update(updater.pinokio_update_payload())
    sha = updater.current_sha()
    if sha:
        out['current_sha'] = sha
    try:
        r = requests.get(f'https://api.github.com/repos/{repo}/releases/latest',
                         timeout=6, headers={'Accept': 'application/vnd.github+json'})
        if r.status_code == 200:
            j = r.json()
            latest = (j.get('tag_name') or '').lstrip('vV').strip()
            out['latest'] = latest or None
            out['url'] = j.get('html_url') or out['url']
            # Date-based versions (YYYY.MM.DD[.N]) -> plain string comparison.
            out['update_available'] = bool(latest) and latest > APP_VERSION
            # can_apply: this release ships a downloadable ZIP asset, so a packaged
            # (non-git) install can update in-app instead of only linking out to the
            # releases page. The button keys off this for a ZIP install.
            zip_size = 0
            for a in (j.get('assets') or []):
                name = (a.get('name') or '').lower()
                if name.endswith('.zip') and a.get('browser_download_url'):
                    zip_size = int(a.get('size') or 0)
                    if 'windows' in name:
                        break
            # Neither container nor Pinokio install may advertise an in-app
            # apply just because the release carries a ZIP: both update from
            # outside this process.
            if not docker_runtime and not updater.is_pinokio_runtime():
                out['can_apply'] = bool(zip_size) or any(
                    (a.get('name') or '').lower().endswith('.zip') and a.get('browser_download_url')
                    for a in (j.get('assets') or []))
                out['zip_size'] = zip_size      # bytes, for the "download ~XX MB" hint
        else:
            out['reason'] = (f'release feed answered {r.status_code} '
                             '(no public release yet?)')
    except requests.RequestException:
        out['reason'] = 'offline or GitHub unreachable'
    _update_cache.update(ts=now, data=out)
    return jsonify(out)


@bp.post('/update/apply')
def update_apply():
    """Update to the latest version and, if anything changed, restart the server.

    A git checkout pulls the latest commits and restarts synchronously (fast).
    A packaged (ZIP) install starts the download+swap on a background thread and
    returns {ok, async:true, from, to, total}; the client then polls
    /api/update/progress and, when it reports 'restarting', /api/health. The
    trivial ZIP outcomes (up to date / no ZIP asset / offline) come back inline
    just like the git path. Both defer changed requirements to the restart helper."""
    from ..services import updater
    # Refuse before even probing .git or the release feed: /app is immutable in
    # the GPU image and an in-container update would be both incomplete and
    # discarded on recreation.
    if updater.is_docker_runtime():
        return jsonify({
            'ok': False,
            'reason': 'Docker GPU installs must be updated by rebuilding the image.',
            **updater.docker_update_payload(),
        })
    # Pinokio owns the process: pulling here would work, but the restart that
    # follows would detach the server from the launcher that is supposed to
    # stop and start it. Refuse before touching git.
    if updater.is_pinokio_runtime():
        return jsonify({
            'ok': False,
            'reason': 'Pinokio installs are updated from the launcher: '
                      'Stop, then Update, then Start.',
            **updater.pinokio_update_payload(),
        })
    if updater.is_git_checkout():
        res = updater.apply_update()
        res['restarting'] = bool(res.get('ok') and res.get('changed'))
        if res['restarting']:
            # invalidate the cached checks so the banner/badge re-evaluate post-update
            _update_cache.update(ts=0.0, data=None)
            _git_check_cache.update(ts=0.0, data=None)
            updater.schedule_restart(install_requirements=bool(res.get('deps_changed')))
        return jsonify(res)
    # Packaged install: async release update with a progress poll.
    _update_cache.update(ts=0.0, data=None)
    return jsonify(updater.start_zip_update())


@bp.get('/update/progress')
def update_progress():
    """Live progress of an in-flight release-ZIP update (packaged install). The
    client polls this after an async /update/apply: phase is one of downloading /
    extracting / installing / restarting / done / error, with byte counts while
    downloading and an honest `error` (already rolled back) on failure."""
    from ..services import updater
    return jsonify(updater.zip_update_progress())


@bp.post('/settings/restart')
def settings_restart():
    """Manual restart — used after saving server.host/server.port (a live bind
    change needs a fresh process; Flask can't rebind mid-request) and as a plain
    troubleshooting action. Same schedule_restart() as the updater, so it
    survives both a git checkout and the packaged build.

    Outside a managed container, pins the restarted process to the SAVED
    host/port via env: the launcher
    (start.bat) exports LDS_PORT, which otherwise wins over config.json forever
    — so without this, changing the port in Settings + restart would keep coming
    back on the launcher's port and the field would look broken. schedule_restart
    passes os.environ down to the relaunch, so setting it here is what makes the
    saved port actually take effect."""
    from ..services import updater
    bind_managed = os.environ.get('LDS_BIND_MANAGED', '').strip().lower() \
        in {'1', 'true', 'yes', 'on'}
    environment_updates = None if bind_managed else {
        'LDS_HOST': str(cfg.get('server.host') or '127.0.0.1'),
        'LDS_PORT': str(cfg.get('server.port') or 5050),
    }
    try:
        updater.schedule_restart(block_during_update=True,
                                 environment_updates=environment_updates)
    except updater.UpdateInProgressError as exc:
        phase = exc.phase
        return jsonify({'ok': False, 'restarting': False, 'phase': phase,
                        'reason': 'An update is already in progress.'}), 409
    return jsonify({'ok': True, 'restarting': True})


@bp.get('/trash')
def trash_info():
    """Trash size for the Settings card — everything the app 'deletes' lands
    there; only 'Empty trash' below actually destroys bytes."""
    from ..services import trash
    return jsonify({'size_bytes': trash.trash_size()})


@bp.get('/run-archive')
def run_archive_info():
    """Size + ceiling of the training-image archive (the deduplicated copies that
    let a comparison still SHOW an image deleted since it was trained)."""
    from ..services import run_archive
    return jsonify({'size_bytes': run_archive.size_bytes(refresh=True),
                    'max_bytes': run_archive.max_bytes(),
                    'enabled': run_archive.enabled()})


@bp.post('/run-archive/clear')
def run_archive_clear():
    """Drop every archived image. Runs, settings and caption text are in the
    database and survive; only the ability to LOOK at a since-deleted image goes."""
    from ..services import run_archive
    return jsonify({'ok': True, **run_archive.clear()})


@bp.get('/storage/locations')
def storage_locations():
    """The “what lives where” map: every category, its effective path and
    whether it can be relocated. Deliberately size-free — measuring means
    walking tens of thousands of files, which belongs to an explicit click
    (see /storage/sizes), never to a tab mount."""
    from ..services import storage_locations as sl
    payload = {'locations': sl.locations()}
    for entry in payload['locations']:
        if entry['path']:
            entry['volume'] = sl.free_space(entry['path'])
    return jsonify(payload)


@bp.get('/storage/sizes')
def storage_sizes():
    """Bytes held by each requested category (?keys=a,b — all of them when
    absent). This is the walk the map's “Measure” button pays for."""
    from ..services import storage_locations as sl
    raw = (request.args.get('keys') or '').strip()
    keys = [k for k in raw.split(',') if k.strip()] if raw else None
    sizes = sl.sizes(keys)
    return jsonify({'ok': True, 'sizes': sizes,
                    'total_bytes': sum(sizes.values())})


@bp.post('/storage/validate')
def storage_validate():
    """Is this folder usable for that category? Proves writability by writing a
    probe file — permission bits lie on Windows. Writes no config."""
    from ..services import storage_locations as sl
    body = request.get_json(silent=True) or {}
    key = str(body.get('key') or '')
    return jsonify(sl.validate_target(key, body.get('path')))


@bp.post('/storage/move')
def storage_move():
    """Start moving a category's current content into its new folder. The
    location change itself is a normal settings save; this only carries the
    bytes, and never runs unless the user asked for it (the alternative,
    “adopt the empty folder”, saves without calling this)."""
    from ..services import storage_locations as sl
    body = request.get_json(silent=True) or {}
    try:
        res = sl.start_move(str(body.get('key') or ''), body.get('path'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **res})


@bp.get('/storage/move/progress')
def storage_move_progress():
    from ..services import storage_locations as sl
    return jsonify({'ok': True, 'job': sl.move_progress(request.args.get('job_id'))})


@bp.post('/storage/adopt-checkpoints')
def storage_adopt_checkpoints():
    """Re-run the staging → checkpoint-store retrofit on demand. Idempotent: on
    an already-migrated install it moves nothing and says so."""
    from ..services import cloud_training as ct
    return jsonify({'ok': True, **ct.migrate_checkpoints_into_store(force=True)})


@bp.post('/trash/open')
def trash_open():
    """Open the server-resolved trash directory; the client supplies no path."""
    from ..services import trash
    try:
        trash.open_trash_folder()
    except Exception:
        current_app.logger.exception('could not open trash folder')
        return jsonify({'error': 'could not open trash folder'}), 500
    return jsonify({'ok': True})


@bp.post('/trash/empty')
def trash_empty():
    from ..services import trash
    return jsonify({'ok': True, **trash.empty_trash()})


def _log_tail_lines(n):
    """(file_name, last_n_lines) of the server log. Reads data/app.log (the
    app's own rotating log), falling back to data/server.log (the portable
    launcher's raw stdout capture). (None, []) when no log exists yet."""
    import os
    from pathlib import Path
    data_dir = Path(os.environ.get('LDS_DATA_DIR', str(cfg.REPO_ROOT / 'data')))
    for name in ('app.log', 'server.log'):
        p = data_dir / name
        if p.is_file():
            try:
                size = p.stat().st_size
                with open(p, encoding='utf-8', errors='replace') as fh:
                    if size > 512 * 1024:               # tail window, never the whole file
                        fh.seek(size - 512 * 1024)
                    return name, fh.read().splitlines()[-n:]
            except OSError:
                continue
    return None, []


# A logging record starts with the file handler's timestamp+level prefix
# ('%(asctime)s %(levelname)s %(name)s: …', see create_app). Any line that does
# NOT match is a continuation of the record above — i.e. a traceback frame. This
# lets the diagnostic reassemble whole ERROR records (message + full stack)
# instead of the plain last-N-lines tail, which routinely cuts a traceback in half.
import re as _re
_LOG_RECORD_RE = _re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (\w+) ')


def _error_log_records(max_records=2, max_lines_per_record=30):
    """The last ERROR/CRITICAL records from the app log, each WITH its full
    traceback — this is what a bug report actually needs (volksrods had to hunt for
    and paste his stack by hand). Path-redacted (home dir -> ~) and capped so a
    runaway stack can't bloat the pasted report. [] when nothing errored."""
    _, lines = _log_tail_lines(400)          # wider window than the plain tail
    if not lines:
        return []
    records = []                             # [(level, [physical lines]), ...]
    for line in lines:
        m = _LOG_RECORD_RE.match(line)
        if m:
            records.append((m.group(1), [line]))
        elif records:                        # traceback frame of the record above
            records[-1][1].append(line)
        # a leading orphan (window cut mid-record) has no owner -> dropped
    errors = [r for r in records if r[0] in ('ERROR', 'CRITICAL')]
    out = []
    for _level, rec_lines in errors[-max_records:]:
        kept = rec_lines[:max_lines_per_record]
        if len(rec_lines) > max_lines_per_record:
            kept.append(f'    … +{len(rec_lines) - max_lines_per_record} more line(s)')
        out.extend(_redact_user_paths(l) for l in kept)
    return out


def _pillow_health() -> dict:
    """Pillow version + a healthy/mixed verdict, computed from the FILES on disk by
    the SAME check the boot self-heal runs (bootstrap_dependencies): a half-swapped
    Pillow raises "property 'mode' has no setter" on the first image decode. Lets a
    report say 'Pillow mixed' instead of the cryptic decode crash that produced it.
    version None => not installed; healthy None => couldn't inspect (frozen build)."""
    try:
        from bootstrap_dependencies import incompatible_pillow_plugins
        version, bad = incompatible_pillow_plugins()
    except Exception:
        return {'version': None, 'healthy': None, 'incompatible_plugins': []}
    if version is None:
        return {'version': None, 'healthy': None, 'incompatible_plugins': []}
    return {'version': version, 'healthy': not bad,
            'incompatible_plugins': [p.name for p in bad]}   # basenames only, no path


def _disk_free() -> dict:
    """Free/total space of the volume the app lives on, in GB (numbers only — the
    path itself never leaves the machine). 'no space left' is a silent killer of
    generation + downloads, so the number belongs in the report. {} on failure."""
    import shutil
    try:
        u = shutil.disk_usage(str(cfg.REPO_ROOT))
        return {'free_gb': round(u.free / 1024 ** 3, 1),
                'total_gb': round(u.total / 1024 ** 3, 1)}
    except OSError:
        return {}


def _recent_generation_errors(scan=40) -> dict:
    """Most-recent failed-generation reason PER engine, plus the last Studio (LoRA
    test) failure — the text shown on a "failed" tile. Answers "it fails at every
    generation" with the real cause instead of a guess. Paste-safe: fail_reason /
    error hold engine/API/save/ComfyUI messages (prompts live in a SEPARATE column),
    still path-redacted and length-capped here. {} when nothing has failed."""
    def _clean(s):
        return _redact_user_paths((s or '').strip())[:300]

    engines = {}
    try:
        from ..models import FaceDatasetImage
        rows = (FaceDatasetImage.query
                .filter(FaceDatasetImage.status == 'failed',
                        FaceDatasetImage.fail_reason.isnot(None))
                .order_by(FaceDatasetImage.id.desc()).limit(scan).all())
        for r in rows:                       # rows are newest-first
            reason = _clean(r.fail_reason)
            if not reason:
                continue
            low = reason.lower()
            # fail_reason is written as f'{engine}: …' on the generation path, so the
            # engine is the prefix; anything else lands in 'other' (save/queue errors).
            eng = next((e for e in ('klein', 'chatgpt', 'nanobanana', 'openrouter')
                        if low.startswith(e)), 'other')
            engines.setdefault(eng, reason)  # first hit == most recent for that engine
    except Exception:
        pass
    studio = None
    try:
        from ..models import LoraTestImage
        srow = (LoraTestImage.query
                .filter(LoraTestImage.status == 'failed', LoraTestImage.error.isnot(None))
                .order_by(LoraTestImage.id.desc()).first())
        if srow:
            studio = _clean(srow.error) or None
    except Exception:
        pass
    out = {}
    if engines:
        out['engines'] = engines
    if studio:
        out['studio'] = studio
    return out


@bp.get('/logs/tail')
def logs_tail():
    """Last N lines of the server log for the in-app viewer — so a novice can
    copy-paste an error instead of hunting for files."""
    try:
        n = max(10, min(1000, int(request.args.get('n', 300))))
    except ValueError:
        n = 300
    name, lines = _log_tail_lines(n)
    return jsonify({'ok': True, 'file': name, 'lines': lines})


@bp.get('/diagnostic')
def diagnostic():
    """Paste-safe bug-report payload: version, platform, capability booleans and
    the log tail. Secret VALUES never appear (presence booleans only) and paths
    are reduced to *_set booleans — the output is meant to be pasted into a
    public issue or Discord thread as-is. (Log lines may still cite file names;
    the UI tells the user to skim before posting.)"""
    from ..services import vision_llm as _vision_llm
    import platform
    import sys
    import time
    from ..version import APP_VERSION
    from ..services import updater
    from ..services import lineage_backfill as _lineage_backfill
    from ..services import framing_backfill as _framing_backfill
    conf = cfg.load_config()
    caps = capabilities.probe()
    e = caps.get('engines') or {}
    comfy = caps.get('comfyui') or {}
    oll = caps.get('ollama') or {}
    sub = caps.get('chatgpt_subscription') or {}
    # Redact ONLY in this paste-safe payload — /api/logs/tail (the in-app log
    # viewer) keeps the raw lines, they're local-only and never meant to be
    # copy-pasted into a public thread.
    _, log_lines = _log_tail_lines(80)
    log_lines = [_redact_user_paths(l) for l in log_lines]
    return jsonify({
        'app_version': APP_VERSION,
        'git_sha': updater.current_sha(),
        'os': f'{platform.system()} {platform.release()}',
        'python': sys.version.split()[0],
        # This interpreter vs the wheel-supported ML range (3.10–3.12): a 3.13+ Flask
        # venv is the cryptic "pip install -r requirements-ml.txt fails" a fresh clone
        # hits (no numpy<2 / insightface wheels), so it belongs up front in a report.
        'python_ml': capabilities.python_ml_status(),
        # Health of the app's Pillow, from the files on disk (the boot self-heal's own
        # check): a mixed install is the "property 'mode' has no setter" decode crash.
        'pillow': _pillow_health(),
        # Free disk on the app's volume — "no space left" silently kills generation
        # and model downloads. Numbers only; the path never leaves the machine.
        'disk': _disk_free(),
        'secrets_present': _secret_presence(),
        'capabilities': {
            # Every engine the build knows about — the formatter already prints
            # openrouter, so leaving it out of the payload made every report say
            # "openrouter=no" no matter how it was configured.
            'engines': {'nanobanana': bool(e.get('nanobanana')),
                        'chatgpt': bool(e.get('chatgpt')),
                        'openrouter': bool(e.get('openrouter')),
                        'klein': bool(e.get('klein'))},
            'comfyui_reachable': bool(comfy.get('reachable')),
            'klein_model': bool((comfy.get('models') or {}).get('klein')),
            # setup_installer action names for the Klein assets NOT yet on disk — the
            # exact gap antonp's report couldn't name (issue: missing Klein assets not
            # listed). Empty required-trio => the engine is asset-ready.
            'klein_missing': list(comfy.get('klein_missing') or []),
            # ChatGPT engine can be lit by an API key OR a connected subscription — a
            # report that shows neither key nor subscription explains a dark engine.
            'chatgpt_subscription': bool(sub.get('connected')),
            'ollama_reachable': bool(oll.get('reachable')),
            'vision_model_ready': bool(oll.get('vision_model_ready')),
            'face_scoring': bool(caps.get('face_scoring')),
            'masks': bool(caps.get('masks')),
            'aitoolkit_valid': bool((caps.get('aitoolkit') or {}).get('valid')),
            'training_visible': bool(caps.get('training_visible')),
            'studio_visible': bool(caps.get('studio_visible')),
            'cloud_training': bool(caps.get('cloud_training')),
        },
        # Live ComfyUI runtime (version / GPU / VRAM / queue) when it answers — {}
        # when unreachable. Tells a "generation hangs" apart from a busy queue or a
        # VRAM-starved GPU. Network, so outside the network-free probe() above.
        'comfyui_runtime': capabilities.comfyui_runtime(),
        'config': {
            'captioning_backend': (conf.get('captioning') or {}).get('backend'),
            'default_engine': (conf.get('engines') or {}).get('default'),
            'enabled_engines': (conf.get('engines') or {}).get('enabled'),
            'training_default_family': (conf.get('training') or {}).get('default_family'),
            'comfyui_base_dir_set': bool((conf.get('comfyui') or {}).get('base_dir')),
            'aitoolkit_dir_set': bool((conf.get('aitoolkit') or {}).get('dir')),
            # allow_crop is the watermark auto-routing default (the trap: crop vs
            # inpaint). A non-sensitive knob whose value silently changes behaviour.
            'watermark_allow_crop': bool((conf.get('watermark') or {}).get('allow_crop')),
            # 🔤 Find text sensitivity — the launch window's slider reads it
            # here and writes it through PUT /api/settings. One stored value,
            # both surfaces (bank and dataset passes read the same key).
            'text_scan_score_min': (conf.get('text_scan') or {}).get('score_min'),
            'lan_enabled': (conf.get('server') or {}).get('host') not in (None, '', '127.0.0.1', 'localhost', '::1'),
        },
        # The configured vision-model string + the tags the probe actually sees at
        # /api/tags. This is what makes a 'vision_model=no' report self-diagnosing:
        # a reader can tell a truly-missing model from one that IS listed under a
        # slightly different identifier (namespace/registry/field variance, issue #7).
        # Model names are not secrets; paths are still absent from this block.
        'ollama': capabilities.ollama_diagnostic(),
        # Both, always, plus which one is in charge. A report that names only
        # Ollama on an LM Studio install says `ollama_reachable: false` about a
        # daemon nobody is using and nothing at all about the one that is —
        # which is the report we would be asked to diagnose from.
        'local_llm': {'provider': _vision_llm.provider()},
        'lmstudio': capabilities.lmstudio_diagnostic(),
        # Last failed-generation reason per engine + the last Studio failure — the
        # real cause behind "it fails every time". Redacted + capped (see helper).
        'generation_errors': _recent_generation_errors(),
        # Last ERROR/CRITICAL records WITH their tracebacks (not just the raw tail),
        # so the stack is in the report instead of asked for in a follow-up. Like
        # log_tail it is log-derived and may still cite non-home file names.
        'error_log': _error_log_records(),
        # One-shot boot pass that reconnects continuations trained before their
        # lineage edge was persisted: how many edges it reconstructed (0 on a fresh
        # or fully-native database). Paste-safe — counts only, no paths.
        'lineage_backfill': _lineage_backfill.summary(),
        # Same idea for the images promoted from a bank before the promotion
        # carried their framing: how many rows the one-shot pass gave back to the
        # Composition tally (0 on a fresh database). Counts only, no paths.
        'framing_backfill': _framing_backfill.summary(),
        'log_tail': log_lines,
        'generated_at': int(time.time()),
    })


# --- ChatGPT subscription (Codex OAuth) --------------------------------------
# Device-code login for the ChatGPT engine's subscription lane. One upstream
# check per poll call — the SPA polls every few seconds, no server thread.

@bp.post('/settings/chatgpt-oauth/start')
def chatgpt_oauth_start():
    from ..services import chatgpt_oauth
    out = chatgpt_oauth.login_start()
    return jsonify(out), (200 if out.get('ok') else 502)


@bp.get('/settings/chatgpt-oauth/poll')
def chatgpt_oauth_poll():
    from ..services import chatgpt_oauth
    return jsonify(chatgpt_oauth.login_poll())


@bp.post('/settings/chatgpt-oauth/import-codex')
def chatgpt_oauth_import_codex():
    from ..services import chatgpt_oauth
    out = chatgpt_oauth.import_codex_cli()
    return jsonify(out), (200 if out.get('ok') else 404)


@bp.post('/settings/chatgpt-oauth/logout')
def chatgpt_oauth_logout():
    from ..services import chatgpt_oauth
    chatgpt_oauth.logout()
    return jsonify({'ok': True})
