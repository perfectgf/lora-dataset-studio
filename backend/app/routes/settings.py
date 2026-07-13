"""Settings API: config/secrets CRUD + capability probes."""
from flask import Blueprint, current_app, jsonify, request

from .. import capabilities
from .. import config as cfg

bp = Blueprint('settings', __name__, url_prefix='/api')

_TEST_TARGETS = {
    'gemini': capabilities.probe_gemini,
    'openai': capabilities.probe_openai,
    'comfyui': capabilities.probe_comfyui,
    'ollama': capabilities.probe_ollama,
    'aitoolkit': capabilities.probe_aitoolkit,
    'face_scoring': capabilities.probe_face_scoring,
    'masks': capabilities.probe_masks,
    'vast': capabilities.probe_vast,
}


def _secret_presence() -> dict:
    return {name: bool(cfg.secret(name)) for name in cfg.SECRET_KEYS}


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
    return {
        'config': cfg.load_config(), 'secrets': _secret_presence(),
        # What THIS running process is actually bound to — run.py stamps these
        # before app.run(); a dev/test boot that never went through run.py (or a
        # WSGI launch) leaves them unset, so the Server card just hides the
        # "running vs saved" diff instead of showing a misleading n/a:n/a.
        'runtime': {'host': current_app.config.get('LDS_BOUND_HOST'),
                    'port': current_app.config.get('LDS_BOUND_PORT'),
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


@bp.put('/settings')
def put_settings():
    body = request.get_json(force=True, silent=True) or {}
    if 'config' in body and not isinstance(body['config'], dict):
        return jsonify({'error': "'config' must be an object"}), 400
    if 'secrets' in body and not isinstance(body['secrets'], dict):
        return jsonify({'error': "'secrets' must be an object"}), 400
    config_partial = body.get('config') or {}
    unknown = set(config_partial) - set(cfg.DEFAULTS)
    if unknown:
        return jsonify({'error': f"unknown config section '{sorted(unknown)[0]}'"}), 400
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
    cfg.save_config(config_partial)
    cfg.set_secrets(body.get('secrets') or {})
    # A changed ComfyUI location must take effect NOW: the base/model listers cache
    # their scans for 5 min, so without this the training-base dropdowns keep showing
    # the pre-save (often empty) list right after the user points the app at ComfyUI.
    # (The wizard's _scan_models view refreshes via the frontend's forced
    # /api/capabilities?force=1 call, so no probe(force) is needed here.)
    if 'comfyui' in config_partial:
        from ..utils import comfyui
        comfyui.clear_model_caches()
    return jsonify(_settings_payload())


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


@bp.post('/settings/test/<target>')
def test_connection(target):
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
    # A git checkout: the meaningful signal is commits-behind-origin (the user pushes
    # commits to a branch, not tagged releases — a release-only check reads "up to date"
    # while the tree is many commits behind). The fetch runs on an explicit check
    # (force, always fresh) or an auto check (nav badge — served from a TTL cache so
    # SPA loads don't hammer the network); never from the bare passive path.
    if (force or auto) and updater.is_git_checkout():
        now = time.time()
        if auto and not force and _git_check_cache['data'] is not None \
                and (now - _git_check_cache['ts']) < _GIT_CHECK_TTL:
            return jsonify(_git_check_cache['data'])
        gs = updater.git_update_status()
        if gs is not None:
            _git_check_cache.update(ts=now, data=gs)
            return jsonify(gs)
    now = time.time()
    if (_update_cache['data'] is not None and (now - _update_cache['ts']) < _UPDATE_TTL
            and not force):
        return jsonify(_update_cache['data'])
    repo = cfg.get('updates.repo') or 'perfectgf/lora-dataset-studio'
    out = {'ok': True, 'current': APP_VERSION, 'latest': None,
           'update_available': False, 'url': f'https://github.com/{repo}/releases'}
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
        else:
            out['reason'] = (f'release feed answered {r.status_code} '
                             '(no public release yet?)')
    except requests.RequestException:
        out['reason'] = 'offline or GitHub unreachable'
    _update_cache.update(ts=now, data=out)
    return jsonify(out)


@bp.post('/update/apply')
def update_apply():
    """Pull the latest commits (git checkout only) and, if anything changed, restart the
    server. Returns immediately with {ok, changed, from, to, restarting, log, ...}; the
    actual re-launch happens ~1 s after this response flushes, so the client can start
    polling /api/health. A packaged build (no git) gets {manual:true, url} instead."""
    from ..services import updater
    res = updater.apply_update()
    res['restarting'] = bool(res.get('ok') and res.get('changed'))
    if res['restarting']:
        # invalidate the cached checks so the banner/badge re-evaluate post-update
        _update_cache.update(ts=0.0, data=None)
        _git_check_cache.update(ts=0.0, data=None)
        updater.schedule_restart()
    return jsonify(res)


@bp.post('/settings/restart')
def settings_restart():
    """Manual restart — used after saving server.host/server.port (a live bind
    change needs a fresh process; Flask can't rebind mid-request) and as a plain
    troubleshooting action. Same schedule_restart() as the updater, so it
    survives both a git checkout and the packaged build.

    Pins the restarted process to the SAVED host/port via env: the launcher
    (start.bat) exports LDS_PORT, which otherwise wins over config.json forever
    — so without this, changing the port in Settings + restart would keep coming
    back on the launcher's port and the field would look broken. schedule_restart
    passes os.environ down to the relaunch, so setting it here is what makes the
    saved port actually take effect."""
    import os
    from ..services import updater
    os.environ['LDS_HOST'] = str(cfg.get('server.host') or '127.0.0.1')
    os.environ['LDS_PORT'] = str(cfg.get('server.port') or 5050)
    updater.schedule_restart()
    return jsonify({'ok': True, 'restarting': True})


@bp.get('/trash')
def trash_info():
    """Trash size for the Settings card — everything the app 'deletes' lands
    there; only 'Empty trash' below actually destroys bytes."""
    from ..services import trash
    return jsonify({'size_bytes': trash.trash_size()})


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
    import platform
    import sys
    import time
    from ..version import APP_VERSION
    from ..services import updater
    conf = cfg.load_config()
    caps = capabilities.probe()
    e = caps.get('engines') or {}
    comfy = caps.get('comfyui') or {}
    oll = caps.get('ollama') or {}
    _, log_lines = _log_tail_lines(80)
    return jsonify({
        'app_version': APP_VERSION,
        'git_sha': updater.current_sha(),
        'os': f'{platform.system()} {platform.release()}',
        'python': sys.version.split()[0],
        'secrets_present': _secret_presence(),
        'capabilities': {
            'engines': {'nanobanana': bool(e.get('nanobanana')),
                        'chatgpt': bool(e.get('chatgpt')),
                        'klein': bool(e.get('klein'))},
            'comfyui_reachable': bool(comfy.get('reachable')),
            'klein_model': bool((comfy.get('models') or {}).get('klein')),
            'ollama_reachable': bool(oll.get('reachable')),
            'vision_model_ready': bool(oll.get('vision_model_ready')),
            'face_scoring': bool(caps.get('face_scoring')),
            'masks': bool(caps.get('masks')),
            'aitoolkit_valid': bool((caps.get('aitoolkit') or {}).get('valid')),
            'training_visible': bool(caps.get('training_visible')),
            'studio_visible': bool(caps.get('studio_visible')),
            'cloud_training': bool(caps.get('cloud_training')),
        },
        'config': {
            'captioning_backend': (conf.get('captioning') or {}).get('backend'),
            'default_engine': (conf.get('engines') or {}).get('default'),
            'enabled_engines': (conf.get('engines') or {}).get('enabled'),
            'training_default_family': (conf.get('training') or {}).get('default_family'),
            'comfyui_base_dir_set': bool((conf.get('comfyui') or {}).get('base_dir')),
            'aitoolkit_dir_set': bool((conf.get('aitoolkit') or {}).get('dir')),
            'lan_enabled': (conf.get('server') or {}).get('host') not in (None, '', '127.0.0.1', 'localhost', '::1'),
        },
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
