"""Settings API: config/secrets CRUD + capability probes."""
from flask import Blueprint, jsonify, request

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


def _settings_payload() -> dict:
    return {'config': cfg.load_config(), 'secrets': _secret_presence()}


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
_UPDATE_TTL = 6 * 3600
_update_cache = {'ts': 0.0, 'data': None}


@bp.get('/update/check')
def update_check():
    import time
    import requests
    from ..version import APP_VERSION
    from ..services import updater
    force = bool(request.args.get('force'))
    # A git checkout: the meaningful signal is commits-behind-origin (the user pushes
    # commits to a branch, not tagged releases — a release-only check reads "up to date"
    # while the tree is many commits behind). Only fetch on an explicit check (force),
    # never from the passive startup banner, which must not hit the network every load.
    if force and updater.is_git_checkout():
        gs = updater.git_update_status()
        if gs is not None:
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
        # invalidate the cached release check so the banner re-evaluates post-update
        _update_cache.update(ts=0.0, data=None)
        updater.schedule_restart()
    return jsonify(res)


@bp.get('/logs/tail')
def logs_tail():
    """Last N lines of the server log for the in-app viewer — so a novice can
    copy-paste an error instead of hunting for files. Reads data/app.log (the
    app's own rotating log), falling back to data/server.log (the portable
    launcher's raw stdout capture)."""
    import os
    from pathlib import Path
    try:
        n = max(10, min(1000, int(request.args.get('n', 300))))
    except ValueError:
        n = 300
    data_dir = Path(os.environ.get('LDS_DATA_DIR', str(cfg.REPO_ROOT / 'data')))
    for name in ('app.log', 'server.log'):
        p = data_dir / name
        if p.is_file():
            try:
                size = p.stat().st_size
                with open(p, encoding='utf-8', errors='replace') as fh:
                    if size > 512 * 1024:               # tail window, never the whole file
                        fh.seek(size - 512 * 1024)
                    lines = fh.read().splitlines()[-n:]
                return jsonify({'ok': True, 'file': name, 'lines': lines})
            except OSError:
                continue
    return jsonify({'ok': True, 'file': None, 'lines': []})


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
