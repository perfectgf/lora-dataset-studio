import sys, os


def _reexec_into_venv():
    """Run on the project's pinned interpreter, not whatever Python launched us.

    If a project .venv exists and we are not already its interpreter, re-exec
    into it before anything else imports. This makes every launch method — the
    start.bat/start.sh flow, a bare `python backend/run.py`, a double-click, an
    IDE, a shell with a newer Python first on PATH — converge on the SAME
    interpreter. That is what lets the optional ML extras (insightface / numpy<2
    / onnxruntime, which only publish wheels for CPython 3.10-3.12) install into
    a supported Python: the in-app installer and the capability probes both key
    off sys.executable, so if run.py runs on e.g. the machine's default 3.14 the
    extras can never install. Skipped for the frozen/portable build (it bundles
    its own Python) and once we are already the venv's python. Set
    LDS_NO_REEXEC=1 to opt out."""
    if getattr(sys, 'frozen', False) \
            or os.environ.get('LDS_REEXEC') == '1' \
            or os.environ.get('LDS_NO_REEXEC') == '1':
        return
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in (('.venv', 'Scripts', 'python.exe'), ('.venv', 'bin', 'python')):
        venv_py = os.path.join(repo_root, *rel)
        if os.path.exists(venv_py):
            break
    else:
        return                                   # no venv -> nothing to switch to
    try:
        if os.path.samefile(venv_py, sys.executable):
            return                               # already the venv interpreter
    except OSError:
        if os.path.normcase(os.path.realpath(venv_py)) \
                == os.path.normcase(os.path.realpath(sys.executable)):
            return
    os.environ['LDS_REEXEC'] = '1'               # loop guard for the re-exec'd child
    print(f"[LDS] re-launching under the project venv: {venv_py}", flush=True)
    os.execv(venv_py, [venv_py, os.path.abspath(__file__), *sys.argv[1:]])


_reexec_into_venv()

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import create_app

try:
    from app.config import get as cfg_get
except ImportError:
    cfg_get = lambda k, d=None: {'server.host': '127.0.0.1', 'server.port': 5000}.get(k, d)

app = create_app()

if __name__ == '__main__':
    host = os.environ.get('LDS_HOST') or cfg_get('server.host')
    port = int(os.environ.get('LDS_PORT') or cfg_get('server.port'))
    # Opening the bind beyond loopback exposes an unauthenticated single-user app
    # (API keys, GPU, datasets) to the network → make sure the token guard
    # (app/netguard.py) has a token to check, and tell the user how to connect.
    if host not in ('127.0.0.1', 'localhost', '::1') \
            and not os.environ.get('LDS_ACCESS_TOKEN') \
            and os.environ.get('LDS_ALLOW_UNAUTHENTICATED') != '1':
        # Persisted in config.json (not just this process's env) so the token
        # survives a restart instead of rotating every boot -- the Settings
        # "Server" card reads it back from there to show/copy it.
        token = cfg_get('server.access_token') or ''
        if not token:
            import secrets
            token = secrets.token_urlsafe(24)
            try:
                from app.config import save_config
                save_config({'server': {'access_token': token}})
            except ImportError:
                pass   # config module unavailable (see cfg_get fallback above) -> ephemeral this run
        os.environ['LDS_ACCESS_TOKEN'] = token
        print(f"\n[LDS] server.host={host} is reachable from the network -> access token enabled.")
        print(f"[LDS] Open from another device:  http://<this-machine>:{port}/?token={os.environ['LDS_ACCESS_TOKEN']}")
        print("[LDS] (set LDS_ACCESS_TOKEN yourself for a stable token, or "
              "LDS_ALLOW_UNAUTHENTICATED=1 if the network is already locked down)\n")
    # Snapshot of what's ACTUALLY bound, for the Settings "Server" card: config.json
    # may already hold newer values the user saved but hasn't restarted into yet, so
    # reading cfg_get again there would lie about what's currently serving requests.
    app.config['LDS_BOUND_HOST'] = host
    app.config['LDS_BOUND_PORT'] = port
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1',
            host=host,
            # LDS_PORT wins over config so the launcher can dodge a busy 5000
            # (macOS AirPlay, another Flask app, …) without touching config.json.
            port=port, threaded=True, use_reloader=False)
