import atexit
import logging
import sys, os
import threading
import time
import urllib.request
import webbrowser


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
from bootstrap_dependencies import ensure_pillow_consistent

# Must run before importing ``app`` (which eventually imports PIL).  This fixes
# Windows installs left half-upgraded by versions of the in-app updater that ran
# pip while Pillow files were still loaded and locked by the Flask process.
ensure_pillow_consistent()

from app import create_app
from port_utils import find_available_port
from single_instance import live_instance, refusal_message, release_lock, write_lock

try:
    from app.config import get as cfg_get
except ImportError:
    cfg_get = lambda k, d=None: {'server.host': '127.0.0.1', 'server.port': 5000}.get(k, d)

app = create_app()


def _announce_when_ready(url, open_browser=False, timeout=180):
    """Print the address the app is actually serving on — and open the browser
    when asked — once that address answers.

    The URL has to be printed HERE because Werkzeug's own " * Running on ..."
    banner never reaches the terminal: ``create_app`` attaches a rotating file
    handler to the ROOT logger, so werkzeug's INFO-level banner lands in
    ``data/app.log`` instead of stdout. A plain ``python backend/run.py`` used to
    print no address at all, and any launcher that reads the terminal for one
    (the Pinokio launcher does, to light up its "Open Web UI" tab) would wait
    forever. Waiting for /api/health first means the line — and the browser tab —
    appear when the app can actually answer, not on a startup error page.

    On timeout the address is printed anyway: a slow first boot must not leave a
    launcher hanging on a line that never comes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + 'api/health', timeout=1) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.25)
    print(f"[LDS] Ready on {url}", flush=True)
    if open_browser:
        webbrowser.open(url)

if __name__ == '__main__':
    host = os.environ.get('LDS_HOST') or cfg_get('server.host')
    requested_port = int(os.environ.get('LDS_PORT') or cfg_get('server.port'))
    # One data folder, one server — checked BEFORE the port slide below, which
    # is exactly how a double-launch used to become a second server on :5051
    # sharing the first one's database (private in-memory job registries, a
    # pass running in one process while the other swore the bank was idle).
    # Instances on their OWN data folder (worktrees, proof instances with
    # LDS_DATA_DIR) are untouched; LDS_ALLOW_SECOND_INSTANCE=1 overrides.
    data_dir = app.config['LDS_DATA_DIR']
    running = live_instance(data_dir)
    if running:
        print(refusal_message(running), flush=True)
        if os.environ.get('LDS_OPEN_BROWSER') == '1':
            # The double-click case: the person wanted the app on screen, and
            # it exists already — open THAT one instead of printing at them.
            webbrowser.open(f"http://127.0.0.1:{running['port']}/")
        sys.exit(0)
    port = (requested_port if os.environ.get('LDS_AUTO_PORT') == '0'
            else find_available_port(host, requested_port))
    if port != requested_port:
        print(f"[LDS] port {requested_port} is already in use; using {port} instead.",
              flush=True)
    os.environ['LDS_PORT'] = str(port)
    is_lan = host not in ('127.0.0.1', 'localhost', '::1')
    from app import netguard
    access_token = netguard.ensure_access_token(host)
    if access_token:
        why = ('LDS_PUBLIC=1 -> this bind is reachable from the internet'
               if netguard.public_bind() else f'server.host={host} reachable from the network')
        print(f"\n[LDS] {why} -> access token REQUIRED.")
        print(f"[LDS] Open with:  /?token={access_token}")
        if not netguard.public_bind():
            print("[LDS] (turn the token off in Settings -> Server to open the LAN without one)")
        print()
    elif is_lan:
        print(f"\n[LDS] server.host={host} reachable from the network (no token — trusted-LAN mode).")
        print(f"[LDS] Open from another device:  http://<this-machine>:{port}/\n")
    # Snapshot of what's ACTUALLY bound, for the Settings "Server" card: config.json
    # may already hold newer values the user saved but hasn't restarted into yet, so
    # reading cfg_get again there would lie about what's currently serving requests.
    app.config['LDS_BOUND_HOST'] = host
    app.config['LDS_BOUND_PORT'] = port
    # Claim the data folder only once the port is settled, so the lock records
    # the address the next double-launch should be pointed at. Released on
    # clean exit; a crash leaves it behind, where the dead pid reads as stale.
    write_lock(data_dir, host, port)
    atexit.register(release_lock, data_dir)
    local_host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(host, host)
    if ':' in local_host and not local_host.startswith('['):
        local_host = f'[{local_host}]'
    url = f"http://{local_host}:{port}/"
    # Refresh the ComfyUI nodes this app ships, for users who already installed
    # them. "Update & restart" replaces the app's files and nothing else — it has
    # no business writing into somebody's ComfyUI on its own — so without this a
    # user would keep the version of the node they first clicked while the app's
    # graphs moved on. Only STALE copies are touched: an absent one stays absent,
    # because installing is the user's decision and refreshing what they already
    # chose is not a new one.
    #
    # Here rather than in create_app(): the test suite builds hundreds of apps,
    # and none of them should be writing into a real ComfyUI folder.
    try:
        from app import setup_installer
        for action, message in setup_installer.refresh_bundled_node_packs().items():
            print(f"[LDS] {action}: {message} — restart ComfyUI to load it.")
    except Exception:                       # noqa: BLE001 — never block boot
        logging.getLogger(__name__).warning('bundled node refresh failed', exc_info=True)
    threading.Thread(target=_announce_when_ready, args=(url,),
                     kwargs={'open_browser': os.environ.get('LDS_OPEN_BROWSER') == '1'},
                     daemon=True).start()
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1',
            host=host,
            port=port, threaded=True, use_reloader=False)
