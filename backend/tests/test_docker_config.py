"""Static contracts shared by the app and its development entrypoints."""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

from app.config import DEFAULTS


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding='utf-8')


def _docker_env(dockerfile):
    """Return ENV assignments regardless of single or continued-line layout."""
    logical_lines = dockerfile.replace('\\\n', ' ')
    assignments = {}
    for line in logical_lines.splitlines():
        if not line.startswith('ENV '):
            continue
        for key, value in re.findall(r'([A-Z][A-Z0-9_]*)=([^\s]+)', line[4:]):
            assignments[key] = value
    return assignments


def _shell_statements(script):
    """A shell script's executable lines: comments and blanks dropped."""
    return [line.strip() for line in script.splitlines()
            if line.strip() and not line.strip().startswith('#')]


def _load_script_module(name):
    """Import one of packaging/docker's standalone boot scripts by path. They sit
    outside the `app` package on purpose — they run under the container's system
    python, before any venv is active — so there is no import path to them."""
    path = REPO_ROOT / 'packaging' / 'docker' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'lds_docker_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_container_runtime_tracks_server_defaults():
    """The image must work both directly and through Docker Compose."""
    server = DEFAULTS['server']
    port = server['port']
    dockerfile = (REPO_ROOT / 'Dockerfile').read_text(encoding='utf-8')
    compose = _read('docker-compose.yml')
    image_env = _docker_env(dockerfile)

    assert image_env['LDS_DATA_DIR'] == '/data'
    assert image_env['LDS_CONFIG'] == '/data/config.json'
    assert image_env['LDS_HOST'] == '0.0.0.0'
    assert image_env['LDS_PORT'] == str(port)
    assert f'EXPOSE {port}' in dockerfile
    assert f'http://127.0.0.1:{port}/api/health' in dockerfile
    assert f'- target: {port}' in compose
    assert f'published: "${{LDS_HOST_PORT:-{port}}}"' in compose
    assert 'protocol: tcp' in compose
    assert f'LDS_PORT={port}' in compose
    assert 'LDS_HOST=0.0.0.0' in compose
    assert 'LDS_CONFIG=/data/config.json' in compose


def test_developer_entrypoints_track_server_default():
    """Examples and the Vite proxy must follow the backend's real default port."""
    port = DEFAULTS['server']['port']
    example = json.loads(_read('config.example.json'))
    vite = _read('frontend/vite.config.js')

    assert example['server']['port'] == port

    # The proxy target is no longer a literal on the '/api' line — it is an
    # env-var override falling back to a named default, so that `npm run dev`
    # stops driving the real install by accident. What must still hold is the
    # thing this test was written for: that FALLBACK has to track the backend's
    # own default port, or the habit ("npm run dev", hit :5173) silently breaks.
    assert re.search(r"['\"]\/api['\"]\s*:", vite), 'Vite must declare an /api proxy'
    default = re.search(
        r"DEFAULT_DEV_API_TARGET\s*=\s*['\"]http:\/\/127\.0\.0\.1:(\d+)", vite)
    assert default, 'Vite must keep a named loopback default for /api'
    assert int(default.group(1)) == port


def test_docker_context_excludes_generated_artifacts():
    ignored = {
        line.strip().rstrip('/')
        for line in _read('.dockerignore').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    }

    assert {'.worktrees', '.pytest_cache', 'packaging/build', 'packaging/dist', 'run', 'basedir',
            'data-docker-gpu', 'bank-images'} <= ignored


def test_git_ignores_the_docker_host_folders():
    """`docker compose -f docker-compose.gpu.yml up` creates ./run, ./basedir and
    ./data-docker-gpu in the repo root, and after one boot they hold ComfyUI's venv
    and model tree. Untracked, that is tens of GB one `git add -A` away from the
    repo. ./data-docker is the API-only stack's own separate folder, checked here
    too since both stacks live in the same repo root. ./bank-images is the worst
    of the set to get wrong: whatever a user drops in there to triage is by
    definition personal photographs, and this repo is public."""
    for folder in ('run', 'basedir', 'data-docker', 'data-docker-gpu', 'bank-images'):
        result = subprocess.run(
            ['git', 'check-ignore', '-q', f'{folder}/'],
            cwd=REPO_ROOT, capture_output=True)
        assert result.returncode == 0, f'{folder}/ is not gitignored'


def test_launcher_can_never_abort_the_upstream_boot():
    """Upstream's run_userscript does `$script || error_exit`, so a non-zero exit
    from the launcher kills the container's ComfyUI too. A studio problem must
    cost the studio only."""
    script = _read('packaging/docker/studio_launch.sh')
    statements = _shell_statements(script)

    assert script.startswith('#!/bin/bash')
    # The last thing the script DOES, not the last bytes of the file: a trailing
    # comment reading "exit 0" would satisfy endswith() and prove nothing.
    assert statements[-1] == 'exit 0'
    # Prose is free to name `set -e`; only a statement that enables it is the defect.
    assert not [line for line in statements if line.startswith('set -e')]
    assert '/app/.venv/bin/python' in script
    assert 'seed_comfy_config.py' in script
    # The studio is a background job beside ComfyUI's foreground process, so the
    # launcher has to be its own supervisor.
    assert [line for line in statements if line.startswith('while true')]
    # ./.env is bind-mounted at /app/.env, so a chown -R of /app walks into the
    # mount and re-owns the user's real API-key file on the host.
    assert not [line for line in statements
                if re.search(r'chown\s+-R\s+\S+\s+"?\$\{?STUDIO_DIR\}?"?\s*$', line)]
    assert '/app/.venv' in script or '${STUDIO_DIR}/.venv' in script


def test_launcher_creates_a_missing_data_dir_before_judging_it():
    """A bind mount always exists; a path NESTED inside a fresh volume does not.
    On a rented pod LDS_DATA_DIR points inside the network volume, nothing on the
    host ever created it, and `[ ! -w ]` on a missing directory is true — so the
    studio refused to start and only ComfyUI answered. Create it, then judge it."""
    script = _read('packaging/docker/studio_launch.sh')

    create = re.search(r'mkdir\s+-p\s+"\$\{DATA_DIR\}"', script)
    assert create, 'the launcher must create DATA_DIR when it is missing'
    # Ordering is the whole point: creating it after the check fixes nothing.
    assert create.start() < script.index('if [ ! -w "${DATA_DIR}" ]')
    # It must not be able to abort the boot — upstream runs this as `$script ||
    # error_exit`, and the writability branch below already reports the failure.
    line = script[create.start():script.index('\n', create.start())]
    assert '||' in line, 'a failing mkdir must stay non-fatal'


def test_launcher_preserves_backend_exit_status_and_restart_pacing():
    """The logging pipe must not turn every backend exit into sed's status 0.
    Code 75 is the intentional Settings restart and must be immediate; a crash is
    delayed so a broken install cannot become a hot respawn loop."""
    script = _read('packaging/docker/studio_launch.sh')

    capture = script.index('studio_status=${PIPESTATUS[0]}')
    deliberate = script.index('if [ "${studio_status}" -eq 75 ]', capture)
    immediate = script.index('continue', deliberate)
    delay = script.index('sleep 10', immediate)

    assert capture < deliberate < immediate < delay


def test_docker_boot_scripts_are_pinned_to_lf_and_launcher_has_no_crlf():
    attributes = _read('.gitattributes')
    launcher = (REPO_ROOT / 'packaging' / 'docker' / 'studio_launch.sh').read_bytes()

    assert 'packaging/docker/*.sh text eol=lf' in attributes
    assert b'\r\n' not in launcher


def test_healthcheck_covers_both_halves_of_the_gpu_image(monkeypatch):
    """One container, two services: a live ComfyUI with a dead studio is a broken
    stack, and Docker only gets one exit code to say so in. Imported rather than
    grepped, so an endpoint only counts if it is actually wired into TARGETS."""
    monkeypatch.delenv('LDS_PORT', raising=False)
    healthcheck = _load_script_module('healthcheck')
    targets = dict(healthcheck.TARGETS)

    assert f":{DEFAULTS['server']['port']}/api/health" in targets['studio']
    assert ':8188/system_stats' in targets['comfyui']


def test_gpu_image_layers_on_the_comfyui_base_without_hijacking_it():
    """Dockerfile.gpu is a layer on someone else's image. Three of its rules are
    invisible in a diff and fatal at runtime, so they are asserted here."""
    dockerfile = _read('Dockerfile.gpu')
    image_env = _docker_env(dockerfile)
    port = DEFAULTS['server']['port']

    assert 'mmartial/comfyui-nvidia-docker' in dockerfile
    assert image_env['LDS_DATA_DIR'] == '/data'
    assert image_env['LDS_CONFIG'] == '/data/config.json'
    assert image_env['LDS_HOST'] == '0.0.0.0'
    assert image_env['LDS_PORT'] == str(port)
    assert image_env['LDS_RUNTIME'] == 'docker-gpu'
    assert image_env['LDS_RESTART_MODE'] == 'supervisor'
    assert image_env['LDS_BIND_MANAGED'] == '1'
    assert f'EXPOSE {port}' in dockerfile
    assert 'EXPOSE 8188' in dockerfile

    logical = [line.strip()
               for line in dockerfile.replace('\\\n', ' ').splitlines()
               if line.strip()]

    # 1. Upstream's ENTRYPOINT (/comfyui-nvidia_init.bash) must stay in charge.
    assert not [line for line in logical
                if line.startswith('ENTRYPOINT') or line.startswith('CMD ')]
    # 2. The container starts as comfytoo and upstream's init script remaps comfy
    #    to WANTED_UID before switching to it.
    assert logical[-1] == 'USER comfytoo'
    # 3. ComfyUI's venv activation must keep winning; the studio is only ever
    #    invoked through absolute paths.
    assert not [line for line in logical if line.startswith('ENV PATH')]

    # /userscripts_dir/*.sh run in "skip" mode: not executable means not run.
    assert 'install -D -m 755 packaging/docker/studio_launch.sh' in dockerfile
    assert '/userscripts_dir/50-lora-dataset-studio.sh' in dockerfile
    # The studio's venv must never be ComfyUI's.
    assert '/app/.venv' in dockerfile

    # A build-time `chown -R /app` rewrites every file's metadata and so copies the
    # whole tree — torch included — into a second layer: 8.66 GB and 407 s, for an
    # ownership the runtime almost never wants. Measured, not guessed (the image's
    # total size moves as layers change, so it is not cited here).
    assert not re.search(r'chown\s+-R\s+\S+\s+/app\b', dockerfile)


def test_gpu_compose_publishes_both_uis_and_reserves_the_gpu():
    compose = _read('docker-compose.gpu.yml')
    port = DEFAULTS['server']['port']

    assert 'dockerfile: Dockerfile.gpu' in compose
    assert f'- target: {port}' in compose
    assert f'published: "${{LDS_HOST_PORT:-{port}}}"' in compose
    assert '- target: 8188' in compose
    assert 'published: "${LDS_COMFY_HOST_PORT:-8188}"' in compose
    assert compose.count('protocol: tcp') >= 2
    for mount in (':/comfy/mnt', ':/basedir', ':/data', ':/images', './.env:/app/.env'):
        assert mount in compose, mount
    assert 'driver: nvidia' in compose
    assert re.search(r'capabilities:\s*\[\s*gpu', compose)
    assert 'WANTED_UID=' in compose and 'WANTED_GID=' in compose
    assert f'LDS_PORT={port}' in compose
    assert 'LDS_HOST=0.0.0.0' in compose
    assert 'LDS_CONFIG=/data/config.json' in compose
    assert 'LDS_RUNTIME=docker-gpu' in compose
    assert 'LDS_RESTART_MODE=supervisor' in compose
    assert 'LDS_BIND_MANAGED=1' in compose
    assert 'LDS_FORCE_CHOWN=${LDS_FORCE_CHOWN:-false}' in compose

    # FORCE_CHOWN is intentionally limited to the app-data bind. The launcher
    # must never recursively adopt ComfyUI's model/run trees or the image bank.
    launcher = _read('packaging/docker/studio_launch.sh')
    force_block = launcher[launcher.index('if [ "${LDS_FORCE_CHOWN'):
                           launcher.index('/usr/bin/python3')]
    assert '"${DATA_DIR}"' in force_block
    # The parent bind can be writable while files created by a previous root
    # container below it are not.  The explicit opt-in must therefore chown the
    # full tree without using the parent's writability as a gate.
    force_condition = re.search(
        r'if\s+\[\s+"\$\{LDS_FORCE_CHOWN:-false\}"\s+=\s+"true"\s+\];\s+then',
        force_block,
    )
    assert force_condition, 'LDS_FORCE_CHOWN must be the recursive chown gate'
    chown = force_block.index('sudo chown -R', force_condition.end())
    post_chown_check = force_block.index('if [ ! -w "${DATA_DIR}" ]', chown)
    assert '! -w "${DATA_DIR}"' not in force_block[:chown]
    assert force_condition.end() < chown < post_chown_check
    assert '/basedir' not in force_block
    assert '/comfy/mnt' not in force_block
    assert '/images' not in force_block

    # Its own compose project, or `up` here recreates docker-compose.yml's container:
    # both files call the service `studio`.
    assert re.search(r'^name:\s*lora-dataset-studio-gpu\s*$', compose, re.M)

    # LDS_PORT is the port the app BINDS INSIDE the container. Interpolating it on
    # the host side of a mapping would publish a port nothing serves.
    assert '${LDS_PORT' not in compose

    # An explicit resolver, because the one a Compose network inherits from the
    # host is often unreachable from inside the container — and this image fetches
    # from astral.sh on every start, so an unresolvable name is a restart loop.
    # Overridable, since setting it REPLACES the inherited list.
    assert 'dns: ${LDS_DNS:-' in compose


def test_gpu_compose_resource_caps_default_to_no_limit():
    """The caps are for people who ask for them. Shipping a real number as the
    default would throttle — or OOM-kill — every install that never did, and the
    first boot (torch plus ComfyUI's dependency tree) is exactly where a tight cap
    stops looking like slowness and starts looking like a broken image. 0 is
    Docker's own "no limit" for memory and cpus, and for memswap_limit it means
    twice mem_limit."""
    compose = _read('docker-compose.gpu.yml')

    assert 'mem_limit: ${LDS_MEM_LIMIT:-0}' in compose
    assert 'memswap_limit: ${LDS_MEMSWAP_LIMIT:-0}' in compose
    assert 'cpus: ${LDS_CPUS:-0}' in compose


def test_every_gpu_compose_variable_is_documented():
    """.env.example is where these are discovered — the compose file is not read
    by most people who deploy it. A variable that only exists in the YAML is a
    knob nobody knows about."""
    compose = _read('docker-compose.gpu.yml')
    documented = _read('.env.example')
    interpolated = set(re.findall(r'\$\{(LDS_[A-Z0-9_]+)', compose))

    # LDS_PORT and friends are the container's own environment, set in the file
    # itself rather than read from .env.
    undocumented = sorted(name for name in interpolated if name not in documented)
    assert not undocumented, f'not in .env.example: {undocumented}'
