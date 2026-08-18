"""An ML capability install that does not actually work must not report success.

Reported by 1Tomber (GitHub #24) on Linux/Python 3.12: installing "Person masks"
from Setup ▸ Install/repair individually reported success — every requirement
"already satisfied" — while the capability stayed ✗ Not installed and masked
training silently degraded to unmasked. Three defects stacked:

  1. the masks install set omitted onnxruntime, which rembg ≥2.0.50 imports at
     load but no longer declares;
  2. the install reported success without ever re-checking the capability, so the
     app said one thing while knowing another — and the import traceback, which
     names the missing module, was thrown away;
  3. the resulting run trained unmasked with no pre-launch warning.

These tests pin all three, plus the constraint that the fix must not reinstall
onnxruntime on the very many machines that already have one — least of all
replace a user's onnxruntime-gpu with the CPU build.

No test here touches the real network or a real pip: every subprocess seam is
stubbed.
"""
import subprocess

import pytest


@pytest.fixture(autouse=True)
def _fresh_encoder_verdict():
    """The ffmpeg verdict is process-cached; these tests stub it, so it must not
    survive them."""
    from app.services import ffmpeg_tools
    ffmpeg_tools.clear_cache()
    yield
    ffmpeg_tools.clear_cache()


# --- 1. the missing dependency ------------------------------------------------

def test_masks_install_set_carries_onnxruntime():
    """rembg imports onnxruntime at module load and stopped declaring it, so the
    scoped masks install has to name it or the capability cannot come up."""
    from app import setup_installer
    canon = {setup_installer._canon(p)
             for p in setup_installer._CAPABILITY_PACKAGES['masks']}
    assert 'onnxruntime' in canon


def test_every_capability_install_set_covers_its_own_probe_imports():
    """The family guard. A capability whose probe is a FULL import of module X
    while X is absent from its install set is exactly the #24 shape, and it would
    fail the same way. Checked for the top-level modules the probes import that
    the ML requirements file knows as packages."""
    from app import setup_installer, capabilities
    # module name -> the distribution that provides it, for the ML packages whose
    # import name differs from (or matches) the distribution name.
    provider = {'rembg': 'rembg', 'onnxruntime': 'onnxruntime',
                'insightface': 'insightface', 'cv2': 'opencv-python-headless',
                'numpy': 'numpy', 'simple_lama_inpainting': 'simple-lama-inpainting',
                # 🎬 video lane — the import names differ from the distributions,
                # which is exactly the drift this guard exists to catch.
                'av': 'av', 'transnetv2_pytorch': 'transnetv2-pytorch'}
    for action, pkgs in setup_installer._CAPABILITY_PACKAGES.items():
        expr = capabilities.CAPABILITY_IMPORTS.get(action)
        if not expr:
            continue
        owned = {setup_installer._canon(p) for p in pkgs}
        for mod in expr.replace('import', '').replace(',', ' ').split():
            dist = provider.get(mod)
            if dist is None:
                continue
            assert setup_installer._canon(dist) in owned, (
                f'{action}: its probe imports {mod} but its install set does not '
                f'carry {dist} — an install would report success and the '
                f'capability would stay off (the #24 defect)')


# --- 2. an install that does not work must not say it worked -------------------

def _stub_pip(monkeypatch, setup_installer, rc=0):
    seen = {}

    def fake_run_pip(action, cmd):
        seen['cmd'] = cmd
        return rc

    monkeypatch.setattr(setup_installer, '_run_pip', fake_run_pip)
    return seen


def _stub_import(monkeypatch, setup_installer, returncode, stderr=''):
    """Stub the verification subprocess only — pip is stubbed separately."""
    class _Proc:
        pass

    def fake_run(cmd, **kw):
        p = _Proc()
        p.returncode = returncode
        p.stderr = stderr
        p.stdout = ''
        return p

    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)


REMBG_TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "<string>", line 1, in <module>\n'
    '  File "/opt/env/lib/python3.12/site-packages/rembg/__init__.py", line 3, in <module>\n'
    '    from .bg import remove\n'
    '  File "/opt/env/lib/python3.12/site-packages/rembg/bg.py", line 7, in <module>\n'
    '    import onnxruntime as ort\n'
    "ModuleNotFoundError: No module named 'onnxruntime'\n"
)


def test_install_that_leaves_the_capability_broken_fails_and_says_why(app, monkeypatch):
    """THE reported scenario, replayed: pip reports success (everything "already
    satisfied") while `import rembg` dies on onnxruntime. Before the fix this
    returned 0 — "installed successfully" next to "✗ Not installed", with the one
    useful line (the module name) discarded."""
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    _stub_pip(monkeypatch, setup_installer, rc=0)
    _stub_import(monkeypatch, setup_installer, returncode=1, stderr=REMBG_TRACEBACK)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 1, 'a capability that does not import must not report success'
    log = '\n'.join(lines)
    assert 'missing module: onnxruntime' in log, (
        'the import traceback names the missing module — it is the most useful '
        'fact in the whole chain and it must reach the screen')
    assert 'person masks' in log.lower()


def test_install_whose_import_works_reports_success(app, monkeypatch):
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    _stub_pip(monkeypatch, setup_installer, rc=0)
    _stub_import(monkeypatch, setup_installer, returncode=0)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 0
    assert any('import OK' in l for l in lines)


def test_a_slow_cold_import_is_warming_not_a_failure(app, monkeypatch):
    """The first `import rembg` after an install compiles caches while the AV
    scans fresh DLLs (~20 s measured). Timing out must never fail a good install
    — that is the mirror-image lie."""
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    _stub_pip(monkeypatch, setup_installer, rc=0)

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(setup_installer.subprocess, 'run', boom)
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 0
    assert any('warming' in l for l in lines)


def test_the_verification_runs_the_exact_import_the_probe_runs():
    """If the two ever drift, "install OK" and "capability ✗" can disagree again.
    One dict, used by both."""
    from app import setup_installer, capabilities
    for action in setup_installer._CAPABILITY_ML_ACTIONS:
        assert capabilities.CAPABILITY_IMPORTS.get(action)
    assert capabilities.CAPABILITY_IMPORTS['masks'] == 'import rembg'


# --- 2b. an action that installs a BINARY cannot be verified by an import ------
#
# The `video` action installs two halves — PyAV (reading) and imageio-ffmpeg
# (encoding) — and its probe import, `import av`, can only speak for the first.
# So the whole #24 lie came back through the other door: pip said "already
# satisfied", `import av` passed, the action returned 0 and Setup printed
# "✓ installed successfully" while its own "Video bank — clip encoding" row
# stayed ✗ behind the same ↻ button. The user reinstalls the half that worked.

def _stub_video_install(monkeypatch, setup_installer, *, encoder_rc=0,
                        encoder_boom=None, import_rc=0):
    """pip succeeds; `import av` answers import_rc; `ffmpeg -version` answers
    encoder_rc. One router because both go through subprocess.run."""
    _stub_pip(monkeypatch, setup_installer, rc=0)

    def fake_run(cmd, **kw):
        class _P:
            stderr = ''
            stdout = ''
            returncode = 0
        p = _P()
        if str(cmd[0]).endswith('ffmpeg') or 'ffmpeg' in str(cmd[0]):
            if encoder_boom is not None:
                raise encoder_boom
            p.returncode = encoder_rc
            p.stderr = 'not an executable'
            return p
        p.returncode = import_rc
        p.stderr = "ModuleNotFoundError: No module named 'av'" if import_rc else ''
        return p

    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)


def test_video_install_that_cannot_encode_must_not_report_success(app, monkeypatch,
                                                                  tmp_path):
    """The half-success. Reading works, encoding does not — and the action used to
    return 0 for it."""
    from app import setup_installer
    from app.services import ffmpeg_tools
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    stub = tmp_path / 'ffmpeg'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    _stub_video_install(monkeypatch, setup_installer, encoder_rc=1)
    with app.app_context():
        setup_installer._runs['video'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('video')

    assert rc == 1, 'an install that left half the capability off must not say it worked'
    log = '\n'.join(lines).lower()
    assert 'encod' in log, 'the message must name the half that is missing'
    assert 'read' in log, 'and the half that works, so the user does not redo it'
    assert 'path' in log or 'install ffmpeg' in log, 'and the gesture that repairs it'


def test_video_install_reports_success_when_both_halves_work(app, monkeypatch, tmp_path):
    from app import setup_installer
    from app.services import ffmpeg_tools
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    stub = tmp_path / 'ffmpeg'
    stub.write_bytes(b'')
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: str(stub))
    _stub_video_install(monkeypatch, setup_installer, encoder_rc=0)
    with app.app_context():
        setup_installer._runs['video'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('video')

    assert rc == 0
    assert any('clip encoding ready' in l for l in lines)


def test_a_broken_encoder_does_not_fail_the_OTHER_capabilities(app, monkeypatch):
    """The extra check is registered per action. Person masks does not ship an
    encoder and must not start failing because this machine has no ffmpeg."""
    from app import setup_installer, config
    from app.services import ffmpeg_tools
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: None)
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: None)
    _stub_pip(monkeypatch, setup_installer, rc=0)
    _stub_import(monkeypatch, setup_installer, returncode=0)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')

    assert rc == 0


def test_every_installed_binary_is_covered_by_a_post_install_check():
    """The family rule behind this fix: a package whose payload is a BINARY (not
    an importable module) cannot be verified by re-running the probe import, so
    whichever action installs one owes an explicit extra check — otherwise its
    success message covers something it never looked at."""
    from app import setup_installer
    binary_packages = {setup_installer._canon('imageio-ffmpeg')}
    for action, pkgs in setup_installer._CAPABILITY_PACKAGES.items():
        if binary_packages & {setup_installer._canon(p) for p in pkgs}:
            assert action in setup_installer._CAPABILITY_EXTRA_CHECKS, (
                f'{action} installs a binary its probe import cannot see — it must '
                f'register a post-install check or it can report success about a '
                f'capability that is still half off')


def test_shot_detection_verifies_everything_it_installs():
    """The sister action, checked for the same gap: it installs
    transnetv2-pytorch + av into the scoring env and its verification imports
    torch, transnetv2_pytorch AND av — a superset, no binary, no gap. Weights
    ride inside the wheel, so there is nothing else 'installed' to prove."""
    from app import setup_installer, capabilities
    expr = capabilities.CAPABILITY_IMPORTS['shot_detect']
    for pkg in setup_installer._CAPABILITY_PACKAGES['shot_detect']:
        module = pkg.replace('-', '_')
        assert module in expr, f'shot_detect installs {pkg} but never verifies it'


# --- 3. don't break the installs that already work -----------------------------

def test_onnxruntime_is_not_reinstalled_when_the_env_already_imports_one(app, monkeypatch):
    """Idempotence AND a performance guard in one rule. onnxruntime, -gpu,
    -directml and -silicon are different distributions providing the same module;
    pip does not know they conflict, so installing the CPU one over a user's CUDA
    build "succeeds" and silently costs them their accelerator."""
    from app import setup_installer, config
    lines = []
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: lines.append(l))
    seen = _stub_pip(monkeypatch, setup_installer, rc=0)
    monkeypatch.setattr(setup_installer, '_onnxruntime_provided', lambda p: True)
    monkeypatch.setattr(setup_installer, '_verify_capability_import', lambda a, p: True)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        setup_installer._run_ml_capability('masks')
        rembg = setup_installer._requirement_spec('rembg')
    cmd = seen['cmd']
    assert rembg in cmd
    assert not any('onnxruntime' in str(c) for c in cmd), (
        'a working onnxruntime (possibly the GPU build) must be left alone'
    )
    assert any('already imports' in l for l in lines)


def test_onnxruntime_is_installed_when_the_env_cannot_import_it(app, monkeypatch):
    from app import setup_installer, config
    monkeypatch.setattr(setup_installer, '_append', lambda a, l: None)
    seen = _stub_pip(monkeypatch, setup_installer, rc=0)
    monkeypatch.setattr(setup_installer, '_onnxruntime_provided', lambda p: False)
    monkeypatch.setattr(setup_installer, '_verify_capability_import', lambda a, p: True)
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        setup_installer._run_ml_capability('masks')
        onnx = setup_installer._requirement_spec('onnxruntime')
    assert onnx in seen['cmd']


def test_an_onnxruntime_probe_that_times_out_counts_as_present(monkeypatch):
    """A missing module raises instantly; a SLOW import means a real (large)
    runtime is loading. Reading a timeout as "absent" would install the CPU build
    on top of exactly the heavy GPU one we must not touch."""
    from app import setup_installer

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)
    monkeypatch.setattr(setup_installer.subprocess, 'run', boom)
    assert setup_installer._onnxruntime_provided('/some/py') is True


def test_an_unlaunchable_interpreter_does_not_claim_onnxruntime_is_present(monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer.os.path, 'isfile', lambda p: True)

    def boom(cmd, **kw):
        raise OSError('no such interpreter')

    monkeypatch.setattr(setup_installer.subprocess, 'run', boom)
    assert setup_installer._onnxruntime_provided('/nope/py') is False


def test_every_capability_the_app_probes_can_be_installed_from_setup():
    """The hole the family guard above cannot see — and the reason it matters.

    That guard walks ``_CAPABILITY_PACKAGES``, so it only ever examines
    capabilities the installer ALREADY knows about. A capability added to
    ``CAPABILITY_IMPORTS`` and forgotten in the installer is invisible to it, and
    that is the worse failure of the two: the app probes the capability, the
    Setup screen shows it ✗ Not installed, and there is no button that fixes it.
    Whoever built the feature never saw it, because their machine already had the
    dependency — it is the NEW user who meets the dead end.

    An install ACTION is what is required here, not a package list: bank_siglip2
    and watermark_detect are installed by dedicated workers rather than a scoped
    pip set, and that is fine. What is never fine is a probe with no way out.
    """
    from app import setup_installer, capabilities
    missing = sorted(set(capabilities.CAPABILITY_IMPORTS)
                     - set(setup_installer.INSTALL_ACTIONS))
    assert not missing, (
        f"probed but not installable from Setup: {missing}. Every capability the "
        "app probes needs an INSTALL_ACTIONS entry — otherwise Setup shows ✗ with "
        "no button that repairs it. Add the action (plus its packages in "
        "_CAPABILITY_PACKAGES when it installs with pip), or remove the probe.")
