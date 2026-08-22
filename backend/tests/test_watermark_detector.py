"""The dedicated watermark detector — the contracts that must not drift.

Not a test of the models (they are 0.9 GB and this suite runs on CI with no
weights). What is tested is everything that CAN silently rot: the two copies of
the model ids, the fail-open promise, the threshold clamp, the region filtering
and the per-image protocol.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'infer'))

from app import config as cfg                       # noqa: E402
from app.services import watermark_detector as wd   # noqa: E402


def _infer_module():
    """Import the CHILD script the way it actually runs — by path, with no app
    package on the way in. It must stay importable with nothing but the standard
    library plus Pillow, because the interpreter that runs it is not this one."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'watermark_detect_infer.py')
    spec = importlib.util.spec_from_file_location('watermark_detect_infer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_ids_do_not_drift_between_parent_and_child():
    """The parent publishes the repo list (the installer downloads it, the probe
    checks the cache for it) and the child hardcodes what it loads. They live in
    two files that run in two different interpreters, so nothing but this test
    stops them from disagreeing — and a disagreement means an install that
    reports success over a capability that stays ✗."""
    infer = _infer_module()
    assert infer.RANK_MODEL in wd.MODEL_FILES
    assert infer.LOCATE_MODEL in wd.MODEL_FILES
    assert set(wd.MODEL_REPOS) == {infer.RANK_MODEL, infer.LOCATE_MODEL}


def test_every_model_is_permissively_licensed():
    """The whole architecture exists because the obvious choice (an ultralytics
    YOLO) claims AGPL-3.0 over its trained weights, which would contaminate this
    public repository. A future weight swap must not quietly reintroduce that."""
    for repo, meta in wd.MODEL_FILES.items():
        assert meta['license'] in ('Apache-2.0', 'MIT'), repo
        assert meta['files'], repo
        assert meta['approx_mb'] > 0, repo


def test_never_downloads_a_whole_repo():
    """Measured 2026-08-03: the SigLIP2 repo publishes its training checkpoints
    (optimizer state included) beside a 371 MB model, so a whole-repo pull costs
    2.4 GB. The install therefore names files — and must keep naming them."""
    files = wd.MODEL_FILES['prithivMLmods/Watermark-Detection-SigLIP2']['files']
    assert 'model.safetensors' in files
    assert not any(f.startswith('checkpoint') or f.endswith('.bin') for f in files)
    # A duplicate weight file in the other repo would silently double its cost.
    dino = wd.MODEL_FILES['IDEA-Research/grounding-dino-tiny']['files']
    assert 'pytorch_model.bin' not in dino


def test_threshold_is_clamped_not_refused(monkeypatch):
    """Read in the middle of a long pass: a hand-edited config must degrade to a
    usable number, never abort a scan that is already running."""
    monkeypatch.setattr(cfg, 'get', lambda key, *a: 5 if key == 'watermark_detect.threshold' else '')
    assert wd.threshold() == 1.0
    monkeypatch.setattr(cfg, 'get', lambda key, *a: -2 if key == 'watermark_detect.threshold' else '')
    assert wd.threshold() == 0.0
    monkeypatch.setattr(cfg, 'get', lambda key, *a: 'nonsense' if key == 'watermark_detect.threshold' else '')
    assert wd.threshold() == 0.60   # the documented fallback, not a crash


def test_default_threshold_is_the_measured_one():
    """0.94 is not a taste, it is the knee measured on a labelled sample; the
    Settings help and the guide both quote it. Changing it here without changing
    those is how a documented number becomes a lie."""
    assert cfg.DEFAULTS['watermark_detect']['threshold'] == 0.94


def test_detector_python_falls_back_to_the_scoring_environment(monkeypatch):
    """The sharing is the design, not an accident: it is what stops the extra
    from asking for a second 2.5 GB copy of torch."""
    values = {'watermark_detect.python': '', 'bank_scoring.python': '/some/env/python'}
    monkeypatch.setattr(cfg, 'get', lambda key, *a: values.get(key, ''))
    assert wd.detector_python() == '/some/env/python'


def test_scan_of_nothing_is_not_an_error():
    assert list(wd.scan([])) == []


# --- the child's pure helpers (no torch, no weights) --------------------------
def test_boxes_are_normalised_and_frame_sized_ones_dropped():
    infer = _infer_module()
    out = infer._normalise_boxes(
        [[100, 200, 300, 400],        # a real, small mark
         [0, 0, 1000, 1000]],         # the whole frame — a failed localisation
        (1000, 1000))
    assert out == [[0.1, 0.2, 0.3, 0.4]]


def test_boxes_are_clamped_to_the_frame():
    infer = _infer_module()
    out = infer._normalise_boxes([[-50, -50, 120, 120]], (1000, 1000))
    assert out == [[0.0, 0.0, 0.12, 0.12]]


def test_overlapping_boxes_merge_but_two_corners_stay_two_zones():
    """Three phrases produce three boxes over ONE mark; those must merge. Two
    marks in two corners must NOT — a single box spanning them would cover the
    subject sitting between them."""
    infer = _infer_module()
    a = [0.80, 0.90, 0.98, 0.99]
    a2 = [0.81, 0.91, 0.99, 0.995]      # the same mark, seen by another phrase
    b = [0.01, 0.02, 0.10, 0.06]        # a different mark, other corner
    merged = infer._merge_boxes([a, a2, b])
    assert len(merged) == 2
    assert any(m[0] < 0.11 for m in merged)      # the top-left one survived alone
    assert any(m[0] > 0.79 for m in merged)


def test_the_first_box_is_the_most_peripheral_one_not_the_biggest():
    """Measured failure this ordering fixes: an image whose real mark was a
    0.2%-of-frame corner logo also produced a 2.7% box in the MIDDLE of the
    picture. Only the first box is persisted, so "biggest first" aimed the crop
    at the subject."""
    infer = _infer_module()
    corner = [0.8752, 0.9497, 0.9963, 0.9955]      # the real mark, tiny
    middle = [0.3574, 0.3405, 0.4749, 0.5660]      # bigger, and on the subject
    assert infer._merge_boxes([middle, corner])[0] == corner


def test_equally_peripheral_boxes_prefer_the_fuller_cover():
    infer = _infer_module()
    tight = [0.0, 0.0, 0.05, 0.05]
    full = [0.0, 0.0, 0.09, 0.09]
    assert infer._merge_boxes([tight, full])[0] == full


def test_region_area_cap_is_the_measured_one():
    """0.10 sits in the empty gap the 87 measured boxes left between real marks
    (≤0.041) and failed localisations (≥0.069). Loosening it silently is how the
    crop level starts cutting subjects."""
    infer = _infer_module()
    assert infer.MAX_REGION_AREA == 0.10


def test_a_locator_that_cannot_load_degrades_to_no_boxes():
    """Fail-open all the way down: a broken second model costs the POSITION of a
    mark, never the pass."""
    infer = _infer_module()
    locator = infer._Locator('cpu', None)
    locator.failed = 'ImportError: boom'          # as if _ensure had failed
    assert locator.regions(object()) == []


def test_child_declares_no_third_party_import_at_module_level():
    """The child is imported here in the FLASK venv, which has neither torch nor
    transformers. It must therefore keep every heavy import inside a function —
    the same discipline every other infer worker follows, and what lets this
    contract be tested at all on a machine with no extra installed."""
    module = _infer_module()
    assert module.RANK_MODEL and module.LOCATE_MODEL


class _FakeProc:
    """Enough of Popen to exercise the streaming protocol without a child."""

    def __init__(self, lines, rc=0):
        import io
        self.stdout = io.StringIO(''.join(lines))
        self.stderr = io.StringIO('')
        self.stdin = io.StringIO()
        self._rc = rc

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def kill(self):
        pass


def _fake_popen(lines):
    def factory(*_a, **_kw):
        return _FakeProc(lines)
    return factory


def test_the_devices_the_child_ran_on_is_reported_not_dropped(monkeypatch):
    """The child has always said which device it ranked on; the parent used to
    read the summary only on failure, which is how a 4090 machine scanned tens
    of thousands of images on the CPU with no line of UI or log saying so."""
    lines = [
        json.dumps({'path': 'a.jpg', 'state': 'none', 'score': 0.1,
                    'regions': [], 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True, 'device': 'cpu'}}) + '\n',
    ]
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen(lines))
    info = {}
    assert len(list(wd.scan(['a.jpg'], info=info))) == 1
    assert info['device'] == 'cpu'
    # And a caller that passes nothing keeps the old signature, byte for byte.
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen(lines))
    assert len(list(wd.scan(['a.jpg']))) == 1


def test_results_are_yielded_per_image_in_order(monkeypatch):
    lines = [
        json.dumps({'path': 'a.jpg', 'state': 'none', 'score': 0.1,
                    'regions': [], 'error': None}) + '\n',
        json.dumps({'path': 'b.jpg', 'state': 'detected', 'score': 0.99,
                    'regions': [[0.1, 0.2, 0.3, 0.4]], 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen(lines))
    out = list(wd.scan(['a.jpg', 'b.jpg']))
    assert [r[0] for r in out] == ['a.jpg', 'b.jpg']
    assert out[1][1] == 'detected'
    assert out[1][3] == [[0.1, 0.2, 0.3, 0.4]]


def test_a_result_for_the_wrong_image_is_dropped_never_misattributed(monkeypatch):
    """The one thing this module must never do: attach one image's verdict to
    another image's database row."""
    lines = [
        json.dumps({'path': 'SOMETHING-ELSE.jpg', 'state': 'detected',
                    'score': 0.99, 'regions': [], 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen(lines))
    assert list(wd.scan(['a.jpg'])) == []


def test_a_child_that_cannot_load_raises_rather_than_reporting_all_clean(monkeypatch):
    """A load failure that came back as 'no detections' would mark a whole bank
    clean. It must be distinguishable, because the caller falls back on it."""
    lines = [json.dumps({'summary': {'ok': False, 'error': 'no weights'}}) + '\n']
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen(lines))
    with pytest.raises(wd.DetectorUnavailable):
        list(wd.scan(['a.jpg']))


def test_a_silent_child_raises_too(monkeypatch):
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen([]))
    with pytest.raises(wd.DetectorUnavailable):
        list(wd.scan(['a.jpg']))


def test_garbage_on_stdout_is_ignored_not_fatal(monkeypatch):
    """A stray print from a library must not sink a 30 000-image pass."""
    lines = [
        'downloading something...\n',
        json.dumps({'path': 'a.jpg', 'state': 'none', 'score': 0.0,
                    'regions': [], 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]
    monkeypatch.setattr(wd.subprocess, 'Popen', _fake_popen(lines))
    assert len(list(wd.scan(['a.jpg']))) == 1


def test_capability_is_false_without_weights(monkeypatch, tmp_path):
    """Importing torch is necessary but not sufficient — an environment with the
    packages and no weights would light the capability green and then fail a
    whole pass on a network error."""
    from app import capabilities
    monkeypatch.setattr(capabilities, '_cached_import', lambda *a, **kw: True)
    monkeypatch.setattr(wd, 'models_root', lambda: str(tmp_path))
    verdict = capabilities.probe_watermark_detect()
    assert verdict['ok'] is False
    assert 'weights' in verdict['detail']


def test_no_weights_means_no_torch_subprocess_at_all(monkeypatch, tmp_path):
    """The probe runs on every capability poll. On the machine that does NOT have
    this extra — which is nearly every machine, and every CI run — the import
    subprocess can never change the answer, so it must never be spawned. Getting
    this backwards taxes every future test run with a cold `import torch`."""
    from app import capabilities
    calls = []
    monkeypatch.setattr(capabilities, '_cached_import',
                        lambda *a, **kw: calls.append(a) or True)
    monkeypatch.setattr(wd, 'models_root', lambda: str(tmp_path))
    capabilities.probe_watermark_detect()
    assert calls == []


def test_capability_is_true_once_both_snapshots_exist(monkeypatch, tmp_path):
    from app import capabilities
    monkeypatch.setattr(capabilities, '_cached_import', lambda *a, **kw: True)
    monkeypatch.setattr(wd, 'models_root', lambda: str(tmp_path))
    for repo in wd.MODEL_REPOS:
        snap = tmp_path / ('models--' + repo.replace('/', '--')) / 'snapshots' / 'abc'
        snap.mkdir(parents=True)
        (snap / 'config.json').write_text('{}', encoding='utf-8')
    assert capabilities.probe_watermark_detect()['ok'] is True


def test_a_half_finished_download_does_not_count_as_installed(monkeypatch, tmp_path):
    from app import capabilities
    monkeypatch.setattr(capabilities, '_cached_import', lambda *a, **kw: True)
    monkeypatch.setattr(wd, 'models_root', lambda: str(tmp_path))
    for repo in wd.MODEL_REPOS:
        (tmp_path / ('models--' + repo.replace('/', '--'))
         / 'snapshots' / 'abc').mkdir(parents=True)   # empty snapshot folder
    assert capabilities.probe_watermark_detect()['ok'] is False


def test_the_install_action_exists_and_has_a_worker():
    from app import setup_installer
    assert 'watermark_detect' in setup_installer.INSTALL_ACTIONS
    # Installs into the venv bank_scoring owns, so it MUST share the pip queue:
    # two pip processes writing one environment corrupt its dist-info.
    assert 'watermark_detect' in setup_installer._PIP_ACTIONS
    # Deliberately absent from "Install everything": ~0.9 GB nobody asked for.
    assert 'watermark_detect' not in setup_installer._INSTALL_ALL_ORDER


def test_the_manual_command_names_the_files_not_the_repo():
    from app import setup_installer
    cmd = setup_installer.manual_command('watermark_detect')
    assert 'snapshot_download' not in cmd
    assert 'hf_hub_download' in cmd or 'from huggingface_hub import' in cmd


def test_schema_addition_is_declared_for_both_new_columns():
    """Additive, or every existing database keeps a model that no longer matches
    its table (the app never ALTERs on its own)."""
    from app import _SCHEMA_ADDITIONS
    assert ('bank_image', 'watermark_source', 'VARCHAR(16)') in _SCHEMA_ADDITIONS
    assert ('bank_image', 'watermark_score', 'REAL') in _SCHEMA_ADDITIONS


def test_settings_save_cannot_blank_the_managed_interpreter():
    """The frontend echoes back a blank `python` on a full save; left in, it would
    deep-merge over the path the installer wrote and turn a perfect install into
    a permanent ✗ — the exact bug the other managed keys already guard against."""
    import inspect
    from app.routes import settings as settings_routes
    source = inspect.getsource(settings_routes)
    assert "'watermark_detect')" in source or "'watermark_detect'," in source


def test_module_docstring_records_why_florence2_was_dropped():
    """The research picked Florence-2 for the localisation stage and this build
    does not use it. The reason (its remote code no longer loads on current
    transformers, and pinning transformers down would endanger the bank-scoring
    stack that shares the environment) has to survive in the code, or the next
    person 'fixes' it back."""
    infer = _infer_module()
    doc = infer.__doc__ or ''
    assert 'Florence-2' in doc
    assert 'forced_bos_token_id' in doc
