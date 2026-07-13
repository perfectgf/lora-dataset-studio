import io
import json
import os

from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


class _Proc:
    """Minimal stand-in for subprocess.CompletedProcess."""
    def __init__(self, stdout, returncode=0, stderr=''):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _dataset_with_ref_and_kept_image(svc, LOCAL_USER):
    """Real FaceDataset with a reference photo + one kept FaceDatasetImage, both
    backed by real files on disk (score_dataset_faces checks os.path.isfile)."""
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, 'FaceTest', 'facetest')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    fn = 'kept.webp'
    with open(os.path.join(d, fn), 'wb') as fh:
        fh.write(_png((0, 255, 0)))
    img = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep', filename=fn, framing='face')
    svc.db.session.add(img)
    svc.db.session.commit()
    return ds, img


# --- face_similarity.score_dataset_faces / analyze_faces --------------------

def test_analyze_faces_maps_state_and_sim_onto_rows(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        path = svc._img_path(img)
        contract = {"ref_ok": True, "results": {path: {"state": "scorable", "sim": 0.62,
                                                        "det": 0.9, "bbox_frac": 0.2, "yaw": 5.0}}}
        # Noise lines before the JSON line -- the parser must pick the LAST `{`-line.
        noisy_stdout = "some progress line\n[face] loading model\n" + json.dumps(contract)
        monkeypatch.setattr('app.services.face_similarity.subprocess.run',
                            lambda *a, **k: _Proc(noisy_stdout))
        counts, _err = svc.analyze_faces(LOCAL_USER, ds.id)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.face_state == 'scorable'
        assert refreshed.face_score == 0.62
        assert counts == {'scorable': 1}


def test_analyze_faces_ref_not_ok_returns_empty_and_no_row_change(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        contract = {"ref_ok": False, "results": {}, "error": "ref unusable"}
        monkeypatch.setattr('app.services.face_similarity.subprocess.run',
                            lambda *a, **k: _Proc(json.dumps(contract)))
        counts, _err = svc.analyze_faces(LOCAL_USER, ds.id)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert counts == {}
        assert refreshed.face_state is None
        assert refreshed.face_score is None


def test_score_dataset_faces_unavailable_returns_empty_without_subprocess(app, monkeypatch):
    """is_available() False (capability probe) -> score_dataset_faces returns {}
    WITHOUT ever invoking subprocess.run (the never-raise/never-shell-out contract)."""
    from app.services import face_similarity as fsim
    from app.capabilities import probe_face_scoring

    monkeypatch.setattr(fsim, 'is_available', lambda: False)

    def _boom(*a, **k):
        raise AssertionError('subprocess.run must not be called when unavailable')

    monkeypatch.setattr('app.services.face_similarity.subprocess.run', _boom)
    with app.app_context():
        # inputs missing on disk -> ({}, None) before the availability check
        assert fsim.score_dataset_faces('/does/not/matter', ['/also/not']) == ({}, None)


def test_is_available_delegates_to_capability_probe(app, monkeypatch):
    from app.services import face_similarity as fsim
    import app.capabilities as caps_mod

    monkeypatch.setattr(caps_mod, 'probe_face_scoring', lambda: {'ok': True, 'detail': 'x'})
    with app.app_context():
        assert fsim.is_available() is True
    monkeypatch.setattr(caps_mod, 'probe_face_scoring', lambda: {'ok': False, 'detail': 'x'})
    with app.app_context():
        assert fsim.is_available() is False


def test_analyze_faces_returns_cleanly_when_scorer_unavailable(app, monkeypatch):
    """analyze_faces must not raise or crash when the scorer is unavailable --
    score_dataset_faces short-circuits to {}, so no row gets a state/score and
    counts comes back empty."""
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: False)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert counts == {}
        assert err and err['kind'] == 'unavailable'
        assert refreshed.face_state is None


def test_score_dataset_faces_stdin_payload_includes_models_root(app, monkeypatch):
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER, save_config

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = kwargs.get('input')
        return _Proc(json.dumps({"ref_ok": True, "results": {}}))

    monkeypatch.setattr('app.services.face_similarity.subprocess.run', _fake_run)
    with app.app_context():
        save_config({'face_scoring': {'models_root': 'C:/models/insightface'}})
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png')
            img_path = os.path.join(d, 'img.png')
            with open(ref, 'wb') as fh:
                fh.write(_png())
            with open(img_path, 'wb') as fh:
                fh.write(_png())
            fsim.score_dataset_faces(ref, [img_path])
    payload = json.loads(captured['input'])
    assert payload['models_root'] == 'C:/models/insightface'


def test_score_dataset_faces_stdin_payload_models_root_none_when_unconfigured(app, monkeypatch):
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = kwargs.get('input')
        return _Proc(json.dumps({"ref_ok": True, "results": {}}))

    monkeypatch.setattr('app.services.face_similarity.subprocess.run', _fake_run)
    with app.app_context():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png')
            img_path = os.path.join(d, 'img.png')
            with open(ref, 'wb') as fh:
                fh.write(_png())
            with open(img_path, 'wb') as fh:
                fh.write(_png())
            fsim.score_dataset_faces(ref, [img_path])
    payload = json.loads(captured['input'])
    assert payload['models_root'] is None


def test_score_dataset_faces_native_crash_returns_empty_not_exception(app, monkeypatch):
    """A crashed/killed subprocess (OSError) or a timeout must degrade to {},
    never raise up through analyze_faces to the route."""
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(fsim, 'is_available', lambda: True)

    def _crash(*a, **k):
        raise OSError('the interpreter died')

    monkeypatch.setattr('app.services.face_similarity.subprocess.run', _crash)
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        counts, err = svc.analyze_faces(LOCAL_USER, ds.id)  # must not raise
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert counts == {}
        assert err and err['kind'] == 'failed' and 'interpreter died' in err['detail']
        assert refreshed.face_state is None


# --- dataset_payload exposes face_thresholds --------------------------------

def test_dataset_payload_includes_face_thresholds_from_config(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'face_scoring': {'green': 0.55, 'orange': 0.42}})
        ds = svc.create_dataset(LOCAL_USER, 'Thresh', 'thresh')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['face_thresholds'] == {'green': 0.55, 'orange': 0.42}


def test_dataset_payload_face_thresholds_default(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Default', 'default')
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['face_thresholds'] == {'green': 0.50, 'orange': 0.45}


def test_score_dataset_faces_crash_reports_stderr_tail(app, monkeypatch):
    """A subprocess that dies with a traceback and no JSON must surface the
    traceback's LAST LINE as the error detail — that line named the real
    problem (nested antelopev2 AssertionError) in the field."""
    import os, tempfile
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    _stderr = ('Traceback (most recent call last):\n'
               '  File "face_analysis.py", line 61\n'
               'AssertionError')
    monkeypatch.setattr(
        'app.services.face_similarity.subprocess.run',
        lambda *a, **k: _Proc('', stderr=_stderr, returncode=1))
    with app.app_context():
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, 'ref.png'); img_path = os.path.join(d, 'img.png')
            for f in (ref, img_path):
                with open(f, 'wb') as fh:
                    fh.write(_png())
            results, err = fsim.score_dataset_faces(ref, [img_path])
    assert results == {}
    assert err['kind'] == 'failed' and err['detail'] == 'AssertionError'


def test_score_faces_payload_carries_scoring_error(app, monkeypatch):
    """The Studio route payload must carry scoring_error so the toast can say
    WHY instead of a green « done — 0/N » (user-reported)."""
    import os
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as studio
    from app.models import LoraTestImage
    from app.config import LOCAL_USER

    monkeypatch.setattr(
        'app.services.face_similarity.score_dataset_faces',
        lambda ref, paths, timeout=900: ({}, {'kind': 'failed', 'detail': 'AssertionError'}))
    with app.app_context():
        ds, img = _dataset_with_ref_and_kept_image(svc, LOCAL_USER)
        cell_path = os.path.join(svc._dataset_dir(ds.id), 'cell.webp')
        with open(cell_path, 'wb') as fh:
            fh.write(_png())
        svc.db.session.add(LoraTestImage(dataset_id=ds.id, status='done',
                                         filename='cell.webp',
                                         checkpoint='lora_x_000000250.safetensors',
                                         prompt='p', seed=1, strength=1.0))
        svc.db.session.commit()
        out = studio.score_faces(LOCAL_USER, ds.id)
    assert out['scored'] == 0 and out['total'] == 1
    assert out['scoring_error'] == {'kind': 'failed', 'detail': 'AssertionError'}
