"""Guest checkpoints on create_run: a models/loras file that is NOT in this
dataset's trigger-matched pool becomes its own cell (same prompt/seed), not an
extra stacked on every cell.

Fail-closed like Canvas `external_loras`: unsafe names and missing files are
hard errors before any row exists. A file at the loras ROOT (no family folder)
is allowed and inherits the run family.
"""
import json

import pytest

_ST = (b'\x08\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{}}'
       .ljust(32, b'\x00'))


def _guest_tree(tmp_path, monkeypatch, trigger, guests=(), krea_guests=()):
    """Trained z-image checkpoint matching `trigger`, plus guest files at the
    loras ROOT and/or under loras/krea. Returns (dataset, trained_rel)."""
    from app import config
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True, exist_ok=True)
    trained = f'lora_{trigger}_000002000.safetensors'
    (lora_dir / trained).write_bytes(_ST)
    loras_root = base / 'models' / 'loras'
    for name in guests:
        (loras_root / name).write_bytes(_ST)
    if krea_guests:
        krea_dir = loras_root / 'krea'
        krea_dir.mkdir(parents=True, exist_ok=True)
        for name in krea_guests:
            (krea_dir / name).write_bytes(_ST)
    unet_dir = base / 'models' / 'unet' / 'z image'
    unet_dir.mkdir(parents=True, exist_ok=True)
    (unet_dir / 'zmodel.safetensors').write_bytes(_ST)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache',
                        {'data': None, 'timestamp': 0})
    ds = svc.create_dataset(LOCAL_USER, trigger.capitalize(), trigger)
    return ds, 'z image' + chr(92) + trained


def _wire(monkeypatch, lts, seen=None):
    """No GPU/ComfyUI: stub the workflow builder and the queue enqueue. The
    real `_persist_and_enqueue_cell` still writes rows, which is what the
    payload/label tests read back."""
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
    monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)

    def fake_build(*a, **k):
        if seen is not None:
            seen.setdefault('checkpoints', []).append(
                a[1] if len(a) > 1 else k.get('checkpoint'))
            seen['allowed'] = a[6] if len(a) > 6 else k.get('allowed_loras')
            seen['extra_loras'] = k.get('extra_loras')
        return {'1': {}}
    monkeypatch.setattr(lts, '_build_cell_workflow', fake_build)
    monkeypatch.setattr(lts, '_enqueue_cell',
                        lambda *a, **k: k.get('job_id') or 'job-1')


def test_guest_is_its_own_cell_not_an_extra(app, tmp_path, monkeypatch):
    """A root-level file rides as the tested LoRA of its cell — never extra_loras."""
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst1',
                                  guests=['other-char.safetensors'])
        seen = {}
        _wire(monkeypatch, lts, seen)
        lts.create_run(LOCAL_USER, ds.id, [trained, 'other-char.safetensors'],
                       [1.0], lts.StudioGenSettings(prompt='p', count=1, seed=7))
        cps = {r.checkpoint for r in LoraTestImage.query.all()}
        assert trained in cps
        assert 'other-char.safetensors' in cps
        assert len(cps) == 2
        for row in LoraTestImage.query.all():
            extras = json.loads(row.extra_loras) if row.extra_loras else []
            assert extras == []
        assert 'other-char.safetensors' in seen['checkpoints']
        assert 'other-char.safetensors' in seen['allowed']


def test_guest_only_run_is_allowed(app, tmp_path, monkeypatch):
    """Tick none of mine, only theirs — still one create_run on this dataset."""
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, _trained = _guest_tree(tmp_path, monkeypatch, 'gst2',
                                   guests=['solo-theirs.safetensors'])
        seen = {}
        _wire(monkeypatch, lts, seen)
        lts.create_run(LOCAL_USER, ds.id, ['solo-theirs.safetensors'],
                       [1.0], lts.StudioGenSettings(prompt='p', count=1, seed=3))
        rows = LoraTestImage.query.all()
        assert len(rows) == 1
        assert rows[0].checkpoint == 'solo-theirs.safetensors'
        assert rows[0].dataset_id == ds.id


def test_guest_missing_file_is_a_hard_error(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst3')
        _wire(monkeypatch, lts)
        with pytest.raises(ValueError, match='external LoRA not found'):
            lts.create_run(LOCAL_USER, ds.id, [trained, 'ghost.safetensors'],
                           [1.0], lts.StudioGenSettings(prompt='p', count=1))
        assert LoraTestImage.query.count() == 0


def test_guest_wrong_arch_409s_before_any_row(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst4',
                                  guests=['wrong-arch.safetensors'])
        _wire(monkeypatch, lts)
        real_detect = lts.lt.detect_lora_arch

        def fake_detect(path):
            if 'wrong-arch' in str(path):
                return 'sdxl'
            return real_detect(path)
        monkeypatch.setattr(lts.lt, 'detect_lora_arch', fake_detect)
        with pytest.raises(lts.StudioArchMismatch):
            lts.create_run(LOCAL_USER, ds.id, [trained, 'wrong-arch.safetensors'],
                           [1.0], lts.StudioGenSettings(prompt='p', count=1))
        assert LoraTestImage.query.count() == 0


def test_guest_in_other_family_folder_is_refused(app, tmp_path, monkeypatch):
    """A file sitting in loras/krea cannot join a z-image epoch grid."""
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst5',
                                  krea_guests=['krea-char.safetensors'])
        _wire(monkeypatch, lts)
        krea_rel = 'krea' + chr(92) + 'krea-char.safetensors'
        with pytest.raises(ValueError, match='cannot mix multiple families'):
            lts.create_run(LOCAL_USER, ds.id, [trained, krea_rel],
                           [1.0], lts.StudioGenSettings(prompt='p', count=1))
        assert LoraTestImage.query.count() == 0


_TRAVERSAL_NAMES = (
    '..' + chr(92) + '..' + chr(92) + 'evil.safetensors',
    'C:' + chr(92) + 'evil.safetensors',
    '/abs/evil.safetensors',
)


def test_guest_rejects_path_traversal_names(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst6')
        _wire(monkeypatch, lts)
        for bad in _TRAVERSAL_NAMES:
            with pytest.raises(ValueError, match='invalid external LoRA name'):
                lts.create_run(LOCAL_USER, ds.id, [trained, bad],
                               [1.0], lts.StudioGenSettings(prompt='p', count=1))
        assert LoraTestImage.query.count() == 0


def test_guest_capped_at_16(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        names = [f'g{i}.safetensors' for i in range(17)]
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst7', guests=names)
        _wire(monkeypatch, lts)
        with pytest.raises(ValueError, match='at most 16'):
            lts.create_run(LOCAL_USER, ds.id, [trained] + names,
                           [1.0], lts.StudioGenSettings(prompt='p', count=1))
        assert LoraTestImage.query.count() == 0


def test_mine_only_run_unchanged(app, tmp_path, monkeypatch):
    """No guest in the list → the old whitelist path, no guest validation."""
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst8')
        seen = {}
        _wire(monkeypatch, lts, seen)
        lts.create_run(LOCAL_USER, ds.id, [trained], [1.0], lts.StudioGenSettings(prompt='p', count=1, seed=1))
        rows = LoraTestImage.query.all()
        assert len(rows) == 1
        assert rows[0].checkpoint == trained
        extras = json.loads(rows[0].extra_loras) if rows[0].extra_loras else []
        assert extras == []


def test_guest_cell_label_is_prefixed_theirs(app, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst9',
                                  guests=['other-char.safetensors'])
        _wire(monkeypatch, lts, {})
        lts.create_run(LOCAL_USER, ds.id, [trained, 'other-char.safetensors'],
                       [1.0], lts.StudioGenSettings(prompt='p', count=1, seed=1))
        payload = lts.studio_payload(LOCAL_USER, ds.id, family='zimage')
        by_cp = {c['checkpoint']: c['label'] for c in payload['cells']}
        assert 'Theirs · ' in by_cp['other-char.safetensors']
        assert 'Theirs · ' not in by_cp[trained]


def test_root_guest_stays_on_the_same_family_grid(app, tmp_path, monkeypatch):
    """family_of_lora is None at the loras root; the cell must still appear
    next to the trained epoch it was launched with, not vanish."""
    from app.config import LOCAL_USER
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, trained = _guest_tree(tmp_path, monkeypatch, 'gst10',
                                  guests=['root-guest.safetensors'])
        _wire(monkeypatch, lts, {})
        lts.create_run(LOCAL_USER, ds.id, [trained, 'root-guest.safetensors'],
                       [1.0], lts.StudioGenSettings(prompt='p', count=1, seed=1))
        payload = lts.studio_payload(LOCAL_USER, ds.id, family='zimage')
        cps = {c['checkpoint'] for c in payload['cells']}
        assert trained in cps
        assert 'root-guest.safetensors' in cps


def test_guest_only_root_file_stays_on_krea_grid(app, tmp_path, monkeypatch):
    """Tick none of mine — a root guest launched from the Krea studio must
    still list on the Krea grid, not collapse to the historical zimage
    fallback and vanish."""
    from app import config
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = tmp_path / 'Comfy'
        krea_dir = base / 'models' / 'loras' / 'krea'
        krea_dir.mkdir(parents=True, exist_ok=True)
        trained = 'lora_gstk_000002000.safetensors'
        (krea_dir / trained).write_bytes(_ST)
        (base / 'models' / 'loras' / 'only-theirs.safetensors').write_bytes(_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        ds = svc.create_dataset(LOCAL_USER, 'Gstk', 'gstk')
        ds.train_type = 'krea'
        _wire(monkeypatch, lts, {})
        lts.create_run(LOCAL_USER, ds.id, ['only-theirs.safetensors'],
                       [1.0], lts.StudioGenSettings(prompt='p', count=1, seed=1), family='krea')
        assert LoraTestImage.query.count() == 1
        payload = lts.studio_payload(LOCAL_USER, ds.id, family='krea')
        cps = {c['checkpoint'] for c in payload['cells']}
        assert 'only-theirs.safetensors' in cps
        assert payload['family'] == 'krea'
