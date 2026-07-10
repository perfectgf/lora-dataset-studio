"""Training preflight guard-rails: per-family image floors, composition balance,
caption quality, identity leaks, untriaged images, VRAM — blockers vs warnings."""
import pytest

from app.config import LOCAL_USER


def _mk(app, n_keep=0, framing='face', caption='a nice varied caption with many words',
        train_type='zimage', fidelity=None, extra_rows=()):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, 'Pf', 'pf_trig', train_type=train_type,
                            fidelity=fidelity)
    for i in range(n_keep):
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'k{i}.webp', status='keep', framing=framing,
            caption=f'{caption} #{i}'))
    for row in extra_rows:
        svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, **row))
    svc.db.session.commit()
    return ds


def test_preflight_blocks_below_family_floor(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app, n_keep=8)                       # zimage floor = 12
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert any('hard minimum' in b and '12' in b for b in r['blockers'])
    assert r['kept'] == 8 and r['floor'] == 12


def test_preflight_family_floors_differ(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app, n_keep=15, train_type='sdxl')   # sdxl floor = 20
        r = lt.training_preflight(LOCAL_USER, ds.id)
        assert any('hard minimum' in b and '20' in b for b in r['blockers'])
        # same count on zimage -> no blocker, just the "recommended" warning
        r2 = lt.training_preflight(LOCAL_USER, ds.id, train_type='zimage')
    assert not r2['blockers']
    assert any('recommended' in w for w in r2['warnings'])


def test_preflight_warns_all_faces_and_body_fidelity(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app, n_keep=20, framing='face', fidelity='body')
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert any('face shot' in w for w in r['warnings'])
    assert any('body fidelity is ON' in w for w in r['warnings'])


def test_preflight_warns_identical_and_leaking_captions(app):
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = _mk(app, n_keep=0)
        for i in range(20):
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, filename=f'c{i}.webp', status='keep', framing='body',
                caption='her long blonde hair falls over the shoulders'))   # identical + hair leak
        svc.db.session.commit()
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert any('identical' in w for w in r['warnings'])
    assert any('describe the identity' in w for w in r['warnings'])


def test_preflight_warns_untriaged_and_short_captions(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app, n_keep=14, framing='body', caption='short words only',   # 3+1 mots
                 extra_rows=[{'filename': f'p{i}.webp', 'status': 'pending', 'framing': 'face'}
                             for i in range(3)])
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert any('await triage' in w and '3' in w for w in r['warnings'])
    assert any('very short' in w for w in r['warnings'])


def test_preflight_vram_warning_krea_only_when_known(app, monkeypatch):
    from app.services import lora_training as lt
    from app import capabilities
    with app.app_context():
        ds = _mk(app, n_keep=16, framing='body', train_type='krea')
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: 16.0)
        r = lt.training_preflight(LOCAL_USER, ds.id)
        assert any('VRAM' in w for w in r['warnings'])
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)   # unknown GPU
        r2 = lt.training_preflight(LOCAL_USER, ds.id)
        assert not any('VRAM' in w for w in r2['warnings'])


def test_preflight_clean_dataset_no_findings(app, monkeypatch):
    from app.services import lora_training as lt
    from app import capabilities
    with app.app_context():
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
        ds = _mk(app, n_keep=12, framing='body',
                 caption='full body shot of the subject walking through a sunny park wearing jeans')
        # 12 kept zimage -> no blocker; add faces? all-body doesn't trigger the
        # all-face warning; captions long + unique; no pending rows.
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert r['blockers'] == []
    assert [w for w in r['warnings'] if 'recommended' not in w] == []


def test_preflight_route(client, app, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
    with app.app_context():
        from app import config as cfg
        root = __import__('pathlib').Path(cfg.get('aitoolkit.dir') or '')
    # configure a fake ai-toolkit so the gate opens
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / 'venv' / 'Scripts').mkdir(parents=True)
    (tmp / 'venv' / 'Scripts' / 'python.exe').touch()
    (tmp / 'run.py').touch()
    with app.app_context():
        from app import config as cfg
        cfg.save_config({'aitoolkit': {'dir': str(tmp)}})
        ds = _mk(app, n_keep=5)
        ds_id = ds.id
    r = client.get(f'/api/dataset/{ds_id}/train/preflight')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['blockers']
    assert client.get('/api/dataset/999999/train/preflight').status_code == 404
