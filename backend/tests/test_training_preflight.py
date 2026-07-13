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
    # « which ones » : every one of the 20 hair-leak captions is listed, with the
    # shape the UI needs to render a thumbnail + editable caption.
    assert len(r['leak_images']) == 20
    li = r['leak_images'][0]
    assert set(li) == {'id', 'filename', 'caption'} and 'hair' in li['caption']


def test_preflight_concept_skips_identity_leak(app):
    """A concept dataset DESCRIBES identity on purpose (the concept, not the face, binds to
    the trigger). The identity-leak dimension must be skipped entirely — otherwise the SAME
    hair/face captions that legitimately warn for a character trip a false warning on every
    concept training run (the '...identité leak' messages reported in concept mode)."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.models import FaceDatasetImage
    with app.app_context():
        c = svc.create_dataset(LOCAL_USER, 'C', 'c_trig', kind='concept',
                               concept_desc='a candid mirror selfie')
        for i in range(20):
            svc.db.session.add(FaceDatasetImage(       # SAME hair "leak" as the character test
                dataset_id=c.id, filename=f'c{i}.webp', status='keep', framing='body',
                caption=f'her long blonde hair falls over the shoulders #{i}'))
        svc.db.session.commit()
        r = lt.training_preflight(LOCAL_USER, c.id)
    assert r['leak_images'] == []                                        # no drill-down list
    assert not any('describe the identity' in w for w in r['warnings'])  # no false warning


def test_preflight_concept_skips_composition_balance(app):
    """The face/bust/body/back balance is a CHARACTER heuristic (render a face at every
    distance). A concept LoRA learns on whatever framings it has — and its images are often
    left unclassified (framing=None) — so the composition dimension is skipped for concept,
    else it trips a false 'every kept image is a face shot' before every training run."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.models import FaceDatasetImage
    with app.app_context():
        # character, all face shots -> composition WARNS (baseline sanity)
        char = _mk(app, n_keep=20, framing='face')
        rc = lt.training_preflight(LOCAL_USER, char.id)
        assert any('face shot' in w for w in rc['warnings'])
        # concept, SAME all-face framing -> no composition warning / check row
        con = svc.create_dataset(LOCAL_USER, 'C2', 'c2_trig', kind='concept',
                                 concept_desc='a candid mirror selfie')
        for i in range(20):
            svc.db.session.add(FaceDatasetImage(
                dataset_id=con.id, filename=f'x{i}.webp', status='keep', framing='face',
                caption=f'a full-body scene described with a fair number of words #{i}'))
        svc.db.session.commit()
        r = lt.training_preflight(LOCAL_USER, con.id)
    assert not any('face shot' in w for w in r['warnings'])
    assert not any(c.get('id') == 'composition' for c in r['checks'])


def _grad_png(path, reverse=False):
    """64px horizontal grayscale gradient — a stable, non-uniform dHash. reverse
    flips it so its dHash is the bitwise opposite (max hamming distance)."""
    from PIL import Image
    w = h = 64
    img = Image.new('RGB', (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            t = (w - 1 - x) if reverse else x
            v = int(255 * t / (w - 1))
            px[x, y] = (v, v, v)
    img.save(path)


def test_preflight_lists_near_duplicate_pairs(app):
    """dup_pairs names the two offending images per near-duplicate pair (dHash),
    so the UI can show the pair and let one be rejected. Two identical gradients
    form one pair; a reversed gradient is far enough to stay out of it."""
    import os
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = _mk(app, n_keep=0)
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        spec = [('dupA.png', False), ('dupB.png', False), ('other.png', True)]
        for fn, rev in spec:
            _grad_png(os.path.join(d, fn), reverse=rev)
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, filename=fn, status='keep', framing='body',
                caption='a distinct descriptive caption with a fair number of words'))
        svc.db.session.commit()
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert len(r['dup_pairs']) == 1
    pair = r['dup_pairs'][0]
    assert {pair['a']['filename'], pair['b']['filename']} == {'dupA.png', 'dupB.png'}
    assert 'other.png' not in (pair['a']['filename'], pair['b']['filename'])
    assert any('near-duplicate' in w for w in r['warnings'])


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


def test_preflight_checks_and_verdict_mirror_findings(app, monkeypatch):
    """`checks`/`verdict` (pastille de préparation) reflètent la même passe que
    blockers/warnings : fail → blocked, warn seul → warnings, rien → ready.
    Les lignes en défaut portent une cible de section (gf-*) pour le saut."""
    from app.services import lora_training as lt
    from app import capabilities
    with app.app_context():
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
        # 🔴 blocked : sous le plancher famille
        r = lt.training_preflight(LOCAL_USER, _mk(app, n_keep=8).id)
        assert r['verdict'] == 'blocked'
        img = next(c for c in r['checks'] if c['id'] == 'images')
        assert img['status'] == 'fail' and img['target'] == 'gf-generate'
        # 🟡 warnings : au-dessus du plancher, sous la reco (aucun fail)
        r2 = lt.training_preflight(LOCAL_USER, _mk(app, n_keep=15, framing='body').id)
        assert r2['verdict'] == 'warnings'
        assert not any(c['status'] == 'fail' for c in r2['checks'])
        # 🟢 ready : dataset propre (reco atteinte, body, captions variées)
        ds3 = _mk(app, n_keep=20, framing='body',
                  caption='full body shot of the subject walking through a sunny park wearing jeans')
        r3 = lt.training_preflight(LOCAL_USER, ds3.id)
        assert r3['verdict'] == 'ready'
        assert all(c['status'] == 'ok' for c in r3['checks'])


def test_preflight_uncaptioned_kept_is_warning_not_blocker(app, monkeypatch):
    """Une gardée sans caption : WARN (plus un mur) — le launch demande un
    confirm « train anyway » (UNCAPTIONED: dans assert_trainable) au lieu de
    refuser. Demande utilisateur : pouvoir expérimenter sans captions."""
    from app.services import lora_training as lt
    from app import capabilities
    with app.app_context():
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: None)
        ds = _mk(app, n_keep=13, framing='body',
                 extra_rows=[{'filename': 'nocap.webp', 'status': 'keep', 'framing': 'body'}])
        r = lt.training_preflight(LOCAL_USER, ds.id)
    cap = next(c for c in r['checks'] if c['id'] == 'captioned')
    assert cap['status'] == 'warn' and cap['target'] == 'gf-images' and '1/14' in cap['detail']
    assert r['verdict'] == 'warnings'
    assert r['blockers'] == []
    assert any('no caption' in w for w in r['warnings'])


def test_assert_trainable_uncaptioned_is_confirmable(app):
    """Le refus sans-caption porte le marqueur UNCAPTIONED: (déclencheur du
    confirm côté front) et allow_uncaptioned=True le lève."""
    import pytest
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app, n_keep=13, framing='body',
                 extra_rows=[{'filename': 'nocap.webp', 'status': 'keep', 'framing': 'body'}])
        with pytest.raises(ValueError, match='^UNCAPTIONED:'):
            lt.assert_trainable(ds.id)
        lt.assert_trainable(ds.id, allow_uncaptioned=True)   # confirm → passe


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
