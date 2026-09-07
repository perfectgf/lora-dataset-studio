"""The preflight lane: which warnings belong to a run that executes ELSEWHERE.

A cloud launch spends real money, and until now it opened no preflight at all —
the blockers still applied (cloud_training calls assert_trainable), but the whole
WARNING tier and its fix-in-place modal were skipped: leaking captions,
near-duplicates, untriaged images silently left out of the paid run.

Turning it on wholesale would have been worse than useless, though. Two of the
eleven rows read THIS machine — its GPU memory, its ai-toolkit torch build — and
on a cloud-only install (no local training environment at all) they would fire on
every single launch. A warning that cries wolf destroys the credibility of the
ten that don't, so `lane='cloud'` drops them, rows AND warning lines.

The inverse trap is face_mask: it too is a read of the local install, but the
masks are GENERATED locally at export and uploaded with the images — missing
InsightFace here means the PAID run trains unmasked. Local origin, cloud
consequence: it stays.
"""
import os
import random
from unittest.mock import patch

from PIL import Image

import pytest

from app.config import LOCAL_USER, save_config

@pytest.fixture(autouse=True)
def _rembg_installed(monkeypatch):
    """Person masking is ON by default, so the `person_mask` row now depends on
    whether rembg is installed ON THIS MACHINE. Pin it available so these tests
    describe the dataset, not the box they run on (test_masked_dataset_setting.py
    owns the missing-rembg case)."""
    from app.services import person_mask
    monkeypatch.setattr(person_mask, 'is_available', lambda: True)

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc

# A Blackwell torch probe (no sm_120 kernels) — same shape as the diagnostics use.
BLACKWELL = {'torch': '2.5.1+cu124', 'cuda': '12.4', 'capability': [12, 0],
             'arch_list': ['sm_50', 'sm_80', 'sm_86', 'sm_90'],
             'device_name': 'NVIDIA GeForce RTX 5090'}


def _noise(seed):
    rnd = random.Random(seed)
    im = Image.new('RGB', (64, 64))
    im.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
                for _ in range(64 * 64)])
    return im


def _dataset(tmp_path, n=20, train_type='krea', **kw):
    """Enough kept images to clear every family floor, so nothing in these tests
    complains about the count."""
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    ds = svc.create_dataset(LOCAL_USER, 'Lane', 'lane_trig', train_type=train_type, **kw)
    img_dir = svc._dataset_dir(ds.id)
    for i in range(n):
        fn = f'k{i}.png'
        # Deterministic NOISE, not flat colours: a flat image dHashes to all-zeros,
        # so twenty of them would be twenty near-duplicates of each other.
        _noise(i).save(os.path.join(img_dir, fn))
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, status='keep', filename=fn, framing='body',
            caption=f'a person standing outdoors in a wide open field, photo {i}'))
    db.session.commit()
    return ds


def _ids(report):
    return {c['id'] for c in report['checks']}


def _machine_probes(vram=8):
    """Both machine-scope probes firing at once: a small GPU and a torch build
    with no kernels for it."""
    from app import capabilities
    return (patch.object(capabilities, 'gpu_vram_gb', return_value=vram),
            patch.object(capabilities, 'aitoolkit_torch_info', return_value=BLACKWELL))


# --- the machine rows: present locally, gone in the cloud ---------------------

def test_local_lane_still_reports_the_machine_rows(app, tmp_path):
    """The contract that must NOT move: with no lane at all, the payload is the
    historical one — GPU memory and torch build included."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path)
        vram_p, torch_p = _machine_probes()
        with vram_p, torch_p:
            r = lt.training_preflight(LOCAL_USER, ds.id)
    assert {'vram', 'torch_arch'} <= _ids(r)
    assert any('GB of VRAM' in w for w in r['warnings'])
    assert any('no kernel image' in w or 'sm_120' in w for w in r['warnings'])
    assert r['lane'] == 'local'


def test_cloud_lane_never_shows_vram_or_torch_arch(app, tmp_path):
    """THE guard-rail of this feature. A rented pod's GPU has nothing to do with
    the one in this box; showing these rows would teach people to click through
    every warning without reading it."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path)
        vram_p, torch_p = _machine_probes()
        with vram_p, torch_p:
            r = lt.training_preflight(LOCAL_USER, ds.id, lane='cloud')
    assert 'vram' not in _ids(r)
    assert 'torch_arch' not in _ids(r)
    # And their WARNING LINES too — the modal renders that flat string list
    # verbatim, so filtering only `checks` would have left the noise on screen.
    assert not any('GB of VRAM' in w for w in r['warnings'])
    assert not any('no kernel image' in w or 'sm_120' in w for w in r['warnings'])
    assert r['lane'] == 'cloud'


def test_cloud_verdict_ignores_a_machine_only_complaint(app, tmp_path):
    """A dataset whose ONLY problem is this machine's GPU is 🟢 for a cloud run."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path)
        vram_p, torch_p = _machine_probes()
        with vram_p, torch_p:
            local = lt.training_preflight(LOCAL_USER, ds.id)
            cloud = lt.training_preflight(LOCAL_USER, ds.id, lane='cloud')
    assert local['verdict'] == 'warnings'
    assert cloud['verdict'] == 'ready'
    assert not cloud['warnings']


# The silent-CPU trap (acontentsheltie, Discord, RTX 3090): a torch that cannot
# see the card. Same probe, same lane rule as the two rows above.
CPU_ONLY = {'torch': '2.13.0+cpu', 'cuda': None, 'cuda_available': False, 'cuda_reason': '',
            'capability': None, 'device_name': None, 'arch_list': [],
            'torchvision': '0.28.0+cpu', 'accelerator_device': 'cpu'}


def test_the_cpu_only_torch_row_blocks_locally_and_vanishes_in_the_cloud(app, tmp_path):
    """Locally it is a blocker the ack cannot waive — ai-toolkit would not fail,
    it would train on the CPU for days. A rented pod brings its own torch: the
    row, its warning line and the verdict all drop."""
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path)
        with patch.object(capabilities, 'gpu_vram_gb', return_value=24), \
             patch.object(capabilities, 'aitoolkit_torch_info', return_value=CPU_ONLY):
            local = lt.training_preflight(LOCAL_USER, ds.id)
            cloud = lt.training_preflight(LOCAL_USER, ds.id, lane='cloud')
    assert 'torch_cuda' in _ids(local)
    assert local['verdict'] == 'blocked' and local['can_override'] is False
    assert any('CPU-only' in b for b in local['blockers'])     # what the launch button reads
    assert any('CPU-only' in w for w in local['warnings'])
    assert 'torch_cuda' not in _ids(cloud)
    assert cloud['blockers'] == []
    assert not any('CPU-only' in w for w in cloud['warnings'])
    assert cloud['verdict'] == 'ready'


def test_every_row_declares_its_scope(app, tmp_path):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path)
        vram_p, torch_p = _machine_probes()
        with vram_p, torch_p:
            r = lt.training_preflight(LOCAL_USER, ds.id)
    scopes = {c['id']: c['scope'] for c in r['checks']}
    assert scopes.get('vram') == 'machine'
    assert scopes.get('torch_arch') == 'machine'
    assert all(v in ('machine', 'dataset') for v in scopes.values())
    assert {k for k, v in scopes.items() if v == 'machine'} == {'vram', 'torch_arch'}


# --- the dataset rows: the whole point of turning this on for the cloud -------

def test_cloud_lane_still_reports_what_the_pod_will_actually_train(app, tmp_path):
    """Leaking captions, near-duplicates and untriaged images ride to the pod in
    the export — they are exactly what the paid run would learn."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path, n=20)
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        # a leaking caption, a near-duplicate pair, and an untriaged image
        rows[0].caption = 'a woman with long blonde hair and green eyes, freckled skin'
        img_dir = svc._dataset_dir(ds.id)
        Image.open(os.path.join(img_dir, rows[1].filename)).save(
            os.path.join(img_dir, 'dupe.png'))
        db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                        filename='dupe.png', framing='body',
                                        caption=rows[1].caption))
        db.session.add(FaceDatasetImage(dataset_id=ds.id, status='pending',
                                        filename='waiting.png', framing='body'))
        db.session.commit()
        vram_p, torch_p = _machine_probes()
        with vram_p, torch_p:
            r = lt.training_preflight(LOCAL_USER, ds.id, lane='cloud')
    ids = _ids(r)
    assert 'leaks' in ids and 'duplicates' in ids and 'triage' in ids
    assert 'images' in ids and 'composition' in ids
    assert r['verdict'] == 'warnings'
    # The modal's drill-down payload survives the lane filter: without these the
    # cloud modal would be a dialog box with nothing to act on.
    assert r['leak_images'] and r['dup_pairs']
    assert any(li['filename'] == rows[0].filename for li in r['leak_images'])


def test_face_mask_is_reported_on_the_cloud_lane(app, tmp_path, monkeypatch):
    """The inverse trap. InsightFace runs LOCALLY at export and the masks are
    uploaded with the images — its absence means the PAID run trains unmasked."""
    from app.services import lora_training as lt
    from app.services import face_mask
    with app.app_context():
        ds = _dataset(tmp_path, kind='concept', concept_desc='balancing a spoon')
        lt.update_train_settings(LOCAL_USER, ds.id, {'mask_faces': True})
        monkeypatch.setattr(face_mask, 'is_available', lambda: False)
        vram_p, torch_p = _machine_probes()
        with vram_p, torch_p:
            r = lt.training_preflight(LOCAL_USER, ds.id, lane='cloud')
    row = next((c for c in r['checks'] if c['id'] == 'face_mask'), None)
    assert row is not None and row['status'] == 'warn'
    assert row['scope'] == 'dataset'
    assert any('unmasked' in w for w in r['warnings'])


def test_blockers_are_lane_independent(app, tmp_path):
    """Blockers were never the gap — cloud_training already calls assert_trainable.
    The lane filter must not touch them."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(tmp_path, n=3)                   # krea floor is well above 3
        local = lt.training_preflight(LOCAL_USER, ds.id)
        cloud = lt.training_preflight(LOCAL_USER, ds.id, lane='cloud')
    assert local['blockers'] and cloud['blockers'] == local['blockers']
    assert cloud['verdict'] == 'blocked'
    assert cloud['can_override'] == local['can_override']


# --- the route ----------------------------------------------------------------

def test_route_without_lane_is_unchanged(app, client, tmp_path):
    from app import capabilities
    with app.app_context():
        ds = _dataset(tmp_path)
        dsid = ds.id
    with patch.object(capabilities, 'probe',
                      return_value={'aitoolkit': {'valid': True}, 'cloud_training': True}):
        plain = client.get(f'/api/dataset/{dsid}/train/preflight').get_json()
        local = client.get(f'/api/dataset/{dsid}/train/preflight?lane=local').get_json()
    assert plain['ok'] and plain['lane'] == 'local'
    assert plain['checks'] == local['checks']


def test_route_lane_cloud_filters_the_machine_rows(app, client, tmp_path):
    from app import capabilities
    with app.app_context():
        ds = _dataset(tmp_path)
        dsid = ds.id
    vram_p, torch_p = _machine_probes()
    with patch.object(capabilities, 'probe',
                      return_value={'aitoolkit': {'valid': True}, 'cloud_training': True}), \
            vram_p, torch_p:
        cloud = client.get(f'/api/dataset/{dsid}/train/preflight?lane=cloud').get_json()
        local = client.get(f'/api/dataset/{dsid}/train/preflight').get_json()
    assert {'vram', 'torch_arch'} & {c['id'] for c in local['checks']}
    assert not ({'vram', 'torch_arch'} & {c['id'] for c in cloud['checks']})


def test_cloud_only_install_still_gets_its_cloud_preflight(app, client, tmp_path):
    """No ai-toolkit on this machine, a vast.ai key: the cloud preflight must
    ANSWER. The historical gate 409'd, and the caller reads a non-200 as "no
    objection" — which would have made the feature a silent no-op exactly where
    money is at stake."""
    from app import capabilities
    with app.app_context():
        ds = _dataset(tmp_path)
        dsid = ds.id
    with patch.object(capabilities, 'probe',
                      return_value={'aitoolkit': {'valid': False}, 'cloud_training': True}):
        cloud = client.get(f'/api/dataset/{dsid}/train/preflight?lane=cloud')
        local = client.get(f'/api/dataset/{dsid}/train/preflight')
    assert cloud.status_code == 200 and cloud.get_json()['ok']
    assert local.status_code == 409          # the local lane still says what's missing
