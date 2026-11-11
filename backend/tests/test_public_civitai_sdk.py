"""📤 Publish to Civitai — the engine, the link store and the routes.

The network is replaced through the ONE seam (`civitai_publish._transport`):
the ported suite uses the actual plugin and SDK and proves the call chain, the envelopes and the guards, never Civitai's
uptime. The shapes mirror the answers the live site gave when the chain was
exercised for real (superjson `{"json": …}` for mutations, devalue for GET
queries, `{urls, bucket, key, uploadId}` for /api/upload, `{id, uploadURL}` for
/api/v1/image-upload).
"""
import io
import json
import os
import struct
import time

import pytest
from PIL import Image

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bundled' / 'civitai_publish'))
from lds_civitai_publish import publish as cp
from test_public_sdk_integration import host, activate  # noqa: F401 — shared fixture


@pytest.fixture(autouse=True)
def forbid_external_io(monkeypatch):
    def refuse(*_args, **_kwargs):
        pytest.fail('Publisher SDK tests must not connect or launch a worker')
    monkeypatch.setattr('socket.socket.connect', refuse)
    monkeypatch.setattr('socket.socket.connect_ex', refuse)
    monkeypatch.setattr('socket.create_connection', refuse)
    monkeypatch.setattr('requests.sessions.Session.request', refuse)
    monkeypatch.setattr('subprocess.Popen', refuse)


@pytest.fixture
def app(host):
    loaded = activate(host, {'civitai_publish'})
    assert loaded.records['civitai_publish'].state == 'loaded'
    return host[0]


@pytest.mark.parametrize('endpoint', ['/api/civitai/links',
    '/api/civitai/checkpoint/1/2500/publish-model', '/api/civitai/images/publish'])
@pytest.mark.parametrize('payload', [['bad'], 'bad', 1, True, None, False, []])
def test_write_payloads_require_a_json_object(client, app, civitai, endpoint, payload):
    app.config['PROPAGATE_EXCEPTIONS'] = False
    response = client.post(endpoint, json=payload)
    assert response.status_code == 400
    assert response.is_json
    assert civitai.calls == []


@pytest.mark.parametrize('absolute', [False, True])
def test_gallery_export_refuses_a_file_outside_its_dataset_before_transport(
        client, app, civitai, tmp_path, absolute):
    from app.extensions import db
    from app.services import face_dataset_service as fds
    from lds_sdk.gallery_exports import GalleryExports
    with app.app_context():
        dataset_id = _create(client)
        rec = _record(db, dataset_id)
        image = _image(db, dataset_id, rec)
        outside = tmp_path / 'outside.png'
        Image.new('RGB', (2, 2), 'red').save(outside)
        image.filename = str(outside) if absolute else os.path.relpath(outside, fds._dataset_dir(dataset_id))
        db.session.commit()
        link = cp.save_link(rec.id, 2500, NUMBERED, dataset_id, model_id=1, version_id=2)
        exports = GalleryExports('local')
        assert exports.path(image.id) is None
        assert exports.finished([image.id]) == ()
        with pytest.raises(cp.CivitaiPublishError) as err:
            cp.post_images([image.id], link, KEY)
        assert err.value.code == 'image_missing'
        assert civitai.calls == []


def test_training_history_keeps_records_owned_and_cloud_tables_distinct(app, monkeypatch):
    from dataclasses import FrozenInstanceError
    from app.extensions import db
    from app.models import CloudTrainingRun, FaceDataset
    from app.services import cloud_training, lora_training
    from lds_sdk.training import TrainingHistory
    with app.app_context():
        ds = FaceDataset(name='Owner', trigger_word='owner', user_id='owner')
        db.session.add(ds)
        db.session.commit()
        rec = _record(db, ds.id, source='cloud')
        cloud = CloudTrainingRun(dataset_id=ds.id, dataset_table='face_dataset', status='done')
        db.session.add(cloud)
        db.session.commit()
        rec.cloud_run_id = cloud.id
        db.session.commit()
        calls = []
        monkeypatch.setattr(cloud_training, 'run_checkpoint_files',
                            lambda run: calls.append(run.id) or {NUMBERED: 'fixture.safetensors'})
        owner, other = TrainingHistory('owner'), TrainingHistory('other')
        snapshot = owner.record(rec.id)
        assert snapshot.id == rec.id
        with pytest.raises(FrozenInstanceError):
            snapshot.dataset_id = 2
        assert other.record(rec.id) is None
        assert other.cloud_checkpoint_files(rec.id) is None
        assert owner.cloud_checkpoint_files(rec.id) == {NUMBERED: 'fixture.safetensors'}
        cloud.dataset_table = 'video_dataset'
        db.session.commit()
        assert owner.cloud_checkpoint_files(rec.id) is None
        cloud.dataset_table = 'face_dataset'
        cloud.train_params = json.dumps({'dataset_deleted': True})
        db.session.commit()
        assert owner.cloud_checkpoint_files(rec.id) is None
        assert calls == [cloud.id]
        monkeypatch.setattr(lora_training, 'list_checkpoints',
                            lambda *_a, **_k: pytest.fail('Foreign record must not inspect files'))
        monkeypatch.setattr(lora_training, 'checkpoint_file_path',
                            lambda *_a, **_k: pytest.fail('Foreign record must not resolve files'))
        assert other.local_checkpoints(rec.id) == ()
        assert other.local_checkpoint_path(rec.id, NUMBERED) is None


def test_finished_gallery_exports_require_owned_existing_done_images(client, app):
    from app.extensions import db
    from lds_sdk.gallery_exports import GalleryExports
    with app.app_context():
        ds = _create(client)
        good = _image(db, ds)
        pending = _image(db, ds, step=1000, status='pending')
        missing = _image(db, ds, step=500)
        os.remove(cp._image_path(missing))
        assert [r.id for r in GalleryExports('local').finished([good.id, pending.id, missing.id])] == [good.id]
        assert GalleryExports('other').finished([good.id]) == ()


def test_a_derived_picture_exports_the_generation_it_was_made_from(client, app):
    """A camera angle, an upscale or an improve keeps its own columns for the pass
    that made it. An export names the render it comes from, through a chain; a link
    that is missing or names another render leaves the picture its own facts."""
    from app.extensions import db
    from lds_sdk.gallery_exports import GalleryExports
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        source = _image(db, ds, rec, step=2500)
        source.strength, source.z_model = 0.8, 'base.safetensors'
        source.extra_loras = json.dumps([{'filename': 'second.safetensors', 'strength': 0.85}])
        pass_loras = json.dumps([{'filename': 'details.safetensors', 'strength': 0.6}])

        def derived(step, parent, **own):
            row = _image(db, ds, rec, step=step)
            row.checkpoint, row.parent_image_id = source.checkpoint, parent
            row.derivation_kind, row.z_model, row.extra_loras = 'camera_angle', None, pass_loras
            for key, value in own.items():
                setattr(row, key, value)
            db.session.commit()
            return row

        angle = derived(2400, source.id)
        chain = derived(2300, angle.id)
        orphan = derived(2200, 999999)
        stale = derived(2100, source.id, checkpoint='another.safetensors')
        exports = {r.id: r for r in GalleryExports('local').finished(
            [source.id, angle.id, chain.id, orphan.id, stale.id])}
        for row in (angle, chain):
            assert (exports[row.id].z_model, exports[row.id].extra_loras, exports[row.id].strength) == (
                'base.safetensors', source.extra_loras, 0.8)
            assert exports[row.id].step == 2500
        for row in (orphan, stale):
            assert (exports[row.id].z_model, exports[row.id].extra_loras) == (None, pass_loras)
        # Nothing is written back: the row still describes its own pass.
        assert db.session.get(type(angle), angle.id, populate_existing=True).z_model is None


def test_civitai_registered_hooks_use_the_existing_link_table(client, app, civitai):
    from app.extensions import db
    from app.models import CivitaiLink
    from app.plugins.hooks import run_filter, run_hook
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        link = cp.save_link(rec.id, 2500, NUMBERED, ds, model_id=1, version_id=2, model_name='Fixture')
        checkpoints = run_filter('lineage.checkpoints', [{'filename': NUMBERED}], rec.id, strict=True)
        assert checkpoints[0]['civitai']['model_name'] == 'Fixture'
        assert run_filter('training_run.delete_summary', {}, rec, strict=True)['civitai_links_detached'] == 1
        run_hook('training_run.delete', rec)
        db.session.flush()
        main_link = db.session.get(CivitaiLink, link.id, populate_existing=True)
        assert main_link.record_id is None and main_link.dataset_id == ds
        run_hook('dataset.delete', ds)
        db.session.flush()
        assert db.session.get(CivitaiLink, link.id, populate_existing=True) is None
        assert civitai.calls == []


def test_h3_targets_are_exact_public_host_symbols():
    from lds_sdk import h3_targets
    from app.services import video_targets
    assert len(h3_targets.__all__) == 8
    for name in h3_targets.__all__:
        assert getattr(h3_targets, name) is getattr(video_targets, name)

# The `civitai` fixture turns `time.sleep` into a no-op so the service's retry
# back-off costs the suite nothing — `cp.time` IS the time module, so that
# reaches every sleep in the process, this file's own included. A poll that
# has to wait for a background thread keeps the genuine one, taken before any
# fixture runs — and the genuine clock with it: one test below freezes
# `time.monotonic`, and a deadline read off a frozen clock never arrives.
_real_sleep = time.sleep
_real_monotonic = time.monotonic

KEY = 'test-key-not-real'
SERVER_DATE = 'Wed, 02 Sep 2026 12:00:00 GMT'


# --- fixtures -------------------------------------------------------------------

class FakeCivitai:
    """Answers the site's endpoints from canned shapes and records every call in
    order, so a test can assert the CHAIN and the BODIES, not just the outcome."""

    def __init__(self):
        self.calls = []          # (method, url, headers, json_body|data)
        self.model_id = 2755270
        self.version_id = 3100001
        self.post_id = 30670842
        self.fail = {}           # url fragment -> (status, body dict)

    def __call__(self, method, url, headers=None, data=None, json_body=None, timeout=None):
        self.calls.append((method, url, headers or {}, json_body if json_body is not None else data))
        for frag, (status, body) in self.fail.items():
            if frag in url:
                return status, {'date': SERVER_DATE}, json.dumps(body).encode()
        if url.startswith('https://s3.example/part/') or url.startswith('https://b2.example/'):
            return 200, {'etag': f'"etag-{len(self.calls)}"'}, b''
        if url.endswith('/api/v1/me'):
            return 200, {'date': SERVER_DATE}, json.dumps({'id': 1, 'username': 'creator'}).encode()
        if '/api/trpc/' in url and method == 'POST':
            proc = url.split('/api/trpc/')[1]
            answer = {
                'model.upsert': {'id': self.model_id},
                'modelVersion.upsert': {'id': self.version_id},
                'modelFile.upsert': {'id': 777},
                'model.publish': {'id': self.model_id},
                'post.create': {'id': self.post_id},
                'post.addImage': {'id': 9000 + len(self.calls)},
                'post.update': {'id': self.post_id},
            }.get(proc, {'id': 1})
            return 200, {'date': SERVER_DATE}, json.dumps({'result': {'data': {'json': answer}}}).encode()
        if '/api/trpc/model.getById' in url:
            # devalue-encoded, like every GET query of the site
            arr = [{'id': 1, 'name': 2, 'type': 3, 'nsfw': 4, 'status': 5, 'modelVersions': 6},
                   self.model_id, 'Nova', 'LORA', False, 'Published',
                   [7], {'id': 8, 'name': 9, 'baseModel': 10, 'status': 5},
                   self.version_id, 'v1.0', 'Krea 2']
            return 200, {'date': SERVER_DATE}, json.dumps({'result': {'data': json.dumps(arr)}}).encode()
        if url.endswith('/api/upload'):
            n = max(1, -(-json_body['size'] // cp._CHUNK))
            return 200, {}, json.dumps({
                'urls': [{'url': f'https://s3.example/part/{i + 1}', 'partNumber': i + 1} for i in range(n)],
                'bucket': 'b', 'key': 'k/lora.safetensors', 'uploadId': 'u1'}).encode()
        if url.endswith('/api/upload/complete'):
            return 200, {}, json.dumps('https://r2.example/civitai/k/lora.safetensors').encode()
        if url.endswith('/api/v1/image-upload'):
            return 200, {}, json.dumps({'id': f'uuid-{len(self.calls)}',
                                        'uploadURL': 'https://b2.example/upload'}).encode()
        raise AssertionError(f'unexpected call {method} {url}')

    def procs(self):
        return [u.split('/api/trpc/')[1] for _m, u, _h, _b in self.calls if '/api/trpc/' in u]

    def body_of(self, proc):
        for _m, u, _h, b in self.calls:
            if u.endswith('/api/trpc/' + proc):
                return b
        return None


@pytest.fixture
def civitai(monkeypatch):
    fake = FakeCivitai()
    monkeypatch.setattr(cp, '_transport', fake)
    monkeypatch.setattr(cp, 'api_key', lambda: KEY)
    cp._who_cache.update(key=None, at=0.0, value=None)
    cp._server_clock.update(date=None, at=0.0)
    monkeypatch.setattr(cp.time, 'sleep', lambda *_a: None)
    return fake


def _safetensors(path, metadata=None, dtype='F16', n=3):
    """A tiny but REAL safetensors container: header + n zero tensors."""
    header = {f't{i}': {'dtype': dtype, 'shape': [2], 'data_offsets': [i * 4, i * 4 + 4]}
              for i in range(n)}
    if metadata is not None:
        header['__metadata__'] = metadata
    blob = json.dumps(header).encode()
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(blob)) + blob + b'\0' * (4 * n))
    return path


def _create(client, name='Nova', trigger='nova'):
    from app.services import face_dataset_service
    from app.config import LOCAL_USER
    with client.application.app_context():
        return face_dataset_service.create_dataset(LOCAL_USER, name, trigger).id


def _record(db, dataset_id, family='krea', source='local', version=2, steps=3000,
            base_model='', variant=None):
    from app.models import FaceDataset, TrainingRunRecord
    if db.session.get(FaceDataset, dataset_id) is None:
        db.session.add(FaceDataset(id=dataset_id, name='Fixture dataset', trigger_word='fixture'))
        db.session.commit()
    rec = TrainingRunRecord(dataset_id=dataset_id, family=family, source=source,
                            fingerprint='f', version=version, steps=steps,
                            base_model=base_model, variant=variant,
                            manifest=json.dumps([[1, 'a', 'b'], [2, 'c', 'd']]),
                            settings=json.dumps({'rank': 16, 'alpha': 16, 'network_type': 'lora'}))
    db.session.add(rec)
    db.session.commit()
    return rec


NUMBERED = 'lora_nova_000002500.safetensors'
FINAL = 'lora_nova.safetensors'


def _image(db, dataset_id, rec=None, step=2500, **kw):
    from app.models import LoraTestImage
    from app.services import face_dataset_service as fds
    d = fds._dataset_path(dataset_id)
    os.makedirs(d, exist_ok=True)
    fn = kw.pop('filename', f'cell_{step}.png')
    im = Image.new('RGB', (64, 48), (200, 30, 30))
    # A PNG WITH a text chunk — what a ComfyUI output carries — so the test can
    # prove the staged copy has none.
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text('workflow', '{"nodes": "C:\\\\Users\\\\someone\\\\ComfyUI"}')
    im.save(os.path.join(d, fn), 'PNG', pnginfo=info)
    img = LoraTestImage(dataset_id=dataset_id,
                        checkpoint=kw.pop('checkpoint', 'krea\\lora_nova_000002500_Krea-2-Raw_rl1_v2.safetensors'),
                        strength=0.85, status=kw.pop('status', 'done'), filename=fn, seed=208607443,
                        prompt='a candid portrait, film grain', negative='blurry',
                        cfg=3.5, steps=20, sampler='euler', scheduler='simple',
                        z_model='krea\\Krea-2-Turbo.safetensors',
                        extra_loras=json.dumps([{'filename': 'krea\\film-look.safetensors', 'strength': 0.6}]),
                        record_id=rec.id if rec else None, step=step if rec else None, **kw)
    db.session.add(img)
    db.session.commit()
    return img


# --- envelopes ------------------------------------------------------------------

def test_superjson_names_its_dates_and_nothing_else():
    assert cp._superjson({'id': 1}) == {'json': {'id': 1}}
    body = cp._superjson({'id': 1, 'publishedAt': '2026-09-02T10:00:00Z'}, date_fields=('publishedAt',))
    assert body == {'json': {'id': 1, 'publishedAt': '2026-09-02T10:00:00Z'},
                    'meta': {'values': {'publishedAt': ['Date']}}}
    # A date field that is not in the input must not be announced.
    assert 'meta' not in cp._superjson({'id': 1}, date_fields=('publishedAt',))


def test_devalue_and_superjson_envelopes_both_unwrap():
    assert cp._unwrap_trpc({'json': {'id': 5}}) == {'id': 5}
    arr = [{'id': 1, 'tags': 2, 'when': 3}, 42, [4, 5], ['Date', '2026-09-02'], 'a', 'b']
    assert cp._unwrap_trpc(json.dumps(arr)) == {'id': 42, 'tags': ['a', 'b'], 'when': '2026-09-02'}


def test_a_refused_key_and_a_missing_scope_get_their_own_sentences(civitai):
    civitai.fail['model.upsert'] = (401, {'error': 'unauthorized'})
    with pytest.raises(cp.CivitaiPublishError) as e:
        cp.trpc_mutation('model.upsert', {'name': 'x'}, KEY)
    assert e.value.code == 'auth'
    assert 'Settings' in e.value.message
    civitai.fail['model.upsert'] = (403, {'error': 'forbidden'})
    with pytest.raises(cp.CivitaiPublishError) as e:
        cp.trpc_mutation('model.upsert', {'name': 'x'}, KEY)
    assert e.value.code == 'forbidden'
    assert 'Media write' in e.value.message and 'Models write' in e.value.message
    civitai.fail['model.upsert'] = (400, {'error': {'json': {'message': 'invalid_type at baseModel'}}})
    with pytest.raises(cp.CivitaiPublishError) as e:
        cp.trpc_mutation('model.upsert', {'name': 'x'}, KEY)
    assert 'invalid_type at baseModel' in e.value.message


def test_the_publish_stamp_follows_the_servers_clock_not_the_pcs(civitai, monkeypatch):
    # Before any answer: the PC clock, minus the margin.
    cp._server_clock.update(date=None, at=0.0)
    assert cp._publish_stamp_iso().endswith('Z')
    # After an answer carrying `Date`: that clock, plus the seconds since, minus 15 s.
    cp.trpc_mutation('post.create', {'modelVersionId': 1}, KEY)
    monkeypatch.setattr(cp.time, 'monotonic', lambda: cp._server_clock['at'] + 20)
    assert cp._publish_stamp_iso() == '2026-09-02T12:00:05Z'


# --- naming ----------------------------------------------------------------------

@pytest.mark.parametrize('family,variant,base,expected', [
    ('zimage', 'turbo', '', 'ZImageTurbo'), ('zimage', None, '', 'ZImageTurbo'),
    ('zimage', 'base', '', 'ZImageBase'),
    # De-Turbo is the Turbo weights de-distilled: the LoRA is used on Turbo.
    ('zimage', 'deturbo', '', 'ZImageTurbo'),
    ('krea', 'base', '', 'Krea 2'), ('krea', 'turbo', '', 'Krea 2'),
    ('krea', 'base', 'krea\\some-finetune.safetensors', 'Krea 2'),
    ('sdxl', None, '', 'SDXL 1.0'), ('flux', None, '', 'Flux.1 D'),
    # LDS trains Klein on the official *-base weights, which Civitai files
    # apart from the distilled lineage.
    ('flux2klein', '4b', '', 'Flux.2 Klein 4B-base'), ('flux2klein', '9b', '', 'Flux.2 Klein 9B-base'),
    ('flux2klein', None, '', 'Flux.2 Klein 4B-base'), ('anima', None, '', 'Anima'),
    ('wan', None, '', 'Other'), ('', None, '', 'Other'),
    # A custom base where the site distinguishes lineages: no honest answer.
    ('sdxl', None, 'sdxl\\ponyDiffusionV6XL.safetensors', ''),
    ('zimage', 'turbo', 'zimage\\some-finetune.safetensors', ''),
    ('flux2klein', '9b', 'klein\\custom.safetensors', ''),
])
def test_every_family_names_a_civitai_base_model_or_declines(family, variant, base, expected):
    assert cp.civitai_base_model(family, variant, base) == expected
    assert expected == '' or expected in cp.CIVITAI_BASE_MODELS


def test_the_klein_base_lineages_are_offered_in_the_select():
    for s in ('Flux.2 Klein 4B-base', 'Flux.2 Klein 9B-base', 'Pony', 'Illustrious', 'NoobAI'):
        assert s in cp.CIVITAI_BASE_MODELS


@pytest.mark.parametrize('ref,expected', [
    ('https://civitai.com/models/2755270/nova-krea-2', (2755270, None)),
    ('https://civitai.red/models/2755270?modelVersionId=3100001', (2755270, 3100001)),
    ('https://civitai.com/models/2755270/nova?modelVersionId=3100001&x=1', (2755270, 3100001)),
    ('civitai.com/models/12/', (12, None)),
    ('2755270', (2755270, None)),
    ('https://civitai.com/images/999', None),
    ('not a url', None), ('', None), (None, None),
])
def test_a_pasted_page_address_resolves_to_its_ids(ref, expected):
    assert cp.parse_model_ref(ref) == expected


def test_the_public_stem_drops_the_deploy_tag():
    assert cp._public_stem('krea\\lora_nova_000002500_Krea-2-Raw_rl1_v2.safetensors') == 'lora_nova_000002500_Krea-2-Raw'
    # A cloud run is tagged `_rc<pod run>` — the tag every real cloud deploy carries.
    assert cp._public_stem('krea\\lora_sushi_tv_000001500_Krea-2-Raw_rc158_v1.safetensors') == 'lora_sushi_tv_000001500_Krea-2-Raw'
    assert cp._public_stem('lora_nova.safetensors') == 'lora_nova'
    assert cp._step_block('krea\\lora_sushi_tv_000001500_Krea-2-Raw_rc158_v1.safetensors') == '000001500'
    assert cp._step_block('lora_nova_Krea-2-Raw_rc158_v1.safetensors') is None


def test_the_link_host_setting_never_produces_a_link_to_nowhere(app, monkeypatch):
    from app import config as cfg
    with app.app_context():
        assert cp.link_host() == 'civitai.com'
        cfg.save_config({'civitai': {'link_host': 'civitai.red'}})
        assert cp.link_host() == 'civitai.red'
        assert cp.model_url(5, 6) == 'https://civitai.red/models/5?modelVersionId=6'
        assert cp.model_wizard_url(5) == 'https://civitai.red/models/5/wizard?step=1'
        cfg.save_config({'civitai': {'link_host': 'evil.example'}})
        assert cp.link_host() == 'civitai.com'


# --- the checkpoint file ----------------------------------------------------------

def test_the_file_is_inspected_header_only_and_a_home_path_in_its_metadata_is_named(tmp_path):
    clean = _safetensors(str(tmp_path / 'clean.safetensors'),
                         {'ss_output_name': 'lora_nova', 'training_info': '{"epoch": 23}',
                          'software': '{"name": "ai-toolkit"}'})
    info = cp.inspect_checkpoint(clean)
    assert info['leaks'] == []
    assert info['fp'] == 'fp16' and info['tensors'] == 3 and info['epoch'] == 23
    assert info['software'] == 'ai-toolkit'
    leaky = _safetensors(str(tmp_path / 'leaky.safetensors'),
                         {'ss_output_dir': 'C:\\Users\\someone\\ai-toolkit\\output',
                          'contact': 'someone@example.com', 'fine': 'x'}, dtype='BF16')
    info = cp.inspect_checkpoint(leaky)
    assert info['leaks'] == ['contact', 'ss_output_dir']
    assert info['fp'] == 'bf16'
    with open(tmp_path / 'not.safetensors', 'wb') as fh:
        fh.write(b'nope')
    with pytest.raises(cp.CivitaiPublishError) as e:
        cp.inspect_checkpoint(str(tmp_path / 'not.safetensors'))
    assert e.value.code == 'not_safetensors'


def test_an_unknown_run_is_said_as_such(app):
    with app.app_context():
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.checkpoint_file_for(999999, 2500)
        assert e.value.code == 'run_missing'


def test_a_cloud_save_is_resolved_by_name_or_by_step_and_a_shared_step_is_refused(app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        run = CloudTrainingRun(dataset_id=1, status='done')
        db.session.add(run)
        db.session.commit()
        rec = _record(db, 1, source='cloud', steps=3000)
        rec.cloud_run_id = run.id
        db.session.commit()
        files = {'lora_nova_000002500.safetensors': str(tmp_path / 'a.safetensors'),
                 'lora_nova.safetensors': str(tmp_path / 'final.safetensors')}
        monkeypatch.setattr(ct, 'run_checkpoint_files', lambda r: files)
        path, got = cp.checkpoint_file_for(rec.id, 2500)
        assert path.endswith('a.safetensors') and got.id == rec.id
        path, _ = cp.checkpoint_file_for(rec.id, 3000, filename='lora_nova.safetensors')
        assert path.endswith('final.safetensors')
        # The final is numbered at the run's target step when the run overshot
        # the last numbered save; when it did NOT, two files share a step…
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.checkpoint_file_for(rec.id, 1000)
        assert e.value.code == 'checkpoint_missing'
        rec.steps = 2500
        db.session.commit()
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.checkpoint_file_for(rec.id, 2500)
        assert e.value.code == 'ambiguous'
        assert 'lora_nova_000002500.safetensors' in e.value.message and 'lora_nova.safetensors' in e.value.message
        # …and naming the file is what resolves it.
        path, _ = cp.checkpoint_file_for(rec.id, 2500, filename='lora_nova.safetensors')
        assert path.endswith('final.safetensors')


def test_a_local_run_that_ended_on_a_numbered_save_refuses_the_bare_step(app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.services import lora_training as lt
    with app.app_context():
        rec = _record(db, 1, steps=2500)
        listed = [
            {'step': 2500, 'filename': NUMBERED, 'run_source': 'local', 'run_id': rec.id},
            {'step': 2500, 'filename': FINAL, 'final': True, 'run_source': 'local', 'run_id': rec.id},
            {'step': 2500, 'filename': 'other.safetensors', 'run_source': 'local', 'run_id': rec.id + 1},
        ]
        monkeypatch.setattr(lt, 'list_checkpoints', lambda *a, **k: listed)
        monkeypatch.setattr(lt, 'checkpoint_file_path',
                            lambda user, ds, fn, *a, **k: str(tmp_path / fn))
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.checkpoint_file_for(rec.id, 2500)
        assert e.value.code == 'ambiguous'
        path, _ = cp.checkpoint_file_for(rec.id, 2500, filename=FINAL)
        assert path.endswith(FINAL)
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.checkpoint_file_for(rec.id, 2500, filename='other.safetensors')
        assert e.value.code == 'checkpoint_missing', 'another run\'s file is not this run\'s'
        # A PICTURE has no file name, only the deployed LoRA name it ran with:
        # its step block picks the numbered save, no block picks the final.
        deployed_numbered = 'krea\\lora_nova_000002500_Krea-2-Raw_rc158_v1.safetensors'
        deployed_final = 'krea\\lora_nova_Krea-2-Raw_rc158_v1.safetensors'
        path, _ = cp.checkpoint_file_for(rec.id, 2500, hint=deployed_numbered)
        assert path.endswith(NUMBERED)
        path, _ = cp.checkpoint_file_for(rec.id, 2500, hint=deployed_final)
        assert path.endswith(FINAL)
        assert cp.resolve_save_filename(rec, 2500, deployed_numbered) == NUMBERED
        assert cp.resolve_save_filename(rec, 2500, deployed_final) == FINAL
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.resolve_save_filename(rec, 2500, None)
        assert e.value.code == 'ambiguous'
        # No save at that step: nothing to name, and that is an answer, not an error.
        assert cp.resolve_save_filename(rec, 1000, deployed_numbered) is None


def test_a_picture_marks_its_page_without_a_file_name(client, app, civitai, monkeypatch):
    """The refusal the maintainer hit from the viewer ("Which save is this? The
    file name is missing."): a picture must resolve its save from the deployed
    name it ran with, and a run whose saves are not on this machine still marks."""
    from app.extensions import db
    from app.services import lora_training as lt
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, steps=2500)
        deployed = 'krea\\lora_nova_000002500_Krea-2-Raw_rl1_v2.safetensors'
        # Saves not listable (no ai-toolkit here): remembered under ''.
        link, _ = cp.link_checkpoint_to_page(rec.id, 2500, 'https://civitai.red/models/2755270/nova',
                                             KEY, hint=deployed)
        assert link.filename == '' and link.model_id == 2755270
        assert cp.link_for(rec.id, 2500, hint=deployed).id == link.id
        assert cp.link_for(rec.id, 2500).id == link.id
        # Saves listable, two at the step: the deployed name's block decides.
        listed = [
            {'step': 2500, 'filename': NUMBERED, 'run_source': 'local', 'run_id': rec.id},
            {'step': 2500, 'filename': FINAL, 'final': True, 'run_source': 'local', 'run_id': rec.id},
        ]
        monkeypatch.setattr(lt, 'list_checkpoints', lambda *a, **k: listed)
        named, _ = cp.link_checkpoint_to_page(rec.id, 2500, '2755270', KEY, hint=deployed)
        assert named.filename == NUMBERED and named.id != link.id
        # …and a picture at that step now finds THE numbered save's link.
        row = _image(db, ds, rec, checkpoint=deployed)
        assert cp.link_for_image(row).id == named.id
        assert cp.link_for(rec.id, 2500, hint=deployed).id == named.id
        # A shared step with nothing to decide it stays a refusal, not a guess.
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.link_checkpoint_to_page(rec.id, 2500, '2755270', KEY)
        assert e.value.code == 'ambiguous'


# --- the draft form -----------------------------------------------------------------

def test_draft_defaults_derive_the_page_from_the_run_and_say_when_the_file_is_missing(client, app, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (_ for _ in ()).throw(
            cp.CivitaiPublishError('checkpoint_missing', 'gone')))
        d = cp.draft_defaults(rec.id, 2500)
        assert d['name'] == 'Nova (Krea 2)'
        assert d['version_name'] == 'v2 · step 2 500'
        assert d['base_model'] == 'Krea 2' and d['base_model_hint'] is None
        assert d['trained_words'] == ['nova']
        assert d['tags'][:2] == ['character', 'krea 2']
        assert d['file'] is None and d['file_error'] == 'gone'
        assert d['link'] is None
        assert 'Trigger word: <code>nova</code>' in d['description']
        assert 'rank 16 / alpha 16' in d['description']
        assert '2 training images (dataset v2)' in d['description']
        assert '2 500 steps' in d['description']


def test_a_custom_base_leaves_the_base_model_open_with_a_hint(client, app, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, family='sdxl', base_model='sdxl\\ponyDiffusionV6XL.safetensors')
        monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (_ for _ in ()).throw(
            cp.CivitaiPublishError('checkpoint_missing', 'gone')))
        d = cp.draft_defaults(rec.id, 2500)
        assert d['base_model'] == ''
        assert 'ponyDiffusionV6XL' in d['base_model_hint'] and 'Pony' in d['base_model_hint']


def test_the_form_is_validated_and_redacted_before_anything_leaves(app):
    with app.app_context():
        with pytest.raises(cp.CivitaiPublishError):
            cp._validate_model_form({'name': ' ', 'base_model': 'Krea 2'})
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp._validate_model_form({'name': 'Nova', 'base_model': ''})
        assert 'Pick the base model' in e.value.message
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp._validate_model_form({'name': 'Nova', 'base_model': 'Krea 3'})
        assert 'Krea 3' in e.value.message
        with pytest.raises(cp.CivitaiPublishError):
            cp._validate_model_form({'name': 'Nova', 'base_model': 'Krea 2', 'file_name': '../x.safetensors'})
        spec = cp._validate_model_form({
            'name': 'Nova', 'base_model': 'Krea 2', 'trained_words': 'nova, NOVA2',
            'tags': ['Character', ' krea 2 '], 'description': 'trained in C:\\Users\\someone\\lds',
            'license': {'allowCommercialUse': False, 'allowNoCredit': False},
        })
        assert spec['trained_words'] == ['nova', 'NOVA2']
        assert spec['tags'] == ['character', 'krea 2']
        assert 'someone' not in spec['description'] and '~' in spec['description']
        assert spec['license']['allowCommercialUse'] == []
        assert spec['license']['allowNoCredit'] is False
        assert spec['publish'] is False


# --- the model page ---------------------------------------------------------------

def test_publish_model_runs_the_chain_as_a_draft_and_links_the_save(client, app, civitai, tmp_path, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        path = _safetensors(str(tmp_path / NUMBERED), {'training_info': '{"epoch": 23}'})
        # A file bigger than one part, so the multipart bookkeeping is exercised.
        monkeypatch.setattr(cp, '_CHUNK', 8)
        monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (path, rec))
        phases = []
        out = cp.publish_model(rec.id, 2500, {
            'name': 'Nova', 'version_name': 'v2', 'base_model': 'Krea 2',
            'trained_words': ['nova'], 'tags': ['character'], 'description': '<p>hi</p>',
            'nsfw': True, 'file_name': 'Nova_v2.safetensors',
        }, KEY, progress=lambda ph, fr: phases.append((ph, round(fr, 2))))
        assert civitai.procs() == ['model.upsert', 'modelVersion.upsert', 'modelFile.upsert']
        m = civitai.body_of('model.upsert')['json']
        assert m['status'] == 'Draft' and m['uploadType'] == 'Created' and m['type'] == 'LORA'
        assert m['nsfw'] is True and m['poi'] is False
        assert m['tagsOnModels'] == [{'name': 'character'}]
        v = civitai.body_of('modelVersion.upsert')['json']
        assert v == {'modelId': civitai.model_id, 'name': 'v2', 'baseModel': 'Krea 2',
                     'trainedWords': ['nova'], 'steps': 2500, 'epochs': 23}
        f = civitai.body_of('modelFile.upsert')['json']
        assert f['url'] == 'https://r2.example/civitai/k/lora.safetensors'
        assert f['name'] == 'Nova_v2.safetensors' and f['modelVersionId'] == civitai.version_id
        assert f['metadata'] == {'format': 'SafeTensor', 'size': 'full', 'fp': 'fp16'}
        puts = [c for c in civitai.calls if c[0] == 'PUT']
        assert len(puts) == -(-os.path.getsize(path) // 8)
        complete = next(b for _m, u, _h, b in civitai.calls if u.endswith('/api/upload/complete'))
        assert [p['PartNumber'] for p in complete['parts']] == list(range(1, len(puts) + 1))
        assert all(p['ETag'].startswith('"etag-') for p in complete['parts'])
        assert phases[0] == ('creating', 0.0) and phases[-1] == ('registering', 1.0)
        assert out['published'] is False
        assert out['url'].endswith(f'/models/{civitai.model_id}/wizard?step=1')
        link = cp.link_for(rec.id, 2500, NUMBERED)
        assert link.model_id == civitai.model_id and link.version_id == civitai.version_id
        assert link.published is False and link.base_model == 'Krea 2'
        assert link.filename == NUMBERED
        assert out['link']['model_name'] == 'Nova' and out['link']['filename'] == NUMBERED


def test_publish_now_adds_model_publish_without_a_pc_clock(client, app, civitai, tmp_path, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        path = _safetensors(str(tmp_path / 'lora.safetensors'), {})
        monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (path, rec))
        out = cp.publish_model(rec.id, 2500, {'name': 'Nova', 'base_model': 'Krea 2', 'publish': True}, KEY)
        assert civitai.procs()[-1] == 'model.publish'
        body = civitai.body_of('model.publish')
        # The server stamps its own "now": a PC clock running ahead would
        # otherwise file the page as SCHEDULED.
        assert body == {'json': {'id': civitai.model_id, 'versionIds': [civitai.version_id]}}
        assert out['published'] is True
        assert out['url'] == cp.model_url(civitai.model_id, civitai.version_id)
        assert cp.link_for(rec.id, 2500, 'lora.safetensors').published is True


def test_the_numbered_save_and_the_final_at_one_step_are_two_links(client, app, civitai, tmp_path, monkeypatch):
    """The refuted design: keyed on (record_id, step) alone, publishing the
    final overwrote the numbered save's link. Now each file is its own row."""
    from app.extensions import db
    from lds_civitai_publish.models import CivitaiLink
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, steps=2500)
        numbered = _safetensors(str(tmp_path / NUMBERED), {})
        final = _safetensors(str(tmp_path / FINAL), {})
        monkeypatch.setattr(cp, 'checkpoint_file_for',
                            lambda rid, step, filename=None, hint=None: (final if filename == FINAL else numbered, rec))
        cp.publish_model(rec.id, 2500, {'name': 'Nova 2500', 'base_model': 'Krea 2'}, KEY, filename=NUMBERED)
        civitai.version_id = 3100002
        cp.publish_model(rec.id, 2500, {'name': 'Nova final', 'base_model': 'Krea 2'}, KEY, filename=FINAL)
        rows = CivitaiLink.query.filter_by(record_id=rec.id, step=2500).order_by(CivitaiLink.id).all()
        assert [(r.filename, r.version_id) for r in rows] == [(NUMBERED, 3100001), (FINAL, 3100002)]
        # By file: exact. By step alone: the NUMBERED save's — what a stamped
        # picture was made with (the final generates with no stamp).
        assert cp.link_for(rec.id, 2500, FINAL).version_id == 3100002
        assert cp.link_for(rec.id, 2500).version_id == 3100001
        assert set(cp.links_for_record(rec.id)) == {NUMBERED, FINAL}


def test_a_checkpoint_whose_metadata_names_the_machine_is_refused_before_any_call(client, app, civitai, tmp_path, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        path = _safetensors(str(tmp_path / 'lora.safetensors'),
                            {'ss_output_dir': '/home/someone/ai-toolkit/output'})
        monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (path, rec))
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.publish_model(rec.id, 2500, {'name': 'Nova', 'base_model': 'Krea 2'}, KEY)
        assert e.value.code == 'file_metadata_leak'
        assert 'ss_output_dir' in e.value.message
        assert civitai.calls == []
        assert cp.link_for(rec.id, 2500, 'lora.safetensors') is None


# --- the link store -------------------------------------------------------------------

def test_marking_a_page_resolves_it_and_remembers_the_version(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        link, page = cp.link_checkpoint_to_page(
            rec.id, 2500, 'https://civitai.red/models/2755270/nova', KEY, filename=NUMBERED)
        assert page['name'] == 'Nova' and page['versions'][0]['base_model'] == 'Krea 2'
        assert link.model_id == 2755270 and link.version_id == civitai.version_id
        assert link.model_name == 'Nova' and link.version_name == 'v1.0'
        assert link.published is True and link.filename == NUMBERED
        # Relinking the same save retargets the ONE row.
        again, _ = cp.link_checkpoint_to_page(rec.id, 2500, '2755270', KEY, filename=NUMBERED)
        assert again.id == link.id
        from lds_civitai_publish.models import CivitaiLink
        assert CivitaiLink.query.count() == 1
        assert cp.links_for_record(rec.id)[NUMBERED]['model_url'].endswith('/models/2755270?modelVersionId=3100001')
        assert [l['id'] for l in cp.links_for_dataset(ds)] == [link.id]
        assert cp.delete_link(link.id) is True
        assert cp.link_for(rec.id, 2500, NUMBERED) is None


def test_a_page_that_is_not_a_lora_or_a_version_not_of_that_page_is_refused(client, app, civitai, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.link_checkpoint_to_page(rec.id, 2500, 'https://civitai.com/models/2755270?modelVersionId=1',
                                       KEY, filename=NUMBERED)
        assert e.value.code == 'no_version'
        # The refusal NAMES the versions the page does have (the maintainer hit
        # this with a mistyped id and had nothing to go on).
        assert 'v1.0 (#3100001)' in e.value.message and 'Look the page up' in e.value.message
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.link_checkpoint_to_page(rec.id, 2500, 'garbage', KEY, filename=NUMBERED)
        assert e.value.code == 'bad_ref'
        monkeypatch.setattr(cp, 'fetch_model_page', lambda *a, **k: {
            'id': 5, 'name': 'A checkpoint', 'type': 'Checkpoint', 'versions': [{'id': 6, 'name': 'v1'}]})
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.link_checkpoint_to_page(rec.id, 2500, '5', KEY, filename=NUMBERED)
        assert e.value.code == 'not_a_lora'




# --- images → a post -------------------------------------------------------------------

def test_a_picture_finds_its_page_through_its_stamp_and_prefers_the_file_it_was_made_with(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, steps=2500)
        final_link = cp.save_link(rec.id, 2500, FINAL, ds, model_id=1, version_id=20)
        numbered_link = cp.save_link(rec.id, 2500, NUMBERED, ds, model_id=1, version_id=10)
        row = _image(db, ds, rec)   # deployed name carries the numbered save's step block
        assert cp.link_for_image(row).id == numbered_link.id
        # A deployed FINAL (no step block) made with a stamp — a legacy name —
        # prefers the step-less link; a prefix test would have picked it for
        # every numbered save too, which is why the block is what is compared.
        made_with_final = _image(db, ds, rec, filename='f.png',
                                 checkpoint='krea\\lora_nova_Krea-2-Raw_rl1_v2.safetensors')
        assert cp.link_for_image(made_with_final).id == final_link.id
        unstamped = _image(db, ds, None, filename='u.png')
        assert cp.link_for_image(unstamped) is None
        # Only the final is linked: a stamped picture at that step still lands somewhere.
        cp.delete_link(numbered_link.id)
        assert cp.link_for_image(row).id == final_link.id


def test_the_image_meta_is_what_actually_ran(client, app):
    from app.extensions import db
    from app.services import face_dataset_service as fds
    from app.config import LOCAL_USER
    with app.app_context():
        ds_id = _create(client)
        ds = fds.get_dataset(LOCAL_USER, ds_id)
        assert ds is not None
        rec = _record(db, ds_id)
        row = _image(db, ds_id, rec)
        link = cp.save_link(rec.id, 2500, NUMBERED, ds_id, model_id=1, version_id=77,
                            model_name='Nova', base_model='Krea 2')
        meta = cp.image_meta(row, ds, link, 64, 48)
        # The trigger is prefixed exactly as the workflow prefixed it…
        assert meta['prompt'] == 'nova, a candid portrait, film grain'
        assert meta['negativePrompt'] == 'blurry'
        assert meta['cfgScale'] == 3.5 and meta['steps'] == 20 and meta['seed'] == 208607443
        assert meta['sampler'] == 'euler' and meta['scheduler'] == 'simple'
        assert meta['Size'] == '64x48' and meta['baseModel'] == 'Krea 2'
        assert meta['Model'] == 'Krea-2-Turbo'
        # The LoRA is named as its page names it; internal deploy tags never leave.
        assert meta['resources'] == [
            {'type': 'lora', 'name': 'Nova', 'weight': 0.85},
            {'type': 'lora', 'name': 'film-look', 'weight': 0.6}]
        assert meta['civitaiResources'] == [{'type': 'lora', 'weight': 0.85, 'modelVersionId': 77}]
        assert cp.image_meta(row, ds, None, 64, 48)['resources'][0]['name'] == 'lora_nova_000002500_Krea-2-Raw'
        # …and NOT when the box was unticked for that launch.
        row.inject_trigger = False
        assert cp.image_meta(row, ds, link, 64, 48)['prompt'] == 'a candid portrait, film grain'
        # A prompt that already carries the trigger is not doubled.
        row.inject_trigger = None
        row.prompt = 'Nova smiling'
        assert cp.image_meta(row, ds, link, 64, 48)['prompt'] == 'Nova smiling'


def test_post_images_uploads_a_metadata_free_png_with_the_meta_and_publishes(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds_id = _create(client)
        rec = _record(db, ds_id)
        row = _image(db, ds_id, rec)
        link = cp.save_link(rec.id, 2500, NUMBERED, ds_id, model_id=2755270, version_id=civitai.version_id,
                            model_name='Nova', base_model='Krea 2')
        phases = []
        out = cp.post_images([row.id], link, KEY, title='Studio picks',
                             progress=lambda ph, fr: phases.append((ph, fr)))
        assert civitai.procs() == ['post.create', 'post.addImage', 'post.update']
        assert civitai.body_of('post.create')['json'] == {'modelVersionId': civitai.version_id,
                                                          'title': 'Studio picks'}
        added = civitai.body_of('post.addImage')['json']
        assert added['postId'] == civitai.post_id and added['modelVersionId'] == civitai.version_id
        assert added['index'] == 0 and added['type'] == 'image' and added['mimeType'] == 'image/png'
        assert added['width'] == 64 and added['height'] == 48
        assert added['url'].startswith('uuid-')
        assert added['meta']['prompt'] == 'nova, a candid portrait, film grain'
        assert added['meta']['civitaiResources'][0]['modelVersionId'] == civitai.version_id
        # The bytes that left: a PNG with NO text chunk — the workflow blob that
        # named a home path in the source file is gone.
        put = next(c for c in civitai.calls if c[0] == 'PUT')
        with Image.open(io.BytesIO(put[3])) as im:
            assert im.format == 'PNG' and im.size == (64, 48)
            assert 'workflow' not in im.info and not im.text
        assert b'someone' not in put[3]
        pub = civitai.body_of('post.update')
        assert pub['json']['id'] == civitai.post_id and pub['meta'] == {'values': {'publishedAt': ['Date']}}
        # Stamped by the SERVER's clock (its Date header), minus the margin.
        assert pub['json']['publishedAt'].startswith('2026-09-02T11:59:')
        assert out['published'] is True and out['count'] == 1
        assert out['url'] == cp.post_url(civitai.post_id, True)
        assert phases[-1] == ('uploading', 1.0)


def test_a_draft_post_skips_the_publish_and_answers_the_edit_address(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds_id = _create(client)
        rec = _record(db, ds_id)
        row = _image(db, ds_id, rec)
        link = cp.save_link(rec.id, 2500, NUMBERED, ds_id, model_id=1, version_id=civitai.version_id)
        out = cp.post_images([row.id], link, KEY, publish=False)
        assert civitai.procs() == ['post.create', 'post.addImage']
        assert out['published'] is False and out['url'].endswith(f'/posts/{civitai.post_id}/edit')


def test_post_images_refuses_without_a_link_a_key_or_a_finished_image(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds_id = _create(client)
        rec = _record(db, ds_id)
        row = _image(db, ds_id, rec)
        link = cp.save_link(rec.id, 2500, NUMBERED, ds_id, model_id=1, version_id=2)
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.post_images([row.id], None, KEY)
        assert e.value.code == 'link_missing'
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.post_images([row.id], link, None)
        assert e.value.code == 'no_key'
        pending = _image(db, ds_id, rec, status='pending', filename='p.png')
        with pytest.raises(cp.CivitaiPublishError) as e:
            cp.post_images([pending.id], link, KEY)
        assert e.value.code == 'image_missing'
        assert civitai.calls == []


# --- the routes ------------------------------------------------------------------------

def test_status_says_whether_a_key_exists_and_whose_it_is(client, civitai, monkeypatch):
    r = client.get('/api/civitai/status')
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'has_key': True, 'username': 'creator', 'link_host': 'civitai.com'}
    monkeypatch.setattr(cp, 'api_key', lambda: None)
    assert client.get('/api/civitai/status').get_json()['has_key'] is False


def test_every_write_route_names_the_missing_key(client, monkeypatch):
    monkeypatch.setattr(cp, 'api_key', lambda: None)
    for url, body in (('/api/civitai/links', {'record_id': 1, 'step': 1, 'filename': 'x', 'url': '1'}),
                      ('/api/civitai/checkpoint/1/1/publish-model', {'name': 'x'}),
                      ('/api/civitai/images/publish', {'image_ids': [1], 'link_id': 1})):
        r = client.post(url, json=body)
        assert r.status_code == 400, url
        assert r.get_json()['error_code'] == 'no_key'
        assert 'Settings' in r.get_json()['error']


def test_link_routes_round_trip(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        rid = rec.id
    r = client.get(f'/api/civitai/links/{rid}/2500?filename={NUMBERED}')
    assert r.status_code == 200 and r.get_json()['link'] is None
    r = client.post('/api/civitai/links', json={'record_id': rid, 'step': 2500, 'filename': NUMBERED,
                                                'url': 'https://civitai.com/models/2755270/nova'})
    assert r.status_code == 200, r.get_json()
    link = r.get_json()['link']
    assert link['model_id'] == 2755270 and link['model_url'].startswith('https://civitai.com/models/2755270')
    assert link['filename'] == NUMBERED
    assert r.get_json()['page']['name'] == 'Nova'
    assert client.get(f'/api/civitai/links/{rid}/2500?filename={NUMBERED}').get_json()['link']['id'] == link['id']
    assert client.get(f'/api/civitai/links/{rid}/2500').get_json()['link']['id'] == link['id']
    assert client.get(f'/api/civitai/links/{rid}/2500?filename=other.safetensors').get_json()['link'] is None
    r = client.get(f'/api/civitai/links?dataset_id={ds}')
    assert [l['id'] for l in r.get_json()['links']] == [link['id']]
    assert client.get('/api/civitai/links').status_code == 400
    r = client.post('/api/civitai/links', json={'record_id': rid, 'step': 2500, 'filename': NUMBERED, 'url': 'nope'})
    assert r.status_code == 400 and r.get_json()['error_code'] == 'bad_ref'
    # The image door: no file name, the deployed name instead. The run's saves
    # are not listable here, so the save is remembered under '' — and the
    # picture's own lookup (`?checkpoint=`) finds it.
    deployed = 'krea\\lora_nova_000002500_Krea-2-Raw_rc158_v1.safetensors'
    r = client.post('/api/civitai/links', json={'record_id': rid, 'step': 2500, 'url': '2755270',
                                                'checkpoint': deployed})
    assert r.status_code == 200, r.get_json()
    unnamed = r.get_json()['link']
    assert unnamed['filename'] == '' and unnamed['id'] != link['id']
    # Two links at the step: the picture's step block picks the NAMED numbered
    # save's link first; the unnamed one answers once that is gone.
    r = client.get(f'/api/civitai/links/{rid}/2500?checkpoint={deployed}')
    assert r.get_json()['link']['id'] == link['id']
    assert client.post(f'/api/civitai/links/{link["id"]}/delete').status_code == 200
    r = client.get(f'/api/civitai/links/{rid}/2500?checkpoint={deployed}')
    assert r.get_json()['link']['id'] == unnamed['id']
    assert client.post(f'/api/civitai/links/{unnamed["id"]}/delete').status_code == 200
    assert client.get(f'/api/civitai/links/{rid}/2500').get_json()['link'] is None
    assert client.post(f'/api/civitai/links/{link["id"]}/delete').status_code == 404






def test_a_pasted_address_is_looked_up_before_it_is_linked(client, civitai, monkeypatch):
    r = client.get('/api/civitai/page?ref=https://civitai.red/models/2755270/nova?modelVersionId=3100001')
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d['page']['name'] == 'Nova' and d['page']['type'] == 'LORA'
    assert [v['id'] for v in d['page']['versions']] == [3100001]
    assert d['version_id'] == 3100001
    # No version in the address: the modal preselects the newest itself.
    assert client.get('/api/civitai/page?ref=2755270').get_json()['version_id'] is None
    r = client.get('/api/civitai/page?ref=garbage')
    assert r.status_code == 400 and r.get_json()['error_code'] == 'bad_ref'
    monkeypatch.setattr(cp, 'api_key', lambda: None)
    assert client.get('/api/civitai/page?ref=2755270').get_json()['error_code'] == 'no_key'


def test_publishing_images_without_a_link_answers_what_to_do(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        row = _image(db, rec.dataset_id, rec)
        ids = (rec.id, row.id)
    r = client.post('/api/civitai/images/publish', json={'image_ids': [ids[1]], 'record_id': ids[0], 'step': 2500})
    assert r.status_code == 409
    assert r.get_json()['error_code'] == 'link_missing'
    assert 'mark' in r.get_json()['error']
    r = client.post('/api/civitai/images/publish', json={'image_ids': [ids[1]], 'link_id': 424242})
    assert r.status_code == 409 and r.get_json()['error_code'] == 'link_missing'
    assert client.post('/api/civitai/images/publish', json={'image_ids': [], 'link_id': 1}).status_code == 400
    assert client.post('/api/civitai/images/publish', json={'image_ids': [1]}).status_code == 400
    assert client.get('/api/civitai/jobs/nope').status_code == 404


def test_the_publish_image_route_runs_the_job_to_its_result(client, app, civitai):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        row = _image(db, ds, rec)
        link = cp.save_link(rec.id, 2500, NUMBERED, ds, model_id=2755270, version_id=civitai.version_id,
                            model_name='Nova')
        ids = (link.id, row.id)
    r = client.post('/api/civitai/images/publish',
                    json={'image_ids': [ids[1]], 'link_id': ids[0], 'publish': False})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['link']['model_name'] == 'Nova'
    job_id = r.get_json()['job_id']
    # A wall-clock deadline, not an iteration count: with `time.sleep` stubbed
    # by the fixture the old 200 × 0.02 s was a spin of 200 GETs, and the
    # release runner's job thread had not finished inside it.
    deadline = _real_monotonic() + 30
    while True:
        j = client.get(f'/api/civitai/jobs/{job_id}').get_json()
        if j['state'] in ('done', 'error') or _real_monotonic() > deadline:
            break
        _real_sleep(0.02)
    assert j['state'] == 'done', j
    assert j['kind'] == 'post' and j['result']['count'] == 1 and j['result']['published'] is False
    assert civitai.procs() == ['post.create', 'post.addImage']


def test_publish_model_route_validates_the_form_and_the_file_before_starting(client, app, civitai, tmp_path, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        rid = rec.id
    r = client.post(f'/api/civitai/checkpoint/{rid}/2500/publish-model', json={'name': 'Nova', 'base_model': 'Nope'})
    assert r.status_code == 400 and 'Nope' in r.get_json()['error']
    r = client.post(f'/api/civitai/checkpoint/{rid}/2500/publish-model', json={'name': 'Nova', 'base_model': 'Krea 2'})
    assert r.status_code == 409 and r.get_json()['error_code'] == 'checkpoint_missing'
    leaky = _safetensors(str(tmp_path / 'l.safetensors'), {'p': 'C:\\Users\\someone\\x'})
    monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (leaky, None))
    r = client.post(f'/api/civitai/checkpoint/{rid}/2500/publish-model', json={'name': 'Nova', 'base_model': 'Krea 2'})
    assert r.status_code == 409 and r.get_json()['error_code'] == 'file_metadata_leak'
    assert civitai.calls == []




def test_draft_defaults_route_answers_the_link_of_the_named_file(client, app, civitai, monkeypatch):
    from app.extensions import db
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        cp.save_link(rec.id, 2500, NUMBERED, ds, model_id=2755270, version_id=civitai.version_id, model_name='Nova',
                     version_name='v1.0')
        rid = rec.id
    monkeypatch.setattr(cp, 'checkpoint_file_for', lambda *a, **k: (_ for _ in ()).throw(
        cp.CivitaiPublishError('checkpoint_missing', 'gone')))
    r = client.get(f'/api/civitai/checkpoint/{rid}/2500/draft-defaults?filename={NUMBERED}')
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d['link']['model_name'] == 'Nova' and d['file_error'] == 'gone'
    assert d['base_model_choices'] == list(cp.CIVITAI_BASE_MODELS)
    r = client.get(f'/api/civitai/checkpoint/{rid}/2500/draft-defaults?filename={FINAL}')
    assert r.get_json()['link'] is None
