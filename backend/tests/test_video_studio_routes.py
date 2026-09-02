"""The Video Test Studio's blueprint: what it refuses, and what it records.

The builder has its own file. These are the questions only the route can answer:
does it refuse before spending anything, does the clip row carry what the graph
was actually built from, and does a finished job find its way back to the row.
"""
from pathlib import Path
import pytest


def _comfy(monkeypatch, reachable=True):
    monkeypatch.setattr('app.capabilities.probe',
                        lambda *a, **k: {'comfyui': {'reachable': reachable}})


@pytest.fixture()
def queued(app, monkeypatch):
    """Enqueue without ComfyUI: preflight open, queue captured."""
    from app.services import video_test_studio as vts
    seen = {}
    monkeypatch.setattr(vts, 'preflight', lambda wf: None)
    from app.job_queue import queue_manager
    monkeypatch.setattr(queue_manager, 'add_job',
                        lambda **kw: seen.update(kw) or kw.get('job_id'))
    return seen


# --- refusals that cost nothing ---------------------------------------------

def test_generate_is_gated_on_comfyui(client, monkeypatch):
    _comfy(monkeypatch, False)
    r = client.post('/api/video-studio/generate', json={'prompt': 'p', 'mode': 't2v'})
    assert r.status_code == 409
    assert r.get_json()['error'] == 'ComfyUI is not reachable'


def test_i2v_without_a_start_frame_is_refused_in_its_own_words(client, monkeypatch):
    _comfy(monkeypatch)
    r = client.post('/api/video-studio/generate', json={'prompt': 'she turns'})
    assert r.status_code == 400
    assert 'start image' in r.get_json()['error']


def test_a_clip_without_a_motion_prompt_is_refused(client, monkeypatch):
    _comfy(monkeypatch)
    r = client.post('/api/video-studio/generate',
                    json={'mode': 't2v', 'prompt': '   '})
    assert r.status_code == 400
    assert 'motion' in r.get_json()['error']


@pytest.mark.parametrize('name', [
    'C:\\weights\\evil.safetensors', '/etc/passwd', '..\\..\\x.safetensors',
    '../../x.safetensors',
])
def test_a_rooted_or_climbing_lora_name_never_reaches_the_loader(client, monkeypatch, name):
    """The name is handed to a loader that resolves it under the loras roots.
    A rooted or `..`-bearing name is the one shape that walks out of them — the
    same guard the image studio applies, deliberately shared rather than
    re-implemented."""
    _comfy(monkeypatch)
    r = client.post('/api/video-studio/generate',
                    json={'mode': 't2v', 'prompt': 'p', 'lora': name})
    assert r.status_code == 400
    assert r.get_json()['error'] == 'invalid LoRA name'


def test_source_without_anything_attached_says_what_to_attach(client):
    r = client.post('/api/video-studio/source', json={})
    assert r.status_code == 400
    assert 'image' in r.get_json()['error']


# --- what a queued clip records ---------------------------------------------

def test_a_queued_clip_records_the_graph_it_was_built_from(client, app, monkeypatch, queued):
    _comfy(monkeypatch)
    r = client.post('/api/video-studio/generate', json={
        'mode': 't2v', 'prompt': 'a street at night', 'aspect': 'portrait',
        'turbo': True, 'sparse': 'conservative', 'frames': 56, 'seed': 99,
        'megapixels': 0.5,
    })
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['seed'] == 99 and body['frames'] == 56

    with app.app_context():
        from app.extensions import db
        from app.models import VideoTestClip
        clip = db.session.get(VideoTestClip, body['clip_id'])
        assert clip.status == 'pending'
        assert clip.mode == 't2v' and clip.seed == 99
        assert clip.turbo is True and clip.sparse == 'conservative'
        assert clip.steps == 6           # the turbo default, as built
        assert clip.base_model  # the base that actually ran is stored, not implied

    # The queue job carries the marker its completion callback keys off. Without
    # it a finished clip is a job nobody claims and an mp4 nobody can attribute.
    assert queued['metadata']['is_video_test'] is True
    assert queued['metadata']['clip_id'] == body['clip_id']
    assert queued['workflow_data']['104']['inputs']['length'] == 56


def test_an_unknown_sparse_level_is_stored_as_off(client, app, monkeypatch, queued):
    _comfy(monkeypatch)
    r = client.post('/api/video-studio/generate',
                    json={'mode': 't2v', 'prompt': 'p', 'sparse': 'turbocharged'})
    with app.app_context():
        from app.extensions import db
        from app.models import VideoTestClip
        assert db.session.get(VideoTestClip, r.get_json()['clip_id']).sparse == ''


def test_a_missing_asset_answers_the_studios_own_409(client, app, monkeypatch):
    """Same structured refusal as the image studio, so the banner that lists
    missing weights and node packs needed no new plumbing."""
    from app.services import lora_test_studio as lts
    from app.services import video_test_studio as vts
    _comfy(monkeypatch)

    def boom(_wf):
        raise lts.StudioAssetsMissing(
            'h3video', [{'path': 'models/unet/minimax.safetensors', 'kind': 'diffusion model'}],
            ['H3SparseAttentionAdvanced'])
    monkeypatch.setattr(vts, 'preflight', boom)
    r = client.post('/api/video-studio/generate', json={'mode': 't2v', 'prompt': 'p'})
    assert r.status_code == 409
    body = r.get_json()
    assert body['studio_missing']['files'][0]['kind'] == 'diffusion model'
    # And the node hint names the pack to install, not just the class.
    assert body['studio_missing']['node_packs'][0]['pack'] == 'H3-Optimizations'


# --- the completion callback -------------------------------------------------

def _clip(app, **kw):
    from app.extensions import db
    from app.models import VideoTestClip
    row = VideoTestClip(job_id=kw.pop('job_id', 'job-1'), status='pending',
                        prompt='p', mode='t2v', **kw)
    db.session.add(row)
    db.session.commit()
    return row.id


def test_a_finished_job_lands_on_its_clip(app, monkeypatch):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        monkeypatch.setattr(vts, '_bring_clip_home', lambda fn: None)
        vts.link_completed_clip('job-1', 'clip_00001.mp4')
        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'done' and row.filename == 'clip_00001.mp4'


def test_a_clip_lands_even_when_comfyui_still_holds_the_mp4(app, tmp_path, monkeypatch):
    """The claim brings the bytes home; the locked original is left behind.

    `_bring_clip_home` used to `shutil.move`, whose copy+unlink fallback raises
    AFTER the copy — on a freshly flushed mp4 that Windows still reports as in
    use. The `except OSError` swallowed it, so the clip looked done while `dst`
    held a half-written file that the early `os.path.exists(dst)` return then
    accepted forever. A video is the biggest, slowest output this app claims,
    so it is the one most likely to still be open.
    """
    import os
    from app.services import lora_test_studio as lts
    from app.services import video_test_studio as vts
    from app.utils import comfy_fs

    out_dir = tmp_path / 'output'
    out_dir.mkdir()
    (out_dir / 'held.mp4').write_bytes(b'MP4DATA')
    clips = tmp_path / 'clips'
    clips.mkdir()
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: clips)
    monkeypatch.setattr(lts, '_comfy_output_dir', lambda: str(out_dir))
    monkeypatch.setattr(comfy_fs, '_OUTPUT_LOCK_RETRY_DELAY', 0)

    def locked_unlink(path):
        if os.path.basename(path) == 'held.mp4':
            err = PermissionError(32, 'used by another process')
            err.winerror = 32
            raise err
        return os.unlink(path)

    monkeypatch.setattr(comfy_fs.os, 'unlink', locked_unlink)

    with app.app_context():
        vts._bring_clip_home('held.mp4')

    assert (clips / 'held.mp4').read_bytes() == b'MP4DATA'
    assert (out_dir / 'held.mp4').exists()   # still ComfyUI's; ours is complete


def test_a_failed_job_keeps_the_reason_comfyui_gave(app):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        vts.link_completed_clip('job-1', None, failed=True,
                                reason='Value not in list: lora_name')
        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'failed'
        assert 'lora_name' in row.error


def test_a_late_completion_never_overwrites_a_settled_clip(app, monkeypatch):
    """A job whose row was already resolved (cancelled, replaced) must not write
    its result over the good one — the image studio learned this the same way."""
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        row = db.session.get(VideoTestClip, cid)
        row.status = 'cancelled'
        db.session.commit()
        monkeypatch.setattr(vts, '_bring_clip_home', lambda fn: None)
        vts.link_completed_clip('job-1', 'late.mp4')
        assert db.session.get(VideoTestClip, cid).filename is None


def test_the_queue_dispatches_a_video_job_to_this_lane(app, monkeypatch):
    """The metadata marker is only useful if the worker branches on it."""
    import json
    from app import job_queue
    from app.extensions import db
    from app.models import ImageGenerationQueue
    seen = {}
    monkeypatch.setattr(job_queue, '_drop_staged_inputs', lambda md: None)
    with app.app_context():
        job = ImageGenerationQueue(
            job_id='job-9', user_id='1', status='processing',
            job_metadata=json.dumps({'is_video_test': True, 'clip_id': 3}))
        db.session.add(job)
        db.session.commit()
        from app.services import video_test_studio as vts
        monkeypatch.setattr(vts, 'link_completed_clip',
                            lambda *a, **k: seen.update({'args': a, 'kw': k}))
        job_queue._dispatch_completion(job, 'out.mp4', False)
    assert seen['args'] == ('job-9', 'out.mp4')
    assert seen['kw']['failed'] is False


# --- deploy ------------------------------------------------------------------

def test_deploy_refuses_a_run_that_is_not_a_video_run(client, app):
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        run = CloudTrainingRun(dataset_id=1, status='done', job_name='j',
                               dataset_table='face_dataset')
        db.session.add(run)
        db.session.commit()
        rid = run.id
    r = client.post('/api/video-studio/deploy',
                    json={'run_id': rid, 'filename': 'x.safetensors'})
    assert r.status_code == 400
    assert 'video training run not found' in r.get_json()['error']


def test_deploy_copies_the_checkpoint_and_names_it_for_the_loader(client, app, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import video_test_studio as vts
    store = tmp_path / 'run_7'
    store.mkdir()
    (store / 'lds7_jessy.safetensors').write_bytes(b'W' * 32)
    loras = tmp_path / 'loras'
    (loras / vts.LORA_SUBDIR).mkdir(parents=True)
    with app.app_context():
        run = CloudTrainingRun(dataset_id=4, status='done', job_name='j',
                               dataset_table='video_dataset', staging_dir=str(store))
        db.session.add(run)
        db.session.commit()
        rid = run.id
    monkeypatch.setattr('app.services.comfy_model_paths.search_roots',
                        lambda kind: [str(loras)] if kind == 'loras' else [])
    r = client.post('/api/video-studio/deploy',
                    json={'run_id': rid, 'filename': 'lds7_jessy.safetensors'})
    assert r.status_code == 200, r.get_json()
    name = r.get_json()['filename']
    assert name.replace('\\', '/') == 'h3/lds/lds7_jessy.safetensors'
    assert (loras / vts.LORA_SUBDIR / 'lds7_jessy.safetensors').is_file()

    # And it is then listed as deployable, which is what stops the picker from
    # offering "Deploy" forever on a LoRA that is already there.
    with app.app_context():
        assert any(e['filename'].replace('\\', '/').endswith('lds7_jessy.safetensors')
                   for e in vts.deployed_loras())


@pytest.mark.parametrize('bad', ['../escape.safetensors', 'sub/dir.safetensors',
                                 'C:\\abs.safetensors'])
def test_deploy_only_ever_resolves_a_basename(client, app, tmp_path, bad):
    """`run_checkpoint_path` is basename-only by construction, so a crafted
    filename resolves to nothing rather than to a file outside the store."""
    from app.extensions import db
    from app.models import CloudTrainingRun
    store = tmp_path / 'run_8'
    store.mkdir()
    (store / 'ok.safetensors').write_bytes(b'W')
    with app.app_context():
        run = CloudTrainingRun(dataset_id=4, status='done', job_name='j',
                               dataset_table='video_dataset', staging_dir=str(store))
        db.session.add(run)
        db.session.commit()
        rid = run.id
    r = client.post('/api/video-studio/deploy', json={'run_id': rid, 'filename': bad})
    assert r.status_code == 400


# --- the read side -----------------------------------------------------------

def test_options_publishes_the_catalogue_rather_than_restating_it(client):
    from app.services import video_targets
    body = client.get('/api/video-studio/options').get_json()
    profile = video_targets.get('minimax_h3')
    assert body['fps'] == profile['fps']
    assert body['frame_choices'] == list(profile['frame_choices'])
    assert body['sparse_modes'] == ['', 'default', 'conservative', 'max']
    assert 'eros_available' in body      # the checkbox needs to know


def test_clips_history_rates_and_deletes(client, app, tmp_path, monkeypatch):
    from app.services import video_test_studio as vts
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: tmp_path)
    with app.app_context():
        cid = _clip(app, job_id='job-h')
    listed = client.get('/api/video-studio/clips').get_json()['clips']
    assert [c['id'] for c in listed] == [cid]

    assert client.post(f'/api/video-studio/clip/{cid}/rate',
                       json={'rating': 5}).get_json()['rating'] == 1
    assert client.post(f'/api/video-studio/clip/{cid}/rate',
                       json={'rating': -3}).get_json()['rating'] == -1
    assert client.delete(f'/api/video-studio/clip/{cid}').status_code == 200
    assert client.get('/api/video-studio/clips').get_json()['clips'] == []


def test_a_clip_with_no_file_is_a_404_not_a_500(client, app):
    with app.app_context():
        cid = _clip(app, job_id='job-nofile')
    assert client.get(f'/api/video-studio/clip/{cid}/video').status_code == 404


# --- the Gallery as a start frame (2026-09-01) --------------------------------

class _FakeReq:
    """The two things _resolve_source reads off a request."""

    def __init__(self, data):
        self.files = {}
        self._data = data

    def get_json(self, silent=False):
        return self._data


def test_a_gallery_image_resolves_to_the_picture_the_user_is_looking_at(app, tmp_path):
    """Asked for from live use: the picture someone wants to animate is very
    often one this app just generated, and the picker sent them back through
    disk to use it. Resolved at FULL size from the folder that SERVES it
    (/api/dataset/<id>/img/<name>) — a thumbnail fed to a 1 MP generation would
    blame the LoRA for a softness the source never had."""
    from PIL import Image
    from app.extensions import db
    from app.models import FaceDataset, LoraTestImage
    from app.routes.video_studio import _resolve_source
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        ds = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(ds)
        db.session.commit()
        folder = Path(dataset_path(ds.id))
        folder.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 48), 'red').save(folder / 'gen_0001.png')
        row = LoraTestImage(dataset_id=ds.id, filename='gen_0001.png',
                            prompt='a woman', checkpoint='ckpt.safetensors',
                            strength=1.0)
        db.session.add(row)
        db.session.commit()

        path, temp = _resolve_source(_FakeReq({'gallery_image_id': row.id}))
        assert path == str(folder / 'gen_0001.png')
        # Not a temp file: it is the user's own picture, and deleting it after
        # staging would delete what the Gallery is still showing.
        assert temp is False

        # A row whose file went away, and an id that never existed: both refuse
        # in words, never with a path that is not there.
        (folder / 'gen_0001.png').unlink()
        with pytest.raises(ValueError, match='disk'):
            _resolve_source(_FakeReq({'gallery_image_id': row.id}))
        with pytest.raises(ValueError, match='gallery'):
            _resolve_source(_FakeReq({'gallery_image_id': 999999}))


def test_the_refusal_names_every_way_in(client):
    """Four sources now; the sentence that lists them is what someone reads
    when they attached none."""
    r = client.post('/api/video-studio/source', json={})
    body = r.get_json()['error'].lower()
    for word in ('image', 'bank', 'clip', 'gallery'):
        assert word in body


def test_a_clip_publishes_the_start_frame_reuse_needs(app):
    """↻ Reuse restored every dial of an image-to-video clip and left the start
    frame empty, so Generate stayed blocked on 'Pick a start frame' — every
    setting back except the one that decides whether the button works. The row
    had stored the staged file all along; the payload never published it."""
    from app.routes.video_studio import _clip_dict
    from app.extensions import db
    from app.models import VideoTestClip

    with app.app_context():
        clip = VideoTestClip(prompt='she turns', mode='i2v', status='done',
                               source_image='lds_vstudio_abc123.png',
                               frames=56, fps=24, steps=6, seed=7)
        db.session.add(clip)
        db.session.commit()
        row = _clip_dict(clip)

    assert row['source_image'] == 'lds_vstudio_abc123.png'
    assert row['mode'] == 'i2v'


def test_options_carry_the_launch_advice_the_running_comfyui_earns(client, monkeypatch):
    from app.services import video_test_studio as vts
    # Started without the flag on a 48 GB machine, on a ComfyUI that knows it: told.
    monkeypatch.setattr(vts, 'comfyui_launch_facts',
                        lambda timeout=3: (['main.py', '--listen', '127.0.0.1'], 47.7, '0.30.1'))
    body = client.get('/api/video-studio/options').get_json()
    assert body['launch_advice'] == {'flag': '--fast-disk', 'add': True, 'remove': None,
                                     'ram_total_gb': 47.7, 'weights_gb': vts.H3_HOST_RAM_GB}
    # Started by LDS's own launcher (which passes the flag): nothing to say.
    monkeypatch.setattr(vts, 'comfyui_launch_facts',
                        lambda timeout=3: (['main.py', '--fast-disk'], 47.7, '0.30.1'))
    assert client.get('/api/video-studio/options').get_json()['launch_advice'] is None
    # Unreachable, or too old to echo its argv: silence, never a guess.
    monkeypatch.setattr(vts, 'comfyui_launch_facts', lambda timeout=3: (None, None, None))
    assert client.get('/api/video-studio/options').get_json()['launch_advice'] is None


class _Resp:
    def __init__(self, payload, status=200, raises=None):
        self.status_code = status
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise self._raises
        return self._payload


def _facts_with(app, monkeypatch, response=None, raising=None):
    from app.services import video_test_studio as vts
    import requests
    seen = {}

    def fake_get(*args, **kwargs):
        seen.update(kwargs)
        if raising:
            raise raising
        return response

    monkeypatch.setattr(requests, 'get', fake_get)
    with app.app_context():
        from app import config as cfg
        monkeypatch.setattr(cfg, 'get', lambda key, default=None: (
            'http://127.0.0.1:8188' if key == 'comfyui.api_url' else default))
        return vts.comfyui_launch_facts(), seen


def test_launch_facts_read_argv_ram_and_version_from_system_stats(app, monkeypatch):
    # A raw, unaligned byte count: the GiB figure is handed on as a float and
    # rounded ONCE, by the advice, so the seam bytes → GiB is really crossed.
    payload = {'system': {'argv': ['main.py', '--fast-disk'], 'ram_total': 50465476608,
                          'comfyui_version': '0.30.1'}}
    facts, seen = _facts_with(app, monkeypatch, _Resp(payload))
    assert facts == (['main.py', '--fast-disk'], 50465476608 / 1024 ** 3, '0.30.1')
    # The call is bounded and never follows a redirect off the configured host:
    # it sits in the synchronous body of the options route the panel fetches on mount.
    assert seen['timeout'] == 3
    assert seen['allow_redirects'] is False


def test_launch_facts_keep_the_argv_when_an_older_server_sends_no_ram(app, monkeypatch):
    facts, _ = _facts_with(app, monkeypatch, _Resp({'system': {'argv': ['main.py']}}))
    assert facts == (['main.py'], None, None)


@pytest.mark.parametrize('response', [
    _Resp({'system': {}}),                       # a server that echoes nothing useful
    _Resp({}),                                   # no system block at all
    _Resp([]),                                   # not even an object
    _Resp({'system': {'argv': ['main.py']}}, status=500),
    _Resp(None, raises=ValueError('not json')),
])
def test_launch_facts_answer_none_for_every_shape_of_nothing(app, monkeypatch, response):
    facts, _ = _facts_with(app, monkeypatch, response)
    assert facts == (None, None, None)


def test_launch_facts_never_raise_when_the_request_fails(app, monkeypatch):
    import requests
    for exc in (requests.ConnectionError('down'), requests.Timeout('slow')):
        facts, _ = _facts_with(app, monkeypatch, raising=exc)
        assert facts == (None, None, None)


# --- ⏱ the render time ---------------------------------------------------------

def _job(app, job_id='job-1', started=None, completed=None, status='completed'):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    row = ImageGenerationQueue(job_id=job_id, user_id='local', status=status,
                               started_at=started, completed_at=completed)
    db.session.add(row)
    db.session.commit()


def test_a_finished_clip_records_how_long_the_queue_spent_on_it(app, monkeypatch):
    from datetime import datetime, timedelta
    from app.extensions import db
    from app.models import VideoTestClip
    from app.routes.video_studio import _clip_dict
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        t0 = datetime(2026, 9, 2, 22, 0, 0)
        _job(app, started=t0, completed=t0 + timedelta(seconds=347.64))
        monkeypatch.setattr(vts, '_bring_clip_home', lambda fn: None)
        vts.link_completed_clip('job-1', 'clip_00001.mp4')
        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'done'
        assert row.render_seconds == 347.6           # rounded once, here
        assert _clip_dict(row)['render_seconds'] == 347.6


def test_a_failed_clip_keeps_how_long_it_ran_before_dying(app):
    from datetime import datetime, timedelta
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        t0 = datetime(2026, 9, 2, 22, 0, 0)
        _job(app, started=t0, completed=t0 + timedelta(seconds=240), status='failed')
        vts.link_completed_clip('job-1', None, failed=True, reason='ComfyUI KSampler: boom')
        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'failed' and row.render_seconds == 240.0


def test_render_time_is_null_whenever_the_queue_cannot_say(app, monkeypatch):
    from datetime import datetime, timedelta
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    monkeypatch.setattr(vts, '_bring_clip_home', lambda fn: None)
    t0 = datetime(2026, 9, 2, 22, 0, 0)
    cases = {
        'no-row': None,                                    # settled by hand, or a pruned queue
        'no-start': dict(started=None, completed=t0),
        'no-end': dict(started=t0, completed=None),
        'backwards': dict(started=t0, completed=t0 - timedelta(seconds=5)),
    }
    with app.app_context():
        for job_id, stamps in cases.items():
            cid = _clip(app, job_id=job_id)
            if stamps is not None:
                _job(app, job_id=job_id, **stamps)
            vts.link_completed_clip(job_id, 'clip.mp4')
            row = db.session.get(VideoTestClip, cid)
            assert row.status == 'done', job_id          # the clip still lands
            assert row.render_seconds is None, job_id    # a guess is not a measurement


def test_a_job_the_queue_cancelled_gets_no_render_time(app, monkeypatch):
    # The ComfyUI-restart path: the job stalls with its start kept, the barrier
    # is reconciled hours later and stamps completed_at THEN. That difference
    # measures the outage; the card must say nothing rather than "6 h".
    from datetime import datetime, timedelta
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        t0 = datetime(2026, 9, 2, 22, 0, 0)
        _job(app, started=t0, completed=t0 + timedelta(hours=6), status='cancelled')
        vts.link_completed_clip('job-1', None, failed=True, reason='ComfyUI restarted')
        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'failed' and row.render_seconds is None


def test_a_job_settled_within_the_same_tick_measures_zero_not_nothing(app, monkeypatch):
    from datetime import datetime
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import video_test_studio as vts
    with app.app_context():
        cid = _clip(app)
        t0 = datetime(2026, 9, 2, 22, 0, 0)
        _job(app, started=t0, completed=t0)
        monkeypatch.setattr(vts, '_bring_clip_home', lambda fn: None)
        vts.link_completed_clip('job-1', 'clip.mp4')
        assert db.session.get(VideoTestClip, cid).render_seconds == 0.0


def test_the_render_time_survives_the_queues_real_completion_path(app, monkeypatch):
    """Everything above fabricates the stamps. This one lets the queue stamp
    them the way it does in production — update_status('processing') at the
    claim, update_status('completed') then commit BEFORE the dispatch — so the
    test is the one that reddens if that order is ever reversed."""
    import json
    import time
    from app import job_queue
    from app.extensions import db
    from app.models import ImageGenerationQueue, VideoTestClip
    from app.services import video_test_studio as vts
    monkeypatch.setattr(job_queue, '_drop_staged_inputs', lambda md: None)
    monkeypatch.setattr(vts, '_bring_clip_home', lambda fn: None)
    with app.app_context():
        cid = _clip(app, job_id='job-real')
        job = ImageGenerationQueue(job_id='job-real', user_id='local', status='pending',
                                   job_metadata=json.dumps({'is_video_test': True, 'clip_id': cid}))
        db.session.add(job)
        db.session.commit()
        job.update_status('processing')
        db.session.commit()
        time.sleep(0.01)
        job.update_status('completed', result_filename='clip.mp4')
        db.session.commit()
        job_queue._dispatch_completion(job, 'clip.mp4', False)
        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'done' and row.filename == 'clip.mp4'
        assert row.render_seconds is not None and row.render_seconds >= 0.0


def test_the_render_time_column_is_declared_to_the_additive_migration():
    # An existing database only gains the column through _SCHEMA_ADDITIONS;
    # a model column missing there is invisible on every install but a fresh one.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, '..', 'app', '__init__.py'), encoding='utf-8').read()
    assert "('video_test_clip', 'render_seconds', 'FLOAT')" in src
