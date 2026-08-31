"""The Video Test Studio's blueprint: what it refuses, and what it records.

The builder has its own file. These are the questions only the route can answer:
does it refuse before spending anything, does the clip row carry what the graph
was actually built from, and does a finished job find its way back to the row.
"""
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
