"""Checkpoints & LoRAs of a VIDEO dataset — the workspace section's routes.

Every verb the image workspace's Checkpoints section has, for a video set, at
the unit of the STEP: list both lanes with the deployed state, 📦 deploy a step
(every file of a Wan pair), ⏏ undeploy the app's own copy, 🗑 trash a step,
download a LOCAL save (the cloud twin already existed), ⓘ details of a run.

The refusals ARE the contract here — a hand-placed LoRA never trashed, a step
never deleted from under a running lane, a face run of the same id never
served — so most tests below are about what a route refuses, not what it does.
"""
import json
import os

from app.extensions import db
from app.services import cloud_run_dataset as crd
from app.services import cloud_training as ct
from test_cloud_video_launch import _face_dataset, _run, _video_dataset
from test_cloud_video_lifecycle import _saves

PAIR_100 = ['video_surf_000000100_high_noise.safetensors',
            'video_surf_000000100_low_noise.safetensors']
PAIR_50 = ['video_surf_000000050_high_noise.safetensors',
           'video_surf_000000050_low_noise.safetensors']
FINAL = ['video_surf.safetensors']


def _loras_root(tmp_path, monkeypatch):
    """ComfyUI's loras root, faked the way the Studio's own tests fake it."""
    root = tmp_path / 'loras'
    root.mkdir(exist_ok=True)
    monkeypatch.setattr('app.services.comfy_model_paths.search_roots',
                        lambda folder_type: [str(root)])
    return root


def _local_saves(tmp_path, monkeypatch, names):
    """Give the dataset's LOCAL run saves on disk, by pointing the lane's
    save root at a folder the test owns (the real one is ai-toolkit's output
    dir, resolved from config)."""
    from app.services import video_training_local as vtl
    d = tmp_path / 'local_saves'
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b'L' * 16)
    monkeypatch.setattr(vtl, 'save_root', lambda ds: d)
    monkeypatch.setattr(vtl, 'video_training_progress',
                        lambda dataset_id, user_id=None: {'active': False})
    return d


def _deployed(root, *names, sub=None):
    folder = root / (sub or os.path.join('h3', 'lds'))
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b'W' * 16)


def _trash_names(app):
    from app.services import trash
    out = []
    for dirpath, _dirs, files in os.walk(trash.trash_root()):
        out.extend(files)
    return out


def _post(client, url, body):
    return client.post(url, data=json.dumps(body), content_type='application/json')


# ── 1. The listing ──────────────────────────────────────────────────────


def test_the_listing_groups_both_lanes_by_step_with_the_deployed_state(
        app, client, tmp_path, monkeypatch):
    root = _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        _saves(run, tmp_path, PAIR_100)
        _local_saves(tmp_path, monkeypatch, PAIR_50 + FINAL)
        # Half of the cloud pair is in ComfyUI; a LOCAL half was dropped by hand
        # under h3/ (listed as deployed, never ours to undeploy).
        _deployed(root, PAIR_100[0])
        _deployed(root, PAIR_50[1], sub='h3')
        ds_id = ds.id
        run_id = run.id
    r = client.get(f'/api/video-dataset/{ds_id}/train/checkpoints')
    assert r.status_code == 200
    d = r.get_json()
    assert d['can_deploy'] is True
    assert d['delete_mode'] == 'app_trash'
    assert d['deploy_folder'] == 'h3/lds'

    local = d['local']
    assert local['run_name'].endswith(f'_ds{ds_id}')
    assert [(s['step'], s['final']) for s in local['steps']] == [(50, False), (None, True)]
    by_name = {f['filename']: f for s in local['steps'] for f in s['files']}
    assert by_name[PAIR_50[1]]['deployed_as'].replace('\\', '/') == 'h3/' + PAIR_50[1]
    assert by_name[PAIR_50[1]]['undeployable'] is False
    assert by_name[PAIR_50[0]]['deployed_as'] is None
    assert local['steps'][0]['deployed'] is False    # half a pair is not deployed

    (cloud,) = d['cloud']
    assert cloud['run_id'] == run_id and cloud['active'] is False
    (step,) = cloud['steps']
    assert step['step'] == 100 and step['final'] is False and step['deployed'] is False
    files = {f['filename']: f for f in step['files']}
    assert files[PAIR_100[0]]['deployed_as'].replace('\\', '/') == 'h3/lds/' + PAIR_100[0]
    assert files[PAIR_100[0]]['undeployable'] is True
    assert files[PAIR_100[1]]['deployed_as'] is None
    assert all(f['size'] == 16 for f in step['files'])


def test_the_listing_shows_nothing_of_the_face_run_of_the_same_id(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        face = _face_dataset()
        vds = _video_dataset(tmp_path)
        assert face.id == vds.id
        face_run = _run(face.id, steps=100)      # NULL table = face
        _saves(face_run, tmp_path, ['lora_trg_000000100.safetensors'])
        ds_id = vds.id
    d = client.get(f'/api/video-dataset/{ds_id}/train/checkpoints').get_json()
    assert d['cloud'] == [] and d['local'] is None


def test_an_unknown_dataset_is_a_404(client):
    assert client.get('/api/video-dataset/999/train/checkpoints').status_code == 404


# ── 2. 📦 Deploy ────────────────────────────────────────────────────────


def test_deploying_a_cloud_step_copies_every_file_of_the_pair(
        app, client, tmp_path, monkeypatch):
    root = _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        _saves(run, tmp_path, PAIR_100)
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/deploy',
              {'run_id': run_id, 'step': 100})
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert sorted(os.path.basename(n) for n in d['deployed']) == sorted(PAIR_100)
    assert all(n.replace('\\', '/').startswith('h3/lds/') for n in d['deployed'])
    assert d['folder'] == 'h3/lds'
    for n in PAIR_100:
        assert (root / 'h3' / 'lds' / n).is_file()
    # And the listing now says so, at the step level.
    lst = client.get(f'/api/video-dataset/{ds_id}/train/checkpoints').get_json()
    assert lst['cloud'][0]['steps'][0]['deployed'] is True


def test_deploying_a_local_step_lands_in_the_same_folder(
        app, client, tmp_path, monkeypatch):
    """The Studio only ever deployed CLOUD runs (a CloudTrainingRun resolves
    the file). A local run's save has no row — it goes through the same copy,
    into the same folder, so the Studio's picker lists it as deployed too."""
    root = _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        _local_saves(tmp_path, monkeypatch, PAIR_50 + FINAL)
        ds_id = ds.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/deploy',
              {'run_id': None, 'step': None, 'final': True})
    assert r.status_code == 200, r.get_json()
    assert (root / 'h3' / 'lds' / FINAL[0]).is_file()
    assert not (root / 'h3' / 'lds' / PAIR_50[0]).exists()   # only the step asked


def test_deploy_refuses_a_step_the_run_never_saved(app, client, tmp_path, monkeypatch):
    root = _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        _saves(run, tmp_path, PAIR_100)
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/deploy',
              {'run_id': run_id, 'step': 999})
    assert r.status_code == 404
    assert not (root / 'h3').exists()


def test_deploy_without_a_loras_root_is_a_stated_refusal(
        app, client, tmp_path, monkeypatch):
    monkeypatch.setattr('app.services.comfy_model_paths.search_roots',
                        lambda folder_type: [])
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        _saves(run, tmp_path, PAIR_100)
        ds_id, run_id = ds.id, run.id
    lst = client.get(f'/api/video-dataset/{ds_id}/train/checkpoints').get_json()
    assert lst['can_deploy'] is False
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/deploy',
              {'run_id': run_id, 'step': 100})
    assert r.status_code == 400
    assert 'loras folder' in r.get_json()['error']


def test_deploy_never_resolves_a_run_of_another_dataset(
        app, client, tmp_path, monkeypatch):
    root = _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        mine = _video_dataset(tmp_path, name='mine', out_dir=str(tmp_path / 'a'))
        other = _video_dataset(tmp_path, name='other', out_dir=str(tmp_path / 'b'))
        run = _run(other.id, crd.VIDEO, steps=100)
        _saves(run, tmp_path, PAIR_100)
        mine_id, run_id = mine.id, run.id
    r = _post(client, f'/api/video-dataset/{mine_id}/train/checkpoint/deploy',
              {'run_id': run_id, 'step': 100})
    assert r.status_code == 404
    assert not (root / 'h3').exists()


# ── 3. ⏏ Undeploy ───────────────────────────────────────────────────────


def test_undeploy_moves_only_the_apps_own_copy_to_the_trash(
        app, client, tmp_path, monkeypatch):
    root = _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        ds_id = ds.id
        _deployed(root, PAIR_100[0])                    # ours: h3/lds/
        _deployed(root, 'by_hand.safetensors', sub='h3')  # the user's: h3/
    url = f'/api/video-dataset/{ds_id}/train/checkpoint/undeploy'
    r = _post(client, url, {'deployed_as': 'h3/lds/' + PAIR_100[0]})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['delete_mode'] == 'app_trash'
    assert not (root / 'h3' / 'lds' / PAIR_100[0]).exists()
    with app.app_context():
        assert PAIR_100[0] in _trash_names(app)
    # The hand-placed one is refused, and stays — and a name given as h3/<x>
    # moves NOTHING even when the app's own folder holds a homonym: the refusal
    # is on the name's folder, not on whether something resolves.
    _deployed(root, 'by_hand.safetensors')             # ours too: h3/lds/by_hand…
    r = _post(client, url, {'deployed_as': 'h3/by_hand.safetensors'})
    assert r.status_code == 400
    assert (root / 'h3' / 'by_hand.safetensors').is_file()
    assert (root / 'h3' / 'lds' / 'by_hand.safetensors').is_file()
    # So is anything that is not a name inside the app's folder.
    for bad in ('h3/lds/../by_hand.safetensors', '../../x.safetensors',
                'h3/lds/notes.txt', ''):
        assert _post(client, url, {'deployed_as': bad}).status_code == 400, bad
    assert (root / 'h3' / 'by_hand.safetensors').is_file()


# ── 4. 🗑 Delete a step ─────────────────────────────────────────────────


def test_deleting_a_cloud_step_trashes_every_file_of_the_pair(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        d = _saves(run, tmp_path, PAIR_100 + PAIR_50)
        run.checkpoint_local_path = str(d / PAIR_100[0])
        db.session.commit()
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/delete',
              {'run_id': run_id, 'step': 100})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert sorted(body['removed']) == sorted(PAIR_100)
    assert body['files_kept'] == [] and body['delete_mode'] == 'app_trash'
    for n in PAIR_100:
        assert not (d / n).exists()
    for n in PAIR_50:
        assert (d / n).is_file()            # the other step is untouched
    with app.app_context():
        names = _trash_names(app)
        assert all(n in names for n in PAIR_100)
        from app.models import CloudTrainingRun
        assert db.session.get(CloudTrainingRun, run_id).checkpoint_local_path is None


def test_deleting_a_step_of_an_active_cloud_run_is_refused(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100, status=next(iter(ct.ACTIVE_STATES)))
        d = _saves(run, tmp_path, PAIR_100)
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/delete',
              {'run_id': run_id, 'step': 100})
    assert r.status_code == 409
    assert 'stop the run' in r.get_json()['error']
    assert all((d / n).is_file() for n in PAIR_100)


def test_deleting_a_local_step_trashes_its_files(app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        d = _local_saves(tmp_path, monkeypatch, PAIR_50 + FINAL)
        ds_id = ds.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/delete',
              {'run_id': None, 'step': 50})
    assert r.status_code == 200, r.get_json()
    assert sorted(r.get_json()['removed']) == sorted(PAIR_50)
    assert not any((d / n).exists() for n in PAIR_50)
    assert (d / FINAL[0]).is_file()
    with app.app_context():
        assert all(n in _trash_names(app) for n in PAIR_50)


def test_deleting_a_local_step_is_refused_while_training_writes_it(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    from app.services import video_training_local as vtl
    with app.app_context():
        ds = _video_dataset(tmp_path)
        d = _local_saves(tmp_path, monkeypatch, PAIR_50)
        monkeypatch.setattr(vtl, 'video_training_progress',
                            lambda dataset_id, user_id=None: {'active': True})
        ds_id = ds.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/delete',
              {'run_id': None, 'step': 50})
    assert r.status_code == 409
    assert all((d / n).is_file() for n in PAIR_50)


def test_a_held_file_is_kept_and_named_rather_than_reported_gone(
        app, client, tmp_path, monkeypatch):
    """The clips' rule (`remove_dataset_clips`): a file the OS holds open stays,
    and the answer says which — never "removed" for a file still on disk."""
    _loras_root(tmp_path, monkeypatch)
    from app.services import trash
    real = trash.send_to_trash

    def holding(path, context=''):
        if os.path.basename(path) == PAIR_100[1]:
            raise trash.TrashLockError('held open')
        return real(path, context=context)
    monkeypatch.setattr(trash, 'send_to_trash', holding)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        d = _saves(run, tmp_path, PAIR_100)
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/checkpoint/delete',
              {'run_id': run_id, 'step': 100})
    assert r.status_code == 200
    body = r.get_json()
    assert body['removed'] == [PAIR_100[0]] and body['files_kept'] == [PAIR_100[1]]
    assert (d / PAIR_100[1]).is_file() and not (d / PAIR_100[0]).exists()


def test_a_step_delete_and_a_clip_removal_share_the_trash_destination():
    """Two verbs of one workspace name ONE destination — the wording on screen
    comes from `delete_mode`, and it must be the same word for both."""
    from app.services import video_bank_service, video_checkpoints
    assert video_checkpoints.DELETE_MODE == video_bank_service.DATASET_CLIP_DELETE_MODE


# ── 5. Download a LOCAL save ────────────────────────────────────────────


def test_the_local_download_serves_a_basename_and_nothing_else(
        app, client, tmp_path, monkeypatch):
    with app.app_context():
        ds = _video_dataset(tmp_path)
        _local_saves(tmp_path, monkeypatch, PAIR_50)
        ds_id = ds.id
    url = f'/api/video-dataset/{ds_id}/train/checkpoint'
    r = client.get(url + '?filename=' + PAIR_50[0])
    assert r.status_code == 200 and r.data == b'L' * 16
    assert client.get(url + '?filename=../' + PAIR_50[0]).status_code == 404
    assert client.get(url + '?filename=nope.safetensors').status_code == 404
    assert client.get(url).status_code == 404
    assert client.get('/api/video-dataset/999/train/checkpoint?filename=x').status_code == 404


# ── 6. ⓘ Details ────────────────────────────────────────────────────────


def test_run_details_are_allow_listed_and_owned(app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        face = _face_dataset()
        ds = _video_dataset(tmp_path)
        other = _video_dataset(tmp_path, name='other', out_dir=str(tmp_path / 'b'))
        run = _run(ds.id, crd.VIDEO, steps=100, gpu_name='RTX 4090',
                   price_per_hour=0.5)
        run.train_params = json.dumps({
            'steps': 100, 'do_i2v': True, 'target_profile': 'wan22_14b',
            'parent_run_id': 7, 'resume_step': 50,
            'resume_ckpt_paths': ['/workspace/old_000000050.safetensors'],
        })
        run.auth_token = 'secret-token'
        db.session.commit()
        _saves(run, tmp_path, PAIR_100)
        face_run = _run(face.id, steps=100)
        other_run = _run(other.id, crd.VIDEO, steps=100)
        ds_id, run_id = ds.id, run.id
        face_run_id, other_run_id = face_run.id, other_run.id
    r = client.get(f'/api/video-dataset/{ds_id}/train/cloud/run/{run_id}')
    assert r.status_code == 200
    d = r.get_json()
    assert d['run_id'] == run_id and d['status'] == 'done'
    assert d['gpu'] == 'RTX 4090' and d['price_per_hour'] == 0.5
    assert d['saves'] == 2 and d['parent_run_id'] == 7
    assert d['params'] == {'steps': 100, 'do_i2v': True, 'target_profile': 'wan22_14b',
                           'parent_run_id': 7, 'resume_step': 50}
    assert 'resume_ckpt_paths' not in json.dumps(d)
    assert 'secret-token' not in json.dumps(d)
    # A face run of the same integer, and another video dataset's run: not ours.
    assert client.get(f'/api/video-dataset/{ds_id}/train/cloud/run/{face_run_id}').status_code == 404
    assert client.get(f'/api/video-dataset/{ds_id}/train/cloud/run/{other_run_id}').status_code == 404


# ── 7. ▶ Continue / ↻ Retry answer the PARALLEL_RUN question ──────────────


def _guardrails_spy(monkeypatch):
    """The launch relay test's idiom (test_video_training_preflight): stop at
    the guardrails and record the answer they were handed."""
    from app.services import cloud_video_training as cvt
    seen = {}

    def spy(dataset_id, fam, dataset_table=None, allow_parallel_run=False):
        seen['allow_parallel_run'] = allow_parallel_run
        raise RuntimeError('stop here — the guardrails were consulted')
    monkeypatch.setattr(ct, '_assert_launch_guardrails', spy)
    monkeypatch.setattr(ct.cfg, 'secret', lambda key, *a, **k: 'k' if key == 'VAST_API_KEY' else None)
    monkeypatch.setattr(cvt, '_count_clips', lambda folder: 2)
    monkeypatch.setattr(cvt.video_training, 'build_job_config', lambda *a, **k: {})
    return seen


def test_continue_from_a_step_relays_allow_parallel_run_to_the_guardrails(
        app, client, tmp_path, monkeypatch):
    """Found by the live check: the section's ▶ asked the PARALLEL_RUN question
    (an active sibling run), the user said yes, and the second POST was refused
    with the SAME question — the continue route dropped the answer the launch
    route relays. A question that cannot be answered is a dead end dressed as
    a dialog."""
    seen = _guardrails_spy(monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        _saves(run, tmp_path, PAIR_100)
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/cloud/continue',
              {'run_id': run_id, 'extra_steps': 500, 'from_step': 100,
               'allow_parallel_run': True})
    assert r.status_code == 409, r.get_json()
    assert seen.get('allow_parallel_run') is True


def test_retry_relays_allow_parallel_run_to_the_guardrails(
        app, client, tmp_path, monkeypatch):
    seen = _guardrails_spy(monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100, status='error')
        ds_id, run_id = ds.id, run.id
    r = _post(client, f'/api/video-dataset/{ds_id}/train/cloud/retry',
              {'run_id': run_id, 'allow_parallel_run': True})
    assert r.status_code == 409, r.get_json()
    assert seen.get('allow_parallel_run') is True
    # And without the answer, the guardrails are asked with the default — the
    # relay is the user's word, never an always-on bypass.
    seen.clear()
    _post(client, f'/api/video-dataset/{ds_id}/train/cloud/retry', {'run_id': run_id})
    assert seen.get('allow_parallel_run') is False
