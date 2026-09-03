"""The ◉ Graph of a VIDEO dataset and the previews behind its pills.

The tree must be the shape the image graph draws — nodes keyed by record_id,
edges parent→child with the resumed step — because that layout is what the
video workspace reuses. Samples are served by NAME through the lane's own
listing, and a poster is cut once and cached.
"""
import json
import os

from app.extensions import db
from app.services import cloud_run_dataset as crd
from app.services import video_lineage
from test_cloud_video_launch import _face_dataset, _run, _video_dataset
from test_cloud_video_lifecycle import _saves
from test_video_checkpoints import _deployed, _local_saves, _loras_root

PAIR_100 = ['video_surf_000000100_high_noise.safetensors',
            'video_surf_000000100_low_noise.safetensors']
FINAL = ['video_surf.safetensors']


def _samples(folder, names):
    os.makedirs(folder, exist_ok=True)
    for n in names:
        with open(os.path.join(folder, n), 'wb') as fh:
            fh.write(b'\x00' * 32)


def _continued(ds_id, parent_id, resume_step, steps=200):
    run = _run(ds_id, crd.VIDEO, steps=steps)
    run.train_params = json.dumps({'steps': steps, 'parent_run_id': parent_id,
                                   'resume_step': resume_step,
                                   'target_profile': 'wan22_14b'})
    db.session.commit()
    return run


# ── 1. The tree ──────────────────────────────────────────────────────────


def test_the_tree_links_a_continuation_to_the_step_it_resumed_from(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        a = _run(ds.id, crd.VIDEO, steps=100)
        _saves(a, tmp_path, PAIR_100)
        b = _continued(ds.id, a.id, 100)
        _saves(b, tmp_path, ['video_surf_000000200.safetensors'])
        _local_saves(tmp_path, monkeypatch, FINAL)
        ds_id, a_id, b_id = ds.id, a.id, b.id
    r = client.get(f'/api/video-dataset/{ds_id}/train/lineage')
    assert r.status_code == 200
    t = r.get_json()
    assert t['root_id'] is None and t['current_id'] is None and t['single'] is False
    ids = [n['record_id'] for n in t['nodes']]
    assert ids == [a_id, b_id, -ds_id], 'cloud runs oldest first, then the local node'
    assert t['edges'] == [{'parent': a_id, 'child': b_id, 'resumed_from': 100, 'superseded': False}]
    by_id = {n['record_id']: n for n in t['nodes']}
    assert by_id[b_id]['parent_record_id'] == a_id and by_id[b_id]['resumed_from'] == 100
    assert by_id[b_id]['origin_unknown'] is False
    assert all(n['train_type'] == 'video' and n['version'] is None for n in t['nodes'])
    assert by_id[a_id]['source'] == 'cloud' and by_id[a_id]['run_id'] == a_id
    local = by_id[-ds_id]
    assert local['source'] == 'local' and local['run_id'] is None
    assert local['run_name'].endswith(f'_ds{ds_id}')
    # A cloud pill is ONE step with BOTH files; a download per file.
    (pill,) = by_id[a_id]['checkpoints']
    assert pill['step'] == 100 and pill['final'] is False
    assert [f['filename'] for f in pill['files']] == PAIR_100
    assert pill['download_urls'] == [
        f'/api/video-dataset/{ds_id}/train/cloud/checkpoint?run_id={a_id}&filename={n}' for n in PAIR_100]
    assert pill['download_url'] == pill['download_urls'][0] and pill['filename'] == PAIR_100[0]
    assert pill['present'] is True and pill['testable'] is False
    assert pill['preview_url'] is None and pill['preview_count'] == 0
    # The local final carries no number (no job config on this test machine).
    (lp,) = local['checkpoints']
    assert lp['step'] is None and lp['final'] is True
    assert lp['download_urls'] == [f'/api/video-dataset/{ds_id}/train/checkpoint?filename={FINAL[0]}']


def test_a_parent_that_is_gone_leaves_an_honest_root(app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        b = _continued(ds.id, 999, 100)
        _saves(b, tmp_path, ['video_surf_000000200.safetensors'])
        ds_id, b_id = ds.id, b.id
    t = client.get(f'/api/video-dataset/{ds_id}/train/lineage').get_json()
    (n,) = t['nodes']
    assert n['record_id'] == b_id and n['parent_record_id'] is None
    assert n['origin_unknown'] is True and n['resumed_from'] == 100
    assert t['edges'] == [] and t['single'] is True


def test_the_tree_shows_nothing_of_the_face_run_of_the_same_id(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        face = _face_dataset()
        ds = _video_dataset(tmp_path)
        assert face.id == ds.id
        face_run = _run(face.id, steps=100)
        _saves(face_run, tmp_path, ['lora_trg_000000100.safetensors'])
        ds_id = ds.id
    t = client.get(f'/api/video-dataset/{ds_id}/train/lineage').get_json()
    assert t['nodes'] == [] and t['edges'] == []
    assert client.get('/api/video-dataset/999/train/lineage').status_code == 404


def test_the_local_final_takes_its_number_from_the_job_config(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    monkeypatch.setattr(lt, '_jobs_dir', lambda: jobs)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        _local_saves(tmp_path, monkeypatch, FINAL)
        from app.services import video_training_local as vtl
        (jobs / (vtl.local_run_name(ds) + '.json')).write_text(json.dumps({
            'job': 'extension', 'config': {'name': 'x', 'process': [
                {'type': 'sd_trainer', 'train': {'steps': 1500}}]}}), encoding='utf-8')
        assert video_lineage.local_total_steps(ds) == 1500
        t = video_lineage.tree('local', ds.id)
        (pill,) = t['nodes'][0]['checkpoints']
        assert pill['step'] == 1500 and pill['final'] is True
        assert t['nodes'][0]['steps'] == 1500


# ── 2. Samples and posters ───────────────────────────────────────────────


def test_samples_are_listed_by_step_and_served_by_name_only(
        app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        d = _saves(run, tmp_path, PAIR_100)
        _samples(d / 'samples', ['1725000000__000000100_0.mp4', '1725000000__000000100_1.mp4',
                                 '1725000000__000000050_0.mp4', 'notes.txt', 'grid.png'])
        _deployed(_loras_root(tmp_path, monkeypatch), *PAIR_100)
        ds_id, run_id = ds.id, run.id
    r = client.get(f'/api/video-dataset/{ds_id}/train/samples?run_id={run_id}')
    assert r.status_code == 200
    d = r.get_json()
    assert d['run_id'] == run_id
    assert [(s['step'], s['prompt_idx']) for s in d['samples']] == [(100, 0), (100, 1), (50, 0)]
    assert all(s['kind'] == 'video' for s in d['samples'])
    assert d['samples'][0]['url'] == (
        f'/api/video-dataset/{ds_id}/train/sample?run_id={run_id}&filename=1725000000__000000100_0.mp4')
    assert d['samples'][0]['poster_url'].endswith('/train/sample/poster?run_id='
                                                 f'{run_id}&filename=1725000000__000000100_0.mp4')
    # The pill of step 100 previews prompt 0 and counts both samples; deployed
    # (both files) reads as testable with the LoraLoader name.
    t = client.get(f'/api/video-dataset/{ds_id}/train/lineage').get_json()
    (pill,) = t['nodes'][0]['checkpoints']
    assert pill['preview_count'] == 2 and pill['preview_status'] == 'ready'
    assert pill['preview_url'] == d['samples'][0]['poster_url']
    assert pill['sample_url'] == d['samples'][0]['url']
    assert pill['testable'] is True
    assert pill['deployed_filename'].replace('\\', '/') == 'h3/lds/' + PAIR_100[0]
    # Serving: by name, video/mp4, and never a path.
    r = client.get(d['samples'][0]['url'])
    assert r.status_code == 200 and r.mimetype == 'video/mp4' and r.data == b'\x00' * 32
    assert client.get(f'/api/video-dataset/{ds_id}/train/sample?run_id={run_id}'
                      '&filename=../samples/1725000000__000000100_0.mp4').status_code == 404
    assert client.get(f'/api/video-dataset/{ds_id}/train/sample?run_id={run_id}'
                      '&filename=notes.txt').status_code == 404
    assert client.get(f'/api/video-dataset/{ds_id}/train/sample?run_id=999'
                      '&filename=1725000000__000000100_0.mp4').status_code == 404


def test_the_local_lane_answers_without_a_run_id(app, client, tmp_path, monkeypatch):
    _loras_root(tmp_path, monkeypatch)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        d = _local_saves(tmp_path, monkeypatch, FINAL)
        _samples(d / 'samples', ['1725000000__000000300_0.mp4'])
        ds_id = ds.id
    d = client.get(f'/api/video-dataset/{ds_id}/train/samples').get_json()
    assert d['run_id'] is None and [s['step'] for s in d['samples']] == [300]
    assert d['samples'][0]['url'] == (
        f'/api/video-dataset/{ds_id}/train/sample?filename=1725000000__000000300_0.mp4')
    assert client.get(d['samples'][0]['url']).status_code == 200


def test_a_poster_is_cut_once_and_cached(app, client, tmp_path, monkeypatch):
    """The frame cutter is the video bank's (PyAV); it is stubbed here because a
    32-byte 'clip' decodes to nothing — what is under test is the caching
    and the honest 404 when no frame can be cut."""
    _loras_root(tmp_path, monkeypatch)
    from app.services import video_bank_service as vbs
    calls = []

    def cut(src, ts, dst):
        calls.append(src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as fh:
            fh.write(b'\xff\xd8JPEG')
        return True
    monkeypatch.setattr(vbs, '_write_thumbnail', cut)
    with app.app_context():
        ds = _video_dataset(tmp_path)
        run = _run(ds.id, crd.VIDEO, steps=100)
        d = _saves(run, tmp_path, PAIR_100)
        _samples(d / 'samples', ['1725000000__000000100_0.mp4', '1725000000__000000100_1.png'])
        ds_id, run_id = ds.id, run.id
    url = f'/api/video-dataset/{ds_id}/train/sample/poster?run_id={run_id}&filename='
    r = client.get(url + '1725000000__000000100_0.mp4')
    assert r.status_code == 200 and r.mimetype == 'image/jpeg' and r.data == b'\xff\xd8JPEG'
    r = client.get(url + '1725000000__000000100_0.mp4')
    assert r.status_code == 200
    assert len(calls) == 1, 'the second request served the cached poster'
    # An image sample is its own poster; no frame is cut for it.
    r = client.get(url + '1725000000__000000100_1.png')
    assert r.status_code == 200 and r.mimetype == 'image/png'
    assert len(calls) == 1
    # No frame can be cut (no PyAV, unreadable file): an honest 404.
    monkeypatch.setattr(vbs, '_write_thumbnail', lambda *a: False)
    with app.app_context():
        _samples(d / 'samples', ['1725000000__000000200_0.mp4'])
    assert client.get(url + '1725000000__000000200_0.mp4').status_code == 404
    assert client.get(url + 'nope.mp4').status_code == 404
