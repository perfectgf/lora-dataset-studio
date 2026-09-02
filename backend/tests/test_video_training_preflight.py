"""☁ The video lane's pre-launch report, and the one flag its launch dropped.

The video launcher refuses everything the preflight reports — one refusal at a
time, AFTER the click, which on the cloud lane is after the money question was
answered. This route asks all of it at once, before, in the image preflight's
shape (`checks` + `verdict`) so the same readiness card renders both lanes.

No ffmpeg, no PyAV, no torch: the media seams are the routes suite's.
"""
import pytest

from app.services import video_bank_service as svc

from _video_extra import detect_source_stub, video_extra_ready  # noqa: F401


@pytest.fixture()
def seams(monkeypatch):
    def _run(args):
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''
    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 120.0, 'fps_native': 30.0, 'width': 1920, 'height': 1080,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096})
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub([
        {'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
        {'start_s': 41.25, 'end_s': 50.0, 'start_frame': 1237, 'end_frame': 1500}]))
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')


def _promoted(client, tmp_path, **promote):
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'a.mp4').write_bytes(b'\x00' * 32)
    bank_id = client.post('/api/video-bank/create',
                          json={'name': 'rushes', 'folder': str(folder)}).get_json()['id']
    assert client.post(f'/api/video-bank/{bank_id}/pipeline', json={}).status_code == 202
    assert client.post(f'/api/video-bank/{bank_id}/triage',
                       json={'ids': [], 'status': 'keep'}).status_code == 200
    body = {'name': 'set', 'target_profile': 'wan22_14b', 'frames': 81, **promote}
    r = client.post(f'/api/video-bank/{bank_id}/promote', json=body)
    assert r.status_code == 202, r.get_json()
    return r.get_json()['id']


def _rows(report):
    return {c['id']: c for c in report['checks']}


def test_the_report_has_the_image_preflight_shape_and_a_verdict(client, tmp_path, seams):
    ds_id = _promoted(client, tmp_path)

    r = client.get(f'/api/video-dataset/{ds_id}/train/preflight')

    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['ok'] is True
    assert body['verdict'] in ('ready', 'warnings', 'blocked')
    for row in body['checks']:
        assert {'id', 'label', 'status', 'detail', 'target', 'scope'} <= set(row)
        assert row['status'] in ('ok', 'warn', 'fail')
    # A fact about the set, never a quality guard-rail: the card must never
    # offer "Continue anyway" over these.
    assert body['can_override'] is False


def test_the_two_lanes_read_different_things(client, tmp_path, seams):
    """Machine rows (ai-toolkit, weights) mean nothing for a rented pod; account
    rows (the vast key, the run limit, the budget) mean nothing for this box."""
    ds_id = _promoted(client, tmp_path)

    local = _rows(client.get(f'/api/video-dataset/{ds_id}/train/preflight').get_json())
    cloud = _rows(client.get(f'/api/video-dataset/{ds_id}/train/preflight?lane=cloud').get_json())

    assert all(c['scope'] != 'cloud' for c in local.values())
    assert all(c['scope'] != 'machine' for c in cloud.values())
    assert 'aitoolkit' in local and 'aitoolkit' not in cloud
    assert 'vast' in cloud and 'vast' not in local
    # The dataset rows are on BOTH lanes — they are true wherever the job runs.
    for lane in (local, cloud):
        assert 'clips' in lane and 'target' in lane


def test_no_vast_key_is_a_BLOCKER_on_the_cloud_lane(client, tmp_path, seams, monkeypatch):
    from app.services import cloud_training as ct
    monkeypatch.setattr(ct.cfg, 'secret', lambda key, *a, **k: '' if key == 'VAST_API_KEY' else None)
    ds_id = _promoted(client, tmp_path)

    body = client.get(f'/api/video-dataset/{ds_id}/train/preflight?lane=cloud').get_json()

    assert _rows(body)['vast']['status'] == 'fail'
    assert body['verdict'] == 'blocked'
    assert any('vast.ai' in b for b in body['blockers'])


def test_an_empty_folder_blocks_both_lanes(client, tmp_path, seams):
    import os
    ds_id = _promoted(client, tmp_path)
    out = client.get(f'/api/video-dataset/{ds_id}').get_json()['output_dir']
    for name in os.listdir(out):
        os.remove(os.path.join(out, name))

    for lane in ('', '?lane=cloud'):
        body = client.get(f'/api/video-dataset/{ds_id}/train/preflight{lane}').get_json()
        assert _rows(body)['clips']['status'] == 'fail', lane
        assert body['verdict'] == 'blocked', lane


def test_a_missing_reference_set_is_a_blocker_with_its_destination(client, tmp_path, seams,
                                                                    monkeypatch):
    """The row has to point at the section that fixes it: the readiness card's
    Fix → button jumps there, and a blocker with no destination is a dead end."""
    from app.services import video_targets
    profile = dict(video_targets.get('wan22_14b') or {})
    profile['requires_references'] = True
    monkeypatch.setattr(video_targets, 'get',
                        lambda key: profile if key == 'wan22_14b' else None)
    ds_id = _promoted(client, tmp_path)

    body = client.get(f'/api/video-dataset/{ds_id}/train/preflight').get_json()

    row = _rows(body)['references']
    assert row['status'] == 'fail'
    assert row['target'] == 'references'
    assert body['verdict'] == 'blocked'


def test_an_unknown_dataset_is_a_404(client):
    assert client.get('/api/video-dataset/9999/train/preflight').status_code == 404


def test_the_cloud_launch_relays_allow_parallel_run_to_the_guardrails(client, tmp_path,
                                                                       seams, monkeypatch):
    """The guardrails' `PARALLEL_RUN:` refusal is CONFIRMABLE by contract, and the
    answer travels as `allow_parallel_run`. The image route relayed it; the
    video route dropped it, so the question the server asked could never be
    answered from the video lane."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    seen = {}

    def spy(dataset_id, fam, dataset_table=None, allow_parallel_run=False):
        seen['allow_parallel_run'] = allow_parallel_run
        raise RuntimeError('stop here — the guardrails were consulted')
    monkeypatch.setattr(ct, '_assert_launch_guardrails', spy)
    monkeypatch.setattr(ct.cfg, 'secret', lambda key, *a, **k: 'k' if key == 'VAST_API_KEY' else None)
    monkeypatch.setattr(cvt, '_count_clips', lambda folder: 2)
    # The validation build runs BEFORE the guardrails and wants a clip size the
    # seams' promote never records; its result is discarded by design, so a
    # stand-in is exact here — the guardrails are the thing under test.
    monkeypatch.setattr(cvt.video_training, 'build_job_config', lambda *a, **k: {})
    ds_id = _promoted(client, tmp_path)

    r = client.post(f'/api/video-dataset/{ds_id}/train/cloud',
                    json={'steps': 500, 'allow_parallel_run': True})

    assert r.status_code == 409, r.get_json()
    assert seen.get('allow_parallel_run') is True
