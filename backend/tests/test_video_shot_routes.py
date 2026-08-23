"""The Find shots API: the threshold, the preview, the re-cut, the single take.

These routes ride the same envelope as a trim — 404 unknown, 409 a pass owns
the bank, 400 bad input — because they change BOUNDS, which is what every later
pass and the encoder read.

The one status code worth arguing about is on a file with no cached vector:
503, not 400. Nothing about the request is wrong; the file was simply detected
before the cache existed, and the fix is one pass, not a different body.
"""
import pytest

from app.config import LOCAL_USER
from app.models import VideoSource
from app.services import shot_probs
from app.services import video_bank_service as svc

# Imported for its autouse effect; see _video_extra.py for why not importorskip.
from _video_extra import video_extra_ready  # noqa: F401


def _probe(_path):
    return {'duration_s': 4.0, 'fps_native': 25.0, 'width': 640, 'height': 360,
            'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096}


def _detect_seam(single):
    def run(path, fps_native=None, **kwargs):
        from app.services import shot_detect as sd
        probs = {'single': single, 'all': None}
        return {'clips': sd.clips_from_probs(probs, fps_native=fps_native or 25.0,
                                             threshold=kwargs.get('threshold'),
                                             min_shot_frames=1),
                'probs': probs, 'fps_native': fps_native or 25.0,
                'frame_count': len(single)}
    return run


@pytest.fixture()
def bank(app, client, tmp_path, monkeypatch):
    folder = tmp_path / 'rushes'
    folder.mkdir()
    (folder / 'a.mp4').write_bytes(b'\x00' * 32)
    single = [0.0] * 100
    single[40] = 0.6
    single[70] = 0.95
    monkeypatch.setattr(svc, '_probe_file', _probe)
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    monkeypatch.setattr(svc, '_detect_source', _detect_seam(single))
    with app.app_context():
        row, _added = svc.create_bank(LOCAL_USER, 'rushes', str(folder))
        bank_id = row.id
        svc.start_probe(app, LOCAL_USER, bank_id)
        svc.start_detect(app, LOCAL_USER, bank_id)
        source_id = VideoSource.query.filter_by(bank_id=bank_id).one().id
    return bank_id, source_id


def test_the_bank_threshold_round_trips_through_the_payload(client, bank):
    bank_id, _source_id = bank

    saved = client.post(f'/api/video-bank/{bank_id}/shot-threshold',
                        json={'threshold': 0.75})

    assert saved.status_code == 200
    payload = client.get(f'/api/video-bank/{bank_id}').get_json()
    assert payload['shot_detect']['threshold'] == 0.75


def test_clearing_the_threshold_is_null_and_not_zero(client, bank):
    bank_id, _source_id = bank
    client.post(f'/api/video-bank/{bank_id}/shot-threshold', json={'threshold': 0.75})

    client.post(f'/api/video-bank/{bank_id}/shot-threshold', json={'threshold': None})

    payload = client.get(f'/api/video-bank/{bank_id}').get_json()
    assert payload['shot_detect']['threshold'] is None


def test_a_threshold_outside_the_range_is_a_400_with_a_sentence(client, bank):
    bank_id, _source_id = bank

    refused = client.post(f'/api/video-bank/{bank_id}/shot-threshold',
                          json={'threshold': 7})

    assert refused.status_code == 400
    assert 'between 0 and 1' in refused.get_json()['error']


def test_a_per_file_threshold_answers_with_what_now_applies(client, bank):
    bank_id, source_id = bank

    out = client.post(
        f'/api/video-bank/{bank_id}/source/{source_id}/shot-threshold',
        json={'threshold': 0.4}).get_json()

    assert out['threshold'] == 0.4 and out['effective'] == 0.4


def test_the_dry_run_previews_without_touching_a_single_clip(client, bank):
    bank_id, source_id = bank
    before = client.get(f'/api/video-bank/{bank_id}/clips').get_json()['clips']

    out = client.post(f'/api/video-bank/{bank_id}/shot-dry-run',
                      json={'source_id': source_id,
                            'thresholds': [0.5, 0.8]}).get_json()

    assert out['rows'] == [{'threshold': 0.5, 'shots': 3},
                           {'threshold': 0.8, 'shots': 2}]
    after = client.get(f'/api/video-bank/{bank_id}/clips').get_json()['clips']
    assert [c['id'] for c in after] == [c['id'] for c in before]


def test_recutting_the_bank_answers_with_what_it_did_and_what_it_left(client, bank):
    bank_id, _source_id = bank

    out = client.post(f'/api/video-bank/{bank_id}/recut',
                      json={'threshold': 0.8}).get_json()

    assert out['clips'] == 2 and out['skipped'] == 0 and out['single_shot'] == 0


def test_recutting_one_file_says_how_many_hand_made_cuts_it_replaced(client, bank):
    bank_id, source_id = bank

    out = client.post(f'/api/video-bank/{bank_id}/source/{source_id}/recut',
                      json={'threshold': 0.8}).get_json()

    assert out['clips'] == 2 and out['replaced_manual'] == 0


def test_a_file_with_no_cached_vector_is_a_503_not_a_400(app, client, bank):
    """Nothing about the request is wrong. The file was detected before the
    cache existed, and the fix is one pass — the same shape of answer this lane
    already gives for a missing extra."""
    bank_id, source_id = bank
    with app.app_context():
        shot_probs.forget(bank_id, source_id)

    refused = client.post(f'/api/video-bank/{bank_id}/source/{source_id}/recut',
                          json={})

    assert refused.status_code == 503
    assert 'Find shots' in refused.get_json()['error']


def test_declaring_a_file_a_single_take_leaves_it_with_exactly_one_clip(app,
                                                                       client,
                                                                       bank):
    bank_id, source_id = bank

    out = client.post(
        f'/api/video-bank/{bank_id}/source/{source_id}/single-shot').get_json()

    assert out['clips'] == 1
    clips = client.get(f'/api/video-bank/{bank_id}/clips').get_json()['clips']
    assert len(clips) == 1
    assert clips[0]['start_s'] == 0.0 and clips[0]['end_s'] == 4.0


def test_the_source_list_says_which_files_can_be_re_cut_instantly(client, bank):
    bank_id, _source_id = bank

    sources = client.get(f'/api/video-bank/{bank_id}').get_json()['sources']

    assert sources[0]['has_probs'] is True


def test_every_shot_route_404s_on_a_bank_that_does_not_exist(client):
    for path, body in (('/api/video-bank/9999/shot-threshold', {'threshold': 0.5}),
                       ('/api/video-bank/9999/recut', {}),
                       ('/api/video-bank/9999/shot-dry-run', {}),
                       ('/api/video-bank/9999/source/1/single-shot', None)):
        assert client.post(path, json=body).status_code == 404


def test_a_shot_route_is_refused_while_a_pass_owns_the_bank(app, client, bank,
                                                            monkeypatch):
    """Same 409 a trim gets, for the same reason: the thumbnails pass reads a
    clip's bounds, shells out, then stamps the state — an edit landing between
    those two writes produces a thumbnail of a span that no longer exists."""
    bank_id, _source_id = bank
    from app.services import bank_jobs
    monkeypatch.setattr(bank_jobs, 'get',
                        lambda key: {'kind': 'thumbs', 'finished': False})

    refused = client.post(f'/api/video-bank/{bank_id}/recut', json={})

    assert refused.status_code == 409
    assert refused.get_json()['busy_kind'] == 'thumbs'
