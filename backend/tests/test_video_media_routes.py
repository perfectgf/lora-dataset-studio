"""🎬 Serving the bytes — the two routes that let anyone actually WATCH a shot.

A video bank writes no clip files: cutting a shot means re-encoding it, and that
cost is only paid at promotion, for the shots that were kept. So the only way to
play a detected shot is to point a player at the SOURCE with a media fragment
(`#t=41,46`) and let the browser fetch that span.

That sentence is only true if the response is Range-capable, which is why the
206 test below is the point of this file rather than a nicety. A `send_file`
without `conditional=True` answers 200 with the whole body; the fragment still
"works" — the player seeks to 41 s — but it has downloaded a two-hour rush to
show five seconds of it. There is no error, no warning and no visible symptom on
a small test file, which is exactly how that regression would ship.

The second thing pinned here is that a refusal is a 404. `relpath` and `filename`
are columns in a database the user can reach; resolving either without checking
it still lands under the folder it belongs to is how `..` reads a file the bank
was never pointed at. Answering 500 (or 403) on the escape would confirm to
whoever tried it that the path exists — so every refusal says the same thing.
"""
import os

import pytest

from app.services import video_bank_service as svc

from _video_extra import detect_source_stub

SOURCE_BYTES = b'\x11' * 4096


@pytest.fixture()
def seams(monkeypatch):
    """The four media seams, with an "encoder" that writes a file big enough to
    Range-request. The real one is ffmpeg; nothing here needs it."""
    def _run(args):
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x22' * 2048)
        return 0, ''

    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 120.0, 'fps_native': 30.0, 'width': 1920, 'height': 1080,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': len(SOURCE_BYTES)})
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub([
        {'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
        {'start_s': 41.25, 'end_s': 50.0, 'start_frame': 1237, 'end_frame': 1500}]))
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)
    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')


def _bank_with_one_source(client, tmp_path, name='a.mp4'):
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(SOURCE_BYTES)
    r = client.post('/api/video-bank/create',
                    json={'name': 'rushes', 'folder': str(folder)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']
    source_id = client.get(f'/api/video-bank/{bank_id}/sources') \
        .get_json()['sources'][0]['id']
    return bank_id, source_id, folder


# --- the bank's source file -----------------------------------------------------

def test_a_range_request_on_a_source_is_answered_206_with_a_content_range(
        client, tmp_path):
    """THE test of this file.

    A media fragment only costs its own span if the server serves 206. On a 200
    the browser downloads the entire rush to show one shot — invisible here, and
    the difference between "instant" and "four minutes" on a real 6 GB file."""
    bank_id, source_id, _ = _bank_with_one_source(client, tmp_path)

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media',
                   headers={'Range': 'bytes=0-1023'})

    assert r.status_code == 206, 'a 200 here means the whole file was sent'
    assert r.headers['Content-Range'] == f'bytes 0-1023/{len(SOURCE_BYTES)}'
    assert r.headers.get('Accept-Ranges') == 'bytes'
    assert len(r.get_data()) == 1024


def test_a_range_past_the_first_block_returns_that_block_and_not_the_head(
        client, tmp_path):
    """The lightbox seeks INTO the file — a server that ignores the offset and
    replies with the head would play the wrong shot with no error anywhere."""
    bank_id, source_id, folder = _bank_with_one_source(client, tmp_path)
    payload = bytes(range(256)) * 16                      # 4096 distinguishable bytes
    (folder / 'a.mp4').write_bytes(payload)

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media',
                   headers={'Range': 'bytes=2048-2063'})

    assert r.status_code == 206
    assert r.get_data() == payload[2048:2064]


def test_without_a_range_header_the_whole_file_is_served(client, tmp_path):
    bank_id, source_id, _ = _bank_with_one_source(client, tmp_path)

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media')

    assert r.status_code == 200
    assert r.get_data() == SOURCE_BYTES
    assert r.headers['Content-Type'].startswith('video/mp4')


def test_the_container_decides_the_mimetype(client, tmp_path):
    """`mimetypes.guess_type` is registry-driven on Windows and answers None for
    .mkv on plenty of installs — and send_file RAISES on a name it cannot guess.
    That would turn "your browser cannot decode Matroska" into a 500."""
    bank_id, source_id, _ = _bank_with_one_source(client, tmp_path, name='a.mkv')

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media')

    assert r.status_code == 200
    assert r.headers['Content-Type'].startswith('video/x-matroska')


def test_a_relpath_that_escapes_the_bank_folder_is_a_404_not_a_500(
        app, client, tmp_path):
    """404, and the same 404 as "no such source".

    A 500 tells whoever tried that the file exists and the read got far enough to
    crash; a distinct 403 tells them the path was real and the guard caught it.
    Both are answers. One answer for every refusal is the only one that isn't."""
    from app.models import VideoSource
    from app.extensions import db
    secret = tmp_path / 'secret.txt'
    secret.write_bytes(b'API_KEY=hunter2')
    bank_id, source_id, _ = _bank_with_one_source(client, tmp_path)
    with app.app_context():
        row = db.session.get(VideoSource, source_id)
        row.relpath = os.path.join('..', 'secret.txt')
        db.session.commit()

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media')

    assert r.status_code == 404
    assert b'hunter2' not in r.get_data()


def test_a_sibling_folder_sharing_the_banks_prefix_is_not_reachable(
        app, client, tmp_path):
    """The prefix check has to carry the separator: without it a bank rooted at
    `…/rushes` accepts `…/rushes-secret/x.mp4` as "contained"."""
    from app.models import VideoSource
    from app.extensions import db
    sibling = tmp_path / 'rushes-secret'
    sibling.mkdir()
    (sibling / 'x.mp4').write_bytes(b'private')
    bank_id, source_id, _ = _bank_with_one_source(client, tmp_path)
    with app.app_context():
        row = db.session.get(VideoSource, source_id)
        row.relpath = os.path.join('..', 'rushes-secret', 'x.mp4')
        db.session.commit()

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media')

    assert r.status_code == 404


def test_a_source_whose_file_moved_away_is_a_404(client, tmp_path):
    """Ordinary weather: a bank points at a LIVE folder people reorganise."""
    bank_id, source_id, folder = _bank_with_one_source(client, tmp_path)
    (folder / 'a.mp4').unlink()

    r = client.get(f'/api/video-bank/{bank_id}/source/{source_id}/media')

    assert r.status_code == 404


def test_an_unknown_bank_or_source_is_a_404(client, tmp_path):
    bank_id, source_id, _ = _bank_with_one_source(client, tmp_path)

    assert client.get(f'/api/video-bank/9999/source/{source_id}/media') \
        .status_code == 404
    assert client.get(f'/api/video-bank/{bank_id}/source/9999/media') \
        .status_code == 404


# --- a promoted clip ------------------------------------------------------------

def _promote(client, tmp_path):
    bank_id, _source_id, _folder = _bank_with_one_source(client, tmp_path)
    assert client.post(f'/api/video-bank/{bank_id}/pipeline',
                       json={}).status_code == 202
    assert client.post(f'/api/video-bank/{bank_id}/triage',
                       json={'ids': [], 'status': 'keep'}).status_code == 200
    r = client.post(f'/api/video-bank/{bank_id}/promote',
                    json={'name': 'wan set', 'target_profile': 'wan22_14b',
                          'frames': 81})
    assert r.status_code == 202, r.get_json()
    ds_id = r.get_json()['id']
    items = client.get(f'/api/video-dataset/{ds_id}').get_json()['items']
    return ds_id, items


def test_a_promoted_clip_can_be_played_back(client, tmp_path, seams):
    """A dataset you cannot re-watch is a list of filenames. Watching the cut is
    how a wrong length is caught before a training run pays for it."""
    ds_id, items = _promote(client, tmp_path)

    r = client.get(f'/api/video-dataset/{ds_id}/clip/{items[0]["id"]}/media')

    assert r.status_code == 200
    assert len(r.get_data()) == 2048
    assert r.headers['Content-Type'].startswith('video/mp4')


def test_a_promoted_clip_also_serves_ranges(client, tmp_path, seams):
    ds_id, items = _promote(client, tmp_path)

    r = client.get(f'/api/video-dataset/{ds_id}/clip/{items[0]["id"]}/media',
                   headers={'Range': 'bytes=0-99'})

    assert r.status_code == 206
    assert len(r.get_data()) == 100


def test_a_dataset_clip_filename_that_escapes_its_folder_is_a_404(
        app, client, tmp_path, seams):
    """We write these filenames ourselves — and "we wrote it ourselves" is the
    assumption every path-traversal write-up opens on. The column is reachable."""
    from app.models import VideoDatasetClip
    from app.extensions import db
    secret = tmp_path / 'secret.txt'
    secret.write_bytes(b'API_KEY=hunter2')
    ds_id, items = _promote(client, tmp_path)
    with app.app_context():
        row = db.session.get(VideoDatasetClip, items[0]['id'])
        row.filename = os.path.join('..', '..', '..', '..', 'secret.txt')
        db.session.commit()

    r = client.get(f'/api/video-dataset/{ds_id}/clip/{items[0]["id"]}/media')

    assert r.status_code == 404
    assert b'hunter2' not in r.get_data()


def test_an_unknown_dataset_or_clip_is_a_404(client, tmp_path, seams):
    ds_id, items = _promote(client, tmp_path)

    assert client.get(f'/api/video-dataset/9999/clip/{items[0]["id"]}/media') \
        .status_code == 404
    assert client.get(f'/api/video-dataset/{ds_id}/clip/9999/media') \
        .status_code == 404


def test_the_media_routes_sit_behind_the_same_network_guard_as_every_route(app):
    """Not a per-route decorator: `install_network_guard` is an app-level
    before_request, so a route that serves arbitrary file bytes cannot be the one
    door that forgot to close. Pinned rather than assumed — this was checked once,
    and a future refactor to per-blueprint guards would silently exempt it."""
    from app import netguard
    funcs = [f.__name__ for f in app.before_request_funcs.get(None, [])]
    assert '_network_guard' in funcs
    assert netguard.install_network_guard is not None
