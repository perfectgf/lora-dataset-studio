"""🔎 The HTTP surface of video search — the same envelope as every other pass.

The embedding pass is a pass like the others (202, one job per bank, 409 with
`busy_kind` when the bank is occupied), and the search is a plain read. What is
worth pinning here is the DIFFERENCE between the two refusals a user meets, since
they need two different actions:

  * 400 — "the bank has no embeddings yet, run the pass". Fixable here, now.
  * 503 — "this install cannot run CLIP at all". A different problem entirely,
    and telling someone to run a pass that cannot start is how a UI sends them
    round a loop.

Also pinned: the ranking comes back as ROWS in ranked order. The clip list
endpoint sorts by file and start time on purpose, so re-fetching the rows in a
second request would quietly throw the ranking away — the exact bug that shape
exists to prevent.
"""
import pytest

from app.services import video_bank_service as svc
from app.services import clip_text_encoder
from app.services import video_clip_search as vcs

# The video-extra gate answers for the MACHINE, so without this these route
# tests pass where PyAV/ffmpeg are installed and 503 where they are not.
# Imported for its autouse effect; see _video_extra.py for why not importorskip.
from _video_extra import detect_source_stub, video_extra_ready  # noqa: F401


@pytest.fixture()
def seams(monkeypatch):
    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 120.0, 'fps_native': 30.0, 'width': 1920, 'height': 1080,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096})
    monkeypatch.setattr(svc, '_detect_source', detect_source_stub([
        {'start_s': 0.0, 'end_s': 8.0, 'start_frame': 0, 'end_frame': 240},
        {'start_s': 41.25, 'end_s': 50.0, 'start_frame': 1237, 'end_frame': 1500}]))
    monkeypatch.setattr(svc, '_write_thumbnail', lambda *a, **k: True)


def _bank(client, tmp_path):
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'a.mp4').write_bytes(b'\x00' * 32)
    r = client.post('/api/video-bank/create',
                    json={'name': 'rushes', 'folder': str(folder)})
    bank_id = r.get_json()['id']
    assert client.post(f'/api/video-bank/{bank_id}/pipeline',
                       json={}).status_code == 202
    return bank_id


def _unit(*c):
    import numpy as np
    v = np.asarray([float(x) for x in c], dtype='float32')
    return v / (float(np.linalg.norm(v)) + 1e-8)


@pytest.fixture()
def embed_seams(monkeypatch):
    """The real pass, with only its two heavy seams faked — so the routes, the
    job envelope and the vector store are all exercised without PyAV or torch."""
    from app.services import clip_image_encoder

    class _NoModel:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(clip_image_encoder, 'ImageEncoder', _NoModel)
    monkeypatch.setattr(vcs, '_write_frames',
                        lambda src, times, dest, stem:
                        [(lbl, t, f'{dest}/{stem}_{lbl}.jpg') for lbl, t in times])
    order = {'n': 0}

    def encode(paths, **kw):
        # Later shots get vectors closer to the query, so the ranking is NOT the
        # database order — a route that lost it would be visible.
        out = []
        for _ in paths:
            order['n'] += 1
            out.append(_unit(order['n'], 1.0))
        return out
    monkeypatch.setattr(vcs, '_encode_frame_files', encode)
    monkeypatch.setattr(svc, '_embed_available', lambda: None)


# --- the pass ------------------------------------------------------------------

def test_starting_the_embedding_pass_answers_202(client, tmp_path, seams,
                                                 embed_seams):
    bank_id = _bank(client, tmp_path)

    r = client.post(f'/api/video-bank/{bank_id}/embed', json={})

    assert r.status_code == 202


def test_an_install_that_cannot_run_clip_is_refused_with_503(client, tmp_path,
                                                             seams, monkeypatch):
    """Not a 400: nothing the user types here can fix a missing torch, and the
    sentence they need is about Setup, not about this bank."""
    bank_id = _bank(client, tmp_path)
    monkeypatch.setattr(svc, '_embed_available',
                        lambda: 'frame embedding needs torch + open_clip')

    r = client.post(f'/api/video-bank/{bank_id}/embed', json={})

    assert r.status_code == 503
    assert 'open_clip' in r.get_json()['error']


def test_the_embedding_pass_on_an_unknown_bank_is_a_404(client):
    assert client.post('/api/video-bank/999999/embed', json={}).status_code == 404


def test_the_workspace_counts_how_many_shots_are_searchable(client, tmp_path,
                                                            seams, embed_seams, app):
    """The counter the next-step line reads to offer the pass — and the one that
    makes "2 of 40 shots searchable" honest after a stopped run."""
    bank_id = _bank(client, tmp_path)
    before = client.get(f'/api/video-bank/{bank_id}').get_json()['counts']
    assert before['embedded'] == 0

    with app.app_context():
        vcs.run_embed(bank_id)

    after = client.get(f'/api/video-bank/{bank_id}').get_json()['counts']
    assert after['embedded'] == 2


# --- the search ----------------------------------------------------------------

def test_a_search_returns_the_ranked_rows_in_ranked_order(client, tmp_path, seams,
                                                          embed_seams, monkeypatch,
                                                          app):
    bank_id = _bank(client, tmp_path)
    with app.app_context():
        vcs.run_embed(bank_id)
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    r = client.get(f'/api/video-bank/{bank_id}/search?q=a+red+car')

    body = r.get_json()
    assert r.status_code == 200
    assert len(body['results']) == 2
    assert [c['id'] for c in body['clips']] == body['clip_ids']
    # Descending score: the ranking survived the trip.
    assert body['results'][0]['score'] >= body['results'][1]['score']
    assert body['results'][0]['frame_s'] is not None


def test_searching_a_bank_with_no_embeddings_is_a_400_that_says_what_to_run(
        client, tmp_path, seams, monkeypatch):
    bank_id = _bank(client, tmp_path)
    monkeypatch.setattr(clip_text_encoder, 'encode_query',
                        lambda text: (_unit(1, 0), True))

    r = client.get(f'/api/video-bank/{bank_id}/search?q=a+red+car')

    assert r.status_code == 400
    assert 'Search' in r.get_json()['error']


def test_a_search_on_an_install_that_cannot_encode_the_phrase_is_a_503(
        client, tmp_path, seams, embed_seams, monkeypatch, app):
    """The feature is unavailable, not the answer — and the UI says those two
    differently."""
    bank_id = _bank(client, tmp_path)
    with app.app_context():
        vcs.run_embed(bank_id)

    def boom(text):
        raise clip_text_encoder.TextEncodeError('no interpreter can run CLIP here')
    monkeypatch.setattr(clip_text_encoder, 'encode_query', boom)

    r = client.get(f'/api/video-bank/{bank_id}/search?q=a+red+car')

    assert r.status_code == 503
    assert r.get_json()['reason'] == 'encoder_unavailable'


def test_a_search_with_no_query_is_a_400(client, tmp_path, seams):
    bank_id = _bank(client, tmp_path)

    assert client.get(f'/api/video-bank/{bank_id}/search?q=').status_code == 400


def test_searching_an_unknown_bank_is_a_404(client):
    assert client.get('/api/video-bank/999999/search?q=car').status_code == 404
