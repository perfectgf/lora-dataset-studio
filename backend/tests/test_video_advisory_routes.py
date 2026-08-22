"""The two advisory passes, as endpoints — the refusals, mostly.

A pass that answers 202 and then does nothing is worse than one that refuses:
the user watches a progress bar reach the end and reads the empty result as an
answer. So what is pinned here is which refusal each pass gives, and that the two
503s of the watermark pass stay two different sentences.
"""

from app.services import video_bank_service as svc


def test_dedup_refuses_a_bank_that_was_never_embedded(client, app):
    """400 with the pass to run named. Zero groups over zero vectors is not an
    answer about duplicates."""
    bank_id = _bank(app)
    r = client.post(f'/api/video-bank/{bank_id}/dedup', json={})
    assert r.status_code == 400
    assert 'Find scenes' in r.get_json()['error']


def test_dedup_starts_once_vectors_exist(client, app, monkeypatch):
    bank_id = _bank(app)
    monkeypatch.setattr('app.services.video_clip_search.load_embeddings',
                        lambda bid: {1: [{'vec': [1.0, 0.0]}]})
    monkeypatch.setattr(svc.bank_jobs, 'start', lambda *a, **k: {'kind': 'dedup'})
    r = client.post(f'/api/video-bank/{bank_id}/dedup', json={})
    assert r.status_code == 202


def test_dedup_on_an_unknown_bank_is_404_not_400(client):
    """Not something the user can fix by editing the body — the bank was deleted
    in another tab."""
    assert client.post('/api/video-bank/424242/dedup', json={}).status_code == 404


def test_watermark_refuses_without_the_decode_extra(client, app, monkeypatch):
    bank_id = _bank(app)
    monkeypatch.setattr('app.capabilities.probe_video',
                        lambda: {'ok': False, 'decode': False, 'detect': False,
                                 'encode': False, 'detail': 'install the video extra'})
    r = client.post(f'/api/video-bank/{bank_id}/watermark', json={})
    assert r.status_code == 503
    assert 'video extra' in r.get_json()['error']


def test_watermark_refuses_without_the_detector_in_its_own_words(client, app, monkeypatch):
    """The SECOND 503, and it must not be the first one's sentence: one is fixed
    by installing the video extra, the other by downloading the detector."""
    bank_id = _bank(app)
    monkeypatch.setattr('app.capabilities.probe_video',
                        lambda: {'ok': True, 'decode': True, 'detect': True,
                                 'encode': True, 'detail': ''})
    monkeypatch.setattr('app.capabilities.probe_watermark_detect',
                        lambda: {'ok': False, 'detail': 'the detector weights are '
                                                        'not downloaded yet'})
    r = client.post(f'/api/video-bank/{bank_id}/watermark', json={})
    assert r.status_code == 503
    assert 'weights' in r.get_json()['error']


def test_both_passes_are_refused_while_another_owns_the_bank(client, app, monkeypatch):
    """409 with `busy_kind`, the same envelope as every other pass — so the UI
    names what to wait for instead of repeating "busy"."""
    bank_id = _bank(app)
    monkeypatch.setattr('app.services.video_clip_search.load_embeddings',
                        lambda bid: {1: [{'vec': [1.0, 0.0]}]})

    def busy(*a, **k):
        raise svc.bank_jobs.BankJobBusy('detect')
    monkeypatch.setattr(svc.bank_jobs, 'start', busy)
    r = client.post(f'/api/video-bank/{bank_id}/dedup', json={})
    assert r.status_code == 409
    assert r.get_json()['busy_kind'] == 'detect'


def _bank(app):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import VideoBank
    with app.app_context():
        bank = VideoBank(user_id=LOCAL_USER, name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.commit()
        return bank.id
