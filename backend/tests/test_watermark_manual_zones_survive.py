"""The zones you draw by hand must survive a clean, so ↩ Restore can offer them
back.

WHY THIS EXISTS. 🚩 Mark a watermark is the answer to a detector MISS — the
zones drawn there are the user's own correction, made precisely because the
automatic box was absent or wrong. A successful clean used to delete them.
Nothing said so, and the loss only showed up one step later: ↩ Restore original
puts the watermarked pixels back and re-flags the image 'detected' so it can be
cleaned again — often with the other engine, which is the button's whole reason
to exist — but the zones the retry needed were already gone, so it silently fell
back to the detected bbox. `restore_watermark_original`'s own docstring promised
the opposite ("the user may want to re-clean the same zones").

It is also the divergence CLAUDE.md warns about: the Bank runs the same feature
and has never nulled this field. Two surfaces, one contract.
"""
import json

from PIL import Image

from test_watermarks import _kept_image

ZONES = [[0.1, 0.1, 0.3, 0.2]]


def _klein_that_succeeds(monkeypatch, calls):
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: True)

    def _fake(user_id, path, boxes, **kwargs):
        calls.append(boxes)
        with Image.open(path) as im:
            im.convert('RGB').save(path)
        return True, None
    monkeypatch.setattr(wk, 'inpaint_watermark_klein', _fake)


def _zones(img):
    raw = img.watermark_regions
    return json.loads(raw) if isinstance(raw, str) else raw


def test_a_clean_does_not_delete_the_zones_you_drew(app, monkeypatch):
    from app.services import face_dataset_service as fds
    calls = []
    _klein_that_succeeds(monkeypatch, calls)

    with app.app_context():
        ds = fds.create_dataset('local', 'Zones', 'z')
        img = _kept_image(fds, ds.id, 'a.webp', bbox=ZONES[0], regions=ZONES)

        fds.clean_watermarks('local', ds.id, [img.id], method='klein')
        assert img.watermark_state == 'cleaned'
        assert _zones(img) == ZONES, 'the clean threw away the hand-drawn zones'


def test_restore_hands_the_zones_back_so_a_retry_uses_them(app, monkeypatch):
    """The step that made the loss visible, and the one the docstring promises."""
    from app.services import face_dataset_service as fds
    calls = []
    _klein_that_succeeds(monkeypatch, calls)

    with app.app_context():
        ds = fds.create_dataset('local', 'Zones', 'z')
        img = _kept_image(fds, ds.id, 'a.webp', bbox=ZONES[0], regions=ZONES)

        fds.clean_watermarks('local', ds.id, [img.id], method='klein')
        payload = fds.restore_watermark_original('local', ds.id, img.id)

        assert img.watermark_state == 'detected'
        assert _zones(img) == ZONES
        # …and the screen that draws them is told, rather than having to guess.
        assert payload['watermark_regions'] == ZONES
        assert payload['effective_watermark_regions'] == ZONES


def test_the_restored_image_is_the_original_byte_for_byte(app, monkeypatch):
    """The other half of the promise, and the reason a retry is worth offering."""
    import hashlib
    from app.services import face_dataset_service as fds
    _klein_that_succeeds(monkeypatch, [])

    with app.app_context():
        ds = fds.create_dataset('local', 'Zones', 'z')
        img = _kept_image(fds, ds.id, 'a.webp', bbox=ZONES[0], regions=ZONES)
        path = fds._img_path(img)
        digest = lambda: hashlib.sha256(open(path, 'rb').read()).hexdigest()
        before = digest()

        fds.clean_watermarks('local', ds.id, [img.id], method='klein')
        assert digest() != before, 'the clean did nothing at all'
        fds.restore_watermark_original('local', ds.id, img.id)
        assert digest() == before


def test_dismissing_still_clears_them(app, monkeypatch):
    """The wipe is right HERE and must stay: '✓ Not a watermark' says the zones
    described nothing, so keeping them would contradict the verdict."""
    from app.services import face_dataset_service as fds

    with app.app_context():
        ds = fds.create_dataset('local', 'Zones', 'z')
        img = _kept_image(fds, ds.id, 'a.webp', bbox=ZONES[0], regions=ZONES)

        fds.dismiss_watermarks('local', ds.id, [img.id])
        assert img.watermark_state == 'dismissed'
        assert _zones(img) is None
