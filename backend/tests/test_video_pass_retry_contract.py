"""No shot is retired for good by one bad pass — in ANY of the video passes.

THE DEAD END THIS PINS SHUT. Every pass in this lane writes 'unreadable' about
a shot it could not handle, and then skips it forever: `pending_clips` stops
offering it and the button above says "nothing to do". That is the right
behaviour for a genuinely broken file and a trap for everything else, because
the reasons a pass fails are mostly NOT about the file — a decoder that stopped
loading, a bank folder that moved, an ML interpreter an unrelated
`pip install --user` broke.

It happened: a bank of 861 shots was retired in one pass by a stale package in
a user site-packages directory. 🔎 Find scenes then answered "0 shots to embed"
while ✂ Duplicates answered "run Find scenes first" — a closed loop, with no way
out of the app.

The rule, and the reason it is one line and not a button: the pass offers the
failed shots again ONLY when it has nothing else to do. A bank that grew never
reaches the second tier, so the normal path pays nothing, and the recovery rides
the button the user was going to press anyway.

This file reads all five passes so a sixth cannot be written without it.
"""
import json

import pytest

from app.extensions import db
from app.models import VideoBank, VideoClip, VideoSource


# (module, how one shot is marked failed, how one shot is marked done)
PASSES = [
    ('video_clip_search', 'embed_state', 'ok'),
    ('video_ai_check', 'ai_check_state', 'ok'),
    ('video_camera_motion', 'camera_state', 'ok'),
    ('video_safe_zone', 'safe_zone_state', 'ok'),
    ('video_watermark', 'watermark_state', 'ok'),
]


def _module(name):
    import importlib
    return importlib.import_module(f'app.services.{name}')


def _bank(app, n=3):
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=90.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        for i in range(n):
            db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                     start_s=float(i * 10), end_s=float(i * 10 + 5)))
        db.session.commit()
        return bank.id


def _mark(app, bank_id, key, value, limit=None):
    """Write one pass's verdict — a column for the embedding pass, a blob key for
    the advisory ones, which is exactly the split the passes themselves make."""
    with app.app_context():
        rows = VideoClip.query.filter_by(bank_id=bank_id).order_by(VideoClip.id)
        for clip in rows.all()[:limit]:
            if key == 'embed_state':
                clip.embed_state = value
            else:
                blob = json.loads(clip.metrics_json or '{}')
                blob[key] = value
                clip.metrics_json = json.dumps(blob)
        db.session.commit()


@pytest.mark.parametrize('name, key, done', PASSES)
def test_a_pass_offers_its_failed_shots_again_when_it_has_nothing_else_to_do(
        app, name, key, done):
    """The way out of the dead end, and the only one a user gets: the pass's own
    button. Nothing else in the app can un-retire a shot."""
    bank_id = _bank(app, 3)
    _mark(app, bank_id, key, 'unreadable')

    with app.app_context():
        assert len(list(_module(name).pending_clips(bank_id))) == 3


@pytest.mark.parametrize('name, key, done', PASSES)
def test_a_pass_with_real_work_left_does_not_redo_its_failures(app, name, key, done):
    """The retry is the SECOND tier and must stay there: a bank that grew must
    not pay to re-decode its known-bad files before touching the new shots."""
    bank_id = _bank(app, 3)
    _mark(app, bank_id, key, 'unreadable', limit=2)

    with app.app_context():
        pending = list(_module(name).pending_clips(bank_id))

    assert len(pending) == 1, 'only the shot nobody has looked at yet'


@pytest.mark.parametrize('name, key, done', PASSES)
def test_a_shot_a_pass_finished_is_never_offered_again(app, name, key, done):
    """The resume contract the retry must not break. A finished verdict is a
    measurement, not a guess, and re-running it would cost the whole bank."""
    bank_id = _bank(app, 3)
    _mark(app, bank_id, key, done)

    with app.app_context():
        assert len(list(_module(name).pending_clips(bank_id))) == 0
