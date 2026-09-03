"""✨ Neural render in the Video Test Studio — a NEW clip, never an edit.

Same rule as ↗ Smooth (test_video_studio_vfi.py): the studio exists to compare,
so the render is its own row, pointing back at its source through `nr_of`, and
the source keeps its file and its status. The child is replaced by a stand-in;
the thread that flips the row is run to completion before asserting.
"""
import os

import pytest

from app.services import neural_render as nr
from app.services import video_test_studio as vts


def _clip(app, **kw):
    from app.extensions import db
    from app.models import VideoTestClip
    with app.app_context():
        row = VideoTestClip(**{'status': 'done', 'filename': 'clip.mp4', 'mode': 'i2v',
                               'fps': 24, 'frames': 56, 'prompt': 'she turns',
                               'lora': 'x.safetensors', 'seed': 7, **kw})
        db.session.add(row)
        db.session.commit()
        return row.id


def _ready(monkeypatch):
    monkeypatch.setattr(nr, 'status', lambda root=None, os_name=None, driver=None: {
        'ready': True, 'missing': [], 'driver_nvof': True})


def _join_thread(src_id):
    thread = nr._STUDIO_THREADS.get(src_id)
    if thread is not None:
        thread.join(timeout=10)


def test_the_render_is_a_new_row_pointing_at_its_source(app, tmp_path, monkeypatch):
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'ORIGINAL')
    _ready(monkeypatch)
    seen = {}

    def fake(src, dst, params, **kw):
        seen['src'] = src
        with open(dst, 'wb') as fh:
            fh.write(b'RENDERED')
        return {'frames': 56, 'mode_note': 'still mode', 'mean_ms': 12.0}
    monkeypatch.setattr(nr, 'render_video', fake)
    src_id = _clip(app)
    with app.app_context():
        out = nr.start_studio_render(app, 'local', src_id, {'tone': 0.5, 'temporal': 'off'})
        assert out['params']['tone'] == 0.5
        new_id = out['clip_id']
        assert new_id != src_id
    _join_thread(src_id)
    with app.app_context():
        new = VideoTestClip.query.get(new_id)
        src = VideoTestClip.query.get(src_id)
        assert new.nr_of == src_id
        assert new.status == 'done' and new.filename and new.filename != 'clip.mp4'
        assert (tmp_path / new.filename).read_bytes() == b'RENDERED'
        # The source is untouched: same file, same bytes, still done.
        assert src.status == 'done' and src.filename == 'clip.mp4'
        assert (tmp_path / 'clip.mp4').read_bytes() == b'ORIGINAL'
        assert seen['src'] == os.path.join(str(tmp_path), 'clip.mp4')
        # The settings travel so the card still says what made the source.
        assert new.lora == 'x.safetensors' and new.seed == 7 and new.prompt == 'she turns'


def test_a_failed_render_lands_as_failed_with_the_childs_sentence(app, tmp_path, monkeypatch):
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'ORIGINAL')
    _ready(monkeypatch)

    def boom(src, dst, params, **kw):
        raise nr.NeuralRenderError('the model refused the frame')
    monkeypatch.setattr(nr, 'render_video', boom)
    src_id = _clip(app)
    with app.app_context():
        new_id = nr.start_studio_render(app, 'local', src_id, {})['clip_id']
    _join_thread(src_id)
    with app.app_context():
        new = VideoTestClip.query.get(new_id)
        assert new.status == 'failed' and 'refused the frame' in new.error
        assert new.filename is None


def test_refusals_are_sentences(app, tmp_path, monkeypatch):
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    _ready(monkeypatch)
    with app.app_context():
        with pytest.raises(nr.NeuralRenderError, match='not found'):
            nr.start_studio_render(app, 'local', 999, {})
        pending = _clip(app, status='pending', filename=None)
        with pytest.raises(nr.NeuralRenderError, match='not finished'):
            nr.start_studio_render(app, 'local', pending, {})
        gone = _clip(app, filename='gone.mp4')
        with pytest.raises(nr.NeuralRenderError, match='no longer on disk'):
            nr.start_studio_render(app, 'local', gone, {})
    monkeypatch.setattr(nr, 'status', lambda root=None, os_name=None, driver=None: {
        'ready': False, 'missing': ['Windows — x'], 'driver_nvof': False})
    with app.app_context():
        with pytest.raises(nr.NeuralRenderError, match='Windows'):
            nr.start_studio_render(app, 'local', _clip(app), {})


def test_the_route_and_the_clip_payload(app, client, tmp_path, monkeypatch):
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'ORIGINAL')
    _ready(monkeypatch)
    monkeypatch.setattr(nr, 'render_video', lambda src, dst, params, **kw: (
        open(dst, 'wb').write(b'R') and {'frames': 1, 'mode_note': 'still mode'}))
    src_id = _clip(app)
    r = client.post(f'/api/video-studio/clip/{src_id}/neural-render', json={'tone': 3})
    assert r.status_code == 400 and 'tone' in r.get_json()['error']
    r = client.post(f'/api/video-studio/clip/{src_id}/neural-render', json={'tone': 0})
    assert r.status_code == 200
    new_id = r.get_json()['clip_id']
    _join_thread(src_id)
    r = client.get('/api/video-studio/clips')
    rows = {c['id']: c for c in r.get_json()['clips']}
    assert rows[new_id]['nr_of'] == src_id
    assert rows[src_id]['nr_of'] is None
    r = client.post('/api/video-studio/clip/999/neural-render', json={})
    assert r.status_code == 400


def test_the_column_is_migrated_for_legacy_databases():
    """A database created before this wave has no `nr_of`; the additive
    migration list must carry it or every studio read dies on a legacy file."""
    from app import __init__ as pkg  # noqa: F401 — the list lives in the package init
    import app as app_pkg
    src = open(os.path.join(os.path.dirname(app_pkg.__file__), '__init__.py'), encoding='utf-8').read()
    assert "('video_test_clip', 'nr_of', 'INTEGER')" in src


def test_the_history_page_carries_the_source_of_its_renders_and_pages_back(app, client, tmp_path, monkeypatch):
    """A render's source is older than the render by construction, so it falls
    off the newest page after a few renders — and a pair that cannot be seen
    together reads as "the original was deleted" (reported the first evening).
    The page appends every listed render's source, whatever its age, and
    `before` pages further back."""
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    ids = [_clip(app, prompt=f'clip {i}') for i in range(30)]       # 30 plain clips
    source = ids[0]                                                  # the oldest one
    render = _clip(app, prompt='render', nr_of=source)               # newest, points at the oldest
    r = client.get('/api/video-studio/clips?limit=5')
    body = r.get_json()
    listed = [c['id'] for c in body['clips']]
    assert listed[0] == render and listed == sorted(listed, reverse=True)
    assert source in listed, 'the source rides along even though it is 30 clips older'
    assert len(listed) == 6 and body['has_more'] is True
    # Paging back from the oldest id of the page proper (not the carried source).
    page_proper = [i for i in listed if i != source]
    assert body['oldest_id'] == min(page_proper), 'the boundary is the page proper, never the carried source'
    r2 = client.get(f'/api/video-studio/clips?limit=5&before={min(page_proper)}')
    older = [c['id'] for c in r2.get_json()['clips']]
    assert older and max(older) < min(page_proper)
    # The last page says so.
    r3 = client.get(f'/api/video-studio/clips?limit=100&before={ids[3]}')
    assert r3.get_json()['has_more'] is False


def test_a_render_remembers_its_dials_and_the_mode_it_used(app, client, tmp_path, monkeypatch):
    """The card can say seed and steps; for a render the dials are what
    differed, so they travel with the row — as asked, then as used."""
    import json
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'ORIGINAL')
    _ready(monkeypatch)
    monkeypatch.setattr(nr, 'render_video', lambda src, dst, params, **kw: (
        open(dst, 'wb').write(b'R') and {'frames': 56, 'temporal': True, 'mean_ms': 31.7, 'mode_note': 'temporal mode'}))
    src_id = _clip(app)
    with app.app_context():
        new_id = nr.start_studio_render(app, 'local', src_id, {'strength': 2, 'passes': 1, 'scale': 2, 'tone': 0})['clip_id']
        asked = json.loads(VideoTestClip.query.get(new_id).nr_params)
        assert asked['strength'] == 2.0 and asked['scale'] == 2 and asked['tone'] == 0.0
        assert 'temporal_used' not in asked
    _join_thread(src_id)
    with app.app_context():
        used = json.loads(VideoTestClip.query.get(new_id).nr_params)
        assert used['temporal_used'] is True and used['ms_per_frame'] == 31.7 and used['frames'] == 56
    r = client.get('/api/video-studio/clips')
    row = next(c for c in r.get_json()['clips'] if c['id'] == new_id)
    assert row['nr_params']['strength'] == 2.0 and row['nr_params']['temporal_used'] is True
    src_row = next(c for c in r.get_json()['clips'] if c['id'] == src_id)
    assert src_row['nr_params'] is None


def test_a_neural_render_measures_its_own_time_done_or_failed(app, tmp_path, monkeypatch):
    """This lane never goes through the queue, so nothing stamps it: the thread
    times itself, on both outcomes, and the card can say how long the pass took."""
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'ORIGINAL')
    _ready(monkeypatch)

    def slow_ok(src, dst, params, **kw):
        import time
        time.sleep(0.05)
        with open(dst, 'wb') as fh:
            fh.write(b'RENDERED')
        return {'frames': 56, 'mode_note': 'still mode', 'mean_ms': 12.0}
    monkeypatch.setattr(nr, 'render_video', slow_ok)
    src_id = _clip(app)
    with app.app_context():
        new_id = nr.start_studio_render(app, 'local', src_id, {})['clip_id']
    _join_thread(src_id)
    with app.app_context():
        new = VideoTestClip.query.get(new_id)
        assert new.status == 'done'
        assert isinstance(new.render_seconds, float) and new.render_seconds >= 0.0

    def boom(src, dst, params, **kw):
        raise nr.NeuralRenderError('the bridge refused the clip')
    monkeypatch.setattr(nr, 'render_video', boom)
    src2 = _clip(app)
    with app.app_context():
        failed_id = nr.start_studio_render(app, 'local', src2, {})['clip_id']
    _join_thread(src2)
    with app.app_context():
        row = VideoTestClip.query.get(failed_id)
        assert row.status == 'failed' and isinstance(row.render_seconds, float)
