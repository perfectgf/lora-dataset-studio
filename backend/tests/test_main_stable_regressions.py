"""Stable-lane regressions: existing video controls and training pod ownership."""
from pathlib import Path
from types import SimpleNamespace
import os

import pytest


@pytest.fixture
def queue_capture(app, monkeypatch):
    from app import capabilities
    from app.job_queue import queue_manager
    from app.services import video_test_studio as vts
    monkeypatch.setattr(capabilities, 'probe_comfyui',
                        lambda: {'ok': True, 'status': 'ok', 'detail': '', 'hint': ''})
    monkeypatch.setattr(vts, 'preflight', lambda wf: None)
    monkeypatch.setattr(vts, 'registered_classes', lambda: None)
    captured = {}
    monkeypatch.setattr(queue_manager, 'add_job', lambda **kw: captured.update(kw))
    return captured


def test_training_reconcile_spares_other_lanes_and_keeps_active_training(app, monkeypatch):
    from app.services import cloud_training as ct
    monkeypatch.setenv('VAST_API_KEY', 'test-key')
    labels = ['lds-123', 'lds-234', 'lds-quantize-abcd', 'lds-live-abcd',
              'lds-user-box', 'lds-123-extra', 'lds-123\n', 'lds-\u0661', 'unrelated']
    fleet = [{'instance_id': str(i), 'label': label} for i, label in enumerate(labels)]
    destroyed = []
    monkeypatch.setattr(ct, 'get_active_runs',
                        lambda: [SimpleNamespace(vast_instance_id='1')])
    monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: fleet)
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda ident: destroyed.append(ident) or True)
    assert ct.reconcile_orphans(app) == 1
    assert destroyed == ['0'], 'only an orphan carrying an exact training label may be destroyed'


@pytest.mark.parametrize('fails', [False, True])
def test_last_frame_is_published_only_after_successful_extraction(app, tmp_path, monkeypatch, fails):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import ffmpeg_tools, video_test_studio as vts
    clip_dir = tmp_path / 'clips'
    clip_dir.mkdir()
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: clip_dir)
    monkeypatch.setattr(ffmpeg_tools, 'ffmpeg_path', lambda: 'test-ffmpeg')
    source = clip_dir / 'source.mp4'
    source.write_bytes(b'video')
    with app.app_context():
        clip = VideoTestClip(status='done', filename=source.name, mode='t2v')
        db.session.add(clip)
        db.session.commit()
        dest = clip_dir / f'clip_{clip.id}_last.png'
        dest.write_bytes(b'previous complete frame')
        os.utime(dest, (1, 1))
        os.utime(source, (2, 2))

        def extract(command, **kwargs):
            target = Path(command[-1])
            target.write_bytes(b'partial frame')
            assert dest.read_bytes() == b'previous complete frame', 'partial PNG became visible'
            if not fails:
                target.write_bytes(b'new complete frame')
            return SimpleNamespace(returncode=int(fails), stderr='decode failed' if fails else '')

        monkeypatch.setattr(vts, '_run_ffmpeg', extract)
        if fails:
            with pytest.raises(ValueError, match='decode failed'):
                vts.last_frame_png(clip.id)
            assert dest.read_bytes() == b'previous complete frame'
        else:
            assert vts.last_frame_png(clip.id) == str(dest)
            assert dest.read_bytes() == b'new complete frame'
    assert sorted(p.name for p in clip_dir.iterdir()) == sorted([source.name, dest.name])


@pytest.mark.parametrize('aspect', ['portrait', 'landscape', 'square'])
def test_t2v_history_replays_its_original_canvas(client, queue_capture, aspect):
    body = {'mode': 't2v', 'prompt': 'A person turns.', 'aspect': aspect, 'seed': 42}
    first = client.post('/api/video-studio/generate', json=body)
    assert first.status_code == 200, first.get_json()
    canvas = queue_capture['workflow_data']['104']['inputs']
    original_size = (canvas['width'], canvas['height'])
    different = 'landscape' if aspect != 'landscape' else 'portrait'
    assert client.post('/api/video-studio/generate', json=dict(body, aspect=different)).status_code == 200
    clip = client.get(f'/api/video-studio/clip/{first.get_json()["clip_id"]}').get_json()
    assert clip.get('aspect') == aspect
    reused = client.post('/api/video-studio/generate', json={
        'mode': clip['mode'], 'prompt': clip['prompt'], 'seed': clip['seed'], 'aspect': clip['aspect']})
    assert reused.status_code == 200, reused.get_json()
    replay = queue_capture['workflow_data']['104']['inputs']
    assert (replay['width'], replay['height']) == original_size


@pytest.mark.parametrize('mode,aspect,expected', [
    ('t2v', ' PORTRAIT ', 'portrait'), ('t2v', 'unknown', 'auto'),
    ('i2v', 'landscape', 'auto'),
])
def test_recorded_aspect_matches_the_canvas_choice(client, queue_capture, mode, aspect, expected):
    response = client.post('/api/video-studio/generate', json={
        'mode': mode, 'prompt': 'A slow camera move.', 'image': 'first.png', 'aspect': aspect})
    assert response.status_code == 200, response.get_json()
    clip = client.get(f'/api/video-studio/clip/{response.get_json()["clip_id"]}').get_json()
    assert clip.get('aspect') == expected


@pytest.mark.parametrize('operation', ['vfi', 'neural-render'])
@pytest.mark.parametrize('accel', ['parasyte', 'dareties'])
def test_derived_clips_keep_existing_acceleration_and_canvas(
        app, client, monkeypatch, queue_capture, operation, accel):
    from app.extensions import db
    from app.models import VideoTestClip
    from app.services import neural_render as nr, video_test_studio as vts
    response = client.post('/api/video-studio/generate', json={
        'mode': 't2v', 'prompt': 'A person turns.', 'aspect': 'portrait', 'accel': accel})
    assert response.status_code == 200, response.get_json()
    ident = response.get_json()['clip_id']
    with app.app_context():
        source = db.session.get(VideoTestClip, ident)
        source.status = 'done'
        source.filename = 'source.mp4'
        (vts.clips_dir() / source.filename).write_bytes(b'test video')
        db.session.commit()
    monkeypatch.setattr(nr, 'status', lambda: {'ready': True, 'missing': []})

    def render(source, destination, params):
        Path(destination).write_bytes(b'test derived video')
        return {'frames': 56, 'mode_note': 'test'}

    monkeypatch.setattr(nr, 'render_video', render)
    try:
        derived = client.post(f'/api/video-studio/clip/{ident}/{operation}', json={})
        assert derived.status_code == 200, derived.get_json()
        if operation == 'neural-render':
            nr._STUDIO_THREADS[ident].join(timeout=5)
            assert not nr._STUDIO_THREADS[ident].is_alive()
        clip = client.get(f'/api/video-studio/clip/{derived.get_json()["clip_id"]}').get_json()
        assert clip['accel'] == accel
        assert clip.get('aspect') == 'portrait'
    finally:
        thread = nr._STUDIO_THREADS.pop(ident, None)
        if thread is not None:
            thread.join(timeout=5)


def test_legacy_aspect_migration_defaults_to_auto_and_can_repeat(app, client):
    from sqlalchemy import text
    from app import _apply_additive_migrations
    from app.extensions import db
    with app.app_context():
        # On the old schema there is nothing to drop; it is the migration's
        # job to add the column while preserving the existing row.
        columns = {row[1] for row in db.session.execute(text('PRAGMA table_info(video_test_clip)'))}
        if 'aspect' in columns:
            db.session.execute(text('ALTER TABLE video_test_clip DROP COLUMN aspect'))
        db.session.execute(text("INSERT INTO video_test_clip (id,status,mode,turbo,latent_upscale,rating) "
                                "VALUES (123,'done','t2v',0,0,0)"))
        db.session.commit()
        _apply_additive_migrations()
        _apply_additive_migrations()
        columns = {row[1] for row in db.session.execute(text('PRAGMA table_info(video_test_clip)'))}
        assert 'aspect' in columns
        assert db.session.execute(text('SELECT aspect FROM video_test_clip WHERE id=123')).scalar() == 'auto'
    assert client.get('/api/video-studio/clip/123').get_json()['aspect'] == 'auto'
