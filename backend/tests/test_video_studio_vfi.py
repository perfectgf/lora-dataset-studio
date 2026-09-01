"""↗ VFI — the maintainer's own interpolation recipe, in the video studio.

The graph is read out of his image generator (workflows/video-generation/
vfi.json) rather than invented: both apps drive the same ComfyUI, and a clip
smoothed in one must be the clip smoothed in the other. What this file pins is
that recipe, and the two rules around it — a NEW row rather than an edit, and a
refusal when the nodes are absent.
"""
import pytest

from app.services import video_test_studio as vts


def test_the_graph_is_the_generators_own_recipe():
    wf = vts.build_vfi_workflow(video_path='C:/clips/a.mp4', fps=24)
    kinds = {n['class_type'] for n in wf.values()}
    assert kinds == {'VHS_LoadVideoPath', 'RIFE VFI', 'VHS_VideoCombine'}

    load = next(n for n in wf.values() if n['class_type'] == 'VHS_LoadVideoPath')
    rife = next(n for n in wf.values() if n['class_type'] == 'RIFE VFI')
    save = next(n for n in wf.values() if n['class_type'] == 'VHS_VideoCombine')

    # An absolute path, because the clip lives in this app's folder and never in
    # ComfyUI's output — which is exactly why VHS_LoadVideoPath is the loader.
    assert load['inputs']['video'] == 'C:/clips/a.mp4'
    # The interpolator's dials, verbatim from vfi.json.
    assert rife['inputs']['ckpt_name'] == 'rife49.pth'
    assert rife['inputs']['multiplier'] == 2
    assert rife['inputs']['fast_mode'] is True
    assert rife['inputs']['ensemble'] is True
    assert rife['inputs']['scale_factor'] == 2
    assert rife['inputs']['clear_cache_after_n_frames'] == 16
    # Twice the rate: the clip keeps its DURATION and gains frames. A graph that
    # wrote the source rate here would deliver slow motion instead.
    assert save['inputs']['frame_rate'] == 48
    assert save['inputs']['format'] == 'video/h264-mp4'
    assert save['inputs']['crf'] == 19
    # Wired end to end, not three loose nodes.
    assert rife['inputs']['frames'][0] in wf
    assert save['inputs']['images'][0] in wf


def test_the_rate_follows_the_source_and_the_multiplier():
    assert vts.build_vfi_workflow(video_path='a.mp4', fps=16)['v3'][
        'inputs']['frame_rate'] == 32
    assert vts.build_vfi_workflow(video_path='a.mp4', fps=24, multiplier=4)['v3'][
        'inputs']['frame_rate'] == 96


def _clip(app, **kw):
    from app.extensions import db
    from app.models import VideoTestClip
    with app.app_context():
        row = VideoTestClip(**{'status': 'done', 'filename': 'clip.mp4',
                               'mode': 'i2v', 'fps': 24, 'frames': 56,
                               'prompt': 'she turns', **kw})
        db.session.add(row)
        db.session.commit()
        return row.id


def test_smoothing_makes_a_NEW_clip_and_leaves_the_original_alone(app, tmp_path, monkeypatch):
    """The studio exists to compare; overwriting the thing being compared would
    end that. The new row carries the source's settings, twice its rate, and a
    pointer back."""
    from app.extensions import db
    from app.models import VideoTestClip
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'\x00' * 8)
    monkeypatch.setattr(vts, 'registered_classes',
                        lambda: {'RIFE VFI', 'VHS_LoadVideoPath', 'VHS_VideoCombine'})
    queued = {}
    monkeypatch.setattr('app.job_queue.queue_manager.add_job',
                        lambda **kw: queued.update(kw))
    src_id = _clip(app, seed=7, turbo=True, lora='h3/x.safetensors', lora_strength=0.8)
    with app.app_context():
        out = vts.interpolate_clip('local', src_id)
        new = db.session.get(VideoTestClip, out['clip_id'])
        src = db.session.get(VideoTestClip, src_id)
        assert new.id != src.id
        assert src.status == 'done' and src.fps == 24        # untouched
        assert new.fps == 48 and new.frames == 112
        assert new.vfi_of == src.id
        # The settings ride along so the card still says what made it.
        assert new.seed == 7 and new.turbo is True
        assert new.lora == 'h3/x.safetensors' and new.lora_strength == 0.8
    assert queued['metadata']['is_video_test'] is True


def test_it_refuses_rather_than_queueing_a_job_this_comfyui_cannot_run(
        app, tmp_path, monkeypatch):
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    (tmp_path / 'clip.mp4').write_bytes(b'\x00' * 8)
    monkeypatch.setattr(vts, 'registered_classes', lambda: {'VHS_LoadVideoPath'})
    src_id = _clip(app)
    with app.app_context():
        with pytest.raises(ValueError, match='RIFE VFI'):
            vts.interpolate_clip('local', src_id)


def test_an_unfinished_or_missing_clip_says_so(app, tmp_path, monkeypatch):
    monkeypatch.setattr(vts, 'clips_dir', lambda create=True: str(tmp_path))
    monkeypatch.setattr(vts, 'registered_classes', lambda: None)   # probe failed
    pending = _clip(app, status='pending', filename=None)
    with app.app_context():
        with pytest.raises(ValueError, match='finished'):
            vts.interpolate_clip('local', pending)
        with pytest.raises(ValueError, match='not found'):
            vts.interpolate_clip('local', 999999)
    gone = _clip(app, filename='vanished.mp4')
    with app.app_context():
        with pytest.raises(ValueError, match='disk'):
            vts.interpolate_clip('local', gone)
