"""A job the machine cancels because ComfyUI restarted must not read as a
ComfyUI error.

Found on a real clip (2026-08-31): a video was rendering when the app was
updated; ComfyUI came back with no memory of the prompt, the queue correctly
noticed it was gone and settled the card — and the card said "Generation failed
(see the server log in Settings for the ComfyUI error)". There was no ComfyUI
error to find. The graph was fine, the render simply disappeared under it.

The distinction matters because the two failures want opposite reactions: a
graph error means "change something before relaunching", a lost prompt means
"launch exactly the same thing again".
"""
import json

import pytest


def _cancelled_job(app, *, metadata, error_message=None):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    job = ImageGenerationQueue(job_id='job-restart', user_id='1',
                               status='cancelled',
                               error_message=error_message,
                               job_metadata=json.dumps(metadata))
    db.session.add(job)
    db.session.commit()
    return job


def test_an_auto_resolved_cancellation_says_the_restart_not_a_comfy_error(app, monkeypatch):
    from app import job_queue
    from app.extensions import db
    from app.models import ImageGenerationQueue, VideoTestClip
    with app.app_context():
        _cancelled_job(app, metadata={'is_video_test': True, 'clip_id': 1})
        clip = VideoTestClip(job_id='job-restart', status='pending', prompt='p',
                             mode='t2v')
        db.session.add(clip)
        db.session.commit()
        cid = clip.id

        job_queue._dispatch_auto_resolved_cancellation('job-restart')

        row = db.session.get(VideoTestClip, cid)
        assert row.status == 'failed'
        assert 'restarted' in row.error
        # The sentence the user must NOT be given here: there is no such error.
        assert 'ComfyUI error' not in row.error
        # And it says what to do, because relaunching is the right move.
        assert 'again' in row.error
        # The reason is written on the JOB, so every other lane's card gets it
        # too without knowing this path exists.
        job = ImageGenerationQueue.query.filter_by(job_id='job-restart').first()
        assert 'restarted' in (job.error_message or '')


def test_a_real_comfyui_error_is_never_overwritten(app):
    """The recovery path only fills a SILENCE. A node error, a validation body —
    anything the queue already captured — is what the user needs, and it wins."""
    from app import job_queue
    from app.extensions import db
    from app.models import VideoTestClip
    with app.app_context():
        _cancelled_job(app, metadata={'is_video_test': True, 'clip_id': 1},
                       error_message='Value not in list: unet_name')
        clip = VideoTestClip(job_id='job-restart', status='pending', prompt='p',
                             mode='t2v')
        db.session.add(clip)
        db.session.commit()
        cid = clip.id

        job_queue._dispatch_auto_resolved_cancellation('job-restart')
        assert 'unet_name' in db.session.get(VideoTestClip, cid).error


@pytest.mark.parametrize('metadata', [
    {'is_lora_test': True, 'dataset_id': 1},
    {'is_video_test': True, 'clip_id': 1},
])
def test_the_sentence_is_written_once_for_every_lane(app, metadata):
    """It is set on the job, not on one lane's row, precisely so the image grid
    and the video clips cannot end up telling two different stories about the
    same event."""
    from app import job_queue
    from app.models import ImageGenerationQueue
    with app.app_context():
        _cancelled_job(app, metadata=metadata)
        job_queue._dispatch_auto_resolved_cancellation('job-restart')
        job = ImageGenerationQueue.query.filter_by(job_id='job-restart').first()
        assert 'restarted' in (job.error_message or '')
