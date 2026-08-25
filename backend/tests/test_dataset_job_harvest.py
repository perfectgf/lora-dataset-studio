"""The finished image must come BACK, and an engine that forgets to say so must
fail a test rather than a user's batch.

WHAT HAPPENED (reported by the maintainer from a real run, 2026-08-02)
---------------------------------------------------------------
SeedVR2 rendered perfectly. ComfyUI reported `execution_success`, the queue row
went `completed` with `result_filename` set, and the 2.2 MB PNG sat in ComfyUI's
output folder. The dataset candidate row stayed `status='pending'` with
`filename` NULL, so the workspace showed nothing at all and the log was clean —
because nothing had failed.

Cause: `_dispatch_completion` routes a dataset job by testing
`model_name in DATASET_IMAGE_JOB_NAMES`, and `seedvr2_upscale` was not in that
set. The `elif` chain simply fell through. No branch, no exception, no log.

THIS IS THE SECOND TIME. Krea 2 Edit shipped with exactly this bug (twelve
images generated and stranded, panel stuck at 0/12). The guard written then —
`test_the_declared_set_matches_what_the_engines_actually_stamp` — promises in its
own docstring that "a third engine added to the enqueue side without being added
here fails here". It does not: it iterates a HARDCODED tuple of two helper
modules, so a third helper module is invisible to it. It could only ever catch a
name declared in the set but stamped by nobody — the opposite direction from the
one that breaks twice.

So this file guards the direction that actually fails: it DISCOVERS the engines
instead of listing them.
"""
import ast
import json
import pathlib


SERVICES = pathlib.Path(__file__).resolve().parents[1] / 'app' / 'services'

# Job names a service stamps that are deliberately NOT dataset-row jobs. Each one
# must say how its result gets home instead, because "it is not in the set" is
# exactly the sentence that shipped this bug twice.
NON_DATASET_JOB_NAMES = {
    # The Test Studio grid routes on `is_lora_test`, checked before model_name.
    'zimage_lora_test',
    # The watermark cleaner POLLS its own queue row for status/result_filename
    # (watermark_klein._await_job); it never wants a dispatch callback.
    'watermark_klein',
    # Same lane, masked geometry: _run_klein_mask_job waits on _wait_for_job and
    # reads the render itself, then composites it over the user's file in place.
    # The finished image is never a dataset row of its own — the row it belongs
    # to already exists and keeps its filename.
    'watermark_klein_mask',
    # 📷 Camera angles. Its rows are `lora_test_image` (the produced view lands
    # in the gallery beside the picture it was shot from), so every job carries
    # `is_lora_test` and takes the SAME first branch as the Test Studio grid
    # above — checked before model_name. Listing it in DATASET_IMAGE_JOB_NAMES
    # instead would point the dispatch at face_dataset_service and look up a
    # dataset row that does not exist.
    'qwen_camera_angle',
}


def _stamped_job_names():
    """{model_name literal: module} for every `'model_name': '<literal>'` a
    service writes into job metadata.

    AST, not a regex over a hardcoded module list: the whole point is that a
    module nobody thought to name is found anyway."""
    found = {}
    for path in sorted(SERVICES.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == 'model_name'
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)):
                    found.setdefault(value.value, path.name)
    return found


def test_every_engine_that_stamps_a_job_name_is_classified():
    """THE guard the Krea-era one only claimed to be.

    Every `model_name` any service stamps must be either a declared dataset job
    (so `_dispatch_completion` links its result back) or explicitly listed above
    as getting home another way. A new engine fails here until its author has
    made that choice, which is the choice both incidents skipped."""
    from app import job_queue
    stamped = _stamped_job_names()
    assert stamped, 'no model_name stamps found at all — the AST walk is broken'
    unclassified = {name: mod for name, mod in stamped.items()
                    if name not in job_queue.DATASET_IMAGE_JOB_NAMES
                    and name not in NON_DATASET_JOB_NAMES}
    assert not unclassified, (
        f'these engines stamp a job name the dispatch does not route: {unclassified}. '
        'Add it to job_queue.DATASET_IMAGE_JOB_NAMES so its finished image is linked '
        'back to its row, or to NON_DATASET_JOB_NAMES with the reason it gets home '
        'another way. Skipping this is what stranded Krea (12 images) and SeedVR2.')


def test_no_declared_dataset_job_name_is_stamped_by_nobody():
    """The other direction, kept from the original test: a name in the set that
    no engine stamps is dead weight that makes the set look maintained."""
    from app import job_queue
    stamped = set(_stamped_job_names())
    orphans = set(job_queue.DATASET_IMAGE_JOB_NAMES) - stamped
    assert not orphans, f'declared but stamped by no engine: {orphans}'


def test_seedvr2_result_reaches_the_row_linker(app, monkeypatch):
    """The exact regression: a completed SeedVR2 job must reach the linker.

    Fails before the fix — `seedvr2_upscale` fell through the elif chain and the
    candidate row kept filename NULL forever."""
    from app import job_queue
    from app.models import ImageGenerationQueue
    from app.services import face_dataset_service
    with app.app_context():
        seen = {}
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda jid, fn, failed=False, reason=None:
                                seen.update(job_id=jid, filename=fn, failed=failed))
        job = ImageGenerationQueue(
            job_id='sv2-1', user_id='local', status='completed',
            workflow_data='{}', prompt='',
            job_metadata=json.dumps({
                'model_name': 'seedvr2_upscale', 'dataset_id': 44,
                'derivation_kind': 'klein_image_improve', 'parent_image_id': 13224,
                'action': 'upscale_improve', 'improve_engine': 'seedvr2'}))
        job_queue._dispatch_completion(job, 'local_DatasetSeedVR2_e9650fde_00001_.png', False)
        assert seen.get('job_id') == 'sv2-1', (
            'the SeedVR2 result never reached the linker — the image is rendered, '
            'paid for in GPU time, and stranded as a pending row with no file')
        assert seen['filename'] == 'local_DatasetSeedVR2_e9650fde_00001_.png'
        assert seen['failed'] is False


# --- The retroactive repair --------------------------------------------------
# Fixing the routing table helps the NEXT run. It gives nobody back the image
# they already paid GPU time for: the row stays pending with a NULL filename
# forever, and the only remedy was hand-written SQL. These cover the boot sweep
# that makes "Update & restart" the whole fix.

def _stranded_pair(app, *, job_status='completed', row_status='pending',
                   filename='local_DatasetSeedVR2_e9650fde_00001_.png',
                   model_name='seedvr2_upscale', job_id='sv2-strand'):
    """A finished queue row plus the dataset candidate it never reached — the
    exact shape observed in production (queue completed with a result_filename,
    candidate pending with filename NULL)."""
    from app.models import FaceDatasetImage, ImageGenerationQueue, db
    from app.services import face_dataset_service as svc
    ds = svc.create_dataset('local', f'harvest {job_id}', 'harvesttrigger')
    parent = FaceDatasetImage(dataset_id=ds.id, source='import', status='keep',
                              filename='parent.webp')
    db.session.add(parent)
    db.session.commit()
    candidate = FaceDatasetImage(
        dataset_id=ds.id, source='generated', status=row_status,
        parent_image_id=parent.id, derivation_kind=svc.KLEIN_IMAGE_IMPROVE,
        job_id=job_id, filename=None)
    job = ImageGenerationQueue(
        job_id=job_id, user_id='local', status=job_status,
        workflow_data='{}', prompt='', result_filename=filename,
        job_metadata=json.dumps({'model_name': model_name,
                                 'dataset_id': ds.id,
                                 'derivation_kind': svc.KLEIN_IMAGE_IMPROVE,
                                 'parent_image_id': parent.id,
                                 'improve_engine': 'seedvr2'}))
    db.session.add_all([candidate, job])
    db.session.commit()
    return candidate, job


def test_boot_harvests_a_finished_job_that_was_never_linked(app, monkeypatch):
    """THE repair: a candidate stranded by a past routing miss is picked up at
    boot, with no SQL and no user gesture beyond Update & restart."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        candidate, job = _stranded_pair(app)
        linked = {}
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda jid, fn, failed=False, reason=None:
                                linked.update(job_id=jid, filename=fn, failed=failed))
        job_queue.queue_manager._harvest_unlinked_completed_jobs()
        assert linked.get('job_id') == 'sv2-strand', (
            'the stranded candidate was not harvested — the rendered image stays '
            'invisible and only manual SQL could recover it')
        assert linked['filename'] == 'local_DatasetSeedVR2_e9650fde_00001_.png'
        assert linked['failed'] is False


def test_the_harvest_leaves_a_job_that_is_still_running_alone(app, monkeypatch):
    """A pending row with a live job is normal in-flight work, not damage."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        _stranded_pair(app, job_status='sent_to_comfy', job_id='sv2-inflight')
        called = []
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda *a, **k: called.append(a))
        job_queue.queue_manager._harvest_unlinked_completed_jobs()
        assert called == [], 'the harvest grabbed a job that had not finished'


def test_the_harvest_ignores_a_row_that_already_has_its_file(app, monkeypatch):
    """Only rows with NO file are damage. Re-linking a healthy one would move a
    file that is already in place."""
    from app import job_queue
    from app.services import face_dataset_service
    from app.models import db
    with app.app_context():
        candidate, _job = _stranded_pair(app, job_id='sv2-healthy')
        candidate.filename = 'already-here.png'
        db.session.commit()
        called = []
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda *a, **k: called.append(a))
        job_queue.queue_manager._harvest_unlinked_completed_jobs()
        assert called == []


def test_a_failed_job_is_harvested_too_so_the_tile_stops_spinning(app, monkeypatch):
    """A row stranded on a FAILED job is the same damage wearing the other face:
    the tile reads 'generating' forever instead of saying what went wrong."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        _stranded_pair(app, job_status='failed', filename=None, job_id='sv2-failed')
        seen = {}
        monkeypatch.setattr(face_dataset_service, 'link_completed_dataset_image',
                            lambda jid, fn, failed=False, reason=None:
                                seen.update(job_id=jid, failed=failed))
        job_queue.queue_manager._harvest_unlinked_completed_jobs()
        assert seen.get('job_id') == 'sv2-failed'
        assert seen.get('failed') is True


def test_the_harvest_survives_one_broken_row(app, monkeypatch):
    """A boot that cannot repair must still boot, and one bad row must not stop
    the others being rescued."""
    from app import job_queue
    from app.services import face_dataset_service
    with app.app_context():
        _stranded_pair(app, job_id='sv2-bad')
        _stranded_pair(app, job_id='sv2-good')
        ok = []

        def _linker(jid, fn, failed=False, reason=None):
            if jid == 'sv2-bad':
                raise RuntimeError('disk gone')
            ok.append(jid)

        monkeypatch.setattr(face_dataset_service,
                            'link_completed_dataset_image', _linker)
        job_queue.queue_manager._harvest_unlinked_completed_jobs()   # must not raise
        assert ok == ['sv2-good']
