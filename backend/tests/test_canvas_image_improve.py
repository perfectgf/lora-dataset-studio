"""✨ Upscale & improve, run on an image of the ◉ Canvas board.

WHY THIS IS ITS OWN LANE, and not a reuse of `/api/dataset/image/<id>/improve`:
the board's pictures are `lora_test_image` rows (that is what
`canvas_image_node.image_id` stores) while the dataset route resolves a
`face_dataset_image`. The two tables have INDEPENDENT autoincrement id spaces,
so handing a board id to the dataset route does NOT 404 — it finds a real,
unrelated dataset image and improves that one. That silent wrong answer is the
bug this whole file exists to make impossible.

The result row lives in `lora_test_image` because the board must be able to pin
it, and that is the hazard: an audit of the 21 read sites in lora_test_studio.py
found TEN that would read it as a Test Studio CELL. The tests below pin the
boundary in BOTH directions — invisible to the studio, visible in the gallery —
because a future contributor "harmonising" either half silently deletes a
feature or resurrects a bug.
"""
import io
import os

import pytest
from PIL import Image


def _png(color=(30, 60, 90)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 48), color).save(buf, 'PNG')
    return buf.getvalue()


def _board_image(svc, *, record_id=7, step=1200, filename='render.png',
                 write=True, status='done', **kw):
    """One finished board render: a LoraTestImage with a file on disk."""
    from app.extensions import db
    from app.models import LoraTestImage
    ds = svc.create_dataset('local', 'Canvas improve', 'canvastrigger')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    if filename and write:
        with open(os.path.join(svc._dataset_dir(ds.id), filename), 'wb') as fh:
            fh.write(_png())
    row = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\Lola-1200.safetensors',
                        strength=0.9, status=status, filename=filename,
                        record_id=record_id, step=step, run_id='run-a', seed=4242,
                        prompt='a portrait', z_model='zturbo.safetensors',
                        **kw)
    db.session.add(row)
    db.session.commit()
    return ds, row


@pytest.fixture()
def engine_stub(monkeypatch):
    """Capture the job hand-off instead of talking to ComfyUI.

    Deliberately stubs at `_enqueue_improve` — BELOW the engine resolution and the
    preflight, so those two stay under test. No GPU work is ever started here.
    """
    from app.services import face_dataset_service as svc
    calls = []

    def fake(engine, *, user_id, source, source_path, prompt, label, dataset,
             extra_metadata=None, profile=None):
        calls.append({'engine': engine, 'user_id': user_id, 'source': source,
                      'source_path': source_path, 'prompt': prompt,
                      'dataset': dataset, 'extra_metadata': extra_metadata,
                      'profile': profile})
        return f'job-{len(calls)}'

    monkeypatch.setattr(svc, '_enqueue_improve', fake)
    monkeypatch.setattr(svc, '_improve_preflight', lambda engine: None)
    return calls


# --- the hand-off: right engine, right source, right routing ------------------

def test_the_submitted_job_carries_the_source_file_and_routes_back_to_this_table(
        app, engine_stub):
    """The payload is the proof: a wrong `source_path` or a missing
    `is_lora_test` is exactly how this feature would improve the wrong picture,
    or improve the right one and lose the result."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, row = _board_image(svc)
        result = lts.improve_canvas_image('local', row.id, engine='klein')

        assert result['job_id'] == 'job-1'
        assert result['engine'] == 'klein'
        call = engine_stub[0]
        # The file actually improved is THIS row's file, resolved in its own
        # dataset directory — not a same-numbered dataset image.
        assert call['source_path'] == os.path.join(svc._dataset_dir(ds.id), 'render.png')
        assert call['source'].id == row.id
        meta = call['extra_metadata']
        # `is_lora_test` is what job_queue._dispatch_completion checks FIRST; without
        # it the completion goes looking for a FaceDatasetImage that does not exist
        # and the candidate stays pending forever.
        assert meta['is_lora_test'] is True
        assert meta['derivation_kind'] == lts.CANVAS_IMAGE_IMPROVE
        assert meta['parent_image_id'] == row.id
        assert meta['improve_engine'] == 'klein'
        assert meta['cell_id'] == result['candidate_id']


def test_seedvr2_is_handed_off_without_a_prompt(app, engine_stub):
    """A restoration sends no instruction; storing Klein's would put a sentence
    on screen that had no effect on the picture."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        ds, row = _board_image(svc)
        result = lts.improve_canvas_image('local', row.id, engine='seedvr2')
        assert engine_stub[0]['engine'] == 'seedvr2'
        assert engine_stub[0]['prompt'] == ''
        candidate = svc.db.session.get(LoraTestImage, result['candidate_id'])
        assert 'no prompt' in candidate.prompt


def test_the_engine_falls_back_to_the_setting_exactly_like_the_dataset_lane(
        app, engine_stub):
    """Reused, not re-implemented: `resolve_improve_engine` is the shared one, so
    a stale tab degrades to the historical Klein instead of refusing."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        _ds, row = _board_image(svc)
        lts.improve_canvas_image('local', row.id, engine='not-an-engine')
        assert engine_stub[0]['engine'] == 'klein'


def test_the_preflight_of_the_chosen_engine_still_runs(app, monkeypatch):
    """The 409 that offers to install SeedVR2 on demand must reach this surface
    too — that is the point of routing through the shared preflight."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.services.seedvr2_helper import SeedVR2ModelsMissing

    def boom(engine):
        raise SeedVR2ModelsMissing([])

    monkeypatch.setattr(svc, '_improve_preflight', boom)
    with app.app_context():
        _ds, row = _board_image(svc)
        with pytest.raises(SeedVR2ModelsMissing):
            lts.improve_canvas_image('local', row.id, engine='seedvr2')


# --- refusals, stated before the click ----------------------------------------

def test_an_improvement_cannot_be_improved_again(app, engine_stub):
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        ds, row = _board_image(svc)
        result = lts.improve_canvas_image('local', row.id)
        candidate = svc.db.session.get(LoraTestImage, result['candidate_id'])
        candidate.status = 'done'
        candidate.filename = 'improved.png'
        svc.db.session.commit()
        with open(os.path.join(svc._dataset_dir(ds.id), 'improved.png'), 'wb') as fh:
            fh.write(_png())
        with pytest.raises(ValueError, match='cannot be improved again'):
            lts.improve_canvas_image('local', candidate.id)


def test_a_second_click_returns_the_same_candidate_instead_of_a_second_job(
        app, engine_stub):
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        _ds, row = _board_image(svc)
        first = lts.improve_canvas_image('local', row.id)
        second = lts.improve_canvas_image('local', row.id)
        assert second['candidate_id'] == first['candidate_id']
        assert len(engine_stub) == 1          # the GPU is not asked twice


def test_a_failed_attempt_can_be_retried_by_pressing_the_button_again(
        app, engine_stub):
    """The studio's resume path deliberately no longer picks these rows up (it
    would re-queue them as Z-Image cells — wrong engine, wrong workflow), so
    ✨ again IS the retry. A failed candidate must therefore not block."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        _ds, row = _board_image(svc)
        first = lts.improve_canvas_image('local', row.id)
        failed = svc.db.session.get(LoraTestImage, first['candidate_id'])
        failed.status = 'failed'
        failed.error = 'ComfyUI said no'
        svc.db.session.commit()
        second = lts.improve_canvas_image('local', row.id)
        assert second['candidate_id'] != first['candidate_id']
        assert len(engine_stub) == 2


def test_a_render_that_is_not_finished_or_whose_file_is_gone_is_refused(
        app, engine_stub):
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        _ds, pending = _board_image(svc, status='pending', filename=None, write=False)
        with pytest.raises(ValueError, match='still rendering'):
            lts.improve_canvas_image('local', pending.id)
        _ds2, gone = _board_image(svc, filename='vanished.png', write=False)
        with pytest.raises(ValueError, match='no longer on disk'):
            lts.improve_canvas_image('local', gone.id)


def test_an_unknown_image_answers_not_found_rather_than_improving_a_neighbour(
        app, engine_stub):
    """The whole hazard in one assertion: an id this table does not hold must
    resolve to nothing, never to a row of the OTHER table."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        assert lts.improve_canvas_image('local', 987654) is None


# --- no ghost rows ------------------------------------------------------------

def test_a_failed_enqueue_leaves_no_pending_row_behind(app, monkeypatch):
    """A `pending` row with no file is the exact shape `_active_run_count` counts.
    Left behind by a failed enqueue it would be a permanent 'a test run is
    already in progress' the day the derivation filter ever slipped."""
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts

    def boom(*a, **kw):
        raise RuntimeError('ComfyUI is not configured')

    monkeypatch.setattr(svc, '_improve_preflight', lambda engine: None)
    monkeypatch.setattr(svc, '_enqueue_improve', boom)
    with app.app_context():
        ds, row = _board_image(svc)
        before = LoraTestImage.query.count()
        with pytest.raises(RuntimeError):
            lts.improve_canvas_image('local', row.id)
        db.session.expire_all()
        assert LoraTestImage.query.count() == before
        assert lts._active_run_count(ds.id) == 0


def test_an_improvement_in_flight_never_blocks_a_new_test_run(app, engine_stub):
    """THE regression this design had to avoid. `_active_run_count` filters on
    `status='pending'` + no file and applies `dataset_id` only when given — so
    with none it is a GLOBAL guard, and an improve counted there would have
    blocked a multi-LoRA comparison across EVERY dataset."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        ds, row = _board_image(svc)
        lts.improve_canvas_image('local', row.id)
        assert lts._active_run_count(ds.id) == 0
        assert lts._active_run_count() == 0        # the global guard too


def test_an_improvement_is_deleted_with_its_run_and_with_its_dataset(
        app, engine_stub):
    """`run_id` is NULL on these rows; the cleanup must key on `record_id`, which
    is copied. A row that escaped both sweeps would be a file nothing can reach."""
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.services import run_cascade_delete as rcd
    with app.app_context():
        _ds, row = _board_image(svc, record_id=31)
        result = lts.improve_canvas_image('local', row.id)
        candidate = db.session.get(LoraTestImage, result['candidate_id'])
        # While it is still rendering the sweep protects it, like every live row
        # (_LIVE_IMAGE_STATES). Finished, it must be swept with its run.
        assert result['candidate_id'] not in rcd._image_split(31)[0]
        candidate.status = 'done'
        candidate.filename = 'improved.png'
        db.session.commit()
        doomed, _kept = rcd._image_split(31)
        assert result['candidate_id'] in doomed
        assert row.id in doomed                   # its source, by the same record_id

        ds2, row2 = _board_image(svc, record_id=32)
        second = lts.improve_canvas_image('local', row2.id)
        finished = db.session.get(LoraTestImage, second['candidate_id'])
        # Finished, like any row the dataset delete walks. (Still in flight, the
        # delete refuses the whole dataset until the ComfyUI job is cancelled —
        # existing behaviour, and it covers these rows for free.)
        finished.status = 'done'
        finished.filename = 'improved.png'
        db.session.commit()
        svc.delete_dataset('local', ds2.id)
        db.session.expire_all()
        assert db.session.get(LoraTestImage, second['candidate_id']) is None


# --- THE BOUNDARY, both directions -------------------------------------------

def test_the_improvement_is_invisible_to_the_test_studio(app, engine_stub):
    """Direction 1. Ten readers break otherwise; these two are the ones a user
    would SEE (a phantom cell in the grid) and the one that silently corrupts a
    decision (a vote for a checkpoint that did not make the picture)."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        ds, row = _board_image(svc)
        result = lts.improve_canvas_image('local', row.id)
        candidate = svc.db.session.get(LoraTestImage, result['candidate_id'])
        candidate.status = 'done'
        candidate.filename = 'improved.png'
        candidate.rating = 1              # rated in the gallery, like any picture
        svc.db.session.commit()

        payload = lts.studio_payload('local', ds.id)
        shown = {c['id'] for c in payload['cells']}
        assert shown == {row.id}, 'the improvement showed up as a grid cell'
        # The ranking must not have gained a vote from an UPSCALE: one cell, and
        # the 👍 given to the improvement counts for nothing here.
        scores = lts.cell_scores(ds.id)
        assert sum(e['images'] for e in scores) == 1
        assert all(e['likes'] == 0 for e in scores)
        assert all(e['likes'] == 0 for e in lts.cell_scores(ds.id, family='zimage'))


def test_the_improvement_IS_in_the_checkpoint_gallery_and_can_be_pinned(
        app, engine_stub):
    """Direction 2, and the reason the row lives in this table at all: the
    improved picture was asked for NEXT TO the original, on the board.

    If someone ever "harmonises" cloud_training.py onto lora_test_studio._cells(),
    this test is what says the feature just disappeared."""
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        ds, row = _board_image(svc, record_id=44, step=900)
        result = lts.improve_canvas_image('local', row.id)
        candidate = svc.db.session.get(LoraTestImage, result['candidate_id'])
        candidate.status = 'done'
        candidate.filename = 'improved.png'
        svc.db.session.commit()
        with open(os.path.join(svc._dataset_dir(ds.id), 'improved.png'), 'wb') as fh:
            fh.write(_png())

        gallery = ct.checkpoint_gallery(44, 900)
        ids = [i['id'] for i in gallery['images']]
        assert result['candidate_id'] in ids
        assert row.id in ids                      # next to its source
        published = next(i for i in gallery['images'] if i['id'] == result['candidate_id'])
        # The front needs both to badge the tile and to refuse improving an
        # improvement before the click.
        assert published['derivation_kind'] == lts.CANVAS_IMAGE_IMPROVE
        assert published['parent_image_id'] == row.id

        # Pinnable: canvas_image_nodes resolves board nodes against this table and
        # keeps only done+filed rows, so the improvement qualifies by construction.
        ct.save_canvas_image_nodes('local', ds.id, [
            {'image_id': result['candidate_id'], 'x': 10, 'y': 20, 'w': 260, 'h': 260,
             'visible': True}])
        nodes = ct.canvas_image_nodes('local')
        pinned = [n for n in nodes['nodes'].get(str(ds.id), [])
                  if n['image_id'] == result['candidate_id']]
        assert len(pinned) == 1
        assert pinned[0]['image']['derivation_kind'] == lts.CANVAS_IMAGE_IMPROVE


def test_the_improvement_stays_out_of_the_checkpoint_timeline(app, engine_stub):
    """Free exclusion, and a deliberate one: the timeline filters
    `run_id IS NOT NULL` (checkpoint_timeline.py:194 and :413), and a 2 MP upscale
    spliced into an epoch-by-epoch morph would be a lie about what changed."""
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    with app.app_context():
        _ds, row = _board_image(svc)
        result = lts.improve_canvas_image('local', row.id)
        candidate = svc.db.session.get(LoraTestImage, result['candidate_id'])
        assert candidate.run_id is None


# --- the invariant is ENFORCED, not remembered --------------------------------

def test_no_bare_lora_test_query_escapes_the_cells_helper():
    """The filter is spread over 11 call sites, so it cannot rely on memory.

    A query that legitimately needs EVERY row (the four that resolve one row by
    `job_id`, and the idempotency lookup that wants derived rows on purpose)
    declares itself with `lds-allow-bare-lora-test-query:` and says why — an
    escape hatch that is visible in review beats a rule people delete.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'app' / 'services' / 'lora_test_studio.py').read_text(encoding='utf-8')
    lines = src.splitlines()
    offenders = []
    for i, line in enumerate(lines):
        if 'LoraTestImage.query' not in line:
            continue
        if line.strip().startswith('#') or '`' in line:
            continue                       # a comment, or prose quoting the name
        if 'derivation_kind' in line or '_is_cell()' in line:
            continue                       # _cells() itself: this line IS the filter
        window = '\n'.join(lines[max(0, i - 5):i])
        if 'lds-allow-bare-lora-test-query' in window:
            continue
        offenders.append((i + 1, line.strip()))
    # …and the OTHER spelling. A query that only JOINS the table never says
    # `LoraTestImage.query`, so the check above cannot see it — which is exactly
    # how `measured_seconds_per_image` (added by a parallel change) started
    # averaging improve durations into the Test Studio's pace estimate without a
    # single git conflict. A join must carry `_is_cell()` in the same statement.
    for i, line in enumerate(lines):
        if 'join(LoraTestImage' not in line:
            continue
        statement = '\n'.join(lines[i:i + 25])
        if '_is_cell()' in statement or 'lds-allow-bare-lora-test-query' in statement:
            continue
        offenders.append((i + 1, line.strip()))
    assert not offenders, (
        'bare LoraTestImage.query outside _cells(): these would read ✨ improve '
        f'results as Test Studio cells — {offenders}')


def test_an_improve_never_enters_the_studios_measured_pace(app, engine_stub):
    """The duration estimate shown before a launch is a median over the last
    finished CELLS. An improve is a 2 MP Klein edit or a SeedVR2 restoration —
    minutes where a Turbo cell takes seconds — and it copies its source's
    `checkpoint`, so it passes the family filter and lands in the same bucket.
    With a 30-sample median and a 0.5-900 s accept window, a few of them visibly
    inflate the time the studio promises the user."""
    from app.extensions import db
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from datetime import datetime, timedelta

    def finished(job_id, seconds, row_id):
        start = datetime(2026, 8, 3, 12, 0, 0)
        db.session.add(ImageGenerationQueue(
            job_id=job_id, status='completed', started_at=start,
            completed_at=start + timedelta(seconds=seconds)))
        row = db.session.get(LoraTestImage, row_id)
        row.job_id = job_id
        db.session.commit()

    with app.app_context():
        ds, row = _board_image(svc)
        finished('cell-a', 10, row.id)
        # Two more ordinary cells, so the median has its minimum sample.
        for n, secs in (('cell-b', 12), ('cell-c', 8)):
            _ds2, extra = _board_image(svc)
            finished(n, secs, extra.id)
        assert lts.measured_seconds_per_image() == 10.0

        result = lts.improve_canvas_image('local', row.id)
        finished('improve-a', 240, result['candidate_id'])
        # Unchanged: the four-minute upscale is not a sample of anything the
        # studio is about to run.
        assert lts.measured_seconds_per_image() == 10.0


def test_the_studio_grid_export_reads_cells_only():
    """The nastiest of the ten: `collect_grid` picks the current run with
    `max(done, key=id)`, and an improvement is newest by construction — it would
    have hijacked the run_seed/prompt/aspect of the whole export."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'app' / 'services' / 'studio_grid_export.py').read_text(encoding='utf-8')
    assert 'lts._cells().filter_by(dataset_id=dataset_id)' in src
    assert 'LoraTestImage.query.filter_by(dataset_id=dataset_id)' not in src


# --- legacy databases ---------------------------------------------------------

def test_the_new_columns_reach_a_database_that_predates_them(app):
    """A repo requirement, and the reason these are nullable: an install that
    never had the columns must boot, keep its rows, and read NULL — which means
    'an ordinary cell' and therefore behaves exactly as it always did."""
    from sqlalchemy import text
    from app import _SCHEMA_ADDITIONS, _apply_additive_migrations
    from app.extensions import db
    from app.services import face_dataset_service as svc

    assert ('lora_test_image', 'derivation_kind', 'VARCHAR(32)') in _SCHEMA_ADDITIONS
    assert ('lora_test_image', 'parent_image_id', 'INTEGER') in _SCHEMA_ADDITIONS
    with app.app_context():
        _ds, row = _board_image(svc)
        row_id = row.id
        # The index has to go first: SQLite refuses to drop a column an index
        # still names. `_apply_additive_migrations` re-creates both (IF NOT
        # EXISTS), which is exactly what a legacy database gets on its next boot.
        db.session.execute(text('DROP INDEX IF EXISTS ix_lora_test_image_parent_image_id'))
        db.session.execute(text('ALTER TABLE lora_test_image DROP COLUMN derivation_kind'))
        db.session.execute(text('ALTER TABLE lora_test_image DROP COLUMN parent_image_id'))
        db.session.commit()

        def columns():
            return {r[1] for r in db.session.execute(text('PRAGMA table_info(lora_test_image)'))}

        assert 'derivation_kind' not in columns()
        _apply_additive_migrations()
        _apply_additive_migrations()       # runs on every boot: must stay a no-op
        assert {'derivation_kind', 'parent_image_id'} <= columns()

        legacy = db.session.execute(text(
            'SELECT checkpoint, derivation_kind, parent_image_id '
            'FROM lora_test_image WHERE id=:id'), {'id': row_id}).one()
        assert legacy[1] is None and legacy[2] is None
        # NULL means "an ordinary cell": the pre-existing row is still a cell.
        assert lts_cells_contains(row_id)


def lts_cells_contains(row_id):
    from app.services import lora_test_studio as lts
    return any(r.id == row_id for r in lts._cells().all())


# --- the route ----------------------------------------------------------------

def test_the_route_answers_ok_and_404s_an_id_this_table_does_not_hold(
        app, client, engine_stub):
    from app.services import face_dataset_service as svc
    with app.app_context():
        _ds, row = _board_image(svc)
        image_id = row.id
    r = client.post(f'/api/canvas/image/{image_id}/improve', json={'engine': 'klein'})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['engine'] == 'klein'
    assert client.post('/api/canvas/image/999999/improve', json={}).status_code == 404
