"""The two RISKY data changes of the LoRA Canvas generation slice, and what they
buy: a checkpoint that keeps every preview it was given, and an exact
image→checkpoint link written at generation time.

  * recreating ``checkpoint_preview`` without its unique constraint — the only
    step of the feature that can lose rows WITHOUT AN ERROR, so it is guarded
    here by a ROW COUNT before and after, not by "the table still exists";
  * the ``(record_id, step)`` columns on ``lora_test_image``, written by the
    engine and backfilled once from evidence only — a row that cannot be
    attributed stays unlinked rather than being attached to a plausible guess.
"""
import pytest


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _legacy_preview_table(db):
    """Recreate the table EXACTLY as it shipped: one preview per checkpoint."""
    from sqlalchemy import text
    db.session.execute(text('DROP TABLE IF EXISTS checkpoint_preview'))
    db.session.execute(text("""
        CREATE TABLE checkpoint_preview (
            id INTEGER NOT NULL PRIMARY KEY,
            record_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            dataset_id INTEGER NOT NULL,
            lora_test_image_id INTEGER,
            prompt TEXT NOT NULL,
            seed BIGINT,
            created_at DATETIME,
            CONSTRAINT uq_checkpoint_preview UNIQUE (record_id, step)
        )"""))
    db.session.commit()


# --- lifting the unique constraint ------------------------------------------

def test_legacy_constraint_is_detected_and_a_fresh_table_is_left_alone(app):
    from app.extensions import db
    from app.services import checkpoint_preview_migration as mig
    with app.app_context():
        # A database created by this build has no constraint to lift.
        assert mig.has_unique_constraint() is False
        assert mig.run_if_needed()['reason'] == 'already-lifted'
        # …a database created by the shipped build does.
        _legacy_preview_table(db)
        assert mig.has_unique_constraint() is True


def test_recreating_the_table_keeps_EVERY_row(app):
    """THE guard. A missing column in the INSERT SELECT destroys rows and returns
    success, so what is asserted is the count, value by value, not the table."""
    from sqlalchemy import text
    from app.extensions import db
    from app.services import checkpoint_preview_migration as mig
    with app.app_context():
        _legacy_preview_table(db)
        for i in range(1, 18):
            db.session.execute(text(
                'INSERT INTO checkpoint_preview '
                '(id, record_id, step, dataset_id, lora_test_image_id, prompt, seed) '
                'VALUES (:i, :r, :s, 1, :img, :p, :seed)'),
                {'i': i, 'r': (i % 4) + 1, 's': i * 500, 'img': 900 + i,
                 'p': f'prompt {i}', 'seed': 1000 + i})
        db.session.commit()
        before = db.session.execute(
            text('SELECT COUNT(*) FROM checkpoint_preview')).scalar()
        assert before == 17

        res = mig.run_if_needed()
        assert res['migrated'] is True
        assert res['rows'] == before

        after = db.session.execute(
            text('SELECT COUNT(*) FROM checkpoint_preview')).scalar()
        assert after == before
        # Not just the count: the ids and every value survive (lora_test_image_id
        # pointers are addressed BY id — a renumbered copy would be silent rot).
        rows = db.session.execute(text(
            'SELECT id, record_id, step, lora_test_image_id, prompt, seed '
            'FROM checkpoint_preview ORDER BY id')).all()
        assert [r[0] for r in rows] == list(range(1, 18))
        assert rows[4] == (5, 2, 2500, 905, 'prompt 5', 1005)
        assert mig.has_unique_constraint() is False


def test_after_the_lift_a_checkpoint_KEEPS_its_previous_previews(app):
    """The point of the migration: regenerating no longer erases what was there."""
    from app.extensions import db
    from app.models import CheckpointPreview
    from app.services import checkpoint_preview_migration as mig
    with app.app_context():
        _legacy_preview_table(db)
        db.session.add(CheckpointPreview(record_id=7, step=1000, dataset_id=1,
                                         lora_test_image_id=11, prompt='a'))
        db.session.commit()
        mig.run_if_needed()
        db.session.add(CheckpointPreview(record_id=7, step=1000, dataset_id=1,
                                         lora_test_image_id=12, prompt='b'))
        db.session.commit()            # would have raised IntegrityError before
        kept = CheckpointPreview.query.filter_by(record_id=7, step=1000).all()
        assert sorted(k.lora_test_image_id for k in kept) == [11, 12]


def test_the_migration_is_idempotent_and_survives_a_crashed_attempt(app):
    from sqlalchemy import text
    from app.extensions import db
    from app.services import checkpoint_preview_migration as mig
    with app.app_context():
        _legacy_preview_table(db)
        db.session.execute(text(
            'INSERT INTO checkpoint_preview (id, record_id, step, dataset_id, prompt) '
            'VALUES (1, 1, 500, 1, "p")'))
        # A temp table left behind by a process killed mid-migration.
        db.session.execute(text(
            'CREATE TABLE checkpoint_preview__migrating (id INTEGER PRIMARY KEY)'))
        db.session.commit()
        assert mig.run_if_needed()['migrated'] is True
        assert mig.run_if_needed()['reason'] == 'already-lifted'   # second boot
        assert db.session.execute(
            text('SELECT COUNT(*) FROM checkpoint_preview')).scalar() == 1


def test_a_short_copy_ABORTS_and_leaves_the_original_untouched(app, monkeypatch):
    """If the copy ever came up short, the original must still be there. Simulated
    by making the temp table reject half the rows (a UNIQUE the real one lacks)."""
    from sqlalchemy import text
    from app.extensions import db
    from app.services import checkpoint_preview_migration as mig
    with app.app_context():
        _legacy_preview_table(db)
        for i in (1, 2, 3):
            db.session.execute(text(
                'INSERT INTO checkpoint_preview (id, record_id, step, dataset_id, prompt) '
                'VALUES (:i, 1, :s, 1, "p")'), {'i': i, 's': i * 100})
        db.session.commit()

        def lossy_temp():
            db.session.execute(text(
                'CREATE TABLE checkpoint_preview__migrating ('
                ' id INTEGER PRIMARY KEY, record_id INTEGER, step INTEGER,'
                ' dataset_id INTEGER, lora_test_image_id INTEGER, prompt TEXT,'
                ' seed BIGINT, created_at DATETIME)'))
        monkeypatch.setattr(mig, '_create_temp_table', lossy_temp)
        # …and make the copy itself lose a row.
        real_execute = db.session.execute

        def filtered(stmt, *a, **kw):
            s = str(stmt)
            if 'INSERT INTO checkpoint_preview__migrating' in s:
                return real_execute(text(s + ' WHERE id < 3'), *a, **kw)
            return real_execute(stmt, *a, **kw)
        monkeypatch.setattr(db.session, 'execute', filtered)

        res = mig.lift_unique_constraint()
        monkeypatch.undo()
        assert res['migrated'] is False
        assert res['reason'] == 'row-count-mismatch'
        # The original table is intact, with all three rows and its constraint.
        assert db.session.execute(
            text('SELECT COUNT(*) FROM checkpoint_preview')).scalar() == 3
        assert mig.has_unique_constraint() is True


def test_a_broken_database_does_not_stop_the_pass_from_returning(app, monkeypatch):
    from app.services import checkpoint_preview_migration as mig
    with app.app_context():
        monkeypatch.setattr(mig, 'has_unique_constraint',
                            lambda: (_ for _ in ()).throw(RuntimeError('disk')))
        assert mig.run_if_needed() == {'migrated': False, 'rows': 0, 'reason': 'failed'}


# --- the exact image → checkpoint link ---------------------------------------

def _record(db, dataset_id, source='local', cloud_run_id=None, family='krea'):
    from app.models import TrainingRunRecord
    rec = TrainingRunRecord(dataset_id=dataset_id, family=family, source=source,
                            cloud_run_id=cloud_run_id, fingerprint='f', version=1)
    db.session.add(rec)
    db.session.commit()
    return rec


def _image(db, dataset_id, checkpoint, **kw):
    from app.models import LoraTestImage
    img = LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint, strength=1.0,
                        status=kw.pop('status', 'done'),
                        filename=kw.pop('filename', 'x.png'), **kw)
    db.session.add(img)
    db.session.commit()
    return img


# A minimal safetensors header: listed and arch-sniffed like a deploy, never loaded.
_ST = (b'\x08\x00\x00\x00\x00\x00\x00\x00{"__metadata__":{}}'
       .ljust(32, b'\x00'))


def _comfy_loras(tmp_path, names):
    """Configure a ComfyUI whose LoRA folder holds exactly `names` (LoraLoader
    form, 'krea\\lora_….safetensors') — the pool a lineage pill is joined to."""
    from app import config
    base = tmp_path / 'Comfy'
    loras = base / 'models' / 'loras'
    loras.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = loras.joinpath(*name.split('\\'))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_ST)
    config.save_config({'comfyui': {'base_dir': str(base)}})


def _cloud_run_with_final_save(db, ds, monkeypatch, final_step=6000):
    """A cloud record whose save list ends on a final save at `final_step`."""
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    crun = CloudTrainingRun(dataset_id=ds, status='done')
    db.session.add(crun)
    db.session.commit()
    rec = _record(db, ds, source='cloud', cloud_run_id=crun.id)
    monkeypatch.setattr(ct, '_node_checkpoints', lambda rec, crun: [
        {'step': final_step // 2, 'filename': 'a', 'present': True},
        {'step': final_step, 'filename': 'b', 'present': True, 'final': True}])
    return rec, crun.id


def test_a_deployed_name_resolves_to_its_run_and_step(client, app):
    from app.extensions import db
    from app.services.checkpoint_link_backfill import resolve_checkpoint_name
    with app.app_context():
        ds = _create(client)
        local = _record(db, ds, source='local')
        cloud = _record(db, ds, source='cloud', cloud_run_id=74)
        assert resolve_checkpoint_name(
            f'krea\\lora_nova_000001500_Krea-2-Raw_rl{local.id}_v2.safetensors'
        ) == (local.id, 1500, ds)
        assert resolve_checkpoint_name(
            'krea\\lora_nova_000003000_Krea-2-Raw_rc74_v1.safetensors'
        ) == (cloud.id, 3000, ds)


@pytest.mark.parametrize('name', [
    # No run tag at all — imported before tagging existed.
    'krea\\lora_nova_000001500_Krea-2-Raw.safetensors',
    # A tag pointing at a run that does not exist.
    'krea\\lora_nova_000001500_Krea-2-Raw_rl9999_v1.safetensors',
    'krea\\lora_nova_000001500_Krea-2-Raw_rc9999_v1.safetensors',
    # A final save: tagged, but with no step in the name. Its step is only
    # knowable by scanning the run folder — so it is NOT guessed.
    'krea\\lora_nova_Krea-2-Raw_rl1_v2.safetensors',
])
def test_an_unattributable_name_resolves_to_nothing(client, app, name):
    from app.extensions import db
    from app.services.checkpoint_link_backfill import resolve_checkpoint_name
    with app.app_context():
        ds = _create(client)
        _record(db, ds, source='local')
        assert resolve_checkpoint_name(name) is None


def test_two_records_claiming_one_cloud_run_leave_the_image_unlinked(client, app):
    from app.extensions import db
    from app.services.checkpoint_link_backfill import resolve_checkpoint_name
    with app.app_context():
        ds = _create(client)
        _record(db, ds, source='cloud', cloud_run_id=42)
        _record(db, ds, source='cloud', cloud_run_id=42)
        assert resolve_checkpoint_name(
            'krea\\lora_nova_000500_Krea_rc42_v1.safetensors') is None


def test_the_backfill_links_by_pointer_then_by_tag_and_states_the_gap(client, app):
    from app.extensions import db
    from app.models import CheckpointPreview, LoraTestImage
    from app.services import checkpoint_link_backfill as bf
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, source='local')

        # 1. an image the Lab generated inline: the pointer IS the evidence, and
        #    its filename carries nothing to parse.
        by_pointer = _image(db, ds, 'krea\\anything.safetensors')
        db.session.add(CheckpointPreview(record_id=rec.id, step=2000, dataset_id=ds,
                                         lora_test_image_id=by_pointer.id, prompt=''))
        # 2. an image whose deployed name attributes itself.
        by_tag = _image(db, ds,
                        f'krea\\lora_nova_000001500_Krea-2-Raw_rl{rec.id}_v1.safetensors')
        # 3. an image that attributes itself to nothing.
        orphan = _image(db, ds, 'krea\\lora_nova_Krea-2-Raw.safetensors')
        db.session.commit()

        out = bf.backfill_checkpoint_links()
        assert out['by_pointer'] == 1
        assert out['by_name'] == 1
        assert out['unlinked'] == 1
        db.session.expire_all()
        assert (db.session.get(LoraTestImage, by_pointer.id).record_id,
                db.session.get(LoraTestImage, by_pointer.id).step) == (rec.id, 2000)
        assert (db.session.get(LoraTestImage, by_tag.id).record_id,
                db.session.get(LoraTestImage, by_tag.id).step) == (rec.id, 1500)
        assert db.session.get(LoraTestImage, orphan.id).record_id is None
        assert bf.unlinked_count() == 1

        # Idempotent: a second pass finds nothing left to do and changes nothing.
        again = bf.backfill_checkpoint_links()
        assert (again['by_pointer'], again['by_name']) == (0, 0)


def test_the_backfill_never_overwrites_a_link_already_written(client, app):
    from app.extensions import db
    from app.models import LoraTestImage
    from app.services import checkpoint_link_backfill as bf
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds, source='local')
        img = _image(db, ds,
                     f'krea\\lora_nova_000001500_Krea_rl{rec.id}_v1.safetensors',
                     record_id=rec.id, step=777)
        db.session.commit()
        bf.backfill_checkpoint_links()
        db.session.expire_all()
        assert db.session.get(LoraTestImage, img.id).step == 777


def test_the_run_marker_makes_the_pass_run_once(client, app, monkeypatch):
    from app.services import checkpoint_link_backfill as bf
    with app.app_context():
        _create(client)
        first = bf.run_if_needed()
        assert first['version'] == bf.BACKFILL_VERSION
        calls = []
        monkeypatch.setattr(bf, 'backfill_checkpoint_links',
                            lambda: calls.append(1) or {'by_pointer': 0, 'by_name': 0,
                                                        'unlinked': 0})
        bf.run_if_needed()
        assert calls == []


# --- what the link is FOR ----------------------------------------------------

def test_the_gallery_returns_every_image_of_a_checkpoint_newest_first(client, app):
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        first = _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
                       filename='one.png')
        second = _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
                        filename='two.png')
        _image(db, ds, 'a.safetensors', record_id=rec.id, step=2000, filename='other.png')
        # A pending cell is not a gallery entry (there is no image yet).
        _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
               status='pending', filename=None)
        db.session.commit()

        out = ct.checkpoint_gallery(rec.id, 1000)
        assert out['count'] == 2
        assert [i['id'] for i in out['images']] == [second.id, first.id]
        assert out['images'][0]['url'] == f'/api/dataset/{ds}/img/two.png'
        assert out['unlinked'] == 0


def test_the_gallery_route_answers_and_an_empty_checkpoint_is_not_an_error(client, app):
    r = client.get('/api/train/checkpoint/424242/1000/images')
    assert r.status_code == 200
    body = r.get_json()
    # `delete_mode` is where a 🗑 would send these files — announced with the
    # gallery so the confirmation can name it before arming (see
    # test_checkpoint_gallery_delete). It depends on whether send2trash is
    # installed, so only its domain is asserted here.
    assert body.pop('delete_mode') in ('trash', 'app_trash')
    assert body == {'record_id': 424242, 'step': 1000, 'count': 0,
                    'unlinked': 0, 'images': []}


def test_the_node_reports_the_newest_preview_and_the_gallery_size(client, app):
    from app.extensions import db
    from app.models import CheckpointPreview
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        old = _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
                     filename='old.png')
        new = _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
                     filename='new.png')
        db.session.add(CheckpointPreview(record_id=rec.id, step=1000, dataset_id=ds,
                                         lora_test_image_id=old.id, prompt=''))
        db.session.add(CheckpointPreview(record_id=rec.id, step=1000, dataset_id=ds,
                                         lora_test_image_id=new.id, prompt=''))
        db.session.commit()
        out = ct.checkpoint_previews_for(rec.id)
        assert out[1000]['url'] == f'/api/dataset/{ds}/img/new.png'
        assert out[1000]['count'] == 2


def test_a_checkpoint_tested_from_the_studio_still_reports_its_gallery(client, app):
    """Test-Studio cells never create a preview pointer. The node must still say
    "there are images here" — that is decision 5 (the gallery holds EVERYTHING)."""
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, 'a.safetensors', record_id=rec.id, step=500, filename='g.png')
        db.session.commit()
        out = ct.checkpoint_previews_for(rec.id)
        assert out[500]['count'] == 1
        assert out[500]['url'] is None      # nothing was pinned as THE preview


# --- the link is written at generation time ----------------------------------

def test_a_canvas_launch_stamps_the_checkpoint_it_was_told_about(client, app):
    """The canvas KNOWS the run and the step — the user clicked that pill. The
    explicit origin wins over anything the filename might suggest."""
    from app.services.lora_test_studio import checkpoint_origins
    with app.app_context():
        got = checkpoint_origins(
            ['krea\\lora_nova_000001500_Krea_rl3_v1.safetensors'],
            {'krea\\lora_nova_000001500_Krea_rl3_v1.safetensors':
             {'record_id': 88, 'step': 4000}})
        assert got['krea\\lora_nova_000001500_Krea_rl3_v1.safetensors'] == (88, 4000)


def test_a_studio_launch_recovers_the_origin_from_the_deploy_tag(client, app):
    from app.extensions import db
    from app.services.lora_test_studio import checkpoint_origins
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        fn = f'krea\\lora_nova_000002500_Krea-2-Raw_rl{rec.id}_v1.safetensors'
        assert checkpoint_origins([fn])[fn] == (rec.id, 2500)
        # …and an untagged pick is left unlinked rather than attached to a guess.
        assert checkpoint_origins(['krea\\lora_nova.safetensors']) == {
            'krea\\lora_nova.safetensors': (None, None)}


def test_a_studio_launch_on_a_step_less_final_deploy_records_its_final_step(
        client, app, monkeypatch, tmp_path):
    """ai-toolkit deploys a run's FINAL save with the run tag and no step
    (`…_rc192_v2`). Cells tested from it were stored with neither run nor step,
    so the Canvas and the checkpoint galleries never showed them. The run's
    lineage already pairs that file with its final pill; the launch asks the
    same join."""
    from app.extensions import db
    from app.services.lora_test_studio import checkpoint_origins
    with app.app_context():
        ds = _create(client)
        rec, cid = _cloud_run_with_final_save(db, ds, monkeypatch)
        final = f'krea\\lora_nova_Krea-2-Raw_rc{cid}_v2.safetensors'
        older = f'krea\\lora_nova_Krea-2-Raw_rc{cid}_v1.safetensors'
        _comfy_loras(tmp_path, [final, older])
        assert checkpoint_origins([final], family='krea')[final] == (rec.id, 6000)
        # An older deploy of the same final save: no pill shows it (the newest
        # version takes the final pill), so it keeps its run and no step.
        assert checkpoint_origins([older], family='krea')[older] == (rec.id, None)


def test_the_gallery_publishes_WHICH_LAUNCH_made_each_image(client, app):
    """``run_id`` groups every cell of one "Generate" and had never left the
    database. The ◉ Canvas needs it: two runs fired at the SAME checkpoint are
    otherwise indistinguishable, so pinning the second one appended its pictures
    to the first one's strip and the board showed one lot where there were two.
    Null on images that predate the column — the canvas falls back to the
    checkpoint there, so an old board draws what it always drew."""
    from app.extensions import db
    from app.services import cloud_training as ct
    with app.app_context():
        ds = _create(client)
        rec = _record(db, ds)
        _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
               filename='a.png', run_id='launch-A')
        _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
               filename='b.png', run_id='launch-B')
        _image(db, ds, 'a.safetensors', record_id=rec.id, step=1000,
               filename='legacy.png')
        db.session.commit()

        by_url = {i['url'].rsplit('/', 1)[-1]: i
                  for i in ct.checkpoint_gallery(rec.id, 1000)['images']}
        assert by_url['a.png']['run_id'] == 'launch-A'
        assert by_url['b.png']['run_id'] == 'launch-B'
        assert by_url['legacy.png']['run_id'] is None
