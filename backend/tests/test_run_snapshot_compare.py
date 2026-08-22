"""Freezing a launch, and comparing two of them.

What these tests defend:
  * a launch records the caption TEXT and the true content hash of every image,
    plus the machine, in one JSON blob;
  * the content hash is CACHED on the row and reused while size+mtime hold, so a
    repeat launch does not re-read the dataset;
  * the fingerprint (and therefore every dataset version already in every user's
    database) is NOT affected by any of it;
  * the compare payload says what changed — including the caption text and a
    picture of an image that has since been deleted;
  * a run recorded before snapshots existed degrades to an explicit statement,
    never to a silent "nothing changed".
"""
import json
import os
from datetime import datetime, timedelta


from app.extensions import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import checkpoint_registry as reg
from app.services import run_archive, run_compare, run_environment, run_snapshot


_T0 = datetime(2026, 7, 20, 10, 0, 0)


def _dataset(name='marion', trigger='mrn', **kw):
    ds = FaceDataset(user_id='local', name=name, trigger_word=trigger, **kw)
    db.session.add(ds)
    db.session.commit()
    return ds


def _image(ds, caption, body=b'pixels', filename=None, **kw):
    """A kept image with a REAL file on disk — the content hash has to have
    something to read."""
    img = FaceDatasetImage(dataset_id=ds.id, status='keep', caption=caption,
                           filename=filename or f'img{len(caption)}.png', **kw)
    db.session.add(img)
    db.session.commit()
    img.filename = f'img{img.id}.png'
    db.session.commit()
    from app import config as cfg
    folder = cfg.dataset_images_root() / str(ds.id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / img.filename).write_bytes(body)
    return img


def _rewrite(ds, img, body):
    from app import config as cfg
    path = cfg.dataset_images_root() / str(ds.id) / img.filename
    path.write_bytes(body)
    # Force a distinct mtime so the size:mtime cache key can never coincide.
    os.utime(path, (0, 0))


# --- the freeze itself --------------------------------------------------------

def test_launch_records_caption_text_and_content_hash(app):
    with app.app_context():
        ds = _dataset()
        a = _image(ds, 'a woman in a red coat', b'AAAA')
        b = _image(ds, 'a woman on a beach', b'BBBB')
        rec = reg.register_launch('local', ds.id, family='krea', source='local')
        snap = run_snapshot.loads(rec)
        assert snap is not None
        assert run_snapshot.caption_of(snap, a.id)[0] == 'a woman in a red coat'
        assert run_snapshot.caption_of(snap, b.id)[0] == 'a woman on a beach'
        # Different bytes -> different content hashes, and both are recorded.
        sig_a = run_snapshot.content_of(snap, a.id)
        sig_b = run_snapshot.content_of(snap, b.id)
        assert sig_a and sig_b and sig_a != sig_b


def test_content_hash_is_cached_and_reused_while_the_file_is_untouched(app):
    """The launch path must not re-read the whole dataset every time. The second
    prepare hashes NOTHING: the cached signature still matches size+mtime."""
    with app.app_context():
        ds = _dataset()
        img = _image(ds, 'caption', b'ZZZZ')
        first = reg.prepare_launch('local', ds.id)
        assert len(first['sig_updates']) == 1        # cold: one hash computed
        db.session.commit()
        for row, sig, stat in first['sig_updates']:
            row.content_sig, row.content_sig_stat = sig, stat
        db.session.commit()
        second = reg.prepare_launch('local', ds.id)
        assert second['sig_updates'] == []           # warm: nothing re-read
        assert (second['snapshot']['images'][str(img.id)]['c']
                == first['snapshot']['images'][str(img.id)]['c'])
        # ...and an actual pixel change invalidates it.
        _rewrite(ds, img, b'QQQQQQQQ')
        third = reg.prepare_launch('local', ds.id)
        assert len(third['sig_updates']) == 1
        assert (third['snapshot']['images'][str(img.id)]['c']
                != first['snapshot']['images'][str(img.id)]['c'])


def test_snapshot_does_not_change_the_fingerprint(app):
    """Everything added here is EXTRA. If it leaked into the fingerprint, every
    dataset in every existing database would allocate a new version on its next
    launch and announce a change that never happened."""
    with app.app_context():
        ds = _dataset()
        _image(ds, 'caption', b'AAAA')
        rows = reg.kept_images(ds.id)
        manifest = reg.dataset_manifest(ds.id, rows)
        before = reg.fingerprint_of(manifest, ds.trigger_word, '')
        rec1 = reg.register_launch('local', ds.id, family='krea', source='local')
        rec2 = reg.register_launch('local', ds.id, family='krea', source='local')
        assert rec1.fingerprint == before == rec2.fingerprint
        assert rec1.version == rec2.version == 1     # unchanged dataset keeps v1


def test_snapshot_records_engine_and_dataset_facts(app):
    with app.app_context():
        ds = _dataset(kind='concept', fidelity='body')
        _image(ds, 'caption', b'AAAA', klein_model='nanobanana', source='generated')
        rec = reg.register_launch('local', ds.id, family='krea', source='local')
        snap = run_snapshot.loads(rec)
        entry = next(iter(snap['images'].values()))
        assert entry['e'] == 'nanobanana'
        assert entry['s'] == 'generated'
        assert snap['dataset']['kind'] == 'concept'
        assert snap['dataset']['fidelity'] == 'body'


def test_registration_survives_a_dataset_whose_files_are_gone(app):
    """Provenance is never a gate: an image row whose file vanished must not stop
    a launch, and must not fabricate a hash for it either."""
    with app.app_context():
        ds = _dataset()
        img = FaceDatasetImage(dataset_id=ds.id, status='keep',
                               caption='ghost', filename='missing.png')
        db.session.add(img)
        db.session.commit()
        rec = reg.register_launch('local', ds.id, family='krea', source='local')
        assert rec is not None
        snap = run_snapshot.loads(rec)
        assert run_snapshot.caption_of(snap, img.id)[0] == 'ghost'
        assert run_snapshot.content_of(snap, img.id) is None


# --- comparing two runs -------------------------------------------------------

def _launch(ds, order=0, family='krea'):
    rec = reg.register_launch('local', ds.id, family=family, source='local',
                              steps=1000 + order)
    rec.created_at = _T0 + timedelta(minutes=order)
    db.session.commit()
    return rec


def test_compare_shows_the_caption_text_on_both_sides(app):
    with app.app_context():
        ds = _dataset()
        img = _image(ds, 'a woman in a red coat', b'AAAA')
        a = _launch(ds, 0)
        img.caption = 'a woman in a blue coat'
        db.session.commit()
        b = _launch(ds, 1)
        out = run_compare.compare(a.id, b.id)
        changed = out['images']['caption_changed']
        assert len(changed) == 1
        assert changed[0]['before'] == 'a woman in a red coat'
        assert changed[0]['after'] == 'a woman in a blue coat'
        assert changed[0]['text_recorded'] is True


def test_compare_reports_added_and_removed_images_chronologically(app):
    """A/B are ordered by creation, so 'removed' always means 'gone by the later
    run' no matter which card the user ticked first."""
    with app.app_context():
        ds = _dataset()
        gone = _image(ds, 'to be deleted', b'AAAA')
        gone_id = gone.id
        a = _launch(ds, 0)
        # Added BEFORE the delete on purpose: SQLite hands the freed rowid back
        # to the next insert, and a recycled id would make this test pass for the
        # wrong reason (one id playing both roles).
        fresh = _image(ds, 'brand new', b'CCCC')
        db.session.delete(gone)
        db.session.commit()
        assert fresh.id != gone_id
        b = _launch(ds, 1)
        for pair in ((a.id, b.id), (b.id, a.id)):   # order of the click is irrelevant
            out = run_compare.compare(*pair)
            assert [e['id'] for e in out['images']['removed']] == [gone_id]
            assert [e['id'] for e in out['images']['added']] == [fresh.id]


def test_compare_sees_a_pixel_edit_behind_an_unchanged_id(app):
    """The whole point of the content hash: same id, same caption, new pixels."""
    with app.app_context():
        ds = _dataset()
        img = _image(ds, 'unchanged caption', b'AAAA')
        a = _launch(ds, 0)
        _rewrite(ds, img, b'DIFFERENTBYTES')
        b = _launch(ds, 1)
        out = run_compare.compare(a.id, b.id)
        assert [e['id'] for e in out['images']['content_changed']] == [img.id]
        assert out['images']['caption_changed'] == []
        # Both runs carry a true hash, so the verdict is NOT the approximate one.
        assert not any('size and timestamp' in n for n in out['notes'])


def test_compare_says_so_when_a_run_predates_snapshots(app):
    """A legacy record must produce an explicit statement, never a comparison
    that reads like 'nothing changed'."""
    with app.app_context():
        ds = _dataset()
        _image(ds, 'caption', b'AAAA')
        a = _launch(ds, 0)
        a.snapshot = None                    # exactly what a pre-feature row looks like
        db.session.commit()
        b = _launch(ds, 1)
        out = run_compare.compare(a.id, b.id)
        assert out['a']['predates_snapshot'] is True
        assert out['b']['predates_snapshot'] is False
        assert any('has no full snapshot' in n for n in out['notes'])


def test_a_backup_restore_keeps_the_machine_and_admits_dropping_the_rest(app):
    """A restore allocates FRESH image ids, so caption/content maps keyed on the
    source machine's ids cannot travel — carrying them would attach one run's
    caption to a different picture. The machine is id-free and does travel."""
    with app.app_context():
        ds = _dataset()
        _image(ds, 'a caption that will not survive the trip', b'AAAA')
        rec = reg.register_launch('local', ds.id, family='krea', source='local')
        rec.snapshot = json.dumps({**json.loads(rec.snapshot),
                                   'env': {'torch': '2.9.1+cu128'}})
        db.session.commit()
        carried = run_snapshot.portable(rec.snapshot)
        data = json.loads(carried)
        assert data['env'] == {'torch': '2.9.1+cu128'}
        assert 'captions' not in data and 'images' not in data
        assert data['restored'] is True
        assert run_snapshot.portable(None) is None
        assert run_snapshot.portable('not json') is None


def test_compare_falls_back_to_the_legacy_proxy_and_flags_it(app):
    """One side without a snapshot can still be compared on the old size:mtime
    proxy — but the payload must ADMIT the result is approximate."""
    with app.app_context():
        ds = _dataset()
        img = _image(ds, 'caption', b'AAAA')
        a = _launch(ds, 0)
        a.snapshot = None
        db.session.commit()
        _rewrite(ds, img, b'MUCHLONGERBYTES')
        b = _launch(ds, 1)
        out = run_compare.compare(a.id, b.id)
        assert [e['id'] for e in out['images']['content_changed']] == [img.id]
        assert any('size and timestamp' in n for n in out['notes'])


def test_compare_config_carries_the_record_columns_not_only_the_snapshot(app):
    """steps / base model / masked / dataset version live on the RECORD. They were
    absent from the compared config, so the panel could not show that one run
    trained masked and the other did not."""
    with app.app_context():
        ds = _dataset()
        _image(ds, 'caption', b'AAAA')
        a = reg.register_launch('local', ds.id, family='krea', source='local',
                                steps=1000, masked=True)
        a.created_at = _T0
        db.session.commit()
        b = reg.register_launch('local', ds.id, family='krea', source='local',
                                steps=4000, masked=False)
        b.created_at = _T0 + timedelta(minutes=1)
        db.session.commit()
        cfg = run_compare.compare(a.id, b.id)['config']
        assert cfg['a']['steps'] == 1000 and cfg['b']['steps'] == 4000
        assert cfg['a']['masked'] == 'yes' and cfg['b']['masked'] == 'no'


def test_compare_unknown_run_is_an_error_not_a_crash(app):
    with app.app_context():
        assert run_compare.compare(9999, 9998).get('error')


# --- the image archive --------------------------------------------------------

def test_archive_keeps_a_deleted_image_visible(app):
    """The change named first — 'suppression ou ajout d'image' — is the one
    that used to be unanswerable: once deleted there was nothing left to look at."""
    with app.app_context():
        ds = _dataset()
        img = _image(ds, 'about to be deleted', b'UNIQUEPIXELS')
        rec = reg.register_launch('local', ds.id, family='krea', source='local')
        sig = run_snapshot.content_of(run_snapshot.loads(rec), img.id)
        # The launch archives asynchronously; do it synchronously here.
        from app import config as cfg
        src = str(cfg.dataset_images_root() / str(ds.id) / img.filename)
        run_archive.store([(src, sig, img.filename)])
        path = run_archive.path_for(sig)
        assert path and open(path, 'rb').read() == b'UNIQUEPIXELS'


def test_compare_shows_the_pixels_the_run_SAW_not_the_current_file(app):
    """An image re-cropped after a run is still in the dataset, so serving the
    live file would show pixels that run never trained on — a confident wrong
    answer. The archived blob of the recorded hash wins."""
    with app.app_context():
        from app import config as cfg
        ds = _dataset()
        img = _image(ds, 'caption', b'ORIGINALPIXELS')
        a = _launch(ds, 0)
        _rewrite(ds, img, b'THE PIXELS RUN B TRAINED ON')
        b = _launch(ds, 1)
        src = str(cfg.dataset_images_root() / str(ds.id) / img.filename)
        sig_b = run_snapshot.content_of(run_snapshot.loads(b), img.id)
        run_archive.store([(src, sig_b, img.filename)])   # what the launch does async
        # …and the image is re-cropped AGAIN, after both runs.
        _rewrite(ds, img, b'A THIRD VERSION NOBODY TRAINED ON')

        entry = run_compare.compare(a.id, b.id)['images']['content_changed'][0]
        assert entry['thumb'] == f'/api/dataset/runs/archive/{sig_b}'
        assert open(run_archive.path_for(sig_b), 'rb').read() == b'THE PIXELS RUN B TRAINED ON'


def test_archive_stores_one_blob_per_distinct_content(app):
    """Content-addressed: ten launches on an unchanged dataset add nothing the
    second time. This is what makes keeping the pixels affordable."""
    with app.app_context():
        ds = _dataset()
        img = _image(ds, 'caption', b'SAMEBYTES')
        from app import config as cfg
        src = str(cfg.dataset_images_root() / str(ds.id) / img.filename)
        sig = run_snapshot._content_sig(src)
        assert run_archive.store([(src, sig, img.filename)])['added'] == 1
        again = run_archive.store([(src, sig, img.filename)])
        assert again['added'] == 0 and again['skipped'] == 1


def test_delayed_archive_never_publishes_new_pixels_under_an_old_signature(app):
    with app.app_context():
        from app import config as cfg

        ds = _dataset()
        img = _image(ds, 'caption', b'GENERATION-A')
        src = str(cfg.dataset_images_root() / str(ds.id) / img.filename)
        generation_a = run_snapshot._content_sig(src)
        _rewrite(ds, img, b'GENERATION-B')

        result = run_archive.store([(src, generation_a, img.filename)])

        assert result['added'] == 0
        assert result['skipped'] == 1
        assert run_archive.path_for(generation_a) is None


def test_archive_keeps_a_preserved_bmp_addressable(app, tmp_path):
    """A dataset BMP must not fall through the archive's old PNG fallback: the
    comparison route needs the real extension to serve it as image/bmp later."""
    source = tmp_path / 'master.bmp'
    source.write_bytes(b'BMnot-a-real-bitmap-but-the-archive-must-copy-bytes')
    with app.app_context():
        sig = 'beefcafe'
        result = run_archive.store([(str(source), sig, source.name)])
        archived = run_archive.path_for(sig)
    assert result['added'] == 1
    assert archived and archived.endswith('.bmp')
    assert open(archived, 'rb').read() == source.read_bytes()


def test_archive_stops_at_its_ceiling_instead_of_eating_the_disk(app):
    with app.app_context():
        from app import config as cfg
        cfg.save_config({**cfg.load_config(), 'provenance': {'archive_max_gb': 0}})
        assert run_archive.enabled() is False
        ds = _dataset()
        img = _image(ds, 'caption', b'BYTES')
        src = str(cfg.dataset_images_root() / str(ds.id) / img.filename)
        out = run_archive.store([(src, 'abc123', img.filename)])
        assert out['added'] == 0 and out['full'] is True


def test_archive_rejects_a_traversal_shaped_signature(app):
    """`path_for` is reachable from a URL. A signature is hex; anything else must
    resolve to nothing rather than to a file somewhere else on the disk."""
    with app.app_context():
        assert run_archive.path_for('../../studio') is None
        assert run_archive.path_for('..') is None
        assert run_archive.path_for('') is None


# --- the environment probes ---------------------------------------------------

def test_environment_capture_never_raises_on_a_bare_machine(app):
    """A machine with no ai-toolkit, no CUDA and no GPU must produce an empty-ish
    stamp, not an exception on the launch path."""
    with app.app_context():
        run_environment.clear_cache()
        env = run_environment.capture(base_model=None)
        assert isinstance(env, dict)
        assert 'app' in env                  # the one fact that is always knowable
        run_environment.clear_cache()


def test_torch_version_is_parsed_from_the_venv_file_not_imported(app, tmp_path, monkeypatch):
    """Reading `torch/version.py` is the whole reason the launch does not pay a
    multi-second cold `import torch`."""
    with app.app_context():
        venv = tmp_path / 'ai-toolkit' / 'venv'
        sp = venv / 'Lib' / 'site-packages' / 'torch'
        sp.mkdir(parents=True)
        (sp / 'version.py').write_text(
            "__version__ = '2.9.1+cu128'\ndebug = False\ncuda: str = '12.8'\n",
            encoding='utf-8')
        monkeypatch.setattr(run_environment.cfg, 'aitoolkit_path',
                            lambda kind: venv / 'Scripts' / 'python.exe')
        run_environment.clear_cache()
        assert run_environment.torch_info() == {'torch': '2.9.1+cu128', 'cuda': '12.8'}
        run_environment.clear_cache()


def test_base_model_identity_full_hash_detects_middle_change_with_same_metadata(
        app, tmp_path):
    """A stale size/mtime cache must not hide a middle-byte model mutation."""
    with app.app_context():
        big = tmp_path / 'base.safetensors'
        big.write_bytes(b'A' * (3 << 20))
        run_environment.clear_cache()
        first = run_environment.base_model_identity(str(big))
        assert first['sampled'] is False and first['size'] == (3 << 20)
        assert len(first['sha256']) == 64
        before = big.stat()
        with big.open('r+b') as stream:
            stream.seek(1 << 20)
            stream.write(b'B' * 4096)
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(big, ns=(before.st_atime_ns, before.st_mtime_ns))

        second = run_environment.base_model_identity(str(big))

        assert second['sha256'] != first['sha256']
        run_environment.clear_cache()


def test_base_model_identity_is_none_for_the_official_hosted_base(app):
    with app.app_context():
        run_environment.clear_cache()
        assert run_environment.base_model_identity('') is None
        assert run_environment.base_model_identity(None) is None
        run_environment.clear_cache()


# --- interrupted copies must not eat the ceiling ----------------------------

def test_an_interrupted_copy_neither_counts_nor_survives_the_next_pass(app):
    """`store` copies to `<blob>.part` then renames, so a crash mid-copy leaves an
    orphan nothing can ever address (`path_for` only looks for final names). It
    still counted towards the archive ceiling, so a few interrupted runs quietly
    shrank the usable archive until the user wiped EVERYTHING with "Clear
    archive" — good blobs included."""
    with app.app_context():
        from app import config as cfg
        root = run_archive.archive_root()
        (root / 'ab').mkdir(parents=True, exist_ok=True)
        orphan = root / 'ab' / 'abcdef.png.part'
        orphan.write_bytes(b'X' * 4096)

        # It is not archive content: excluded from the size the ceiling reads.
        assert run_archive.size_bytes(refresh=True) == 0

        src = cfg.dataset_images_root() / 'sweep.png'
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b'REAL PIXELS')
        out = run_archive.store([(str(src), 'abcdef', 'sweep.png')])

        assert out['added'] == 1
        assert not orphan.exists(), 'the next pass sweeps what a crash left behind'
        assert run_archive.path_for('abcdef')
        assert run_archive.size_bytes(refresh=True) == len(b'REAL PIXELS')


def test_the_sweep_never_touches_a_real_blob(app):
    with app.app_context():
        from app import config as cfg
        src = cfg.dataset_images_root() / 'keepme.png'
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b'KEEP')
        run_archive.store([(str(src), 'beef01', 'keepme.png')])
        assert run_archive.sweep_partials() == 0
        assert run_archive.path_for('beef01')
