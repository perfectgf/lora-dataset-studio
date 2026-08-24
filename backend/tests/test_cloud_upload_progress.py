"""The phase that cost two hours of a rented pod without reporting one byte.

Incident (run #138, 2026-08-02). A dense Krea run staged a 24 GB dataset —
12 422 files — and entered 'uploading'. `upload_dataset` was a single blocking
call that wrote nothing, so:

  * the run card said 'Uploading the dataset' and nothing else for 2 h 07;
  * every watchdog in cloud_training judges a run on database/file evidence,
    and this phase produced none, so the only limit that applied was the blind
    two-hour floor every 'silent by design' phase gets;
  * two app restarts re-adopted the run and restarted the upload from file 0,
    which no reading could reveal either;
  * the owner cancelled by hand at 2 h 07. The pod had billed ~2.73 $ and had
    never reached step 1.

The fix is evidence, not a shorter timer: the transfer reports bytes, the
progress clock is fed from them, and the phase is judged on 'nothing arrived'
instead of 'nobody wrote a row'. A slow upload must stay untouched — that is
the second half of every test below.

Everything here is offline: no pod, no network, no thread, no dollar.
"""
import json
from app.utils.timestamps import naive_utcnow
import os
from datetime import timedelta

import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    yield cloud_training


def _mkrun(ct, tmp_path, **kw):
    staging = tmp_path / f"run_{kw.get('job_name', 'j1')}"
    staging.mkdir(parents=True, exist_ok=True)
    fields = dict(dataset_id=1, status='uploading', vast_instance_id='90001',
                  vast_label='lds-1', job_name='j1', price_per_hour=1.29,
                  staging_dir=str(staging))
    fields.update(kw)
    run = ct.CloudTrainingRun(**fields)
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


def _dataset(tmp_path, pairs=20, image_bytes=1000):
    """A staged dataset folder shaped like the real one: <n>.png + <n>.txt."""
    folder = tmp_path / 'dataset'
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(pairs):
        (folder / f'img_{i:03d}.png').write_bytes(b'x' * image_bytes)
        (folder / f'img_{i:03d}.txt').write_text('a caption', encoding='utf-8')
    return str(folder)


class _OkResponse:
    status_code = 200
    text = 'ok'


def _remote(monkeypatch, response=None):
    """A RemoteAiToolkit whose HTTP layer is stubbed, recording each POST."""
    from app.services.aitoolkit_remote import RemoteAiToolkit
    remote = RemoteAiToolkit('http://pod.invalid:1234', 'tok')
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return response() if response else _OkResponse()

    monkeypatch.setattr(remote, '_request', fake_request)
    return remote, calls


# ------------------------------------------- 1. the transfer reports bytes --

def test_upload_dataset_reports_files_and_bytes_as_they_land(tmp_path, monkeypatch):
    """The reading the incident lacked entirely. 40 files (20 pairs) in batches
    of 8 = 5 POSTs, and the caller must be able to watch them go."""
    folder = _dataset(tmp_path, pairs=20, image_bytes=1000)
    remote, calls = _remote(monkeypatch)
    seen = []
    total = remote.upload_dataset('ds', folder,
                                  on_progress=lambda *a: seen.append(a))

    assert total == 40
    assert len(calls) == 5
    # One opening report (so the card shows 0/40 immediately) + one per batch.
    assert len(seen) == 6
    assert seen[0] == (0, 40, 0, 20 * (1000 + len('a caption')))
    assert seen[-1][0] == 40 and seen[-1][1] == 40
    assert seen[-1][2] == seen[-1][3] > 0
    # Strictly increasing: a counter that can go backwards would read as
    # progress to the fingerprint and defeat the stall detection.
    assert [s[2] for s in seen] == sorted(s[2] for s in seen)
    assert len(set(s[2] for s in seen)) == 6


def test_a_raising_progress_callback_never_breaks_the_upload(tmp_path, monkeypatch):
    """Same contract as the download side. The callback exists to describe the
    transfer; it must never be able to cost a rented pod."""
    folder = _dataset(tmp_path, pairs=12)
    remote, calls = _remote(monkeypatch)
    hits = []

    def explode(*args):
        hits.append(args)
        raise RuntimeError('database is on fire')

    assert remote.upload_dataset('ds', folder, on_progress=explode) == 24
    assert len(calls) == 3          # every batch still went up
    assert len(hits) == 1           # ... and the callback was disabled at once


def test_upload_without_a_callback_still_works(tmp_path, monkeypatch):
    """The parameter is optional: seed_checkpoint and any older caller keep
    calling this with two arguments."""
    folder = _dataset(tmp_path, pairs=4)
    remote, calls = _remote(monkeypatch)
    assert remote.upload_dataset('ds', folder) == 8
    assert len(calls) == 1


def test_an_unreadable_file_is_counted_as_a_file_worth_zero_bytes(tmp_path, monkeypatch):
    """Sizing the folder must not be able to fail the upload."""
    from app.services import aitoolkit_remote
    folder = _dataset(tmp_path, pairs=2)
    real_getsize = os.path.getsize

    def flaky(path):
        if path.endswith('img_000.png'):
            raise OSError('gone')
        return real_getsize(path)

    monkeypatch.setattr(aitoolkit_remote.os.path, 'getsize', flaky)
    remote, _ = _remote(monkeypatch)
    seen = []
    assert remote.upload_dataset('ds', folder,
                                 on_progress=lambda *a: seen.append(a)) == 4
    assert seen[0][1] == 4          # the file is still counted


# ------------------------------------------------- 2. the durable evidence --

def test_the_heartbeat_records_every_batch_and_throttles_the_sentence(
        ct, app, tmp_path, monkeypatch):
    """Two different jobs, deliberately not throttled the same way: the byte
    record feeds the watchdog and is written every time, the phase sentence is
    cosmetic and is not."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        clock = {'t': 1000.0}
        monkeypatch.setattr(ct, '_now', lambda: clock['t'])
        beat = ct._upload_heartbeat(run, 'Uploading the dataset')

        beat(8, 400, 8_000_000, 400_000_000)
        assert ct._read_upload_bytes(run) == 8_000_000
        first = run.phase_detail

        clock['t'] += 1                       # inside the throttle window
        beat(16, 400, 16_000_000, 400_000_000)
        assert ct._read_upload_bytes(run) == 16_000_000     # recorded anyway
        ct.db.session.refresh(run)
        assert run.phase_detail == first                    # sentence unchanged

        clock['t'] += ct._UPLOAD_HEARTBEAT_SECONDS + 1
        beat(24, 400, 24_000_000, 400_000_000)
        ct.db.session.refresh(run)
        assert run.phase_detail != first
        assert '24/400 files' in run.phase_detail
        assert '400 MB' in run.phase_detail


def test_the_heartbeat_sentence_keeps_the_launch_step_on_upload(ct, app, tmp_path):
    """_active_launch_step reads phase_detail: only a sentence starting with
    'Job ' may advance the checklist to 'Starting the training job'. The
    progress text must not move the checklist by accident."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        ct._upload_heartbeat(run, 'Uploading the dataset')(8, 400, 8, 400)
        ct.db.session.refresh(run)
        assert ct._active_launch_step(run.status, run.phase_detail) == 'upload'


def test_a_write_locked_database_skips_the_sentence_without_failing_the_run(
        ct, app, tmp_path, monkeypatch):
    """The run #137 lesson (a539370), applied to this heartbeat: a local write
    lock may cost a cosmetic refresh, never a paid run. The durable byte record
    is a plain file and is unaffected."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)

        def locked(*a, **k):
            raise RuntimeError('(sqlite3.OperationalError) database is locked')

        monkeypatch.setattr(ct, '_set', locked)
        ct._upload_heartbeat(run, 'Uploading the dataset')(8, 400, 8_000, 400_000)
        assert ct._read_upload_bytes(run) == 8_000


def test_a_hard_database_failure_never_stops_the_byte_clock(
        ct, app, tmp_path, monkeypatch):
    """The trap this fix could have walked into. The transfer driver disables a
    callback that raises — correct for the transfer, fatal here: losing the
    callback would also stop the byte clock, and the upload watchdog would then
    kill a perfectly healthy upload for going quiet. Describing the transfer
    may fail; recording it may not."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)

        def boom(*a, **k):
            raise RuntimeError('disk full, not a lock')

        monkeypatch.setattr(ct, '_set', boom)
        beat = ct._upload_heartbeat(run, 'Uploading the dataset')
        beat(8, 400, 8_000, 400_000)            # must not raise
        beat(16, 400, 16_000, 400_000)
        assert ct._read_upload_bytes(run) == 16_000


def test_a_run_without_a_staging_dir_writes_nothing_anywhere(ct, app, tmp_path, monkeypatch):
    """The record is addressed by staging_dir; without one, a relative path
    would drop the file in the process's working directory, where every such
    run would then read one run's bytes as its own."""
    monkeypatch.chdir(tmp_path)
    with app.app_context():
        run = _mkrun(ct, tmp_path, job_name='nodir')
        run.staging_dir = None
        ct.db.session.commit()
        ct._upload_heartbeat(run, 'Uploading the dataset')(8, 400, 8_000, 400_000)
        assert ct._read_upload_bytes(run) is None
        assert not os.path.exists(tmp_path / 'upload_progress.json')


def test_upload_bytes_move_the_progress_fingerprint(ct, app, tmp_path):
    """The whole point: 'the upload is alive' becomes an observable fact."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        ct._write_upload_progress(run, 8, 400, 8_000_000, 400_000_000)
        frozen = ct._progress_fingerprint(run)
        assert ct._progress_fingerprint(run) == frozen      # nothing happened
        ct._write_upload_progress(run, 16, 400, 16_000_000, 400_000_000)
        assert ct._progress_fingerprint(run) != frozen      # bytes arrived


def test_a_missing_or_corrupt_record_is_not_an_error(ct, app, tmp_path):
    """Read on every supervisor tick and every card render: it may never
    raise, and a run that has not started uploading has no record at all."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        assert ct._read_upload_bytes(run) is None
        with open(os.path.join(run.staging_dir, 'upload_progress.json'), 'w',
                  encoding='utf-8') as fh:
            fh.write('{not json')
        assert ct._read_upload_bytes(run) is None
        assert ct._progress_fingerprint(run)                # still answers


# --------------------------------------------------- 3. the stall watchdog --

def test_uploading_no_longer_gets_the_blind_two_hour_floor(ct, app, tmp_path):
    """The exact number that let run #138 bill for two hours: 'uploading' was
    scored with _SILENT_PHASE_FREEZE_SECONDS like every other quiet phase."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        assert ct._freeze_limit_seconds(run, {}) == ct._UPLOAD_STALL_MINUTES * 60
        assert ct._freeze_limit_seconds(run, {}) < ct._SILENT_PHASE_FREEZE_SECONDS
        # every other silent phase is untouched
        run.status = 'provisioning'
        assert ct._freeze_limit_seconds(run, {}) == ct._SILENT_PHASE_FREEZE_SECONDS


def test_the_upload_limit_is_configurable_and_respects_the_master_switch(ct, app, tmp_path):
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        assert ct._freeze_limit_seconds(run, {'upload_stall_minutes': 40}) == 2400
        assert ct._freeze_limit_seconds(run, {'upload_stall_minutes': 0}) == 0
        # Turning the freeze watchdog off turns this off too: it is a
        # tightening of that watchdog, never a way around the user's choice.
        assert ct._freeze_limit_seconds(
            run, {'freeze_watchdog_minutes': 0, 'upload_stall_minutes': 25}) == 0


def test_a_stalled_upload_destroys_the_pod_with_its_own_message(
        ct, app, tmp_path, monkeypatch):
    """THE incident scenario, replayed: an 'uploading' run whose byte counter
    has not moved. The pod must be terminated — the billing is the damage —
    and the error must say the DATASET never arrived, not that the run froze."""
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(str(iid)) or True)
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        ct._write_upload_progress(run, 8, 12422, 25_000_000, 24_000_000_000)
        ct.note_progress(run, naive_utcnow() - timedelta(minutes=40))
        ct._set(run, phase_detail='Uploading the dataset — 8/12422 files')

        acted = ct.supervise_active_runs()

        assert destroyed == ['90001']
        assert [a['reason'] for a in acted] == ['upload_stall']
        ct.db.session.refresh(run)
        assert run.status == 'stopped'
        assert run.error == 'upload stall watchdog'
        assert 'upload stalled' in (run.phase_detail or '').lower()


def test_a_slow_but_moving_upload_is_left_alone(ct, app, tmp_path, monkeypatch):
    """The half that matters just as much. 24 GB over a home connection can
    run for hours; as long as bytes keep arriving the pod is doing exactly what
    it was rented for and must not be touched — not at 25 min, not at 3 h."""
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(str(iid)) or True)
    with app.app_context():
        run = _mkrun(ct, tmp_path,
                     created_at=naive_utcnow() - timedelta(hours=3))
        ct._write_upload_progress(run, 8, 12422, 25_000_000, 24_000_000_000)
        ct.note_progress(run, naive_utcnow() - timedelta(hours=3))
        # ... and then a batch lands, three hours in.
        ct._write_upload_progress(run, 16, 12422, 50_000_000, 24_000_000_000)

        assert ct.supervise_active_runs() == []
        assert destroyed == []
        ct.db.session.refresh(run)
        assert run.status == 'uploading'


def test_a_restart_does_not_kill_the_run_on_its_predecessors_byte_count(
        ct, app, tmp_path, monkeypatch):
    """A re-adopted run restarts its upload from file 0, so the byte counter
    drops. That is a CHANGE, and the clock treats every change as progress —
    the same direction of error the fingerprint chooses everywhere else."""
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(str(iid)) or True)
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        ct._write_upload_progress(run, 900, 12422, 2_000_000_000, 24_000_000_000)
        ct.note_progress(run, naive_utcnow() - timedelta(minutes=40))
        ct._write_upload_progress(run, 0, 12422, 0, 24_000_000_000)  # restarted

        assert ct.supervise_active_runs() == []
        assert destroyed == []


# ------------------------------------------------------- 4. the launch card --

def test_launch_view_publishes_the_upload_deadline(ct, app, tmp_path):
    """The card can only announce a limit the backend hands it — that is what
    turns a long upload from 'it hung' into 'it has room as long as it moves'."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        view = ct.launch_view(run, cloud_cfg={})
        assert view['active_step'] == 'upload'
        assert view['upload_stall_limit_seconds'] == ct._UPLOAD_STALL_MINUTES * 60
        booting = _mkrun(ct, tmp_path, job_name='j2', status='provisioning')
        assert ct.launch_view(booting, cloud_cfg={})['upload_stall_limit_seconds'] == 0


def test_the_run_payload_exposes_the_shorter_upload_limit(ct, app, tmp_path):
    """`idle_limit_seconds` is what the card warns from; it must carry the
    upload's own limit while the run is uploading, not the two-hour floor."""
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        payload = ct._run_payload(run)
        assert payload['idle_limit_seconds'] == ct._UPLOAD_STALL_MINUTES * 60


# -------------------------------------------------- 5. the monitor call site --

def test_the_monitor_hands_the_heartbeat_to_both_uploads(ct, app, tmp_path, monkeypatch):
    """Wiring check: the dataset AND the masks folder must both report. A
    masked run that goes silent for its mask upload is the same incident."""
    import inspect
    src = inspect.getsource(ct._monitor)
    assert src.count('_upload_heartbeat(run,') == 2
    assert 'Uploading the masks' in src

    # ... and the record a heartbeat writes is readable through the seam the
    # supervisor uses, whichever of the two wrote it.
    with app.app_context():
        run = _mkrun(ct, tmp_path)
        ct._upload_heartbeat(run, 'Uploading the masks')(3, 3, 900, 900)
        assert ct._read_upload_bytes(run) == 900
        with open(os.path.join(run.staging_dir, 'upload_progress.json'),
                  encoding='utf-8') as fh:
            assert json.load(fh)['files_total'] == 3
