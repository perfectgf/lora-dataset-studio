"""Concurrency contracts for Bank destinations built in the background.

These tests deliberately exercise the seams that are invisible in ordinary
``TESTING`` mode (where Bank jobs run inline): folder-name allocation, the
multi-Bank registry reservation, and the short interval between persisting a
destination Bank and starting its copy.  A partial destination must never be a
writable, deletable Bank.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

from flask import Flask
from PIL import Image
import pytest


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _runner_app():
    """A job-only app: no DB access, and importantly no inline TESTING runner."""
    runner = Flask('bank-job-reservation-tests')
    runner.config['TESTING'] = False
    return runner


def _seed_bank(app, folder, name, colour=(30, 60, 90)):
    from app.extensions import db
    from app.models import BankImage, ImageBank

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'image.png')
    Image.new('RGB', (8, 8), colour).save(path, format='PNG')
    bank = ImageBank(user_id='local', name=name, source_path=str(folder))
    db.session.add(bank)
    db.session.flush()
    image = BankImage(
        bank_id=bank.id, relpath='image.png', file_size=os.path.getsize(path),
        status='keep',
    )
    db.session.add(image)
    db.session.commit()
    return bank.id, image.id


def test_import_folder_names_are_reserved_atomically_under_concurrency(app):
    """Concurrent same-name exports must never write into the same directory."""
    from app.services import image_bank_service as banks

    workers = 12
    gate = threading.Barrier(workers)

    def allocate():
        gate.wait(timeout=3)
        return banks._import_folder_for('Same / Bank')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(lambda _n: allocate(), range(workers)))

    canonical = [os.path.normcase(os.path.realpath(path)) for path in paths]
    assert len(set(canonical)) == workers
    assert all(os.path.isdir(path) for path in paths)
    assert len({os.path.dirname(path) for path in canonical}) == 1


def test_overlapping_multi_bank_reservations_are_atomic_and_only_guard_their_keys():
    """Two jobs racing for one destination have exactly one winner.

    The source and explicit destination are guarded by the shared reservation;
    an unrelated Bank remains free.  Cancelling through the destination alias is
    allowed and reaches the same shared job.
    """
    from app.services import bank_jobs

    runner = _runner_app()
    shared_destination = 9003
    release = threading.Event()
    entered = threading.Event()
    start_gate = threading.Barrier(3)

    def hold(_job):
        entered.set()
        # A SAFETY NET, not the mechanism: the test's own `finally` sets `release`,
        # so a passing run never waits here at all. It was 3 s, which made the
        # reservation expire in the MIDDLE of the test body — a dozen HTTP round
        # trips happen before the confirmed relocate, and on a loaded CI runner
        # they take longer than that. The lock then quietly lapsed and the write
        # this test exists to see REFUSED came back 200/applied. A timeout that
        # decides the verdict is a clock, not an assertion.
        release.wait(timeout=120)

    def attempt(primary):
        start_gate.wait(timeout=3)
        try:
            bank_jobs.start(
                runner, primary, f'copy-{primary}', hold,
                reserve_ids=(shared_destination,),
            )
            return ('started', primary)
        except bank_jobs.BankJobBusy as exc:
            return ('busy', primary, exc.kind)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(attempt, primary) for primary in (9001, 9002)]
            start_gate.wait(timeout=3)
            outcomes = [future.result(timeout=3) for future in futures]

        winners = [item for item in outcomes if item[0] == 'started']
        losers = [item for item in outcomes if item[0] == 'busy']
        assert len(winners) == len(losers) == 1
        winner = winners[0][1]
        loser = losers[0][1]
        assert entered.wait(timeout=3)
        assert bank_jobs.running(winner)
        assert bank_jobs.running(shared_destination)
        assert not bank_jobs.running(loser)

        unrelated_done = threading.Event()
        bank_jobs.start(
            runner, 9004, 'unrelated', lambda _job: unrelated_done.set())
        assert unrelated_done.wait(timeout=3)

        assert bank_jobs.cancel(shared_destination) is True
        assert bank_jobs.get(winner)['cancelled'] is True
        assert bank_jobs.get(shared_destination)['cancelled'] is True
    finally:
        release.set()
        _wait_until(lambda: not any(
            bank_jobs.running(bank_id)
            for bank_id in (9001, 9002, shared_destination, 9004)
        ))


def test_explicit_reservation_is_adopted_only_by_object_identity(app):
    """A copied/lookalike token cannot launch or steal a staged destination."""
    from app.services import bank_jobs

    reservation = bank_jobs.reserve(
        9051, 'copy', total=1, reserve_ids=(9052, 9053))
    impostor = dict(reservation)

    with pytest.raises(RuntimeError, match='missing, stale, or belongs'):
        bank_jobs.start(
            app, 9051, 'copy', lambda _job: None, total=1,
            reserve_ids=(9052, 9053), reservation=impostor)

    assert all(bank_jobs._jobs[bank_id] is reservation
               for bank_id in (9051, 9052, 9053))
    assert bank_jobs.launched(reservation) is False

    adopted = bank_jobs.start(
        app, 9051, 'copy', lambda _job: None, total=1,
        reserve_ids=(9052, 9053), reservation=reservation)
    assert adopted is reservation
    assert all(bank_jobs.get(bank_id)['finished'] is True
               for bank_id in (9051, 9052, 9053))


@pytest.mark.parametrize('start_kind,reserve_ids', [
    ('other-kind', (9062,)),
    ('copy', (9062, 9063)),
    ('copy', ()),
])
def test_explicit_reservation_rejects_kind_or_key_mismatch(
        app, start_kind, reserve_ids):
    from app.services import bank_jobs

    reservation = bank_jobs.reserve(9061, 'copy', reserve_ids=(9062,))
    with pytest.raises(RuntimeError, match='missing, stale, or belongs'):
        bank_jobs.start(
            app, 9061, start_kind, lambda _job: None,
            reserve_ids=reserve_ids, reservation=reservation)

    assert bank_jobs._jobs[9061] is reservation
    assert bank_jobs._jobs[9062] is reservation
    assert bank_jobs.launched(reservation) is False


def test_explicit_reservation_cannot_revive_an_expired_job(app):
    from app.services import bank_jobs

    reservation = bank_jobs.reserve(9071, 'copy', reserve_ids=(9072,))
    reservation['_touched'] = time.time() - bank_jobs._STALE_TTL - 1

    with pytest.raises(RuntimeError, match='missing, stale, or belongs'):
        bank_jobs.start(
            app, 9071, 'copy', lambda _job: None,
            reserve_ids=(9072,), reservation=reservation)

    assert bank_jobs.get(9071) is None
    assert bank_jobs.get(9072) is None


def test_thread_launch_abort_releases_every_reserved_bank(monkeypatch):
    """A thread-construction failure must not leave a phantom live alias."""
    from app.services import bank_jobs

    class BrokenThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError('thread launch failed')

    monkeypatch.setattr(bank_jobs.threading, 'Thread', BrokenThread)
    reservation = bank_jobs.reserve(
        9101, 'copy', reserve_ids=(9102, 9103))
    with pytest.raises(RuntimeError, match='thread launch failed'):
        bank_jobs.start(
            _runner_app(), 9101, 'copy', lambda _job: None,
            reserve_ids=(9102, 9103), reservation=reservation,
        )

    assert all(bank_jobs.get(bank_id) is None for bank_id in (9101, 9102, 9103))
    assert not any(bank_id in bank_jobs._jobs for bank_id in (9101, 9102, 9103))


@pytest.mark.parametrize('finished,ttl_name', [
    (False, '_STALE_TTL'),
    (True, '_FINISHED_TTL'),
])
def test_expiring_one_alias_purges_the_whole_multi_bank_reservation(
        app, finished, ttl_name):
    """No stale source/destination alias may survive its shared job's expiry."""
    from app.services import bank_jobs

    job = bank_jobs.start(
        app, 9201, 'copy', lambda _job: None,
        reserve_ids=(9202, 9203),
    )
    job['finished'] = finished
    job['_touched'] = time.time() - getattr(bank_jobs, ttl_name) - 1

    assert bank_jobs.get(9202) is None
    assert all(bank_jobs.get(bank_id) is None for bank_id in (9201, 9202, 9203))
    assert not any(bank_id in bank_jobs._jobs for bank_id in (9201, 9202, 9203))


def test_cancellation_fence_requires_every_alias_but_keeps_legacy_jobs_usable():
    """Losing one destination capability stops a modern multi-Bank worker.

    Plain pre-reservation job mappings remain supported for synchronous service
    callers; they have no registry capability and therefore follow their flag.
    """
    from app.services import bank_jobs

    reservation = bank_jobs.reserve(9251, 'copy', reserve_ids=(9252, 9253))
    assert bank_jobs.cancelled(reservation) is False
    with bank_jobs._lock:
        bank_jobs._jobs.pop(9252)
    assert bank_jobs.cancelled(reservation) is True
    bank_jobs.abort(reservation)

    legacy = {'cancelled': False}
    assert bank_jobs.cancelled(legacy) is False
    legacy['cancelled'] = True
    assert bank_jobs.cancelled(legacy) is True


def test_synchronous_mutation_lease_reenters_only_with_exact_job_capability():
    """Pipeline helpers reuse their job slot; lookalike tokens fail closed."""
    from app.services import bank_jobs

    reservation = bank_jobs.reserve(9261, 'pipeline')
    try:
        with bank_jobs.mutation_lease(
                9261, 'apply_flags', capability=reservation) as adopted:
            assert adopted is reservation
            assert bank_jobs.get(9261)['kind'] == 'pipeline'

        with pytest.raises(RuntimeError, match='missing, stale, or belongs'):
            with bank_jobs.mutation_lease(
                    9261, 'apply_flags', capability=dict(reservation)):
                pass
    finally:
        bank_jobs.abort(reservation)


def test_synchronous_lease_restores_finished_ui_snapshot_one_alias_at_a_time(app):
    """A GET inventory fence must not erase the pass result it is about to show."""
    from app.services import bank_jobs

    bank_jobs.start(
        app, 9265, 'score', lambda job: bank_jobs.progress(
            job, detail='done — 4 scored'), reserve_ids=(9266,))
    assert bank_jobs.get(9265)['finished'] is True

    with bank_jobs.mutation_lease(9265, 'folder_sync'):
        assert bank_jobs.get(9265)['kind'] == 'folder_sync'
        # Only the leased alias is displaced; the completed destination/source
        # peer keeps its useful snapshot throughout.
        assert bank_jobs.get(9266)['kind'] == 'score'

    restored = bank_jobs.get(9265)
    assert restored['kind'] == 'score'
    assert restored['detail'] == 'done — 4 scored'
    assert restored['finished'] is True


def test_rotate_losing_race_after_before_request_is_still_a_409(
        app, client, tmp_path, monkeypatch):
    """The blueprint check is advisory; the service lease is the real fence.

    Install a promotion reservation *inside* the guard's first ``running`` read
    while returning the stale ``False`` it observed.  This deterministically
    recreates the formerly unsafe interleaving: the rotate request was admitted,
    but promotion acquired the Bank before ``rotate_images`` reached its write.
    """
    from app.services import bank_jobs

    with app.app_context():
        from app.extensions import db
        from app.models import BankImage

        bank_id, image_id = _seed_bank(
            app, tmp_path / 'rotate-race', 'Rotate race')
        original_running = bank_jobs.running
        captured = {}

        def admit_then_promote(candidate_id):
            if candidate_id == bank_id and 'reservation' not in captured:
                captured['reservation'] = bank_jobs.reserve(
                    bank_id, 'bank_promote')
                return False
            return original_running(candidate_id)

        monkeypatch.setattr(bank_jobs, 'running', admit_then_promote)
        try:
            response = client.post(
                f'/api/bank/{bank_id}/rotate',
                json={'ids': [image_id], 'degrees': 90})
        finally:
            bank_jobs.abort(captured.get('reservation'))

        assert response.status_code == 409, response.get_json()
        assert response.get_json()['busy_kind'] == 'bank_promote'
        db.session.expire_all()
        assert db.session.get(BankImage, image_id).rotation is None


def test_service_mutators_cannot_bypass_an_existing_promotion_reservation(
        app, tmp_path):
    """Every synchronous metadata/pixel lane shares the same atomic slot."""
    with app.app_context():
        from app.services import bank_jobs, folder_person
        from app.services import image_bank_service as banks

        bank_id, image_id = _seed_bank(
            app, tmp_path / 'mutator-fence', 'Mutation fence')
        reservation = bank_jobs.reserve(bank_id, 'bank_promote')
        calls = (
            lambda: banks.set_status('local', bank_id, [image_id], 'reject'),
            lambda: banks.rotate_images('local', bank_id, [image_id], 90),
            lambda: banks.apply_flags('local', bank_id, ['blur']),
            lambda: banks.resolve_dups('local', bank_id),
            lambda: banks.undo_last('local', bank_id),
            lambda: banks.set_watermark_regions(
                'local', bank_id, image_id, None),
            lambda: banks.undo_watermark_clean('local', bank_id, [image_id]),
            lambda: banks.dismiss_watermarks('local', bank_id, [image_id]),
            lambda: folder_person.assert_single_person('local', bank_id, ''),
            lambda: folder_person.revoke('local', bank_id, ''),
            lambda: folder_person.accept_suggestions('local', bank_id, ['']),
        )
        try:
            for call in calls:
                with pytest.raises(bank_jobs.BankJobBusy) as refused:
                    call()
                assert refused.value.kind == 'bank_promote'
        finally:
            bank_jobs.abort(reservation)


def test_bank_to_bank_destination_is_never_listed_before_it_is_reserved(
        app, tmp_path, monkeypatch):
    """The first observable destination generation already owns a job guard.

    The commit spy makes the formerly sub-millisecond create/launch race fully
    deterministic.  An implementation may either keep a building row hidden or
    expose it with its reservation; both satisfy this externally visible rule.
    """
    with app.app_context():
        from app.extensions import db
        from app.services import bank_jobs
        from app.services import image_bank_service as banks

        source_id, image_id = _seed_bank(
            app, tmp_path / 'source', 'Source', colour=(10, 80, 140))
        real_commit = db.session.commit
        visible_checks = []

        def commit_then_observe():
            real_commit()
            destinations = [
                row for row in banks.list_banks('local')
                if row['name'] == 'Building destination'
            ]
            for destination in destinations:
                visible_checks.append(bank_jobs.get(destination['id']) is not None)

        monkeypatch.setattr(db.session, 'commit', commit_then_observe)
        destination_id = banks.start_bank_promote(
            app, 'local', source_id, [image_id], 'Building destination')

        assert destination_id != source_id
        assert visible_checks, 'the completed destination never became list-visible'
        assert all(visible_checks), (
            'a partially built destination became visible before its guard existed')


def test_reserved_destination_allows_reads_and_cancel_but_refuses_writes_and_delete(
        app, client, tmp_path, monkeypatch):
    """A Bank being built is inspectable, never mutable or deletable."""
    from app.models import BankImage
    from app.services import bank_jobs

    with app.app_context():
        source_id, _source_image = _seed_bank(
            app, tmp_path / 'source', 'Source', colour=(1, 2, 3))
        destination_id, destination_image = _seed_bank(
            app, tmp_path / 'destination', 'Destination', colour=(4, 5, 6))
        other_id, other_image = _seed_bank(
            app, tmp_path / 'other', 'Other', colour=(7, 8, 9))

    release = threading.Event()
    entered = threading.Event()

    def hold(_job):
        entered.set()
        # A SAFETY NET, not the mechanism: the test's own `finally` sets `release`,
        # so a passing run never waits here at all. It was 3 s, which made the
        # reservation expire in the MIDDLE of the test body — a dozen HTTP round
        # trips happen before the confirmed relocate, and on a loaded CI runner
        # they take longer than that. The lock then quietly lapsed and the write
        # this test exists to see REFUSED came back 200/applied. A timeout that
        # decides the verdict is a clock, not an assertion.
        release.wait(timeout=120)

    bank_jobs.start(
        _runner_app(), source_id, 'bank_promote', hold,
        reserve_ids=(destination_id,),
    )
    assert entered.wait(timeout=3)
    try:
        from app.services import image_bank_service as banks
        monkeypatch.setattr(
            banks, 'select_diverse',
            lambda *_args, **_kwargs: {'image_ids': [], 'pool': 0,
                                       'requested': 10, 'typicality': 0.5})
        monkeypatch.setattr(
            banks, 'select_balanced',
            lambda *_args, **_kwargs: {'image_ids': [], 'pool': 0,
                                       'requested': 10, 'typicality': 0.5})
        monkeypatch.setattr(
            banks, 'select_similar',
            lambda *_args, **_kwargs: {'image_ids': [], 'pool': 0,
                                       'requested': 10})
        monkeypatch.setattr(
            banks, 'search_by_text',
            lambda *_args, **_kwargs: {'image_ids': [], 'pool': 0,
                                       'requested': 10})
        # Read-only workspace/list endpoints and the cooperative Stop endpoint
        # stay usable while the destination is reserved.
        assert client.get(f'/api/bank/{destination_id}').status_code == 200
        assert client.get(f'/api/bank/{destination_id}/images').status_code == 200

        # These POSTs are queries, not writes. They may answer 400 when this
        # deliberately unscored fixture cannot satisfy the query, but the Bank
        # reservation itself must not reject them with 409.
        for suffix, body in (
            ('select-diverse', {'n': 10}),
            ('select-balanced', {'n': 10}),
            ('select-similar', {'ref_id': destination_image, 'n': 10}),
            ('search-text', {'query': 'portrait', 'n': 10}),
        ):
            read_query = client.post(
                f'/api/bank/{destination_id}/{suffix}', json=body)
            assert read_query.status_code != 409, (
                suffix, read_query.status_code, read_query.get_json())

        relocate_preview = client.post(
            f'/api/bank/{destination_id}/relocate',
            json={'folder': str(tmp_path / 'destination'), 'confirm': False},
        )
        assert relocate_preview.status_code == 200, relocate_preview.get_json()
        assert relocate_preview.get_json()['applied'] is False

        relocate_confirm = client.post(
            f'/api/bank/{destination_id}/relocate',
            json={'folder': str(tmp_path / 'destination'), 'confirm': True},
        )
        assert relocate_confirm.status_code == 409, relocate_confirm.get_json()
        assert relocate_confirm.get_json()['busy_kind'] == 'bank_promote'

        # The reservation is scoped: an unrelated Bank remains writable.
        unrelated = client.post(
            f'/api/bank/{other_id}/images/status',
            json={'ids': [other_image], 'status': 'reject'},
        )
        assert unrelated.status_code == 200, unrelated.get_json()

        write = client.post(
            f'/api/bank/{destination_id}/images/status',
            json={'ids': [destination_image], 'status': 'reject'},
        )
        assert write.status_code == 409, write.get_json()
        assert write.get_json()['busy_kind'] == 'bank_promote'

        delete = client.delete(f'/api/bank/{destination_id}')
        assert delete.status_code == 409, delete.get_json()
        assert delete.get_json()['busy_kind'] == 'bank_promote'

        cancelled = client.post(f'/api/bank/{destination_id}/cancel')
        assert cancelled.status_code == 200
        assert cancelled.get_json() == {'ok': True}
        assert client.get(f'/api/bank/{destination_id}').status_code == 200

        with app.app_context():
            assert BankImage.query.filter_by(id=destination_image).one().status == 'keep'
    finally:
        release.set()
        assert _wait_until(lambda: not bank_jobs.running(source_id))


def test_bank_dataset_id_routes_normalize_strings_and_reject_coercions(
        client, monkeypatch):
    """Every Bank route that addresses a Dataset shares the strict id contract."""
    from app.services import image_bank_service as banks

    promoted = []
    counted = []

    def start_promote(_app, _user, _bank, ids, dataset_id):
        banks._normalize_promotion_ids(ids)
        promoted.append(dataset_id)

    monkeypatch.setattr(
        banks, 'start_promote', start_promote)
    monkeypatch.setattr(
        banks, 'promotable_count',
        lambda _user, _bank, dataset_id: counted.append(dataset_id) or 4)

    promote = client.post('/api/bank/123/promote', json={
        'dataset_id': '12', 'image_ids': [],
    })
    assert promote.status_code == 202, promote.get_json()
    assert promoted == [12]

    promotable = client.get('/api/bank/123/promotable?dataset_id=12')
    assert promotable.status_code == 200, promotable.get_json()
    assert promotable.get_json() == {'count': 4}
    assert counted == [12]
    for invalid in (True, False, 1.0, '1.0', None):
        response = client.post('/api/bank/123/promote', json={
            'dataset_id': invalid, 'image_ids': [],
        })
        assert response.status_code == 400, (invalid, response.get_json())
    assert promoted == [12]

    for invalid_ids in ('', False, 0, {}):
        response = client.post('/api/bank/123/promote', json={
            'dataset_id': 12, 'image_ids': invalid_ids,
        })
        assert response.status_code == 400, (
            invalid_ids, response.get_json())
    assert client.post('/api/bank/123/promote', json=[]).status_code == 400
    assert promoted == [12]

    for invalid in ('1.0', '', str(1 << 63)):
        response = client.get(
            f'/api/bank/123/promotable?dataset_id={invalid}')
        assert response.status_code == 400, (invalid, response.get_json())
    assert counted == [12]


def test_bank_to_dataset_activity_conflict_is_a_409(client, monkeypatch):
    from app.services import dataset_activity
    from app.services import image_bank_service as banks

    def busy(*_args, **_kwargs):
        raise dataset_activity.DatasetActivityBusy(
            'This dataset already has work in progress')

    monkeypatch.setattr(banks, 'start_promote', busy)
    response = client.post('/api/bank/123/promote', json={
        'dataset_id': 12, 'image_ids': [1],
    })
    assert response.status_code == 409
    assert response.get_json()['busy_kind'] == 'bank_import'
