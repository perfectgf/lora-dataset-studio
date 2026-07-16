import threading
from types import SimpleNamespace

import pytest


class _CoordinatedQueue:
    """Force deux lecteurs à prendre le même snapshot sans verrou externe."""

    def __init__(self, items=()):
        self.items = [dict(item) for item in items]
        self.first_read = threading.Event()
        self.second_read = threading.Event()
        self._read_count = 0
        self._count_lock = threading.Lock()

    def get(self):
        with self._count_lock:
            self._read_count += 1
            read_number = self._read_count
            snapshot = [dict(item) for item in self.items]
        if read_number == 1:
            self.first_read.set()
            # Sans le verrou de production, le second lecteur entre et libère
            # immédiatement celui-ci. Avec le correctif, il attend la fin de la
            # transaction et ce court timeout crée deux snapshots successifs.
            self.second_read.wait(timeout=0.25)
        elif read_number == 2:
            self.second_read.set()
        return snapshot

    def save(self, items):
        self.items = [dict(item) for item in items]


def _run_threads(first, second, first_read):
    errors = []

    def guarded(call):
        try:
            call()
        except BaseException as exc:  # surfaced in the pytest thread
            errors.append(exc)

    thread_a = threading.Thread(target=guarded, args=(first,))
    thread_b = threading.Thread(target=guarded, args=(second,))
    thread_a.start()
    assert first_read.wait(timeout=1), 'first queue read never started'
    thread_b.start()
    thread_a.join(timeout=2)
    thread_b.join(timeout=2)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert errors == []


def _stub_enqueue_preflights(monkeypatch, lt):
    def fake_dataset(_user_id, dataset_id):
        return SimpleNamespace(
            id=int(dataset_id),
            train_type='flux',
            train_base_model=None,
            train_variant=None,
            train_vae_path=None,
            train_te_path=None,
        )

    monkeypatch.setattr(lt.fds, 'get_dataset', fake_dataset)
    monkeypatch.setattr(
        lt.fds,
        'db',
        SimpleNamespace(session=SimpleNamespace(commit=lambda: None)),
    )
    monkeypatch.setattr(lt, 'assert_trainable', lambda *args, **kwargs: None)
    monkeypatch.setattr(lt, 'preflight_custom_paths', lambda *args, **kwargs: None)
    monkeypatch.setattr(lt, 'find_run_collision', lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    ('dataset_ids', 'expected_ids', 'expected_queued'),
    [
        ((1, 2), {1, 2}, [True, True]),
        ((1, 1), {1}, [False, True]),
    ],
)
def test_concurrent_enqueues_preserve_items_and_reject_duplicates(
        monkeypatch, dataset_ids, expected_ids, expected_queued):
    from app.services import lora_training as lt

    _stub_enqueue_preflights(monkeypatch, lt)
    queue = _CoordinatedQueue()
    monkeypatch.setattr(lt, 'get_train_queue', queue.get)
    monkeypatch.setattr(lt, '_save_queue', queue.save)
    results = []
    results_lock = threading.Lock()

    def enqueue(dataset_id):
        result = lt.enqueue_training('local', dataset_id)
        with results_lock:
            results.append(result)

    _run_threads(
        lambda: enqueue(dataset_ids[0]),
        lambda: enqueue(dataset_ids[1]),
        queue.first_read,
    )

    assert {item['dataset_id'] for item in queue.items} == expected_ids
    assert sorted(result['queued'] for result in results) == expected_queued


def test_concurrent_dequeues_do_not_resurrect_removed_items(monkeypatch):
    from app.services import lora_training as lt

    queue = _CoordinatedQueue(({'dataset_id': 1}, {'dataset_id': 2}))
    monkeypatch.setattr(lt, 'get_train_queue', queue.get)
    monkeypatch.setattr(lt, '_save_queue', queue.save)

    _run_threads(
        lambda: lt.dequeue_training(1),
        lambda: lt.dequeue_training(2),
        queue.first_read,
    )

    assert queue.items == []


def test_enqueue_collision_check_receives_captured_variant(monkeypatch):
    from app.services import lora_training as lt

    _stub_enqueue_preflights(monkeypatch, lt)
    seen = {}
    monkeypatch.setattr(
        lt, 'find_run_collision',
        lambda *args, **kwargs: seen.update(kwargs) or None)
    monkeypatch.setattr(lt, 'get_train_queue', lambda: [])
    monkeypatch.setattr(lt, '_save_queue', lambda _items: None)

    result = lt.enqueue_training('local', 4, variant='base')
    assert result['queued'] is True
    assert seen['variant'] == 'base'


def test_enqueue_continue_requires_custom_zimage_base_confirmation(monkeypatch):
    from app.services import lora_training as lt

    ds = SimpleNamespace(
        id=4, train_type='zimage', train_base_model=None,
        train_variant='base', train_vae_path=None, train_te_path=None)
    monkeypatch.setattr(lt.fds, 'get_dataset', lambda *_a, **_kw: ds)
    with pytest.raises(ValueError, match='^CUSTOM_WEIGHTS_UNVERIFIED:'):
        lt.enqueue_training(
            'local', 4, extra_steps=500,
            base_model=r'merges\unknown.safetensors', variant='base')


def test_advance_and_dequeue_share_the_same_queue_lock(monkeypatch):
    from app.services import lora_training as lt

    advance_entered = threading.Event()
    release_advance = threading.Event()
    dequeue_read = threading.Event()

    def blocked_advance():
        advance_entered.set()
        assert release_advance.wait(timeout=1)
        return 'advanced'

    def read_queue():
        dequeue_read.set()
        return []

    monkeypatch.setattr(lt, '_advance_training_queue', blocked_advance)
    monkeypatch.setattr(lt, 'get_train_queue', read_queue)
    monkeypatch.setattr(lt, '_save_queue', lambda _items: None)

    process_thread = threading.Thread(target=lt.process_training_queue)
    dequeue_thread = threading.Thread(target=lambda: lt.dequeue_training(1))
    process_thread.start()
    assert advance_entered.wait(timeout=1)
    dequeue_thread.start()

    # process_training_queue garde le verrou pendant _advance_training_queue :
    # dequeue ne doit donc pas pouvoir lire/réécrire le même snapshot en parallèle.
    dequeue_was_blocked = not dequeue_read.wait(timeout=0.1)
    release_advance.set()
    process_thread.join(timeout=2)
    dequeue_thread.join(timeout=2)

    assert not process_thread.is_alive() and not dequeue_thread.is_alive()
    assert dequeue_was_blocked
    assert dequeue_read.is_set()


def test_stop_and_dequeue_share_the_same_queue_lock(monkeypatch):
    from app.services import lora_training as lt

    stop_clear_entered = threading.Event()
    release_stop = threading.Event()
    dequeue_started = threading.Event()
    dequeue_read = threading.Event()

    def blocked_clear(items):
        assert items == []
        stop_clear_entered.set()
        assert release_stop.wait(timeout=1)

    def read_queue():
        dequeue_read.set()
        return []

    monkeypatch.setattr(
        lt.queue_manager,
        '_get_system_state',
        lambda _key, default=None: default,
    )
    monkeypatch.setattr(
        lt.queue_manager,
        '_set_system_state',
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(lt, '_save_queue', blocked_clear)
    monkeypatch.setattr(lt, 'get_train_queue', read_queue)

    def dequeue():
        dequeue_started.set()
        lt.dequeue_training(1)

    stop_thread = threading.Thread(target=lt.stop_training)
    dequeue_thread = threading.Thread(target=dequeue)
    stop_thread.start()
    assert stop_clear_entered.wait(timeout=1)
    dequeue_thread.start()
    assert dequeue_started.wait(timeout=1)

    # Stop garde le verrou pendant le clear et la transition des flags : une
    # suppression concurrente ne doit lire la file qu'après cette transition.
    dequeue_was_blocked = not dequeue_read.wait(timeout=0.1)
    release_stop.set()
    stop_thread.join(timeout=2)
    dequeue_thread.join(timeout=2)

    assert not stop_thread.is_alive() and not dequeue_thread.is_alive()
    assert dequeue_was_blocked
    assert dequeue_read.is_set()


def test_targeted_stop_does_not_touch_a_newer_local_run(monkeypatch):
    from app.services import lora_training as lt

    state = {
        'training_dataset_id': 22,
        'training_in_progress': True,
        'training_pid': 4242,
    }
    writes = []
    queue_writes = []
    monkeypatch.setattr(lt.queue_manager, '_get_system_state',
                        lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(lt.queue_manager, '_set_system_state',
                        lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(lt, '_save_queue', lambda items: queue_writes.append(items))

    assert lt.stop_training(expected_dataset_id=21) is False
    assert writes == []
    assert queue_writes == []


def test_targeted_stop_token_rejects_same_dataset_newer_run(monkeypatch):
    from app.services import lora_training as lt

    state = {
        'training_dataset_id': 22,
        'training_run_token': 'new-run-token',
        'training_in_progress': True,
        'training_pid': 4242,
    }
    writes = []
    queue_writes = []
    monkeypatch.setattr(
        lt.queue_manager, '_get_system_state',
        lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(
        lt.queue_manager, '_set_system_state',
        lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(lt, '_save_queue', lambda items: queue_writes.append(items))

    assert lt.stop_training(
        expected_dataset_id=22,
        expected_run_token='old-run-token') is False
    assert writes == []
    assert queue_writes == []


def test_stop_waits_until_launch_publishes_the_new_pid(
        app, tmp_path, monkeypatch):
    """Popen + PID publication and Stop share one lock: Stop cannot clear the
    flag while launch is blocked inside Popen and let an orphan process escape."""
    from app import config as cfg
    from app.config import LOCAL_USER
    from app.services import checkpoint_registry
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    exported = tmp_path / 'exported'
    exported.mkdir()
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
        ds = svc.create_dataset(
            LOCAL_USER, 'Launch lock', 'launch_lock', train_type='zimage')
        dataset_id = ds.id
        lt._clear_training_identity(ttl_seconds=None)

    popen_entered = threading.Event()
    release_popen = threading.Event()
    stop_started = threading.Event()
    stop_done = threading.Event()
    identity = {}
    killed = []
    launch_errors = []
    stop_results = []

    def blocked_popen(_args, **kwargs):
        identity['token'] = lt.queue_manager._get_system_state(
            'training_run_token', None)
        popen_entered.set()
        assert release_popen.wait(timeout=2)
        kwargs['stdout'].close()
        return SimpleNamespace(pid=7878)

    monkeypatch.setattr(lt, 'assert_free_disk', lambda *_a, **_kw: None)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    monkeypatch.setattr(lt, 'preflight_custom_paths', lambda *_a, **_kw: None)
    monkeypatch.setattr(lt, 'find_run_collision', lambda *_a, **_kw: None)
    monkeypatch.setattr(
        lt, 'export_dataset_to_aitoolkit',
        lambda *_a, **_kw: str(exported))
    monkeypatch.setattr(
        lt, 'write_job_config', lambda *_a, **_kw: str(tmp_path / 'job.json'))
    monkeypatch.setattr(
        checkpoint_registry, 'register_launch', lambda *_a, **_kw: None)
    monkeypatch.setattr(lt.subprocess, 'Popen', blocked_popen)
    if lt.os.name == 'nt':
        monkeypatch.setattr(
            lt.subprocess, 'run',
            lambda args, **_kw: killed.append(args)
            or SimpleNamespace(returncode=0))
    else:
        monkeypatch.setattr(
            lt.os, 'kill', lambda pid, sig: killed.append([str(pid), str(sig)]))
    monkeypatch.setattr(lt, '_watch_training', lambda *_a, **_kw: None)

    def launch():
        try:
            with app.app_context():
                lt.launch_training(
                    LOCAL_USER, dataset_id, steps=500, masked=False)
        except BaseException as exc:
            launch_errors.append(exc)

    def stop():
        stop_started.set()
        with app.app_context():
            stop_results.append(lt.stop_training(
                expected_dataset_id=dataset_id,
                expected_run_token=identity['token']))
        stop_done.set()

    launch_thread = threading.Thread(target=launch)
    stop_thread = threading.Thread(target=stop)
    launch_thread.start()
    assert popen_entered.wait(timeout=2)
    assert identity['token']
    stop_thread.start()
    assert stop_started.wait(timeout=1)

    stop_was_blocked = not stop_done.wait(timeout=0.1)
    release_popen.set()
    launch_thread.join(timeout=3)
    stop_thread.join(timeout=3)

    assert not launch_thread.is_alive() and not stop_thread.is_alive()
    assert launch_errors == []
    assert stop_was_blocked
    assert stop_results == [True]
    assert any('7878' in args for args in killed)


def test_queued_continue_replays_captured_base_variant_and_confirmation(monkeypatch):
    from app.services import lora_training as lt

    captured = {}
    monkeypatch.setattr(
        lt, 'continue_training',
        lambda *args, **kwargs: captured.update({'args': args, 'kwargs': kwargs}))
    lt._launch_queued_item({
        'dataset_id': 9,
        'user_id': 'local',
        'extra_steps': 750,
        'base_model': r'merges\base.safetensors',
        'variant': 'deturbo',
        'train_type': 'zimage',
        'masked': False,
        'allow_unverified_weights': True,
    })
    assert captured['args'] == ('local', 9)
    assert captured['kwargs'] == {
        'extra_steps': 750,
        'base_model': r'merges\base.safetensors',
        'variant': 'deturbo',
        'train_type': 'zimage',
        'masked': False,
        'allow_unverified_weights': True,
        '_allow_dead_predecessor': True,
    }


def test_queued_continue_accepts_dead_predecessor_flag(monkeypatch):
    from app.services import lora_training as lt

    ds = SimpleNamespace(
        id=9, train_type='zimage', train_base_model=None,
        train_variant='base', train_vae_path=None, train_te_path=None)
    state = {'training_in_progress': True, 'training_pid': 4242}
    launched = {}
    monkeypatch.setattr(
        lt.queue_manager, '_get_system_state',
        lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(lt, '_pid_alive', lambda _pid: False)
    monkeypatch.setattr(lt.fds, 'get_dataset', lambda *_a, **_kw: ds)
    monkeypatch.setattr(
        lt, 'list_checkpoints',
        lambda *_a, **_kw: [{'step': 1000, 'filename': 'ck.safetensors'}])
    monkeypatch.setattr(
        lt, 'launch_training',
        lambda *_a, **kw: launched.update(kw) or {'started': True})

    result = lt.continue_training(
        'local', 9, extra_steps=500, base_model='', variant='base',
        train_type='zimage', _allow_dead_predecessor=True)

    assert result['target_steps'] == 1500
    assert launched['base_model'] == ''
    assert launched['variant'] == 'base'
    assert launched['train_type'] == 'zimage'


def test_stop_holds_queue_lock_during_kill_before_watcher_advance(monkeypatch):
    from app.services import lora_training as lt

    state = {
        'training_pid': 4242,
        'training_in_progress': True,
        'vision_in_progress': False,
    }
    queue_items = [{'dataset_id': 2, 'user_id': 'local'}]
    kill_entered = threading.Event()
    release_kill = threading.Event()
    watcher_started = threading.Event()
    advance_entered = threading.Event()
    launch_calls = []
    observations = []
    errors = []

    def get_state(key, default=None):
        return state.get(key, default)

    def set_state(key, value, ttl_seconds=None):
        del ttl_seconds
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value

    def get_queue():
        return [dict(item) for item in queue_items]

    def save_queue(items):
        queue_items[:] = [dict(item) for item in items]

    def blocked_kill(*_args, **_kwargs):
        kill_entered.set()
        assert release_kill.wait(timeout=1)

    real_advance = lt._advance_training_queue

    def observed_advance():
        observations.append({
            'queue': get_queue(),
            'in_progress': state.get('training_in_progress'),
            'pid': state.get('training_pid'),
        })
        advance_entered.set()
        return real_advance()

    monkeypatch.setattr(lt.queue_manager, '_get_system_state', get_state)
    monkeypatch.setattr(lt.queue_manager, '_set_system_state', set_state)
    monkeypatch.setattr(lt, 'get_train_queue', get_queue)
    monkeypatch.setattr(lt, '_save_queue', save_queue)
    monkeypatch.setattr(lt, '_pid_alive', lambda _pid: False)
    monkeypatch.setattr(lt, '_snapshot_final_checkpoint', lambda *_args: None)
    monkeypatch.setattr(lt, '_launch_queued_item',
                        lambda item: launch_calls.append(item['dataset_id']))
    monkeypatch.setattr(lt, '_advance_training_queue', observed_advance)
    if lt.os.name == 'nt':
        monkeypatch.setattr(lt.subprocess, 'run', blocked_kill)
    else:
        monkeypatch.setattr(lt.os, 'kill', blocked_kill)

    def guarded(call):
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    def watcher():
        watcher_started.set()
        lt.process_training_queue()

    stop_thread = threading.Thread(target=guarded, args=(lt.stop_training,))
    watcher_thread = threading.Thread(target=guarded, args=(watcher,))
    stop_thread.start()
    assert kill_entered.wait(timeout=1)
    watcher_thread.start()
    assert watcher_started.wait(timeout=1)

    # Même si le watcher considère déjà l'ancien PID comme mort, il ne peut pas
    # avancer la file pendant que Stop est encore bloqué dans le kill.
    advance_was_blocked = not advance_entered.wait(timeout=0.1)
    release_kill.set()
    stop_thread.join(timeout=2)
    watcher_thread.join(timeout=2)

    assert not stop_thread.is_alive() and not watcher_thread.is_alive()
    assert errors == []
    assert advance_was_blocked
    assert observations == [{'queue': [], 'in_progress': False, 'pid': None}]
    assert launch_calls == []
    assert queue_items == []
