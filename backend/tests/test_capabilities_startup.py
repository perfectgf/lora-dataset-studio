"""Startup probes share work without weakening readiness or refresh semantics."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from flask import current_app

from app import capabilities as caps


@pytest.fixture(autouse=True)
def isolated_probes(monkeypatch):
    caps.clear_import_cache()
    monkeypatch.setattr(caps, '_http_ok', lambda *a, **k: False)
    monkeypatch.setattr(caps, '_import_ok', lambda *a, **k: False)
    monkeypatch.setattr(caps, 'gpu_vram_gb', lambda: None)
    monkeypatch.setattr(caps.ffmpeg_tools, 'ffmpeg_ready',
                        lambda: {'ok': False, 'reason': 'not installed'})
    yield
    caps.clear_import_cache()


def in_app(app, *, force=False):
    with app.app_context():
        return caps.probe(force=force)


def test_independent_probes_overlap_with_at_most_four_workers(app, monkeypatch):
    """Four blocked probes must start, while the remaining probes wait."""
    guard, release, four_started = Lock(), Event(), Event()
    active = peak = calls = 0

    def blocked_probe():
        nonlocal active, peak, calls
        assert current_app._get_current_object() is app
        with guard:
            calls += 1
            active += 1
            peak = max(peak, active)
            if active == 4:
                four_started.set()
        try:
            assert release.wait(5), 'test did not release the probe'
            return {'ok': True, 'detail': 'ready', 'model': 'test-model'}
        finally:
            with guard:
                active -= 1

    names = ('probe_face_scoring', 'probe_masks', 'probe_bank_scoring',
             'probe_bank_siglip2', 'probe_watermark_inpaint', 'probe_video_text')
    for name in names:
        monkeypatch.setattr(caps, name, blocked_probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(in_app, app)
        try:
            assert four_started.wait(3), 'independent probes still run serially'
            with guard:
                assert calls == peak == 4
        finally:
            release.set()
        result = pending.result(timeout=5)
    assert calls == 6 and peak == 4
    assert all(result[key] is True for key in (
        'face_scoring', 'masks', 'bank_scoring', 'bank_siglip2',
        'watermark_inpaint', 'video_text'))


def test_main_cache_ttl_starts_after_long_scan(app, monkeypatch):
    clock, calls = [100.0], []
    monkeypatch.setattr(caps.time, 'time', lambda: clock[0])

    def slow_probe():
        calls.append(1)
        clock[0] += caps._CACHE_TTL + 1
        return {'ok': True}

    monkeypatch.setattr(caps, 'probe_masks', slow_probe)
    first = in_app(app)
    assert in_app(app) == first
    assert calls == [1]
    first['engines']['klein'] = 'caller mutation'
    assert in_app(app)['engines']['klein'] is False


@pytest.mark.parametrize('verdict', [True, False, None])
def test_import_cache_ttl_starts_after_long_import(monkeypatch, verdict):
    clock, calls = [100.0], []
    monkeypatch.setattr(caps.time, 'time', lambda: clock[0])

    def slow_import(*args):
        calls.append(1)
        clock[0] += caps._IMPORT_TTL + 1
        return verdict

    monkeypatch.setattr(caps, '_import_ok', slow_import)
    assert caps._cached_import_state('startup', 'python', 'import example') is verdict
    assert caps._cached_import_state('startup', 'python', 'import example') is verdict
    assert calls == [1]


@pytest.mark.parametrize('verdict', [True, False, None])
def test_same_import_is_shared_by_overlapping_callers(monkeypatch, verdict):
    started, release, second_miss = Event(), Event(), Event()
    calls = []

    class ObservedCache(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if started.is_set() and value is None:
                second_miss.set()
            return value

    monkeypatch.setattr(caps, '_import_cache', ObservedCache())

    def blocked_import(*args):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return verdict

    monkeypatch.setattr(caps, '_import_ok', blocked_import)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(caps._cached_import_state, 'startup', 'python', 'import example')
        try:
            assert started.wait(3)
            second = executor.submit(caps._cached_import_state, 'startup', 'python', 'import example')
            assert second_miss.wait(3)
        finally:
            release.set()
        assert first.result(timeout=3) is verdict
        assert second.result(timeout=3) is verdict
    assert calls == [1]


def test_clear_during_import_does_not_republish_old_verdict(monkeypatch):
    started, release = Event(), Event()
    calls = []

    def changed_import(*args):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            return False
        return True

    monkeypatch.setattr(caps, '_import_ok', changed_import)
    with ThreadPoolExecutor(max_workers=1) as executor:
        old = executor.submit(caps._cached_import_state, 'startup', 'python', 'import example')
        try:
            assert started.wait(3)
            caps.clear_import_cache()
        finally:
            release.set()
        assert old.result(timeout=3) is False
    assert caps._import_cache == {}
    assert caps._cached_import_state('startup', 'python', 'import example') is True
    assert calls == [1, 1]


def test_clear_during_scan_keeps_next_read_fresh(app, monkeypatch):
    started, release = Event(), Event()
    calls = []

    def changed_probe():
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            return {'ok': False}
        return {'ok': True}

    monkeypatch.setattr(caps, 'probe_masks', changed_probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        old = executor.submit(in_app, app)
        try:
            assert started.wait(3)
            caps.clear_import_cache()
        finally:
            release.set()
        assert old.result(timeout=3)['masks'] is False
    assert caps._cache is None
    assert in_app(app)['masks'] is True
    assert calls == [1, 1]


@pytest.mark.parametrize('force', [False, True])
def test_waiting_reads_share_scan_but_force_reads_changed_settings(app, monkeypatch, force):
    started, release, waiting = Event(), Event(), Event()
    calls = []

    class ObservedLock:
        """Let the test release scan one only after request two reaches it."""
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if started.is_set():
                waiting.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    def scan():
        result = {'model': caps.cfg.get('ollama.vision_model'),
                  'key_set': bool(caps.cfg.secret('GEMINI_API_KEY'))}
        calls.append(result)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(caps, '_probe_lock', ObservedLock())
    monkeypatch.setattr(caps, '_probe_uncached', scan)
    caps.cfg.save_config({'ollama': {'vision_model': 'before'}})
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(in_app, app)
        try:
            assert started.wait(3)
            caps.cfg.save_config({'ollama': {'vision_model': 'after'}})
            monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
            second = executor.submit(in_app, app, force=force)
            assert waiting.wait(3)
            assert len(calls) == 1, 'two whole scans ran concurrently'
        finally:
            release.set()
        before = first.result(timeout=3)
        after = second.result(timeout=3)
    assert before == {'model': 'before', 'key_set': False}
    assert after == ({'model': 'after', 'key_set': True} if force else before)
    assert len(calls) == (2 if force else 1)
    # A force issued later is also a fresh read, even with a valid cache.
    assert in_app(app, force=True) == {'model': 'after', 'key_set': True}
    assert len(calls) == (3 if force else 2)


def test_different_imports_can_run_together(monkeypatch):
    started = {name: Event() for name in ('import first', 'import second')}
    release = Event()

    def blocked_import(python, expression):
        started[expression].set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(caps, '_import_ok', blocked_import)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(caps._cached_import_state, 'startup', 'python', expr)
                   for expr in started]
        try:
            assert all(event.wait(3) for event in started.values())
        finally:
            release.set()
        assert all(future.result(timeout=3) is True for future in pending)


@pytest.mark.parametrize('kind', ['import', 'scan'])
def test_failed_probe_releases_locks_for_retry(app, monkeypatch, kind):
    calls = []

    def fail_once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('broken probe')
        return True if kind == 'import' else {'ok': True}

    if kind == 'import':
        monkeypatch.setattr(caps, '_import_ok', fail_once)

        def request():
            return caps._cached_import_state('startup', 'python', 'import example')
    else:
        monkeypatch.setattr(caps, 'probe_masks', fail_once)

        def request():
            return in_app(app)['masks']

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match='broken probe'):
            executor.submit(request).result(timeout=5)
        assert executor.submit(request).result(timeout=5) is True
    assert calls == [1, 1]


def test_parallel_snapshot_matches_serial_probes(app, monkeypatch):
    monkeypatch.setattr(caps, '_PROBE_WORKERS', 1)
    serial = in_app(app, force=True)
    monkeypatch.setattr(caps, '_PROBE_WORKERS', 4)
    parallel = in_app(app, force=True)
    assert parallel == serial


@pytest.mark.parametrize('fails', [False, True])
def test_clear_discards_late_ffmpeg_cache_even_when_another_probe_fails(monkeypatch, fails):
    def invalidated_scan():
        caps.clear_import_cache()
        # An encoder subprocess launched before the clear answers afterwards.
        caps.ffmpeg_tools._ready_cache = (0, {'ok': False})
        if fails:
            raise ValueError('another probe failed')
        return {'video_encode': False}

    monkeypatch.setattr(caps, '_probe_uncached', invalidated_scan)
    if fails:
        with pytest.raises(ValueError, match='another probe failed'):
            caps.probe()
    else:
        assert caps.probe() == {'video_encode': False}
    assert caps.ffmpeg_tools._ready_cache is None
    assert caps._cache is None
