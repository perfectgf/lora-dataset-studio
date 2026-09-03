"""/object_info is served stale at once and refreshed behind: a launch that
comes a minute after the previous one no longer waits the 6.7 MB read out."""
import threading
import time

from app.utils import comfyui as cu


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _prime(monkeypatch, classes, timestamp):
    cu._object_info_cache.update(data=set(classes), timestamp=timestamp, key=cu.api_address(),
                                 enums={}, files={})
    cu._object_info_last.update(timestamp=timestamp, key=cu.api_address(), status='ok', waited=5)


def test_a_stale_set_is_served_at_once_and_refreshed_behind(monkeypatch):
    calls = []
    gate = threading.Event()

    def slow_get(url, timeout=None, **kw):
        calls.append(url)
        gate.wait(2)                                  # the 2.2 s read, held by the test
        return _Resp({'NewNode': {}, 'OldNode': {}})
    monkeypatch.setattr(cu.requests, 'get', slow_get)
    monkeypatch.setattr(cu, '_object_info_refreshing', False)
    _prime(monkeypatch, {'OldNode'}, time.time() - cu._OBJECT_INFO_TTL - 5)   # stale, not ancient

    t0 = time.perf_counter()
    classes = cu.fetch_object_info_classes()
    assert time.perf_counter() - t0 < 0.5, 'the stale answer came back without waiting the read out'
    assert classes == {'OldNode'}
    assert cu.fetch_object_info_classes() == {'OldNode'}
    assert len(calls) == 1, 'one refresh in flight, not one per caller'
    gate.set()
    deadline = time.time() + 3
    while cu._object_info_cache['data'] != {'NewNode', 'OldNode'} and time.time() < deadline:
        time.sleep(0.02)
    assert cu._object_info_cache['data'] == {'NewNode', 'OldNode'}, 'the refresh landed'
    assert cu.fetch_object_info_classes() == {'NewNode', 'OldNode'}


def test_an_ancient_set_blocks_for_a_fresh_read_as_before(monkeypatch):
    def quick_get(url, timeout=None, **kw):
        return _Resp({'Fresh': {}})
    monkeypatch.setattr(cu.requests, 'get', quick_get)
    monkeypatch.setattr(cu, '_object_info_refreshing', False)
    _prime(monkeypatch, {'Ancient'}, time.time() - cu._OBJECT_INFO_STALE_MAX - 5)
    assert cu.fetch_object_info_classes() == {'Fresh'}, 'older than the stale window: read now'
