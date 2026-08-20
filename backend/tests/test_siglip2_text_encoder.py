"""The SigLIP2 query tower never crosses CLIP cache/worker boundaries."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


np = pytest.importorskip('numpy')


def _unit(index):
    value = np.zeros(768, dtype='float32')
    value[index] = 1.0
    return value


def test_clip_and_siglip2_queries_use_separate_provenance_bound_files(
        app, tmp_path, monkeypatch):
    from app.services import bank_semantic_engine
    from app.services import clip_text_encoder as encoder

    monkeypatch.setattr(encoder.cfg, 'banks_root', lambda: tmp_path)
    monkeypatch.setattr(
        encoder, 'unavailable_reason', lambda engine='clip': None)
    calls = {'clip': 0, 'siglip2': 0}

    def clip_encode(_texts, **_kwargs):
        calls['clip'] += 1
        return [_unit(0)]

    def siglip_encode(_texts):
        calls['siglip2'] += 1
        return [_unit(1)]

    monkeypatch.setattr(encoder, '_encode_uncached', clip_encode)
    monkeypatch.setattr(encoder, '_encode_siglip2_uncached', siglip_encode)
    encoder.forget_memory_cache()
    clip, clip_cached = encoder.encode_query('same phrase')
    siglip, siglip_cached = encoder.encode_query('same phrase', engine='siglip2')
    assert clip_cached is False and siglip_cached is False
    assert int(np.argmax(clip)) == 0 and int(np.argmax(siglip)) == 1
    assert calls == {'clip': 1, 'siglip2': 1}
    assert encoder._cache_path() != encoder._siglip2_cache_path()
    assert encoder._cache_path().is_file()
    assert encoder._siglip2_cache_path().is_file()

    contract = bank_semantic_engine.semantic_contract('siglip2')
    with np.load(encoder._siglip2_cache_path(), allow_pickle=False) as cache:
        assert cache['engine'].shape == (1,)
        assert cache['model_key'].shape == (1,)
        assert cache['dimension'].shape == (1,)
        assert str(cache['engine'][0]) == 'siglip2'
        assert str(cache['model_key'][0]) == contract['model_key']
        assert int(cache['dimension'][0]) == contract['dimension']

    encoder.forget_memory_cache()
    monkeypatch.setattr(
        encoder, '_encode_uncached',
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('CLIP re-encoded')))
    monkeypatch.setattr(
        encoder, '_encode_siglip2_uncached',
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('SigLIP re-encoded')))
    assert encoder.encode_query('same phrase')[1] is True
    assert encoder.encode_query('same phrase', engine='siglip2')[1] is True


class _Input:
    closed = False

    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(line)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _Output:
    def __init__(self, messages):
        self.messages = list(messages)

    def readline(self):
        return json.dumps(self.messages.pop(0)) + '\n'


class _Worker:
    def __init__(self, messages):
        self.stdin = _Input()
        self.stdout = _Output(messages)
        self.stderr = None
        self.killed = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def test_siglip2_text_worker_uses_semantic_python(app, monkeypatch):
    from app.services import bank_semantic_engine
    from app.services import bank_semantic_models as models
    from app.services import clip_text_encoder as encoder

    contract = bank_semantic_engine.text_worker_handshake('siglip2')
    worker = _Worker([{
        'ok': True, 'ready': True, 'engine': 'siglip2',
        'model_key': contract['model_key'],
        'dimension': contract['dimension'],
    }])
    seen = {}

    def popen(command, **kwargs):
        seen['command'] = command
        return worker

    monkeypatch.setattr(models, 'semantic_python',
                        lambda: '/managed/semantic/python')
    monkeypatch.setattr(encoder.subprocess, 'Popen', popen)
    monkeypatch.setattr(encoder, '_siglip2_proc', None)
    # Avoid starting the idle-reaper thread; its behaviour is covered separately.
    monkeypatch.setattr(encoder, '_siglip2_reaper', object())
    assert encoder._start_siglip2_worker_locked() is worker
    # ...without the machine's user site-packages, like every other infer
    # worker and like the probe that vouches for this one (services/infer_env).
    assert seen['command'] == [
        '/managed/semantic/python', '-s', encoder._SIGLIP2_SCRIPT]
    encoder._stop_siglip2_worker_locked()


def test_siglip2_encoding_never_reuses_the_live_clip_worker(app, monkeypatch):
    from app.services import bank_semantic_engine
    from app.services import clip_text_encoder as encoder

    contract = bank_semantic_engine.text_worker_handshake('siglip2')
    siglip_worker = _Worker([{
        'ok': True, 'engine': 'siglip2', 'model_key': contract['model_key'],
        'dimension': contract['dimension'], 'vector': _unit(2).tolist(),
    }])
    poison_clip = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(encoder, '_proc', poison_clip)
    monkeypatch.setattr(encoder, '_siglip2_proc', siglip_worker)
    monkeypatch.setattr(encoder, 'idle_minutes', lambda: 10)
    vector = encoder._encode_siglip2_uncached(['query'])[0]
    assert int(np.argmax(vector)) == 2
    assert len(siglip_worker.stdin.lines) == 1
    assert encoder._proc is poison_clip
    encoder.release(engine='siglip2')
    assert encoder._proc is poison_clip


def test_siglip2_status_and_release_are_engine_scoped(app, monkeypatch):
    from app.services import clip_text_encoder as encoder

    clip_worker = _Worker([])
    siglip_worker = _Worker([])
    monkeypatch.setattr(encoder, '_proc', clip_worker)
    monkeypatch.setattr(encoder, '_siglip2_proc', siglip_worker)
    monkeypatch.setattr(
        encoder, 'unavailable_reason', lambda engine='clip': None)
    monkeypatch.setattr(encoder, 'cached_queries', lambda engine='clip': 0)
    monkeypatch.setattr(encoder, 'weights_warning', lambda engine='clip': None)
    assert encoder.status(engine='siglip2')['warm'] is True
    assert encoder.release(engine='siglip2') is True
    assert encoder._siglip2_proc is None
    assert encoder._proc is clip_worker
    encoder.release()


@pytest.mark.parametrize('engine', ['clip', 'siglip2'])
def test_query_limit_is_enforced_before_any_model_probe(app, monkeypatch, engine):
    from app.services import clip_text_encoder as encoder

    monkeypatch.setattr(
        encoder, 'unavailable_reason',
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('oversized input must fail before probing a model')))
    with pytest.raises(encoder.TextEncodeError, match='maximum'):
        encoder.encode_query('x' * (encoder.MAX_QUERY_CHARS + 1), engine=engine)


@pytest.mark.parametrize('engine', ['clip', 'siglip2'])
def test_query_cache_is_lru_bounded_for_each_engine(
        app, tmp_path, monkeypatch, engine):
    from app.services import clip_text_encoder as encoder

    monkeypatch.setattr(encoder.cfg, 'banks_root', lambda: tmp_path)
    monkeypatch.setattr(encoder, 'MAX_CACHED_QUERIES', 2)
    monkeypatch.setattr(encoder, 'unavailable_reason', lambda *a, **k: None)
    monkeypatch.setattr(encoder, '_encode_uncached', lambda texts: [_unit(0) for _ in texts])
    monkeypatch.setattr(
        encoder, '_encode_siglip2_uncached', lambda texts: [_unit(1) for _ in texts])
    encoder.forget_memory_cache()
    encoder.encode_query('first', engine=engine)
    encoder.encode_query('second', engine=engine)
    assert encoder.encode_query('first', engine=engine)[1] is True  # newest
    encoder.encode_query('third', engine=engine)
    memory = encoder._memory if engine == 'clip' else encoder._siglip2_memory
    assert list(memory) == ['first', 'third']


def test_oversized_disk_cache_is_ignored_before_numpy_load(
        app, tmp_path, monkeypatch):
    from app.services import clip_text_encoder as encoder

    monkeypatch.setattr(encoder.cfg, 'banks_root', lambda: tmp_path)
    path = encoder._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path), queries=np.asarray(['safe']),
        vecs=np.asarray([_unit(0)], dtype='float32'))
    monkeypatch.setattr(encoder, '_QUERY_CACHE_MAX_FILE_BYTES', 1)
    encoder.forget_memory_cache()
    assert encoder.cached_queries('clip') == 0
