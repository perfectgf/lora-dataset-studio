"""The caption engine follows what the machine HAS (2026-09-01).

Born of a maintainer's objection that survived the bench built to refute it:
the app ships to users who run Ollama or LM Studio and no torch Python, and the
video pass answered them with a dead ✗. The 2026-08-04 claim that Ollama fails
silently on video was remeasured the day of this file and had expired (16/16
full answers, one call per shot). What never moved, and is pinned across every
engine: an empty answer is stored as an error, never as a caption.

The resolver's contract: LDS's own transformers worker when the ✨ Score
interpreter can run it — native timestamps, the real umT5 count — else the
local LLM the user already operates, through the SAME vision_llm waist and the
SAME settings as the image passes. No new which-server or which-model dial.
"""
import json

import pytest

from app.services import video_caption as vc
from app.services import video_caption_worker as vcw


def _no_cache(monkeypatch):
    monkeypatch.setattr(vc, '_backend_cache', {'at': 0.0, 'value': None})


def _cfg(monkeypatch, values):
    from app import config as cfg
    real = cfg.get

    def fake(key, *a, **kw):
        if key in values:
            return values[key]
        return real(key, *a, **kw)
    monkeypatch.setattr(cfg, 'get', fake)


# --- resolution ----------------------------------------------------------------------

def test_auto_prefers_the_transformers_worker_when_it_can_run(app, monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(vcw, 'unavailable_reason', lambda: None)
    calls = []
    monkeypatch.setattr(vc, '_ollama_reachable',
                        lambda url: calls.append(url) or (True, ''))
    with app.app_context():
        resolved = vc.resolve_backend(fresh=True)
    assert resolved['backend'] == 'transformers'
    assert resolved['available'] is True
    assert resolved['record_id'] == resolved['model']
    # The winning side settles it — nobody probes a server for nothing.
    assert calls == []


def test_auto_falls_back_to_the_local_llm_and_says_which(app, monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(vcw, 'unavailable_reason', lambda: 'no torch here')
    monkeypatch.setattr(vc, '_ollama_reachable', lambda url: (True, ''))
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama, 'get_vision_model',
                        lambda: 'qwen3-vl:4b-instruct')
    with app.app_context():
        resolved = vc.resolve_backend(fresh=True)
    assert resolved['backend'] == 'local_llm'
    assert resolved['engine'] == 'ollama'
    assert resolved['available'] is True
    # Engine-prefixed: a bank captioned across engines stays readable.
    assert resolved['record_id'] == 'ollama:qwen3-vl:4b-instruct'


def test_when_nothing_can_run_the_reason_names_both_absences(app, monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(vcw, 'unavailable_reason', lambda: 'no torch here.')
    monkeypatch.setattr(vc, '_ollama_reachable',
                        lambda url: (False, 'Ollama is not reachable at x'))
    with app.app_context():
        resolved = vc.resolve_backend(fresh=True)
        reason = vc.caption_unavailable_reason()
    assert resolved['available'] is False
    # "Install a torch Python" alone is the wrong advice for an Ollama user.
    assert 'no torch here' in reason and 'Ollama is not reachable' in reason


def test_forcing_transformers_never_touches_the_network(app, monkeypatch):
    _no_cache(monkeypatch)
    _cfg(monkeypatch, {'video_caption.backend': 'transformers'})
    monkeypatch.setattr(vcw, 'unavailable_reason', lambda: 'no torch here')

    def boom(url):
        raise AssertionError('probed the network for a forced side')
    monkeypatch.setattr(vc, '_ollama_reachable', boom)
    with app.app_context():
        resolved = vc.resolve_backend(fresh=True)
    assert resolved['backend'] == 'transformers'
    assert resolved['available'] is False


def test_forcing_local_llm_reports_its_own_absence(app, monkeypatch):
    _no_cache(monkeypatch)
    _cfg(monkeypatch, {'video_caption.backend': 'local_llm'})
    monkeypatch.setattr(vc, '_ollama_reachable', lambda url: (False, 'down.'))
    with app.app_context():
        resolved = vc.resolve_backend(fresh=True)
    assert resolved['backend'] == 'local_llm'
    assert resolved['available'] is False
    assert resolved['reason'] == 'down.'


def test_the_lmstudio_provider_resolves_through_its_own_driver(app, monkeypatch):
    _no_cache(monkeypatch)
    _cfg(monkeypatch, {'video_caption.backend': 'local_llm',
                       'local_llm.provider': 'lmstudio'})
    from app.services import vision_lmstudio
    monkeypatch.setattr(vision_lmstudio, 'resolve_model',
                        lambda preferred=None, **kw: 'qwen3-vl-8b')
    with app.app_context():
        resolved = vc.resolve_backend(fresh=True)
    assert resolved['engine'] == 'lmstudio'
    assert resolved['available'] is True
    assert resolved['record_id'] == 'lmstudio:qwen3-vl-8b'


# --- the run records the engine ------------------------------------------------------

def _bank_with_one_clip(app):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=60.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                 start_s=0.0, end_s=10.0))
        db.session.commit()
        return bank.id


def test_run_captions_records_the_local_engine_and_estimates_tokens(app, monkeypatch):
    from app.models import VideoClip

    class _Fake:
        def __init__(self, **kw):
            self.kw = kw
            self.last_tokens = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    monkeypatch.setattr(vc, 'resolve_backend', lambda fresh=False: {
        'backend': 'local_llm', 'engine': 'ollama', 'label': 'Ollama',
        'model': 'qwen3-vl:4b-instruct',
        'record_id': 'ollama:qwen3-vl:4b-instruct',
        'available': True, 'reason': None})
    # run_captions imports the class from its OWN module at call time.
    monkeypatch.setattr(vcw, 'LocalLlmCaptionWorker', _Fake)
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/{stem}_0.jpg'])
    monkeypatch.setattr(vc, '_caption_frames',
                        lambda paths, prompt, **kw: 'A woman turns and walks away.')
    bank_id = _bank_with_one_clip(app)
    with app.app_context():
        out = vc.run_captions(bank_id)
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        assert clip.caption == 'A woman turns and walks away.'
        assert clip.caption_model == 'ollama:qwen3-vl:4b-instruct'
        # No umT5 behind an HTTP server: the preflight estimates, and says so.
        assert clip.caption_tokens is None
    assert out['model'] == 'ollama:qwen3-vl:4b-instruct'


# --- the local worker's contract -----------------------------------------------------

def test_the_time_preamble_tells_the_truth_or_says_nothing():
    line = vcw.frames_time_preamble(8, 5.0)
    assert line.startswith('The 8 frames')
    assert '5.0-second' in line and '0.0s' in line and '5.0s' in line
    # One frame carries no motion; no span carries no time. Silence beats a guess.
    assert vcw.frames_time_preamble(1, 5.0) == ''
    assert vcw.frames_time_preamble(8, 0) == ''
    assert vcw.frames_time_preamble(8, None) == ''


def test_the_local_worker_prepends_time_and_refuses_empty_out_loud(
        app, tmp_path, monkeypatch, caplog):
    from app.services import vision_llm
    seen = {}

    def fake(frames, prompt, **kw):
        seen['frames'] = len(frames)
        seen['prompt'] = prompt
        return ''
    monkeypatch.setattr(vision_llm, 'describe_frames', fake)
    monkeypatch.setattr(vision_llm, 'unload_vision_model', lambda **kw: True)
    p1, p2 = tmp_path / 'a.jpg', tmp_path / 'b.jpg'
    p1.write_bytes(b'x')
    p2.write_bytes(b'y')
    with app.app_context():
        with vcw.LocalLlmCaptionWorker(provider='ollama', model='m') as worker:
            with caplog.at_level('WARNING'):
                out = worker.caption([str(p1), str(p2)], 'PROMPT', span_s=4.0)
    assert out == ''
    assert worker.last_tokens is None
    assert seen['frames'] == 2
    assert seen['prompt'].startswith('The 2 frames')
    assert seen['prompt'].endswith('PROMPT')
    # The Ollama failure mode of 2026-08-04, guarded on every engine, out loud.
    assert 'caption worker refused a shot' in caplog.text


def test_the_local_worker_records_an_engine_prefixed_id(app):
    with app.app_context():
        worker = vcw.LocalLlmCaptionWorker(provider='ollama', model='qwen3-vl:4b')
    assert worker.loaded_model == 'ollama:qwen3-vl:4b'
    assert worker.token_counter is None


# --- the waist dispatches ------------------------------------------------------------

def test_describe_frames_dispatches_on_the_provider(app, monkeypatch):
    from app.services import vision_llm, vision_lmstudio, vision_ollama
    hits = []
    monkeypatch.setattr(vision_ollama, 'describe_frames_ollama',
                        lambda frames, prompt, **kw: hits.append('ollama') or 'o')
    monkeypatch.setattr(vision_lmstudio, 'describe_frames',
                        lambda frames, prompt, **kw: hits.append('lmstudio') or 'l')
    with app.app_context():
        _cfg(monkeypatch, {'local_llm.provider': 'ollama'})
        assert vision_llm.describe_frames([b'x'], 'p') == 'o'
        _cfg(monkeypatch, {'local_llm.provider': 'lmstudio'})
        assert vision_llm.describe_frames([b'x'], 'p') == 'l'
    assert hits == ['ollama', 'lmstudio']


def test_the_ollama_video_door_sends_one_deterministic_request(app, monkeypatch):
    from app.services import vision_ollama
    sent = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'response': 'a caption'}

    def fake_post(url, json=None, timeout=None):
        sent['url'] = url
        sent['payload'] = json
        return _Resp()
    monkeypatch.setattr(vision_ollama, '_ensure_ollama_decodable', lambda b: b)
    monkeypatch.setattr(vision_ollama, '_admit_local_ollama',
                        lambda *a, **kw: None)
    monkeypatch.setattr(vision_ollama.requests, 'post', fake_post)
    with app.app_context():
        out = vision_ollama.describe_frames_ollama(
            [b'f1', b'f2', b'f3'], 'PROMPT', model='qwen3-vl:4b-instruct')
    assert out == 'a caption'
    assert sent['url'].endswith('/api/generate')
    payload = sent['payload']
    assert len(payload['images']) == 3
    # Training captions, not conversation: the same answer twice.
    assert payload['options']['temperature'] == 0
    assert payload['keep_alive'] == '5m'
    assert payload['stream'] is False


def test_the_runtime_line_reports_the_resolved_engine(app, monkeypatch):
    from app.services import video_bank_service as svc
    monkeypatch.setattr(vc, 'resolve_backend', lambda fresh=False: {
        'backend': 'local_llm', 'engine': 'ollama', 'label': 'Ollama',
        'model': 'qwen3-vl:4b-instruct', 'record_id': 'ollama:qwen3-vl:4b-instruct',
        'available': True, 'reason': None})
    with app.app_context():
        info = svc.caption_model_info()
    assert info['runtime']['engine'] == 'ollama'
    assert info['runtime']['backend'] == 'local_llm'
    assert info['runtime']['model'] == 'qwen3-vl:4b-instruct'
    assert info['runtime']['available'] is True


def test_the_fields_tail_survives_a_local_engine(app, monkeypatch):
    """C12-C is engine-independent: the labelled tail is parsed from whatever
    text comes back, so a local LLM's caption stores fields too."""
    from app.models import VideoClip

    class _Fake:
        def __init__(self, **kw):
            self.last_tokens = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    tailed = ('A woman walks to the window.\n---\nSubject: a woman\n'
              'Motion: walks to the window\nSetting: a flat\nStyle: daylight\n'
              'Short: a woman walks to a window')
    monkeypatch.setattr(vc, 'resolve_backend', lambda fresh=False: {
        'backend': 'local_llm', 'engine': 'lmstudio', 'label': 'LM Studio',
        'model': 'q8', 'record_id': 'lmstudio:q8', 'available': True,
        'reason': None})
    monkeypatch.setattr(vcw, 'LocalLlmCaptionWorker', _Fake)
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/{stem}_0.jpg'])
    monkeypatch.setattr(vc, '_caption_frames', lambda paths, prompt, **kw: tailed)
    bank_id = _bank_with_one_clip(app)
    with app.app_context():
        vc.run_captions(bank_id)
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        assert clip.caption == 'A woman walks to the window.'
        assert json.loads(clip.caption_fields)['motion'] == 'walks to the window'
        assert clip.caption_model == 'lmstudio:q8'


def test_the_resolver_result_is_cached_for_the_polls(app, monkeypatch):
    _no_cache(monkeypatch)
    calls = []

    def probe():
        calls.append(1)
        return None
    monkeypatch.setattr(vcw, 'unavailable_reason', probe)
    with app.app_context():
        vc.resolve_backend(fresh=True)
        vc.resolve_backend()
        vc.resolve_backend()
    assert len(calls) == 1
    with pytest.raises(TypeError):
        # Guard against a silent signature change: fresh is keyword-friendly.
        vc.resolve_backend(True, True)
