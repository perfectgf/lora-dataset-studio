"""Choosing the model of the Nano Banana and ChatGPT engines.

NOTHING here talks to Google or OpenAI: `requests.post` is patched in every test,
so the suite never needs a key, never spends a cent, and never depends on either
service being up. That is also the honest limit of this file — it proves the
REQUEST we build and how we read an answer we hand ourselves.

The invariants, in order of how badly they hurt when broken:
  1. the model the user configured is the model actually SENT (a setting that
     doesn't reach the wire is worse than no setting at all);
  2. a blank setting keeps the historical default — nobody's behaviour changes
     because a field appeared;
  3. the documented precedence holds, including for someone who had set the
     pre-existing environment variable: setting > env var > built-in default;
  4. a model the provider refuses fails with a NAMED cause that says it is the
     model, and stops the batch instead of paying for the same refusal per row;
  5. a moderation refusal is still reported as a refusal (it is one) — and on
     Gemini it is now reported with the provider's own reason instead of a
     silent None the caller had to guess about.
"""

import pytest

pytestmark = pytest.mark.plugins('api_engines')

import base64
import logging
from unittest.mock import MagicMock, patch


GEMINI_KEY = 'AIza-test-gemini-key-0123456789'
OPENAI_KEY = 'sk-proj-testopenaikey0123456789'
PNG = b'\x89PNG\r\n\x1a\nfake-pixels'


def _resp(status=200, json_body=None, text=''):
    r = MagicMock(status_code=status)
    if json_body is None:
        r.json.side_effect = ValueError('no json')
    else:
        r.json.return_value = json_body
    r.text = text
    return r


def _gemini_ok():
    return {'candidates': [{'content': {'parts': [
        {'inlineData': {'mimeType': 'image/png',
                        'data': base64.b64encode(PNG).decode()}}]}}]}


def _gemini_err(msg, code=400, status='INVALID_ARGUMENT'):
    return {'error': {'code': code, 'message': msg, 'status': status}}


def _openai_ok():
    return {'data': [{'b64_json': base64.b64encode(PNG).decode()}]}


def _openai_err(msg, code=None, param=None, type_='invalid_request_error'):
    return {'error': {'message': msg, 'type': type_, 'param': param, 'code': code}}


# --- 1. the configured model really is the one sent -------------------------

def test_nanobanana_asks_for_the_model_from_settings(app, monkeypatch):
    """The Gemini model rides in the URL path, not the body — a setting that never
    reached that URL would be a silent no-op the user could not tell from a
    provider quirk."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    monkeypatch.delenv('NANOBANANA_MODEL', raising=False)
    from app import config as cfg
    from lds_api_engines import nanobanana
    with app.app_context():
        cfg.save_config({'engines': {'nanobanana_model': 'gemini-4-flash-image'}})
        assert nanobanana.get_model() == 'gemini-4-flash-image'
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(200, _gemini_ok())) as post:
            assert nanobanana.generate_variation(b'ref', 'a portrait') == PNG
    assert post.call_args.args[0] == (
        'https://generativelanguage.googleapis.com/v1beta/models/'
        'gemini-4-flash-image:generateContent')


def test_chatgpt_sends_the_image_model_from_settings(app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    monkeypatch.delenv('CHATGPT_IMAGE_MODEL', raising=False)
    from app import config as cfg
    from lds_api_engines import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_image_model': 'gpt-image-3'}})
        assert chatgpt_image.get_image_model() == 'gpt-image-3'
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(200, _openai_ok())) as post:
            assert chatgpt_image.generate_variation(b'ref', 'a portrait') == PNG
    assert post.call_args.kwargs['data']['model'] == 'gpt-image-3'


def test_a_model_typed_in_settings_applies_without_a_restart(app, monkeypatch):
    """The old env-var-only override was read at IMPORT. Resolving at call time is
    the whole point: two generations in the same process must be able to use two
    different models."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    monkeypatch.delenv('NANOBANANA_MODEL', raising=False)
    from app import config as cfg
    from lds_api_engines import nanobanana
    with app.app_context():
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(200, _gemini_ok())) as post:
            cfg.save_config({'engines': {'nanobanana_model': 'model-one'}})
            nanobanana.generate_variation(b'r', 'p')
            cfg.save_config({'engines': {'nanobanana_model': 'model-two'}})
            nanobanana.generate_variation(b'r', 'p')
    assert 'model-one:generateContent' in post.call_args_list[0].args[0]
    assert 'model-two:generateContent' in post.call_args_list[1].args[0]


# --- 2. non-regression: a blank field changes nothing -----------------------

@pytest.mark.parametrize('blank', ['', '   '])
def test_a_blank_setting_keeps_the_historical_nanobanana_model(app, monkeypatch, blank):
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    monkeypatch.delenv('NANOBANANA_MODEL', raising=False)
    from app import config as cfg
    from lds_api_engines import nanobanana
    with app.app_context():
        cfg.save_config({'engines': {'nanobanana_model': blank}})
        assert nanobanana.get_model() == 'gemini-3-pro-image' == nanobanana.DEFAULT_MODEL
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(200, _gemini_ok())) as post:
            nanobanana.generate_variation(b'r', 'p')
    assert 'gemini-3-pro-image:generateContent' in post.call_args.args[0]


@pytest.mark.parametrize('blank', ['', '   '])
def test_a_blank_setting_keeps_the_historical_chatgpt_model(app, monkeypatch, blank):
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    monkeypatch.delenv('CHATGPT_IMAGE_MODEL', raising=False)
    from app import config as cfg
    from lds_api_engines import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_image_model': blank}})
        assert chatgpt_image.get_image_model() == 'gpt-image-2.5-sunburst' \
            == chatgpt_image.DEFAULT_IMAGE_MODEL
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(200, _openai_ok())) as post:
            chatgpt_image.generate_variation(b'r', 'p')
    assert post.call_args.kwargs['data']['model'] == 'gpt-image-2.5-sunburst'


def test_a_fresh_install_ships_the_two_model_settings_blank(app):
    """Blank, NOT a copy of the default: a literal in DEFAULTS would be a non-empty
    cfg.get() that silently outranked an environment variable someone had set."""
    from app import config as cfg
    with app.app_context():
        assert cfg.get('engines.nanobanana_model') == ''
        assert cfg.get('engines.chatgpt_image_model') == ''
        # The Codex ROUTER model is a different setting and is untouched.
        assert cfg.get('engines.chatgpt_subscription_model') == 'gpt-5.4-mini'


# --- 3. the precedence, including the pre-existing env vars -----------------

@pytest.mark.parametrize('setting,env,expected', [
    ('gemini-from-settings', 'gemini-from-env', 'gemini-from-settings'),
    ('',                     'gemini-from-env', 'gemini-from-env'),
    ('   ',                  'gemini-from-env', 'gemini-from-env'),
    ('gemini-from-settings', None,              'gemini-from-settings'),
    ('',                     None,              'gemini-3-pro-image'),
])
def test_nanobanana_precedence_is_setting_then_env_then_default(app, monkeypatch,
                                                                setting, env, expected):
    """Somebody who exported NANOBANANA_MODEL before this field existed must not
    have their choice ignored in silence — the env var still beats the built-in
    default, and only an explicit slug in Settings outranks it."""
    from app import config as cfg
    from lds_api_engines import nanobanana
    if env is None:
        monkeypatch.delenv('NANOBANANA_MODEL', raising=False)
    else:
        monkeypatch.setenv('NANOBANANA_MODEL', env)
    with app.app_context():
        cfg.save_config({'engines': {'nanobanana_model': setting}})
        assert nanobanana.get_model() == expected


@pytest.mark.parametrize('setting,env,expected', [
    ('gpt-from-settings', 'gpt-from-env', 'gpt-from-settings'),
    ('',                  'gpt-from-env', 'gpt-from-env'),
    ('   ',               'gpt-from-env', 'gpt-from-env'),
    ('gpt-from-settings', None,           'gpt-from-settings'),
    ('',                  None,           'gpt-image-2.5-sunburst'),
])
def test_chatgpt_precedence_is_setting_then_env_then_default(app, monkeypatch,
                                                             setting, env, expected):
    from app import config as cfg
    from lds_api_engines import chatgpt_image
    if env is None:
        monkeypatch.delenv('CHATGPT_IMAGE_MODEL', raising=False)
    else:
        monkeypatch.setenv('CHATGPT_IMAGE_MODEL', env)
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_image_model': setting}})
        assert chatgpt_image.get_image_model() == expected


def test_an_env_var_set_after_import_is_still_honoured(app, monkeypatch):
    """Both variables used to be read at IMPORT, so a value exported later in the
    process was ignored. Reading at call time fixes that too."""
    monkeypatch.delenv('NANOBANANA_MODEL', raising=False)
    from lds_api_engines import chatgpt_image, nanobanana
    with app.app_context():
        assert nanobanana.get_model() == nanobanana.DEFAULT_MODEL
        monkeypatch.setenv('NANOBANANA_MODEL', 'late-gemini')
        monkeypatch.setenv('CHATGPT_IMAGE_MODEL', 'late-gpt')
        assert nanobanana.get_model() == 'late-gemini'
        assert chatgpt_image.get_image_model() == 'late-gpt'


def test_an_explicit_model_argument_still_wins_over_everything(app, monkeypatch):
    """The per-call override the fan-out may pass is above all of it."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    monkeypatch.setenv('NANOBANANA_MODEL', 'from-env')
    from app import config as cfg
    from lds_api_engines import nanobanana
    with app.app_context():
        cfg.save_config({'engines': {'nanobanana_model': 'from-settings'}})
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(200, _gemini_ok())) as post:
            nanobanana.generate_variation(b'r', 'p', model='from-the-caller')
    assert 'from-the-caller:generateContent' in post.call_args.args[0]


# --- 4. a refused model fails loudly, and names the model -------------------

def test_an_unknown_gemini_model_is_fatal_and_names_it(app, monkeypatch):
    """404 on a slug that does not exist: every other row would 404 too, so the
    batch stops — and the message points at the field that caused it."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    from app import config as cfg
    from lds_api_engines import nanobanana
    with app.app_context():
        cfg.save_config({'engines': {'nanobanana_model': 'gemini-9-imaginary'}})
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(404, _gemini_err(
                       'models/gemini-9-imaginary is not found for API version v1beta',
                       404, 'NOT_FOUND'))):
            with pytest.raises(nanobanana.NanoBananaFatal) as e:
                nanobanana.generate_variation(b'r', 'p')
    msg = str(e.value)
    assert 'gemini-9-imaginary' in msg
    assert 'is not found' in msg                    # Google's own words, verbatim
    assert 'Settings > Image engines' in msg


def test_a_gemini_model_that_cannot_take_reference_images_is_fatal(app, monkeypatch):
    """THE failure this feature makes possible: a text-only model. The dataset
    generator always sends reference photos, so it can never work — and the user
    must read that, not 'often a content-policy refusal'."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    from app import config as cfg
    from lds_api_engines import nanobanana
    detail = ('Model does not support the requested response modalities: image')
    with app.app_context():
        cfg.save_config({'engines': {'nanobanana_model': 'gemini-3-pro'}})
        # 400 twice: the first is consumed by the imageConfig retry, so the
        # classification has to survive that retry to be reached at all.
        with patch('lds_api_engines.nanobanana.requests.post',
                   side_effect=[_resp(400, _gemini_err(detail)),
                                _resp(400, _gemini_err(detail))]) as post:
            with pytest.raises(nanobanana.NanoBananaFatal) as e:
                nanobanana.generate_variation(b'r', 'p')
    assert post.call_count == 2
    msg = str(e.value)
    assert 'gemini-3-pro' in msg and 'reference images' in msg
    assert 'modalities' in msg
    assert 'Settings > Image engines' in msg


def test_an_unknown_openai_model_is_fatal_and_names_it(app, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    from app import config as cfg
    from lds_api_engines import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_image_model': 'gpt-image-99'}})
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(404, _openai_err(
                       "The model 'gpt-image-99' does not exist",
                       code='model_not_found', param='model'))):
            with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
                chatgpt_image.generate_variation(b'r', 'p')
    msg = str(e.value)
    assert 'gpt-image-99' in msg and 'does not exist' in msg
    assert 'Settings > Image engines' in msg


def test_the_organization_verification_403_keeps_the_provider_reason_and_settings_action(app, monkeypatch):
    """Keep the upstream cause and the owner's settings action without promising
    that switching model will bypass the provider's account requirements."""
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    from app import config as cfg
    from lds_api_engines import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_image_model': 'gpt-image-1.5'}})
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(403, _openai_err(
                       'Your organization must be verified to use the model '
                       '`gpt-image-1.5`'))):
            with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
                chatgpt_image.generate_variation(b'r', 'p')
    msg = str(e.value)
    assert 'gpt-image-1.5' in msg
    assert 'verification' in msg or 'verified' in msg
    assert 'Plugins → API image engines → Settings' in msg


def test_a_model_openai_will_not_edit_with_is_fatal(app, monkeypatch):
    """A text-to-image-only model refused by /images/edits: a 400 that blames the
    model, not the prompt. It must stop the batch, unlike a moderation 400."""
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    from app import config as cfg
    from lds_api_engines import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_image_model': 'dall-e-3'}})
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(400, _openai_err(
                       "Invalid value: 'dall-e-3'. Supported values are: 'gpt-image-2'",
                       param='model'))):
            with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
                chatgpt_image.generate_variation(b'r', 'p')
    assert 'dall-e-3' in str(e.value)
    assert 'image input' in str(e.value)


def test_a_moderation_400_is_still_a_plain_row_failure(app, monkeypatch):
    """The mirror invariant, and the non-regression that matters most: a refused
    PROMPT must not be dressed up as a model problem, and must not stop a batch
    the rest of which would have rendered.

    Rewritten on purpose: this used to assert a silent `None`, which the fan-out
    worded as "empty response", i.e. as if the app had come back empty-handed for
    unknown reasons. OpenAI named the cause in that body, so the engine now says
    it (ChatGPTImageRefused). What must NOT change is fatality — a refusal still
    costs one row, never the run. See test_chatgpt_refusal.py."""
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    from lds_api_engines import chatgpt_image
    from app.services.engine_errors import EngineFatal
    with app.app_context():
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(400, _openai_err(
                       'Your request was rejected as a result of our safety system',
                       code='moderation_blocked'))):
            with pytest.raises(chatgpt_image.ChatGPTImageRefused) as e:
                chatgpt_image.generate_variation(b'r', 'p')
    assert 'safety system refused' in str(e.value)
    assert not isinstance(e.value, EngineFatal)         # one row, not the batch


def test_a_missing_key_says_so_instead_of_looking_like_a_refusal(app, monkeypatch):
    """Both engines used to return None here, which the fan-out words as 'empty
    response (often a content-policy refusal)' — sending the user to rewrite a
    prompt when the fix is pasting a key."""
    from lds_api_engines import chatgpt_image, nanobanana
    with app.app_context():
        with patch('lds_api_engines.nanobanana.requests.post') as post:
            with pytest.raises(nanobanana.NanoBananaFatal) as e:
                nanobanana.generate_variation(b'r', 'p')
        post.assert_not_called()
        assert 'GEMINI_API_KEY' in str(e.value) and 'Settings' in str(e.value)
        with patch('lds_api_engines.chatgpt_image.requests.post') as post:
            with pytest.raises(chatgpt_image.ChatGPTImageFatal) as e:
                chatgpt_image.generate_variation(b'r', 'p', model='x')
        post.assert_not_called()
        assert 'OPENAI_API_KEY' in str(e.value) and 'Settings' in str(e.value)


@pytest.mark.parametrize('status,expected', [
    (429, 'rate-limited'),
    (503, 'HTTP 503'),
])
def test_a_transient_gemini_failure_stays_per_row(app, monkeypatch, status, expected):
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    from lds_api_engines import nanobanana
    with app.app_context():
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(status, _gemini_err('busy', status))):
            with pytest.raises(nanobanana.NanoBananaError) as e:
                nanobanana.generate_variation(b'r', 'p')
    assert not isinstance(e.value, nanobanana.NanoBananaFatal)
    assert expected in str(e.value)


@pytest.mark.parametrize('status,expected', [
    (429, 'rate-limited'),
    (500, 'HTTP 500'),
])
def test_a_transient_openai_failure_stays_per_row(app, monkeypatch, status, expected):
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    from lds_api_engines import chatgpt_image
    with app.app_context():
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(status, _openai_err('busy'))):
            with pytest.raises(chatgpt_image.ChatGPTImageError) as e:
                chatgpt_image.generate_variation(b'r', 'p')
    assert not isinstance(e.value, chatgpt_image.ChatGPTImageFatal)
    assert expected in str(e.value)


def test_a_200_with_no_image_on_gemini_now_names_the_refusal(app, monkeypatch):
    """A safety block: Gemini answers 200 with a text-only candidate.

    CONTRACT CHANGE, on purpose. This used to assert `is None` — the engine
    returned nothing and the fan-out guessed at the cause, wording it "empty
    response (often a content-policy refusal or a transient API error - retry
    usually works)". That guess was wrong in both directions at once, so the
    engine now RAISES with the cause read out of the response. `None` on this
    engine is gone, not accidentally lost; see test_nanobanana_refusal.py for the
    full behaviour, and nanobanana.py for why no remedy is offered."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    from lds_api_engines import nanobanana
    from app.services.engine_errors import EngineFatal, EngineRefused
    with app.app_context():
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(200, {'candidates': [
                       {'content': {'parts': [{'text': 'I cannot help with that'}]}}]})):
            with pytest.raises(EngineRefused) as e:
                nanobanana.generate_variation(b'r', 'p')
    # The model answered in words: relay them rather than paraphrase a block.
    assert 'I cannot help with that' in str(e.value)
    # A refusal costs one row, never the batch (invariant 4 is about the MODEL).
    assert not isinstance(e.value, EngineFatal)
    assert GEMINI_KEY not in str(e.value)


@pytest.mark.parametrize('status,body', [
    (401, 'unauthorized'), (403, 'forbidden'), (404, 'no such model'),
    (429, 'slow down'), (500, 'boom'),
])
def test_no_message_or_log_line_can_carry_either_key(app, monkeypatch, caplog,
                                                     status, body):
    """The invariant that must hold on EVERY failure path, including the ones a
    user is most likely to paste into a public help channel."""
    monkeypatch.setenv('GEMINI_API_KEY', GEMINI_KEY)
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI_KEY)
    from lds_api_engines import chatgpt_image, nanobanana
    caplog.set_level(logging.DEBUG)
    with app.app_context():
        with patch('lds_api_engines.nanobanana.requests.post',
                   return_value=_resp(status, _gemini_err(body, status), text=body)):
            with pytest.raises(nanobanana.NanoBananaError) as e:
                nanobanana.generate_variation(b'r', 'p')
        assert GEMINI_KEY not in str(e.value) and GEMINI_KEY[6:] not in str(e.value)
        with patch('lds_api_engines.chatgpt_image.requests.post',
                   return_value=_resp(status, _openai_err(body), text=body)):
            with pytest.raises(chatgpt_image.ChatGPTImageError) as e:
                chatgpt_image.generate_variation(b'r', 'p')
        assert OPENAI_KEY not in str(e.value) and OPENAI_KEY[8:] not in str(e.value)
    assert GEMINI_KEY not in caplog.text and OPENAI_KEY not in caplog.text


# --- 5. the batch stop, now shared by all three API engines -----------------

class _SerialPool:
    """Deterministic stand-in for ThreadPoolExecutor (the real 3-worker pool races
    on the shared in-memory sqlite). Concurrency is not what is under test."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, items): return [fn(i) for i in items]


@pytest.mark.parametrize('engine,exc_path', [
    ('nanobanana', 'lds_api_engines.nanobanana.NanoBananaFatal'),
    ('chatgpt', 'lds_api_engines.chatgpt_image.ChatGPTImageFatal'),
])
def test_a_wrong_model_stops_the_batch_on_every_engine(app, monkeypatch, engine, exc_path):
    """Asking the provider the same refused question once per row costs the user
    time and, on a partially-billed batch, money. OpenRouter already did this; the
    two older engines now do too, through the shared EngineFatal."""
    import concurrent.futures
    import importlib
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    mod_name, cls_name = exc_path.rsplit('.', 1)
    fatal_cls = getattr(importlib.import_module(mod_name), cls_name)
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise fatal_cls('does not serve the model "typo-9000" — check the model '
                        'in Settings > Image engines')

    monkeypatch.setattr(svc, '_api_generate_fn', lambda e: boom)
    with app.app_context():
        import io
        import os
        from PIL import Image
        buf = io.BytesIO(); Image.new('RGB', (64, 64), (255, 0, 0)).save(buf, 'PNG')
        ds = svc.create_dataset(LOCAL_USER, 'M', 'm')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model=engine)
                for _ in range(3)]
        svc.db.session.add_all(rows); svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(r.id, 'p', '1:1') for r in rows],
                                  [buf.getvalue()], engine=engine)
        assert len(calls) == 1                      # rows 2-3 never reached the API
        svc.db.session.expire_all()
        for r in rows:
            row = svc.db.session.get(FaceDatasetImage, r.id)
            assert row.status == 'failed'
            assert f'{engine}:' in row.fail_reason
            assert 'typo-9000' in row.fail_reason
            assert 'remaining rows were stopped' in row.fail_reason


def test_the_openrouter_fatal_still_stops_the_batch_through_the_shared_base(app):
    """Rebasing OpenRouter's exceptions onto the shared taxonomy must not have
    unhooked the behaviour it shipped with."""
    from app.services.engine_errors import EngineError, EngineFatal
    from lds_api_engines.openrouter import OpenRouterError, OpenRouterFatal
    assert issubclass(OpenRouterFatal, EngineFatal)
    assert issubclass(OpenRouterError, EngineError)
    assert not issubclass(OpenRouterError, EngineFatal)
