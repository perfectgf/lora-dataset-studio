"""Gemini's output filter: a refusal must be reported AS a refusal.

NOTHING here calls the real Gemini API. `requests.post` is patched in every
test, so no key is read, no quota is spent, and the suite does not depend on
Google being up — which is also this file's honest limit: it proves how we READ
an answer we hand ourselves, not that Google produces that shape today.

WHAT IS BEING LOCKED, and why it is worth a file of its own
-----------------------------------------------------------
Gemini screens the image it just produced. When that screen trips, the API
answers **HTTP 200 with an empty candidate list** — a success envelope with no
picture in it. Before this file existed, `generate_variation` handed that back
as `None` and the dataset fan-out wrote one sentence for it:

    "empty response (often a content-policy refusal or a transient API error -
     retry usually works)"

That sentence had to guess, and it guessed wrongly in both directions at once:
it told a user Google had permanently refused that "retry usually works", and it
told a user with a broken connection that they had written something forbidden.

So, in order of how badly each hurts when broken:
  1. a 200 with no image says the FILTER refused it, and relays Google's own
     reason code when there is one;
  2. a real malfunction (unreachable host, rejected key, quota, 5xx) keeps its
     own distinct message and never reads as a content refusal — nor the reverse;
  3. a refusal is NOT fatal: a 40-image batch with 12 refusals runs all 40 and
     ends with an exact count, instead of stopping at the first one or leaving
     12 silent holes;
  4. a normal response is untouched.

The messages promise no workaround, because there is none to promise: the output
filter is not configurable, and it is not deterministic (see nanobanana.py).
"""
import base64
import logging
from unittest.mock import MagicMock, patch

import pytest

KEY = 'AIzaSyTESTKEYVALUE0123456789abcdefghij'
PNG = b'\x89PNG\r\n\x1a\nfake-pixels'


def _resp(status=200, json_body=None, text=''):
    r = MagicMock(status_code=status)
    if json_body is None:
        r.json.side_effect = ValueError('no json')
    else:
        r.json.return_value = json_body
    r.text = text
    return r


def _image_body():
    """A normal, successful generateContent answer."""
    return {'candidates': [{'content': {'parts': [
        {'inlineData': {'mimeType': 'image/png',
                        'data': base64.b64encode(PNG).decode()}}]},
        'finishReason': 'STOP'}]}


def _filtered_body(reason='IMAGE_SAFETY'):
    """What the output filter actually returns: 200, a candidate, no image."""
    return {'candidates': [{'content': {'parts': []}, 'finishReason': reason}],
            'usageMetadata': {'promptTokenCount': 12}}


def _prompt_blocked_body(reason='SAFETY'):
    """The other half of the safety stack: refused BEFORE generating."""
    return {'promptFeedback': {'blockReason': reason, 'safetyRatings': []}}


# --- 1. the refusal is named ------------------------------------------------

def test_a_200_with_no_image_raises_instead_of_returning_none(app, monkeypatch):
    """THE regression this file exists for. The silent `None` is gone: a refused
    request now carries its cause out of the engine."""
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    from lds_api_engines import nanobanana
    from app.services.engine_errors import EngineRefused
    with patch('lds_api_engines.nanobanana.requests.post',
               return_value=_resp(200, _filtered_body())):
        with pytest.raises(EngineRefused) as e:
            nanobanana.generate_variation([b'ref'], 'a portrait')
    msg = str(e.value)
    assert "image filter refused this image" in msg
    assert 'IMAGE_SAFETY' in msg                      # Google's own reason relayed
    assert 'not configurable' in msg                  # the fact that has no remedy
    assert 'LDS cannot turn it off' in msg


def test_the_refusal_message_never_promises_a_workaround():
    """The measured behaviour is that the same prompt passes about half the time,
    so "retry" is a coin toss. The message must not sell it as a fix, and must
    not invent prompt advice the filter's false positives would make false."""
    from lds_api_engines.nanobanana import refusal_message
    for body in (_filtered_body(), _filtered_body('PROHIBITED_CONTENT'),
                 _prompt_blocked_body(), {}, {'candidates': []}):
        msg = refusal_message(body).lower()
        assert 'retry' not in msg
        assert 'try again' not in msg
        assert 'rephrase' not in msg
        assert 'usually works' not in msg


def test_a_prompt_side_block_is_told_apart_from_an_output_side_one():
    """These are genuinely different: the prompt-side categories ARE the four the
    API exposes, the output filter is not exposed at all. One sentence for both
    would misdescribe whichever it was not written for."""
    from lds_api_engines.nanobanana import refusal_message, refusal_detail
    prompt_side = refusal_message(_prompt_blocked_body('SAFETY'))
    assert 'blocked the prompt before generating' in prompt_side
    assert 'SAFETY' in prompt_side
    assert refusal_detail(_prompt_blocked_body())['scope'] == 'prompt'

    image_side = refusal_message(_filtered_body())
    assert 'image filter refused this image' in image_side
    assert refusal_detail(_filtered_body())['scope'] == 'image'
    assert prompt_side != image_side


def test_a_text_only_answer_relays_the_words_gemini_actually_wrote():
    """Not every empty response is the filter: the model sometimes answers in
    prose. Paraphrasing that as a policy block would be a second guess."""
    from lds_api_engines.nanobanana import refusal_message
    body = {'candidates': [{'content': {'parts': [
        {'text': "I can't help with that request."}]}, 'finishReason': 'STOP'}]}
    msg = refusal_message(body)
    assert 'answered with text instead of an image' in msg
    assert "I can't help with that request." in msg


def test_an_unknown_reason_code_still_reads_as_a_refusal():
    """Google adds finishReason values without notice. An unrecognised one on a
    200-with-no-image must not fall through to a sentence about the network."""
    from lds_api_engines.nanobanana import refusal_message
    msg = refusal_message(_filtered_body('SOME_FUTURE_CODE_2027'))
    assert 'refused' in msg
    assert 'SOME_FUTURE_CODE_2027' in msg


@pytest.mark.parametrize('body', [
    None, 'a string', 42, [], {'candidates': 'not-a-list'},
    {'candidates': ['not-a-dict']}, {'candidates': [{'content': 'not-a-dict'}]},
    {'candidates': [{'content': {'parts': 'not-a-list'}}]},
    {'promptFeedback': 'not-a-dict'},
])
def test_a_malformed_body_still_produces_a_sentence_instead_of_a_crash(body):
    """This runs on whatever Google sends, outside any try/except in the caller.
    A shape we did not anticipate must degrade to the generic refusal, not turn
    a content refusal into an AttributeError the user reads as an app bug."""
    from lds_api_engines.nanobanana import refusal_message, refusal_detail
    assert isinstance(refusal_detail(body), dict)
    msg = refusal_message(body)
    assert isinstance(msg, str) and 'refused' in msg


def test_a_refusal_with_no_reason_at_all_says_so_rather_than_inventing_one():
    from lds_api_engines.nanobanana import refusal_message
    msg = refusal_message({'candidates': [{'content': {'parts': []}}]})
    assert 'no reason given' in msg


def test_snake_case_spellings_are_read_too():
    """The REST envelope is camelCase and the protos are snake_case; both have
    been seen. Reading only one spelling would silently lose the reason."""
    from lds_api_engines.nanobanana import refusal_detail
    assert refusal_detail(
        {'prompt_feedback': {'block_reason': 'SAFETY'}})['reason'] == 'SAFETY'
    assert refusal_detail({'candidates': [
        {'finish_reason': 'IMAGE_SAFETY'}]})['scope'] == 'image'


# --- 2. a malfunction is NOT a refusal --------------------------------------

@pytest.mark.parametrize('status,body,expected,fatal', [
    (401, {'error': {'message': 'API key not valid'}}, 'rejected the API key', True),
    (404, {'error': {'message': 'not found'}}, 'does not serve the model', True),
    (429, {'error': {'message': 'Quota exceeded'}}, 'rate-limited', False),
    (500, {'error': {'message': 'internal'}}, 'HTTP 500', False),
    (503, {'error': {'message': 'overloaded'}}, 'HTTP 503', False),
])
def test_a_real_malfunction_keeps_its_own_message(app, monkeypatch, status, body,
                                                  expected, fatal):
    """Replacing a silence with the WRONG explanation would be worse than the
    silence. A key, a quota or an outage must never read as a content refusal."""
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    from lds_api_engines import nanobanana
    from app.services.engine_errors import EngineFatal, EngineRefused
    with patch('lds_api_engines.nanobanana.requests.post',
               return_value=_resp(status, body)):
        with pytest.raises(nanobanana.NanoBananaError) as e:
            nanobanana.generate_variation([b'ref'], 'a portrait')
    assert expected in str(e.value)
    assert not isinstance(e.value, EngineRefused)       # never mislabelled
    assert 'image filter' not in str(e.value)
    assert isinstance(e.value, EngineFatal) is fatal
    assert KEY not in str(e.value)


def test_an_unreachable_host_is_not_a_content_refusal(app, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    import requests
    from lds_api_engines import nanobanana
    from app.services.engine_errors import EngineRefused
    with patch('lds_api_engines.nanobanana.requests.post',
               side_effect=requests.ConnectionError('name resolution failed')):
        with pytest.raises(nanobanana.NanoBananaError) as e:
            nanobanana.generate_variation([b'ref'], 'a portrait')
    assert 'could not reach Gemini' in str(e.value)
    assert not isinstance(e.value, EngineRefused)


def test_a_missing_key_is_fatal_and_not_a_refusal(app):
    from lds_api_engines import nanobanana
    from app.services.engine_errors import EngineFatal, EngineRefused
    with pytest.raises(EngineFatal) as e:
        nanobanana.generate_variation([b'ref'], 'a portrait')
    assert 'no Gemini API key saved' in str(e.value)
    assert not isinstance(e.value, EngineRefused)


def test_no_message_or_log_record_ever_carries_the_key(app, monkeypatch, caplog):
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    from lds_api_engines import nanobanana
    from app.services.engine_errors import EngineRefused
    caplog.set_level(logging.DEBUG)
    with patch('lds_api_engines.nanobanana.requests.post',
               return_value=_resp(200, _filtered_body())):
        with pytest.raises(EngineRefused) as e:
            nanobanana.generate_variation([b'ref'], 'a portrait')
    assert KEY not in str(e.value)
    assert KEY not in caplog.text


# --- 3. a partially-refused batch runs to the end and counts ----------------

class _SerialPool:
    """ThreadPoolExecutor stand-in: same API, one thread, deterministic order."""
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, items):
        return [fn(i) for i in items]


def _real_png():
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(buf, format='PNG')
    return buf.getvalue()


def test_a_batch_with_refusals_finishes_every_row_and_counts_them_exactly(
        app, monkeypatch, caplog):
    """40 rows, 12 refused. The refusals must NOT stop the run (the filter is not
    deterministic — the other 28 deserve their attempt), every refused tile must
    name the cause, and the totals must add up. Twelve silent holes is the
    outcome this whole change exists to prevent."""
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from lds_api_engines.nanobanana import NanoBananaRefused
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    caplog.set_level(logging.INFO)

    attempts = []

    def flaky(*a, **k):
        n = len(attempts)
        attempts.append(n)
        if n % 10 in (3, 7, 8):            # 12 of 40
            raise NanoBananaRefused(
                "Google's image filter refused this image (IMAGE_SAFETY). "
                'That filter is not configurable — LDS cannot turn it off.')
        return _real_png()

    monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: flaky)
    with app.app_context():
        import os
        ds = svc.create_dataset(LOCAL_USER, 'NB', 'nb')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending',
                                 klein_model='nanobanana') for _ in range(40)]
        svc.db.session.add_all(rows)
        svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(r.id, 'p', '1:1') for r in rows],
                                  [_real_png()], engine='nanobanana')

        assert len(attempts) == 40, 'the batch stopped early on a refusal'
        svc.db.session.expire_all()
        done = [svc.db.session.get(FaceDatasetImage, r.id) for r in rows]
        refused = [d for d in done if d.fail_kind == 'refused']
        made = [d for d in done if d.filename]
        assert len(refused) == 12
        assert len(made) == 28
        assert len(refused) + len(made) == 40      # nothing fell in a hole
        for d in refused:
            assert d.status == 'failed'
            assert d.fail_reason.startswith('nanobanana: ')
            assert 'image filter refused' in d.fail_reason
        # …and the run itself reports the count instead of leaving it to be
        # discovered tile by tile.
        assert '12 refused by the provider' in caplog.text


def test_a_refusal_is_not_fatal_but_a_rejected_key_still_is(app, monkeypatch):
    """The pair that must never merge: a refusal costs one row, a rejected key
    costs the run. Same engine, same batch entry point, opposite handling."""
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from lds_api_engines.nanobanana import NanoBananaFatal, NanoBananaRefused
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)

    for raised, expect_calls, expect_kind in (
            (NanoBananaRefused("Google's image filter refused this image."), 3, 'refused'),
            (NanoBananaFatal('Gemini rejected the API key (HTTP 401)'), 1, 'error')):
        calls = []

        def boom(*a, _e=raised, **k):
            calls.append(1)
            raise _e

        monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: boom)
        with app.app_context():
            import os
            ds = svc.create_dataset(LOCAL_USER, f'NB{expect_kind}', f'nb{expect_kind}')
            os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
            rows = [FaceDatasetImage(dataset_id=ds.id, status='pending',
                                     klein_model='nanobanana') for _ in range(3)]
            svc.db.session.add_all(rows)
            svc.db.session.commit()
            svc._run_nanobanana_batch(app, [(r.id, 'p', '1:1') for r in rows],
                                      [_real_png()], engine='nanobanana')
            assert len(calls) == expect_calls
            svc.db.session.expire_all()
            first = svc.db.session.get(FaceDatasetImage, rows[0].id)
            assert first.fail_kind == expect_kind
        # The fatal one, and only the fatal one, announces that it cut the run.
        assert ('remaining rows were stopped' in (first.fail_reason or '')) \
            is (expect_kind == 'error')


def test_an_engine_that_cannot_explain_itself_says_that_instead_of_guessing(
        app, monkeypatch):
    """ChatGPT and OpenRouter can return 200-with-no-image and genuinely cannot
    tell a refusal from a hiccup. They keep a vague message — but an HONEST one:
    it states the ambiguity instead of promising that retrying works."""
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    monkeypatch.setattr(svc, '_api_generate_fn', lambda engine: (lambda *a, **k: None))
    with app.app_context():
        import os
        ds = svc.create_dataset(LOCAL_USER, 'ORX', 'orx')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        row = FaceDatasetImage(dataset_id=ds.id, status='pending',
                               klein_model='openrouter')
        svc.db.session.add(row)
        svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(row.id, 'p', '1:1')], [_real_png()],
                                  engine='openrouter')
        svc.db.session.expire_all()
        saved = svc.db.session.get(FaceDatasetImage, row.id)
        assert saved.fail_kind == 'empty'          # NOT counted as a refusal
        assert 'no image and no reason given' in saved.fail_reason
        assert 'retry usually works' not in saved.fail_reason


def test_a_database_created_before_fail_kind_gains_it_on_boot(app):
    """The counting relies on a column that did not exist yesterday, and a live
    SQLite schema is frozen at creation. So the additive migration is exercised
    for real: drop the column, boot the migration, check it comes back and that
    the rows already in the table survive with a NULL kind (never a guessed one).
    """
    from sqlalchemy import text
    from app import _SCHEMA_ADDITIONS, _apply_additive_migrations
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage, db
    from app.services import face_dataset_service as svc

    assert ('face_dataset_image', 'fail_kind', 'VARCHAR(16)') in _SCHEMA_ADDITIONS

    def columns():
        return {r[1] for r in db.session.execute(
            text('PRAGMA table_info(face_dataset_image)'))}

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'legacy', 'legacy')
        row = FaceDatasetImage(dataset_id=ds.id, status='failed',
                               fail_reason='nanobanana: something old')
        db.session.add(row)
        db.session.commit()
        row_id = row.id

        db.session.execute(text('ALTER TABLE face_dataset_image DROP COLUMN fail_kind'))
        db.session.commit()
        assert 'fail_kind' not in columns()               # a genuinely older schema

        _apply_additive_migrations()
        assert 'fail_kind' in columns()
        _apply_additive_migrations()                      # idempotent: runs every boot
        assert 'fail_kind' in columns()

        db.session.expire_all()
        old = db.session.get(FaceDatasetImage, row_id)
        assert old.fail_reason == 'nanobanana: something old'
        assert old.fail_kind is None                      # unclassified, not invented


# --- 4. anti-regression: a normal answer is untouched ------------------------

def test_a_normal_response_still_returns_the_image(app, monkeypatch):
    """The whole point of raising on refusals is lost if the success path moved."""
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    from lds_api_engines import nanobanana
    with patch('lds_api_engines.nanobanana.requests.post',
               return_value=_resp(200, _image_body())) as post:
        out = nanobanana.generate_variation([b'ref-a', b'ref-b'], 'a portrait',
                                            aspect_ratio='3:4')
    assert out == PNG
    payload = post.call_args.kwargs['json']
    parts = payload['contents'][0]['parts']
    assert parts[0]['text'] == 'a portrait'
    assert len(parts) == 3                                  # prompt + both refs
    assert payload['generationConfig']['imageConfig']['aspectRatio'] == '3:4'


def test_a_successful_batch_row_records_no_failure_kind(app, monkeypatch):
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (lambda *a, **k: _real_png()))
    with app.app_context():
        import os
        ds = svc.create_dataset(LOCAL_USER, 'NBOK', 'nbok')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        row = FaceDatasetImage(dataset_id=ds.id, status='pending',
                               klein_model='nanobanana')
        svc.db.session.add(row)
        svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(row.id, 'p', '1:1')], [_real_png()],
                                  engine='nanobanana')
        svc.db.session.expire_all()
        saved = svc.db.session.get(FaceDatasetImage, row.id)
        assert saved.filename and saved.fail_kind is None and saved.fail_reason is None


def test_a_wide_generated_image_is_not_stored_below_the_dataset_minimum_side(app, monkeypatch):
    import concurrent.futures
    import io
    import os
    from PIL import Image
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    def rendered(width, height):
        buf = io.BytesIO()
        Image.new('RGB', (width, height), 'red').save(buf, format='PNG')
        return buf.getvalue()

    outputs = iter([rendered(1536, 1024), rendered(2048, 2048)])
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    monkeypatch.setattr(svc, '_api_generate_fn',
                        lambda engine: (lambda *a, **k: next(outputs)))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NBWIDE', 'nbwide')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='nanobanana')
                for _ in range(2)]
        svc.db.session.add_all(rows)
        svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(rows[0].id, 'p', '3:2'), (rows[1].id, 'p', '1:1')],
                                  [_real_png()], engine='nanobanana')
        svc.db.session.expire_all()
        sizes = []
        for row in rows:
            saved = svc.db.session.get(FaceDatasetImage, row.id)
            with Image.open(os.path.join(svc._dataset_dir(ds.id), saved.filename)) as stored:
                sizes.append(stored.size)
        # The limit is 1024 on the SHORT side. A long-side limit stored the wide
        # frame at 1024x683, under the 768 minimum a dataset accepts.
        assert sizes == [(1536, 1024), (1024, 1024)]


def test_the_imageconfig_retry_still_happens_before_any_refusal_is_declared(
        app, monkeypatch):
    """A model that rejects imageConfig answers 400 on the first payload. That is
    a retry, not a refusal, and it must not be reported as one."""
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    from lds_api_engines import nanobanana
    responses = [_resp(400, {'error': {'message': 'imageConfig unsupported'}}),
                 _resp(200, _image_body())]
    with patch('lds_api_engines.nanobanana.requests.post',
               side_effect=responses) as post:
        out = nanobanana.generate_variation([b'ref'], 'a portrait')
    assert out == PNG
    assert post.call_count == 2
    assert 'imageConfig' not in str(post.call_args.kwargs['json'])
