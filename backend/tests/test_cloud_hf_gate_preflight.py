"""A gated Hugging Face repo the account was never granted answers 403 ON THE POD —
after a GPU is rented and billed. Three runs were lost that way to krea/Krea-2-Raw,
each costing a rental and ~3 minutes, and the card showed only "403 Client Error
(Request ID…)" while the sentence naming the repo sat unseen in the payload.

The gate is checkable in one request before anything is reserved. These tests pin
both halves: it refuses when HF refuses, and it never blocks a launch for any other
reason — an outage must not ground a run that would have worked.
"""
import urllib.error

import pytest

from app.services import cloud_training as ct


def _http_error(code):
    return urllib.error.HTTPError('u', code, 'x', None, None)


def _patch(monkeypatch, raiser):
    monkeypatch.setattr(ct.urllib.request if hasattr(ct, 'urllib') else __import__('urllib.request', fromlist=['x']),
                        'urlopen', raiser, raising=False)


def test_a_refused_gate_stops_the_launch_before_anything_is_rented(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(_http_error(403)))
    with pytest.raises(ValueError) as err:
        ct._assert_official_base_reachable('krea/Krea-2-Raw', 'hf_token')
    msg = str(err.value)
    assert 'krea/Krea-2-Raw' in msg                     # names the repo to unlock
    assert 'huggingface.co/krea/Krea-2-Raw' in msg      # and the page to open
    assert 'cost' in msg.lower()                        # says nothing was rented


def test_an_unauthenticated_refusal_is_treated_the_same(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(_http_error(401)))
    with pytest.raises(ValueError):
        ct._assert_official_base_reachable('black-forest-labs/FLUX.1-dev', None)


@pytest.mark.parametrize('boom', [
    _http_error(500), _http_error(404), TimeoutError('slow'), OSError('offline'),
])
def test_it_fails_OPEN_on_anything_that_is_not_a_refusal(monkeypatch, boom):
    """An HF outage, a rename, a flaky network — none of those mean the user cannot
    train. The pod stays the real authority; grounding a good run would be worse
    than the problem this solves."""
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(boom))
    ct._assert_official_base_reachable('krea/Krea-2-Turbo', 'hf_token')   # no raise


def test_no_repo_to_check_is_a_no_op(monkeypatch):
    """Custom local weights resolve to None — there is nothing for HF to refuse."""
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *a, **k: (_ for _ in ()).throw(_http_error(403)))
    ct._assert_official_base_reachable(None, 'hf_token')


def test_the_resolver_names_the_repo_each_family_actually_downloads(app):
    """The check is only worth as much as the repo id it is given, and that id must
    track the recipes rather than restate them."""
    from types import SimpleNamespace
    from app.services import lora_training as lt
    with app.app_context():
        def ds(train_type, variant=None, base=None):
            return SimpleNamespace(id=1, user_id='local', trigger_word='t',
                                   train_type=train_type, train_variant=variant,
                                   train_base_model=base)
        assert lt.official_base_repo(ds('krea')) == 'krea/Krea-2-Raw'
        assert lt.official_base_repo(ds('krea', 'turbo')) == 'krea/Krea-2-Turbo'
        assert lt.official_base_repo(ds('flux')) == 'black-forest-labs/FLUX.1-dev'
        assert lt.official_base_repo(ds('zimage')) == lt.ZIMAGE_TURBO_BASE
        assert lt.official_base_repo(ds('zimage', 'base')) == lt.ZIMAGE_BASE
        assert lt.official_base_repo(ds('zimage', 'deturbo')) == lt.ZIMAGE_DETURBO_BASE
        # custom local weights: nothing is fetched from HF, so nothing to check
        assert lt.official_base_repo(ds('krea', base='C:/models/mine.safetensors')) is None
