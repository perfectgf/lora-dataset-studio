"""The provider router — and the one property that matters most: nothing moved.

Adding a second local LLM must be invisible to every install that has Ollama.
That is not a nice-to-have here: LDS is public, people are running it, and three
docker-compose files exist purely to wire Ollama up. So the first tests below are
about the DEFAULT, not about LM Studio.

The two accessors at the bottom exist because an adversarial pass found that
`vision_pool` and `vision_keepalive` read `ollama.*` directly: without routing
them, an LM Studio user would get two Settings dials that change nothing.
"""
import pytest

from app import config
from app.services import vision_llm


def test_an_install_that_never_heard_of_this_setting_still_uses_ollama(app):
    """No `local_llm` section at all — the shape every existing config.json has."""
    with app.app_context():
        config.save_config({'ollama': {'url': 'http://127.0.0.1:11434'}})
        assert vision_llm.provider() == 'ollama'
        assert vision_llm.label() == 'Ollama'


def test_a_provider_from_a_newer_version_degrades_to_ollama_instead_of_breaking(app):
    """Downgrade path: a config written by a future build naming a provider this
    one does not have must not brick captioning. Falling back is the only answer
    that keeps the app usable while the user works out what happened."""
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'some-future-thing'}})
        assert vision_llm.provider() == 'ollama'


def test_choosing_lmstudio_routes_the_calls_there(app, monkeypatch):
    seen = {}
    from app.services import vision_lmstudio, vision_ollama

    def _mark(who):
        def _fn(*a, **kw):
            seen['who'] = who
            return 'caption'
        return _fn

    monkeypatch.setattr(vision_lmstudio, 'describe_image', _mark('lmstudio'))
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', _mark('ollama'))
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'}})
        assert vision_llm.describe_image(b'x', 'p') == 'caption'
    assert seen['who'] == 'lmstudio'


def test_the_default_still_reaches_the_ollama_driver_untouched(app, monkeypatch):
    """The Ollama path must keep receiving its OWN kwargs — `fmt`, `num_ctx` and
    `keep_alive` have no LM Studio equivalent and are not reinterpreted for it,
    but they must still arrive intact where they do mean something."""
    got = {}
    from app.services import vision_ollama
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda img, prompt, **kw: got.update(kw) or 'ok')
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'ollama'}})
        vision_llm.describe_image(b'x', 'p', fmt='json', num_ctx=4096, keep_alive=30)
    assert got == {'fmt': 'json', 'num_ctx': 4096, 'keep_alive': 30}


def test_the_model_list_keeps_the_shape_both_pickers_already_read(app, monkeypatch):
    """Dataset and Bank read the same `{ok, reachable, models: [str]}`. Changing
    that shape per provider would break one surface and not the other — exactly
    the Bank/Dataset divergence the repo's rules exist to prevent."""
    from app.services import vision_lmstudio
    monkeypatch.setattr(vision_lmstudio, 'list_models', lambda **kw: {
        'ok': True, 'reachable': True, 'surface': 'v1',
        'models': [{'id': 'qwen/qwen3-vl-4b', 'type': 'vlm', 'loaded': True},
                   {'id': 'nomic-embed', 'type': 'embeddings', 'loaded': False}]})
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'}})
        out = vision_llm.list_models()
    assert out['ok'] is True and out['reachable'] is True
    assert out['models'] == ['qwen/qwen3-vl-4b'], 'an embedding model is not a captioner'


# --- the two dials that would otherwise be inert ------------------------------

@pytest.mark.parametrize('who, concurrency, warm', [
    ('ollama', 4, 120),
    ('lmstudio', 7, 45),
])
def test_the_concurrency_and_keep_warm_dials_follow_the_active_provider(
        app, who, concurrency, warm):
    with app.app_context():
        config.save_config({
            'local_llm': {'provider': who},
            'ollama': {'vision_concurrency': 4, 'vision_keep_warm_seconds': 120},
            'lmstudio': {'vision_concurrency': 7, 'vision_keep_warm_seconds': 45},
        })
        assert vision_llm.vision_concurrency() == concurrency
        assert vision_llm.keep_warm_seconds() == warm


def test_an_unreadable_dial_falls_back_instead_of_raising(app):
    """A corrupted value must cost speed, never the pass — the same contract the
    Ollama concurrency setting already documents."""
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'vision_concurrency': 'lots',
                                         'vision_keep_warm_seconds': None}})
        assert vision_llm.vision_concurrency() == 4
        assert vision_llm.keep_warm_seconds() == 120
