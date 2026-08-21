"""Editable identity / quality prompts (feature request @bbsorry / 雨田壹).

The four identity "locks" (API single/multi guard, Klein identity block, Klein
improve instruction) are exposed as GLOBAL overrides with a Restore-default path,
plus an enable toggle for the Klein-improve step. The load-bearing invariant: with
NO override configured, every prompt stays byte-identical to the historical
hardcoded constant — the reproducibility guarantee test_face_variations already
locks. These tests cover the NEW override / toggle paths without disturbing it.
"""
import pytest

from app.services import face_variations as fv


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """load_config() caches on a module global not keyed on LDS_CONFIG. These
    tests monkeypatch cfg.get (never save_config), but reset the cache around
    each case anyway so an override can never leak into a later default-path test
    (the ordering trap the conftest documents)."""
    import app.config as _cfg
    _cfg._cache = None
    yield
    _cfg._cache = None


def _patch_overrides(monkeypatch, mapping):
    """Make cfg.get answer identity_prompts.* from a nested `mapping` shaped like
    the real config node, walked dotted-path style exactly as config.get does."""
    import app.config as cfg

    def fake_get(key, default=None):
        if not key.startswith('identity_prompts.'):
            return default
        node = mapping
        for part in key.split('.')[1:]:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    monkeypatch.setattr(cfg, 'get', fake_get)


# --- defaults / registry -----------------------------------------------------

def test_default_registry_matches_constants():
    assert fv.identity_prompt_default('face_single') == fv.IDENTITY_GUARD
    assert fv.identity_prompt_default('face_multi') == fv.IDENTITY_GUARD_MULTI
    assert fv.identity_prompt_default('klein_identity') == fv.IDENTITY_GUARD_KLEIN
    assert fv.identity_prompt_default('klein_improve') == fv.KLEIN_IMAGE_IMPROVE_PROMPT
    # The registry now also holds the five prompt PARTS that used to be
    # hardcoded; the four identity kinds keep their exact meaning above.
    assert (set(fv.IDENTITY_PROMPT_KINDS) | set(fv.PROMPT_PART_KINDS)
            == set(fv._IDENTITY_PROMPT_DEFAULTS))


def test_identity_prompt_defaults_returns_all_four_constants():
    d = fv.identity_prompt_defaults()
    for kind, const in (('face_single', fv.IDENTITY_GUARD),
                        ('face_multi', fv.IDENTITY_GUARD_MULTI),
                        ('klein_identity', fv.IDENTITY_GUARD_KLEIN),
                        ('klein_improve', fv.KLEIN_IMAGE_IMPROVE_PROMPT)):
        assert d[kind] == const, kind
    # ...alongside the five parts that became editable in the same mechanism.
    assert set(d) == set(fv.IDENTITY_PROMPT_KINDS) | set(fv.PROMPT_PART_KINDS)
    # a copy, not the live registry — a mutating caller cannot corrupt defaults
    d['face_single'] = 'x'
    assert fv.identity_prompt_default('face_single') == fv.IDENTITY_GUARD


def test_get_identity_prompt_returns_default_when_unconfigured(monkeypatch):
    _patch_overrides(monkeypatch, {})
    for kind in fv.IDENTITY_PROMPT_KINDS:
        assert fv.get_identity_prompt(kind) == fv.identity_prompt_default(kind)


@pytest.mark.parametrize('blank', ['', '   ', '\n\t '])
def test_blank_override_falls_back_to_default(monkeypatch, blank):
    _patch_overrides(monkeypatch, {k: blank for k in fv.IDENTITY_PROMPT_KINDS})
    for kind in fv.IDENTITY_PROMPT_KINDS:
        assert fv.get_identity_prompt(kind) == fv.identity_prompt_default(kind)


def test_nonblank_override_wins(monkeypatch):
    _patch_overrides(monkeypatch, {'face_single': 'MY CUSTOM GUARD'})
    assert fv.get_identity_prompt('face_single') == 'MY CUSTOM GUARD'
    # the other kinds keep their defaults
    assert fv.get_identity_prompt('face_multi') == fv.IDENTITY_GUARD_MULTI


# --- wrappers honour the override -------------------------------------------

def test_wrap_variation_uses_single_and_multi_overrides(monkeypatch):
    _patch_overrides(monkeypatch, {'face_single': 'SINGLE-LOCK.', 'face_multi': 'MULTI-LOCK.'})
    assert fv.wrap_variation('a portrait') == 'SINGLE-LOCK. a portrait'
    assert fv.wrap_variation('a portrait', ref_count=3) == 'MULTI-LOCK. a portrait'


def test_wrap_variation_klein_uses_identity_override(monkeypatch):
    _patch_overrides(monkeypatch, {'klein_identity': 'KEEP THE FACE.'})
    out = fv.wrap_variation_klein('a portrait', framing='face')
    assert 'KEEP THE FACE. Professional realistic photograph, SFW.' in out
    # the default block text is gone, replaced by the override
    assert 'Restage the shot to match this description' not in out


def test_default_path_is_byte_identical_without_override(monkeypatch):
    """The invariant: unconfigured, every wrapper equals its hardcoded default."""
    _patch_overrides(monkeypatch, {})
    assert fv.wrap_variation('p').startswith(fv.IDENTITY_GUARD)
    assert fv.wrap_variation('p', ref_count=2).startswith(fv.IDENTITY_GUARD_MULTI)
    kl = fv.wrap_variation_klein('p', framing='bust')
    assert fv.IDENTITY_GUARD_KLEIN in kl


# --- config defaults are additive & blank -----------------------------------

def test_config_defaults_are_additive_and_blank():
    from app.config import DEFAULTS
    ip = DEFAULTS['identity_prompts']
    assert ip == {'face_single': '', 'face_multi': '', 'klein_identity': '',
                  'klein_improve': '', 'klein_improve_enabled': True,
                  'markings_lock': '', 'outfit_vary': '', 'expression_neutral': '',
                  'outfit_palette': '', 'render_tail_sfw': '', 'render_tail_nsfw': '',
                  'framing_face': '', 'framing_bust': '', 'framing_body': '',
                  'framing_back': '',
                  'by_subject': {}}
    # EVERY editable prompt part ships blank: blank is what makes the default path
    # byte-identical, and a shipped copy of the default text would freeze it.
    assert all(ip[k] == '' for k in fv.PROMPT_PART_KINDS)


# --- D: Klein-improve toggle + override (service path) -----------------------

def _improve_source(svc, user_id):
    import io
    import os
    from PIL import Image
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(user_id, 'Improve', 'improve')
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    buf = io.BytesIO(); Image.new('RGB', (96, 64), (10, 20, 30)).save(buf, 'PNG')
    with open(os.path.join(svc._dataset_dir(ds.id), 'source.png'), 'wb') as fh:
        fh.write(buf.getvalue())
    image = FaceDatasetImage(dataset_id=ds.id, filename='source.png', source='import',
                             status='keep', framing='body', caption='c',
                             variation_label='Imported low-resolution image',
                             variation_prompt='p')
    svc.db.session.add(image); svc.db.session.commit()
    return ds, image


def _run_improve(app, monkeypatch, overrides):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import klein_edit_helper as keh
    queued = []
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(keh, 'enqueue_klein_edit',
                        lambda **kw: (queued.append(kw) or 'improve-job'))
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda _d: None)

    def fake_get(key, default=None):
        if key.startswith('identity_prompts.'):
            return overrides.get(key.split('.', 1)[1], default)
        return default
    monkeypatch.setattr(svc.cfg, 'get', fake_get)
    with app.app_context():
        _ds, source = _improve_source(svc, LOCAL_USER)
        result = svc.improve_existing_image(LOCAL_USER, source.id)
        candidate = svc.db.session.get(FaceDatasetImage, result['candidate_id'])
        return candidate, queued


def test_klein_improve_default_when_enabled_no_override(app, monkeypatch):
    from app.services import face_dataset_service as svc
    candidate, queued = _run_improve(app, monkeypatch, {})
    assert queued[0]['edit_prompt'] == svc.KLEIN_IMAGE_IMPROVE_PROMPT
    assert candidate.variation_prompt == svc.KLEIN_IMAGE_IMPROVE_PROMPT


def test_klein_improve_uses_override_when_enabled(app, monkeypatch):
    candidate, queued = _run_improve(
        app, monkeypatch, {'klein_improve_enabled': True, 'klein_improve': 'sharpen only'})
    assert queued[0]['edit_prompt'] == 'sharpen only'
    assert candidate.variation_prompt == 'sharpen only'


def test_klein_improve_disabled_applies_no_prompt(app, monkeypatch):
    candidate, queued = _run_improve(
        app, monkeypatch, {'klein_improve_enabled': False, 'klein_improve': 'ignored'})
    assert queued[0]['edit_prompt'] == ''
    assert candidate.variation_prompt == ''


# --- settings API: persist + restore round-trip -----------------------------

def test_settings_api_persists_and_restores_override(client):
    # set an override
    r = client.put('/api/settings', json={'config': {'identity_prompts': {
        'face_single': 'CUSTOM API GUARD', 'klein_improve_enabled': False}}})
    assert r.status_code == 200
    ip = r.get_json()['config']['identity_prompts']
    assert ip['face_single'] == 'CUSTOM API GUARD'
    assert ip['klein_improve_enabled'] is False
    # restore default = save blank; blank is preserved (no guard-loop pop)
    r2 = client.put('/api/settings', json={'config': {'identity_prompts': {
        'face_single': '', 'klein_improve_enabled': True}}})
    ip2 = r2.get_json()['config']['identity_prompts']
    assert ip2['face_single'] == ''
    assert ip2['klein_improve_enabled'] is True


# --- the label contract: klein_identity belongs to EVERY local engine --------

def test_every_local_engine_sends_the_klein_identity_lock(monkeypatch):
    """`identity_prompts.klein_identity` is the LOCAL-ENGINE identity lock, not
    Klein's alone: Klein and Krea 2 Edit share `_compose_edit_prompt` and both
    read it, deliberately — one lock to keep in sync instead of two.

    The UI is now labelled on that fact ("Local engines — restage & face-identity
    block"), so this has to stay a fact. It was mislabelled for one release and a
    Krea 2 user asked on Discord whether the box reached them at all; the failure
    mode of the reverse — a new local engine that skips this lock, or an API
    engine that starts reading it — is a user carefully editing a text that does
    nothing, with nothing on screen to tell them.

    Driven off face_dataset_service.LOCAL_ENGINES, so adding a third local engine
    fails HERE, at the contract, rather than quietly making the label a lie.
    """
    from app.services.face_dataset_service import API_ENGINES, LOCAL_ENGINES

    sentinel = 'ZZ-LOCAL-ENGINE-IDENTITY-LOCK-ZZ'
    # Both storage layouts at once: human keeps the flat key, every other subject
    # lives under by_subject.<type> (see identity_prompt_config_key).
    _patch_overrides(monkeypatch, {
        'klein_identity': sentinel,
        'by_subject': {st: {'klein_identity': sentinel} for st in fv.SUBJECT_TYPES},
    })

    for st in fv.SUBJECT_TYPES:
        for engine in LOCAL_ENGINES:
            res = fv.compose_preview(engine, subject_type=st)
            # An engine compose_preview does not know falls back to Klein — which
            # DOES carry the lock, so the assertion below would pass on an engine
            # nobody wired up. The echoed id is what makes this test bite.
            assert res['engine'] == engine, f'{engine} is not wired into compose_preview'
            assert sentinel in res['prompt'], f'{engine} / {st} ignores klein_identity'
        for engine in API_ENGINES:
            prompt = fv.compose_preview(engine, subject_type=st)['prompt']
            assert sentinel not in prompt, f'{engine} / {st} reads klein_identity'


def test_the_preview_dispatch_covers_every_known_engine():
    """The contract above only proves what compose_preview routes. An engine
    missing from PREVIEW_ENGINES falls back to Klein there, which would make the
    test above pass by accident on an engine nobody wired up."""
    from app.services.face_dataset_service import KNOWN_ENGINES

    assert set(KNOWN_ENGINES) == set(fv.PREVIEW_ENGINES)
