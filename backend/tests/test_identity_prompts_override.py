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
    """Make cfg.get answer identity_prompts.<kind> from `mapping`, default else."""
    import app.config as cfg

    def fake_get(key, default=None):
        if key.startswith('identity_prompts.'):
            return mapping.get(key.split('.', 1)[1], default)
        return default

    monkeypatch.setattr(cfg, 'get', fake_get)


# --- defaults / registry -----------------------------------------------------

def test_default_registry_matches_constants():
    assert fv.identity_prompt_default('face_single') == fv.IDENTITY_GUARD
    assert fv.identity_prompt_default('face_multi') == fv.IDENTITY_GUARD_MULTI
    assert fv.identity_prompt_default('klein_identity') == fv.IDENTITY_GUARD_KLEIN
    assert fv.identity_prompt_default('klein_improve') == fv.KLEIN_IMAGE_IMPROVE_PROMPT
    assert set(fv.IDENTITY_PROMPT_KINDS) == set(fv._IDENTITY_PROMPT_DEFAULTS)


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
                  'klein_improve': '', 'klein_improve_enabled': True}


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
