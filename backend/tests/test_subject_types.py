"""Subject-type selector (Human / Animal / Creature / Object / Other) — pure module.

A dataset declares WHAT its subject is; the generation prompts stop assuming a
person. Two load-bearing invariants guard the feature:
  1. the HUMAN path is byte-identical to the pre-feature behaviour (every legacy
     dataset stores subject_type NULL -> 'human'), and
  2. labels are GLOBALLY UNIQUE across every catalog, because the by-label
     resolvers (prompt_by_label / aspect_for_label / is_nsfw_label) search the
     union with no subject_type threading — a collision would resolve in silence.
These tests lock both, plus the identity-lock / catalog / preset wiring.
"""
import re

import pytest

from app.services import face_variations as fv


# --- normalize / registry ----------------------------------------------------

def test_subject_types_registry():
    assert fv.SUBJECT_TYPES == ('human', 'animal', 'creature', 'object', 'other')
    assert set(fv._SUBJECT_CATALOGS) == set(fv.SUBJECT_TYPES)


@pytest.mark.parametrize('raw,expected', [
    ('human', 'human'), ('ANIMAL', 'animal'), ('  object ', 'object'),
    ('', 'human'), (None, 'human'), ('nonsense', 'human'), ('person', 'human'),
])
def test_normalize_subject_type(raw, expected):
    assert fv.normalize_subject_type(raw) == expected


# --- GARDE-FOU 1: human path byte-identical ----------------------------------

def test_human_catalog_is_the_original_object():
    assert fv.variation_catalog('human') is fv.VARIATION_CATALOG
    assert fv.variation_catalog(None) is fv.VARIATION_CATALOG
    assert fv.variation_catalog('unknown') is fv.VARIATION_CATALOG
    assert fv.nsfw_variation_catalog('human') is fv.NSFW_VARIATION_CATALOG


def test_human_identity_defaults_are_byte_identical():
    assert fv.identity_prompt_default('face_single') == fv.IDENTITY_GUARD
    assert fv.identity_prompt_default('face_single', 'human') == fv.IDENTITY_GUARD
    assert fv.identity_prompt_default('face_multi', 'human') == fv.IDENTITY_GUARD_MULTI
    assert fv.identity_prompt_default('klein_identity', 'human') == fv.IDENTITY_GUARD_KLEIN


def test_human_wrappers_are_byte_identical_default():
    """No subject_type given -> exactly the historical output (the reproducibility
    invariant every existing dataset relies on)."""
    assert fv.wrap_variation('a portrait').startswith(fv.IDENTITY_GUARD)
    assert fv.wrap_variation('a portrait', ref_count=3).startswith(fv.IDENTITY_GUARD_MULTI)
    kl = fv.wrap_variation_klein('upper body portrait', framing='bust')
    assert 'same person' in kl and fv.IDENTITY_GUARD_KLEIN in kl
    # explicit 'human' must equal the implicit default
    assert fv.wrap_variation('a portrait', subject_type='human') == fv.wrap_variation('a portrait')
    assert fv.wrap_variation_klein('p', framing='bust', subject_type='human') == \
        fv.wrap_variation_klein('p', framing='bust')


def test_human_presets_unchanged():
    assert len(fv.select_preset('balanced_25')) == 25
    assert len(fv.select_preset('balanced_25', 'human')) == 25
    assert fv.preset_meta_for('human') == []   # human keeps its frontend PRESET_META


# --- identity locks per subject ---------------------------------------------

def test_non_human_identity_defaults():
    assert fv.identity_prompt_default('face_single', 'animal') == fv.IDENTITY_GUARD_ANIMAL
    assert fv.identity_prompt_default('face_multi', 'object') == fv.IDENTITY_GUARD_OBJECT_MULTI
    assert fv.identity_prompt_default('klein_identity', 'creature') == fv.IDENTITY_GUARD_CREATURE_KLEIN
    # klein_improve is subject-agnostic
    for st in fv.SUBJECT_TYPES:
        assert fv.identity_prompt_default('klein_improve', st) == fv.KLEIN_IMAGE_IMPROVE_PROMPT


def test_non_human_locks_drop_human_face_vocabulary():
    """An animal/object lock must not talk about jawline/skin tone/facial identity."""
    human_only = re.compile(r'jawline|skin tone|facial identity|facial expression|beautify, slim', re.I)
    for st in ('animal', 'object', 'creature', 'other'):
        for kind in ('face_single', 'face_multi', 'klein_identity'):
            assert not human_only.search(fv.identity_prompt_default(kind, st)), (st, kind)


def _patch_overrides(monkeypatch, mapping):
    import app.config as cfg

    def fake_get(key, default=None):
        if key.startswith('identity_prompts.'):
            return mapping.get(key.split('.', 1)[1], default)
        return default
    monkeypatch.setattr(cfg, 'get', fake_get)


@pytest.fixture(autouse=True)
def _reset_config_cache():
    import app.config as _cfg
    _cfg._cache = None
    yield
    _cfg._cache = None


def test_blank_override_falls_back_to_subject_default(monkeypatch):
    _patch_overrides(monkeypatch, {'face_single': '   '})
    assert fv.get_identity_prompt('face_single', 'animal') == fv.IDENTITY_GUARD_ANIMAL
    assert fv.get_identity_prompt('face_single', 'human') == fv.IDENTITY_GUARD


def test_global_override_wins_for_every_subject_type(monkeypatch):
    """Documented corner: the override is GLOBAL — it wins for all subject types,
    keeping the editable-identity feature's flat config schema untouched."""
    _patch_overrides(monkeypatch, {'face_single': 'MY LOCK'})
    for st in fv.SUBJECT_TYPES:
        assert fv.get_identity_prompt('face_single', st) == 'MY LOCK'


# --- wrappers thread subject_type -------------------------------------------

def test_wrap_variation_uses_subject_lock():
    an = fv.wrap_variation('a full body shot', subject_type='animal')
    assert an.startswith(fv.IDENTITY_GUARD_ANIMAL)
    assert 'jawline' not in an and 'skin tone' not in an
    ob_multi = fv.wrap_variation('a product shot', ref_count=2, subject_type='object')
    assert ob_multi.startswith(fv.IDENTITY_GUARD_OBJECT_MULTI)


def test_wrap_variation_klein_uses_subject_noun_and_lock():
    an = fv.wrap_variation_klein('a full body shot', framing='body', subject_type='animal')
    assert 'same animal' in an and fv.IDENTITY_GUARD_ANIMAL_KLEIN in an
    ob = fv.wrap_variation_klein('a shot', framing='face', subject_type='object')
    assert 'same object' in ob and fv.IDENTITY_GUARD_OBJECT_KLEIN in ob
    other = fv.wrap_variation_klein('a shot', framing='body', subject_type='other')
    assert 'same subject' in other


# --- catalogs ----------------------------------------------------------------

def test_non_human_catalog_shape():
    for st in ('animal', 'creature', 'object', 'other'):
        cat = fv.variation_catalog(st)
        assert cat and cat is not fv.VARIATION_CATALOG
        ids = [e['id'] for e in cat]
        assert len(ids) == len(set(ids)), f'{st} ids not unique'
        for e in cat:
            assert e['framing'] in ('face', 'bust', 'body', 'back'), (st, e['id'])
            assert set(e) >= {'id', 'axis', 'framing', 'label', 'prompt'}


def test_non_human_catalogs_are_not_outfit_expression_augmented():
    """The outfit/expression bake is a HUMAN concern — it must never touch an
    animal/object catalog (an object has no outfit or expression)."""
    for st in ('animal', 'creature', 'object', 'other'):
        for e in fv.variation_catalog(st):
            assert fv.OUTFIT_VARY not in e['prompt'], (st, e['id'])
            assert fv.EXPRESSION_NEUTRAL not in e['prompt'], (st, e['id'])


def test_all_catalog_prompts_and_labels_are_english():
    # French accented letters, or a few unambiguous French words that never occur
    # in English shot descriptions — a robust "did a prompt slip into French?" net
    # that will not false-positive on English words like "animal" or "profile".
    accents = re.compile(r'[àâäéèêëïîôöùûüç]', re.I)
    french_words = re.compile(
        r'\b(objet|corps|visage|debout|assis|couch\w+|derriere|tete|cote)\b', re.I)
    # Non-human catalogs only: the human catalog legitimately carries the English
    # loanword "café" (its own test guards its French), and is byte-frozen here.
    non_human = (fv.ANIMAL_CATALOG + fv.CREATURE_CATALOG + fv.OBJECT_CATALOG + fv.OTHER_CATALOG)
    for e in non_human:
        text = f"{e['prompt']} {e['label']}"
        assert not accents.search(text), f"Accented French in {e['id']}: {text!r}"
        assert not french_words.search(text), f"French word in {e['id']}: {text!r}"


def test_no_nsfw_catalog_for_non_human_types():
    for st in ('animal', 'creature', 'object', 'other'):
        assert fv.nsfw_variation_catalog(st) == []


# --- GARDE-FOU 2: labels globally unique across every catalog ----------------

def test_labels_globally_unique():
    """Load-bearing invariant, NOT decorative: prompt_by_label / aspect_for_label /
    is_nsfw_label search the union of every catalog with only the label. A label
    shared between two catalogs would resolve to the wrong entry in silence, so
    this test MUST fail if any collision is ever introduced."""
    labels = [e['label'] for e in fv._ALL_CATALOGS]
    dupes = sorted({lbl for lbl in labels if labels.count(lbl) > 1})
    assert not dupes, f'label collision across catalogs: {dupes}'
    assert len(labels) == len(set(labels))


def test_non_human_labels_resolve_via_union():
    """A stored non-human label recovers its prompt/aspect from the union (the path
    regenerate uses when the raw prompt is missing), and is never mistaken for NSFW."""
    for st in ('animal', 'object', 'creature', 'other'):
        for e in fv.variation_catalog(st):
            assert fv.prompt_by_label(e['label']) == e['prompt'], e['id']
            assert fv.aspect_for_label(e['label'], e['framing']) == fv.aspect_for_entry(e), e['id']
            assert fv.is_nsfw_label(e['label']) is False, e['id']


# --- presets -----------------------------------------------------------------

def test_non_human_presets_resolve_and_have_meta():
    for st in ('animal', 'creature', 'object', 'other'):
        presets = fv.presets_for(st)
        assert presets, st
        meta = fv.preset_meta_for(st)
        assert meta and all(m['key'] in presets for m in meta), st
        for name in presets:
            got = fv.select_preset(name, st)
            assert len(got) == len(presets[name]) > 0, (st, name)


# --- persistence (DB): every query wrapped in app_context (release-workflow trap) ---

def test_subject_type_persists_and_defaults_to_human(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        # Legacy-style create (no subject_type) -> column NULL -> 'human' at read.
        ds0 = svc.create_dataset(LOCAL_USER, 'Legacy', 'zlegacy')
        assert ds0.subject_type is None
        assert svc.subject_type_of(ds0) == 'human'
        # Explicit animal create persists.
        ds1 = svc.create_dataset(LOCAL_USER, 'Rex', 'zrex', subject_type='animal')
        assert ds1.subject_type == 'animal'
        assert svc.subject_type_of(ds1) == 'animal'
        # Update round-trips and normalises garbage back to 'human'.
        svc.update_dataset_settings(LOCAL_USER, ds1.id, subject_type='object')
        assert svc.get_dataset(LOCAL_USER, ds1.id).subject_type == 'object'
        svc.update_dataset_settings(LOCAL_USER, ds1.id, subject_type='nonsense')
        assert svc.get_dataset(LOCAL_USER, ds1.id).subject_type == 'human'


# --- route: human byte-identical, non-human carries meta --------------------

def test_variations_route_human_is_byte_identical(client):
    base = client.get('/api/dataset/variations').get_json()
    explicit = client.get('/api/dataset/variations?subject_type=human').get_json()
    assert base == explicit
    # No extra keys on the human payload — exactly the pre-feature shape.
    assert set(base) == {'catalog', 'nsfw_catalog', 'presets'}
    assert len(base['catalog']) == len(fv.VARIATION_CATALOG)
    assert 'balanced_25' in base['presets']


def test_variations_route_animal_payload(client):
    d = client.get('/api/dataset/variations?subject_type=animal').get_json()
    assert d['subject_type'] == 'animal'
    assert d['nsfw_catalog'] == []          # no NSFW catalog for animals
    assert d['preset_meta'] and d['catalog']
    assert any('Animal' in e['label'] for e in d['catalog'])
    # An unknown subject_type falls back to the human payload shape.
    unknown = client.get('/api/dataset/variations?subject_type=zzz').get_json()
    assert set(unknown) == {'catalog', 'nsfw_catalog', 'presets'}
