from app.services.face_variations import (VARIATION_CATALOG, select_preset,
    aspect_for_framing, composition_counts, drop_identity_tags)


def test_catalog_shape():
    assert len(VARIATION_CATALOG) == 45          # 36 + 9 body-emphasis entries
    frames = [e['framing'] for e in VARIATION_CATALOG]
    assert frames.count('face') == 17 and frames.count('bust') == 10
    assert frames.count('body') == 17 and frames.count('back') == 1
    for e in VARIATION_CATALOG:
        assert set(e) >= {'id', 'axis', 'framing', 'label', 'prompt'}


def test_presets():
    assert len(select_preset('balanced_25')) == 25
    assert len(select_preset('zimage_12')) == 12


def test_body_emphasis_preset():
    """Body-fidelity preset: every id resolves, composition is 8/8/8/1, and the
    figure-visible outfits stay in the API-accepted register (swimwear/fitted —
    the SFW identity guard is prepended by wrap_variation at generation time)."""
    entries = select_preset('body_emphasis')
    assert len(entries) == 25
    c = composition_counts(entries)
    assert c == {'face': 8, 'bust': 8, 'body': 8, 'back': 1}
    prompts = ' '.join(e['prompt'] for e in entries)
    assert 'bikini' in prompts and 'bodycon' in prompts and 'sportswear' in prompts
    # no explicit terms — these prompts must pass API-provider policy as-is
    for banned in ('nude', 'naked', 'topless', 'nsfw'):
        assert banned not in prompts.lower()


def test_aspects():
    assert aspect_for_framing('face') == '1:1'
    assert aspect_for_framing('body') == '3:4'


def test_composition_counts():
    c = composition_counts(select_preset('balanced_25'))
    assert sum(c.values()) == 25
