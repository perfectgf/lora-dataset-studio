from app.services.face_variations import (VARIATION_CATALOG, select_preset,
    aspect_for_framing, composition_counts, drop_identity_tags)


def test_catalog_shape():
    assert len(VARIATION_CATALOG) == 36
    frames = [e['framing'] for e in VARIATION_CATALOG]
    assert frames.count('face') == 17 and frames.count('bust') == 7
    assert frames.count('body') == 11 and frames.count('back') == 1
    for e in VARIATION_CATALOG:
        assert set(e) >= {'id', 'axis', 'framing', 'label', 'prompt'}


def test_presets():
    assert len(select_preset('balanced_25')) == 25
    assert len(select_preset('zimage_12')) == 12


def test_aspects():
    assert aspect_for_framing('face') == '1:1'
    assert aspect_for_framing('body') == '3:4'


def test_composition_counts():
    c = composition_counts(select_preset('balanced_25'))
    assert sum(c.values()) == 25
