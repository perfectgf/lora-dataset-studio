import re

from app.services.face_variations import (VARIATION_CATALOG, NSFW_VARIATION_CATALOG,
    select_preset, aspect_for_framing, composition_counts, drop_identity_tags,
    OUTFIT_VARY, EXPRESSION_NEUTRAL, _HAS_OUTFIT, _HAS_EXPRESSION,
    wrap_variation, wrap_variation_klein)


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


# --- Anti-leak guardrails (outfit / expression) ------------------------------
def test_every_shot_directs_outfit_and_expression():
    """Owner's field report: edit models copy the reference's OUTFIT and EXPRESSION
    when the prompt is silent. Guard: every SFW shot must name an outfit source
    (explicit garment OR the variation directive) so nothing is left to the
    reference default, and every face-bearing shot must name an expression."""
    for e in VARIATION_CATALOG:
        p = e['prompt']
        assert _HAS_OUTFIT.search(p) or OUTFIT_VARY in p, f"{e['id']}: no outfit directive"
        if e['framing'] != 'back':
            assert _HAS_EXPRESSION.search(p) or EXPRESSION_NEUTRAL in p, \
                f"{e['id']}: no expression directive"


def test_silent_shots_carry_the_variation_directive():
    """The entries that used to be SILENT on outfit (e.g. bust_outdoor, every plain
    body/face framing) must now carry the explicit vary directive — the exact leak
    the owner hit on Bust shots."""
    for eid in ('bust_outdoor', 'bust_studio', 'body_walk', 'body_cafe', 'back_34',
                'face_window', 'face_studio'):
        e = next(x for x in VARIATION_CATALOG if x['id'] == eid)
        assert OUTFIT_VARY in e['prompt'], eid
    # ...and the expression directive on the shots that never named one.
    for eid in ('bust_outdoor', 'bust_jacket', 'body_walk', 'face_window', 'face_golden'):
        e = next(x for x in VARIATION_CATALOG if x['id'] == eid)
        assert EXPRESSION_NEUTRAL in e['prompt'], eid


def test_explicit_outfits_are_preserved_not_overwritten():
    """Shots that DO name an outfit keep it (no double 'casual outfit' directive)."""
    for eid, token in (('bust_swim', 'bikini'), ('body_bodycon', 'bodycon'),
                       ('body_athletic', 'sportswear'), ('bust_jacket', 'jacket'),
                       ('body_swim_pool', 'swimsuit')):
        e = next(x for x in VARIATION_CATALOG if x['id'] == eid)
        assert token in e['prompt'] and OUTFIT_VARY not in e['prompt'], eid


def test_nsfw_shots_keep_state_but_neutralise_expression():
    """NSFW nude/lingerie shots must NOT get a 'casual outfit' directive (it would
    fight the described state) but still get a neutral expression."""
    nude = next(x for x in NSFW_VARIATION_CATALOG if x['id'] == 'nsfw_body_nude_stand')
    assert OUTFIT_VARY not in nude['prompt'] and 'nude' in nude['prompt']
    assert EXPRESSION_NEUTRAL in nude['prompt']


def test_catalog_prompts_are_english():
    """Generation PROMPTS must be English. (Display LABELS stay as their French
    persisted keys by design — they key is_nsfw_label / prompt_by_label on stored
    DB rows, so translating them is a data migration, not a prompt tweak.)"""
    french = re.compile(
        r'\b(buste|corps|dos|visage|robe|haut|tenue|maillot|assis|debout|marche|'
        r'moulante|ajust\w*|s[ée]rieux|sourire|veste|plage|piscine|champ|paysage|'
        r'lumi[èe]re|fen[êe]tre|serviette|allong\w*|douche)\b', re.I)
    for e in VARIATION_CATALOG + NSFW_VARIATION_CATALOG:
        assert not french.search(e['prompt']), f"French in prompt {e['id']}: {e['prompt']}"


def test_wrappers_scope_reference_to_face_identity():
    """Both assembly wrappers must tell the model the reference is for FACE identity
    only, and that outfit + expression come from the description (not the reference)."""
    api = wrap_variation('upper body portrait, front view')
    assert 'do NOT copy the outfit' in api and 'facial expression' in api
    api_multi = wrap_variation('upper body portrait', ref_count=3)
    assert 'do NOT copy the outfit' in api_multi
    kl = wrap_variation_klein('upper body portrait', framing='bust')
    assert 'do not copy' in kl and 'outfit' in kl and 'facial expression' in kl
