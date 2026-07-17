import re

from app.services.face_variations import (VARIATION_CATALOG, NSFW_VARIATION_CATALOG,
    select_preset, aspect_for_framing, composition_counts, drop_identity_tags,
    OUTFIT_VARY, EXPRESSION_NEUTRAL, _HAS_OUTFIT, _HAS_EXPRESSION,
    wrap_variation, wrap_variation_klein,
    LEGACY_LABEL_ALIASES, canonical_label, is_nsfw_label, prompt_by_label,
    aspect_for_label)


def test_catalog_shape():
    assert len(VARIATION_CATALOG) == 53          # 45 existing + 8 profile variations
    frames = [e['framing'] for e in VARIATION_CATALOG]
    assert frames.count('face') == 25 and frames.count('bust') == 10
    assert frames.count('body') == 17 and frames.count('back') == 1
    for e in VARIATION_CATALOG:
        assert set(e) >= {'id', 'axis', 'framing', 'label', 'prompt'}


def test_left_and_right_profile_variations_are_symmetric():
    by_id = {e['id']: e for e in VARIATION_CATALOG}
    for suffix in ('smile', 'serious', 'look_up', 'rim_light'):
        left = by_id[f'face_profile_l_{suffix}']
        right = by_id[f'face_profile_r_{suffix}']
        assert left['framing'] == right['framing'] == 'face'
        assert left['axis'] == right['axis']
        assert 'left profile view' in left['prompt']
        assert 'right profile view' in right['prompt']


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
    """Generation PROMPTS must be English. (Display LABELS are now English too — see
    test_catalog_labels_are_english — after the FR->EN translation from waltm's PR;
    the pre-migration French labels stay resolvable through LEGACY_LABEL_ALIASES.)"""
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


# --- Label translation + legacy alias (rattraper l'existant) -----------------
# The catalog labels used to be French and are persisted verbatim on every
# generated row (variation_label) and inside dataset backups. is_nsfw_label /
# prompt_by_label / aspect_for_label all resolve a stored label against the live
# catalog, so the FR->EN translation would orphan pre-migration datasets without
# LEGACY_LABEL_ALIASES. (Empirical orphaning-vs-repair proof: prove_orphaning.py.)
_CUR_LABELS = {e['label'] for e in VARIATION_CATALOG + NSFW_VARIATION_CATALOG}


def test_catalog_labels_are_english():
    """The display labels are English now (they used to be French persisted keys).
    Guards against a French label sneaking back into the catalog."""
    french = re.compile(
        r'\b(visage|buste|corps|dos|profil|gauche|droite|serieux|sourire|rire|'
        r'veste|robe|maillot|plage|piscine|champ|paysage|lumiere|fenetre|serviette|'
        r'debout|assis|marche|allong\w*|douche|regard|cadre|habille|exterieur)\b', re.I)
    for e in VARIATION_CATALOG + NSFW_VARIATION_CATALOG:
        assert not french.search(e['label']), f"French label survived: {e['label']!r}"


def test_all_labels_unique():
    """No two entries share a display label, so a stored label resolves to one entry."""
    labels = [e['label'] for e in VARIATION_CATALOG + NSFW_VARIATION_CATALOG]
    assert len(labels) == len(set(labels))


def test_legacy_aliases_are_well_formed():
    """Every alias VALUE is a live catalog label; every KEY is a retired (French)
    label, never a current one; and the targets cover the whole catalog 1:1."""
    assert len(LEGACY_LABEL_ALIASES) == len(VARIATION_CATALOG) + len(NSFW_VARIATION_CATALOG)
    for fr, en in LEGACY_LABEL_ALIASES.items():
        assert en in _CUR_LABELS, f"alias target not in catalog: {en!r}"
        assert fr not in _CUR_LABELS, f"alias key is still a live label: {fr!r}"
    assert set(LEGACY_LABEL_ALIASES.values()) == _CUR_LABELS


def test_canonical_label_passthrough():
    assert canonical_label('Face front, neutral') == 'Face front, neutral'   # current
    assert canonical_label('Visage face, neutre') == 'Face front, neutral'   # legacy
    assert canonical_label('\U0001f51e custom scene').startswith('\U0001f51e')  # custom prompt
    assert canonical_label('') == '' and canonical_label(None) is None


def test_legacy_labels_resolve_identically_to_new_labels():
    """A pre-migration row (French label) resolves to the SAME prompt, NSFW verdict
    and aspect as the current English label of the same shot — the orphaning the
    alias prevents (without it: None prompt, NSFW->False, aspect override lost)."""
    for fr, en in LEGACY_LABEL_ALIASES.items():
        assert is_nsfw_label(fr) == is_nsfw_label(en), fr
        assert prompt_by_label(fr) is not None, fr
        assert prompt_by_label(fr) == prompt_by_label(en), fr
        assert aspect_for_label(fr) == aspect_for_label(en), fr


def test_legacy_nsfw_labels_stay_fail_closed():
    """Every retired French NSFW label must still read as NSFW so the row stays
    pinned to local Klein and never leaks to a third-party API engine."""
    for e in NSFW_VARIATION_CATALOG:
        fr = next(k for k, v in LEGACY_LABEL_ALIASES.items() if v == e['label'])
        assert is_nsfw_label(fr) is True, fr


def test_mixed_era_dataset_resolves_without_duplication():
    """A dataset mixing a legacy FR row and a new EN row of the same shot resolves
    both to one entry (identical prompt + aspect)."""
    assert prompt_by_label('Corps, nu debout') == prompt_by_label('Body, nude standing')
    assert aspect_for_label('Corps, plan large urbain') == aspect_for_label('Body, wide urban shot') == '16:9'


def test_unknown_label_is_inert():
    """An unknown/custom label is not NSFW, has no catalog prompt, and falls back to
    the framing aspect (unchanged behaviour, alias must not swallow it)."""
    assert is_nsfw_label('totally unknown shot') is False
    assert prompt_by_label('totally unknown shot') is None
    assert aspect_for_label('totally unknown shot', 'body') == '3:4'
