"""Dataset coverage — the variety read under the composition meter.

Two things are worth proving and they are different. The lexicon
(``caption_coverage``) is a pure function of strings: it can be pinned exactly,
so it is, including the false positive it is known to have. The service on top
is about the POOL and the honesty counters — that a rejected image does not
count, that "no captions" says so instead of drawing an empty panel.

Nothing here runs a model: the whole feature is DB rows plus regex.
"""
import pytest


# --- the lexicon, alone -----------------------------------------------------

def test_scan_caption_reads_each_axis():
    from app.services.caption_coverage import scan_caption
    hits = scan_caption('A woman in profile, wearing a leather jacket, '
                        'backlit at sunset on a city street, smiling.')
    assert 'profile' in hits['view']
    assert 'outerwear' in hits['outfit']
    assert {'golden', 'backlit'} <= hits['lighting']
    assert 'urban' in hits['setting']
    assert 'smile' in hits['expression']


def test_hyphen_and_space_spellings_both_match():
    """'three-quarter', 'three quarter' and 'three_quarter' are the same bucket —
    and 't-shirt' must survive the separator handling (re.escape turns the hyphen
    into '\\-', which a naive substitution corrupts into a literal '[')."""
    from app.services.caption_coverage import scan_caption
    for spelling in ('three-quarter view', 'three quarter view', 'three_quarter view'):
        assert 'three_quarter' in scan_caption(spelling)['view'], spelling
    assert 'casual' in scan_caption('wearing a blue t-shirt')['outfit']
    assert 'casual' in scan_caption('wearing a blue tshirt')['outfit']


def test_matching_is_word_bounded():
    """'profiles' or a word merely containing a keyword must not trip a bucket."""
    from app.services.caption_coverage import scan_caption
    assert scan_caption('the darkroom door')['lighting'] == set()
    assert scan_caption('a sundress')['outfit'] == {'dress'}


def test_negation_is_not_parsed_and_we_know_it():
    """Documented false positive: the lexicon is positive-only. Pinned so that a
    future 'clever' change has to face the claim the UI makes about it."""
    from app.services.caption_coverage import scan_caption
    assert 'smile' in scan_caption('she is not smiling')['expression']


def test_axes_are_kind_aware():
    from app.services import caption_coverage as cc
    assert 'expression' in cc.axes_for_kind('character')
    assert 'expression' not in cc.axes_for_kind('concept')
    assert 'outfit' not in cc.axes_for_kind('style')
    assert cc.axes_for_kind(None) == cc.axes_for_kind('character')


def test_advice_names_the_missing_core_buckets():
    """The point of the whole feature: eight front-on studio portraits must be
    TOLD they have no profile and no three-quarter, not shown a green bar."""
    from app.services import caption_coverage as cc
    caps = ['a woman facing the camera, plain background, studio lighting, smiling'] * 8
    rep = cc.analyse(caps, kind='character')
    text = ' | '.join(a['text'] for a in cc.advice(rep))
    assert 'profile' in text and 'three-quarter' in text
    assert 'outfit' in text.lower()        # one outfit type -> variety warning
    assert any(a['tone'] == 'warn' for a in cc.advice(rep))


def test_an_axis_nobody_described_is_unmeasured_not_uniform():
    """Caught by the headless screenshot: every camera-height chip read 0 while
    the advice said "every caption describes the same one". Nothing named an
    angle at all — that is unmeasured, and claiming uniformity from silence is
    the one thing this panel must never do."""
    from app.services import caption_coverage as cc
    caps = ['a woman facing the camera, studio lighting, plain background, '
            'wearing a t-shirt, smiling'] * 8
    lines = {a['text'] for a in cc.advice(cc.analyse(caps, kind='character'))}
    height = [t for t in lines if t.startswith('Camera height')]
    assert height and 'cannot be judged' in height[0]
    assert not any('every caption' in t for t in height)


def test_a_single_named_value_still_reads_as_uniform():
    """The other half of the same branch: when captions DO name the axis and all
    name the same value, that is a real uniformity warning — and it says which."""
    from app.services import caption_coverage as cc
    caps = ['a woman at eye level, facing the camera, daylight, outdoors in a '
            'park, wearing a t-shirt, smiling'] * 8
    warns = [a['text'] for a in cc.advice(cc.analyse(caps, kind='character'))
             if a['text'].startswith('Camera height')]
    assert warns and 'eye level' in warns[0] and 'the same one' in warns[0]


def test_advice_says_when_captions_are_missing_instead_of_showing_nothing():
    from app.services import caption_coverage as cc
    rep = cc.analyse(['', '', None], kind='character')
    notes = cc.advice(rep)
    assert rep['captioned'] == 0 and rep['uncaptioned'] == 3
    assert 'No captions yet' in notes[0]['text']


def test_advice_refuses_to_judge_a_tiny_sample():
    from app.services import caption_coverage as cc
    rep = cc.analyse(['a woman facing the camera'] * 3, kind='character')
    text = ' | '.join(a['text'] for a in cc.advice(rep))
    assert 'too few' in text
    assert 'No profile' not in text


def test_advice_stays_quiet_on_a_varied_set():
    from app.services import caption_coverage as cc
    caps = [
        'facing the camera, daylight, outdoors in a park, wearing jeans, smiling',
        'profile, indoor lighting in a room, formal suit, serious',
        'three-quarter view, overcast light, city street, leather jacket, neutral expression',
        'from behind, night, urban rooftop, bikini, laughing',
        'facing the camera, sunset, beach, sportswear, surprised',
        'side view, studio lighting, plain background, uniform, pensive',
        'low angle, daylight, garden, dress, smiling',
        'high angle, indoor lamp, kitchen, hoodie, serious',
    ]
    warns = [a for a in cc.advice(cc.analyse(caps, kind='character')) if a['tone'] == 'warn']
    assert warns == []


# --- the service on top -----------------------------------------------------

def _seed(svc, ds_id, rows):
    from app.models import FaceDatasetImage
    for i, (status, framing, caption) in enumerate(rows):
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds_id, filename=f'{i}.png', status=status,
            framing=framing, caption=caption))
    svc.db.session.commit()


def test_coverage_endpoint_reports_pool_and_advice(app, client):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Cov', 'cov')
        _seed(svc, ds.id, [('keep', 'face',
                            'a woman facing the camera, studio lighting, '
                            'plain background, wearing a t-shirt, smiling')] * 8)
        ds_id = ds.id
    r = client.get(f'/api/dataset/{ds_id}/coverage')
    assert r.status_code == 200
    p = r.get_json()
    assert p['total'] == 8 and p['captioned'] == 8 and p['uncaptioned'] == 0
    assert p['kind'] == 'character'
    text = ' | '.join(a['text'] for a in p['advice'])
    assert 'profile' in text
    # The panel keys on axis ids -- they are payload contract, not decoration.
    assert [a['id'] for a in p['axes']] == \
        ['view', 'angle', 'lighting', 'setting', 'outfit', 'expression']


def test_coverage_ignores_rejected_and_failed_like_the_meter_above_it(app):
    from app.services import face_dataset_service as svc
    from app.services import dataset_coverage as cov
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Pool', 'pool')
        _seed(svc, ds.id, [
            ('keep', 'face', 'facing the camera'),
            ('pending', 'bust', 'profile view'),
            ('reject', 'body', 'from behind at night'),
            ('failed', 'back', 'three-quarter view'),
        ])
        p = cov.coverage(LOCAL_USER, ds.id)
    assert p['total'] == 2                      # reject + failed excluded
    views = {b['id']: b['count'] for b in p['axes'][0]['buckets']}
    assert views['frontal'] == 1 and views['profile'] == 1
    assert views['from_behind'] == 0 and views['three_quarter'] == 0


def test_coverage_counts_unframed_images_the_bar_silently_drops(app):
    from app.services import face_dataset_service as svc
    from app.services import dataset_coverage as cov
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Unframed', 'unf')
        _seed(svc, ds.id, [('keep', 'face', 'x'), ('keep', None, 'x'),
                           ('keep', 'unknown', 'x')])
        p = cov.coverage(LOCAL_USER, ds.id)
    assert p['framed'] == 1 and p['unframed'] == 2
    assert 'no shot type yet' in p['advice'][0]['text']


def test_coverage_of_a_style_dataset_drops_the_character_axes(app):
    from app.services import face_dataset_service as svc
    from app.services import dataset_coverage as cov
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Sty', 'sty')
        ds.kind = 'style'
        svc.db.session.commit()
        _seed(svc, ds.id, [('keep', 'face', 'daylight outdoors, facing the camera')] * 6)
        p = cov.coverage(LOCAL_USER, ds.id)
    assert p['kind'] == 'style'
    assert [a['id'] for a in p['axes']] == ['lighting', 'setting', 'view']
    text = ' | '.join(a['text'] for a in p['advice'])
    assert 'outfit' not in text.lower() and 'xpression' not in text


def test_coverage_404_on_a_missing_dataset(client):
    assert client.get('/api/dataset/999999/coverage').status_code == 404


# --- which images a bucket counted, not just how many ------------------------

def test_analyse_reports_the_ids_of_each_bucket_in_the_same_pass_as_the_counts():
    """A chip is only a filter if it can name its images, and the number it shows
    must be reproducible from that list — so the two come out of one walk."""
    from app.services.caption_coverage import analyse
    report = analyse(
        ['a woman in profile, smiling', 'facing the camera, neutral', 'in profile at night'],
        ids=[11, 22, 33])
    view = next(a for a in report['axes'] if a['id'] == 'view')
    profile = next(b for b in view['buckets'] if b['id'] == 'profile')
    assert profile['image_ids'] == [11, 33]
    assert profile['count'] == len(profile['image_ids'])
    frontal = next(b for b in view['buckets'] if b['id'] == 'frontal')
    assert frontal['image_ids'] == [22]
    # A bucket nobody mentioned reports an EMPTY list, never a missing key: the
    # panel keys on it to decide whether a chip can be clicked at all.
    behind = next(b for b in view['buckets'] if b['id'] == 'from_behind')
    assert behind['image_ids'] == [] and behind['count'] == 0


def test_analyse_without_ids_keeps_its_old_payload_exactly():
    """The ids are additive. Every caller that does not ask for them (the bank
    variety read among them) must not start receiving a new key."""
    from app.services.caption_coverage import analyse
    report = analyse(['a woman in profile'])
    for axis in report['axes']:
        for bucket in axis['buckets']:
            assert 'image_ids' not in bucket


def test_ids_stay_lined_up_when_an_image_has_no_caption():
    """The empties are dropped before counting; dropping them before PAIRING
    would shift every id after the gap onto the wrong caption."""
    from app.services.caption_coverage import analyse
    report = analyse(['', 'a woman in profile', None, 'in profile too'],
                     ids=[1, 2, 3, 4])
    view = next(a for a in report['axes'] if a['id'] == 'view')
    profile = next(b for b in view['buckets'] if b['id'] == 'profile')
    assert profile['image_ids'] == [2, 4]
    assert report['uncaptioned'] == 2


def test_ids_of_the_wrong_length_are_refused_rather_than_silently_zipped():
    import pytest as _pytest
    from app.services.caption_coverage import analyse
    with _pytest.raises(ValueError):
        analyse(['a', 'b', 'c'], ids=[1, 2])


def test_coverage_buckets_carry_the_ids_of_the_pool_they_counted(app):
    """And only that pool: a rejected image is out of the panel, so it must be
    out of the chip filter too, or clicking would show an image the number
    above it never counted."""
    from app.services import face_dataset_service as svc
    from app.services import dataset_coverage as cov
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Ids', 'ids')
        _seed(svc, ds.id, [
            ('keep', 'face', 'a woman in profile, smiling'),
            ('reject', 'face', 'a woman in profile, smiling'),
            ('keep', 'body', 'facing the camera'),
        ])
        rows = (FaceDatasetImage.query
                .filter(FaceDatasetImage.dataset_id == ds.id)
                .order_by(FaceDatasetImage.id).all())
        kept_profile, rejected_profile = rows[0].id, rows[1].id
        payload = cov.coverage(LOCAL_USER, ds.id)
    view = next(a for a in payload['axes'] if a['id'] == 'view')
    profile = next(b for b in view['buckets'] if b['id'] == 'profile')
    assert profile['image_ids'] == [kept_profile]
    assert rejected_profile not in profile['image_ids']
    assert profile['count'] == 1
