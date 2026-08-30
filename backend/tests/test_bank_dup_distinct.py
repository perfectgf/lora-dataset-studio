"""≠ "not duplicates" — the answer the duplicate panel never had.

Before this, a group the user had genuinely settled ("these are a burst, not
copies") could only be answered by rejecting a picture they wanted to keep, or
by skipping — which writes nothing, so the group came back on every run, forever.

The tests that matter here are not "does the button work". They are:

  * the veto is stored as PAIRS, so it survives the renumbering that every run
    of the grouping pass performs (the trap this design exists to avoid);
  * a group that GAINS a member is asked again — a new copy is a new question;
  * nothing is ever rejected by a veto, on either stage;
  * "Resolve ALL" does not collapse a group the user already ruled on;
  * the counters the user reads agree with the list they are given.
"""
import pytest
from PIL import Image

from test_image_bank import _mkbank, checkerboard  # noqa: F401  (shared helpers)


def _group_rows(app, bank_id, gid=1, attr='dup_group'):
    """Put every image of the bank in one group, all undecided."""
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for r in rows:
            setattr(r, attr, gid)
            r.status = 'pending'
        db.session.commit()
        return [r.id for r in rows]


def _unresolved(client, bank_id, path='dup-groups'):
    return client.get(f'/api/bank/{bank_id}/{path}').get_json()['total']


@pytest.fixture()
def three_copies(client, tmp_path, app):
    """A bank of three images the pass has grouped together."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': im, 'b.jpg': im, 'c.jpg': im})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    ids = _group_rows(app, bank_id)
    return bank_id, ids


def test_veto_hides_the_group_and_rejects_nothing(client, three_copies, app):
    bank_id, ids = three_copies
    assert _unresolved(client, bank_id) == 1

    r = client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    body = r.get_json()
    assert body['ok'] is True
    assert body['pairs'] == 3, 'three copies = three pairs'

    assert _unresolved(client, bank_id) == 0, 'the group stops being proposed'
    with app.app_context():
        from app.models import BankImage
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert {r.status for r in rows} == {'pending'}, 'a veto decides nothing'
        assert all(r.reject_reason is None for r in rows)


def test_the_veto_survives_the_group_being_RENUMBERED(client, three_copies, app):
    """The trap this design exists to avoid.

    Both grouping passes renumber the WHOLE bank from scratch on every run, so a
    veto keyed on "group #1" would, after the next scan, silently apply to some
    other set of images. Keyed on the PAIRS it survives, because the pairs are
    what the user actually ruled on."""
    bank_id, ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    assert _unresolved(client, bank_id) == 0

    _group_rows(app, bank_id, gid=4242)          # the pass renumbers everything
    assert _unresolved(client, bank_id) == 0, \
        'same images, new group number — still answered'


def test_a_group_that_GAINS_a_member_is_asked_again(client, tmp_path, app):
    """A new copy is a new question: the pairs with it were never ruled on, so
    the group comes back — with the unruled copy on screen."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': im, 'b.jpg': im, 'c.jpg': im, 'd.jpg': im,
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for r in rows[:3]:
            r.dup_group, r.status = 1, 'pending'
        rows[3].dup_group, rows[3].status = None, 'pending'
        db.session.commit()
        newcomer = rows[3].id

    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    assert _unresolved(client, bank_id) == 0

    with app.app_context():                       # a looser re-group pulls d in
        db.session.get(BankImage, newcomer).dup_group = 1
        db.session.commit()
    assert _unresolved(client, bank_id) == 1, 'the pairs with d were never ruled on'
    listed = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups'][0]
    assert newcomer in [i['id'] for i in listed['images']]


def test_a_split_group_stays_answered(client, three_copies, app):
    """The other half of the same promise: a TIGHTER re-group that splits {a,b,c}
    into {a,b} leaves a group whose every pair is already vetoed, so it does not
    come back to ask a question that was answered."""
    bank_id, ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        db.session.get(BankImage, ids[2]).dup_group = None   # c leaves the group
        db.session.commit()
    assert _unresolved(client, bank_id) == 0


def test_restore_puts_the_group_back(client, three_copies):
    bank_id, _ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    assert _unresolved(client, bank_id) == 0
    r = client.post(f'/api/bank/{bank_id}/dups/distinct',
                    json={'restore': True, 'group': 1})
    assert r.get_json()['restored'] == 3
    assert _unresolved(client, bank_id) == 1


def test_restore_without_a_group_clears_the_whole_bank(client, three_copies):
    bank_id, _ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    r = client.post(f'/api/bank/{bank_id}/dups/distinct', json={'restore': True})
    assert r.get_json()['restored'] == 3
    assert _unresolved(client, bank_id) == 1


def test_resolve_ALL_leaves_a_vetoed_group_alone(client, three_copies, app):
    """"Resolve ALL" means the groups still to rule on. Collapsing a vetoed group
    would reject copies the user explicitly asked to keep — the exact damage the
    feature exists to prevent."""
    bank_id, _ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'best'})
    assert r.get_json()['rejected'] == 0
    with app.app_context():
        from app.models import BankImage
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert all(x.status == 'pending' for x in rows)


def test_naming_a_vetoed_group_explicitly_still_resolves_it(client, three_copies):
    """Naming a group IS ruling on it again — the user is allowed to change their
    mind without hunting for a restore button first."""
    bank_id, _ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    r = client.post(f'/api/bank/{bank_id}/dups/resolve',
                    json={'strategy': 'first', 'group': 1})
    assert r.get_json()['rejected'] == 2


def test_the_counters_agree_with_the_list(client, three_copies):
    bank_id, _ids = three_copies
    before = client.get(f'/api/bank/{bank_id}').get_json()['dup']
    assert before['unresolved'] == 1 and before['not_duplicates'] == 0
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    after = client.get(f'/api/bank/{bank_id}').get_json()['dup']
    assert after['unresolved'] == 0, 'the chip must not send you back to a done group'
    assert after['not_duplicates'] == 1
    assert after['groups'] == before['groups'], 'the pass found the same groups'


def test_the_veto_is_idempotent(client, three_copies):
    bank_id, _ids = three_copies
    assert client.post(f'/api/bank/{bank_id}/dups/distinct',
                       json={'group': 1}).get_json()['pairs'] == 3
    assert client.post(f'/api/bank/{bank_id}/dups/distinct',
                       json={'group': 1}).get_json()['pairs'] == 0
    assert _unresolved(client, bank_id) == 0


def test_semantic_stage_has_the_same_verb(client, tmp_path, app):
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': im, 'b.jpg': im})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    _group_rows(app, bank_id, attr='semantic_dup_group')
    assert _unresolved(client, bank_id, 'semantic-dup-groups') == 1
    r = client.post(f'/api/bank/{bank_id}/semantic-dups/distinct', json={'group': 1})
    assert r.get_json()['pairs'] == 1
    assert _unresolved(client, bank_id, 'semantic-dup-groups') == 0


def test_a_veto_on_one_stage_answers_the_other(client, tmp_path, app):
    """"a and b are different shots" is a fact about the IMAGES, not about which
    algorithm grouped them — so one veto answers both stages. Saying it twice
    would be asking the same question twice."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': im, 'b.jpg': im})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    _group_rows(app, bank_id, attr='dup_group')
    _group_rows(app, bank_id, attr='semantic_dup_group')
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    assert _unresolved(client, bank_id, 'dup-groups') == 0
    assert _unresolved(client, bank_id, 'semantic-dup-groups') == 0


def test_a_bulk_veto_is_refused(client, three_copies):
    """There is deliberately no "mark every group distinct": it would answer
    questions the user never looked at, which is the failure mode of the bulk
    buttons this feature was built to give an alternative to."""
    bank_id, _ids = three_copies
    r = client.post(f'/api/bank/{bank_id}/dups/distinct', json={})
    assert r.status_code == 400
    assert 'group is required' in r.get_json()['error']


def test_a_huge_group_is_refused_with_the_dial_named(client, tmp_path, app):
    """n²/2 rows is the wrong storage for a 3 000-frame group, and a group that
    size means the threshold is wrong. The refusal has to say which dial."""
    from app.services import image_bank_service as banks
    im = Image.new('RGB', (32, 32), 'white')
    bank_id, _src = _mkbank(client, tmp_path, {f'{i}.jpg': im for i in range(4)})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    _group_rows(app, bank_id)
    with app.app_context():
        original = banks._DISTINCT_MAX_MEMBERS
        banks._DISTINCT_MAX_MEMBERS = 3
        try:
            r = client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
        finally:
            banks._DISTINCT_MAX_MEMBERS = original
    assert r.status_code == 400
    assert 'filter thresholds' in r.get_json()['error']


def test_deleting_the_images_takes_their_verdicts(client, three_copies, app):
    """Hygiene, not correctness — an orphan row is inert — but a bank re-scanned
    from the same folder must not accumulate verdicts about ids nobody can see."""
    bank_id, ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    with app.app_context():
        from app.models import BankDupDistinct
        from app.services import image_bank_service as banks
        assert BankDupDistinct.query.filter_by(bank_id=bank_id).count() == 3
        banks.drop_distinct_for_images(bank_id, ids[:1])
        from app.extensions import db
        db.session.commit()
        # a is in both of its pairs, so removing it takes two of the three
        assert BankDupDistinct.query.filter_by(bank_id=bank_id).count() == 1


# --- what the /verif swarm found, pinned so it cannot come back ---------------

def test_restoring_one_stage_leaves_the_OTHER_stage_verdicts_alone(client, tmp_path, app):
    """"Put them back" is offered by ONE panel, under ITS own count.

    It used to delete every ≠ row of the bank, so pressing it in ≈ Duplicates
    also threw away every verdict made in ✂ Same shot — decisions the sentence
    never mentioned, and no way to tell they were gone. A veto stays a fact about
    the images; UNDOING it must not reach past what the button counted."""
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': im, 'b.jpg': im, 'c.jpg': im, 'd.jpg': im,
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for r in rows:
            r.dup_group = r.semantic_dup_group = None
            r.status = 'pending'
        rows[0].dup_group = rows[1].dup_group = 1            # stage 1: a + b
        rows[2].semantic_dup_group = rows[3].semantic_dup_group = 1   # stage 2: c + d
        db.session.commit()

    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    client.post(f'/api/bank/{bank_id}/semantic-dups/distinct', json={'group': 1})
    assert _unresolved(client, bank_id, 'dup-groups') == 0
    assert _unresolved(client, bank_id, 'semantic-dup-groups') == 0

    r = client.post(f'/api/bank/{bank_id}/dups/distinct', json={'restore': True})
    assert r.get_json()['restored'] == 1, 'only the pair of THIS stage'
    assert _unresolved(client, bank_id, 'dup-groups') == 1, 'this panel is restored'
    assert _unresolved(client, bank_id, 'semantic-dup-groups') == 0, \
        'the other panel keeps the verdict it was never asked about'


def test_a_group_id_that_is_not_a_number_is_refused(client, three_copies):
    """int() alone let {"group": true} through as group 1 — a veto on a group the
    caller never named — and {"group": {}} raised TypeError, i.e. a 500."""
    bank_id, _ids = three_copies
    for bad in (True, {}, []):
        r = client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': bad})
        assert r.status_code == 400, f'{bad!r} must be refused, not honoured'
    assert _unresolved(client, bank_id) == 1, 'nothing was vetoed by a bad id'
    # A numeric string stays acceptable — that is what a query string sends.
    assert client.post(f'/api/bank/{bank_id}/dups/distinct',
                       json={'group': '1'}).status_code == 200


def test_deleting_rejected_images_takes_their_verdicts_THROUGH_THE_ROUTE(
        client, three_copies, app):
    """The real deletion path, not the helper called by hand — and re-read from a
    fresh app context, because that is what proves the delete was committed.

    An orphan row is NOT inert: SQLite hands out max(rowid)+1, so a deleted
    image's id comes back on the next import and a surviving pair would veto a
    group nobody ruled on."""
    bank_id, ids = three_copies
    client.post(f'/api/bank/{bank_id}/dups/distinct', json={'group': 1})
    with app.app_context():
        from app.models import BankDupDistinct
        assert BankDupDistinct.query.filter_by(bank_id=bank_id).count() == 3
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [ids[0]], 'status': 'reject'})
    client.post(f'/api/bank/{bank_id}/delete-rejected', json={})
    with app.app_context():
        from app.models import BankDupDistinct, BankImage
        assert BankImage.query.filter_by(id=ids[0]).count() == 0, 'the row really went'
        left = BankDupDistinct.query.filter_by(bank_id=bank_id).all()
        assert len(left) == 1, 'the two pairs naming the deleted image went with it'
        assert ids[0] not in (left[0].image_a, left[0].image_b)
