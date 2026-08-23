"""Is the repository still there — and, above all, when may we SAY it is not.

``artifact_status`` is written once, at delivery, and never revisited. Both
panels rendered that record in the present tense, so a repository deleted
overnight still read "Full model available … verified" above a link answering
404. ``hub_presence`` answers the other question with one metadata request.

The contract pinned here is almost entirely about the ANSWER WE MAY NOT GIVE:

  a) 200 → present. A live answer is the only thing entitled to the present tense;
  b) 404 → gone, but ONLY after every configured token has 404'd AND the last
     one is still recognised by the Hub. `huggingface_hub` cannot make this
     distinction at all (it raises RepositoryNotFoundError for 401 and 404
     alike), which is why this module speaks raw HTTP;
  c) no token / offline / 5xx / 401 / 403 / dead token → unknown. Never gone.
     Telling someone eight hours of paid GPU are lost because their Wi-Fi
     dropped is a worse bug than the one being fixed;
  d) the answer is cached with a short TTL — the Checkpoints panel re-polls
     itself and the Hub must not be re-asked at that rate;
  e) a batch stops asking the network once one probe finds nothing answering.

No request ever leaves this process: ``_http_get`` is the single seam and every
test replaces it.
"""
import pytest

from app.services import hub_presence as hp

REPO = 'ns/dense-run-90'
TOKEN = 'hf_test_token'


@pytest.fixture(autouse=True)
def _fresh_cache():
    hp.clear_cache()
    yield
    hp.clear_cache()


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv('HF_CLOUD_TOKEN', TOKEN)
    return TOKEN


class FakeHub:
    """Answers by URL prefix and records every call, so a test can assert both
    the verdict and how many requests it took to reach it."""

    def __init__(self, model=200, whoami=200):
        self.model = model
        self.whoami = whoami
        self.calls = []

    def __call__(self, url, token, timeout=None):
        self.calls.append((url, token))
        status = self.whoami if 'whoami' in url else self.model
        if callable(status):
            status = status(token)
        return status, b''


def install(monkeypatch, hub):
    monkeypatch.setattr(hp, '_http_get', hub)
    return hub


# --- the three answers --------------------------------------------------------

def test_a_repository_that_answers_is_present(monkeypatch, token):
    hub = install(monkeypatch, FakeHub(model=200))
    out = hp.check(REPO)
    assert out['state'] == hp.PRESENT
    assert out['repo_id'] == REPO
    assert out['checked_at']
    # The cheap request and nothing else: no file listing, no whoami.
    assert len(hub.calls) == 1
    assert hub.calls[0][0].endswith(f'/api/models/{REPO}')
    assert hub.calls[0][1] == TOKEN


def test_a_404_from_a_live_token_is_the_only_way_to_say_gone(monkeypatch, token):
    install(monkeypatch, FakeHub(model=404, whoami=200))
    out = hp.check(REPO)
    assert out['state'] == hp.GONE
    # The sentence names both readings of a 404 rather than picking one: the Hub
    # hides a private repo from an unauthorized token behind the same status it
    # uses for a deleted one, and pretending otherwise would be the same class
    # of overclaim this module exists to end.
    assert 'deleted or renamed' in out['detail']
    assert 'no longer allowed to see it' in out['detail']


def test_a_404_from_a_token_the_hub_no_longer_knows_is_NOT_gone(monkeypatch, token):
    """The inverse error, and the expensive one: it would report a loss."""
    install(monkeypatch, FakeHub(model=404, whoami=401))
    out = hp.check(REPO)
    assert out['state'] == hp.UNKNOWN
    assert 'could not be validated' in out['detail']
    assert 'deleted' not in out['detail']


@pytest.mark.parametrize('status', [401, 403, 500, 502, 503, None])
def test_nothing_but_a_qualified_404_may_read_as_an_absence(monkeypatch, token, status):
    install(monkeypatch, FakeHub(model=status))
    out = hp.check(REPO)
    assert out['state'] == hp.UNKNOWN
    assert 'deleted' not in out['detail']
    assert 'gone' not in out['detail'].lower()


def test_no_token_is_a_question_we_could_not_ask(monkeypatch):
    hub = install(monkeypatch, FakeHub(model=404))
    out = hp.check(REPO)
    assert out['state'] == hp.UNKNOWN
    assert 'HF_CLOUD_TOKEN' in out['detail']
    assert hub.calls == []          # and it never touched the network to find out


def test_a_crash_in_the_probe_is_not_an_absence(monkeypatch, token):
    def boom(*_a, **_k):
        raise RuntimeError('unexpected')
    monkeypatch.setattr(hp, '_http_get', boom)
    assert hp.check(REPO)['state'] == hp.UNKNOWN


def test_an_empty_repo_id_asserts_nothing(monkeypatch, token):
    hub = install(monkeypatch, FakeHub(model=200))
    assert hp.check('')['state'] == hp.UNKNOWN
    assert hub.calls == []


@pytest.mark.parametrize('bad', ['../../api/whoami-v2', 'no-namespace',
                                 'ns/name?expand=x', 'ns/na me'])
def test_a_name_that_is_not_a_repo_id_never_reaches_a_request(monkeypatch, token, bad):
    hub = install(monkeypatch, FakeHub(model=200))
    out = hp.check(bad)
    assert out['state'] == hp.UNKNOWN
    assert hub.calls == []


# --- the second token exists only to AVOID a false gone -----------------------

def test_a_second_token_is_tried_before_concluding_gone(monkeypatch):
    monkeypatch.setenv('HF_CLOUD_TOKEN', 'narrow')
    monkeypatch.setenv('HF_TOKEN', 'wide')
    hub = install(monkeypatch, FakeHub(
        model=lambda tok: 200 if tok == 'wide' else 404))
    out = hp.check(REPO)
    assert out['state'] == hp.PRESENT
    # The delivery token first (it is the one that pushed), the general one only
    # as the second opinion that keeps a narrowed scope from reading as a loss.
    assert [tok for _url, tok in hub.calls] == ['narrow', 'wide']


def test_both_tokens_404_and_a_live_token_still_means_gone(monkeypatch):
    monkeypatch.setenv('HF_CLOUD_TOKEN', 'a')
    monkeypatch.setenv('HF_TOKEN', 'b')
    install(monkeypatch, FakeHub(model=404, whoami=200))
    assert hp.check(REPO)['state'] == hp.GONE


# --- caching ------------------------------------------------------------------

def test_a_verdict_is_held_so_a_polling_panel_cannot_hammer_the_hub(monkeypatch, token):
    hub = install(monkeypatch, FakeHub(model=200))
    for _ in range(5):
        assert hp.check(REPO)['state'] == hp.PRESENT
    assert len(hub.calls) == 1
    assert hp.check(REPO)['cached'] is True
    # `force` is the one way past it — the explicit re-check, never the poll.
    hp.check(REPO, force=True)
    assert len(hub.calls) == 2


def test_could_not_check_expires_faster_than_a_verdict():
    # An outage should heal within a poll or two; a deletion will not un-happen.
    assert hp._TTL[hp.UNKNOWN] < hp._TTL[hp.GONE]
    assert hp._TTL[hp.GONE] == hp._TTL[hp.PRESENT]


def test_the_cache_cannot_grow_without_bound_in_a_process_that_never_restarts(
        monkeypatch, token):
    install(monkeypatch, FakeHub(model=200))
    for i in range(hp._MAX_CACHED + 5):
        hp.check(f'ns/repo-{i}')
        # Age every entry as it lands, so the prune has something to collect.
        stamped, payload = hp._cache[f'ns/repo-{i}']
        hp._cache[f'ns/repo-{i}'] = (stamped - hp._TTL[hp.PRESENT] - 1, payload)
    assert len(hp._cache) <= hp._MAX_CACHED


def test_an_expired_entry_is_asked_again(monkeypatch, token):
    hub = install(monkeypatch, FakeHub(model=200))
    hp.check(REPO)
    stamped, payload = hp._cache[REPO]
    hp._cache[REPO] = (stamped - hp._TTL[hp.PRESENT] - 1, payload)
    hp.check(REPO)
    assert len(hub.calls) == 2


# --- a panel's worth of runs --------------------------------------------------

def test_a_batch_de_duplicates_and_is_bounded(monkeypatch, token):
    hub = install(monkeypatch, FakeHub(model=200))
    out = hp.check_many([REPO, REPO, '', None, 'ns/other'])
    assert set(out) == {REPO, 'ns/other'}
    assert len(hub.calls) == 2
    # One panel must never turn into an unbounded burst.
    hp.clear_cache()
    hub.calls.clear()
    hp.check_many([f'ns/repo-{i}' for i in range(hp._MAX_PER_CALL + 10)])
    assert len(hub.calls) == hp._MAX_PER_CALL


def test_an_unreachable_hub_stops_the_batch_instead_of_timing_out_twelve_times(
        monkeypatch, token):
    hub = install(monkeypatch, FakeHub(model=None))
    out = hp.check_many([f'ns/repo-{i}' for i in range(6)])
    assert len(hub.calls) == 1
    assert {r['state'] for r in out.values()} == {hp.UNKNOWN}
    assert all('not checked' in r['detail'] for r in out.values())


# --- the endpoint the panels call ---------------------------------------------

def _dense_run(dataset_id, *, repo='ns/dense-90', status='available'):
    """A delivered dense run, in the shape the panels list."""
    import json

    from app.extensions import db
    from app.models import CloudTrainingRun
    params = {'training_mode': 'full_transformer', 'train_type': 'krea',
              'variant': 'Raw', 'steps': 3000, 'artifact_status': status}
    if repo:
        params.update({'hf_repo_id': repo,
                       'hf_url': f'https://huggingface.co/{repo}'})
    run = CloudTrainingRun(dataset_id=dataset_id, status='done', job_name='j',
                           staging_dir='', train_params=json.dumps(params))
    db.session.add(run)
    db.session.commit()
    return run


def test_the_endpoint_answers_per_run_and_only_for_dense_runs(
        app, client, monkeypatch, token):
    import json

    from app.extensions import db
    from app.models import CloudTrainingRun
    install(monkeypatch, FakeHub(model=404, whoami=200))
    with app.app_context():
        dense = _dense_run(1).id
        no_repo = _dense_run(1, repo=None).id
        lora = CloudTrainingRun(
            dataset_id=1, status='done', job_name='j', staging_dir='',
            train_params=json.dumps({'train_type': 'krea', 'hf_repo_id': 'ns/lora'}))
        db.session.add(lora)
        db.session.commit()
        lora_id = lora.id
    body = client.post('/api/dataset/train/cloud/hub-presence',
                       json={'run_ids': [dense, no_repo, lora_id, 999999]}).get_json()
    assert body['ok'] is True
    # A run with no recorded repository, a LoRA run and an unknown id all have
    # nothing to answer — silence, not a fabricated absence.
    assert set(body['results']) == {str(dense)}
    assert body['results'][str(dense)]['state'] == hp.GONE
    assert body['results'][str(dense)]['run_id'] == dense


def test_the_endpoint_never_rewrites_the_delivery_record(app, client, monkeypatch,
                                                         token):
    """A live 404 must not overwrite `artifact_status`.

    That field is the DELIVERY's minute — the proof a verified model once
    existed, and what the recovery paths reason on. Letting a read-only panel
    check demote it would destroy evidence on a token that was merely narrowed,
    and there is a deliberate, transactional operation for restating it
    (`recheck-delivery`).
    """
    from app.services import cloud_training as ct
    install(monkeypatch, FakeHub(model=404, whoami=200))
    with app.app_context():
        run_id = _dense_run(2).id
    assert client.post('/api/dataset/train/cloud/hub-presence',
                       json={'run_ids': [run_id]}).status_code == 200
    with app.app_context():
        from app.models import CloudTrainingRun
        run = CloudTrainingRun.query.get(run_id)
        assert ct._run_param(run, 'artifact_status') == 'available'


def test_the_endpoint_refuses_an_empty_request(app, client):
    assert client.post('/api/dataset/train/cloud/hub-presence',
                       json={}).status_code == 400
    assert client.post('/api/dataset/train/cloud/hub-presence',
                       json={'run_ids': []}).status_code == 400


def test_the_panel_payload_dates_the_record_it_shows(app, client, monkeypatch):
    """The Checkpoints listing carries WHEN the delivery was verified.

    Without it the card can only say "delivered and verified", which reads as a
    present-tense claim; with it, it says "on 2026-07-11 — not re-checked since",
    which is the same fact told truthfully."""
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(3)
        from app.services import cloud_training as ct
        ct._persist_run_params(run, delivery_last_checked_at='2026-07-11T09:12:33')
        entry = da.describe_run(run)
    assert entry['hub']['checked_at'] == '2026-07-11T09:12:33'
    assert entry['hub']['status'] == 'available'
