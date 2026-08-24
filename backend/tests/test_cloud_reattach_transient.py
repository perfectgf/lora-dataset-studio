"""Reattaching to a pod that is already training (run #146, 2026-08-03).

The app restarted while a dense run was at step 825/3000, with three saved
checkpoints. The boot-wait adopted it at 16:01:58, the vast API answered one
poll without the instance in its listing, and ten seconds later the run was
condemned: the boot budget is anchored to the run's durable created_at, so a
run that has been alive for hours enters its "boot" wait with the whole budget
already spent. Everything after that followed: the dense recovery stopped the
remote job — and the pod, which was perfectly reachable the whole time, obeyed
and killed the training.

Two independent guarantees are asserted here:

* the vast API is not the authority on whether the pod exists. When the row
  already carries a base_url, the pod itself is asked;
* a reattach is not a boot. Its window is the same bounded tolerance the poll
  loop grants a pod that stops answering mid-run — minutes of CONSECUTIVE
  silence, measured from this attempt — never a single unlucky poll.

Plus the anti-masking pair: a resume whose job never started keeps the durable
anchor of the 2026-07-14 incident, and a pod that is really gone is still given
up (without a stop it cannot receive, and without banning a host that was
training fine minutes ago).
"""
import json
from app.extensions import db
from app.utils.timestamps import naive_utcnow
from datetime import timedelta

from test_cloud_training_monitor import ct, FakeRemote, _launch    # noqa: F401


_POD = {'instance_id': '777', 'actual_status': 'running',
        'public_ipaddr': '1.2.3.4', 'label': 'lds-x',
        'ports': {'18675/tcp': [{'HostPort': '40123'}]},
        'jupyter_token': 'jtok-vast'}


def _resumed_training_run(ct, app, run_id, *, age_minutes=180):
    """The row exactly as boot_recover finds it: pod owned, job started, the
    run alive for hours."""
    with app.app_context():
        run = db.session.get(ct.CloudTrainingRun, run_id)
        ct._set(run, vast_instance_id='777', remote_job_id='j-live',
                status='training', base_url='http://1.2.3.4:40123',
                auth_token='tok',
                created_at=naive_utcnow() - timedelta(minutes=age_minutes))


def test_a_vast_listing_gap_never_condemns_a_pod_that_answers(
        ct, app, client, monkeypatch):
    """THE incident. Two polls where the vast API does not list the instance,
    while the pod answers its own URL normally. Before the fix the second poll
    condemned the run (age > boot budget); now the probe settles it and the run
    trains to completion."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    _resumed_training_run(ct, app, run_id)
    listings = {'n': 0}

    def with_a_gap(iid):
        listings['n'] += 1
        return None if listings['n'] <= 2 else dict(_POD)

    monkeypatch.setattr(ct.vast_client, 'get_instance', with_a_gap)
    with app.app_context():
        ct._monitor(app, run_id)
        run = db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'done'
        # It never even needed the listing to come back: the pod's own answer
        # is the evidence, so the boot wait broke out on the first poll.
        assert listings['n'] == 1
        assert remote.uploaded == {}            # still a reattach, not a relaunch
        assert remote.job_config is None


def test_a_pod_silent_for_two_polls_is_not_a_dead_pod(
        ct, app, client, monkeypatch):
    """The second guarantee, isolated: even with NOTHING answering — no listing
    and no reply from the pod — a reattach owes minutes of consecutive silence
    before it may condemn, not the two polls (10 s) that killed #146."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    probes = {'n': 0}

    def slow_ready():
        probes['n'] += 1
        return probes['n'] > 2                  # silent for the first two polls

    remote.is_ready = slow_ready
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    _resumed_training_run(ct, app, run_id)
    listings = {'n': 0}

    def with_a_gap(iid):
        listings['n'] += 1
        return None if listings['n'] <= 2 else dict(_POD)

    monkeypatch.setattr(ct.vast_client, 'get_instance', with_a_gap)
    with app.app_context():
        ct._monitor(app, run_id)
        run = db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'done'
        assert probes['n'] == 3                 # it kept asking, then got in


# --- anti-masking: the tolerance is bounded, and it costs the pod nothing -----

def _dense_run(ct, dataset_id, staging):
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='training', run_name='dense-reattach',
        job_name='Krea_dense_reattach', vast_label='lds-dense',
        vast_instance_id='pod-dense', remote_job_id='remote-dense',
        base_url='https://dense-pod.invalid', auth_token='tok',
        staging_dir=str(staging),
        train_params=json.dumps({
            'training_mode': 'full_transformer', 'train_type': 'krea',
            'variant': 'base', 'steps': 3000, 'artifact_status': 'pending',
            'hf_repo_id': 'tester/Krea-dense-reattach',
        }))
    ct.db.session.add(run)
    ct.db.session.commit()
    return run.id


def test_a_pod_that_never_answers_again_is_given_up_without_a_stop_or_a_ban(
        ct, app, client, monkeypatch, tmp_path):
    """The guard that keeps the fix from being a cover-up — and the third lesson
    of #146. The pod is really gone: nothing lists it, nothing answers it. The
    dense run must still be given up, bounded, but

    * no stop is sent — a job we could not reach for minutes cannot receive one,
      and sending it on a WRONG verdict is precisely what killed the training;
    * the host is not banned — it was training our job minutes ago;
    * the pod is kept, so a dense result stays recoverable.
    """
    stopped, destroyed = [], []

    class Silent:
        def is_ready(self):
            return False

        def stop_job(self, job_id):
            stopped.append(job_id)

    polls = {'n': 0}

    def never_listed(iid):
        polls['n'] += 1
        return None

    monkeypatch.setattr(ct, '_prepare_staging', lambda run: None)
    monkeypatch.setattr(ct, '_make_remote', lambda run: Silent())
    monkeypatch.setattr(ct.vast_client, 'get_instance', never_listed)
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(iid) or True)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + 30.0)
                        or clock['t'])
    staging = tmp_path / 'dense-reattach'
    staging.mkdir()
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Dense', 'trigger_word': 'p'}).get_json()['id']
    with app.app_context():
        run_id = _dense_run(ct, ds_id, staging)
        ct._monitor(app, run_id)
        ct.db.session.expire_all()
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error_pod_kept'
        assert 'could not be reached' in (run.error or '')
        assert stopped == []                    # nothing to stop, nothing sent
        assert destroyed == []                  # the paid pod is kept
        assert ct._load_bad_hosts() == {}       # and the host is not blamed
        # Bounded: minutes of consecutive silence, not two polls and not forever.
        assert 3 <= polls['n'] <= 60


def test_a_resume_whose_job_never_started_keeps_the_durable_boot_anchor(
        ct, app, client, monkeypatch):
    """The boundary. The 2026-07-14 incident was a pod that never came up at
    all, given a brand-new 15-min window at every restart until it had eaten 37
    minutes. That run has no started job, so it is still BOOTING and still
    measured from created_at — this fix must not hand it a fresh window."""
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    monkeypatch.setattr(ct, '_now', lambda: 0.0)
    with app.app_context():
        run = db.session.get(ct.CloudTrainingRun, run_id)
        ct._set(run, vast_instance_id='777', staging_dir='/tmp/x',
                base_url='http://1.2.3.4:40123',        # never any remote_job_id
                created_at=naive_utcnow() - timedelta(minutes=30))
        ct._monitor(app, run_id)
        run = db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error'
        assert 'become ready' in (run.error or '').lower()
        assert destroyed == ['777']
