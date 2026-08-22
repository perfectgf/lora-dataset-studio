"""A reference edit that is READY must survive a server restart.

This is the most expensive thing the app can lose. The edit is a PAID provider
call of one to three minutes; when it lands, its image is written into the
dataset folder and the registry entry that knows the image exists is held in
process memory. A restart dropped that entry, so the modal came back empty, Keep
answered 409, and the TTL sweep deleted the paid result thirty minutes later. The
user was billed for a picture no code path could ever show them again.

What is recovered, and what is not
----------------------------------
Only a candidate that had actually LANDED. An engine still running when the
process died produced nothing retrievable — its provider call is gone with the
thread — so it comes back FAILED, saying why, rather than running. A resurrected
"running" would be a spinner nothing is driving, and Discard would be the only
way out of a modal that promises a result that is never coming.

The trap this file exists to pin
--------------------------------
CANDIDATE_MARKER is in the name of two different kinds of file: the paid RESULT,
and the staged INPUT snapshots a local engine reads (`snapshot_…`, `modalref_…`).
Recovering by filename pattern would offer the user their own input photo back as
an edit result. So recovery is driven by a sidecar written when a candidate turns
ready — a file that only ever exists for a real result — and never by the name.

Freshness stays ONE rule, not two: recovery honours the same `_TTL_SECONDS` the
in-process purge and the disk sweep already use. A restart must not extend the
life of an abandoned edit beyond what staying up would have allowed.
"""
import io
import json
import os

import pytest
from PIL import Image

from app.services import dataset_activity
from app.services import reference_edit_jobs as rej


def _webp(color=(9, 40, 90)):
    b = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(b, 'WEBP')
    return b.getvalue()


@pytest.fixture(autouse=True)
def _clean_registry():
    rej.reset()
    dataset_activity.reset()
    yield
    rej.reset()
    dataset_activity.reset()


def _ready_batch(directory, dataset_id=1, engines=('chatgpt',), ready=('chatgpt',),
                 prompt='add glasses'):
    """Run a batch to the point where `ready` engines have landed a candidate."""
    started = rej.start_batch(dataset_id, directory, engines, prompt)
    names = {}
    for engine in ready:
        fn = f'local{rej.CANDIDATE_MARKER}{engine}cand.webp'
        with open(os.path.join(directory, fn), 'wb') as fh:
            fh.write(_webp())
        assert rej.set_ready(dataset_id, started['tokens'][engine], fn)
        names[engine] = fn
    return started, names


def test_a_paid_candidate_is_still_offered_after_a_restart(app, tmp_path):
    """The whole point: the user is not billed for a picture nobody can show."""
    directory = str(tmp_path)
    started, names = _ready_batch(directory)

    rej.reset()                                   # ← the restart

    recovered = rej.get(1, directory)
    assert recovered is not None, 'the paid candidate did not survive the restart'
    assert recovered['status'] == 'ready'
    assert recovered['candidate_filename'] == names['chatgpt']
    assert recovered['engine'] == 'chatgpt'
    assert recovered['prompt'] == 'add glasses'
    assert recovered['batch_id'] == started['batch_id']
    assert os.path.exists(os.path.join(directory, names['chatgpt']))


def test_a_recovered_candidate_can_actually_be_kept(app, tmp_path):
    """Being visible is not enough — Keep resolves through a CAS on batch_id, so
    a recovered entry that could not be claimed would be a button that 409s."""
    directory = str(tmp_path)
    started, names = _ready_batch(directory)

    rej.reset()

    assert rej.get(1, directory) is not None
    claim = rej.claim_ready(1, 'chatgpt', batch_id=started['batch_id'])
    assert claim is not None, 'the recovered candidate could not be claimed'
    assert claim['candidate_filename'] == names['chatgpt']


def test_a_staged_input_photo_is_never_offered_as_a_result(app, tmp_path):
    """The named trap. Local engines stage their INPUTS under the same marker; a
    filename-driven recovery would hand the user back their own source photo and
    call it an edit."""
    directory = str(tmp_path)
    for name in (f'local{rej.CANDIDATE_MARKER}snapshot_abc_0.webp',
                 f'local{rej.CANDIDATE_MARKER}modalref_abc_0.webp',
                 f'local{rej.CANDIDATE_MARKER}deadbeef.webp'):
        with open(os.path.join(directory, name), 'wb') as fh:
            fh.write(_webp())

    rej.reset()

    assert rej.get(1, directory) is None, 'a non-result file was offered as a result'


def test_an_engine_still_running_at_the_restart_comes_back_failed(app, tmp_path):
    """Its provider call died with the thread. Coming back "running" would be a
    spinner nothing drives; coming back silently absent would hide that the user
    was billed. It says what happened."""
    directory = str(tmp_path)
    _ready_batch(directory, engines=('chatgpt', 'krea'), ready=('chatgpt',))

    rej.reset()

    recovered = rej.get(1, directory)
    assert recovered is not None
    assert recovered['candidates']['chatgpt']['status'] == 'ready'
    krea = recovered['candidates']['krea']
    assert krea['status'] == 'failed'
    assert krea['error'], 'a lost engine must say why it has no result'
    assert 'restart' in krea['error'].lower()
    # One ready result among failures is still an unambiguous, keepable batch.
    assert recovered['status'] == 'ready'
    assert recovered['engine'] == 'chatgpt'


def test_the_batch_keeps_the_engine_order_it_was_started_with(app, tmp_path):
    """The panel lays the engines out in the order the user picked them."""
    directory = str(tmp_path)
    _ready_batch(directory, engines=('krea', 'chatgpt', 'klein'),
                 ready=('chatgpt', 'klein'))

    rej.reset()

    assert rej.get(1, directory)['engines'] == ['krea', 'chatgpt', 'klein']


def test_a_candidate_past_its_ttl_is_not_resurrected(app, tmp_path, monkeypatch):
    """One freshness rule, not two: a restart must not buy an abandoned edit more
    life than staying up would have."""
    directory = str(tmp_path)
    _, names = _ready_batch(directory)

    rej.reset()
    monkeypatch.setattr(rej, '_TTL_SECONDS', -1)   # everything counts as old

    assert rej.get(1, directory) is None
    rej.sweep(directory)
    assert not os.path.exists(os.path.join(directory, names['chatgpt']))


def test_discard_is_durable_across_a_restart(app, tmp_path):
    """A candidate the user threw away must not come back on the next boot."""
    directory = str(tmp_path)
    _, names = _ready_batch(directory)
    rej.clear(1, directory)

    rej.reset()

    assert rej.get(1, directory) is None
    assert not os.path.exists(os.path.join(directory, names['chatgpt']))


def test_keeping_a_candidate_leaves_nothing_to_recover(app, tmp_path):
    """After Keep the candidate IS the reference. Recovering it again would offer
    to keep what is already kept."""
    directory = str(tmp_path)
    started, _ = _ready_batch(directory)
    claim = rej.claim_ready(1, 'chatgpt', batch_id=started['batch_id'])
    assert rej.clear_claimed(1, claim['batch_token'], claim['claim_token'],
                             directory, reference_mutated=True) is not None

    rej.reset()

    assert rej.get(1, directory) is None


def test_a_sidecar_whose_image_vanished_is_not_a_candidate(app, tmp_path):
    """The image is the result; the sidecar only points at it. A pointer to
    nothing must not become an entry Keep would 409 on."""
    directory = str(tmp_path)
    _, names = _ready_batch(directory)
    os.remove(os.path.join(directory, names['chatgpt']))

    rej.reset()

    assert rej.get(1, directory) is None


def test_a_corrupt_sidecar_costs_the_candidate_not_the_page(app, tmp_path):
    """A hand-edited or truncated sidecar must not take the dataset payload down
    with it — every dataset poll goes through this path."""
    directory = str(tmp_path)
    _, names = _ready_batch(directory)
    sidecar = rej._candidate_meta_path(directory, names['chatgpt'])
    assert os.path.isfile(sidecar), 'no sidecar was written for a ready candidate'
    with open(sidecar, 'w', encoding='utf-8') as fh:
        fh.write('{ truncated')

    rej.reset()

    assert rej.get(1, directory) is None


def test_recovery_never_invents_an_activity_to_close(app, tmp_path):
    """The activity registry is empty after a restart. A recovered entry carrying
    a token from the dead process would light a badge nothing can turn off."""
    directory = str(tmp_path)
    _ready_batch(directory)

    rej.reset()

    assert rej.get(1, directory) is not None
    assert dataset_activity.get(1) is None
    rej.clear(1, directory)                 # must not raise on a phantom token
    assert dataset_activity.get(1) is None


def test_a_live_batch_is_never_replaced_by_a_recovered_one(app, tmp_path):
    """Recovery is for the empty-registry case only. A running edit must win over
    anything left on disk by the previous process."""
    directory = str(tmp_path)
    _ready_batch(directory)

    rej.reset()

    fresh = rej.start_batch(1, directory, ('krea',), 'a new edit')
    live = rej.get(1, directory)
    assert live['batch_id'] == fresh['batch_id']
    assert live['status'] == 'running'


def test_looking_for_a_lost_candidate_does_not_reread_the_folder_forever(
        app, tmp_path, monkeypatch):
    """Every dataset poll comes through get(). A dataset folder holds thousands
    of images, so answering "still nothing" by listing it on each poll would put
    a directory scan on the hot path. The question is asked once per process:
    after that, a candidate can only appear through the live registry, and one
    that is cleared takes its sidecar with it."""
    directory = str(tmp_path)
    listings = []
    real_listdir = os.listdir
    monkeypatch.setattr(rej.os, 'listdir',
                        lambda p: (listings.append(p), real_listdir(p))[1])

    for _ in range(5):
        assert rej.get(1, directory) is None
    assert len(listings) == 1, f'the folder was listed {len(listings)} times'

    # ...and a restart asks again, because then there really may be something.
    rej.reset()
    assert rej.get(1, directory) is None
    assert len(listings) == 2


def test_the_sidecar_records_no_absolute_path(app, tmp_path):
    """It travels with the dataset folder. Only the bare filename is written, so
    a folder moved to another drive still resolves."""
    directory = str(tmp_path)
    _, names = _ready_batch(directory)
    payload = json.loads(
        open(rej._candidate_meta_path(directory, names['chatgpt']),
             encoding='utf-8').read())
    assert payload['candidate_filename'] == names['chatgpt']
    assert not any(isinstance(v, str) and (':' in v or '/' in v or '\\' in v)
                   for v in payload.values()), payload
