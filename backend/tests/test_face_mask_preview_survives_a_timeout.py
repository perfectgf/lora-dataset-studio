"""A face-mask preview must never restart from zero — including after the ways
it ends that nobody asked for.

The Stop/Resume bargain was already built and already worked. What it did not
cover is everything else that ends a pass: the watchdog budget elapsing, the
child dying, the server going down mid-run. All three shared one root cause —
the child holds every box it finds IN MEMORY until its final JSON line, so an
ending that never reaches that line destroys the whole pass, and the next start
begins at image 1.

That is not hypothetical. A 138-image set ran healthily to about image 100 and
was then killed by a FIXED 900 s watchdog covering the whole batch: 900 s is a
per-image number in disguise, it fits roughly 45 images on this CPU-only
detector (9-20 s each, plus the load), and it silently executes anything larger.
The route then read the timeout as a failure and called `fail`, which threw away
what the run had produced. Both halves are fixed here:

  1. the child publishes each image's result AS IT GOES (`[facemask] item {...}`),
     the parent banks it debounced, so a death costs at most the flush window;
  2. the budget GROWS with the image count, the watchdog ASKS before it kills,
     and a pass cut short banks what it has under a state of its own —
     interrupted, which is neither a failure nor a user's Stop.

No InsightFace anywhere: stand-in scripts speak the same stdin/stdout/stderr
protocol as infer/face_mask_infer.py, which is the contract under test.
"""
import json
import os
import sys
import time

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import face_mask, face_mask_preview as fmp
from app.services import infer_stream
from app.config import LOCAL_USER, save_config

PREVIEW = '/api/dataset/{}/train/face-mask-preview'


def _dataset(app, tmp_path, n=5):
    """A concept dataset with `n` kept images. Same context discipline as the
    sibling files: the `app` fixture does not push one."""
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FMT', 'fmt_act', kind='concept',
                                concept_desc='balancing a spoon')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(n):
            fn = f'k{i}.png'
            Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
            db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        db.session.commit()
        dataset_id = ds.id
    fmp.reset()
    return dataset_id


def _use(monkeypatch, tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding='utf-8')
    monkeypatch.setattr(face_mask, '_SCRIPT', str(p))
    monkeypatch.setattr(face_mask, '_face_python', lambda: sys.executable)
    monkeypatch.setattr(face_mask, 'is_available', lambda: True)


def _tiny_budget(monkeypatch, per_image=0, load=1):
    """Shrink the watchdog budget so a timeout is a test, not a coffee break.
    The arithmetic under test is `timeout_for`, which is asserted on its own
    values elsewhere in this file."""
    monkeypatch.setattr(face_mask, 'PER_IMAGE_S', per_image)
    monkeypatch.setattr(face_mask, 'LOAD_BUDGET_S', load)


# A child that publishes per-image results, then HANGS — until the sentinel
# appears, at which point it winds up like the real script does. That is the
# graceful timeout: the budget elapses, the watchdog asks, the child answers.
_STALLS = (
    "import json, os, sys, time\n"
    "req = json.loads(sys.stdin.read())\n"
    "imgs, cancel = req['images'], req.get('cancel_file')\n"
    "n = int(os.environ['FMT_N'])\n"
    "print('[facemask] phase=detecting', file=sys.stderr, flush=True)\n"
    "out = {}\n"
    "for i, p in enumerate(imgs[:n], 1):\n"
    "    out[p] = {'state': 'masked', 'boxes': [[.4, .3, .6, .5]]}\n"
    "    print('[facemask] item ' + json.dumps(dict(out[p], path=p)),\n"
    "          file=sys.stderr, flush=True)\n"
    "    print('[facemask] %d/%d masked faces=1' % (i, len(imgs)), file=sys.stderr, flush=True)\n"
    "deadline = time.time() + 30\n"
    "while time.time() < deadline:\n"
    "    if cancel and os.path.exists(cancel):\n"
    "        break\n"
    "    time.sleep(0.02)\n"
    "print(json.dumps({'ok': True, 'written': 0, 'cancelled': True, 'results': out}))\n"
)

# A child that publishes results and then DIES without its final JSON line — a
# crash, a kill, the machine going away. Everything it printed is all there is.
_CRASHES = (
    "import json, os, sys\n"
    "imgs = json.loads(sys.stdin.read())['images']\n"
    "n = int(os.environ['FMT_N'])\n"
    "print('[facemask] phase=detecting', file=sys.stderr, flush=True)\n"
    "for i, p in enumerate(imgs[:n], 1):\n"
    "    rec = {'path': p, 'state': 'masked', 'boxes': [[.4, .3, .6, .5]]}\n"
    "    print('[facemask] item ' + json.dumps(rec), file=sys.stderr, flush=True)\n"
    "    print('[facemask] %d/%d masked faces=1' % (i, len(imgs)), file=sys.stderr, flush=True)\n"
    "sys.stderr.flush()\n"
    "os._exit(9)\n"
)

# A well-behaved child that publishes items AND its final line.
_GOOD = (
    "import json, sys\n"
    "imgs = json.loads(sys.stdin.read())['images']\n"
    "print('[facemask] phase=detecting', file=sys.stderr, flush=True)\n"
    "out = {}\n"
    "for i, p in enumerate(imgs, 1):\n"
    "    out[p] = {'state': 'masked', 'boxes': [[.4, .3, .6, .5]]}\n"
    "    print('[facemask] item ' + json.dumps(dict(out[p], path=p)),\n"
    "          file=sys.stderr, flush=True)\n"
    "    print('[facemask] %d/%d masked faces=1' % (i, len(imgs)), file=sys.stderr, flush=True)\n"
    "print(json.dumps({'ok': True, 'written': 0, 'results': out}))\n"
)


# --- 1. the grammar that makes banking-as-you-go possible ---------------------

def test_a_finished_image_arrives_on_the_progress_channel_as_a_result():
    """RED before this fix: the child's stderr carried only `i/N state faces=K`,
    with no coordinates, so the parent could count the pass but could not keep
    any of it. Nothing could be banked until the final JSON line existed."""
    line = ('[facemask] item {"path": "/d/a.png", "state": "masked", '
            '"boxes": [[0.1, 0.2, 0.3, 0.4]], "coverage": 0.04}')
    assert face_mask.parse_progress_line(line) == {
        'item': {'path': '/d/a.png',
                 'result': {'state': 'masked', 'boxes': [[0.1, 0.2, 0.3, 0.4]],
                            'coverage': 0.04}}}


def test_the_item_line_does_not_disturb_the_counter_grammar():
    """The `i/N` lines drive the progress bar and are under test elsewhere. The
    new shape is a THIRD line, not a replacement — and a malformed one is
    ignored rather than raised, because a progress channel must never be able to
    take the pass down."""
    assert face_mask.parse_progress_line('[facemask] 2/7 masked faces=1') == {
        'phase': 'detecting', 'done': 2, 'total': 7}
    assert face_mask.parse_progress_line('[facemask] item {truncated') is None
    assert face_mask.parse_progress_line('[facemask] item {"state": "masked"}') is None


def test_the_real_child_publishes_every_outcome_it_records(capsys):
    """Asserted on the shipped script, not on a stub: a stub that speaks the
    protocol proves nothing about whether infer/face_mask_infer.py does."""
    sys.path.insert(0, str((__import__('app.config', fromlist=['x']).BACKEND_DIR / 'infer')))
    import face_mask_infer

    face_mask_infer._item('/d/a.png', {'state': 'no_face', 'boxes': []})
    line = capsys.readouterr().err.strip()
    assert face_mask.parse_progress_line(line) == {
        'item': {'path': '/d/a.png', 'result': {'state': 'no_face', 'boxes': []}}}

    src = (__import__('app.config', fromlist=['x']).BACKEND_DIR
           / 'infer' / 'face_mask_infer.py').read_text(encoding='utf-8')
    # One publication per outcome the loop can record — unreadable, no_face,
    # masked/too_large, error. A branch that only counts is a branch whose work
    # dies with the process.
    assert src.count('_item(p, results[p])') == 4


# --- 2. the budget is derived from the work ----------------------------------

def test_the_budget_grows_with_the_number_of_images():
    """The old 900 s covered the whole batch, which fits ~45 images on this
    CPU-only detector. A budget that does not scale is a run that dies at a size
    nobody warned about."""
    assert face_mask.timeout_for(0) == face_mask.LOAD_BUDGET_S
    small, large = face_mask.timeout_for(10), face_mask.timeout_for(138)
    assert large - small == face_mask.PER_IMAGE_S * 128
    # The incident's set: comfortably past the 25-45 min the pass really costs.
    assert large >= 45 * 60


def test_detect_faces_asks_for_that_budget_rather_than_a_fixed_one(
        app, tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=3)
    paths = [os.path.join(svc._dataset_dir(ds_id), f'k{i}.png') for i in range(3)]

    seen = []
    real = face_mask.run_infer_script

    def _spy(python, script, payload, timeout, *a, **kw):
        seen.append(timeout)
        return real(python, script, payload, timeout, *a, **kw)
    monkeypatch.setattr(face_mask, 'run_infer_script', _spy)

    face_mask.detect_faces(paths)
    assert seen == [face_mask.timeout_for(3)]

    # An explicit budget still wins — the suite and any caller that knows better
    # keep their say.
    face_mask.detect_faces(paths, timeout=42)
    assert seen[-1] == 42


def test_the_generation_path_scales_too(app, tmp_path, monkeypatch):
    """Same defect, nastier surface: there the failure is SILENT by design
    (the export just trains unmasked), so a budget too small for the set loses
    the masks with nothing on screen to say why."""
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=4)
    paths = [os.path.join(svc._dataset_dir(ds_id), f'k{i}.png') for i in range(4)]

    seen = []
    real = face_mask.run_infer_script
    monkeypatch.setattr(face_mask, 'run_infer_script',
                        lambda p, s, pl, t, *a, **kw: (seen.append(t),
                                                       real(p, s, pl, t, *a, **kw))[1])

    face_mask.generate_face_masks(paths, str(tmp_path / 'masks'))
    assert seen == [face_mask.timeout_for(4)]


# --- 3. the watchdog asks before it kills ------------------------------------

def test_a_child_that_can_wind_up_still_hands_back_its_results(tmp_path):
    """The heart of it. The watchdog used to `proc.kill()` on the spot, which is
    precisely what destroys a pass whose results exist nowhere else yet."""
    script = tmp_path / 'winds_up.py'
    script.write_text(
        "import json, os, sys, time\n"
        "req = json.loads(sys.stdin.read())\n"
        "deadline = time.time() + 30\n"
        "while time.time() < deadline and not os.path.exists(req['cancel_file']):\n"
        "    time.sleep(0.02)\n"
        "print(json.dumps({'ok': True, 'results': {'a': 1}}))\n", encoding='utf-8')
    sentinel = tmp_path / 'stop'

    stdout, _, rc, timed_out = infer_stream.run_infer_script(
        sys.executable, str(script), json.dumps({'cancel_file': str(sentinel)}),
        timeout=1, on_stop=lambda: sentinel.write_text('stop', encoding='utf-8'),
        stop_grace=10)

    assert timed_out is True, 'the budget did elapse and must be reported'
    assert rc == 0
    assert json.loads(stdout.strip())['results'] == {'a': 1}


def test_a_child_that_cannot_answer_is_still_killed(tmp_path):
    """The ask is not a promise to wait forever. A child deep in a model load or
    a several-hundred-megabyte download reaches no polling point — and there a
    kill costs nothing, because nothing has been computed yet."""
    script = tmp_path / 'deaf.py'
    script.write_text("import sys, time\nsys.stdin.read()\ntime.sleep(120)\n",
                      encoding='utf-8')
    asked = []
    started = time.monotonic()

    _, _, rc, timed_out = infer_stream.run_infer_script(
        sys.executable, str(script), '{}', timeout=1,
        on_stop=lambda: asked.append(1), stop_grace=1)

    assert timed_out is True and asked == [1]
    assert rc != 0, 'the deaf child must not outlive its grace'
    assert time.monotonic() - started < 30


# --- 4. a timed-out pass keeps its work and says what happened ----------------

def test_a_pass_killed_by_its_budget_leaves_a_resume_instead_of_nothing(
        app, client, tmp_path, monkeypatch):
    """THE regression. The route read a timeout through `not data['ok']` and
    called `fail`, which discards everything — so the run that died at image 100
    of 138 offered a full restart."""
    _use(monkeypatch, tmp_path, 'stalls.py', _STALLS)
    monkeypatch.setenv('FMT_N', '3')
    _tiny_budget(monkeypatch)
    ds_id = _dataset(app, tmp_path, n=5)

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['resume'] == {'done': 3, 'total': 5}
    # Its own state: not a failure (the pass was working), not a Stop (nobody
    # clicked). The panel has to be able to tell the user which of the three
    # happened, and only one of them is their doing.
    assert body['job']['interrupted'] is True
    assert body['job']['stopped'] is False
    assert body['job']['error'] is None
    # The budget is IN the message, because the answer to a real timeout is a
    # bigger budget and the user is the one who knows their set size.
    assert 'budget' in body['job']['note']
    # No preview published: `coverage` is a safety figure over the WHOLE kept
    # set, so "masked on 3 of 3" for a 5-image set would be a wrong number.
    assert body['result'] is None


def test_the_next_start_carries_on_from_where_the_budget_ran_out(
        app, client, tmp_path, monkeypatch):
    """A resume credit nobody can spend is a label, not a fix."""
    _use(monkeypatch, tmp_path, 'stalls.py', _STALLS)
    monkeypatch.setenv('FMT_N', '3')
    _tiny_budget(monkeypatch)
    ds_id = _dataset(app, tmp_path, n=5)
    client.post(PREVIEW.format(ds_id), json={'limit': 6})
    assert client.get(PREVIEW.format(ds_id)).get_json()['resume']['done'] == 3

    handed = []
    real = face_mask.detect_faces
    monkeypatch.setattr(face_mask, 'detect_faces',
                        lambda paths, **kw: (handed.append(list(paths)),
                                             real(paths, **kw))[1])
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)      # this time it completes

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    assert len(handed[0]) == 2, 'the second pass re-detected images already banked'
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['result']['coverage'] == {'total': 5, 'masked': 5, 'no_face': 0,
                                          'too_large': 0, 'failed': 0}
    assert body['resume'] is None       # the bank is spent, not left to shadow


# --- 5. a crash, and a restart on top of it ----------------------------------

def test_a_child_that_dies_mid_pass_still_leaves_what_it_published(
        app, client, tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path, 'crashes.py', _CRASHES)
    monkeypatch.setenv('FMT_N', '4')
    ds_id = _dataset(app, tmp_path, n=6)

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    body = client.get(PREVIEW.format(ds_id)).get_json()
    # A crash IS a failure and still says so — but it no longer costs the four
    # images the pass had already paid for.
    assert body['job']['error']
    assert body['resume'] == {'done': 4, 'total': 6}


def test_an_interrupted_pass_survives_a_server_restart(app, client, tmp_path,
                                                       monkeypatch):
    """`fmp.reset()` is the restart — it drops every byte of module state, which
    is what a new process starts with. Without the sidecar write, the resume the
    interruption banked would exist only in the dead process's memory."""
    _use(monkeypatch, tmp_path, 'crashes.py', _CRASHES)
    monkeypatch.setenv('FMT_N', '4')
    ds_id = _dataset(app, tmp_path, n=6)
    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    fmp.reset()

    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['resume'] == {'done': 4, 'total': 6}
    assert body['job'] is None, 'a dead job must not come back as a ghost'


def test_the_pass_writes_its_detections_down_while_it_is_still_running(
        app, client, tmp_path, monkeypatch):
    """The one thing a crash test cannot prove on its own: the bank is written
    DURING the pass, not assembled at the end. Asserted on a run that finishes
    cleanly — that path banks nothing on the way out (it publishes a result and
    clears), so every call seen here came from the in-flight flush.

    Debounced on purpose: the sidecar is rewritten whole, so one write per image
    would be one full rewrite per image. Ten bounds the loss to ten images."""
    from app.routes import training as tr
    _use(monkeypatch, tmp_path, 'good.py', _GOOD)
    ds_id = _dataset(app, tmp_path, n=25)
    # Pin the COUNT bound by taking the time bound out of the picture: this stub
    # runs in milliseconds, so a wall-clock rule would only add flakiness under
    # load without asserting anything the count rule does not.
    monkeypatch.setattr(tr, '_FACE_BANK_EVERY_S', 10_000)

    sizes = []
    real = fmp.remember_partial
    monkeypatch.setattr(fmp, 'remember_partial',
                        lambda ds, results, fp: (sizes.append(len(results)),
                                                 real(ds, results, fp))[1])

    client.post(PREVIEW.format(ds_id), json={'limit': 6})

    assert sizes == [tr._FACE_BANK_EVERY, 2 * tr._FACE_BANK_EVERY]
    # ...and the completed pass supersedes all of it.
    body = client.get(PREVIEW.format(ds_id)).get_json()
    assert body['resume'] is None
    assert body['result']['coverage']['masked'] == 25
