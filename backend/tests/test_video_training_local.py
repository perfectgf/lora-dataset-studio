"""Launching a video training on THIS machine, and the three ways that can lie.

WHY A SECOND LAUNCHER AND NOT A BRANCH OF `launch_training`
-----------------------------------------------------------
`lora_training.launch_training` reads a `FaceDataset` from its first line to its
last: it exports images, checks captions, resolves a trigger word, names the run
from a base-model tag and freezes an image manifest in the provenance registry. A
video dataset has none of those columns and none of those artefacts — its folder
is ALREADY the flat mp4 + homonym .txt shape ai-toolkit reads. Threading a second
entity through that function would mean a `getattr(ds, ..., None)` at every one of
those steps, which is not a branch, it is a config assembled from defaults nobody
can see. What IS shared is the part that must be shared: the ai-toolkit paths, the
GPU admission sequence, the durable ownership fence and the crash watcher.

THE THREE LIES THIS FILE IS WRITTEN AGAINST
-------------------------------------------
1. A button that starts something which cannot finish. MiniMax H3 needs ~43 GB of
   weights that are not on most machines. Starting anyway means the "training" is
   in fact an unannounced multi-hour download, and on a full disk it is a crash
   an hour in. The size is named BEFORE anything is fetched, and fetching is a
   separate, explicit yes.
2. A run attributed to the wrong dataset. The durable training fence keys on
   `training_dataset_id`, an integer — and face dataset #3 and video dataset #3
   both exist. Without a table stamp the image lane's status panel would name a
   face dataset for a video run, and its Stop button would kill it.
3. A resume from someone else's weights. ai-toolkit resumes automatically from
   whatever it finds in the run folder, and the run folder is named after the
   dataset. Two video datasets of the same name — or one dataset re-promoted to a
   different target — share that folder, and the second run loads the first's
   LoRA into a different architecture.

Nothing here spawns a process or touches a GPU: the spawn seam is injected.
"""
import json
import os
import threading

import pytest

from app.services import video_training_local as vtl


def _video_dataset(tmp_path, name='surf clips', profile='wan22_14b', frames=81,
                   fps=16, width=384, height=384, clips=2, out_dir=None):
    """A promoted video dataset: the row AND the flat folder on disk. The folder
    is load-bearing — the launcher counts clips before it takes the GPU."""
    from app.models import VideoDataset
    from app.extensions import db
    if out_dir is None:
        out_dir = str(tmp_path / f'vds_{name.replace(" ", "_")}')
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, clips + 1):
        with open(os.path.join(out_dir, f'clip_{i:04d}.mp4'), 'wb') as fh:
            fh.write(b'\x00')
        with open(os.path.join(out_dir, f'clip_{i:04d}.txt'), 'w') as fh:
            fh.write('a person walking')
    vd = VideoDataset(user_id='local', name=name, target_profile=profile,
                      fps=fps, frames=frames, width=width, height=height,
                      output_dir=out_dir)
    db.session.add(vd)
    db.session.commit()
    return vd


class _FakeProc:
    """Just enough of a Popen to stand in for the child.

    `wait` blocks forever on purpose. The launcher starts a real watcher thread on
    whatever it spawned, and a process that returns instantly makes that thread
    run the crash/queue post-processing — with its own app context — while the
    test still holds the SQLite connection. Blocking keeps the watcher parked
    exactly where it would be during a real run; the thread is a daemon and dies
    with the interpreter."""

    def __init__(self, pid=424242):
        self.pid = pid
        self.returncode = None
        self._never = threading.Event()

    def wait(self, timeout=None):
        self._never.wait()
        return 0

    def poll(self):
        return None


def _aitoolkit(monkeypatch, tmp_path, models=()):
    """Point every ai-toolkit path accessor at a scratch tree, and declare the
    interpreter ready. `models` lists weight files to create under its models
    folder, so a test can choose whether H3's checkpoints are 'installed'."""
    from app.services import lora_training as lt
    root = tmp_path / 'aitk'
    (root / 'config' / 'generated').mkdir(parents=True, exist_ok=True)
    (root / 'output').mkdir(parents=True, exist_ok=True)
    (root / 'models').mkdir(parents=True, exist_ok=True)
    python = root / 'python.exe'
    python.write_bytes(b'')
    for rel in models:
        path = root / 'models' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\x00')
    monkeypatch.setattr(lt, '_aitoolkit_dir', lambda: root)
    monkeypatch.setattr(lt, '_output_dir', lambda: root / 'output')
    monkeypatch.setattr(lt, '_jobs_dir', lambda: root / 'config' / 'generated')
    monkeypatch.setattr(lt, '_venv_python', lambda: python)
    monkeypatch.setattr(lt, 'is_installed', lambda: True)
    monkeypatch.setattr(lt, 'assert_interpreter_ready', lambda: None)
    monkeypatch.setattr(lt, 'assert_free_disk', lambda *a, **k: None)
    monkeypatch.setattr(lt, 'training_subprocess_env', lambda **k: {})
    return root


def _clear_fence():
    from app.job_queue import queue_manager
    from app.services import lora_training as lt
    lt._clear_training_identity()
    queue_manager._set_system_state('training_error', None, ttl_seconds=1)


@pytest.fixture()
def spawned():
    """Collects the argv the launcher would have spawned, so every test can assert
    on the real command line without a process ever existing."""
    calls = []

    def _spawn(argv, cwd, env, stdout):
        calls.append({'argv': list(argv), 'cwd': str(cwd), 'env': dict(env)})
        return _FakeProc()
    _spawn.calls = calls
    return _spawn


# --- the happy path, and what the child is actually told -----------------------

def test_the_child_is_ai_toolkits_own_cli_on_a_config_we_wrote(
        app, tmp_path, monkeypatch, spawned):
    """The same headless invocation the image lane uses — `run.py <config>` from
    the ai-toolkit root. Not a new protocol, not its web UI's database: the one
    entry point that is already proven on this machine."""
    root = _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        res = vtl.start_video_training('local', vid.id, steps=500, _spawn=spawned)
        assert res['started'] is True and res['pid'] == 424242
        argv = spawned.calls[0]['argv']
        assert argv[1:] == ['run.py', res['config_path']]
        assert spawned.calls[0]['cwd'] == str(root)
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        proc = cfg['config']['process'][0]
        assert proc['model']['arch'] == 'wan22_14b'
        assert proc['datasets'][0]['num_frames'] == 81
        assert proc['train']['steps'] == 500
        _clear_fence()


def test_the_clips_are_trained_from_the_dataset_folder_itself(
        app, tmp_path, monkeypatch, spawned):
    """No export, no staging copy. The promoted folder is already the shape
    ai-toolkit reads, and a dataset of 81-frame clips is gigabytes — copying it to
    an `aitoolkit/datasets/` mirror would double the disk for nothing."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        out_dir = vid.output_dir
        res = vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        assert cfg['config']['process'][0]['datasets'][0]['folder_path'] == out_dir
        _clear_fence()


def test_the_run_gets_its_own_folder_and_its_log_sits_above_the_saves(
        app, tmp_path, monkeypatch, spawned):
    """`training_folder` must be resolved locally — `video_training` has no run
    root of its own and would otherwise emit ai-toolkit's relative 'output',
    scattering runs wherever the child's working directory happened to be.

    The log and the run marker live in that folder, and ai-toolkit's save root is
    one level below it. That separation is the image lane's, and its reason is
    that a checkpoint scan of the save root must see checkpoints and nothing
    else — a `training.log` listed as a save is a download button that 404s."""
    root = _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        res = vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        run_root = root / 'output' / res['run_name']
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        assert cfg['config']['process'][0]['training_folder'] == str(run_root)
        assert cfg['config']['name'] == res['run_name']
        # the log exists from the first second, so a crash before ai-toolkit's
        # first line is readable rather than "log missing"
        assert os.path.isfile(res['log_path'])
        assert os.path.dirname(res['log_path']) == str(run_root)
        assert str(vtl.save_root(vid)) == str(run_root / res['run_name'])
        _clear_fence()


# --- lie #1: a button that starts a download it never mentioned ----------------

def test_h3_names_its_repository_and_its_size_instead_of_downloading_it(
        app, tmp_path, monkeypatch, spawned):
    """~43 GB, from `Comfy-Org/MiniMax-H3`. Fetched silently, the button appears
    to hang for hours and then fails on a full disk. The refusal names the repo
    and the size, and it is raised BEFORE the GPU fence is taken — the card stays
    free while the user decides."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24)
        with pytest.raises(vtl.VideoWeightsMissing) as e:
            vtl.start_video_training(
                'local', vid.id, steps=100,
                _spawn=lambda *a, **k: pytest.fail('a run started anyway'))
        msg = str(e.value)
        assert 'Comfy-Org/MiniMax-H3' in msg
        assert '43' in msg
        assert e.value.gigabytes == 43
        assert e.value.repo == 'Comfy-Org/MiniMax-H3'
        _clear_fence()


def test_the_download_happens_only_when_it_is_explicitly_accepted(
        app, tmp_path, monkeypatch, spawned):
    """The consent is a parameter, not a config setting: a user who accepted 43 GB
    once for H3 has not accepted it forever, nor for the next architecture."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24)
        res = vtl.start_video_training('local', vid.id, steps=100,
                                       accept_download=True, _spawn=spawned)
        assert res['started'] is True
        assert res['downloading'] is True
        _clear_fence()


def test_the_refusal_says_how_much_room_the_target_drive_actually_has(
        app, tmp_path, monkeypatch):
    """"43 GB" alone leaves the user to go and check. The number that decides
    whether this is a wait or an impossibility is the FREE space on the drive the
    files would land on — and on a machine where that is 26 GB, "download?" is the
    wrong question entirely. Both numbers, and the drive, in one sentence."""
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(lt, 'free_disk_gb', lambda path: 26.5)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24)
        with pytest.raises(vtl.VideoWeightsMissing) as e:
            vtl.start_video_training('local', vid.id, steps=100)
        msg = str(e.value)
        assert '43' in msg and '26.5' in msg
        assert e.value.free_gigabytes == 26.5
        # ...and it points at the setting that moves the destination, because a
        # full drive here is not a full machine: another disk usually has room.
        assert 'ai-toolkit folder' in msg.lower() or 'settings' in msg.lower()
        _clear_fence()


def test_a_drive_that_cannot_be_measured_still_names_the_download(
        app, tmp_path, monkeypatch):
    """`free_disk_gb` answers None on a stat failure. The refusal must degrade to
    the size alone rather than printing "None GB free" or, worse, deciding the
    disk is empty and going ahead."""
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(lt, 'free_disk_gb', lambda path: None)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24)
        with pytest.raises(vtl.VideoWeightsMissing) as e:
            vtl.start_video_training('local', vid.id, steps=100)
        assert '43' in str(e.value)
        assert 'None' not in str(e.value)
        assert e.value.free_gigabytes is None
        _clear_fence()


# --- the size the model was trained at, stated rather than hidden --------------

def test_a_dataset_far_below_the_targets_own_sizes_is_warned_about_not_refused(
        app, tmp_path, monkeypatch, spawned):
    """MiniMax H3's own recommended sizes are all short-edge 768; a 384x384 set is
    perfectly legal and will train, but it is a long way from what the model saw.
    That is a limit to STATE, not one to enforce — refusing would block a
    deliberate low-resolution run, and saying nothing would let someone spend a
    night discovering it. The note names both numbers."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24,
                             width=384, height=384)
        note = vtl.resolution_note(vid)
        assert note and '384' in note and '768' in note
        res = vtl.start_video_training('local', vid.id, steps=100,
                                       accept_download=True, _spawn=spawned)
        assert note in res['warnings']
        _clear_fence()


def test_a_dataset_at_the_targets_own_size_draws_no_note(
        app, tmp_path, monkeypatch, spawned):
    """The guard has to be quiet when there is nothing to say, or it becomes the
    banner everyone learns to skip."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24,
                             width=768, height=768)
        assert vtl.resolution_note(vid) is None
        res = vtl.start_video_training('local', vid.id, steps=100,
                                       accept_download=True, _spawn=spawned)
        assert res['warnings'] == []
        _clear_fence()


def test_a_target_that_states_no_sizes_says_nothing_rather_than_guessing(
        app, tmp_path, monkeypatch):
    """`recommended_sizes` is deliberately EMPTY for every Wan profile — no local
    source states one, and the catalogue refuses to invent them. A note derived
    from a number we do not have would be exactly the dressed-up guess that field
    exists to avoid."""
    from app.services import video_targets as vt
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        assert vt.get('wan22_14b')['recommended_sizes'] == ()
        vid = _video_dataset(tmp_path, width=384, height=384)
        assert vtl.resolution_note(vid) is None
        _clear_fence()


def test_the_progress_route_carries_the_note_before_anything_is_clicked(
        app, client, tmp_path, monkeypatch):
    """A warning that only arrives after the launch is a warning about a decision
    already taken. The card polls this endpoint on mount, so the note is on screen
    before the button is pressed."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid_id = _video_dataset(tmp_path, profile='minimax_h3', frames=107,
                                fps=24, width=384, height=384).id
    body = client.get(f'/api/video-dataset/{vid_id}/train/progress').get_json()
    assert body['resolution_note'] and '768' in body['resolution_note']


def test_weights_already_on_disk_are_not_re_announced(
        app, tmp_path, monkeypatch, spawned):
    """The check is a file probe, not a flag. With the four checkpoints present
    under the models folder there is nothing to download, no consent to ask for,
    and the launch goes straight through."""
    files = vtl.WEIGHT_FOOTPRINTS['minimax_h3']['files']
    _aitoolkit(monkeypatch, tmp_path, models=files)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24)
        res = vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        assert res['downloading'] is False
        _clear_fence()


def test_a_flat_models_folder_counts_as_installed_too(app, tmp_path, monkeypatch,
                                                      spawned):
    """ai-toolkit's H3 loader looks for each file at its repo-relative path AND
    flat at the root of the models folder. Probing only the first would announce a
    43 GB download to someone who already has the weights."""
    files = vtl.WEIGHT_FOOTPRINTS['minimax_h3']['files']
    _aitoolkit(monkeypatch, tmp_path,
               models=[os.path.basename(f) for f in files])
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='minimax_h3', frames=107, fps=24)
        assert vtl.start_video_training(
            'local', vid.id, steps=100, _spawn=spawned)['downloading'] is False
        _clear_fence()


def test_an_architecture_with_no_declared_footprint_is_not_gated(
        app, tmp_path, monkeypatch, spawned):
    """Wan's base is a diffusers repo resolved through the Hugging Face cache, not
    a list of files under the models folder — this build cannot prove whether it is
    present. Inventing a refusal there would block the one lane that is PROVEN to
    work end to end, on a machine where the weights are already cached."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        res = vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        assert res['downloading'] is None       # not False: unknown, and said so
        _clear_fence()


# --- lie #2: a run attributed to the wrong dataset -----------------------------

def test_the_fence_records_which_table_the_id_points_into(
        app, tmp_path, monkeypatch, spawned):
    """The durable fence is one integer. Stamping the table is what stops the
    image lane's status panel from resolving it as a face dataset."""
    from app.services import cloud_run_dataset as crd
    from app.job_queue import queue_manager
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        assert queue_manager._get_system_state(
            'training_dataset_table', None) == crd.VIDEO
        assert queue_manager._get_system_state('training_dataset_id', None) == vid.id
        assert queue_manager._get_system_state('training_in_progress', False) is True
        _clear_fence()


def test_the_image_status_panel_does_not_rename_a_video_run(
        app, tmp_path, monkeypatch, spawned):
    """`training_status` resolved the fence's id with `FaceDataset.query.get`. On a
    colliding id that names a face dataset the user is not training — the run
    would appear on the wrong page, under the wrong name, with a Stop button."""
    from app.models import FaceDataset
    from app.extensions import db
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        face = FaceDataset(user_id='local', name='portraits', trigger_word='trg')
        db.session.add(face)
        db.session.commit()
        vid = _video_dataset(tmp_path, 'surf clips')
        assert face.id == vid.id                 # the collision this test is about
        vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        current = lt.training_status()['current']
        assert current['name'] == 'surf clips'
        assert current['train_type'] == 'video'
        assert current['dataset_table'] == 'video_dataset'
        _clear_fence()


def test_a_face_datasets_stop_button_cannot_kill_a_video_run(
        app, tmp_path, monkeypatch, spawned):
    """`stop_training(expected_dataset_id=N)` compared integers alone, so the Stop
    button of face dataset #N would have killed the video run of video dataset #N
    — hours of GPU, ended by a click on another page."""
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        assert lt.stop_training(expected_dataset_id=vid.id) is False
        # ...and the video lane's own stop, which names the table, is accepted as
        # far as the identity check goes.
        assert lt.stop_training(expected_dataset_id=vid.id,
                                expected_dataset_table='video_dataset') is not False
        _clear_fence()


# --- lie #3: resuming from another dataset's weights ---------------------------

def test_two_datasets_of_the_same_name_do_not_share_a_run_folder(
        app, tmp_path, monkeypatch, spawned):
    """The run folder is named after the dataset, and ai-toolkit resumes from
    whatever LoRA it finds there. Two promotions called "surf clips" would put the
    second run on top of the first's checkpoints — a continuation the user never
    asked for, reported as a fresh training."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        a = _video_dataset(tmp_path, 'surf clips', out_dir=str(tmp_path / 'a'))
        b = _video_dataset(tmp_path, 'surf clips', out_dir=str(tmp_path / 'b'))
        first = vtl.start_video_training('local', a.id, steps=100, _spawn=spawned)
        _clear_fence()
        second = vtl.start_video_training('local', b.id, steps=100, _spawn=spawned)
        assert first['run_name'] != second['run_name']
        _clear_fence()


def test_a_run_folder_built_for_another_architecture_is_refused(
        app, tmp_path, monkeypatch, spawned):
    """Re-promoting a dataset to a different target keeps its id and its name, so
    the run folder is the same one — and ai-toolkit would load a Wan LoRA into an
    H3 transformer. What that raises, if it raises at all, is a shape error deep
    in a state dict; here it is a sentence naming both targets."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        _clear_fence()
        vid.target_profile = 'wan21'
        from app.extensions import db
        db.session.commit()
        with pytest.raises(ValueError) as e:
            vtl.start_video_training(
                'local', vid.id, steps=100,
                _spawn=lambda *a, **k: pytest.fail('it resumed anyway'))
        assert 'wan22_14b' in str(e.value) and 'wan21' in str(e.value)
        _clear_fence()


def test_the_same_dataset_on_the_same_target_still_continues(
        app, tmp_path, monkeypatch, spawned):
    """The guard must not break the feature it protects: relaunching the SAME
    dataset on the SAME target is how a run is continued past its step count, and
    that is exactly what the marker has to allow."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        _clear_fence()
        assert vtl.start_video_training(
            'local', vid.id, steps=200, _spawn=spawned)['started'] is True
        _clear_fence()


# --- the refusals shared with every other GPU owner ----------------------------

def test_a_vision_pass_on_the_card_refuses_the_launch(
        app, tmp_path, monkeypatch, spawned):
    """One GPU, one owner. A captioning or embedding pass and a training run do
    not share the card, and the video lane must take the SAME guard as the image
    lane rather than inventing a second, weaker one."""
    from app.gpu_window import GpuBusyError
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(lt, '_assert_no_vision_pass_on_gpu',
                        lambda: (_ for _ in ()).throw(GpuBusyError('vision')))
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        with pytest.raises(GpuBusyError):
            vtl.start_video_training(
                'local', vid.id, steps=100,
                _spawn=lambda *a, **k: pytest.fail('it took the GPU anyway'))
        _clear_fence()


def test_a_training_already_running_refuses_the_launch(
        app, tmp_path, monkeypatch, spawned):
    """Two ai-toolkit processes on one card corrupt each other's run. The fence is
    shared with the image lane on purpose: an image training in flight must block
    a video launch exactly as another video launch would."""
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        vtl.start_video_training('local', vid.id, steps=100, _spawn=spawned)
        # The fence only refuses on a process that is not PROVABLY dead; the fake
        # pid is nobody's, so the liveness probe is pinned rather than the guard.
        monkeypatch.setattr(lt, '_training_process_is_definitely_dead',
                            lambda pid: False)
        with pytest.raises(ValueError) as e:
            vtl.start_video_training(
                'local', vid.id, steps=100,
                _spawn=lambda *a, **k: pytest.fail('a second child was spawned'))
        assert 'in progress' in str(e.value)
        _clear_fence()


def test_an_unconfigured_ai_toolkit_is_an_availability_problem(
        app, tmp_path, monkeypatch):
    """RuntimeError, which the route maps to 409 — the request was fine, the
    backend is not installed. A 400 would tell the user they asked wrongly."""
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(lt, 'is_installed', lambda: False)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        with pytest.raises(RuntimeError):
            vtl.start_video_training('local', vid.id, steps=100)
        _clear_fence()


def test_an_empty_dataset_folder_is_refused(app, tmp_path, monkeypatch):
    """The row can outlive its clips (a deleted folder, a failed promotion). Left
    to ai-toolkit this is a run that trains on nothing and saves a LoRA of noise."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, clips=0, out_dir=str(tmp_path / 'empty'))
        with pytest.raises(ValueError) as e:
            vtl.start_video_training('local', vid.id, steps=100)
        assert 'clip' in str(e.value).lower()
        _clear_fence()


def test_an_unknown_dataset_is_refused_by_name(app, tmp_path, monkeypatch):
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        with pytest.raises(ValueError) as e:
            vtl.start_video_training('local', 987654, steps=100)
        assert 'not found' in str(e.value)


def test_an_unsupported_target_is_refused_before_the_gpu(
        app, tmp_path, monkeypatch):
    """The same builder refusal the cloud lane gets, in the same position: before
    anything is reserved. Locally the cost is not money, it is a card taken away
    from ComfyUI for a run that was never going to start."""
    from app.services import video_training as vt
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path, profile='generic', frames=40)
        with pytest.raises(vt.VideoTrainingUnsupported):
            vtl.start_video_training(
                'local', vid.id, steps=100,
                _spawn=lambda *a, **k: pytest.fail('it launched anyway'))
        _clear_fence()


# --- progress, read from the same log the image lane reads ---------------------

def test_progress_reads_the_video_runs_own_log(app, tmp_path, monkeypatch,
                                               spawned):
    """The step/loss parser is the image lane's, unchanged — ai-toolkit writes the
    same lines whatever it trains. What must be video-specific is only WHICH log
    is opened, which is the one fact a face dataset's id cannot supply."""
    from app.services import lora_training as lt
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(lt, '_training_process_is_definitely_dead',
                        lambda pid: False)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        res = vtl.start_video_training('local', vid.id, steps=500, _spawn=spawned)
        with open(res['log_path'], 'a', encoding='utf-8') as fh:
            fh.write('\n 42%|####      | 210/500 [1:02:03<00:40:00,  0.05it/s, '
                     'lr: 1.0e-04 loss: 2.3e-01]\n')
        prog = vtl.video_training_progress(vid.id)
        assert prog['step'] == 210 and prog['total'] == 500
        assert prog['active'] is True
        _clear_fence()


def test_progress_of_a_dataset_that_never_ran_is_empty_not_an_error(
        app, tmp_path, monkeypatch):
    """The panel polls before the first launch too."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        prog = vtl.video_training_progress(vid.id)
        assert prog['active'] is False and prog['log_exists'] is False


# --- the HTTP surface ---------------------------------------------------------

def test_the_launch_route_starts_a_local_run(app, client, tmp_path, monkeypatch,
                                             spawned):
    """Without a route the lane is reachable only from a Python shell. This is
    also the one place a user-supplied step count arrives from outside."""
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(vtl, '_default_spawn', spawned)
    with app.app_context():
        _clear_fence()
        vid_id = _video_dataset(tmp_path).id
    r = client.post(f'/api/video-dataset/{vid_id}/train', json={'steps': 700})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['started'] is True and body['steps'] == 700 and body['clips'] == 2
    with app.app_context():
        _clear_fence()


def test_the_launch_route_hands_the_panel_the_size_of_the_download(
        app, client, tmp_path, monkeypatch):
    """A 409 rather than a 400: nothing is wrong with the request, the machine
    simply does not have the weights yet. The repository and the gigabytes travel
    in the body so the panel can ask "download 43 GB?" instead of "error"."""
    _aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(vtl, '_default_spawn',
                        lambda *a, **k: pytest.fail('a run started anyway'))
    with app.app_context():
        _clear_fence()
        vid_id = _video_dataset(tmp_path, profile='minimax_h3', frames=107,
                                fps=24).id
    r = client.post(f'/api/video-dataset/{vid_id}/train', json={})
    assert r.status_code == 409
    body = r.get_json()
    assert body['needs_download'] is True
    assert body['repo'] == 'Comfy-Org/MiniMax-H3'
    assert body['gigabytes'] == 43


def test_the_launch_route_reports_an_unsupported_target_as_a_refusal(
        app, client, tmp_path, monkeypatch):
    """400, not 500: the user picked a target this build cannot train, and the
    message names what to do about it."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        vid_id = _video_dataset(tmp_path, profile='generic', frames=40).id
    r = client.post(f'/api/video-dataset/{vid_id}/train', json={})
    assert r.status_code == 400
    assert 'generic' in r.get_json()['error']


def test_an_unknown_video_dataset_is_a_404(app, client, tmp_path, monkeypatch):
    _aitoolkit(monkeypatch, tmp_path)
    assert client.post('/api/video-dataset/98765/train', json={}).status_code == 404
    assert client.get(
        '/api/video-dataset/98765/train/progress').status_code == 404


def test_the_stop_route_refuses_a_run_that_is_not_this_datasets(
        app, client, tmp_path, monkeypatch, spawned):
    """The button reports what happened. A stop that named another run must come
    back `ok: false` — silently reporting success would leave the user believing a
    GPU was released while ai-toolkit still owned it."""
    _aitoolkit(monkeypatch, tmp_path)
    with app.app_context():
        _clear_fence()
        a = _video_dataset(tmp_path, 'one', out_dir=str(tmp_path / 'one'))
        b = _video_dataset(tmp_path, 'two', out_dir=str(tmp_path / 'two'))
        a_id, b_id = a.id, b.id
        vtl.start_video_training('local', a_id, steps=100, _spawn=spawned)
    assert client.post(
        f'/api/video-dataset/{b_id}/train/stop').get_json()['ok'] is False
    with app.app_context():
        _clear_fence()


def test_the_recipe_is_offered_only_to_a_toolkit_that_can_run_it(
        app, tmp_path, monkeypatch):
    """ai-toolkit has no version to read - it is a git checkout whose capabilities
    move commit by commit - so the probe reads the ARCHITECTURE'S OWN SOURCE for
    the method that consumes the setting. `load_training_adapter` arrived on
    2026-08-06; on anything older the generic loader takes over and raises "Only
    Flux models can load assistant adapters currently", which ends the run.

    Everything unreadable answers False: no folder, no file, no arch. A false
    negative costs a recipe, a false positive costs the run."""
    from app.services import video_training_local as vtl
    _aitoolkit(monkeypatch, tmp_path)
    arch_dir = (tmp_path / 'aitk' / 'extensions_built_in' / 'diffusion_models'
                / 'minimax_h3')

    # No extensions tree at all - the shape of an install that predates the arch.
    assert vtl.supports_training_adapter('minimax_h3') is False

    # The arch is there, but from before the method existed.
    arch_dir.mkdir(parents=True)
    source = arch_dir / 'minimax_h3.py'
    source.write_text('''class MiniMaxH3:
    def load_model(self): pass
''', encoding='utf-8')
    assert vtl.supports_training_adapter('minimax_h3') is False

    # And with the method present.
    source.write_text('''class MiniMaxH3:
    def load_training_adapter(self, t): pass
''', encoding='utf-8')
    assert vtl.supports_training_adapter('minimax_h3') is True

    # An arch with no recipe never touches the filesystem to find that out.
    assert vtl.supports_training_adapter('wan22_14b') is False
    assert vtl.supports_training_adapter(None) is False
