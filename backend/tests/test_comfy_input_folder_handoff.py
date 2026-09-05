r"""The FILESYSTEM half of the ComfyUI contract must fail out loud.

Reported on Discord by nofaceman: LoRA Dataset Studio in Docker, ComfyUI in a
SECOND container. Setup went green (URL + directory both accepted), then every
generation answered a bare `500` with no detail — "probably no network access".

It was not the network. The app reaches ComfyUI two ways, and only one is HTTP:
the API URL, and the input/ folder it COPIES every source image into. Two
containers do not share that folder by default, so `shutil.copy2` raised an
OSError no route maps -> Flask 500, no body, nothing to act on.

These tests hold the line on what was actually missing: the failure must NAME the
operation, the folder and the cause, must arrive as a mapped 409 rather than a
500, and must be safe to paste in a public thread.
"""
import os
from unittest.mock import patch

import pytest
from PIL import Image

from app.utils import comfy_fs


# --- the message: names the operation, the folder, and what to do -------------

def test_missing_input_folder_is_named_not_swallowed(tmp_path):
    """RED before the fix: the copy died on a raw FileNotFoundError. Now the
    refusal happens BEFORE the copy and says which folder is missing."""
    ghost = tmp_path / 'comfy' / 'input'
    with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
        comfy_fs.ensure_input_usable(str(ghost))
    msg = str(exc.value)
    # The path is REDACTED (tmp dirs live under the home dir) — what must survive
    # is the folder itself, unambiguously named.
    assert comfy_fs.safe_path(ghost) in msg
    assert msg.endswith('folder.') or 'comfy' in msg
    assert 'input folder' in msg
    assert 'does not exist' in msg
    assert 'shared volume' in msg          # the container/remote case, named


def test_unwritable_input_folder_is_named(tmp_path, monkeypatch):
    """The Docker case proper: the folder EXISTS (it is in the image) but this
    process cannot write into it — a read-only bind mount, or a different uid.

    The denial is injected at the probe seam rather than by chmod: a read-only
    directory is not reproducible across Windows/POSIX/CI, and what is under test
    is the MESSAGE. The real probe is exercised by the round-trip test below."""
    folder = tmp_path / 'input'
    folder.mkdir()
    monkeypatch.setattr(comfy_fs, '_write_probe',
                        lambda p: 'PermissionError: [Errno 13] Permission denied')
    with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
        comfy_fs.ensure_input_usable(str(folder))
    msg = str(exc.value)
    assert comfy_fs.safe_path(folder) in msg
    assert 'not writable' in msg
    assert 'PermissionError' in msg        # the CAUSE, not just the symptom
    assert 'shared volume' in msg


def test_write_probe_round_trips_and_leaves_nothing_behind(tmp_path):
    """A usable folder is judged usable, and the probe file never survives."""
    assert comfy_fs.folder_problem('input', str(tmp_path)) == ''
    assert list(tmp_path.iterdir()) == []


def test_copy_failure_names_the_operation_and_the_folder(tmp_path):
    """Everything checks out and the copy STILL fails (disk full, a mount that
    went away between the probe and the write). Same contract."""
    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    src.write_bytes(b'\x89PNG\r\n\x1a\n')
    with patch.object(comfy_fs.shutil, 'copy2', side_effect=OSError('disk full')):
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_copy(str(src), 'x.png', str(folder))
    msg = str(exc.value)
    assert 'Copying the source image into ComfyUI' in msg
    assert comfy_fs.safe_path(folder) in msg
    assert 'OSError: disk full' in msg


def test_generated_file_write_failure_is_named_too(tmp_path):
    """The watermark lane WRITES its crop instead of copying one — same guard."""
    folder = tmp_path / 'input'
    folder.mkdir()

    def _boom(path):
        raise OSError('read-only file system')

    with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
        comfy_fs.stage_input_write('crop.png', _boom, str(folder))
    assert 'Writing the source image into ComfyUI' in str(exc.value)
    assert 'read-only file system' in str(exc.value)


# --- paste-safety: these strings are written to be dropped in a public thread --

@pytest.mark.parametrize('api_url', [
    'https://review:review-password@comfy.invalid/api',
    'https://comfy.invalid/api?token=review-token#review-fragment',
])
def test_input_mismatch_message_hides_url_credentials(api_url, monkeypatch):
    monkeypatch.setattr(comfy_fs.cfg, 'get', lambda key: api_url)
    monkeypatch.setattr(comfy_fs, '_comfy_folder_note', lambda: '')
    msg = comfy_fs._mismatch_message('a file', '/home/somebody/ComfyUI/input')
    assert 'https://comfy.invalid/api' in msg
    assert 'review' not in msg
    assert 'somebody' not in msg
    assert '~/ComfyUI/input' in msg


def test_messages_never_expose_a_user_home_path():
    r"""The whole point of a readable error is that people paste it. A raw
    `C:\Users\<account>\...` would leak the OS account name into #help."""
    home = r'C:\Users\somebody\ComfyUI\input'
    problem = comfy_fs.folder_problem('input', home)
    assert 'somebody' not in problem
    assert '~' in problem and 'ComfyUI' in problem

    posix = '/home/somebody/comfy/input'
    assert 'somebody' not in comfy_fs.folder_problem('input', posix)


def test_staging_error_is_redacted_too(tmp_path):
    """Redaction applies to the CAUSE as well — an exception repr routinely
    carries the destination path."""
    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    src.write_bytes(b'x')
    with patch.object(comfy_fs.shutil, 'copy2',
                      side_effect=OSError(r'cannot write C:\Users\somebody\in\x.png')):
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_copy(str(src), 'x.png', str(folder))
    assert 'somebody' not in str(exc.value)


# --- it is a RuntimeError, i.e. an actionable 409, never a bare 500 -----------

def test_failure_class_maps_to_409_not_500(app):
    """`_map_error` re-raises anything it doesn't know (-> 500). Being a
    RuntimeError is what turns this into a 409 carrying the message; the test
    drives the real mapper rather than trusting the class hierarchy."""
    from app.routes._common import _map_error
    with app.app_context():
        body, status = _map_error(comfy_fs.ComfyFolderUnavailable('nope'))
        assert status == 409
        assert body.get_json()['error'] == 'nope'


# --- end to end through the engines that stage a file ------------------------

def _klein_ready(monkeypatch, keh, comfy_in):
    monkeypatch.setattr(keh, '_comfy_input_dir', lambda: str(comfy_in))
    monkeypatch.setattr(keh, 'resolve_klein_unet', lambda *a, **k: 'unet.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_vae', lambda *a, **k: 'vae.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_text_encoder', lambda *a, **k: 'te.safetensors')
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda *a, **k: [])


def test_klein_generation_refuses_with_a_reason(app, tmp_path, monkeypatch):
    """The user-facing repro: generate with an input folder that isn't there.
    Before: OSError -> 500, no detail. After: a named refusal, and NO job queued."""
    from app.services import klein_edit_helper as keh
    src = tmp_path / 'src.png'
    src.write_bytes(b'\x89PNG\r\n\x1a\n')
    queued = []
    with app.app_context():
        _klein_ready(monkeypatch, keh, tmp_path / 'not-mounted')
        monkeypatch.setattr(keh.queue_manager, 'add_job',
                            lambda **kw: queued.append(kw))
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            keh.enqueue_klein_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='hi')
    assert 'not-mounted' in str(exc.value)
    assert 'shared volume' in str(exc.value)
    assert queued == []          # nothing enqueued that could only have failed


def test_klein_generation_still_works_on_a_normal_install(app, tmp_path, monkeypatch):
    """Non-regression, the half that matters most: a plain single-machine install
    stages its source exactly as before and enqueues the job."""
    from app.services import klein_edit_helper as keh
    comfy_in = tmp_path / 'input'
    comfy_in.mkdir()
    src = tmp_path / 'src.png'
    Image.new('RGB', (32, 24), (30, 120, 220)).save(src, 'PNG')
    seen = {}
    with app.app_context():
        _klein_ready(monkeypatch, keh, comfy_in)
        monkeypatch.setattr(keh.queue_manager, 'add_job', lambda **kw: seen.update(kw))
        keh.enqueue_klein_edit(user_id='local', source_filename='src.png',
                               source_path=str(src), edit_prompt='hi')
    staged = [p.name for p in comfy_in.iterdir()]
    assert len(staged) == 1 and staged[0].endswith('src.png')
    assert seen['workflow_data']['52']['inputs']['image'] == staged[0]


def test_krea_generation_refuses_with_a_reason(app, tmp_path, monkeypatch):
    """The second local engine stages the same way and must answer the same way —
    a fix that only covered Klein would leave the identical 500 one engine away."""
    from app.services import krea_edit_helper as keh2
    src = tmp_path / 'src.png'
    src.write_bytes(b'\x89PNG\r\n\x1a\n')
    with app.app_context():
        monkeypatch.setattr(keh2, '_comfy_input_dir', lambda: str(tmp_path / 'not-mounted'))
        monkeypatch.setattr(keh2, 'preflight', lambda *a, **k: None)
        monkeypatch.setattr(keh2, 'resolve_krea_unet', lambda *a, **k: 'unet.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_text_encoder', lambda *a, **k: 'te.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_vae', lambda *a, **k: 'vae.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_identity_lora',
                            lambda *a, **k: ('lora.safetensors', 'x'))
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            keh2.enqueue_krea_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='hi')
    assert 'not-mounted' in str(exc.value)


def test_watermark_klein_degrades_instead_of_raising(app, tmp_path, monkeypatch):
    """The watermark lane returns (image, error) instead of raising, so its guard
    must feed that contract — 'unavailable' + the same explanatory message."""
    from PIL import Image
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path / 'not-mounted'))
    monkeypatch.setattr(wk.keh, 'resolve_klein_unet', lambda *a, **k: 'unet.safetensors')
    monkeypatch.setattr(wk.keh, 'resolve_klein_vae', lambda *a, **k: 'vae.safetensors')
    monkeypatch.setattr(wk.keh, 'resolve_klein_text_encoder', lambda *a, **k: 'te.safetensors')
    monkeypatch.setattr(wk.keh, 'klein_missing_assets', lambda *a, **k: [])
    with app.app_context():
        img, err = wk._run_klein_job('local', Image.new('RGB', (8, 8)), seed=1)
    assert img is None
    assert err['kind'] == 'unavailable'
    assert 'not-mounted' in err['detail'] and 'shared volume' in err['detail']


# --- Setup / Settings: warn at configuration time, never block ----------------

def test_settings_preview_reports_a_present_but_unusable_folder(tmp_path, monkeypatch):
    from app import capabilities
    base = tmp_path / 'Comfy'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir()
    monkeypatch.setattr(comfy_fs, '_write_probe', lambda p: 'PermissionError: denied')
    r = capabilities.classify_comfyui_folders(str(base), {})
    assert r['input_dir']['exists'] is True          # the old check: green
    assert r['input_dir']['usable'] is False         # the other half: honest
    assert 'not writable' in r['input_dir']['problem']
    # A folder the app only READS from is not write-probed — no phantom warning.
    assert r['output_dir']['usable'] is True and r['output_dir']['problem'] == ''


def test_settings_preview_stays_quiet_when_all_is_well(tmp_path):
    from app import capabilities
    base = tmp_path / 'Comfy'
    (base / 'input').mkdir(parents=True)
    r = capabilities.classify_comfyui_folders(str(base), {})
    assert r['input_dir']['usable'] is True
    assert r['input_dir']['problem'] == ''
    # Nothing to probe when the folder isn't there: `exists` already says it.
    assert r['models_dir']['usable'] is None


def test_setup_wizard_verdict_warns_without_blocking(app, tmp_path, monkeypatch):
    """The wizard's answer stays 'valid' — someone may configure the app before
    mounting their volumes — but it now carries the input-folder verdict."""
    from app import capabilities
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    (comfy / 'input').mkdir()
    monkeypatch.setattr(comfy_fs, '_write_probe', lambda p: 'PermissionError: denied')
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['status'] == 'valid'                      # NOT downgraded, not blocking
    assert r['input_check']['ok'] is False
    assert 'not writable' in r['input_check']['problem']


def test_setup_wizard_verdict_is_green_on_a_working_install(app, tmp_path):
    from app import capabilities
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    (comfy / 'input').mkdir()
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['status'] == 'valid' and r['input_check']['ok'] is True


def test_setup_wizard_honours_a_saved_input_override(app, tmp_path):
    """The wizard must judge the folder the app would REALLY use, not <base>/input
    — someone whose ComfyUI runs with --input-directory already set an override."""
    from app import capabilities
    import app.config as config
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    override = tmp_path / 'mounted-input'
    override.mkdir()
    with app.app_context():
        config.save_config({'comfyui': {'input_dir': str(override)}})
        r = capabilities.classify_comfyui_dir(str(comfy))
    # <base>/input does not even exist; the override does -> the verdict is green.
    assert not (comfy / 'input').exists()
    assert r['input_check']['ok'] is True
    assert r['input_check']['path'].endswith('mounted-input')


def _valid_comfy(tmp_path):
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    (comfy / 'input').mkdir()
    return comfy


def test_setup_wizard_offers_the_input_folder_comfyui_reports_when_it_cannot_see_ours(
        app, tmp_path, monkeypatch):
    """GitHub #64 (mikemil828, Comfy Desktop with its shared folder): the folder
    ComfyUI was started with was known to the app, but offered only inside an
    Advanced fold of Settings. Once ComfyUI has proved it does not read ours, the
    wizard offers the one it reports — reported, never inferred."""
    from app import capabilities
    comfy = _valid_comfy(tmp_path)
    monkeypatch.setattr(comfy_fs, 'input_visibility_problem',
                        lambda p: 'ComfyUI cannot see the source image the app just staged.')
    monkeypatch.setattr(capabilities, 'detect_comfyui_folders',
                        lambda *a, **k: {'input_dir': 'D:/ComfyUI-Shared/input'})
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['status'] == 'valid'
    assert r['input_check']['ok'] is False
    assert r['input_check']['suggestion'] == 'D:/ComfyUI-Shared/input'


def test_no_suggestion_when_comfyui_says_nothing_or_names_our_own_folder(
        app, tmp_path, monkeypatch):
    from app import capabilities
    comfy = _valid_comfy(tmp_path)
    monkeypatch.setattr(comfy_fs, 'input_visibility_problem', lambda p: 'cannot see')
    monkeypatch.setattr(capabilities, 'detect_comfyui_folders', lambda *a, **k: {})
    with app.app_context():
        assert capabilities.classify_comfyui_dir(str(comfy))['input_check']['suggestion'] == ''
    # The same folder in ComfyUI's own spelling is not a fix, so it is not offered.
    monkeypatch.setattr(capabilities, 'detect_comfyui_folders',
                        lambda *a, **k: {'input_dir': str(comfy / 'input')})
    with app.app_context():
        assert capabilities.classify_comfyui_dir(str(comfy))['input_check']['suggestion'] == ''


def test_a_visible_input_folder_never_asks_comfyui_for_a_suggestion(app, tmp_path, monkeypatch):
    from app import capabilities
    comfy = _valid_comfy(tmp_path)

    def _never(*a, **k):
        raise AssertionError('detect_comfyui_folders must not run when the probe is green')
    monkeypatch.setattr(capabilities, 'detect_comfyui_folders', _never)
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['input_check']['ok'] is True and r['input_check']['suggestion'] == ''


def test_setup_route_carries_the_input_check(client, tmp_path):
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    r = client.get(f'/api/setup/comfyui-dir?path={comfy}')
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'valid'
    assert 'input_check' in body
    # <base>/input isn't there: the wizard says so rather than certifying it.
    assert body['input_check']['ok'] is False
    assert os.sep in body['input_check']['path']


# --- the folder is there, writable, and NOT the one ComfyUI reads -------------
#
# GitHub #64 (mikemil828): Krea 2, "it stops instantly without an error message".
# Every local check was green — the folder existed, the app wrote into it — and
# ComfyUI answered 400 `custom_validation_failed: image - Invalid image file:
# krea_source_….png`, a body that reached the log and nowhere else. The tests
# below hold the two halves of the fix: the app ASKS ComfyUI whether it can see
# what was just staged, and a "no" becomes the same named 409 as every other
# failure in this file — never a queued job that can only die.

# The autouse conftest fixture replaces `comfyui_sees_input` with "could not ask"
# so that no test reaches a live ComfyUI (config.py's DEFAULTS make api_url
# resolve on every machine). The probe's OWN contract still has to be exercised,
# so the real functions are captured here, at import time — before any fixture
# runs — and reinstalled by the tests that are about them.
_REAL_SEES_INPUT = comfy_fs.comfyui_sees_input
_REAL_FOLDER_NOTE = comfy_fs._comfy_folder_note


class _Answer:
    """The only two things the probe reads off an HTTP response."""

    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


@pytest.mark.parametrize('status, verdict', [
    (200, True),      # ComfyUI served it: same folder, proven
    (404, False),     # ComfyUI looked and it is not there: THE refusal
    (400, None),      # a name /view rejects — says nothing about the folder
    (403, None),
    (405, None),      # a proxy or a build with no HEAD on /view
    (500, None),
    (302, None),      # a login page in front of ComfyUI
])
def test_the_probe_only_says_no_when_comfyui_says_404(app, monkeypatch, status, verdict):
    """A generation may be refused ONLY on a definite answer. Everything else is
    "could not ask", because the alternative — reading a proxy's 403 as a missing
    file — would break installs that work today."""
    monkeypatch.setattr(comfy_fs.requests, 'head', lambda *a, **k: _Answer(status))
    with app.app_context():
        assert _REAL_SEES_INPUT('krea_source_abcd1234_src.png') is verdict


def test_the_probe_carries_on_when_comfyui_cannot_be_reached(app, monkeypatch):
    """ComfyUI down, DNS gone, TLS refused: staging must behave exactly as it did
    before this check existed."""
    def _boom(*a, **k):
        raise comfy_fs.requests.exceptions.ConnectionError('refused')

    monkeypatch.setattr(comfy_fs.requests, 'head', _boom)
    with app.app_context():
        assert _REAL_SEES_INPUT('krea_source_abcd1234_src.png') is None


def test_the_probe_asks_about_the_input_folder_by_basename(app, monkeypatch):
    """What makes the answer MEAN anything: `/view` resolves `type=input` through
    the same `get_input_directory()` LoadImage validates against, and the graph
    references a bare basename. Sending the full path, or leaving the type off
    (it defaults to `output`), would ask a different question and get a confident
    wrong answer."""
    seen = {}

    def _head(url, **kwargs):
        seen['url'] = url
        seen['params'] = kwargs.get('params')
        return _Answer(200)

    monkeypatch.setattr(comfy_fs.requests, 'head', _head)
    with app.app_context():
        _REAL_SEES_INPUT(os.path.join('C:', 'staged', 'krea_source_abcd1234_src.png'))
    assert seen['url'].endswith('/view')
    assert seen['params'] == {'filename': 'krea_source_abcd1234_src.png',
                              'type': 'input'}


def test_staging_refuses_when_comfyui_cannot_see_what_was_written(app, tmp_path,
                                                                  monkeypatch):
    """THE #64 repro, reduced: the write succeeds and ComfyUI still cannot see the
    file. Before, the job was queued and died on a 400 nobody was shown."""
    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    Image.new('RGB', (16, 12), (200, 30, 30)).save(src, 'PNG')
    monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: False)
    with app.app_context():
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_image(str(src), 'krea_source_abcd1234_src.png',
                                       str(folder))
    msg = str(exc.value)
    assert 'ComfyUI cannot see' in msg
    assert comfy_fs.safe_path(folder) in msg     # the folder the app actually used
    assert 'reads its input folder somewhere else' in msg
    assert 'shared volume' in msg                # the container/remote case, named
    assert 'Settings' in msg                     # and where to fix it
    # Nothing is left in a folder that belongs to ComfyUI: the job will not run.
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize('verdict', [True, None])
def test_staging_proceeds_unless_comfyui_says_no(app, tmp_path, monkeypatch, verdict):
    """Non-regression, the half that matters most. `True` is a healthy install;
    `None` is every install where the question could not be put, and both must
    stage exactly as before."""
    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    Image.new('RGB', (16, 12), (30, 120, 220)).save(src, 'PNG')
    monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: verdict)
    with app.app_context():
        staged = comfy_fs.stage_input_image(str(src), 'krea_source_abcd1234_src.png',
                                            str(folder))
    assert os.path.isfile(staged)
    assert [p.name for p in folder.iterdir()] == ['krea_source_abcd1234_src.png']


def test_the_written_lane_is_guarded_too(app, tmp_path, monkeypatch):
    """The watermark lane GENERATES its crop instead of copying one, and reaches
    ComfyUI through the same LoadImage. A fix covering only `stage_input_image`
    would leave the identical silent death one lane away."""
    folder = tmp_path / 'input'
    folder.mkdir()
    monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: False)
    with app.app_context():
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_write('wmklein_crop_abcd1234.png',
                                       lambda p: open(p, 'wb').write(b'x'),
                                       str(folder))
    assert 'ComfyUI cannot see' in str(exc.value)
    assert list(folder.iterdir()) == []


# --- naming the OTHER side: what ComfyUI itself reports -----------------------

def test_the_note_quotes_the_input_directory_flag(app, monkeypatch):
    """ComfyUI takes its folders on the command line and records them nowhere —
    but it echoes its own argv in /system_stats. Quoting the flag turns "somewhere
    else" into an address."""
    argv = ['/opt/ComfyUI/main.py', '--listen', '--input-directory', '/mnt/shared/input']
    monkeypatch.setattr(comfy_fs.requests, 'get',
                        lambda *a, **k: _Answer(200, {'system': {'argv': argv}}))
    with app.app_context():
        note = _REAL_FOLDER_NOTE()
    # VERBATIM: this path was written by another machine. Normalising it would
    # print `\mnt\shared\input` to a Windows reader and stop being a quote.
    assert note == 'it was started with `--input-directory /mnt/shared/input`'


def test_a_posix_path_is_absolute_even_when_this_machine_is_windows(app, monkeypatch):
    """The container case, which is the one this whole module exists for: LDS on
    Windows, ComfyUI in WSL or Docker reporting `/workspace/ComfyUI/input`.

    `os.path.isabs` answers False for that on Windows — a leading slash is
    drive-relative there, and Python 3.13 made the rule explicit — so judging a
    remote install's path by the local OS's rules drops exactly the case that
    needs naming. RED with `os.path.isabs`, on Windows only."""
    argv = ['/workspace/ComfyUI/main.py', '--input-directory', '/workspace/shared/input']
    monkeypatch.setattr(comfy_fs.requests, 'get',
                        lambda *a, **k: _Answer(200, {'system': {'argv': argv}}))
    with app.app_context():
        assert _REAL_FOLDER_NOTE() == ('it was started with `--input-directory '
                                       '/workspace/shared/input`')
    # ...and the Windows spelling keeps working on a POSIX reader, same reason.
    argv = ['C:\\Comfy\\main.py', '--input-directory', 'D:\\shared\\input']
    monkeypatch.setattr(comfy_fs.requests, 'get',
                        lambda *a, **k: _Answer(200, {'system': {'argv': argv}}))
    with app.app_context():
        assert _REAL_FOLDER_NOTE() == ('it was started with `--input-directory '
                                       'D:\\shared\\input`')


def test_the_note_quotes_the_base_directory_flag(app, monkeypatch):
    """`--base-directory` moves input/ too, and is quoted as the flag it is: the
    app does not invent `<that>/input`, which would be a guess about someone
    else's layout."""
    argv = ['/opt/ComfyUI/main.py', '--base-directory=/mnt/comfy-data']
    monkeypatch.setattr(comfy_fs.requests, 'get',
                        lambda *a, **k: _Answer(200, {'system': {'argv': argv}}))
    with app.app_context():
        note = _REAL_FOLDER_NOTE()
    assert note == 'it was started with `--base-directory /mnt/comfy-data`'


def test_the_note_falls_back_to_the_script_that_is_running(app, monkeypatch):
    """No flags at all: then the cause is usually a SECOND install, and argv[0] is
    the one address that identifies it."""
    argv = ['/opt/OtherComfy/main.py', '--listen']
    monkeypatch.setattr(comfy_fs.requests, 'get',
                        lambda *a, **k: _Answer(200, {'system': {'argv': argv}}))
    with app.app_context():
        assert _REAL_FOLDER_NOTE() == ('the ComfyUI answering there runs from '
                                       '/opt/OtherComfy')


def test_the_note_stays_silent_when_comfyui_says_nothing_useful(app, monkeypatch):
    """A relative path resolves against a working directory this process does not
    know, and a build too old to echo argv says nothing at all. Silence beats a
    confident wrong address."""
    for payload in ({'system': {'argv': ['main.py', '--input-directory', 'input']}},
                    {'system': {}}, {}, None):
        monkeypatch.setattr(comfy_fs.requests, 'get',
                            lambda *a, **k: _Answer(200, payload))
        with app.app_context():
            assert _REAL_FOLDER_NOTE() == ''


def test_the_refusal_carries_the_note(app, tmp_path, monkeypatch):
    """The two halves joined: the folder the app used AND where ComfyUI says it
    looks. Naming only one leaves the reader with the half they already had."""
    folder = tmp_path / 'input'
    folder.mkdir()
    monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: False)
    monkeypatch.setattr(comfy_fs, '_comfy_folder_note',
                        lambda *a, **k: 'it was started with `--input-directory /mnt/x`')
    with app.app_context():
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_write('wmklein_crop_abcd1234.png',
                                       lambda p: open(p, 'wb').write(b'x'), str(folder))
    assert '--input-directory /mnt/x' in str(exc.value)


def test_the_new_message_is_paste_safe(app, monkeypatch):
    r"""Same rule as every other message here: it is written to be dropped in
    #help, so a `C:\Users\<account>\` must never survive it."""
    monkeypatch.setattr(comfy_fs, '_comfy_folder_note', lambda *a, **k: '')
    with app.app_context():
        msg = comfy_fs._mismatch_message('a file', r'C:\Users\somebody\ComfyUI\input')
    assert 'somebody' not in msg and 'ComfyUI' in msg


# --- the same question at CONFIGURATION time ---------------------------------

def test_the_folder_probe_leaves_nothing_behind(app, tmp_path, monkeypatch):
    """It writes a file into a folder that belongs to ComfyUI. It removes it —
    on the "yes" path and on the "no" path alike."""
    folder = tmp_path / 'input'
    folder.mkdir()
    for verdict in (True, False, None):
        monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: verdict)
        with app.app_context():
            comfy_fs.input_visibility_problem(str(folder))
        assert list(folder.iterdir()) == [], f'probe survived the {verdict} path'


def test_the_folder_probe_is_quiet_unless_comfyui_proves_the_mismatch(app, tmp_path,
                                                                     monkeypatch):
    folder = tmp_path / 'input'
    folder.mkdir()
    for verdict in (True, None):
        monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: verdict)
        with app.app_context():
            assert comfy_fs.input_visibility_problem(str(folder)) == ''


def test_settings_preview_flags_a_folder_comfyui_does_not_read(app, tmp_path,
                                                               monkeypatch):
    """The Settings preview already reports "present but not writable". The folder
    that is present, writable and simply not ComfyUI's was the one case it called
    green — and it is the one that costs a generation."""
    from app import capabilities
    base = tmp_path / 'Comfy'
    (base / 'input').mkdir(parents=True)
    monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: False)
    with app.app_context():
        r = capabilities.classify_comfyui_folders(str(base), {})
    assert r['input_dir']['exists'] is True          # both old checks: green
    assert r['input_dir']['usable'] is False         # the new half: honest
    assert 'ComfyUI cannot see' in r['input_dir']['problem']
    # Only input/ is asked about — it is the one folder both sides must agree on.
    assert r['output_dir']['problem'] == ''


def test_setup_wizard_flags_a_folder_comfyui_does_not_read(app, tmp_path, monkeypatch):
    """Same verdict one screen earlier, and still not blocking: someone may
    configure the app before starting ComfyUI."""
    from app import capabilities
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    (comfy / 'input').mkdir()
    monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: False)
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['status'] == 'valid'                    # NOT downgraded, not blocking
    assert r['input_check']['ok'] is False
    assert 'ComfyUI cannot see' in r['input_check']['problem']


# --- end to end: the engine from the report ----------------------------------

def test_krea_generation_refuses_and_queues_nothing(app, tmp_path, monkeypatch):
    """The reported shape, through the engine that reported it: Krea 2, a folder
    that passes every local check, and a ComfyUI that does not read it. The user
    gets a sentence; the queue gets nothing it could only fail."""
    from app.services import krea_edit_helper as keh2
    comfy_in = tmp_path / 'input'
    comfy_in.mkdir()
    src = tmp_path / 'src.png'
    Image.new('RGB', (32, 24), (10, 10, 10)).save(src, 'PNG')
    queued = []
    with app.app_context():
        monkeypatch.setattr(keh2, '_comfy_input_dir', lambda: str(comfy_in))
        monkeypatch.setattr(keh2, 'preflight', lambda *a, **k: None)
        monkeypatch.setattr(keh2, 'resolve_krea_unet', lambda *a, **k: 'unet.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_text_encoder', lambda *a, **k: 'te.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_vae', lambda *a, **k: 'vae.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_identity_lora',
                            lambda *a, **k: ('lora.safetensors', 'x'))
        monkeypatch.setattr(keh2.queue_manager, 'add_job', lambda **kw: queued.append(kw))
        monkeypatch.setattr(comfy_fs, 'comfyui_sees_input', lambda *a, **k: False)
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            keh2.enqueue_krea_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='hi')
    assert 'ComfyUI cannot see' in str(exc.value)
    assert queued == []
    assert list(comfy_in.iterdir()) == []
