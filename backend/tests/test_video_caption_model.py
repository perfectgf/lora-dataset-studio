"""WHICH model writes the captions — and why that had to become a setting.

The pass shipped with one checkpoint hard-coded. On a real corpus that turned out
to be a dataset defect rather than a matter of taste: the default model
EUPHEMISES. It describes plainly visible things in evasive terms and never names
them — 171 captions, zero refusals, and a vocabulary that circled the subject
throughout. A caption that talks around what it shows teaches the model being
trained to look away too, and nothing in the output reveals it. The captions read
fine. They are simply about something slightly other than the footage.

Three consequences, one test section each:

  * THE DEFAULT DOES NOT MOVE. An install that sets nothing captions exactly as
    it did yesterday. A silently changed captioner would silently change every
    dataset built afterwards, and the two would look identical on disk.
  * NOTHING IS DOWNLOADED IN SILENCE. Pointing the setting at a model this
    machine does not have means gigabytes over someone's connection. The pass
    says so BEFORE it starts, in the job line already on screen — the same
    philosophy as clip_text_encoder.weights_warning(), which exists so a first
    search cannot stall for ten minutes with nothing said.
  * EVERY CAPTION REMEMBERS WHO WROTE IT. Two checkpoints do not produce
    comparable captions, and changing the setting mid-corpus is precisely what
    this feature invites. A bank captioned half by one and half by another is
    unreadable unless each row says which.

No model is ever loaded here: the worker is a seam and is monkeypatched.
"""

from app.services import video_caption as vc

DEFAULT_MODEL = 'Qwen/Qwen3-VL-4B-Instruct'


# --- the setting ------------------------------------------------------------------

def test_the_default_is_the_checkpoint_that_already_shipped(app):
    """No behaviour change without an opt-in."""
    with app.app_context():
        assert vc.configured_model() == DEFAULT_MODEL


def test_the_setting_overrides_it(app):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'video_caption': {'model': 'someone/other-vlm'}})
        try:
            assert vc.configured_model() == 'someone/other-vlm'
        finally:
            cfg.save_config({'video_caption': {'model': ''}})


def test_a_blank_setting_falls_back_to_the_default(app):
    """A cleared field means "use the default", not "use the empty string" — the
    second would reach the worker and fail on a model id of nothing."""
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'video_caption': {'model': '   '}})
        try:
            assert vc.configured_model() == DEFAULT_MODEL
        finally:
            cfg.save_config({'video_caption': {'model': ''}})


# --- the download announcement ------------------------------------------------------

def test_a_model_already_on_this_machine_announces_nothing(app, monkeypatch):
    monkeypatch.setattr(vc, 'model_is_cached', lambda mid: True)

    with app.app_context():
        assert vc.download_notice(DEFAULT_MODEL) == ''


def test_a_model_this_machine_lacks_says_so_before_the_pass_starts(app, monkeypatch):
    """BEFORE, in the job line. A pass sitting at 0/470 for twenty minutes while
    gigabytes come down the wire is indistinguishable from a hang."""
    monkeypatch.setattr(vc, 'model_is_cached', lambda mid: False)

    with app.app_context():
        notice = vc.download_notice('someone/huge-vlm')

    assert 'someone/huge-vlm' in notice
    assert 'download' in notice.lower()


def test_the_notice_never_invents_a_size(app, monkeypatch):
    """We cannot know how big an arbitrary checkpoint is without asking the
    network, and a made-up figure is one the user would plan around."""
    monkeypatch.setattr(vc, 'model_is_cached', lambda mid: False)

    with app.app_context():
        notice = vc.download_notice('someone/huge-vlm')

    assert 'GB' not in notice


def test_a_cache_we_cannot_read_is_not_reported_as_missing(app, monkeypatch):
    """Best effort, like weights_warning(): crying wolf about a layout we could
    not inspect trains people to ignore the one warning that matters."""
    def boom():
        raise OSError('unreadable')
    monkeypatch.setattr(vc, '_hf_cache_dirs', boom)

    assert vc.model_is_cached('anything/at-all') is True


def test_the_cache_probe_finds_a_model_by_its_hub_folder(tmp_path, monkeypatch):
    """The HF hub layout: `models--<org>--<name>` holding a snapshot. Probed on
    disk rather than by importing huggingface_hub, which is not in this venv."""
    hub = tmp_path / 'hub'
    (hub / 'models--Qwen--Qwen3-VL-4B-Instruct' / 'snapshots' / 'abc').mkdir(parents=True)
    monkeypatch.setattr(vc, '_hf_cache_dirs', lambda: [str(hub)])

    assert vc.model_is_cached(DEFAULT_MODEL) is True
    assert vc.model_is_cached('Qwen/Not-Here') is False


def test_a_folder_with_no_snapshot_is_not_a_cached_model(tmp_path, monkeypatch):
    """An interrupted download leaves the directory behind. Calling that present
    would skip the warning for exactly the case that needs it."""
    hub = tmp_path / 'hub'
    (hub / 'models--Qwen--Half-Downloaded' / 'blobs').mkdir(parents=True)
    monkeypatch.setattr(vc, '_hf_cache_dirs', lambda: [str(hub)])

    assert vc.model_is_cached('Qwen/Half-Downloaded') is False


# --- the model reaches the worker, and the caption remembers it ----------------------

def test_the_configured_model_is_handed_to_the_worker(app, monkeypatch):
    """A setting the worker never receives is a setting that does nothing."""
    from app.services import video_caption_worker as vcw
    seen = {}

    class _Spy:
        def __init__(self, **kw):
            seen.update(kw)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def caption(self, paths, prompt):
            return 'A woman turns and walks away.'

    monkeypatch.setattr(vcw, 'CaptionWorker', _Spy)
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/x.jpg'])
    bank_id = _bank_with_clips(app, 1)

    with app.app_context():
        vc.run_captions(bank_id, model='someone/other-vlm')

    assert seen.get('model') == 'someone/other-vlm'


def test_each_caption_records_which_model_wrote_it(app, monkeypatch):
    """Changing the setting mid-corpus is exactly what this feature invites, so
    the bank has to stay readable across the change."""
    bank_id = _bank_with_clips(app, 2)
    _fake_seams(monkeypatch)

    with app.app_context():
        vc.run_captions(bank_id, model='someone/other-vlm')

    assert _models(app, bank_id) == ['someone/other-vlm'] * 2


def test_a_clip_that_failed_records_no_model(app, monkeypatch):
    """Nothing wrote it, so nothing is claimed. An id stored beside an empty
    caption would say a model produced that emptiness."""
    bank_id = _bank_with_clips(app, 1)
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/x.jpg'])
    monkeypatch.setattr(vc, '_caption_frames', lambda paths, prompt, **kw: '')

    with app.app_context():
        vc.run_captions(bank_id)

    assert _models(app, bank_id) == [None]


def test_a_caption_written_by_hand_is_credited_to_no_model(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)

    with app.app_context():
        vc.run_captions(bank_id)
        cid = _clip_ids(app, bank_id)[0]
        vc.set_caption('local', bank_id, cid, 'Mine.')

    assert _models(app, bank_id) == [None]


def test_the_pass_reports_which_model_it_used(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)

    with app.app_context():
        out = vc.run_captions(bank_id, model='someone/other-vlm')

    assert out['model'] == 'someone/other-vlm'


# --- helpers -------------------------------------------------------------------------

def _fake_seams(monkeypatch):
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/x.jpg'])
    monkeypatch.setattr(vc, '_caption_frames',
                        lambda paths, prompt, **kw: 'A woman turns and walks away.')


def _bank_with_clips(app, n):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=600.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        for i in range(n):
            db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                     start_s=float(i * 20), end_s=float(i * 20 + 10)))
        db.session.commit()
        return bank.id


def _clip_ids(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.id for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]


def _models(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.caption_model for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]


# --- what the workspace is told ------------------------------------------------------

def test_the_bank_payload_says_which_model_will_caption(app, client, tmp_path,
                                                        monkeypatch):
    """On the payload the workspace already polls, not behind a new route: one
    string that changes when a config file is hand-edited does not deserve a
    request per bank per two seconds."""
    from app.services import video_bank_service as svc
    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 60.0, 'fps_native': 30.0, 'width': 640, 'height': 480,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096})
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'a.mp4').write_bytes(b'0' * 32)
    bank_id = client.post('/api/video-bank/create',
                          json={'name': 'r', 'folder': str(folder)}).get_json()['id']

    info = client.get(f'/api/video-bank/{bank_id}').get_json()['caption_model']

    assert info['model'] == DEFAULT_MODEL
    assert info['is_default'] is True
    assert 'cached' in info


def test_the_infer_script_takes_the_model_from_the_handshake(app):
    """The setting has to reach the process that loads the weights. Asserted on
    the CODE with its prose stripped — the docstring necessarily discusses the
    default it is being checked against."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'infer'
           / 'video_caption_infer.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)

    # The default survives for a bare invocation…
    assert f"MODEL_ID = '{DEFAULT_MODEL}'" in code
    # …but the handshake wins, and the weights are loaded from THAT.
    assert "req.get('model')" in code
    assert 'from_pretrained(model_id' in code
    assert 'from_pretrained(MODEL_ID' not in code
