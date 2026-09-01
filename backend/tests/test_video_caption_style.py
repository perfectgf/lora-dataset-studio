"""WHICH prompt writes the captions — the lock that turned out to dominate.

The model was made configurable first, on the theory that the checkpoint decided
how plainly a caption described its footage. An A/B on real material said
otherwise, and that measurement is why this file exists:

    base model     + the standard prompt            -> evasive, circled the subject
    uncensored 8B  + the standard prompt            -> barely better, hid behind
                                                       the camera
    uncensored 8B  + an explicit-permission prompt  -> named things precisely
    BASE model     + that same prompt               -> named things precisely too,
                                                       with the best action writing
                                                       of the four

So the dominant lock is the PROMPT, not the checkpoint. A captioner describes
what it has been given permission to describe, and the shipped prompt never
granted it. That makes the prompt a setting for the same reason the model became
one: a caption that talks around what it shows teaches the trained model to look
away, and the output reads perfectly well either way.

Two styles, and the boundary between them is a promise:
  * `standard` — unchanged, and the DEFAULT. An install that sets nothing
    captions exactly as it did before.
  * `plain` — grants explicit permission to name what is shown, and forbids the
    two words the measurement caught it hiding behind.

Every caption records the style that produced it, exactly as it records the
model: two styles do not write comparable captions either, and mixing them across
one bank is precisely what making this configurable invites.
"""

from app.services import video_caption as vc

# The video-extra gate answers for the MACHINE, so without this these route
# tests pass where PyAV/ffmpeg are installed and 503 where they are not.
# Imported for its autouse effect; see _video_extra.py for why not importorskip.
from _video_extra import video_extra_ready  # noqa: F401


# --- the styles themselves ---------------------------------------------------------

def test_the_default_style_is_the_prompt_that_already_shipped(app):
    with app.app_context():
        assert vc.configured_style() == 'standard'
        assert vc.caption_prompt('standard') == vc.caption_prompt()


def test_both_styles_still_ask_for_the_action_first():
    """The whole point of captioning a VIDEO. A style that traded the action for
    an inventory would be a regression dressed as an option."""
    for style in vc.CAPTION_STYLES:
        prompt = vc.caption_prompt(style).lower()
        assert 'action' in prompt
        assert 'camera' in prompt


def test_both_styles_forbid_the_preamble():
    for style in vc.CAPTION_STYLES:
        assert 'preamble' in vc.caption_prompt(style).lower()


def test_the_plain_style_grants_permission_and_names_the_hiding_words():
    """The two demands that carried the measured difference: name what is there,
    and do not substitute the words it was caught hiding behind. Pinned because a
    later tidy-up of the wording would quietly restore the euphemism."""
    prompt = vc.caption_prompt('plain').lower()

    assert 'never euphemize' in prompt
    assert 'intimate' in prompt and 'sensual' in prompt
    assert 'body parts' in prompt
    # The third measured failure (2026-08-07 bench): asked politely, a model
    # does not soften the scene — it swaps it for an invented harmless one.
    assert 'never sanitize' in prompt
    assert 'replace the scene' in prompt
    assert 'nsfw' in prompt and 'authorized' in prompt


def test_the_standard_style_says_none_of_that():
    """It is the shipped default and its meaning must not drift — a user who
    never opts in gets the captions they already had."""
    prompt = vc.caption_prompt('standard').lower()

    assert 'euphemize' not in prompt
    assert 'body parts' not in prompt
    assert 'nsfw' not in prompt
    # It DOES carry the content-neutral half of the same lesson: the scene that
    # is shown, never a substitute — hallucinated subjects are a dataset defect
    # on any footage (measured: a woman invented in place of a man, 2026-09-01).
    assert 'never a substitute' in prompt


def test_an_unknown_style_falls_back_to_the_default_rather_than_failing(app):
    """A typo in a config file must not take the pass down — and must not grant a
    permission nobody asked for, so it lands on `standard` and not on `plain`."""
    with app.app_context():
        assert vc.caption_prompt('nonsense') == vc.caption_prompt('standard')


def test_the_setting_selects_the_style(app):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'video_caption': {'style': 'plain'}})
        try:
            assert vc.configured_style() == 'plain'
        finally:
            cfg.save_config({'video_caption': {'style': ''}})


def test_a_blank_or_unknown_setting_is_the_default_style(app):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'video_caption': {'style': 'wat'}})
        try:
            assert vc.configured_style() == 'standard'
        finally:
            cfg.save_config({'video_caption': {'style': ''}})


# --- it reaches the model, and the caption remembers it ------------------------------

def test_the_chosen_style_is_the_prompt_actually_sent(app, monkeypatch):
    seen = {}
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/x.jpg'])

    def spy(paths, prompt, **kw):
        seen['prompt'] = prompt
        return 'A woman turns and walks away.'
    monkeypatch.setattr(vc, '_caption_frames', spy)
    bank_id = _bank_with_clips(app, 1)

    with app.app_context():
        vc.run_captions(bank_id, style='plain')

    assert seen['prompt'] == vc.caption_prompt('plain')


def test_each_caption_records_the_style_that_produced_it(app, monkeypatch):
    bank_id = _bank_with_clips(app, 2)
    _fake_seams(monkeypatch)

    with app.app_context():
        vc.run_captions(bank_id, style='plain')

    assert _styles(app, bank_id) == ['plain', 'plain']


def test_a_hand_written_caption_records_no_style(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)

    with app.app_context():
        vc.run_captions(bank_id, style='plain')
        cid = _clip_ids(app, bank_id)[0]
        vc.set_caption('local', bank_id, cid, 'Mine.')

    assert _styles(app, bank_id) == [None]


def test_the_pass_reports_the_style_it_used(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)

    with app.app_context():
        out = vc.run_captions(bank_id, style='plain')

    assert out['style'] == 'plain'


def test_the_route_takes_a_style_for_one_run_without_changing_the_setting(
        app, client, tmp_path, monkeypatch):
    """A per-run choice next to the button, with the config key as its default.
    Captioning one bank plainly must not silently re-point every other bank."""
    from app.services import video_bank_service as svc
    monkeypatch.setattr(svc, '_probe_file', lambda _p: {
        'duration_s': 60.0, 'fps_native': 30.0, 'width': 640, 'height': 480,
        'codec': 'h264', 'probe_state': 'ok', 'file_size': 4096})
    started = {}
    monkeypatch.setattr(svc, 'start_caption', lambda *a, **kw: started.update(kw))
    folder = tmp_path / 'rushes'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'a.mp4').write_bytes(b'0' * 32)
    bank_id = client.post('/api/video-bank/create',
                          json={'name': 'r', 'folder': str(folder)}).get_json()['id']

    r = client.post(f'/api/video-bank/{bank_id}/caption', json={'style': 'plain'})

    assert r.status_code == 202
    assert started.get('style') == 'plain'


def test_the_payload_offers_the_styles_and_says_which_one_is_current(
        app, client, tmp_path, monkeypatch):
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

    assert info['style'] == 'standard'
    assert [s['key'] for s in info['styles']] == ['standard', 'plain']
    assert all(s.get('label') for s in info['styles'])


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


def _styles(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.caption_style for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]
