"""🗣 Captioning shots — so a search can find what HAPPENS, not only what is visible.

The CLIP pass (video_clip_search.py) embeds frames, so it finds what a moment
LOOKS like. It cannot find an action: "a woman turns and walks away" is a fact
about time, and no single frame carries it. Captions are the other half of the
question the user actually asked, and they are also what the trainer reads — a
promoted clip's `.txt` sidecar is its prompt.

WHY NOT OLLAMA, pinned here because it is the expensive mistake available: Ollama
fails SILENTLY on video on this machine — an empty response, no error, the fence
swallowing everything (grep `text generate skipped` in its logs). A captioner
that returns "" for every clip and reports success would fill a bank with empty
sidecars, which is exactly the failure ai-toolkit turns into an untrained prompt
with no message. So this lane runs transformers directly.

WHAT WAS VERIFIED ON THIS MACHINE BEFORE A LINE WAS WRITTEN (the video lane's
absolute rule — no unverified model claim):
  * `Qwen/Qwen3-VL-4B-Instruct` is already in the HF cache
    (A:\\...\\hub\\models--Qwen--Qwen3-VL-4B-Instruct, 8.3 GB, both safetensors
    shards present);
  * the ai-toolkit venv is python 3.12.9 / transformers 5.5.3 / torch 2.9.1+cu128
    with CUDA, and exposes `Qwen3VLForConditionalGeneration` NATIVELY — no
    trust_remote_code, no pinning;
  * `Qwen3VLProcessor` accepts `videos=` directly and carries a chat template, so
    `qwen_vl_utils` (which is NOT installed) is not needed: the parent samples
    the frames itself with PyAV, which it already does for the embedding pass.

Both heavy seams — the frame decode and the model — are monkeypatched here, so
the suite runs with neither PyAV nor torch.
"""

from app.services import video_caption as vc


# --- which frames the captioner is shown -----------------------------------------

def test_a_shot_is_sampled_across_its_whole_length():
    """An action happens over TIME. Sampling the middle three frames of a
    ten-second shot describes a moment, not the movement — and "turns and walks
    away" is precisely what would be lost."""
    times = vc.caption_frame_times(0.0, 10.0)

    assert len(times) == vc.CAPTION_FRAMES
    assert times == sorted(times)
    assert times[-1] - times[0] > 8.0          # spans nearly the whole shot


def test_the_sampled_frames_stay_inside_the_shot_and_off_its_edges():
    """Same reason the embedding pass insets: a shot boundary is where a cut just
    happened, and a dissolve frame captioned as content is a caption about the
    edit rather than about the scene."""
    times = vc.caption_frame_times(4.0, 14.0)

    assert all(4.0 < t < 14.0 for t in times)


def test_a_very_short_shot_is_not_sampled_more_often_than_it_has_moments():
    """Asking for eight frames of a half-second shot hands the model eight copies
    of one picture and pays for all of them."""
    times = vc.caption_frame_times(0.0, 0.3)

    assert 1 <= len(times) < vc.CAPTION_FRAMES
    assert len(times) == len(set(times))


# --- the prompt ------------------------------------------------------------------

def test_the_prompt_asks_for_the_action_not_an_inventory_of_objects():
    """A video dataset trains on movement. "A woman in a red coat. A street. A
    car." is a caption for a photograph, and it teaches the model nothing about
    what should happen between the first frame and the last."""
    prompt = vc.caption_prompt()

    lowered = prompt.lower()
    assert 'action' in lowered or 'happen' in lowered or 'move' in lowered
    assert 'camera' in lowered          # camera motion is part of what is learned


def test_the_prompt_forbids_the_preamble_that_would_poison_every_sidecar():
    """The sidecar IS the training prompt. Every caption starting "This video
    shows" teaches the model that phrase, and stripping it afterwards is a
    find-and-replace nobody remembers to run."""
    prompt = vc.caption_prompt().lower()

    assert 'this video' in prompt or 'do not begin' in prompt or 'no preamble' in prompt


def test_a_returned_caption_is_cleaned_of_the_preamble_anyway():
    """Belt and braces: the instruction is not a guarantee, and one leaked
    preamble in a thousand clips is a poisoned sidecar nobody will read."""
    assert vc.clean_caption('This video shows a woman walking away.') \
        == 'A woman walking away.'
    assert vc.clean_caption('The video depicts a car turning left.') \
        == 'A car turning left.'
    assert vc.clean_caption('  A woman turns and walks away.\n\n') \
        == 'A woman turns and walks away.'


def test_a_caption_that_came_back_empty_is_an_error_not_a_caption():
    """The Ollama failure mode, defended against structurally: an empty string
    that is stored as a caption becomes an empty sidecar, which ai-toolkit trains
    as an empty prompt without a word."""
    assert vc.clean_caption('') == ''
    assert vc.clean_caption('   \n ') == ''


# --- the pass --------------------------------------------------------------------

def test_every_clip_of_the_bank_gets_a_caption(app, monkeypatch):
    bank_id = _bank_with_clips(app, 3)
    _fake_seams(monkeypatch)

    with app.app_context():
        out = vc.run_captions(bank_id)

    assert out['captioned'] == 3
    assert _captions(app, bank_id) == ['A woman turns and walks away.'] * 3
    assert _states(app, bank_id) == ['ok', 'ok', 'ok']


def test_a_rerun_pays_only_for_what_is_not_captioned_yet(app, monkeypatch):
    """The resume contract this lane shares, and the one that makes Stop safe."""
    bank_id = _bank_with_clips(app, 3)
    seen = _fake_seams(monkeypatch)

    with app.app_context():
        vc.run_captions(bank_id)
        first = len(seen)
        vc.run_captions(bank_id)

    assert first == 3 and len(seen) == 3


def test_a_stop_keeps_what_is_already_captioned(app, monkeypatch):
    """Graceful cancel, the same contract as the image lane's caption batch: a
    stopped pass loses nothing and the next run starts where it stopped."""
    bank_id = _bank_with_clips(app, 4)
    seen = _fake_seams(monkeypatch)
    stop_after = {'n': 2}

    with app.app_context():
        out = vc.run_captions(bank_id,
                              should_stop=lambda: len(seen) >= stop_after['n'])

    assert out['captioned'] == 2
    assert sum(1 for c in _captions(app, bank_id) if c) == 2
    with app.app_context():
        again = vc.run_captions(bank_id)
    assert again['captioned'] == 2          # the remaining two, not all four


def test_one_clip_the_model_refuses_costs_that_clip(app, monkeypatch):
    bank_id = _bank_with_clips(app, 3)
    calls = {'n': 0}

    def flaky(paths, prompt, **kw):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('the model produced nothing')
        return 'A woman turns and walks away.'
    monkeypatch.setattr(vc, '_caption_frames', flaky)
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/{stem}_{i}.jpg'
                                                        for i, _ in enumerate(times)])

    with app.app_context():
        out = vc.run_captions(bank_id)

    assert out['captioned'] == 2 and out['failed'] == 1
    assert sorted(_states(app, bank_id)) == ['error', 'ok', 'ok']


def test_a_caption_edited_by_hand_is_never_overwritten_by_a_rerun(app, monkeypatch):
    """A generated caption is a draft. Losing a correction to the next pass is
    the one thing that would stop anyone from ever editing one."""
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)
    with app.app_context():
        vc.run_captions(bank_id)
        cid = _clip_ids(app, bank_id)[0]
        vc.set_caption('local', bank_id, cid, 'A dog crosses the road.')
        vc.run_captions(bank_id, recaption=True)

    assert _captions(app, bank_id) == ['A dog crosses the road.']
    assert _states(app, bank_id) == ['edited']


def test_an_edited_caption_can_still_be_regenerated_on_purpose(app, monkeypatch):
    """"Never silently" is not "never". The explicit gesture exists, it is just
    not what a bulk re-run does."""
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)
    with app.app_context():
        vc.run_captions(bank_id)
        cid = _clip_ids(app, bank_id)[0]
        vc.set_caption('local', bank_id, cid, 'Mine.')
        vc.run_captions(bank_id, recaption=True, include_edited=True)

    assert _captions(app, bank_id) == ['A woman turns and walks away.']


def test_clearing_a_caption_puts_the_clip_back_in_the_queue(app, monkeypatch):
    bank_id = _bank_with_clips(app, 1)
    _fake_seams(monkeypatch)
    with app.app_context():
        vc.run_captions(bank_id)
        cid = _clip_ids(app, bank_id)[0]
        vc.set_caption('local', bank_id, cid, '')
        pending = vc.pending_clips(bank_id, False).count()

    assert pending == 1
    assert _captions(app, bank_id) == [None]


# --- the model contract ------------------------------------------------------------

def _worker_code():
    """The worker's CODE, with its prose removed.

    Stripped on purpose: the module documents at length why it avoids
    `trust_remote_code` and `qwen_vl_utils`, so a naive substring search over the
    file finds the very names the tests are asserting are absent — and would fail
    on a file that is right for exactly the reason it explains."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'infer'
           / 'video_caption_infer.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    # Drop every docstring, then unparse: comments are already gone (ast does not
    # keep them), which leaves executable code only.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_the_worker_names_a_model_this_machine_actually_has():
    """The video lane's absolute rule: no unverified model claim. This pins the
    id that was checked in the local HF cache, so a future "upgrade" to a model
    nobody has here fails a test instead of a user's first run."""
    code = _worker_code()

    assert "MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'" in code
    # Native class, not trust_remote_code — the Florence-2 lesson recorded in
    # watermark_detect_infer.py: remote modelling code written against an old
    # transformers is what forces a whole environment to be pinned back.
    assert 'trust_remote_code' not in code
    assert 'Qwen3VLForConditionalGeneration' in code


def test_the_worker_never_asks_for_a_package_this_venv_does_not_have():
    """`qwen_vl_utils` is the helper every Qwen-VL example imports and it is NOT
    installed here — verified. The processor takes `videos=` on its own, so the
    parent samples frames with PyAV (which it already does) and the worker stays
    dependency-free."""
    assert 'qwen_vl_utils' not in _worker_code()


# --- helpers ------------------------------------------------------------------------

def _fake_seams(monkeypatch):
    seen = []

    def write(src, times, dest, stem):
        seen.append(stem)
        return [f'{dest}/{stem}_{i}.jpg' for i, _ in enumerate(times)]
    monkeypatch.setattr(vc, '_write_caption_frames', write)
    monkeypatch.setattr(vc, '_caption_frames',
                        lambda paths, prompt, **kw: 'A woman turns and walks away.')
    return seen


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


def _captions(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.caption for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]


def _states(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        return [c.caption_state for c in VideoClip.query.filter_by(bank_id=bank_id)
                .order_by(VideoClip.id).all()]


def test_the_model_is_told_how_long_the_shot_really_is():
    """The frames are pre-sampled, so without a span the processor stamps them
    at a default 24 fps and the model reads a five-second action as a third of
    a second — every judgement of speed and duration wrong at the source. The
    span the sampled instants actually cover must ride the seam to the worker."""
    from app.services import video_caption as vc

    seen = {}

    class _Worker:
        def caption(self, paths, prompt, span_s=None):
            seen['span_s'] = span_s
            return 'a caption'

    times = vc.caption_frame_times(10.0, 15.0)
    assert vc._caption_frames(['a.jpg', 'b.jpg'], 'p', worker=_Worker(),
                              span_s=times[-1] - times[0]) == 'a caption'
    # The span is the sampled instants' cover, inside the shot (edge margins
    # shave it), never the raw clip length and never a made-up number.
    assert 0 < seen['span_s'] <= 5.0
    assert seen['span_s'] == times[-1] - times[0]


# --- the C12 calibration (2026-08-30): frames, length, token budget ------------------

def test_the_frame_budget_is_the_survey_sixteen():
    """Sixteen frames, because the payoff of a longer caption is a payoff of
    MOTION and motion is read between frames — eight was the ceiling on what
    any prompt could extract, whatever it asked for."""
    assert vc.CAPTION_FRAMES == 16


def test_both_prompts_ask_for_the_measured_length():
    """150-200 words is where the primary sources converge (MiraData's ablation,
    the vendors' own caption specs). "One or two sentences" was the folklore
    number, and it starved exactly the dimension a video LoRA learns."""
    for style in vc.CAPTION_STYLES:
        prompt = vc.caption_prompt(style)
        assert '150' in prompt and '200' in prompt
        assert 'one or two plain sentences' not in prompt.lower()


def test_both_prompts_refuse_camera_and_sound():
    """Refuted twice for the camera (no VLM reads it reliably — our homography
    classifier writes that line at export) and never true for sound (the model
    sees silent JPEGs; any audio it wrote would be an invention)."""
    for style in vc.CAPTION_STYLES:
        lowered = vc.caption_prompt(style).lower()
        assert 'do not describe the camera' in lowered
        assert 'sound' in lowered


def test_the_token_budget_matches_the_asked_for_length():
    """96 tokens truncated a 150-200-word paragraph mid-sentence; 400 leaves
    room. Pinned in BOTH processes — the worker default and the infer fallback —
    because they disagree in silence."""
    from pathlib import Path
    worker = (Path(__file__).resolve().parents[1] / 'app' / 'services'
              / 'video_caption_worker.py').read_text(encoding='utf-8')
    infer = (Path(__file__).resolve().parents[1] / 'infer'
             / 'video_caption_infer.py').read_text(encoding='utf-8')
    assert 'max_new_tokens=400' in worker
    assert "req.get('max_new_tokens') or 400" in infer
    assert 'max_new_tokens=96' not in worker
