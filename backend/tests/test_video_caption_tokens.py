"""C12-C (2026-09-01): the caption's structured tail, its REAL token count, and
the short form served where the encoder would otherwise cut.

Three facts this pins, each measured before it was written:
- umT5 (every Wan 2.x text encoder) truncates past 512 tokens in silence, and
  the shipped prompt costs 1.35-1.36 tokens per word (48 captions, three arms).
- transformers 5.3 cannot rebuild umT5's tokenizer from spiece.model, so the
  child counts with sentencepiece first.
- The paragraph is what trains; the labelled tail is what a budgeted target is
  served INSTEAD of a paragraph cut mid-sentence.
"""
import json
from pathlib import Path

from app.services import video_caption as vc
from app.services import video_caption_worker as vcw

INFER = Path(__file__).resolve().parents[1] / 'infer' / 'video_caption_infer.py'
WORKER = Path(__file__).resolve().parents[1] / 'app' / 'services' / 'video_caption_worker.py'

PARA = 'A woman in a red dress walks to the window and turns into the light.'
TAILED = (PARA + '\n---\nSubject: a woman in a red dress\nMotion: walks to the window and turns\n'
          'Setting: a bright apartment\nStyle: soft daylight\nShort: a woman walks to a bright window')
# Complete AND substantial: the serving floor requires all four served fields
# and at least twenty words between them (a tail the generation cap cut would
# fail one or the other).
FIELDS = json.dumps({'subject': 'a young woman in a long red dress',
                     'motion': 'walks slowly toward the tall window and turns',
                     'setting': 'a bright, sparsely furnished apartment',
                     'style': 'soft morning daylight, warm palette',
                     'short': 'a woman walks to a bright window'})


# --- the prompt asks for the tail, after the paragraph -------------------------------

def test_both_prompts_ask_for_the_labelled_tail_after_the_paragraph():
    for style in ('standard', 'plain'):
        p = vc.caption_prompt(style)
        assert '---' in p and 'Subject:' in p and 'Short:' in p, style
        # The tail follows the paragraph instruction: the paragraph stays the caption.
        assert p.index('paragraph') < p.index('Subject:'), style
        # Every field named once, in reading order.
        for a, b in zip(('Subject:', 'Motion:', 'Setting:', 'Style:'),
                        ('Motion:', 'Setting:', 'Style:', 'Short:')):
            assert p.index(a) < p.index(b), style


# --- the pass stores prose, fields and the measured count ----------------------------

def _bank_with_one_clip(app):
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
        db.session.add(VideoClip(bank_id=bank.id, source_id=src.id, start_s=0.0, end_s=10.0))
        db.session.commit()
        return bank.id


class _FakeWorker:
    """What run_captions builds — here never started, and handing back a count."""
    built = []

    def __init__(self, **kw):
        self.kw = kw
        self.loaded_model = kw.get('model')
        self.last_tokens = 187
        _FakeWorker.built.append(kw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_run_captions_stores_the_prose_the_fields_and_the_measured_tokens(app, monkeypatch):
    from app.models import VideoClip
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/{stem}_{i}.jpg'
                                                        for i, _ in enumerate(times)])
    monkeypatch.setattr(vc, '_caption_frames', lambda paths, prompt, **kw: TAILED)
    # run_captions imports the class from its OWN module at call time.
    monkeypatch.setattr(vcw, 'CaptionWorker', _FakeWorker)
    monkeypatch.setattr(vc, 'umt5_tokenizer_dir', lambda: '/models/umt5/tokenizer')
    _FakeWorker.built.clear()
    bank_id = _bank_with_one_clip(app)
    with app.app_context():
        vc.run_captions(bank_id)
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        # The paragraph is the caption — the tail never reaches the sidecar as prose.
        assert clip.caption == PARA
        fields = json.loads(clip.caption_fields)
        assert fields['motion'] == 'walks to the window and turns'
        assert fields['short'] == 'a woman walks to a bright window'
        # The count came from the worker, in the encoder's own tokens.
        assert clip.caption_tokens == 187
    # The tokenizer the parent found was handed to the worker at start.
    assert _FakeWorker.built and _FakeWorker.built[0]['tokenizer_dir'] == '/models/umt5/tokenizer'


def test_a_caption_without_a_tail_stores_no_fields_and_no_invented_count(app, monkeypatch):
    from app.models import VideoClip

    class _Bare(_FakeWorker):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.last_tokens = None       # no tokenizer on this machine
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/{stem}_0.jpg'])
    monkeypatch.setattr(vc, '_caption_frames', lambda paths, prompt, **kw: PARA)
    monkeypatch.setattr(vcw, 'CaptionWorker', _Bare)
    monkeypatch.setattr(vc, 'umt5_tokenizer_dir', lambda: None)
    bank_id = _bank_with_one_clip(app)
    with app.app_context():
        vc.run_captions(bank_id)
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        assert clip.caption == PARA
        assert clip.caption_fields is None
        assert clip.caption_tokens is None


# --- finding the tokenizer, without ever downloading it ------------------------------

def test_tokenizer_discovery_reads_the_declared_caches_and_returns_none_when_absent(
        app, tmp_path, monkeypatch):
    hub = tmp_path / 'hub'
    monkeypatch.setattr(vc, '_hf_cache_dirs', lambda: [str(hub)])
    with app.app_context():
        assert vc.umt5_tokenizer_dir() is None
        snap = hub / 'models--ai-toolkit--umt5_xxl_encoder' / 'snapshots' / 'abc' / 'tokenizer'
        snap.mkdir(parents=True)
        (snap / 'spiece.model').write_bytes(b'\x00')
        assert Path(vc.umt5_tokenizer_dir()) == snap


# --- the sidecar plan -----------------------------------------------------------------

def test_plan_sidecar_serves_the_paragraph_when_it_fits_the_window():
    from app.services.video_bank_service import (
        SIDECAR_TOKEN_RESERVE, plan_sidecar, trigger_token_bound,
    )
    plan = plan_sidecar('mychar', 'a woman walks in soft light', None,
                        fields_json=FIELDS, caption_tokens=200, token_budget=512)
    assert plan['text'] == 'mychar, a woman walks in soft light'
    assert plan['served_short'] is False
    assert plan['measured'] is True
    # Lines reserve + THIS trigger at its byte ceiling — a flat reserve was
    # breached by an invented five-word trigger measuring 42 tokens alone.
    assert plan['tokens'] == 200 + SIDECAR_TOKEN_RESERVE + trigger_token_bound('mychar')


def test_plan_sidecar_serves_the_short_form_where_the_encoder_would_cut():
    from app.services.video_bank_service import plan_sidecar
    plan = plan_sidecar('mychar', 'a long paragraph the encoder would truncate', None,
                        fields_json=FIELDS, caption_tokens=500, token_budget=512)
    assert plan['served_short'] is True
    # Subject, motion, setting, style — the model's own words, as sentences; the
    # trigger still leads, exactly once.
    assert plan['text'] == ('mychar, a young woman in a long red dress. '
                            'walks slowly toward the tall window and turns. '
                            'a bright, sparsely furnished apartment. '
                            'soft morning daylight, warm palette.')
    # The short form was not measured by the pass: its count is the estimate.
    assert plan['measured'] is False


def test_plan_sidecar_never_cuts_and_never_invents_a_window():
    from app.services.video_bank_service import plan_sidecar
    # No fields: the paragraph, whole — and counted over, so the preflight says so.
    long = ' '.join(['word'] * 400)
    plan = plan_sidecar('', long, None, fields_json=None, caption_tokens=None,
                        token_budget=512)
    assert plan['served_short'] is False
    assert plan['text'] == long
    assert plan['tokens'] > 512 and plan['measured'] is False
    # No window published: the paragraph, even with fields and a huge count.
    plan = plan_sidecar('', 'a paragraph', None, fields_json=FIELDS,
                        caption_tokens=5000, token_budget=None)
    assert plan['served_short'] is False and plan['text'] == 'a paragraph'
    # No caption at all: no text, no tokens — the empty-sidecar count owns that.
    assert plan_sidecar('', '', None, token_budget=512)['tokens'] == 0


def test_the_estimate_holds_against_what_refuted_it():
    """Not a tautology against the constant — the numbers the refutation
    measured (2026-09-01). 1.4/word undercounts a quarter of single captions
    but held 0/984 real + 0/1600 synthetic against the 512 window (the ratio
    falls with length); CJK was its structural blind spot: a ~540-token ZH
    caption estimated at 42 and cut in silence."""
    from app.services.video_bank_service import TOKENS_PER_WORD, estimate_tokens
    assert TOKENS_PER_WORD >= 1.36          # the measured mean floor
    assert estimate_tokens('one two three') == 5
    assert estimate_tokens('') == 0 and estimate_tokens(None) == 0
    # A spaceless CJK caption is one "word" to split(); the char term answers.
    zh = '一名年轻女子穿着红色连衣裙走向窗户' * 10          # 170 chars, no spaces
    assert estimate_tokens(zh) >= 170
    # Mixed text errs toward over — the survivable direction.
    assert estimate_tokens('a woman 女子') >= estimate_tokens('a woman woman')


def test_the_trigger_is_bounded_at_the_byte_ceiling_not_guessed_from_words():
    """'zylphraxian_cinematic_style_v3' is two words and sixteen real tokens;
    five invented words measured 42 — word-based estimates fail exactly on the
    strings triggers are made of. One token per input byte is the ceiling
    sentencepiece cannot exceed, so the bound is provable, and its overshoot
    on a friendly trigger only tightens the budget check."""
    from app.services.video_bank_service import plan_sidecar, trigger_token_bound
    assert trigger_token_bound('zylphraxian_cinematic_style_v3') >= 16
    assert trigger_token_bound('') == 0 and trigger_token_bound(None) == 0
    # The refuter's breach, replayed against the fix: a 45-byte invented
    # trigger + a 450-token caption must now read over a 512 window.
    plan = plan_sidecar('zylphraxian glombular vexicated brumthic quorvalent',
                        ' '.join(['word'] * 300), None,
                        caption_tokens=450, token_budget=512)
    assert plan['tokens'] > 512


# --- the published window, and only the published one --------------------------------

def test_every_wan_profile_carries_the_published_umt5_window_and_nobody_invents_one():
    from app.services import video_targets as vt
    for key in ('wan21', 'wan21_i2v', 'wan22_14b', 'wan22_14b_i2v', 'wan22_ti2v5b'):
        assert vt.get(key).get('caption_token_budget') == 512, key
    for key in ('ltx2', 'ltx23', 'minimax_h3', 'minimax_h3_ref2va', 'generic'):
        assert 'caption_token_budget' not in vt.get(key), key


# --- the two halves of the worker ----------------------------------------------------

def test_the_child_counts_with_sentencepiece_first_and_on_the_prose():
    code = INFER.read_text(encoding='utf-8')
    assert "req.get('tokenizer_dir')" in code
    assert 'import sentencepiece as spm' in code
    # +1: the EOS the HF wrapper appends and the raw model does not.
    assert 'len(sp.encode(str(text))) + 1' in code
    # sentencepiece BEFORE transformers — 5.3 cannot rebuild this tokenizer's
    # precompiled normalizer from the .model file (measured 2026-09-01).
    assert code.index('import sentencepiece as spm') < code.index('from transformers import AutoTokenizer')
    # Counted on the PROSE: the parent's stdlib-only splitter, imported by path.
    assert "'caption_fields.py'" in code
    assert 'split_caption_fields(caption)[0]' in code
    assert "'tokens': tokens" in code
    assert "'token_counter': token_counter" in code


def test_the_parent_hands_over_the_tokenizer_and_keeps_the_last_count():
    src = WORKER.read_text(encoding='utf-8')
    assert "'tokenizer_dir': self.tokenizer_dir" in src
    assert 'self.last_tokens = tokens if isinstance(tokens, int)' in src
    # A refusal resets the count: a stale number must never be stored on the next clip.
    refusal = src.index('caption worker refused a shot')
    assert 'self.last_tokens = None' in src[refusal:refusal + 400]

def test_a_human_edit_sheds_every_machine_derived_column(app, monkeypatch):
    """set_caption already refused to credit a checkpoint for human words; the
    same honesty owes the fields and the token count. Stale ones would serve
    the OLD caption's facets — and, past the budget, its short form INSTEAD of
    the human's words in the exported .txt."""
    from app.models import VideoClip
    monkeypatch.setattr(vc, '_write_caption_frames',
                        lambda src, times, dest, stem: [f'{dest}/{stem}_0.jpg'])
    monkeypatch.setattr(vc, '_caption_frames', lambda paths, prompt, **kw: TAILED)
    monkeypatch.setattr(vcw, 'CaptionWorker', _FakeWorker)
    monkeypatch.setattr(vc, 'umt5_tokenizer_dir', lambda: '/m/t')
    bank_id = _bank_with_one_clip(app)
    with app.app_context():
        vc.run_captions(bank_id)
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        assert clip.caption_fields and clip.caption_tokens == 187
        row = vc.set_caption('local', bank_id, clip.id, 'My own words.')
        assert row is not None
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        assert clip.caption == 'My own words.'
        assert clip.caption_state == 'edited'
        assert clip.caption_fields is None
        assert clip.caption_tokens is None

def test_a_tail_cut_by_the_generation_cap_never_becomes_the_caption():
    """Review finding 5, the concrete scenario: the model writes 400 words,
    `---`, `Subject: a woman in a red dre` — cut at the cap. The parse
    (rightly) returns that one label; without a floor the exported .txt became
    `mychar, a woman in a red dre.` and the paragraph was silently thrown
    away. The floor refuses the stump: the paragraph ships WHOLE, stays
    counted over the window, and the plan says the tail was incomplete."""
    from app.services.video_bank_service import plan_sidecar
    stump = json.dumps({'subject': 'a woman in a red dre', 'motion': None,
                        'setting': None, 'style': None, 'short': None})
    long_para = ' '.join(['word'] * 400)
    plan = plan_sidecar('mychar', long_para, None, fields_json=stump,
                        caption_tokens=560, token_budget=512)
    assert plan['served_short'] is False
    assert plan['tail_incomplete'] is True
    assert plan['text'] == f'mychar, {long_para}'
    assert plan['tokens'] > 512


def test_a_complete_but_skeletal_tail_is_refused_too():
    """Four labels of one word each pass the completeness check and still
    cannot stand in for a 400-word paragraph — the length floor (twenty
    words, what the prompt asks of `short` alone) refuses them."""
    from app.services.video_bank_service import plan_sidecar
    thin = json.dumps({'subject': 'woman', 'motion': 'walks', 'setting': 'room',
                       'style': 'soft', 'short': 'x'})
    plan = plan_sidecar('', ' '.join(['word'] * 400), None, fields_json=thin,
                        caption_tokens=560, token_budget=512)
    assert plan['served_short'] is False and plan['tail_incomplete'] is True


def test_the_local_worker_leaves_the_tail_generation_headroom():
    """The paragraph and the tail compete for one cap and the tail is written
    last — 600 was exactly enough to cut it on a chatty model. Pinned with its
    reason so a tidy-up cannot quietly shave it back."""
    src = WORKER.read_text(encoding='utf-8')
    assert 'num_predict=800' in src
    assert 'written LAST' in src

