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
import math
from pathlib import Path

from app.services import video_caption as vc
from app.services import video_caption_worker as vcw

INFER = Path(__file__).resolve().parents[1] / 'infer' / 'video_caption_infer.py'
WORKER = Path(__file__).resolve().parents[1] / 'app' / 'services' / 'video_caption_worker.py'

PARA = 'A woman in a red dress walks to the window and turns into the light.'
TAILED = (PARA + '\n---\nSubject: a woman in a red dress\nMotion: walks to the window and turns\n'
          'Setting: a bright apartment\nStyle: soft daylight\nShort: a woman walks to a bright window')
FIELDS = json.dumps({'subject': 'a woman', 'motion': 'walks', 'setting': 'a room',
                     'style': 'soft light', 'short': 'a woman walks in a room'})


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
    from app.services.video_bank_service import SIDECAR_TOKEN_RESERVE, plan_sidecar
    plan = plan_sidecar('mychar', 'a woman walks in soft light', None,
                        fields_json=FIELDS, caption_tokens=200, token_budget=512)
    assert plan['text'] == 'mychar, a woman walks in soft light'
    assert plan['served_short'] is False
    assert plan['measured'] is True
    assert plan['tokens'] == 200 + SIDECAR_TOKEN_RESERVE


def test_plan_sidecar_serves_the_short_form_where_the_encoder_would_cut():
    from app.services.video_bank_service import plan_sidecar
    plan = plan_sidecar('mychar', 'a long paragraph the encoder would truncate', None,
                        fields_json=FIELDS, caption_tokens=500, token_budget=512)
    assert plan['served_short'] is True
    # Subject, motion, setting, style — the model's own words, as sentences; the
    # trigger still leads, exactly once.
    assert plan['text'] == 'mychar, a woman. walks. a room. soft light.'
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


def test_the_estimate_rounds_up_from_the_measured_ratio():
    from app.services.video_bank_service import TOKENS_PER_WORD, estimate_tokens
    # 1.36 tokens per word measured over 48 captions; the estimate must not undershoot it.
    assert TOKENS_PER_WORD >= 1.36
    assert estimate_tokens('one two three') == math.ceil(3 * TOKENS_PER_WORD)
    assert estimate_tokens('') == 0 and estimate_tokens(None) == 0


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
