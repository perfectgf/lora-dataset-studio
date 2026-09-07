"""The four levers that decide how FAST a local run goes.

They were frozen at the values that make a 12B DiT fit in 24 GB: batch 1,
gradient checkpointing on, weights-only quantisation, no compile. On a bigger
card that calibration is a tax, and until now there was no way to refuse it —
`qtype` was plumbed end to end but had no control at all, and the other three
were literals in seven recipe builders.

The contract these tests hold: **defaults are byte-identical to before**, every
lever reaches the config ai-toolkit actually reads, and the choices we offer are
ones ai-toolkit accepts (a knob the trainer ignores is worse than no knob).
"""
import pytest

from app import config as cfg
from app.config import LOCAL_USER
from app.services import face_dataset_service as svc
from app.services import lora_training as lt


def _prepare(tmp_path, family='krea', **settings):
    """A dataset of `family` with the levers set, and a configured ai-toolkit."""
    cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    ds = svc.create_dataset(LOCAL_USER, f'Speed {family}', f'sp_{family}',
                            train_type=family)
    if settings:
        lt.update_train_settings(LOCAL_USER, ds.id, settings)
        ds = svc.get_dataset(LOCAL_USER, ds.id)
    folder = tmp_path / f'ds_{family}'
    folder.mkdir(exist_ok=True)
    return ds, str(folder)


def _cfg(ds, folder):
    return lt.build_job_config(ds, folder, steps=1000)['config']['process'][0]


# --- the defaults do not move ------------------------------------------------

def test_an_untouched_dataset_still_emits_the_shipped_recipe(app, tmp_path):
    """The whole point of exposing a lever is that nobody who ignores it sees a
    change. Batch 1, checkpointing on, no compile key at all."""
    with app.app_context():
        p = _cfg(*_prepare(tmp_path))
    assert p['train']['batch_size'] == 1
    assert p['train']['gradient_checkpointing'] is True
    assert 'compile' not in p['model']
    assert p['model']['qtype'] == 'qfloat8'


# --- each lever reaches the config ------------------------------------------

def test_batch_size_reaches_the_train_block(app, tmp_path):
    with app.app_context():
        p = _cfg(*_prepare(tmp_path, batch_size=2))
    assert p['train']['batch_size'] == 2


def test_gradient_checkpointing_can_be_switched_off(app, tmp_path):
    """The memory-for-speed trade, made available rather than assumed."""
    with app.app_context():
        p = _cfg(*_prepare(tmp_path, gradient_checkpointing=False))
    assert p['train']['gradient_checkpointing'] is False


def test_compile_is_a_model_key_and_only_appears_when_asked(app, tmp_path):
    """Upstream reads `compile` off ModelConfig, not the train block — and an
    absent key is not the same as False for a config a user may share."""
    with app.app_context():
        on = _cfg(*_prepare(tmp_path, compile=True))
        off = _cfg(*_prepare(tmp_path, family='zimage', compile=False))
    assert on['model']['compile'] is True
    assert 'compile' not in off['model']
    assert 'compile' not in on['train']


def test_the_int8_w8a8_backend_is_offered_and_reaches_the_model_block(app, tmp_path):
    """`convrot8` is the only offered qtype that can be FASTER: the other three
    quantise weights only and promote them back before every matmul."""
    assert 'convrot8' in lt._QTYPE_CHOICES
    with app.app_context():
        p = _cfg(*_prepare(tmp_path, qtype='convrot8'))
    assert p['model']['qtype'] == 'convrot8'


def test_a_qtype_we_do_not_offer_is_refused_at_the_door(app, tmp_path):
    """A typo, or a backend this ai-toolkit does not carry, is rejected when it
    is SAVED rather than forwarded to a trainer that would refuse it twenty
    minutes in — and a rejected value leaves the stored setting untouched."""
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Bad qtype', 'badq', train_type='krea')
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, ds.id, {'qtype': 'no_such_backend'})
        folder = tmp_path / 'ds_badq'
        folder.mkdir(exist_ok=True)
        p = _cfg(svc.get_dataset(LOCAL_USER, ds.id), str(folder))
    assert p['model']['qtype'] == 'qfloat8'


# --- the levers survive the round trip the UI makes --------------------------

def test_the_panel_is_told_the_effective_value_and_the_choices(app, tmp_path):
    with app.app_context():
        ds, _ = _prepare(tmp_path, batch_size=4, gradient_checkpointing=False,
                         compile=True, qtype='convrot8')
        eff = lt.effective_train_settings(ds)
    assert eff['batch_size'] == 4
    assert eff['batch_size_choices'] == [1, 2, 4]
    assert eff['gradient_checkpointing'] is False
    assert eff['compile'] is True
    assert eff['qtype'] == 'convrot8'
    assert 'convrot8' in eff['qtype_choices']


def test_an_untouched_panel_reads_the_shipped_values(app, tmp_path):
    with app.app_context():
        ds, _ = _prepare(tmp_path)
        eff = lt.effective_train_settings(ds)
    assert eff['batch_size'] == 1
    assert eff['gradient_checkpointing'] is True
    assert eff['compile'] is False
    assert eff['qtype'] is None          # nothing stored = the family default


@pytest.mark.parametrize('key,value', [
    ('batch_size', 2), ('gradient_checkpointing', False), ('compile', True),
])
def test_a_run_records_which_levers_it_used(app, tmp_path, key, value):
    """Two runs of one dataset can differ only by these; a snapshot that omits
    them cannot tell an old run from a re-run at another speed."""
    with app.app_context():
        ds, _ = _prepare(tmp_path, **{key: value})
        snap = lt.launch_settings_snapshot(ds)
    assert snap[key] == value


# --- the whole family, not just Krea 2 ---------------------------------------

@pytest.mark.parametrize('family', ['krea', 'zimage', 'flux', 'flux2klein', 'anima'])
def test_every_family_honours_the_levers(app, tmp_path, family):
    """These are properties of the run, not of one architecture: a lever that
    works on Krea 2 and silently does nothing on Z-Image is the worst kind."""
    with app.app_context():
        p = _cfg(*_prepare(tmp_path, family=family, batch_size=2,
                           gradient_checkpointing=False))
    assert p['train']['batch_size'] == 2
    assert p['train']['gradient_checkpointing'] is False
