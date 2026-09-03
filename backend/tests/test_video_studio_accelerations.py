"""⚡ The Render panel's acceleration choice: the arena's top three, grafted
right, stored on the clip, downloadable from Setup."""
import pytest

from app import setup_installer
from app.services import video_test_studio as vts


def _graph(**kw):
    return vts.build_workflow(prompt='a woman turns and smiles', mode='t2v', seed=1,
                              frames=56, megapixels=0.2, sage=False, **kw)


def test_the_three_choices_are_the_arena_podium_at_six_steps():
    assert [a['id'] for a in vts.ACCELERATIONS] == ['turbo', 'parasyte', 'dareties']
    assert all(a['steps'] == vts.TURBO_STEPS == 6 for a in vts.ACCELERATIONS)
    assert vts.accel_spec('turbo')['file'] == vts.TURBO_LORA
    for a in vts.ACCELERATIONS:
        assert a['arena'].startswith('#') and a['hint'] and a['action']


def test_the_boolean_still_means_larryvrh_and_a_stranger_is_refused():
    assert vts.normalise_accel(None, turbo=True) == 'turbo'
    assert vts.normalise_accel('', turbo=False) == ''
    assert vts.normalise_accel('off', turbo=True) == 'turbo', 'the flag decides when nothing is named'
    assert vts.normalise_accel('Parasyte', turbo=False) == 'parasyte'
    with pytest.raises(ValueError):
        vts.normalise_accel('../evil', turbo=False)


def test_turbo_keeps_its_own_nodes_and_the_stock_choices_take_the_stock_path():
    turbo = _graph(turbo=True)
    assert turbo['accel'] == 'turbo' and vts.N_TURBO_LORA in turbo['workflow']
    assert vts.N_ACCEL_LORA not in turbo['workflow'] and vts.N_SHIFT not in turbo['workflow']

    for accel, strength in (('parasyte', 4.0), ('dareties', 0.8)):
        out = _graph(accel=accel)
        wf = out['workflow']
        assert out['accel'] == accel and out['steps'] == 6
        assert vts.N_TURBO_LORA not in wf and vts.N_TURBO_SAMPLER not in wf, 'no larryvrh pack needed'
        lora = wf[vts.N_ACCEL_LORA]
        assert lora['class_type'] == 'LoraLoaderModelOnly'
        assert lora['inputs']['lora_name'] == vts.accel_spec(accel)['file']
        assert lora['inputs']['strength_model'] == strength, 'the arena setting, not 1.0'
        shift = wf[vts.N_SHIFT]
        assert shift['class_type'] == 'MiniMaxH3SigmaShift'
        assert (shift['inputs']['shift_video'], shift['inputs']['shift_audio']) == (8.0, 3.0)
        assert shift['inputs']['model'] == [vts.N_ACCEL_LORA, 0]
        assert wf[vts.N_SAMPLER_SELECT]['inputs']['sampler_name'] == 'euler'
        assert wf[vts.N_SCHEDULER]['inputs']['scheduler'] == 'simple'
        # Everyone who read the base now reads the shifted, accelerated model:
        # the guider path and the scheduler alike (sigmas and sampling agree).
        readers = vts._model_readers(wf, [vts.N_SHIFT, 0])
        assert vts.N_SCHEDULER in readers and readers, 'the scheduler reads the patched chain'
        assert not vts._model_readers(wf, [vts.N_UNET, 0], skip=(vts.N_ACCEL_LORA,)), 'nothing reads the bare base'
        assert any(n.startswith('accel: ') and 'shift 8/3' in n for n in out['notes'])


def test_the_lora_under_test_sits_on_top_of_the_acceleration():
    out = _graph(accel='parasyte', lora='h3/lds/mine.safetensors', lora_strength=1.1)
    wf = out['workflow']
    assert wf[vts.N_TEST_LORA]['inputs']['model'] == [vts.N_ACCEL_LORA, 0]
    assert wf[vts.N_TEST_LORA]['inputs']['strength_model'] == 1.1
    assert wf[vts.N_SHIFT]['inputs']['model'] == [vts.N_TEST_LORA, 0], 'the shift reads the whole LoRA stack'


def test_an_explicit_step_count_still_wins_over_the_accelerations_six():
    out = _graph(accel='dareties', steps=8)
    assert out['steps'] == 8
    assert any(n.endswith('steps=8') for n in out['notes']), 'the note says what the graph runs, not the default'
    turbo = _graph(turbo=True, steps=4)
    assert turbo['steps'] == 4 and any(n.endswith('steps=4') for n in turbo['notes'])


def test_every_optional_weight_with_a_button_is_a_download_setup_knows():
    for action, subs, filename, _what in vts.OPTIONAL_WEIGHTS:
        if action is None:
            continue
        entry = setup_installer._MODEL_DOWNLOADS[action]
        assert action in setup_installer.INSTALL_ACTIONS
        assert entry['dest'] == (subs[0], filename), f'{action}: the file Setup writes is the file the graph loads'
        assert entry['url'].startswith('https://huggingface.co/') and '/resolve/main/' in entry['url']
        assert entry['license_url']


def test_the_status_says_what_this_machine_has(monkeypatch):
    present = {vts.PARASYTE_LORA}
    monkeypatch.setattr(vts, '_weight_present', lambda subs, name: name in present)
    monkeypatch.setattr(vts, 'option_availability',
                        lambda classes=None: {'turbo': {'available': False, 'pack': 'x', 'url': 'u', 'search': 's', 'nodes': ['MiniMaxH3TurboSampler']}})
    rows = {r['id']: r for r in vts.accelerations_status(classes=set())}
    assert rows['parasyte']['available'] is True and rows['parasyte']['pack'] is None
    assert rows['turbo']['available'] is False and rows['turbo']['weight_present'] is False
    assert rows['dareties']['available'] is False and rows['dareties']['action'] == 'h3_dareties_lora'
    present.add(vts.TURBO_LORA)
    rows = {r['id']: r for r in vts.accelerations_status(classes=set())}
    assert rows['turbo']['available'] is False, 'the weight alone is not enough: larryvrh needs its pack'
