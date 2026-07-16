"""write_job_config must name the on-disk job config by the base/family-aware
run name, not the trigger alone -- otherwise a zimage run and a krea run of the
same trigger both write `{trigger}.json` and the second clobbers the first, and
purge_training_artifacts (which keyed by trigger too) would destroy the wrong
family's record. Runtime verification (agent C) reproduced this."""
import os
from types import SimpleNamespace
from unittest.mock import patch
import pytest


def _ds(train_type, train_variant=None):
    # Minimal stub satisfying _run_name / _safe_trigger / _train_type / _base_tag
    # / _krea_is_raw (train_variant None → Krea defaults to Raw).
    return SimpleNamespace(id=1, user_id='local', trigger_word='AgentCTest',
                           train_type=train_type, train_base_model=None,
                           train_variant=train_variant)


@pytest.fixture()
def training(app, tmp_path):
    with app.app_context():
        from app import config as cfg
        aitk = tmp_path / 'aitoolkit'
        (aitk / 'config' / 'generated').mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)}})
        from app.services import lora_training as lt
        yield lt, str(aitk / 'config' / 'generated')


def test_write_job_config_is_run_name_scoped(training):
    lt, generated = training
    with patch('app.services.lora_training.build_job_config', return_value={'x': 1}):
        p_zimage = lt.write_job_config(_ds('zimage'), 'folderA')
        p_krea = lt.write_job_config(_ds('krea'), 'folderB')            # default -> Raw
        p_krea_turbo = lt.write_job_config(_ds('krea', 'turbo'), 'folderC')
    assert os.path.basename(p_zimage) == (
        'ulocal_AgentCTest_Z-Image-Turbo.json')
    assert os.path.basename(p_krea) == 'ulocal_AgentCTest_Krea-2-Raw.json'
    assert os.path.basename(p_krea_turbo) == 'ulocal_AgentCTest_Krea-2-Turbo.json'
    # Raw and Turbo Krea configs must not clobber each other (incompatible weights).
    assert os.path.isfile(p_zimage) and os.path.isfile(p_krea) and os.path.isfile(p_krea_turbo)


def test_purge_removes_all_family_configs_but_not_sibling_trigger(training):
    lt, generated = training
    with patch('app.services.lora_training.build_job_config', return_value={'x': 1}):
        p1 = lt.write_job_config(_ds('zimage'), 'folderA')
        p2 = lt.write_job_config(_ds('krea'), 'folderB')
    assert p1 != p2 and os.path.isfile(p1) and os.path.isfile(p2)
    sibling = os.path.join(generated, 'ulocal_AgentCTest2.json')  # different trigger
    with open(sibling, 'w', encoding='utf-8') as fh:
        fh.write('{}')
    removed = lt.purge_training_artifacts('local', 'AgentCTest')
    assert not os.path.isfile(p1) and not os.path.isfile(p2)
    assert os.path.isfile(sibling)  # trigger-boundary guard: AgentCTest2 survives
