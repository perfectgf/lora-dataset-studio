"""Cloud video rank survives SDK versions, launch stamps and pod rebuilding."""
import json
from types import SimpleNamespace

import pytest

from public_cloud_test_io import no_cloud_provider_io  # noqa: F401
from test_cloud_video_launch import _video_dataset

pytestmark = pytest.mark.plugins('cloud_training', 'video')


def test_default_rank_keeps_an_installed_legacy_provider_working(monkeypatch):
    from lds_sdk import cloud_video_training as sdk
    calls = []

    def legacy(*args):
        calls.append(args)
        return {'run_id': 7}

    monkeypatch.setattr(sdk, 'provider', lambda **kwargs: SimpleNamespace(
        launch_cloud_video_training=legacy))
    assert sdk.launch_cloud_video_training('local', 4) == {'run_id': 7}
    assert len(calls[0]) == 16
    with pytest.raises(RuntimeError, match='Update Cloud training'):
        sdk.launch_cloud_video_training('local', 4, rank=32)
    assert len(calls) == 1


@pytest.mark.parametrize('rank', [True, 0, 257, 16.5, '32'])
def test_invalid_rank_is_refused_before_provider_or_reservation(monkeypatch, rank):
    from lds_sdk import cloud_video_training as sdk
    from lds_cloud_training import cloud_video_training as cvt
    monkeypatch.setattr(sdk, 'provider', lambda **kwargs: pytest.fail('provider accessed'))
    for launch in (sdk.launch_cloud_video_training, cvt.launch_cloud_video_training):
        with pytest.raises(ValueError, match='LoRA rank'):
            launch('local', 999, rank=rank)


@pytest.mark.parametrize('rank', [16, 32])
def test_rank_reaches_validation_stamp_pod_and_relaunch(app, tmp_path, monkeypatch, rank):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from lds_sdk import cloud_video_training as sdk
    from lds_cloud_training import cloud_training as ct, cloud_video_training as cvt

    seen = []
    original = cvt.video_training.build_job_config

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result['config']['process'][0]['network']['linear'])
        return result

    monkeypatch.setattr(cvt.video_training, 'build_job_config', build)
    with app.app_context():
        dataset = _video_dataset(tmp_path)
        result = sdk.launch_cloud_video_training(
            'local', dataset.id, steps=100, rank=rank, _provision=lambda run: None)
        run = db.session.get(CloudTrainingRun, result['run_id'])
        stamp = json.loads(run.train_params)
        assert stamp['rank'] == rank
        config = ct._build_pod_job_config(run, '/staged/video', {
            'DATASETS_FOLDER': '/workspace/datasets', 'TRAINING_FOLDER': '/workspace/output'})
        assert config['config']['process'][0]['network']['linear'] == rank
        assert seen == [rank, rank]
        assert cvt._relaunch_args(stamp)['rank'] == rank
        assert cvt._relaunch_args({})['rank'] == 16
