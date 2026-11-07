"""The trainer image a core pins for a family, as Cloud Training reads it.

The plugin calls this for every launch, with no fallback: a read that silently fell
back to the shared image would rent, and bill, a pod without the family's trainer.
Its manifest asks for LDS 2026.11.06.1 and no higher plugin API, so a core that
loses this name breaks every cloud launch of the plugin it can install.
"""
from app.services import face_dataset_service as fds
from app.services import lora_training as lt
from lds_sdk import training_runtime


def test_the_sdk_serves_the_cores_pins():
    assert 'cloud_trainer_image' in training_runtime.__all__
    pin = lt.CLOUD_TRAINER_IMAGES['qwenimage21']
    assert training_runtime.cloud_trainer_image('qwenimage21') == pin
    assert training_runtime.cloud_trainer_image('QwenImage21') == pin


def test_every_other_family_keeps_the_shared_image():
    assert set(lt.CLOUD_TRAINER_IMAGES) == {'qwenimage21'}
    for family in (*(f for f in fds.TRAIN_TYPES if f != 'qwenimage21'), 'video', '', None):
        assert training_runtime.cloud_trainer_image(family) is None
