"""Shared training recipes, export and checkpoint import contracts.

Cloud owns provisioning and monitoring. The host owns the local dataset export,
recipe validation and training provenance used by every provider. These are
named operations, never a module mirror or a returned host ORM/session.
"""
from app.services.lora_training import (
    _PERSISTED, _SAMPLE_RE as SAMPLE_RE, _UNVERIFIED_MARKER as UNVERIFIED_MARKER,
    _launch_transaction_lock as launch_transaction_lock,
    FULL_TRANSFORMER_MAX_STEP_SAVES, KREA_TURBO_BASE)
from app.services.hf_storage import DENSE_CHECKPOINT_FALLBACK_BYTES

__all__ = ['assert_full_transformer_recipe', 'default_variant_for', 'dir_size', 'parse_training_log', 'run_dir', 'run_name', 'train_settings', 'train_type', 'valid_variants_for', 'assert_resume_checkpoint_record', 'assert_trainable', 'assert_zimage_custom_recipe_confirmed', 'build_job_config', 'checkpoint_file_path', 'default_steps', 'dense_fp8_export_enabled', 'dense_inference_hint', 'dense_keep_bf16_master', 'dense_max_step_saves_for', 'download_bytes_seen', 'export_dataset_to_aitoolkit', 'failed_local_run', 'import_checkpoint', 'launch_settings_snapshot', 'legacy_lokr_resume_error', 'list_checkpoints', 'normalize_training_mode', 'official_base_repo', 'parse_download_progress', 'preflight_custom_paths', 'resolve_masked', 'resolve_resume_lr', 'slider_mode_enabled', 'training_status', 'validate_resume_overrides', 'validate_resume_record_id', 'zimage_recipe_diagnostic', 'zimage_training_recipe', 'custom_combo_hash', 'is_custom_weights', 'foreign_base_message', 'dense_checkpoint_bytes', 'training_mode', 'SAMPLE_RE', 'UNVERIFIED_MARKER', 'launch_transaction_lock', 'FULL_TRANSFORMER_MAX_STEP_SAVES', 'KREA_TURBO_BASE', 'DENSE_CHECKPOINT_FALLBACK_BYTES', 'cloud_trainer_image']



def cloud_trainer_image(family):
    """The container image a cloud trainer boots for ``family`` when the host pins one
    (an architecture newer than the provider's shared image), else None. Every core
    from LDS 2026.11.06.1 has it, whatever its plugin API minor."""
    from app.services.lora_training import cloud_trainer_image as operation
    return operation(family)



def assert_full_transformer_recipe(ds):
    from app.services.lora_training import _assert_full_transformer_recipe as operation
    return operation(ds)



def default_variant_for(family):
    from app.services.lora_training import _default_variant_for as operation
    return operation(family)



def dir_size(path):
    from app.services.lora_training import _dir_size as operation
    return operation(path)



def parse_training_log(text: str):
    from app.services.lora_training import _parse_training_log as operation
    return operation(text)



def run_dir(user_id, dataset_id, base_model=_PERSISTED, family=None, variant=_PERSISTED):
    from app.services.lora_training import _run_dir as operation
    return operation(user_id, dataset_id, base_model, family, variant)



def run_name(ds, base_model=_PERSISTED, family=None, variant=_PERSISTED):
    from app.services.lora_training import _run_name as operation
    return operation(ds, base_model, family, variant)



def train_settings(ds):
    from app.services.lora_training import _train_settings as operation
    return operation(ds)



def train_type(ds, family=None):
    from app.services.lora_training import _train_type as operation
    return operation(ds, family)



def valid_variants_for(family):
    from app.services.lora_training import _valid_variants_for as operation
    return operation(family)



def assert_resume_checkpoint_record(checkpoint, expected_record_id):
    from app.services.lora_training import assert_resume_checkpoint_record as operation
    return operation(checkpoint, expected_record_id)



def assert_trainable(dataset_id, train_type=None, allow_caption_mismatch=False, allow_uncaptioned=False, allow_caption_quality=False, variant=None, allow_not_ready=False):
    from app.services.lora_training import assert_trainable as operation
    return operation(dataset_id, train_type, allow_caption_mismatch, allow_uncaptioned, allow_caption_quality, variant, allow_not_ready)



def assert_zimage_custom_recipe_confirmed(family, base_model, variant, allow_unverified_weights=False):
    from app.services.lora_training import assert_zimage_custom_recipe_confirmed as operation
    return operation(family, base_model, variant, allow_unverified_weights)



def build_job_config(ds, dataset_folder: str, steps: int=3000, training_folder=None):
    from app.services.lora_training import build_job_config as operation
    return operation(ds, dataset_folder, steps, training_folder)



def checkpoint_file_path(user_id, dataset_id, filename, base_model=_PERSISTED, family=None, variant=_PERSISTED):
    from app.services.lora_training import checkpoint_file_path as operation
    return operation(user_id, dataset_id, filename, base_model, family, variant)



def default_steps(ds, train_type=None, variant=None):
    from app.services.lora_training import default_steps as operation
    return operation(ds, train_type, variant)



def dense_fp8_export_enabled(ds):
    from app.services.lora_training import dense_fp8_export_enabled as operation
    return operation(ds)



def dense_inference_hint(ds=None):
    from app.services.lora_training import dense_inference_hint as operation
    return operation(ds)



def dense_keep_bf16_master(ds):
    from app.services.lora_training import dense_keep_bf16_master as operation
    return operation(ds)



def dense_max_step_saves_for(ds):
    from app.services.lora_training import dense_max_step_saves_for as operation
    return operation(ds)



def download_bytes_seen(text: str):
    from app.services.lora_training import download_bytes_seen as operation
    return operation(text)



def export_dataset_to_aitoolkit(user_id, dataset_id, masked: bool=True, dest_dir=None, masked_faces: bool=True):
    from app.services.lora_training import export_dataset_to_aitoolkit as operation
    return operation(user_id, dataset_id, masked, dest_dir, masked_faces)



def failed_local_run():
    from app.services.lora_training import failed_local_run as operation
    return operation()



def import_checkpoint(user_id, dataset_id, filename, base_model=_PERSISTED, family=None, src_dir=None, version=None, variant=_PERSISTED, run_id=None, run_source=None, return_meta=False):
    from app.services.lora_training import import_checkpoint as operation
    return operation(user_id, dataset_id, filename, base_model, family, src_dir, version, variant, run_id, run_source, return_meta)



def launch_settings_snapshot(ds, family=None, masked=None):
    from app.services.lora_training import launch_settings_snapshot as operation
    return operation(ds, family, masked)



def legacy_lokr_resume_error(parent_geometry, fallback_settings=None):
    from app.services.lora_training import legacy_lokr_resume_error as operation
    return operation(parent_geometry, fallback_settings)



def list_checkpoints(user_id, dataset_id, base_model=_PERSISTED, family=None, variant=_PERSISTED):
    from app.services.lora_training import list_checkpoints as operation
    return operation(user_id, dataset_id, base_model, family, variant)



def normalize_training_mode(value):
    from app.services.lora_training import normalize_training_mode as operation
    return operation(value)



def official_base_repo(ds, family=None, variant=_PERSISTED):
    from app.services.lora_training import official_base_repo as operation
    return operation(ds, family, variant)



def parse_download_progress(text: str):
    from app.services.lora_training import parse_download_progress as operation
    return operation(text)



def preflight_custom_paths(family, weights=None, vae_path=None, te_path=None, allow_unverified_weights=False):
    from app.services.lora_training import preflight_custom_paths as operation
    return operation(family, weights, vae_path, te_path, allow_unverified_weights)



def resolve_masked(ds, requested=None):
    from app.services.lora_training import resolve_masked as operation
    return operation(ds, requested)



def resolve_resume_lr(settings: dict, lr_factor):
    from app.services.lora_training import resolve_resume_lr as operation
    return operation(settings, lr_factor)



def slider_mode_enabled(ds):
    from app.services.lora_training import slider_mode_enabled as operation
    return operation(ds)



def training_status(user_id=None):
    from app.services.lora_training import training_status as operation
    return operation(user_id)



def validate_resume_overrides(overrides):
    from app.services.lora_training import validate_resume_overrides as operation
    return operation(overrides)



def validate_resume_record_id(expected_record_id):
    from app.services.lora_training import validate_resume_record_id as operation
    return operation(expected_record_id)



def zimage_recipe_diagnostic(family, variant, effective_base=None, training_adapter=None, recipe_version=None):
    from app.services.lora_training import zimage_recipe_diagnostic as operation
    return operation(family, variant, effective_base, training_adapter, recipe_version)



def zimage_training_recipe(variant=None, base_model=None):
    from app.services.lora_training import zimage_training_recipe as operation
    return operation(variant, base_model)



def custom_combo_hash(ds, base_model=_PERSISTED, family=None):
    from app.services.lora_training import _custom_combo_hash as operation
    return operation(ds, base_model, family)



def is_custom_weights(value):
    from app.services.lora_training import _is_custom_weights as operation
    return operation(value)



def foreign_base_message(family, base_model):
    from app.services.lora_training import foreign_base_message as operation
    return operation(family, base_model)



def dense_checkpoint_bytes(_runs=None):
    from app.services.hf_storage import dense_checkpoint_bytes as operation
    return operation(_runs)



def training_mode(ds, override=None):
    from app.services.lora_training import training_mode as operation
    return operation(ds, override)
