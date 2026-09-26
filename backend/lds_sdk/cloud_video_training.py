"""Versioned cloud_video_training operations; no provider implementation path is exposed."""
from __future__ import annotations
import inspect
from ._cloud_provider import provider



def launch_cloud_video_training(user_id, video_dataset_id, steps=1000, base_model=None, low_vram=False, gpu_name=None, do_i2v=False, sample_prompts=None, distillation='auto', resume_ckpt_paths=None, resume_step=None, parent_run_id=None, auto_retry_of=None, auto_retry_count=0, allow_parallel_run=False, _provision=None, rank=16):
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 256:
        raise ValueError('LoRA rank must be an integer between 1 and 256')
    launch = provider(recovery=False).launch_cloud_video_training
    # Older installed Cloud packages expose API v1 without this optional control.
    # Keep their historical default working; never silently discard another rank.
    supports_rank = 'rank' in inspect.signature(launch).parameters
    if not supports_rank and rank != 16:
        raise RuntimeError('Update Cloud training in Plugins → Updates to choose a video LoRA rank.')
    return launch(user_id, video_dataset_id, steps, base_model, low_vram, gpu_name, do_i2v, sample_prompts, distillation, resume_ckpt_paths, resume_step, parent_run_id, auto_retry_of, auto_retry_count, allow_parallel_run, _provision, **({'rank': rank} if supports_rank else {}))



def video_gpu_tiers(user_id, video_dataset_id, steps=None):
    return provider(recovery=False).video_gpu_tiers(user_id, video_dataset_id, steps)



def delete_cloud_video_run(user_id, run_id):
    return provider(recovery=False).delete_cloud_video_run(user_id, run_id)



def retry_cloud_video_run(user_id, run_id, allow_parallel_run=False):
    return provider(recovery=False).retry_cloud_video_run(user_id, run_id, allow_parallel_run)



def continue_cloud_video_run(user_id, run_id, extra_steps=1000, from_step=None, allow_parallel_run=False):
    return provider(recovery=False).continue_cloud_video_run(user_id, run_id, extra_steps, from_step, allow_parallel_run)


__all__ = ['launch_cloud_video_training', 'video_gpu_tiers', 'delete_cloud_video_run', 'retry_cloud_video_run', 'continue_cloud_video_run']
