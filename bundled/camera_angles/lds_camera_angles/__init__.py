"""📷 Camera angles — the bundled plugin.

Re-shoots ONE picture from other camera positions with Qwen-Image-Edit and
the Multiple-Angles LoRA: the subject stays put and the backdrop reprojects
with the camera, which no edit model does by prompting harder (the
measurement is in ``camera_angles.py``). The standalone workspace imports
images and keeps their views in plugin storage. Gallery and dataset actions
also keep their existing results beside the source.

What the plugin registers through ``lds.api``:

* the Gallery/dataset camera routes, catalog and standalone workspace routes;
* standalone job completion and durable imported images/results;
* the four weight files Setup downloads, as Setup actions, and the one-click
  ``camera`` install group (the Qwen VAE is the Krea 2 lane's download —
  one file, one button — so the group names ``krea_vae`` as a member);
* the lane's readiness under ``comfyui.camera_missing`` /
  ``comfyui.camera_ready``, where every Setup surface reads it;
* the ``caption.stamp`` hook that keeps a view's angle phrase in its caption
  through every re-caption;
* the ``camera_unet`` slot of the model picker, listing the lane's own
  candidate walk.

The ``camera.*`` settings section stays in the core's defaults (a stored key
never moves) and is declared in the manifest's ``owns`` table.
"""
from . import qwen_camera_helper as qch

__version__ = '1.1.0'

# The weights that move the CAMERA rather than the subject.
#
# WHY A SECOND BASE MODEL AT ALL, when a 9 GB one is already installed. Measured
# on this repo's own Klein lane (2026-08-25, one reference, seed held constant):
# asked for a profile or a back view, Klein turns the PERSON and leaves the room
# exactly where it was — every phrasing tried, English, an explicit "only the
# photographer moves", the Chinese cinematography terms, with and without the
# one camera LoRA that exists for Klein 9B. The backdrop delta stayed at 11-14
# (noise) against 64-80 for a real viewpoint change. The Qwen LoRA below was
# trained on 3000+ gaussian-splatting renders — pairs where the subject cannot
# move and the background must — and reaches 64-80 on all eight angles tried.
# That inversion is the feature; it is not reachable by prompting harder.
#
# URL survey 2026-08-26 (anonymous HTTP HEAD, no token): all four answer 200 and
# the signed CDN URL carries `user_id=public` — none is access-gated. The
# 401/403 recovery path is kept anyway, like every other catalog: a
# measurement is a photograph of one moment.
#
# NOT LISTED, on purpose: the Qwen image VAE. The core's `krea_vae` already
# installs the identical file to the identical destination — a second key for
# the same bytes would put the same gigabyte on the Setup screen twice and let
# two copies drift. qwen_camera_helper.CAMERA_VAE_ACTION names that button.
#
# `dest[0]` is 'diffusion_models' for the model and 'loras' for the adapters,
# both under a `qwen/` subfolder — the same shape the Klein and Krea lanes use,
# and what qwen_camera_helper._scan looks for first.
DOWNLOADS = {
    'camera_model': {
        'url': 'https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors',
        'dest': ('diffusion_models', 'qwen', 'qwen_image_edit_2511_fp8mixed.safetensors'),
        'min_free_gb': 25, 'gated': False, 'min_bytes': 4 * 1024 ** 3,
        'license_url': 'https://huggingface.co/Qwen/Qwen-Image-Edit-2511',
    },
    # The lane's REASON. Without it the base model still edits, it just answers
    # `<sks>` the way any edit model does — by turning the subject. A camera view
    # that silently has no camera in it would look like a success, which is why
    # qwen_camera_helper lists this one as REQUIRED, not recommended.
    'camera_lora': {
        'url': 'https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors',
        'dest': ('loras', 'qwen', 'Qwen-Image-Edit-2511-Multiple-Angles.safetensors'),
        'min_free_gb': 2, 'gated': False, 'min_bytes': 32 * 1024 ** 2,
        'license_url': 'https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA',
    },
    # Speed only, and genuinely optional: absent, the lane raises its own step
    # count (STEPS_WITHOUT_SPEED_LORA) and renders correctly, about five times
    # slower. Keeping 4 steps without it would render noise — which is why the
    # step count and this file are decided in the same place.
    'camera_speed_lora': {
        'url': 'https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
        'dest': ('loras', 'qwen', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'),
        'min_free_gb': 3, 'gated': False, 'min_bytes': 128 * 1024 ** 2,
        'license_url': 'https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning',
    },
    # ⚠️ A THIRD Qwen text encoder, and the three are NOT interchangeable:
    # qwen_3_8b_fp8mixed is Klein's, qwen3vl_4b_fp8_scaled is Z-Image/Krea's,
    # and this 2.5-VL 7B build is Qwen-Image-Edit's. They share a folder and a
    # prefix; a resolver matching a bare 'qwen' picks the wrong one and the
    # sampler dies on a shape mismatch. Canonical name first, narrow token after
    # — see qwen_camera_helper.resolve_camera_text_encoder.
    'camera_text_encoder': {
        'url': 'https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors',
        'dest': ('text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors'),
        'min_free_gb': 12, 'gated': False, 'min_bytes': 1024 ** 3,
        'license_url': 'https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI',
    },
}

# The one-click group, in install order. `krea_vae` is a member ON PURPOSE:
# the lane runs on the same Qwen VAE the Krea 2 lane installs, and
# camera_missing_assets reports it under that key — one file, one action,
# whichever engine asks for it first.
INSTALL_GROUP = ('camera_model', 'camera_lora', 'camera_speed_lora',
                 'camera_text_encoder', 'krea_vae')


def _camera_missing():
    return qch.camera_missing_assets()


def _camera_ready():
    # The speed LoRA missing does NOT make the lane un-ready: it renders at 20
    # steps instead of 4. Only the four REQUIRED assets gate it.
    return qch.camera_ready(qch.camera_missing_assets())


def register(ctx):
    from .routes import bp
    from .views import with_camera_pose_phrase
    from . import studio

    ctx.app.extensions['camera_angles.context'] = ctx
    ctx.register_blueprint(bp, url_prefix='/api')     # the URLs the screens already call
    ctx.register_job_handler('is_camera_studio', studio.completed, presentation={
        'title': 'Camera view', 'surface': 'Camera angles', 'engine': 'Qwen-Image-Edit',
        'cancel_scope': 'job',
    })
    for key, spec in DOWNLOADS.items():
        ctx.register_model_download(key, **spec)
    # No integrity lane yet (`camera_invalid` is not a capability): named
    # anyway so the planner reads None today and the verdicts the day the
    # validator learns these files.
    ctx.register_install_group('camera', INSTALL_GROUP,
                               missing_key='camera_missing', invalid_key='camera_invalid')
    # Under `comfyui.*`, where the Setup screens read a lane's readiness. The
    # lane has no pins, no custom-node pack and no invalid-file class — it is
    # four filenames on disk. `camera_missing` names Setup actions, so the
    # screen turns each one into the button that installs it; `camera_ready`
    # is the single verdict every surface reads, so the picker and the Setup
    # card cannot disagree about whether a view can be rendered.
    ctx.register_probe('comfyui.camera_missing', _camera_missing)
    ctx.register_probe('comfyui.camera_ready', _camera_ready)
    # The angle is the one fact the captioner cannot see: re-injected at every
    # stamp, so it survives every re-caption.
    ctx.register_hook('caption.stamp', lambda text, img: with_camera_pose_phrase(img, text))
    # The picker's Model row lists the lane's own candidate walk (qwen-named
    # folders + search-root level), so the picker and the pin agree about what
    # exists.
    ctx.register_model_slot('camera_unet', folder_type='diffusion_models',
                            hint='a qwen-named folder under ComfyUI’s models/diffusion_models',
                            candidates=qch.qwen_unet_candidates)
