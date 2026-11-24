# SPDX-License-Identifier: Apache-2.0
"""Per-block attention backend split for the MiniMax H3 DiT.

Derived from `custom_nodes/h3_block_attention/__init__.py` of FastH3 Live
(jacokon, 2026, Apache License 2.0, huggingface.co/datasets/jacokon/fasth3-live),
whose measurements are the reason this node exists: on one RTX 5090 at a
15 448-token sequence, attention is 64 % of a denoising step, and `sage
attention` runs it in 14.41 ms per block against `pytorch attention`'s 40.97 —
3.24 s → 1.92 s per step, the single largest win available on one GPU.
ComfyUI does not switch the backend for you: there is no global setting, and
"Using pytorch attention" in the startup log is what that looks like.

INT8 attention kernels (comfy kitchen) and sage are fast but lose precision.
The common mitigation is to keep the most quantisation-sensitive transformer
blocks — the first and last few — on full-precision SDPA and run only the
middle stack on the fast kernel.

This node registers a per-block replace patch on the "dit" block family and
swaps transformer_options["optimized_attention_override"] around each block's
forward, so every attention call inside that block uses the chosen backend. It
does not touch any ComfyUI core file. Drop-in: place after the UNETLoader.

What changed from the original, and why (LoRA Dataset Studio, 2026-09-04):
* the class key is namespaced (see __init__.py);
* the list of backends is computed where it is needed rather than as a side
  effect of INPUT_TYPES — the original populated its `sage` entry only when
  INPUT_TYPES had run first, so a graph submitted straight to /prompt could ask
  for sage, run on pytorch, and be told "middle=sage attention" in the report;
* the report names the backend that is ACTUALLY installed for each span. A
  requested backend that is not available falls back to pytorch, and the
  report says so with "requested … ran pytorch". The app reads that string.
"""

import logging

import comfy.ldm.modules.attention as attn_mod

PYTORCH = "pytorch attention"
KITCHEN = "comfy kitchen attention"
SAGE = "sage attention"

_INTERNAL = {
    KITCHEN: "comfy_kitchen_int8",
    PYTORCH: "pytorch",
    SAGE: "sage",
}


def available_backends():
    """The backends this ComfyUI can run right now, pytorch always first."""
    out = [PYTORCH]
    if getattr(attn_mod, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False):
        out.append(KITCHEN)
    if getattr(attn_mod, "SAGE_ATTENTION_IS_AVAILABLE", False):
        out.append(SAGE)
    return out


def _resolve(display_name):
    """(attention function, the display name that will really run)."""
    internal = _INTERNAL.get(display_name)
    fn = attn_mod.get_attention_function(internal, None) if internal else None
    if fn is None or display_name not in available_backends():
        logging.warning("[lds_h3_attention] backend %r unavailable; falling back to %s.",
                        display_name, PYTORCH)
        return attn_mod.get_attention_function("pytorch"), PYTORCH
    return fn, display_name


def _make_override(fn):
    """Build an optimized_attention_override callable for one backend.

    Mirrors ModelPatcher.set_model_optimized_attention: the override is invoked
    as override(original_func, *args, **kwargs); when q/k/v arrive wrapped in
    AttentionTensorContainer the wrapper prefers override.container_function.
    """
    def override(_, *args, **kwargs):
        return fn(*args, **kwargs)

    cf = getattr(fn, "container_function", None)
    if cf is not None:
        override.container_function = cf
    return override


def _block_count(model):
    for path in ("diffusion_model.blocks", "diffusion_model.double_blocks"):
        try:
            obj = model.get_model_object(path)
            if obj is not None:
                return len(obj)
        except Exception:
            pass
    try:
        return len(model.model.diffusion_model.blocks)
    except Exception:
        return 0


def _make_block_patch(override):
    def patch(args, extra):
        to = args["transformer_options"]
        had = "optimized_attention_override" in to
        prev = to.get("optimized_attention_override")
        to["optimized_attention_override"] = override
        try:
            return extra["original_block"](args)
        finally:
            if had:
                to["optimized_attention_override"] = prev
            else:
                to.pop("optimized_attention_override", None)
    return patch


class LDSMiniMaxH3BlockAttentionSplit:
    @classmethod
    def INPUT_TYPES(cls):
        backends = available_backends()
        mid_default = SAGE if SAGE in backends else (KITCHEN if KITCHEN in backends else PYTORCH)
        return {
            "required": {
                "model": ("MODEL",),
                "edge_backend": (backends, {"default": PYTORCH,
                    "tooltip": "Backend for the first and last blocks (the quantisation-sensitive ends)."}),
                "middle_backend": (backends, {"default": mid_default,
                    "tooltip": "Backend for the middle stack (where the speed comes from)."}),
                "head_pct": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Percentage of blocks at the START kept on edge_backend."}),
                "tail_pct": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Percentage of blocks at the END kept on edge_backend."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "split_report")
    FUNCTION = "patch"
    CATEGORY = "model/patch"

    def patch(self, model, edge_backend, middle_backend, head_pct, tail_pct):
        n = _block_count(model)
        if n == 0:
            logging.warning("[lds_h3_attention] could not determine block count; model returned unchanged.")
            return (model, "block count unknown - no patch applied")

        head = int(round(n * head_pct / 100.0))
        tail = int(round(n * tail_pct / 100.0))
        head = max(0, min(head, n))
        tail = max(0, min(tail, n - head))

        edge_fn, edge_ran = _resolve(edge_backend)
        mid_fn, mid_ran = _resolve(middle_backend)
        edge_patch = _make_block_patch(_make_override(edge_fn))
        mid_patch = _make_block_patch(_make_override(mid_fn))

        m = model.clone()
        for i in range(n):
            is_edge = i < head or i >= n - tail
            m.set_model_patch_replace(edge_patch if is_edge else mid_patch, "dit", "double_block", i)

        mid = n - head - tail
        spans = []
        if head:
            spans.append(f"[0..{head-1}]")
        if tail:
            spans.append(f"[{n-tail}..{n-1}]")
        span_s = " + ".join(spans) if spans else "(none)"

        def _ran(requested, ran):
            return ran if ran == requested else f"requested {requested} ran {ran}"

        report = (f"{n} blocks | edge={_ran(edge_backend, edge_ran)}: {span_s} "
                  f"({head}+{tail}={head+tail}) | middle={_ran(middle_backend, mid_ran)}: {mid} blocks")
        logging.info("[lds_h3_attention] %s", report)
        return (m, report)
