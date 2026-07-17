"""On-demand scan of the ComfyUI loras roots for the Klein generation-LoRA picker.

The Settings "Klein generation LoRA presets" card (and the workspace preset
editor) used to take each LoRA as free text. This backs a real picker of the
LoRAs ACTUALLY on disk so the user stops guessing filenames.

WHY a recursive multi-root scan: LoRAs are NOT sorted into a ``loras/klein/``
subfolder on every install — a shared / portable ComfyUI mixes every family in
``models/loras`` and may register extra locations in ``extra_model_paths.yaml``.
So we scan EVERY loras search root recursively, reusing
``comfy_model_paths.list_models('loras')`` — the faithful mirror of ComfyUI's
``folder_paths.get_filename_list``. Its ``rel_name`` (``os.path.relpath`` from the
root, subfolders included, OS separator) is EXACTLY:
  * the string a ``LoraLoader`` / ``LoraLoaderModelOnly`` node expects for
    ``lora_name`` (it matches ComfyUI's own COMBO enum, which is built the same
    way), and
  * the value ``klein_edit_helper`` resolves back to an absolute path
    (``_lora_abs`` joins it onto a search root).
So ``picker == node enum == resolver`` — the value the picker emits is the exact
value that resolves at generate time. We emit it verbatim (no separator
rewriting): the resolver additionally tolerates a ``/`` form, but the canonical
on-disk form is what ComfyUI's enum validates against.

WHY an architecture badge instead of folder-based grouping: since folder can't
tell us the family, each LoRA is badged from its OWN header via
``lora_training.detect_lora_arch`` (metadata + tensor-name signatures, header-only
read, cached per file). A LoRA whose arch's key namespace differs from Klein's
(SDXL / Krea / Z-Image) loads as a SILENT no-op in the Klein graph — ComfyUI drops
every incompatible key and the edit renders as if the LoRA were off. Klein shares
its key namespace with FLUX.1, so a ``flux`` LoRA is marked compatible too (a
wrong FLUX version fails LOUDLY on a shape error, never silently). The single
compatibility rule lives in ``lora_training.lora_arch_conflicts`` — reused here so
the picker's badge and the deploy/Studio guardrail can never disagree.

Cache: the assembled list is cached by a cheap change-signature of the loras
search roots (each root's mtime), so repeated calls (a Settings re-render, several
preset rows) don't re-walk the tree. Never scanned at boot — only when the
endpoint is hit. ``force=True`` (the ↻ rescan button) bypasses the cache and picks
up a LoRA added deep in a subfolder, which a root-level mtime can't see.
"""
from __future__ import annotations

import logging
import os
import threading

from . import comfy_model_paths
from .lora_training import detect_lora_arch, lora_arch_conflicts

logger = logging.getLogger(__name__)

# The family the Klein generation graph runs. A LoRA is "Klein-compatible" when
# its detected arch shares Klein's key namespace (see lora_arch_conflicts).
_KLEIN_FAMILY = 'flux2klein'

# Presentation labels for the badge — a superset-safe copy of the family keys
# detect_lora_arch can return. Kept local (presentation, not logic): the
# compatibility DECISION is delegated to lora_arch_conflicts so there is one
# source of truth for it.
_ARCH_LABEL = {
    'flux2klein': 'FLUX.2 Klein',
    'flux': 'FLUX.1',
    'sdxl': 'SDXL',
    'krea': 'Krea 2',
    'zimage': 'Z-Image',
}

# Sort order of the three compatibility buckets: Klein-compatible first, then the
# undetermined ones, then the positively-incompatible ones last.
_COMPAT_RANK = {'yes': 0, 'unknown': 1, 'no': 2}

_lock = threading.Lock()
_cache: dict = {'sig': None, 'data': None}


def _roots_signature() -> tuple:
    """A cheap change-signature of the loras search roots: ``(root, mtime)`` per
    root, so the scan re-runs only when a root's own listing changed. Mirrors
    comfy_model_paths' mtime-cache philosophy. A missing/unreadable root contributes
    ``(root, None)`` so it still participates (and re-validates when it appears)."""
    sig = []
    for root in comfy_model_paths.search_roots('loras'):
        try:
            sig.append((root, os.path.getmtime(root)))
        except OSError:
            sig.append((root, None))
    return tuple(sig)


def _compatibility(arch) -> str:
    """'yes' | 'no' | 'unknown' for the Klein graph. ``None`` (undetectable header)
    is 'unknown' — never blocked, but not vouched for; a positively-detected arch in
    a different key namespace is 'no' (a silent no-op in the Klein graph)."""
    if arch is None:
        return 'unknown'
    return 'no' if lora_arch_conflicts(arch, _KLEIN_FAMILY) else 'yes'


def scan_generation_loras(force: bool = False) -> list:
    """``[{name, arch, label, compatible}]`` for every LoRA on disk across the loras
    search roots (base ``models/loras`` + every ``extra_model_paths.yaml`` root),
    recursively, sorted Klein-compatible first then case-insensitive by name.

      * ``name``       — the ComfyUI-relative loader string (what a preset row stores
                         and what resolves at generate time).
      * ``arch``       — detected family key ('flux2klein'|'flux'|'sdxl'|'krea'|
                         'zimage') or ``None`` when the header is undetectable.
      * ``label``      — human arch label for the badge, or ``None``.
      * ``compatible`` — 'yes' | 'no' | 'unknown' (see _compatibility).

    ``[]`` when no loras root exists (ComfyUI unconfigured) — the caller then keeps
    the field as free text. Cached by the roots' mtime; ``force`` bypasses it."""
    sig = _roots_signature()
    if not force:
        with _lock:
            if _cache['sig'] == sig and _cache['data'] is not None:
                return _cache['data']
    out = []
    for rel, abs_path in comfy_model_paths.list_models('loras'):
        arch = detect_lora_arch(abs_path)
        out.append({
            'name': rel,
            'arch': arch,
            'label': _ARCH_LABEL.get(arch),
            'compatible': _compatibility(arch),
        })
    out.sort(key=lambda e: (_COMPAT_RANK[e['compatible']], e['name'].lower()))
    with _lock:
        _cache['sig'] = sig
        _cache['data'] = out
    return out


def clear_cache() -> None:
    """Drop the scan cache (test hygiene; production self-invalidates on the mtime
    signature)."""
    with _lock:
        _cache['sig'] = None
        _cache['data'] = None
