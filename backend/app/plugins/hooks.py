"""Hook points the core exposes to plugins.

A hook point is a named place in the core where a plugin may transform a
value (a *filter*): the core calls :func:`run_filter` with the value and the
hook's arguments, every plugin registered on that name runs in registration
order (bundled before external), and the value the last one returned is what
the core uses. A plugin's hook that raises is logged and skipped — a plugin
must never take a core pass down.

Hook points are named by the CORE and listed here, so a plugin author has one
place to read what exists and a rename is a deliberate change:

``caption.stamp`` — ``(text, img) -> text``: the caption a dataset row keeps
    after a VLM pass, before it is stamped. The camera-angles plugin prefixes
    the row's angle phrase here, so the one fact the captioner cannot see
    survives every re-caption.
``lineage.checkpoints`` — ``(checkpoints, record_id) -> checkpoints``: the
    saves of one run as the run graph serialises them, before they leave.
    A plugin stamps its own fact on a pill here (the Civitai publisher marks
    the page a save IS), one call per run — never a request per pill.
``video_lineage.checkpoints`` — ``(rows, dataset_id, run_id, paths) -> rows``:
    the steps of one VIDEO run as the checkpoint list and its graph serialise
    them (``paths`` maps each file name to its resolved path; ``run_id`` is
    None for the local run). Same rule: the publisher stamps the version a
    step IS (``row['civitai']``), one query per run.
``training_run.delete_summary`` — ``(summary, rec) -> summary``: what a
    run's deletion will take with it, as the confirmation states it. A
    plugin adds its own count; the dialog ignores keys it does not know.
``studio.enhance_writer`` — ``(writer_or_none, profile) -> writer_or_none``:
    resolve a named writing profile for the generic Studio Enhance endpoint.
    An enabled product contributes its callable; it keeps its prompts and
    validation. The host calls ``writer(prompt, model=..., creative_direction=...)``
    inside the existing GPU window and acknowledges the requested profile.
    ``studio`` is reserved for the core; missing profiles are refused. Resolving
    a writer must not generate text or overwrite a profile already claimed.

An *event* is a hook point with no value to transform: :func:`run_hook`
calls every plugin registered on the name with the arguments and keeps
nothing back. A plugin's hook that raises is logged and skipped here too.

``training_run.delete`` — ``(rec)``: a run is about to be deleted, in its
    own transaction; a plugin detaches or drops what it keyed on the run.
``dataset.delete`` — ``(dataset_id)``: a dataset is being deleted, inside
    its transaction, before the parent row goes; a plugin deletes its own
    rows keyed on the dataset through the shared session.
"""
from __future__ import annotations

import logging

from .registry import active

log = logging.getLogger(__name__)

HOOK_POINTS = ('caption.stamp', 'lineage.checkpoints', 'video_lineage.checkpoints',
               'training_run.delete_summary', 'training_run.delete', 'dataset.delete',
               'studio.enhance_writer',
               # The queue's boot sweep of ComfyUI's input folder asks what a
               # plugin's lane still points at: `(keep: set) -> set`.
               'job_queue.keep_inputs',
               # API 1.8: pending product rows with an unimported result.
               # `(job_ids: list[str]) -> list[str]`; core rechecks terminal state.
               'job_queue.unlinked_results',
               # A ComfyUI restart asks what a plugin's lane is doing with the card:
               # `(reasons: list) -> list` of sentences; the first refuses with 409.
               'comfyui.restart_blockers',
               # Releasing cached AI memory must leave an automatic local lane
               # alone, including between its queued clips: `(reasons: list) -> list`.
               'system.free_memory_blockers',
               # Disabling a plugin asks what would be left unattended at the next
               # restart: `(reasons: list, plugin_id: str) -> list` of sentences; the
               # first refuses with 409 (a cloud run still billing, a live rental).
               'plugin.disable_blockers')


def run_filter(name: str, value, *args, strict: bool = False):
    """Pass ``value`` through every plugin's filter on ``name``. A failing
    hook is logged and skipped — unless ``strict``: then it propagates, for the
    core passes where 'the value as it was' is the wrong answer (a sweep that
    erases what a lane could not name)."""
    registry = active()
    if registry is None:
        return value
    for plugin_id, fn in registry.hooks.get(name, ()):
        try:
            if name in ('caption.stamp', 'studio.enhance_writer') and not registry.membership.check(
                    registry.records.get(plugin_id))['allowed']:
                continue
            value = fn(value, *args)
        except Exception:  # noqa: BLE001 — one plugin's hook must not take the core pass down
            if strict:
                raise
            log.exception('plugin %r: hook %r failed; value left as it was', plugin_id, name)
    return value


def run_hook(name: str, *args) -> None:
    """Announce an event to every plugin registered on ``name``."""
    registry = active()
    if registry is None:
        return
    for plugin_id, fn in registry.hooks.get(name, ()):
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 — one plugin's hook must not take the core pass down
            log.exception('plugin %r: hook %r failed', plugin_id, name)
