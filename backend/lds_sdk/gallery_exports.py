"""Owned finished gallery images and their explicitly persisted generation facts.

A derived picture (a camera angle, an upscale, an improve, or a chain of them)
keeps its OWN columns for the pass that produced it: the viewer shows them as
"Made with", so a creator never copies the source's base model or LoRAs onto
it. What it was made FROM is still the source render's generation, and that is
what an export names: its LoRA, run and step, base model and applied LoRAs are
read from the render at the start of its lineage (`_GENERATION_FIELDS`)."""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .images import GalleryImages

__all__ = ['GalleryExportRecord', 'GalleryExports', 'prompt_with_trigger']

# Which generation a picture comes from. Every derivation copies the first four
# from its source at creation; the base model and the applied LoRAs it does not.
_GENERATION_FIELDS = ('checkpoint', 'strength', 'record_id', 'step', 'z_model', 'extra_loras')
_MAX_LINEAGE = 32


@dataclass(frozen=True)
class GalleryExportRecord:
    id: int
    dataset_id: int
    filename: str
    status: str
    record_id: int | None
    step: int | None
    checkpoint: str | None
    prompt: str | None
    inject_trigger: bool | None
    negative: str | None
    cfg: float | None
    steps: int | None
    sampler: str | None
    scheduler: str | None
    seed: int | None
    z_model: str | None
    strength: float | None
    extra_loras: str | None


class GalleryExports:
    def __init__(self, user_id):
        self._images = GalleryImages(user_id)

    def path(self, image_id):
        return self._path(self._images._row(image_id))

    @staticmethod
    def _path(row):
        """A current owned media file, confined to its dataset after symlink resolution."""
        if row is None or not row.filename:
            return None
        name = str(row.filename)
        if (Path(name).is_absolute() or PureWindowsPath(name).drive or ':' in name
                or '..' in name.replace('\\', '/').split('/')):
            return None
        from app.services.face_dataset_service import _dataset_dir
        try:
            root = Path(_dataset_dir(row.dataset_id)).resolve()
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                return None
            return str(path)
        except (OSError, ValueError, RuntimeError):
            return None

    def finished(self, image_ids):
        rows = (self._images._row(image_id) for image_id in image_ids)
        return tuple(self._record(row)
                     for row in rows if row is not None and row.status == 'done' and self._path(row))

    def _record(self, row):
        facts = {key: getattr(row, key) for key in GalleryExportRecord.__dataclass_fields__}
        origin = self._generation_origin(row)
        if origin is not None:
            facts.update({key: getattr(origin, key) for key in _GENERATION_FIELDS})
        return GalleryExportRecord(**facts)

    def _generation_origin(self, row):
        """The render a derived picture comes from, through every derivation in
        between, or None: for a picture that is not derived, and for a lineage
        that cannot be trusted. A link is followed only to an owned picture of
        the same dataset, created before it (ids only grow) and stamped with the
        same LoRA, which every derivation copies from its source: a pointer that
        is missing, stale or looping stops the walk, and the picture keeps its
        own facts."""
        current, seen = row, set()
        while getattr(current, 'derivation_kind', None) and getattr(current, 'parent_image_id', None):
            if current.id in seen or len(seen) >= _MAX_LINEAGE:
                return None
            seen.add(current.id)
            parent = self._images._row(current.parent_image_id)
            if (parent is None or parent.dataset_id != current.dataset_id
                    or not parent.id < current.id or parent.checkpoint != current.checkpoint):
                return None
            current = parent
        return None if current is row else current


def prompt_with_trigger(prompt, trigger_word):
    from app.services.lora_test_studio import _prompt_with_trigger
    return _prompt_with_trigger(prompt, trigger_word)
