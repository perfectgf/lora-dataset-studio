"""Seed the SHOWCASE instance — the one the public screenshots are taken from.

Why this exists
---------------
Every screenshot in `docs/screenshots/` and in a release body is a picture of
somebody's data. The maintainer's own banks are NSFW and are out of bounds for
anything public, cropped or not, so the honest source has always been a
throwaway instance built for the occasion. Built per release, that instance is
always empty and always ugly: two datasets, no images, nothing to photograph.
And an ugly demo set is not a cosmetic problem — a public gallery drawn from a
single repeated identity was read as "this tool produces identical faces", which
is the opposite of what a dataset tool wants to say.

So the showcase is kept instead of rebuilt. What is expensive is not the server,
it is the CURATED SET: several distinct identities, real variety of framing and
light. This script builds that set once into a data directory of its own; the
server is started only when a picture is needed, against the tag being released.

Usage
-----
    python scripts/seed_showcase.py --data-dir <dir> --images <dir> --init
    python scripts/seed_showcase.py --data-dir <dir> --images <dir>   # top up

`--images` holds one SUBFOLDER PER IDENTITY; the folder name becomes the dataset
name and its trigger word. Nothing is generated here — see the companion
generator for that; this only imports what you point it at.

The guard
---------
This writes datasets into a live data directory, so pointing it at a real one by
accident would be unrecoverable. `--init` stamps `showcase.json` in the target,
and every later run REQUIRES that stamp. A directory that was not initialised as
a showcase is refused, and so are the two paths that are known to be real
installs. That is a structural refusal, not a warning: the mistake it prevents
has no undo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MARKER = 'showcase.json'

# Data directories that are known to belong to a real install. The stamp check
# below already refuses them (they carry no marker); naming them explicitly buys
# a message that says WHICH mistake was made instead of a generic refusal.
KNOWN_REAL = ('projects/test/data', 'lora-dataset-studio/data')

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}

# A dataset's Composition meter counts `framing` over the kept images, and
# nothing fills that column on import: the app classifies framings with a
# Qwen3-VL pass through Ollama. A showcase machine has no vision model — the
# reference crop below already works around the same absence — so every seeded
# image landed at framing=None and Curation showed 0 face / 0 bust / 0 body /
# 0 back on a set built expressly to fill it.
#
# `import_images` takes a `framings=` list for exactly this case ("a framing
# ALREADY known for the blob… so it lands counted in the composition instead of
# sitting at 0 until something re-classifies it"). For a demo set the framing IS
# known — it was commissioned shot by shot — so it travels in the file name.
# A name that says nothing yields None, which is the same graceful state as
# before: the classifier can still fill it in on a machine that has one.
FRAMING_TOKENS = {
    'portrait': 'face', 'profile': 'face',
    'bust': 'bust',
    'full': 'body',
    'back': 'back',
}


def framing_of(path: Path) -> str | None:
    """The catalog framing a demo file names, or None if it names none."""
    tokens = set(path.stem.lower().split('_'))
    matched = {FRAMING_TOKENS[t] for t in tokens if t in FRAMING_TOKENS}
    # Two framings in one name is a naming mistake, not a blend — refusing to
    # guess leaves the column empty rather than counting the shot in the wrong
    # bucket, and a wrong bucket is worse than an absent one on a meter whose
    # whole job is to say what the set is missing.
    return matched.pop() if len(matched) == 1 else None


def _refuse(message: str) -> None:
    print(f'REFUSED: {message}', file=sys.stderr)
    raise SystemExit(2)


def check_target(data_dir: Path, *, init: bool) -> None:
    """Refuse anything that is not, or is not becoming, a showcase directory."""
    normalised = data_dir.as_posix().lower()
    for real in KNOWN_REAL:
        if normalised.endswith(real):
            _refuse(f'{data_dir} is a real install ({real}). The showcase needs '
                    f'its own directory — never seed into data somebody uses.')
    # The showcase's own path is PUBLIC: the workspace prints "Images folder
    # <path>" in a banner above the grid, so it lands in any screenshot taken
    # there. A path under a user profile therefore publishes the account name.
    # Measured, not guessed — the first showcase shot carried it in plain sight.
    if '/users/' in normalised or '/home/' in normalised:
        print(f'WARNING: {data_dir} contains a user profile. The app prints the '
              f'images folder in a banner, so that path will appear in your '
              f'screenshots. Prefer something neutral like C:/lds-showcase.',
              file=sys.stderr)
    marker = data_dir / MARKER
    if init:
        if marker.exists():
            print(f'already initialised: {marker}')
        return
    if not marker.exists():
        _refuse(f'{data_dir} carries no {MARKER}. Run once with --init if this '
                f'really is the showcase directory; if it is not, you just '
                f'avoided writing datasets into somebody\'s install.')


def stamp(data_dir: Path) -> None:
    (data_dir / MARKER).write_text(json.dumps({
        'showcase': True,
        'why': 'Generated demo data for public screenshots. Never real images.',
    }, indent=2), encoding='utf-8')


def identities(images_dir: Path):
    """One subfolder per identity, each holding that identity's images."""
    for child in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in child.iterdir()
                       if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            yield child.name, files


def set_reference(svc, dataset_id: int, raw: bytes) -> None:
    """Give the dataset its reference photo, the way the app does."""
    import uuid
    from app.config import LOCAL_USER
    from app.routes.datasets import ensure_dataset_dir

    with svc.reference_mutation(dataset_id):
        dsdir = ensure_dataset_dir(dataset_id)
        original = f'{LOCAL_USER}_datasetreforig_{uuid.uuid4().hex[:8]}.webp'
        shown = f'{LOCAL_USER}_datasetref_{uuid.uuid4().hex[:8]}.webp'
        svc.write_image_atomic(os.path.join(dsdir, original),
                               svc.normalize_to_webp(raw, size=2048))
        svc.write_image_atomic(os.path.join(dsdir, shown),
                               svc.normalize_to_webp(raw, size=1024))
        ds = svc.get_dataset(LOCAL_USER, dataset_id)
        ds.ref_original_filename = original
        ds.ref_filename = shown
        svc.db.session.commit()


def seed(data_dir: Path, images_dir: Path) -> tuple[int, int]:
    # Imported here, after the guard has run: importing the app creates the
    # database, so a refused target must never get this far.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))
    from app import create_app                                  # noqa: E402
    from app.config import LOCAL_USER                            # noqa: E402
    from app.services import face_dataset_service as svc         # noqa: E402

    app = create_app()
    total = skipped = 0
    with app.app_context():
        existing = {d.name for d in svc.list_datasets(LOCAL_USER)}
        for name, files in identities(images_dir):
            if name in existing:
                print(f'  {name}: already there, skipped')
                skipped += 1
                continue
            ds = svc.create_dataset(LOCAL_USER, name, name.lower().replace(' ', '_'))
            # RAW BYTES, one per image — `import_images` enumerates the list
            # itself and reads each blob's header. A (name, bytes) pair is
            # silently counted as a refusal, which is how the first run of this
            # script imported nothing while reporting success.
            payload = [f.read_bytes() for f in files]
            framings = [framing_of(f) for f in files]
            ids, failed = svc.import_images(LOCAL_USER, ds.id, payload,
                                            crop=False, framings=framings)
            # The library tile IS the reference photo — a dataset without one
            # renders as a coloured letter, which is not what a showcase is for.
            # Same two files the /ref route writes (full-frame original + the
            # displayed reference), through the same helpers, minus the auto
            # head-crop: that one needs an Ollama vision model, and a showcase
            # must seed on a machine that has none.
            if payload:
                set_reference(svc, ds.id, payload[0])
            if failed:
                # `import_images` refuses a blob it cannot decode as an image.
                # It does NOT apply the 768px short-side floor — that belongs to
                # the scraper's import path, which is a different door. Measured,
                # after this comment first claimed the opposite: 200x200 PNGs
                # import here without complaint.
                print(f'    ({failed} refused — unreadable as images?)')
            # Print the composition, because it is the screen this set exists to
            # photograph and an all-None spread is invisible otherwise — the run
            # reports 12 imported either way, and only Curation shows the zeros.
            counted = [f for f in framings if f]
            spread = ', '.join(f'{k} {counted.count(k)}'
                               for k in ('face', 'bust', 'body', 'back')
                               if counted.count(k))
            print(f'  {name}: {len(ids)} imported, {failed} refused'
                  + (f' — composition: {spread}' if spread
                     else ' — WARNING: no framing in any file name, the '
                          'Composition meter will read 0 everywhere'))
            total += len(ids)
    return total, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-dir', required=True, type=Path,
                    help='the showcase LDS_DATA_DIR (never a real install)')
    ap.add_argument('--images', required=True, type=Path,
                    help='folder holding one subfolder per identity')
    ap.add_argument('--init', action='store_true',
                    help='stamp this directory as the showcase (first run only)')
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    images_dir = args.images.resolve()
    if not images_dir.is_dir():
        _refuse(f'{images_dir} is not a folder of identities')

    data_dir.mkdir(parents=True, exist_ok=True)
    check_target(data_dir, init=args.init)
    if args.init:
        stamp(data_dir)

    # The app reads its home from the environment, so this must be set before
    # create_app() runs — which is why seed() imports it lazily.
    os.environ['LDS_DATA_DIR'] = str(data_dir)
    os.environ.setdefault('LDS_NO_REEXEC', '1')

    found = list(identities(images_dir))
    if not found:
        _refuse(f'{images_dir} holds no identity subfolder with images in it')
    print(f'seeding {len(found)} identities into {data_dir}')
    total, skipped = seed(data_dir, images_dir)
    print(f'done: {total} images')
    # A run that found identities, imported NOTHING, and skipped nothing has
    # failed — whatever it printed. The first version of this script reported
    # exactly that and exited 0, because the payload shape was wrong and every
    # blob was counted as a refusal. A seeder that succeeds silently at doing
    # nothing is worse than one that crashes.
    if total == 0 and skipped == 0:
        _refuse('nothing was imported. The identities were found but every file '
                'was refused — they are probably not decodable images.')


if __name__ == '__main__':
    main()
