"""✂️ Editing an image must not silently re-compress it.

Four operations rewrite the working image the trainer will later `copy2` verbatim:
mirror, rotate, crop and the watermark crop. Mirror and rotate already preserved the
source format; crop re-encoded EVERYTHING to lossy WEBP q92, so cropping a PNG
degraded it and left PNG-named files holding WEBP bytes. These tests pin the rule
they now share (`app.services.image_encoding`), pin that each one STATES its own
encoding policy rather than inheriting a global default, and — just as importantly —
pin that the throwaway derivatives were NOT dragged along with it.

Note on wording: crop also RESIZES — a box LONGER than 1024 is normalised down to a
1024 long side. It no longer ENLARGES a smaller box (that upscale invented no detail
and cost 2.3x the bytes); `upscale_ratio` still records how far under the training
resolution the box was. Downscaling is destructive by nature, so nothing here claims
a lossless CROP. What is proven lossless is the ENCODING step: the bytes on disk
decode back to exactly the crop+resize result.
"""
import io

import pytest
from PIL import Image, ImageOps

from app.services import image_encoding


# --- fixtures ----------------------------------------------------------------
def _photo(w=400, h=300, seed=11):
    """Detailed, non-flat content: a lossy encoder has something to lose on it.

    A flat colour survives q92 untouched, which would make every assertion below
    pass for the wrong reason.
    """
    im = Image.new('RGB', (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (
                (x * 7 + y * 3 + seed) % 256,
                (x * x // 5 + y * 11) % 256,
                (x * 3 + y * y // 7 + seed * 5) % 256,
            )
    return im


def _write(path, fmt, w=400, h=300):
    im = _photo(w, h)
    if fmt == 'JPEG':
        im.save(path, 'JPEG', quality=92, subsampling=0)
    elif fmt == 'WEBP':
        im.save(path, 'WEBP', quality=92)
    else:
        im.save(path, fmt)
    return im


def _reference_crop(path, box, size=1024):
    """The crop+resize result BEFORE any encoder touches it — the honest baseline.

    Mirrors the service's cap: the long side is normalised DOWN to `size`, never up."""
    with Image.open(path) as src:
        src.load()
        work = src.convert(image_encoding.resample_mode(src))
    bw, bh = box[2] - box[0], box[3] - box[1]
    size = min(size, max(bw, bh))
    if bw >= bh:
        out = (size, max(1, round(size * bh / bw)))
    else:
        out = (max(1, round(size * bw / bh)), size)
    return work.crop(box).resize(out, Image.LANCZOS)


def _decoded(path):
    with Image.open(path) as im:
        im.load()
        return (im.format or '').upper(), im.convert('RGB').tobytes(), im.size


def test_visual_header_size_swaps_exif_axes_without_loading_pixels():
    """Route/poll geometry must not full-decode a master just to read W×H."""
    class _HeaderOnly:
        size = (1600, 1000)

        def getexif(self):
            return {274: 6}

        def load(self):
            raise AssertionError('visual-size helper must remain header-only')

    assert image_encoding.visual_size_from_header(_HeaderOnly()) == (1000, 1600)


def test_live_source_transform_uses_bounded_exact_bytes(tmp_path, monkeypatch):
    """A Bank rotation must decode the bytes it bounded, not reopen its live path."""
    from app.services import face_dataset_service as svc

    path = tmp_path / 'live.png'
    _write(path, 'PNG', 24, 18)
    opened = []
    real_open = svc.Image.open

    def _opened_from_bytes(source, *args, **kwargs):
        opened.append(source)
        assert isinstance(source, io.BytesIO)
        return real_open(source, *args, **kwargs)

    monkeypatch.setattr(svc.Image, 'open', _opened_from_bytes)
    payload = svc.transformed_image_bytes(
        path, ImageOps.mirror, max_source_bytes=1024 * 1024)
    assert opened and payload
    with pytest.raises(ValueError, match='too large'):
        svc.transformed_image_bytes(path, ImageOps.mirror, max_source_bytes=1)


def _seed_image(app, filename, fmt, w=400, h=300):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, f'Crop {filename}', f'crop-{filename}')
        import os
        path = os.path.join(svc._dataset_dir(ds.id), filename)
        _write(path, fmt, w, h)
        row = FaceDatasetImage(dataset_id=ds.id, filename=filename,
                               source='import', status='keep')
        svc.db.session.add(row)
        svc.db.session.commit()
        return ds.id, row.id, path


# --- the rule ----------------------------------------------------------------
@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'shot.png'),
    ('WEBP', 'shot.webp'),
    ('BMP', 'shot.bmp'),
])
def test_crop_encoding_is_lossless_for_lossless_sources(app, fmt, filename):
    """A PNG cropped stays a PNG, and the bytes on disk decode back to the exact
    crop+resize result. Before the fix this wrote WEBP q92 into a .png file."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, filename, fmt)
    box = (40, 30, 280, 210)
    expected = _reference_crop(path, box)
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, box[0], box[1],
                              box[2] - box[0], box[3] - box[1]) is True

    detected, pixels, size = _decoded(path)
    assert detected == fmt, f'crop changed the format to {detected}'
    assert size == expected.size
    assert pixels == expected.convert('RGB').tobytes(), 'the encoder lost pixels'


def test_crop_of_a_jpeg_stays_a_jpeg_and_loses_less_than_before(app):
    """JPEG has no lossless mode — we do NOT pretend otherwise.

    The source stays a JPEG (turning a user's JPEG into a 2.4x heavier lossless file
    to protect pixels that were already lossy is a bad trade), re-encoded at q95 with
    no chroma subsampling. What is asserted is that the residual error is far below
    what the old always-WEBP-q92 path left behind.
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.jpg', 'JPEG')
    box = (40, 30, 280, 210)
    expected = _reference_crop(path, box).convert('RGB')
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, box[0], box[1],
                              box[2] - box[0], box[3] - box[1]) is True

    detected, pixels, size = _decoded(path)
    assert detected == 'JPEG'
    assert size == expected.size

    ref = expected.tobytes()
    kept = sum(abs(a - b) for a, b in zip(ref, pixels)) / len(ref)

    old = io.BytesIO()
    expected.save(old, 'WEBP', quality=92)
    with Image.open(old) as im:
        im.load()
        old_err = sum(abs(a - b) for a, b in zip(ref, im.convert('RGB').tobytes())) / len(ref)

    assert kept < old_err, (f'q95 4:4:4 JPEG should beat the old WEBP q92 path '
                            f'({kept:.3f} vs {old_err:.3f})')


def test_repeated_crops_of_a_webp_never_accumulate_damage(app):
    """The failure mode that reaches the LoRA weights: curate, re-crop, re-crop.

    Under q92 each pass re-compressed the previous compression. Lossless WEBP makes
    a second full-frame crop byte-for-byte identical to the first one's output.
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.webp', 'WEBP')
    with app.app_context():
        svc.crop_image(LOCAL_USER, image_id, 0, 0, 400, 300)
    once = open(path, 'rb').read()
    with Image.open(path) as im:
        w, h = im.size
    with app.app_context():
        svc.crop_image(LOCAL_USER, image_id, 0, 0, w, h)
    assert open(path, 'rb').read() == once, 're-cropping the full frame changed bytes'


def test_cropped_file_extension_still_matches_its_real_content(app):
    """A `.png` holding WEBP bytes is a trap for everything that guesses by name —
    and the import path writes the TRUE extension, so the trap was real."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    for filename, fmt in (('shot.png', 'PNG'), ('shot.jpg', 'JPEG'),
                          ('shot.webp', 'WEBP'), ('shot.bmp', 'BMP')):
        _ds, image_id, path = _seed_image(app, filename, fmt)
        with app.app_context():
            svc.crop_image(LOCAL_USER, image_id, 10, 10, 200, 150)
        detected, _pixels, _size = _decoded(path)
        ext = '.' + filename.rsplit('.', 1)[1]
        assert ext in image_encoding.FORMAT_EXTENSIONS[detected], (
            f'{filename} now contains {detected} bytes')


def test_a_small_crop_keeps_its_own_pixels_instead_of_being_enlarged(app):
    """The resize used to be unconditional: a 240x180 box came out 1024x768.

    That enlargement carried almost nothing — shrinking the enlarged result back to
    240 recovers the original at 48.96 dB (max channel error 10) — for 2.3x the bytes,
    which since the lossless switch means ~1 MB of interpolated pixels per small crop.
    A crop now never exceeds the size of what was actually cropped.

    `upscale_ratio` is UNCHANGED in value and meaning as a stored number (target /
    long side, i.e. how far under the training resolution this tile is); only the
    pixels stop pretending. It is a stored column read by dataset_payload, so its
    scale must not drift."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.webp', 'WEBP')
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, 40, 30, 240, 180) is True
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert row.upscale_ratio == pytest.approx(1024 / 240)
    with Image.open(path) as im:
        assert im.size == (240, 180)


def test_a_crop_longer_than_1024_is_still_normalised_down(app):
    """The guard-rail on the fix above: removing the ENLARGEMENT must not remove the
    normalisation. A 2000x1500 box still lands on a 1024 long side, aspect preserved."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'big.webp', 'WEBP', w=2400, h=1800)
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, 100, 100, 2000, 1500) is True
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert row.upscale_ratio == pytest.approx(1024 / 2000)
    with Image.open(path) as im:
        assert im.size == (1024, 768)


def test_crop_of_a_bmp_stays_a_lossless_bmp(app):
    """BMP is a supported dataset format, not a legacy exception: crop keeps a
    BMP-named file as real BMP bytes and preserves every remaining RGB pixel."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    import os

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Exotic', 'exotic')
        path = os.path.join(svc._dataset_dir(ds.id), 'legacy.bmp')
        _photo().save(path, 'BMP')
        row = FaceDatasetImage(dataset_id=ds.id, filename='legacy.bmp',
                               source='import', status='keep')
        svc.db.session.add(row)
        svc.db.session.commit()
        image_id = row.id
    box = (40, 30, 280, 210)
    expected = _reference_crop(path, box)
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, box[0], box[1],
                              box[2] - box[0], box[3] - box[1]) is True
    detected, pixels, _size = _decoded(path)
    assert detected == 'BMP'
    assert pixels == expected.convert('RGB').tobytes()


def test_watermark_crop_is_byte_exact_because_it_never_resizes(app):
    """`_apply_watermark_crop` invents no pixel by design (no resize at all), so here
    'lossless' is the whole operation, not just its encoding."""
    from app.services import face_dataset_service as svc

    _ds, _image_id, path = _seed_image(app, 'shot.png', 'PNG')
    with Image.open(path) as im:
        im.load()
        expected = im.crop((20, 15, 300, 240)).convert('RGB').tobytes()
    assert svc._apply_watermark_crop(path, (20, 15, 300, 240)) is True
    detected, pixels, size = _decoded(path)
    assert detected == 'PNG' and size == (280, 225)
    assert pixels == expected


def test_manual_crop_uses_exif_oriented_browser_coordinates(app):
    """The crop widget sees an upright JPEG, so its coordinates must too."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'camera.jpg', 'JPEG', w=80, h=40)
    source = _photo(80, 40)
    exif = source.getexif()
    exif[274] = 6
    source.save(path, 'JPEG', quality=95, subsampling=0, exif=exif)

    # Visual image is 40×80. A full-height 20px strip must remain 20×80;
    # applying this to raw dimensions would incorrectly return 20×40.
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, 0, 0, 20, 80) is True

    with Image.open(path) as cropped:
        cropped.load()
        assert cropped.format == 'JPEG' and cropped.size == (20, 80)
        assert not cropped.getexif()


def test_direct_watermark_crop_uses_exif_oriented_visual_box(app):
    """The crop helper shares the VLM/browser orientation contract on its own."""
    from app.services import face_dataset_service as svc

    _ds, _image_id, path = _seed_image(app, 'camera.jpg', 'JPEG', w=80, h=40)
    source = _photo(80, 40)
    exif = source.getexif()
    exif[274] = 6
    source.save(path, 'JPEG', quality=95, subsampling=0, exif=exif)

    assert svc._apply_watermark_crop(path, (0, 0, 20, 80)) is True
    with Image.open(path) as cropped:
        cropped.load()
        assert cropped.format == 'JPEG' and cropped.size == (20, 80)
        assert not cropped.getexif()


@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'turn.png'),
    ('WEBP', 'turn.webp'),
])
def test_mirror_and_rotation_come_back_byte_identical(app, fmt, filename):
    """A PUBLISHED promise, not an internal nicety: GitHub #17 was answered with
    "PNG and WEBP come back byte-identical after four quarter turns". Nothing in
    this consolidation may weaken it, so it is pinned on BYTES, not pixels — a
    future tweak aimed at the crop's encoder would have to break this test to reach
    the mirror or the rotation.

    (JPEG is excluded on purpose: Pillow has no DCT-domain path, so every turn of a
    JPEG is a re-encode. The app says so rather than pretending.)
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, filename, fmt)
    with app.app_context():
        svc.mirror_image(LOCAL_USER, image_id)     # normalise through the encoder once
    settled = open(path, 'rb').read()

    with app.app_context():
        svc.mirror_image(LOCAL_USER, image_id)
        svc.mirror_image(LOCAL_USER, image_id)
    assert open(path, 'rb').read() == settled, 'two mirrors did not restore the file'

    with app.app_context():
        for _ in range(4):
            svc.rotate_image(LOCAL_USER, image_id, 90)
    assert open(path, 'rb').read() == settled, 'four quarter turns did not restore the file'


def test_every_edit_states_its_own_encoding_policy(app):
    """The policy is an argument, never a module default: mirror/rotate carry the
    byte-identity promise and crop does not, so the two must stay independently
    tunable. Pin that every call site names a policy, and that none of them silently
    picked the lossy one."""
    import pathlib
    import re

    def arguments(text, start):
        """The call's argument list, counting nested parentheses (the format is
        chosen by a nested `format_for_path(...)` at two of the call sites)."""
        depth, out = 0, []
        for ch in text[start:]:
            if ch == '(':
                depth += 1
                if depth == 1:
                    continue
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
        return ''.join(out)

    root = pathlib.Path(__file__).resolve().parents[1]
    sites = []
    for rel in ('app/services/face_dataset_service.py',
                'app/services/image_bank_service.py',   # the Bank's own three: clean, crop, improve
                'app/services/watermark_klein.py',
                'infer/lama_infer.py'):
        text = (root / rel).read_text(encoding='utf-8')
        for call in re.finditer(r'image_encoding\.save_(?:edit|params)\s*\(', text):
            sites.append((rel, arguments(text, call.end() - 1)))

    assert len(sites) >= 4, f'expected every edit to route through the module, got {sites}'
    for rel, args in sites:
        assert 'image_encoding.LOSSLESS' in args, (
            f'{rel}: this edit does not state the LOSSLESS policy — {args!r}')
        assert 'HIGH_QUALITY' not in args, f'{rel}: an edit silently went lossy'


def test_the_escape_hatch_policy_is_real_and_unknown_policies_are_refused():
    """HIGH_QUALITY exists so an operation CAN opt out of the size cost, and the
    module docstring records the measurement that keeps it unused. Prove it is a
    working lossy branch rather than a decorative constant — and that a typo in a
    policy name fails loudly instead of silently re-compressing a dataset."""
    ref = _photo(240, 180)

    exact = io.BytesIO()
    image_encoding.save_edit(ref, exact, 'WEBP', image_encoding.LOSSLESS)
    cheap = io.BytesIO()
    image_encoding.save_edit(ref, cheap, 'WEBP', image_encoding.HIGH_QUALITY)

    with Image.open(exact) as im:
        im.load()
        assert im.convert('RGB').tobytes() == ref.tobytes()
    with Image.open(cheap) as im:
        im.load()
        assert im.convert('RGB').tobytes() != ref.tobytes(), 'HIGH_QUALITY is not lossy'
    # No size assertion here on purpose: which branch is SMALLER depends entirely on
    # the content (this synthetic fixture is high-frequency and compresses better
    # losslessly than lossy, the opposite of a photograph). The measurement that
    # matters is in the module docstring, on a real photograph.

    with pytest.raises(ValueError, match='unknown encoding policy'):
        image_encoding.save_params(ref, 'WEBP', 'lossles')


def test_mirror_and_crop_share_one_encoder(app):
    """The point of the wave: one rule, not three copies of it. Mirroring then
    cropping a PNG must never produce a format neither of them chose."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.png', 'PNG')
    with app.app_context():
        svc.mirror_image(LOCAL_USER, image_id)
        assert _decoded(path)[0] == 'PNG'
        svc.crop_image(LOCAL_USER, image_id, 10, 10, 200, 150)
    assert _decoded(path)[0] == 'PNG'


# --- the guardrail: derivatives stay cheap -----------------------------------
def test_bank_thumbnails_stay_lossy_and_small(app, client, tmp_path):
    """Do NOT drag the derivatives into the lossless rule while doing good.

    A grid thumbnail is regenerated from the working image at will; inflating it 4x
    would cost a lot of disk for pixels nobody trains on. It must still be the q72
    WEBP it always was.
    """
    import os
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as bank

    src = tmp_path / 'thumb-src'
    os.makedirs(src, exist_ok=True)
    _photo(1200, 900).save(str(src / 'a.png'))

    response = client.post('/api/bank/create', json={'name': 'Thumbs', 'folder': str(src)})
    assert response.status_code == 200, response.get_json()
    bank_id = response.get_json()['id']

    with app.app_context():
        b = db.session.get(ImageBank, bank_id)
        row = BankImage.query.filter_by(bank_id=bank_id).first()
        assert row is not None, 'the bank should have inventoried the file'
        tpath = bank.ensure_thumb(b, row)
        assert tpath is not None and tpath.is_file()

        with Image.open(tpath) as t:
            assert (t.format or '').upper() == 'WEBP'
            t.load()
            probe = io.BytesIO()
            t.convert('RGB').save(probe, 'WEBP', lossless=True, quality=100, method=4)
        assert tpath.stat().st_size * 2 < len(probe.getvalue()), (
            'the thumbnail looks lossless — a derivative was inflated by mistake')


def test_encoding_module_stays_importable_without_the_app(app):
    """`backend/infer/lama_infer.py` runs under the dedicated ML interpreter, where
    the Flask app package does not import. It loads this module by file path, so the
    module must never grow an app/db/config dependency."""
    import ast
    import pathlib

    path = pathlib.Path(image_encoding.__file__)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or '').split('.')[0] or '<relative>')
    assert imported <= {'PIL', '__future__'}, f'unexpected dependency: {imported}'


def test_lama_worker_can_load_the_encoder_the_way_it_actually_does():
    """The worker resolves the module by FILE PATH (`load_image_encoding`), because it
    runs under an interpreter this suite never boots. Exercise that exact bootstrap in
    a clean subprocess: if the file moves, watermark inpainting would die at import
    time on a machine we cannot test here — and only there."""
    import pathlib
    import subprocess
    import sys

    worker = pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'lama_infer.py'
    probe = (
        'import importlib.util, sys\n'
        f'spec = importlib.util.spec_from_file_location("w", r"{worker}")\n'
        'w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n'
        'e = w.load_image_encoding()\n'
        'print(e.format_for_path("a.png"), e.save_edit.__name__)\n'
        # The bootstrap must not leave the services directory on sys.path: it would
        # shadow whatever torch/simple-lama import next.
        'assert not any("services" in p for p in sys.path), sys.path\n'
    )
    done = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'PNG save_edit'
