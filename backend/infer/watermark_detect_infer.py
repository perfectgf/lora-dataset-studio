"""Dedicated overlaid-watermark detector — the fast, discriminative replacement
for asking a chat model "is there a watermark?" 29 000 times.

Runs in the DEDICATED "watermark detector" interpreter (torch/transformers are not
in the Flask venv), same subprocess family as bank_score_infer.py.

Two models, two jobs, both permissively licensed — that is not an accident but the
whole reason this cascade exists (see LICENSES below):

  * RANK  — ``prithivMLmods/Watermark-Detection-SigLIP2`` (Apache-2.0). A SigLIP2
    image classifier fine-tuned on 22 762 images for the single binary question
    "is there an overlaid watermark". One forward per image, no text decoding, so
    it answers in milliseconds where a VLM spends ~1.7 s writing a sentence. It
    returns a PROBABILITY, which is what makes a tunable threshold possible at all.
  * LOCATE — ``IDEA-Research/grounding-dino-tiny`` (Apache-2.0), zero-shot object
    detection prompted with the phrases below. Only ever run on images the ranker
    already flagged, because localisation is the expensive half and most images
    are clean. Its boxes are what the Bank's ✂ crop / 🧽 inpaint levels route on,
    so a detector that only said yes/no would be a downgrade from the VLM it
    replaces.

WHY NOT Florence-2, which the research picked for this second stage: it does not
load. ``microsoft/Florence-2-base-ft`` ships its modelling code IN the repo
(trust_remote_code), written against transformers ~4.4x, and on a current
transformers it dies with ``'Florence2LanguageConfig' object has no attribute
'forced_bos_token_id'`` — that attribute moved out of PretrainedConfig in
transformers 5. Pinning transformers down was the wrong price: this extra
deliberately shares the bank-scoring environment (so the user is not asked for a
second 2.5 GB copy of torch), and holding that whole environment back for one
unmaintained remote file would put the ✨ Score NSFW classifier at risk too.
Grounding DINO answers the same "phrase in, boxes out" question, is integrated
NATIVELY in transformers (no remote code, no flash-attn workaround, no version
roulette) and is Apache-2.0.

LICENSES (verified at the source repo, not from a mirror's README):
  torch                          BSD-3-Clause
  transformers                   Apache-2.0
  timm                           Apache-2.0
  Pillow                         MIT-CMU
  huggingface_hub                Apache-2.0
  safetensors                    Apache-2.0
  Watermark-Detection-SigLIP2    Apache-2.0
  grounding-dino-tiny            Apache-2.0
Nothing here is AGPL. `ultralytics` is deliberately NOT used at any point: it
claims AGPL-3.0 over trained WEIGHTS, which would contaminate this public repo,
and the best public watermark YOLO descends from a checkpoint with no declared
licence at all.

Protocol (streaming, unlike the batch workers — the parent commits per image and
must be able to stop a 30 000-image pass without losing what it already learned):
  stdin  : ONE json object
           {"images": [abs paths], "threshold": 0.60, "locate": true,
            "device": "auto"|"cuda"|"cpu", "models_root": path|null,
            "cancel_file": path|null}
  stdout : ONE json line PER IMAGE, in input order:
           {"path": str, "state": "detected"|"none"|"error",
            "score": float|null, "regions": [[x1,y1,x2,y2], ...],
            "error": str|null}
           then a final line {"summary": {...}}.
  stderr : "[wmdet] i/N state" progress lines + load diagnostics.

A per-image failure is a row with state "error", never a dead pass: one unreadable
file out of thirty thousand must not cost the other 29 999.
"""
from __future__ import annotations
import io
import hashlib
import json
import os
import sys
# A ._pth-pinned interpreter (ComfyUI portable's python_embeded) does not put
# this script's directory on sys.path — restore it or the import below dies
# there. See _harness.py for the whole story.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _harness import _cancel_requested, _emit, _log

# Model ids are CONSTANTS, never config: a detector whose weights can be pointed
# at an arbitrary repo has no calibrated threshold, and the threshold is the
# feature. Changing either of these invalidates the measured numbers in the help.
RANK_MODEL = 'prithivMLmods/Watermark-Detection-SigLIP2'
LOCATE_MODEL = 'IDEA-Research/grounding-dino-tiny'
# Grounding DINO takes a period-separated list of phrases. Three, not one,
# because an overlaid mark is called three different things by the same model
# depending on what it looks like — a semi-transparent site name is a
# "watermark", a corner mark is a "logo", a bottom banner is a "text overlay" —
# and this stage runs ONLY on images already judged watermarked, so widening the
# phrase costs recall of the box, not precision of the verdict.
LOCATE_PROMPT = 'a watermark. a logo. a text overlay.'
# Box confidence floor for the locator. Well below the ranker's threshold on
# purpose: the question here is no longer "is this image marked" (already
# answered) but "where", and a timid box is better than none.
LOCATE_BOX_THRESHOLD = 0.25
# The locator also sweeps the image in TILES (2x2 and 3x3 windows), because a
# small corner logo in a large photo shrinks below DINO's notice at full frame:
# measured on a real 2048px photo carrying seven copies of a logo, the full
# pass found 4, the tiled sweep all 7. Tiles are noisier — a tile promotes rock
# texture to "a logo" — so the tile passes demand a HIGHER confidence, and the
# floor below is measured: at 0.25 the tile junk pushed raw coverage past the
# wall-to-wall guard and zeroed the scan; at 0.35 the same sweep returned every
# true mark with coverage 0.23.
LOCATE_TILE_THRESHOLD = 0.35
TILE_OVERLAP = 0.12
# Second, WIDER prompt used as a validator: a box is kept only when the two
# phrasings agree on the spot (strict consensus). Ground-truth bench, 12
# stamped scenarios over 3 real photos: base prompt alone at these scales is
# 79% recall / 72% precision; the strict consensus costs one recall point
# (77%) and lifts precision to 90% — the boxes it drops are exactly the rock
# and icicle shapes a single phrasing hallucinates in tiles.
LOCATE_VALIDATE_PROMPT = (LOCATE_PROMPT
                          + ' a semi-transparent text. a small logo.'
                          + ' a repeated watermark pattern.')

# A big box is not a watermark, it is a failed localisation — the phrase "a logo"
# matched the subject, or "a text overlay" matched the whole picture — and acting
# on it would route the image to a crop that destroys it or an inpaint that
# repaints the person. Dropped, which leaves the image flagged with NO box: the
# honest state ("we know it is marked, we do not know where"), which the Bank
# already handles.
#
# 0.10 is MEASURED, not inherited: over the 87 boxes the locator returned on 54
# real flagged images, areas form two clean populations — 74 boxes at or under
# 0.041 of the frame (every one of them an actual mark) and 13 at 0.069 and
# above, topping out at 0.594 (every one of them the subject). The cut sits in
# the empty gap between them, and it happens to agree with the ceiling the
# existing crop/inpaint router already applies to a mark's area.
MAX_REGION_AREA = 0.10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_image_guard import read_validated_bank_image  # noqa: E402


def _pick_device(requested):
    import torch
    if requested == 'cpu':
        return 'cpu'
    if requested == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class _Ranker:
    """SigLIP2 watermark classifier. Loaded once, reused for every image."""

    def __init__(self, device, cache_dir):
        import torch
        from transformers import AutoImageProcessor, SiglipForImageClassification
        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            RANK_MODEL, cache_dir=cache_dir)
        self.model = SiglipForImageClassification.from_pretrained(
            RANK_MODEL, cache_dir=cache_dir).to(device).eval()
        # NEVER hardcode the positive index. The published label order is
        # {"0": "No Watermark", "1": "Watermark"} today, and a silent flip in a
        # future revision would invert every verdict in the app while every test
        # still passed. Read it from the config the weights ship with, and refuse
        # to guess if the labels stop being recognisable.
        id2label = {int(k): str(v) for k, v in
                    (self.model.config.id2label or {}).items()}
        positive = [i for i, name in id2label.items()
                    if 'no' not in name.lower().split() and 'watermark' in name.lower()]
        if len(positive) != 1:
            raise RuntimeError(
                f'{RANK_MODEL} labels are not what this build expects '
                f'({id2label!r}) — refusing to guess which class means '
                '"watermarked"')
        self.positive = positive[0]
        self.labels = id2label

    def score(self, image):
        with self.torch.no_grad():
            inputs = self.processor(images=image, return_tensors='pt').to(self.device)
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0]
        return float(probs[self.positive].item())


class _Locator:
    """Grounding DINO zero-shot detection, loaded LAZILY — a bank with no
    watermark at all must never pay a 690 MB model load."""

    def __init__(self, device, cache_dir):
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.torch = None
        self.failed = None      # the load error, reported once, never retried

    def _ensure(self):
        if self.model is not None or self.failed is not None:
            return
        try:
            import torch
            from transformers import (AutoModelForZeroShotObjectDetection,
                                      AutoProcessor)
            self.processor = AutoProcessor.from_pretrained(
                LOCATE_MODEL, cache_dir=self.cache_dir)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                LOCATE_MODEL, cache_dir=self.cache_dir).to(self.device).eval()
            self.torch = torch
            _log(f'[wmdet] locator ready ({LOCATE_MODEL}, {self.device})')
        except Exception as e:      # noqa: BLE001
            # Degrade to rank-only: the pass still flags the right images, they
            # simply arrive without a box (exactly like the pre-box builds did),
            # and the parent says so rather than failing the whole scan.
            self.failed = f'{type(e).__name__}: {e}'
            _log(f'[wmdet] locator unavailable, flagging without boxes: {self.failed}')

    def _detect(self, image, prompt, threshold):
        """One DINO forward on one (sub)image → raw pixel boxes. Tiles are sent
        ONE BY ONE on purpose: batching the sweep pads every tile to the
        largest and was measured at up to 25x the sequential time."""
        inputs = self.processor(images=image, text=prompt,
                                return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'], threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(image.size[1], image.size[0])])[0]
        return [[float(v) for v in b.tolist()] for b in result['boxes']]

    def _collect(self, image, prompt):
        """Raw pixel boxes for one prompt across the whole tile plan, mapped
        back to full-frame coordinates. A failed window contributes nothing and
        the sweep goes on — one bad tile must not cost the other twelve."""
        raw = []
        for grid_n, threshold in tile_plan(image.size):
            for x1, y1, x2, y2 in tile_windows(image.size, grid_n):
                try:
                    tile = image if grid_n == 1 else image.crop((x1, y1, x2, y2))
                    raw += [[b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]
                            for b in self._detect(tile, prompt, threshold)]
                except Exception as e:      # noqa: BLE001 — one window, not the pass
                    _log(f'[wmdet] locate window failed: {type(e).__name__}: {e}')
        return raw

    def regions(self, image):
        """Normalised [x1,y1,x2,y2] boxes in 0..1, most peripheral first. []
        when the locator could not load or found nothing both prompts name.

        Two sweeps (geometry prompt + validator prompt) over the same adaptive
        tile plan, then the strict consensus + wall-to-wall rule — the whole
        recipe and its measured numbers live on effective_regions and the
        constants above."""
        self._ensure()
        if self.model is None:
            return []
        base = self._collect(image, LOCATE_PROMPT)
        validate = self._collect(image, LOCATE_VALIDATE_PROMPT)
        return effective_regions(base, image.size, validate_boxes=validate)


# Above this fraction of the frame (raw DINO boxes, before the per-box area
# cap), the marks are WALL-TO-WALL — a tiled stock watermark repeated across
# the whole picture. Reporting the one corner tile that survives the cap would
# be a lie by omission: the clean would repaint that tile and leave the rest,
# and the user would file the image as handled. Measured case: a stock photo
# tiled with "Watermark stock photo" was flagged with a single small box.
# The honest verdict is "detected, no usable position" — the same unlocated
# lane the parent already routes to 🔍 Review with its own explanation.
WALL_TO_WALL_COVERAGE = 0.40


def _raw_coverage(boxes, size):
    """Fraction of the frame the raw boxes claim, cap-free. Clamped per box and
    summed (tiles barely overlap); bounded to 1.0 so a pathological pile of
    frame-sized boxes cannot report 700% of an image."""
    width, height = size
    if not width or not height:
        return 0.0
    total = 0.0
    for box in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0.0, x1 / width), min(1.0, x2 / width)))
        y1, y2 = sorted((max(0.0, y1 / height), min(1.0, y2 / height)))
        total += max(0.0, (x2 - x1) * (y2 - y1))
    return min(1.0, total)


# Under a wall-to-wall claim, fewer located zones than this is a FAILED
# localisation (the old one-tile-for-a-tiled-mark lie) and reports nothing;
# at or above it the zones are worth showing even though more marks exist —
# the real tiled stock photo localises 12 of ~25 text tiles, and 12 drawn
# boxes in Review tell the user "it is everywhere" far better than zero did.
WALL_TO_WALL_MIN_ZONES = 3


def effective_regions(boxes, size, validate_boxes=None):
    """The zones a scan should REPORT for these raw pixel boxes — the one
    decision point, pure so the parent's tests can hold it.

    ``validate_boxes`` (raw pixel boxes from the second prompt) enables the
    strict consensus: only base boxes the validator agrees on survive. None
    keeps the single-set behaviour (legacy callers, degraded validator).

    A wall-to-wall claim (raw coverage above WALL_TO_WALL_COVERAGE) with fewer
    than WALL_TO_WALL_MIN_ZONES located zones returns [] — unlocated → Review,
    the parent says why. With enough located zones the zones ARE the honest
    report, tiling or not."""
    kept = _normalise_boxes(boxes, size)
    if validate_boxes is not None:
        kept = _strict_consensus(kept, _normalise_boxes(validate_boxes, size))
    merged = _merge_boxes(kept)
    if (_raw_coverage(boxes, size) > WALL_TO_WALL_COVERAGE
            and len(merged) < WALL_TO_WALL_MIN_ZONES):
        return []
    return merged


def _strict_consensus(base, validate):
    """Boxes (normalised) both prompt passes agree on — agreement = overlap.
    The base box carries the geometry (its boxes hug the full mark); the
    validator only vouches for the SPOT. Validator-only boxes are dropped:
    symmetric union was measured at 72% precision against 90% for this."""
    return [b for b in base if any(_overlaps(b, v) for v in validate)]


# A tile below this on its short side is an upscaled thumbnail, not a window —
# DINO has nothing left to see into. 250px keeps every measured win: the
# 1200x800 tiled stock photo owes 11 of its 12 zones to the 3x3 sweep
# (800/3 = 267), and a 640px phone-sized image still earns its 2x2.
TILE_MIN_SIDE = 250


def tile_plan(size):
    """[(grid_n, box_threshold)] for this image size — pure and adaptive.

    Every image gets the legacy full-frame pass at the legacy floor, so a
    small image behaves byte-for-byte as before; each deeper grid joins only
    while its tiles keep TILE_MIN_SIDE on the short side."""
    short = min(size[0] or 0, size[1] or 0)
    plan = [(1, LOCATE_BOX_THRESHOLD)]
    for grid_n in (2, 3):
        if short / grid_n >= TILE_MIN_SIDE:
            plan.append((grid_n, LOCATE_TILE_THRESHOLD))
    return plan


def tile_windows(size, grid_n, overlap=TILE_OVERLAP):
    """Pixel crop windows [(x1, y1, x2, y2)] for an n x n sweep, each grown by
    ``overlap`` of a tile per side so a mark cut by a seam still lands whole in
    one window. grid_n == 1 is the full frame."""
    width, height = size
    if grid_n <= 1:
        return [(0, 0, width, height)]
    tw, th = width / grid_n, height / grid_n
    ox, oy = tw * overlap, th * overlap
    windows = []
    for j in range(grid_n):
        for i in range(grid_n):
            windows.append((max(0, int(i * tw - ox)), max(0, int(j * th - oy)),
                            min(width, int((i + 1) * tw + ox)),
                            min(height, int((j + 1) * th + oy))))
    return windows


def _normalise_boxes(boxes, size):
    """Pixel boxes -> the app's normalised 0..1 contract. Clamped; degenerate and
    frame-sized boxes dropped."""
    width, height = size
    if not width or not height:
        return []
    out = []
    for box in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((max(0.0, x1 / width), min(1.0, x2 / width)))
        y1, y2 = sorted((max(0.0, y1 / height), min(1.0, y2 / height)))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or area > MAX_REGION_AREA:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _merge_boxes(boxes):
    """Union overlapping boxes, biggest first.

    A phrase list of three produces three near-identical boxes over the SAME
    mark, and handing all three to the mask editor would show the user three
    stacked rectangles for one watermark and make the inpaint repaint the region
    three times. Overlap is the right merge test here (not proximity): two marks
    in two different corners must stay two zones, because a single box spanning
    them would cover the subject between them."""
    merged = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                      reverse=True):
        for existing in merged:
            if _overlaps(existing, box):
                existing[0] = min(existing[0], box[0])
                existing[1] = min(existing[1], box[1])
                existing[2] = max(existing[2], box[2])
                existing[3] = max(existing[3], box[3])
                break
        else:
            merged.append(list(box))
    merged = [b for b in merged if (b[2] - b[0]) * (b[3] - b[1]) <= MAX_REGION_AREA]
    # ORDER MATTERS: the caller persists ONE box (the Bank's watermark_bbox is a
    # single rectangle), so first place is a decision, not a display detail.
    #
    # It is NOT the biggest one. Measured on a real bank: an image whose actual
    # mark is a 0.2%-of-frame corner logo also produced a 2.7% box in the middle
    # of the picture — the locator naming a printed word on the subject — and
    # "biggest" picked the middle one, which would send a crop or an inpaint at
    # the person instead of the logo. So the tie is broken by DISTANCE TO THE
    # FRAME EDGE first: an overlaid mark is a thing added at the edge of someone
    # else's picture, and that is exactly the shape the crop level can act on.
    # Area only breaks ties between equally peripheral boxes, where the fuller
    # cover of the same mark is the better mask.
    def edge_distance(b):
        return min(b[0], b[1], 1.0 - b[2], 1.0 - b[3])

    merged.sort(key=lambda b: (edge_distance(b),
                               -(b[2] - b[0]) * (b[3] - b[1])))
    return [[round(v, 4) for v in b] for b in merged]


def _overlaps(a, b) -> bool:
    return (min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1]))


def _file_hash(path):
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.digest()
    except OSError:
        return b''


def _open(path):
    from PIL import Image
    payload = read_validated_bank_image(path)
    digest = hashlib.sha256(payload).digest()
    return Image.open(io.BytesIO(payload)).convert('RGB'), digest


def main():
    try:
        job = json.loads(sys.stdin.read() or '{}')
    except ValueError as e:
        _emit({'summary': {'ok': False, 'error': f'unreadable job: {e}'}})
        return 1
    images = [str(p) for p in (job.get('images') or [])]
    threshold = float(job.get('threshold') or 0.60)
    locate = bool(job.get('locate', True))
    cancel_file = job.get('cancel_file') or ''
    cache_dir = job.get('models_root') or None

    try:
        device = _pick_device(str(job.get('device') or 'auto'))
        ranker = _Ranker(device, cache_dir)
    except Exception as e:      # noqa: BLE001 — a load failure IS the whole pass
        _emit({'summary': {'ok': False,
                           'error': f'{type(e).__name__}: {e}'}})
        return 1
    _log(f'[wmdet] ranker ready ({RANK_MODEL}, {device}, '
         f'positive class {ranker.positive}={ranker.labels.get(ranker.positive)!r})')
    locator = _Locator(device, cache_dir) if locate else None

    detected = clean = errors = 0
    for i, path in enumerate(images, 1):
        if _cancel_requested(cancel_file):
            _log(f'[wmdet] cancelled at {i - 1}/{len(images)}')
            break
        try:
            image, payload_hash = _open(path)
            score = ranker.score(image)
            if _file_hash(path) != payload_hash:
                raise RuntimeError('image changed while it was analysed')
        except Exception as e:      # noqa: BLE001
            errors += 1
            _emit({'path': path, 'state': 'error', 'score': None, 'regions': [],
                   'fingerprint': None,
                   'error': f'{type(e).__name__}: {e}'})
            _log(f'[wmdet] {i}/{len(images)} error')
            continue
        if score >= threshold:
            detected += 1
            regions = locator.regions(image) if locator is not None else []
            if _file_hash(path) != payload_hash:
                _emit({'path': path, 'state': 'error', 'score': None,
                       'regions': [], 'fingerprint': None,
                       'error': 'image changed while it was analysed'})
                errors += 1
                detected -= 1
                continue
            _emit({'path': path, 'state': 'detected', 'score': round(score, 4),
                   'regions': regions, 'fingerprint': payload_hash.hex(),
                   'error': None})
        else:
            clean += 1
            _emit({'path': path, 'state': 'none', 'score': round(score, 4),
                   'regions': [], 'fingerprint': payload_hash.hex(),
                   'error': None})
        _log(f'[wmdet] {i}/{len(images)} {"detected" if score >= threshold else "none"}')
    _emit({'summary': {'ok': True, 'detected': detected, 'clean': clean,
                       'errors': errors, 'device': device,
                       'threshold': threshold,
                       'located': bool(locator is not None and locator.failed is None),
                       'locate_error': (locator.failed if locator is not None else None)}})
    return 0


if __name__ == '__main__':
    sys.exit(main())
