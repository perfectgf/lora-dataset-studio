"""Portable, fail-closed analysis transport between Banks and Datasets.

Version 3 keeps row metadata in the Dataset row and large ML embeddings in one
small, bounded NPZ sidecar per image.  The JSON contains no path and the sidecar
contains no path/signature: both are authorised by the enclosing SHA-256 of the
exact Dataset bytes.  A restored Bank writes fresh destination paths and stat
signatures into its CLIP, SigLIP2 and Face runtime caches. Older two-lane
sidecars remain readable.

Version 2 deterministic-only snapshots remain readable for existing datasets
and backups.  Transformation markers, promotion pointers and source paths are
never members of either schema.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from pathlib import Path

from . import npz_transport


SNAPSHOT_VERSION = 3
LEGACY_SNAPSHOT_VERSION = 2
_MAX_SNAPSHOT_JSON_BYTES = 8 * 1024
_MAX_CACHE_SIDECAR_BYTES = 64 * 1024
CACHE_SIDECAR_MAX_BYTES = _MAX_CACHE_SIDECAR_BYTES
_MAX_RUNTIME_CACHE_BYTES = 1024 * 1024 * 1024
_MAX_RUNTIME_CACHE_ENTRIES = 200_000
_FINGERPRINT_RE = re.compile(r'^[0-9a-f]{64}$')
# New sidecars are named by the full SHA-256 of their exact NPZ bytes.  The
# 32-hex form remains readable for local pre-hardening snapshots, but portable
# backup v2 deliberately accepts only the content-addressed 64-hex form.
_CACHE_REF_RE = re.compile(r'^(?:[0-9a-f]{32}|[0-9a-f]{64})$')
_STAT_SIG_RE = re.compile(r'^\d+:\d+$')


DETERMINISTIC_ANALYSIS_FIELDS = (
    'quality_state', 'blur_score', 'noise_score', 'uniformity_score',
    'dhash', 'detail_ratio', 'bars_ratio', 'jpeg_quality', 'origin',
    'origin_evidence',
)

MODEL_ANALYSIS_FIELDS = (
    'face_state', 'face_det', 'aesthetic_score', 'nsfw_score',
)

SEMANTIC_DUPLICATE_GROUP_FIELDS = (
    'semantic_dup_group',
    'clip_semantic_dup_group',
    'siglip2_semantic_dup_group',
)
BANK_COPY_DUPLICATE_GROUP_FIELDS = (
    'dup_group', *SEMANTIC_DUPLICATE_GROUP_FIELDS)
BANK_LOCAL_GROUP_FIELDS = (
    *BANK_COPY_DUPLICATE_GROUP_FIELDS, 'face_cluster', 'style_cluster',
)

BANK_DIRECT_COPY_ANALYSIS_FIELDS = (
    *DETERMINISTIC_ANALYSIS_FIELDS,
    *MODEL_ANALYSIS_FIELDS,
    'framing',
    'watermark_state', 'watermark_bbox', 'watermark_regions',
    'watermark_source', 'watermark_score',
    'medium', 'medium_margin', 'face_yaw',
    *BANK_COPY_DUPLICATE_GROUP_FIELDS,
    'face_cluster', 'face_cluster_origin', 'style_cluster',
)

# Version 3 predates the two durable engine lanes.  Its outer schema and version
# stay unchanged: old snapshots have this exact analysis key set and are expanded
# with NULL lanes on read, while newly written v3 snapshots carry all fields.
_LEGACY_V3_BANK_ANALYSIS_FIELDS = tuple(
    name for name in BANK_DIRECT_COPY_ANALYSIS_FIELDS
    if name not in ('clip_semantic_dup_group',
                    'siglip2_semantic_dup_group'))

BANK_TRANSFORM_STALE_ANALYSIS_FIELDS = tuple(
    name for name in BANK_DIRECT_COPY_ANALYSIS_FIELDS
    if name not in DETERMINISTIC_ANALYSIS_FIELDS
)

# Every persisted value derived from image pixels belongs to exactly one lane.
# ``analysis_fingerprint`` on BankImage is shared across the lanes: a writer
# that sees different bytes clears this complete union before recording its own
# lane.  Keeping the map beside the transport schema prevents a new persisted
# field from accidentally being transferable without being invalidated.
BANK_ANALYSIS_LANE_FIELDS = {
    'quality': (*DETERMINISTIC_ANALYSIS_FIELDS, 'dup_group'),
    'score': ('aesthetic_score', 'nsfw_score', 'medium', 'medium_margin',
              'style_cluster'),
    'semantic': SEMANTIC_DUPLICATE_GROUP_FIELDS,
    'face': ('face_state', 'face_det', 'face_yaw', 'face_cluster',
             'face_cluster_origin'),
    'watermark': ('watermark_state', 'watermark_bbox', 'watermark_regions',
                  'watermark_source', 'watermark_score',
                  # 🔤 Find text rides in the watermark lane: its zones live in
                  # watermark_regions, so its memory must be cleared, counted
                  # and transferred with the geometry it authored.
                  'text_state'),
    'framing': ('framing',),
}
BANK_WATERMARK_ANALYSIS_FIELDS = BANK_ANALYSIS_LANE_FIELDS['watermark']
BANK_EFFECTIVE_ANALYSIS_FIELDS = tuple(dict.fromkeys(
    name for lane, fields in BANK_ANALYSIS_LANE_FIELDS.items()
    if lane != 'watermark' for name in fields))
# Backward-compatible public name: callers that invalidate the shared analysis
# fingerprint now clear only lanes attached to the effective displayed bytes.
BANK_PIXEL_DERIVED_FIELDS = BANK_EFFECTIVE_ANALYSIS_FIELDS

# A Bank and a Dataset do not expose the same row columns.  This small envelope
# carries the source-side fields that have no active destination column, so an
# otherwise lossless round-trip does not silently throw them away.  Destination
# ids, paths and in-flight job pointers remain historical only: readers restore
# just the explicitly safe Dataset fields below, while the full normalized
# envelope keeps everything available for the next hop and for backups.
TRANSFER_METADATA_VERSION = 1
# The envelope may legitimately contain both long Dataset captions, a failure
# explanation, source attribution, hand masks and the prior analysis snapshot.
# 256 KiB is a hard per-row bound, but it is above the aggregate of every
# accepted portable field maximum so valid metadata never makes a copy fail.
TRANSFER_METADATA_MAX_BYTES = 256 * 1024
DATASET_PORTABLE_FIELDS = (
    'filename', 'source', 'variation_label',
    'caption_short', 'caption_short_origin',
    'job_id', 'variation_prompt', 'klein_model', 'parent_image_id',
    'derivation_kind', 'bank_image_id', 'face_score', 'face_state',
    'fail_reason', 'fail_kind', 'upscale_ratio', 'content_sig',
    'content_sig_stat', 'bank_analysis_snapshot', 'created_at',
)
BANK_PORTABLE_FIELDS = (
    'relpath', 'file_size', 'watermark_clean_method', 'rotation',
    'status', 'reject_reason', 'promoted_dataset_id', 'promoted_bank_id',
    'created_at', 'semantic_engine',
)
DATASET_RESTORE_FIELDS = (
    'source', 'variation_label', 'caption_short', 'caption_short_origin',
    'variation_prompt', 'klein_model', 'fail_reason', 'fail_kind',
)
DATASET_PIXEL_RESTORE_FIELDS = ('face_score', 'face_state', 'upscale_ratio')

_PORTABLE_INTEGER_FIELDS = frozenset((
    'parent_image_id', 'bank_image_id', 'file_size', 'width', 'height',
    'dup_group', *SEMANTIC_DUPLICATE_GROUP_FIELDS,
    'face_cluster', 'style_cluster',
    'rotation', 'promoted_dataset_id', 'promoted_bank_id',
))
_PORTABLE_FLOAT_FIELDS = frozenset((
    'face_score', 'upscale_ratio', 'blur_score', 'noise_score',
    'uniformity_score', 'detail_ratio', 'bars_ratio', 'jpeg_quality',
    'face_det', 'aesthetic_score', 'nsfw_score', 'watermark_score',
    'medium_margin', 'face_yaw',
))
_PORTABLE_TEXT_LIMITS = {
    'filename': 255, 'source': 12, 'framing': 12,
    'variation_label': 120, 'status': 16, 'caption': 10000,
    'caption_short': 10000, 'caption_origin': 16,
    'caption_short_origin': 16, 'job_id': 64,
    'variation_prompt': 500, 'klein_model': 255,
    'derivation_kind': 64, 'face_state': 32, 'fail_reason': 32768,
    'fail_kind': 32, 'watermark_state': 32, 'watermark_bbox': 8192,
    'watermark_regions': 32768, 'watermark_source': 32,
    'source_metadata': 32768, 'content_sig': 128,
    'content_sig_stat': 128, 'bank_analysis_snapshot': 16384,
    'created_at': 64, 'relpath': 4096, 'quality_state': 32,
    'dhash': 32, 'origin': 16, 'origin_evidence': 64,
    'watermark_clean_method': 32, 'analysis_fingerprint': 128,
    'watermark_fingerprint': 128, 'reject_reason': 64,
    'medium': 32, 'face_cluster_origin': 32,
    'semantic_engine': 16,
}
_PORTABLE_INVALID = object()

_TEXT_LIMITS = {
    'quality_state': 12, 'dhash': 16, 'origin': 8, 'origin_evidence': 24,
}
_OPTIONAL_NUMERIC_FIELDS = frozenset(('detail_ratio', 'bars_ratio', 'jpeg_quality'))
_NUMERIC_LIMITS = {
    'blur_score': (0.0, float((4 * 255) ** 2)),
    'noise_score': (0.0, 255.0),
    'uniformity_score': (0.0, 128.0),
    'detail_ratio': (0.0, 1.0),
    'bars_ratio': (0.0, 1.0),
    'jpeg_quality': (1.0, 100.0),
}
_FACE_STATES = frozenset(
    ('scorable', 'no_face', 'low_det', 'too_small', 'extreme_pose',
     'unreadable', 'error'))
_SCORE_STATES = frozenset(('ok', 'error'))
_SEMANTIC_STATES = frozenset(('ok', 'error'))
_FRAMINGS = frozenset(('face', 'bust', 'body', 'back', 'unknown'))
_WATERMARK_STATES = frozenset(
    ('none', 'detected', 'dismissed', 'cleaned', 'failed', 'error'))
_WATERMARK_SOURCES = frozenset(('detector', 'vision'))
_MEDIUMS = frozenset(('photo', 'anime', 'render3d', 'illustration', 'unsure'))
_FACE_CLUSTER_ORIGINS = frozenset(('asserted',))

_SIDECAR_KEYS_V1 = frozenset((
    'score_present', 'score_state', 'score_aes', 'score_nsfw', 'score_emb',
    'face_present', 'face_state', 'face_det', 'face_bfrac', 'face_yaw',
    'face_emb',
))
_SIDECAR_KEYS = frozenset((*_SIDECAR_KEYS_V1,
    'semantic_present', 'semantic_engine', 'semantic_model_id',
    'semantic_revision', 'semantic_model_key', 'semantic_dimension',
    'semantic_state', 'semantic_emb',
))


def content_fingerprint_bytes(data: bytes | bytearray) -> str | None:
    if not isinstance(data, (bytes, bytearray)):
        return None
    return hashlib.sha256(data).hexdigest()


def content_fingerprint_path(path: str | os.PathLike | None) -> str | None:
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as fh:
            while chunk := fh.read(1 << 20):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _portable_scalar(field, value):
    if value is None:
        return None
    if field == 'created_at' and not isinstance(value, str):
        try:
            value = value.isoformat()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return _PORTABLE_INVALID
    if field in _PORTABLE_INTEGER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            return _PORTABLE_INVALID
        return int(value) if -(2**63) < value < 2**63 else _PORTABLE_INVALID
    if field in _PORTABLE_FLOAT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _PORTABLE_INVALID
        number = float(value)
        return (number if math.isfinite(number) and abs(number) <= 1e15
                else _PORTABLE_INVALID)
    limit = _PORTABLE_TEXT_LIMITS.get(field)
    if limit is None or not isinstance(value, str) or len(value) > limit:
        return _PORTABLE_INVALID
    try:
        return value if len(value.encode('utf-8')) <= limit * 4 else _PORTABLE_INVALID
    except UnicodeError:
        return _PORTABLE_INVALID


def _portable_namespace(value, fields):
    if value is None:
        return None
    if not isinstance(value, dict) or not set(value).issubset(fields):
        return None
    out = {}
    for field in fields:
        normalized = _portable_scalar(field, value.get(field))
        if normalized is _PORTABLE_INVALID:
            return None
        out[field] = normalized
    return out


def parse_transfer_metadata(value) -> dict | None:
    """Parse the bounded, path-inert Bank/Dataset metadata envelope."""
    if isinstance(value, str):
        try:
            if (len(value) > TRANSFER_METADATA_MAX_BYTES
                    or len(value.encode('utf-8')) > TRANSFER_METADATA_MAX_BYTES):
                return None
            value = json.loads(value)
        except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
            return None
    expected_keys = {
        'v', 'bank', 'dataset', 'bank_fingerprint', 'dataset_fingerprint',
        'dataset_analysis_fingerprint'}
    if (not isinstance(value, dict) or set(value) != expected_keys
            or value.get('v') != TRANSFER_METADATA_VERSION):
        return None
    bank = _portable_namespace(value.get('bank'), BANK_PORTABLE_FIELDS)
    dataset = _portable_namespace(value.get('dataset'), DATASET_PORTABLE_FIELDS)
    bank_fingerprint = value.get('bank_fingerprint')
    dataset_fingerprint = value.get('dataset_fingerprint')
    dataset_analysis_fingerprint = value.get('dataset_analysis_fingerprint')
    if ((value.get('bank') is not None and bank is None)
            or (value.get('dataset') is not None and dataset is None)):
        return None
    if bank is not None and bank.get('semantic_engine') not in (
            None, 'clip', 'siglip2'):
        return None
    if (bank_fingerprint is not None
            and (not isinstance(bank_fingerprint, str)
                 or not _FINGERPRINT_RE.fullmatch(bank_fingerprint))):
        return None
    if (dataset_fingerprint is not None
            and (not isinstance(dataset_fingerprint, str)
                 or not _FINGERPRINT_RE.fullmatch(dataset_fingerprint))):
        return None
    if (dataset_analysis_fingerprint is not None
            and (not isinstance(dataset_analysis_fingerprint, str)
                 or not _FINGERPRINT_RE.fullmatch(
                     dataset_analysis_fingerprint))):
        return None
    if bank is None and bank_fingerprint is not None:
        return None
    if dataset is None and dataset_fingerprint is not None:
        return None
    if dataset is None and dataset_analysis_fingerprint is not None:
        return None
    return {
        'v': TRANSFER_METADATA_VERSION,
        'bank': bank, 'bank_fingerprint': bank_fingerprint,
        'dataset': dataset, 'dataset_fingerprint': dataset_fingerprint,
        'dataset_analysis_fingerprint': dataset_analysis_fingerprint,
    }


def _portable_capture(source, fields):
    if source is None:
        return None
    values = {
        field: (source.get(field) if isinstance(source, dict)
                else getattr(source, field, None))
        for field in fields
    }
    return _portable_namespace(values, fields)


def capture_transfer_metadata(
        value=None, *, bank=None, bank_fingerprint=None, dataset=None,
        dataset_fingerprint=None, rebind_dataset_from=None) -> str | None:
    """Merge current source metadata into a normalized portable envelope.

    Existing opposite-side history is retained.  A malformed non-empty envelope
    fails closed instead of being silently replaced with a weaker one.
    """
    if value is None:
        envelope = {
            'v': TRANSFER_METADATA_VERSION,
            'bank': None, 'bank_fingerprint': None,
            'dataset': None, 'dataset_fingerprint': None,
            'dataset_analysis_fingerprint': None,
        }
    else:
        envelope = parse_transfer_metadata(value)
        if envelope is None:
            return None
    envelope = dict(envelope)
    if bank is not None:
        if (not isinstance(bank_fingerprint, str)
                or not _FINGERPRINT_RE.fullmatch(bank_fingerprint)):
            return None
        captured = _portable_capture(bank, BANK_PORTABLE_FIELDS)
        if captured is None:
            return None
        envelope['bank'] = captured
        envelope['bank_fingerprint'] = bank_fingerprint
    if dataset is not None:
        if (not isinstance(dataset_fingerprint, str)
                or not _FINGERPRINT_RE.fullmatch(dataset_fingerprint)):
            return None
        captured = _portable_capture(dataset, DATASET_PORTABLE_FIELDS)
        if captured is None:
            return None
        envelope['dataset'] = captured
        envelope['dataset_fingerprint'] = dataset_fingerprint
        envelope['dataset_analysis_fingerprint'] = dataset_fingerprint
    # A tracked Bank rotation/clean is lineage-preserving only while its raw
    # source still equals the Dataset bytes that entered the Bank. Rebind the
    # Dataset-only metadata to the baked effective payload in that proven case;
    # an external same-path replacement cannot satisfy this comparison.
    if (envelope.get('dataset') is not None
            and isinstance(rebind_dataset_from, str)
            and _FINGERPRINT_RE.fullmatch(rebind_dataset_from)
            and envelope.get('dataset_fingerprint') == rebind_dataset_from
            and isinstance(bank_fingerprint, str)
            and _FINGERPRINT_RE.fullmatch(bank_fingerprint)):
        envelope['dataset_fingerprint'] = bank_fingerprint
    try:
        stored = json.dumps(
            envelope, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        return (stored if len(stored.encode('utf-8')) <= TRANSFER_METADATA_MAX_BYTES
                else None)
    except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None


def normalized_transfer_metadata_storage(value) -> str | None:
    parsed = parse_transfer_metadata(value)
    if parsed is None:
        return None
    try:
        stored = json.dumps(
            parsed, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        return (stored if len(stored.encode('utf-8')) <= TRANSFER_METADATA_MAX_BYTES
                else None)
    except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None


def dataset_restore_values(value, fingerprint=None) -> dict:
    """Safe active Dataset fields from a prior Dataset hop.

    Local ids, paths, job pointers and graph relationships stay in the envelope
    as history and are intentionally never activated in another container.
    """
    parsed = parse_transfer_metadata(value)
    stored = parsed.get('dataset') if parsed else None
    if (not stored or not isinstance(fingerprint, str)
            or parsed.get('dataset_fingerprint') != fingerprint):
        return {}
    out = {field: stored.get(field) for field in DATASET_RESTORE_FIELDS}
    if parsed.get('dataset_analysis_fingerprint') == fingerprint:
        out.update({
            field: stored.get(field)
            for field in DATASET_PIXEL_RESTORE_FIELDS
        })
    if out.get('source') not in ('generated', 'import'):
        out['source'] = 'import'
    if out.get('caption_short_origin') not in (
            None, 'asserted', 'joycaption', 'ollama'):
        out['caption_short_origin'] = None
    return out


def bank_semantic_engine_for_fingerprint(value, fingerprint=None) -> str | None:
    """Return the source Bank choice only when it is bound to these exact bytes.

    Dataset rows can come from several Banks. Callers therefore collect this
    per-row value and restore it only when every compatible row agrees.
    """
    parsed = parse_transfer_metadata(value)
    stored = parsed.get('bank') if parsed else None
    if (not stored or not isinstance(fingerprint, str)
            or parsed.get('bank_fingerprint') != fingerprint):
        return None
    engine = stored.get('semantic_engine')
    return engine if engine in ('clip', 'siglip2') else None


def is_content_addressed_cache_ref(value) -> bool:
    """True only for the hardened full-SHA sidecar namespace."""
    return (isinstance(value, str) and len(value) == 64
            and _FINGERPRINT_RE.fullmatch(value) is not None)


def cache_ref_matches_bytes(cache_ref, data) -> bool:
    return (is_content_addressed_cache_ref(cache_ref)
            and content_fingerprint_bytes(data) == cache_ref)


def runtime_stat_signature(path: str | os.PathLike | None) -> str:
    """Cheap runtime invalidation signature; SHA-256 stays transfer-only."""
    if not path:
        return ''
    try:
        st = os.stat(path)
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return ''


def _finite_number(value, low=None, high=None, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    if low is not None and number < low:
        raise ValueError
    if high is not None and number > high:
        raise ValueError
    return number


def _normalized_analysis(value) -> dict | None:
    """Strict v2 deterministic analysis."""
    if not isinstance(value, dict) or set(value) != set(DETERMINISTIC_ANALYSIS_FIELDS):
        return None
    out = {}
    try:
        for name in DETERMINISTIC_ANALYSIS_FIELDS:
            item = value.get(name)
            if item is None:
                if name in _OPTIONAL_NUMERIC_FIELDS or name == 'origin_evidence':
                    out[name] = None
                    continue
                return None
            if name in _TEXT_LIMITS:
                if not isinstance(item, str) or len(item) > _TEXT_LIMITS[name]:
                    return None
                if name == 'dhash' and not re.fullmatch(r'[0-9a-fA-F]{16}', item):
                    return None
                if name == 'quality_state' and item != 'ok':
                    return None
                if name == 'origin' and item not in ('ai', 'camera', 'unknown'):
                    return None
                out[name] = item.lower() if name == 'dhash' else item
                continue
            low, high = _NUMERIC_LIMITS[name]
            out[name] = _finite_number(item, low, high)
    except (TypeError, ValueError, OverflowError):
        return None
    origin, evidence = out['origin'], out['origin_evidence']
    if origin == 'unknown':
        if evidence is not None:
            return None
    elif not isinstance(evidence, str) or not evidence.strip():
        return None
    return out


def _normalized_partial_deterministic(value) -> dict | None:
    """Nullable v3 form: every available measurement is validated independently."""
    if not isinstance(value, dict) or set(value) != set(DETERMINISTIC_ANALYSIS_FIELDS):
        return None
    out = {}
    try:
        for name in DETERMINISTIC_ANALYSIS_FIELDS:
            item = value.get(name)
            if item is None:
                out[name] = None
                continue
            if name in _TEXT_LIMITS:
                if not isinstance(item, str) or len(item) > _TEXT_LIMITS[name]:
                    return None
                if name == 'quality_state' and item not in ('ok', 'unreadable'):
                    return None
                if name == 'dhash' and not re.fullmatch(r'[0-9a-fA-F]{16}', item):
                    return None
                if name == 'origin' and item not in ('ai', 'camera', 'unknown'):
                    return None
                out[name] = item.lower() if name == 'dhash' else item
                continue
            low, high = _NUMERIC_LIMITS[name]
            out[name] = _finite_number(item, low, high)
    except (TypeError, ValueError, OverflowError):
        return None
    origin, evidence = out['origin'], out['origin_evidence']
    if origin in (None, 'unknown'):
        if evidence is not None:
            return None
    elif not isinstance(evidence, str) or not evidence.strip():
        return None
    return out


def _normalized_box_json(value, *, regions=False):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError
    parsed = json.loads(value)
    boxes = parsed if regions else [parsed]
    if not isinstance(boxes, list) or len(boxes) > 64:
        raise ValueError
    normalized = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError
        nums = [_finite_number(item, 0.0, 1.0) for item in box]
        if not (nums[0] < nums[2] and nums[1] < nums[3]):
            raise ValueError
        normalized.append(nums)
    result = normalized if regions else normalized[0]
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))


def _optional_token(value, vocabulary):
    if value is None:
        return None
    if not isinstance(value, str) or value not in vocabulary:
        raise ValueError
    return value


def _optional_group(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 2**31 - 1):
        raise ValueError
    return int(value)


def normalized_full_analysis(value) -> dict | None:
    """Validate every Bank analysis field accepted by a v3 snapshot."""
    if not isinstance(value, dict) or set(value) not in (
            set(BANK_DIRECT_COPY_ANALYSIS_FIELDS),
            set(_LEGACY_V3_BANK_ANALYSIS_FIELDS)):
        return None
    deterministic = _normalized_partial_deterministic({
        name: value.get(name) for name in DETERMINISTIC_ANALYSIS_FIELDS
    })
    if deterministic is None:
        return None
    out = dict(deterministic)
    try:
        out['face_state'] = _optional_token(value.get('face_state'), _FACE_STATES)
        out['face_det'] = _finite_number(value.get('face_det'), 0.0, 1.0, optional=True)
        out['aesthetic_score'] = _finite_number(
            value.get('aesthetic_score'), -100.0, 100.0, optional=True)
        out['nsfw_score'] = _finite_number(value.get('nsfw_score'), 0.0, 1.0, optional=True)
        out['framing'] = _optional_token(value.get('framing'), _FRAMINGS)
        out['watermark_state'] = _optional_token(
            value.get('watermark_state'), _WATERMARK_STATES)
        out['watermark_bbox'] = _normalized_box_json(value.get('watermark_bbox'))
        out['watermark_regions'] = _normalized_box_json(
            value.get('watermark_regions'), regions=True)
        out['watermark_source'] = _optional_token(
            value.get('watermark_source'), _WATERMARK_SOURCES)
        out['watermark_score'] = _finite_number(
            value.get('watermark_score'), 0.0, 1.0, optional=True)
        out['medium'] = _optional_token(value.get('medium'), _MEDIUMS)
        out['medium_margin'] = _finite_number(
            value.get('medium_margin'), 0.0, 2.0, optional=True)
        out['face_yaw'] = _finite_number(
            value.get('face_yaw'), -180.0, 180.0, optional=True)
        for name in BANK_LOCAL_GROUP_FIELDS:
            out[name] = _optional_group(value.get(name))
        out['face_cluster_origin'] = _optional_token(
            value.get('face_cluster_origin'), _FACE_CLUSTER_ORIGINS)
        if out['face_cluster'] is None and out['face_cluster_origin'] is not None:
            return None
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError,
            RecursionError, MemoryError):
        return None
    return out


def _deterministic_only_full(analysis) -> dict | None:
    deterministic = _normalized_analysis(analysis)
    if deterministic is None:
        return None
    out = {name: None for name in BANK_DIRECT_COPY_ANALYSIS_FIELDS}
    out.update(deterministic)
    return out


def _normalized_group_scope(value, analysis):
    has_groups = any(analysis.get(name) is not None for name in BANK_LOCAL_GROUP_FIELDS)
    if value is None:
        return None if not has_groups else False
    if not isinstance(value, str) or not _CACHE_REF_RE.fullmatch(value):
        return False
    return value


def _snapshot_json(payload) -> str | None:
    try:
        value = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        if len(value.encode('utf-8')) > _MAX_SNAPSHOT_JSON_BYTES:
            return None
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None


def captured_bank_analysis(analysis, image_bytes, *, group_scope=None,
                           cache_bundle=None, assurance='exact',
                           watermark_fingerprint=None) -> dict | None:
    """Create an in-memory, SHA-bound promotion capture.

    ``cache_bundle`` stays in memory and never enters JSON/DB.  The Dataset
    importer writes it to a bounded sidecar only after its final bytes match.
    """
    analysis = normalized_full_analysis(analysis)
    fingerprint = content_fingerprint_bytes(image_bytes)
    if (analysis is None or fingerprint is None
            or assurance not in ('exact', 'legacy_tofu')):
        return None
    if (watermark_fingerprint is not None
            and (not isinstance(watermark_fingerprint, str)
                 or not _FINGERPRINT_RE.fullmatch(watermark_fingerprint))):
        return None
    scope = _normalized_group_scope(group_scope, analysis)
    if scope is False:
        return None
    caches = normalized_cache_bundle(cache_bundle or {})
    if caches is None:
        caches = {}
    return {'fingerprint': fingerprint, 'analysis': analysis,
            'assurance': assurance,
            'watermark_fingerprint': watermark_fingerprint,
            'group_scope': scope, 'caches': caches}


def snapshot_storage(analysis, image_bytes: bytes | bytearray, *, captured=None,
                     cache_ref=None) -> str | None:
    """Seal final Dataset bytes as v3, reusing full capture only if byte-identical."""
    final_analysis = _normalized_analysis(analysis)
    fingerprint = content_fingerprint_bytes(image_bytes)
    if final_analysis is None or fingerprint is None:
        return None
    full = _deterministic_only_full(final_analysis)
    scope = None
    assurance = 'exact'
    watermark_fingerprint = None
    if isinstance(captured, dict):
        candidate = normalized_full_analysis(captured.get('analysis'))
        candidate_assurance = captured.get('assurance')
        candidate_scope = (_normalized_group_scope(captured.get('group_scope'), candidate)
                           if candidate is not None else False)
        if (candidate is not None and candidate_scope is not False
                and candidate_assurance in ('exact', 'legacy_tofu')
                and captured.get('fingerprint') == fingerprint):
            full = dict(candidate)
            # The final Dataset bytes were independently measured.  Their
            # deterministic values are authoritative and also fill holes from
            # a manually-kept Bank that never ran Scan.
            full.update(final_analysis)
            scope = candidate_scope
            assurance = candidate_assurance
            candidate_watermark_fingerprint = captured.get(
                'watermark_fingerprint')
            if (candidate_watermark_fingerprint is None
                    or (isinstance(candidate_watermark_fingerprint, str)
                        and _FINGERPRINT_RE.fullmatch(
                            candidate_watermark_fingerprint))):
                watermark_fingerprint = candidate_watermark_fingerprint
    if cache_ref is not None and (
            not isinstance(cache_ref, str) or not _CACHE_REF_RE.fullmatch(cache_ref)):
        return None
    return _snapshot_json({
        'v': SNAPSHOT_VERSION, 'fingerprint': fingerprint, 'analysis': full,
        'assurance': assurance,
        'watermark_fingerprint': watermark_fingerprint,
        'group_scope': scope, 'cache_ref': cache_ref,
    })


def _parse_v2(value):
    if (len(value) != 3 or set(value) != {'v', 'fingerprint', 'analysis'}
            or value.get('v') != LEGACY_SNAPSHOT_VERSION):
        return None
    fingerprint = value.get('fingerprint')
    analysis = _normalized_analysis(value.get('analysis'))
    if (not isinstance(fingerprint, str)
            or not _FINGERPRINT_RE.fullmatch(fingerprint) or analysis is None):
        return None
    return {'v': LEGACY_SNAPSHOT_VERSION, 'fingerprint': fingerprint,
            'analysis': analysis}


def _parse_v3(value):
    if (len(value) != 7
            or set(value) != {'v', 'fingerprint', 'analysis', 'assurance',
                              'watermark_fingerprint', 'group_scope', 'cache_ref'}
            or value.get('v') != SNAPSHOT_VERSION):
        return None
    fingerprint = value.get('fingerprint')
    analysis = normalized_full_analysis(value.get('analysis'))
    scope = (_normalized_group_scope(value.get('group_scope'), analysis)
             if analysis is not None else False)
    cache_ref = value.get('cache_ref')
    assurance = value.get('assurance')
    watermark_fingerprint = value.get('watermark_fingerprint')
    if (not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint)
            or analysis is None or scope is False
            or assurance not in ('exact', 'legacy_tofu')
            or (watermark_fingerprint is not None
                and (not isinstance(watermark_fingerprint, str)
                     or not _FINGERPRINT_RE.fullmatch(watermark_fingerprint)))
            or (cache_ref is not None and (
                not isinstance(cache_ref, str) or not _CACHE_REF_RE.fullmatch(cache_ref)))):
        return None
    return {'v': SNAPSHOT_VERSION, 'fingerprint': fingerprint,
            'analysis': analysis, 'assurance': assurance,
            'watermark_fingerprint': watermark_fingerprint,
            'group_scope': scope, 'cache_ref': cache_ref}


def parse_snapshot(value) -> dict | None:
    if isinstance(value, str):
        if len(value) > _MAX_SNAPSHOT_JSON_BYTES:
            return None
        try:
            if len(value.encode('utf-8')) > _MAX_SNAPSHOT_JSON_BYTES:
                return None
            value = json.loads(value)
        except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
            return None
    if not isinstance(value, dict):
        return None
    if value.get('v') == LEGACY_SNAPSHOT_VERSION:
        return _parse_v2(value)
    if value.get('v') == SNAPSHOT_VERSION:
        return _parse_v3(value)
    return None


def normalized_snapshot_storage(value, *, drop_cache=False) -> str | None:
    snapshot = parse_snapshot(value)
    if snapshot is None:
        return None
    if drop_cache and snapshot['v'] == SNAPSHOT_VERSION:
        snapshot = dict(snapshot)
        snapshot['cache_ref'] = None
    return _snapshot_json(snapshot)


def compatible_snapshot(value, path: str | os.PathLike | None) -> dict | None:
    snapshot = parse_snapshot(value)
    if snapshot is None or content_fingerprint_path(path) != snapshot['fingerprint']:
        return None
    if snapshot['v'] == LEGACY_SNAPSHOT_VERSION:
        return {
            **snapshot,
            'analysis': _deterministic_only_full(snapshot['analysis']),
            'assurance': 'exact',
            'watermark_fingerprint': None,
            'group_scope': None,
            'cache_ref': None,
        }
    return snapshot


def _normalized_vector(value, dimension):
    try:
        arr = tuple(float(item) for item in value)
        if (len(arr) != dimension or not all(math.isfinite(item) for item in arr)):
            return None
        if max((abs(item) for item in arr), default=0.0) > 1.001:
            return None
        return arr
    except (TypeError, ValueError, OverflowError, MemoryError):
        return None


def _normalized_score_cache(value):
    if not isinstance(value, dict) or set(value) != {
            'state', 'aesthetic', 'nsfw', 'embedding'}:
        return None
    try:
        state = _optional_token(value.get('state'), _SCORE_STATES)
        if state is None:
            return None
        aesthetic = _finite_number(value.get('aesthetic'), -100.0, 100.0, optional=True)
        nsfw = _finite_number(value.get('nsfw'), 0.0, 1.0, optional=True)
        emb = _normalized_vector(value.get('embedding'), 768)
        if emb is None:
            return None
        return {'state': state, 'aesthetic': aesthetic, 'nsfw': nsfw, 'embedding': emb}
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_semantic_cache(value):
    """Validate one portable SigLIP2 vector and its complete provenance."""
    if not isinstance(value, dict) or set(value) != {
            'state', 'engine', 'model_id', 'revision', 'model_key',
            'dimension', 'embedding'}:
        return None
    try:
        from . import bank_semantic_models as assets
        if (value.get('engine') != 'siglip2'
                or value.get('model_id') != assets.MODEL_ID
                or value.get('revision') != assets.REVISION
                or value.get('model_key') != assets.MODEL_KEY
                or isinstance(value.get('dimension'), bool)
                or int(value.get('dimension')) != assets.DIMENSION):
            return None
        state = _optional_token(value.get('state'), _SEMANTIC_STATES)
        embedding = _normalized_vector(value.get('embedding'), assets.DIMENSION)
        if state is None or embedding is None:
            return None
        norm = math.sqrt(sum(component * component for component in embedding))
        if ((state == 'ok'
             and not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-4))
                or (state == 'error' and norm != 0.0)):
            return None
        return {
            'state': state, 'engine': 'siglip2',
            'model_id': assets.MODEL_ID, 'revision': assets.REVISION,
            'model_key': assets.MODEL_KEY, 'dimension': assets.DIMENSION,
            'embedding': embedding,
        }
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_face_cache(value):
    if not isinstance(value, dict) or set(value) != {
            'state', 'det', 'bbox_frac', 'yaw', 'embedding'}:
        return None
    try:
        state = _optional_token(value.get('state'), _FACE_STATES)
        if state is None:
            return None
        det = _finite_number(value.get('det'), 0.0, 1.0)
        bbox_frac = _finite_number(value.get('bbox_frac'), 0.0, 2.0)
        yaw = _finite_number(value.get('yaw'), -180.0, 180.0, optional=True)
        emb = _normalized_vector(value.get('embedding'), 512)
        if emb is None:
            return None
        return {'state': state, 'det': det, 'bbox_frac': bbox_frac,
                'yaw': yaw, 'embedding': emb}
    except (TypeError, ValueError, OverflowError):
        return None


def normalized_cache_bundle(value) -> dict | None:
    if (not isinstance(value, dict)
            or not set(value).issubset({'score', 'semantic', 'face'})):
        return None
    out = {}
    if value.get('score') is not None:
        score = _normalized_score_cache(value.get('score'))
        if score is None:
            return None
        out['score'] = score
    if value.get('semantic') is not None:
        semantic = _normalized_semantic_cache(value.get('semantic'))
        if semantic is None:
            return None
        out['semantic'] = semantic
    if value.get('face') is not None:
        face = _normalized_face_cache(value.get('face'))
        if face is None:
            return None
        out['face'] = face
    return out


def write_cache_sidecar(root, cache_bundle) -> str | None:
    """Atomically write one bounded, path-free embedding sidecar."""
    bundle = normalized_cache_bundle(cache_bundle)
    if not bundle:
        return None
    root = Path(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    staged = root / f'.sidecar-{secrets.token_hex(16)}.npz'
    score = bundle.get('score')
    semantic = bundle.get('semantic')
    face = bundle.get('face')
    nan = float('nan')
    arrays = {
        'score_present': npz_transport.uint8([score is not None], (1,)),
        'score_state': npz_transport.unicode([score['state'] if score else '']),
        'score_aes': npz_transport.floats([
            score['aesthetic'] if score and score['aesthetic'] is not None else nan], (1,)),
        'score_nsfw': npz_transport.floats([
            score['nsfw'] if score and score['nsfw'] is not None else nan], (1,)),
        'score_emb': npz_transport.floats(
            score['embedding'] if score else (0.0,) * 768, (768,)),
        'semantic_present': npz_transport.uint8([semantic is not None], (1,)),
        'semantic_engine': npz_transport.unicode([
            semantic['engine'] if semantic else '']),
        'semantic_model_id': npz_transport.unicode([
            semantic['model_id'] if semantic else '']),
        'semantic_revision': npz_transport.unicode([
            semantic['revision'] if semantic else '']),
        'semantic_model_key': npz_transport.unicode([
            semantic['model_key'] if semantic else '']),
        'semantic_dimension': npz_transport.int32([
            semantic['dimension'] if semantic else 0], (1,)),
        'semantic_state': npz_transport.unicode([
            semantic['state'] if semantic else '']),
        'semantic_emb': npz_transport.floats(
            semantic['embedding'] if semantic else (0.0,) * 768, (768,)),
        'face_present': npz_transport.uint8([face is not None], (1,)),
        'face_state': npz_transport.unicode([face['state'] if face else '']),
        'face_det': npz_transport.floats([face['det'] if face else 0.0], (1,)),
        'face_bfrac': npz_transport.floats([face['bbox_frac'] if face else 0.0], (1,)),
        'face_yaw': npz_transport.floats([
            face['yaw'] if face and face['yaw'] is not None else nan], (1,)),
        'face_emb': npz_transport.floats(
            face['embedding'] if face else (0.0,) * 512, (512,)),
    }
    if any(array is None for array in arrays.values()):
        return None
    try:
        if not npz_transport.write_atomic(
                staged, arrays, max_file_bytes=_MAX_CACHE_SIDECAR_BYTES):
            return None
        ref = content_fingerprint_path(staged)
        if not is_content_addressed_cache_ref(ref):
            return None
        final = root / f'{ref}.npz'
        # Content addressing makes concurrent identical writers converge on one
        # immutable filename. Replacing it is safe because both payloads carry
        # the same complete SHA-256; no random reference can be rebound later.
        os.replace(staged, final)
        return ref
    except OSError:
        return None
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def _cache_bundle_from_arrays(arrays) -> dict | None:
    try:
        if arrays is None or set(arrays) not in (_SIDECAR_KEYS_V1, _SIDECAR_KEYS):
            return None
        extended = set(arrays) == _SIDECAR_KEYS
        shapes = {
            'score_present': (1,), 'score_state': (1,),
            'score_aes': (1,), 'score_nsfw': (1,), 'score_emb': (768,),
            'face_present': (1,), 'face_state': (1,), 'face_det': (1,),
            'face_bfrac': (1,), 'face_yaw': (1,), 'face_emb': (512,),
        }
        if extended:
            shapes.update({
                'semantic_present': (1,), 'semantic_engine': (1,),
                'semantic_model_id': (1,), 'semantic_revision': (1,),
                'semantic_model_key': (1,), 'semantic_dimension': (1,),
                'semantic_state': (1,), 'semantic_emb': (768,),
            })
        if any(arrays[name].shape != shape for name, shape in shapes.items()):
            return None
        out = {}
        score_present = arrays['score_present'].uint8()
        semantic_present = arrays['semantic_present'].uint8() if extended else 0
        face_present = arrays['face_present'].uint8()
        if (score_present not in (0, 1) or semantic_present not in (0, 1)
                or face_present not in (0, 1)
                or not (score_present or semantic_present or face_present)):
            return None
        score_state = arrays['score_state'].string(0)
        score_aes = arrays['score_aes'].float(0)
        score_nsfw = arrays['score_nsfw'].float(0)
        score_embedding = tuple(
            arrays['score_emb'].float(i) for i in range(768))
        face_state = arrays['face_state'].string(0)
        face_det = arrays['face_det'].float(0)
        face_bfrac = arrays['face_bfrac'].float(0)
        face_yaw = arrays['face_yaw'].float(0)
        face_embedding = tuple(
            arrays['face_emb'].float(i) for i in range(512))
        if score_present:
            out['score'] = {
                'state': score_state,
                'aesthetic': None if math.isnan(score_aes) else score_aes,
                'nsfw': None if math.isnan(score_nsfw) else score_nsfw,
                'embedding': score_embedding,
            }
        elif (score_state or not math.isnan(score_aes)
              or not math.isnan(score_nsfw)
              or any(value != 0.0 for value in score_embedding)):
            return None
        if extended:
            semantic_engine = arrays['semantic_engine'].string(0)
            semantic_model_id = arrays['semantic_model_id'].string(0)
            semantic_revision = arrays['semantic_revision'].string(0)
            semantic_model_key = arrays['semantic_model_key'].string(0)
            semantic_dimension = arrays['semantic_dimension'].int32()
            semantic_state = arrays['semantic_state'].string(0)
            semantic_embedding = tuple(
                arrays['semantic_emb'].float(i) for i in range(768))
            if semantic_present:
                out['semantic'] = {
                    'state': semantic_state,
                    'engine': semantic_engine,
                    'model_id': semantic_model_id,
                    'revision': semantic_revision,
                    'model_key': semantic_model_key,
                    'dimension': semantic_dimension,
                    'embedding': semantic_embedding,
                }
            elif (semantic_engine or semantic_model_id or semantic_revision
                  or semantic_model_key or semantic_dimension != 0
                  or semantic_state
                  or any(value != 0.0 for value in semantic_embedding)):
                return None
        if face_present:
            out['face'] = {
                'state': face_state, 'det': face_det,
                'bbox_frac': face_bfrac,
                'yaw': None if math.isnan(face_yaw) else face_yaw,
                'embedding': face_embedding,
            }
        elif (face_state or face_det != 0.0 or face_bfrac != 0.0
              or not math.isnan(face_yaw)
              or any(value != 0.0 for value in face_embedding)):
            return None
        return normalized_cache_bundle(out)
    except (OSError, ValueError, TypeError, KeyError, IndexError, MemoryError):
        return None


def read_cache_sidecar(root, cache_ref) -> dict | None:
    if not isinstance(cache_ref, str) or not _CACHE_REF_RE.fullmatch(cache_ref):
        return None
    path = Path(root) / f'{cache_ref}.npz'
    if (is_content_addressed_cache_ref(cache_ref)
            and content_fingerprint_path(path) != cache_ref):
        return None
    arrays = npz_transport.read(
        path, max_file_bytes=_MAX_CACHE_SIDECAR_BYTES,
        max_uncompressed_bytes=32 * 1024, max_elements=2048)
    return _cache_bundle_from_arrays(arrays)


def read_cache_sidecar_bytes(data, *, expected_ref=None) -> dict | None:
    """Validate the exact bounded bytes being copied into/out of a backup."""
    if expected_ref is not None and not cache_ref_matches_bytes(expected_ref, data):
        return None
    arrays = npz_transport.read_bytes(
        data, max_file_bytes=_MAX_CACHE_SIDECAR_BYTES,
        max_uncompressed_bytes=32 * 1024, max_elements=2048)
    return _cache_bundle_from_arrays(arrays)


def remove_cache_sidecar(root, cache_ref) -> None:
    if isinstance(cache_ref, str) and _CACHE_REF_RE.fullmatch(cache_ref):
        try:
            (Path(root) / f'{cache_ref}.npz').unlink(missing_ok=True)
        except OSError:
            pass


def _runtime_npz(path, *, wanted_paths=None):
    if not path:
        return None
    return npz_transport.read(
        path, max_file_bytes=_MAX_RUNTIME_CACHE_BYTES,
        max_uncompressed_bytes=_MAX_RUNTIME_CACHE_BYTES,
        max_elements=_MAX_RUNTIME_CACHE_ENTRIES * 768,
        wanted_paths=wanted_paths)


def load_runtime_cache_index(score_path=None, face_path=None, *, semantic_path=None,
                             wanted_paths=None) -> dict:
    """Load strictly shaped runtime entries, retaining source signatures internally."""
    if not any(path and Path(path).is_file()
               for path in (score_path, semantic_path, face_path)):
        return {}
    def canonical(path):
        try:
            return os.path.normcase(os.path.realpath(str(path)))
        except (OSError, TypeError, ValueError):
            return str(path)

    wanted = ({str(path) for path in wanted_paths}
              if wanted_paths is not None else None)
    wanted_canonical = ({canonical(path) for path in wanted_paths}
                        if wanted_paths is not None else None)
    out = {}
    z = _runtime_npz(score_path, wanted_paths=wanted_paths)
    score_required = {'paths', 'states', 'aes', 'nsfw', 'embs'}
    if (z is not None and score_required.issubset(z)
            and set(z).issubset(score_required | {'sigs', 'hashes'})):
        try:
            paths, states, aes, nsfw, embs = (z[k] for k in ('paths', 'states', 'aes', 'nsfw', 'embs'))
            sigs = z.get('sigs')
            hashes = z.get('hashes')
            n = paths.shape[0] if len(paths.shape) == 1 else -1
            if (0 <= n <= _MAX_RUNTIME_CACHE_ENTRIES
                    and paths.shape == states.shape == aes.shape == nsfw.shape == (n,)
                    and (sigs is None or sigs.shape == (n,))
                    and (hashes is None or (
                        hashes.shape == (n, 32) and hashes.descr == '|u1'))
                    and embs.shape == (n, 768)):
                for i in range(n):
                    p = paths.string(i)
                    if (wanted is not None and p not in wanted
                            and canonical(p) not in wanted_canonical):
                        continue
                    a, ns = aes.float(i), nsfw.float(i)
                    payload = _normalized_score_cache({
                        'state': states.string(i), 'aesthetic': None if math.isnan(a) else a,
                        'nsfw': None if math.isnan(ns) else ns,
                        'embedding': embs.float_row(i),
                    })
                    sig = sigs.string(i) if sigs is not None else ''
                    digest = hashes.uint8_row(i) if hashes is not None else b''
                    if payload is not None and len(p) <= 4096 and len(sig) <= 128:
                        out.setdefault(p, {})['score'] = (payload, sig, digest)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, MemoryError):
            pass
    z = _runtime_npz(semantic_path, wanted_paths=wanted_paths)
    semantic_required = {
        'version', 'engine', 'model_id', 'revision', 'model_key', 'dimension',
        'paths', 'states', 'embs', 'sigs', 'hashes',
    }
    if z is not None and set(z) == semantic_required:
        try:
            from . import bank_semantic_models as assets
            paths, states, embs = (z[k] for k in ('paths', 'states', 'embs'))
            sigs, hashes = z['sigs'], z['hashes']
            metadata_shapes = all(z[name].shape == (1,) for name in (
                'version', 'engine', 'model_id', 'revision', 'model_key',
                'dimension'))
            provenance_ok = (
                metadata_shapes
                and z['version'].int32() == 1
                and z['engine'].string(0) == 'siglip2'
                and z['model_id'].string(0) == assets.MODEL_ID
                and z['revision'].string(0) == assets.REVISION
                and z['model_key'].string(0) == assets.MODEL_KEY
                and z['dimension'].int32() == assets.DIMENSION)
            n = paths.shape[0] if len(paths.shape) == 1 else -1
            if (provenance_ok and 0 <= n <= _MAX_RUNTIME_CACHE_ENTRIES
                    and paths.shape == states.shape == sigs.shape == (n,)
                    and hashes.shape == (n, 32) and hashes.descr == '|u1'
                    and embs.shape == (n, assets.DIMENSION)):
                for i in range(n):
                    p = paths.string(i)
                    if (wanted is not None and p not in wanted
                            and canonical(p) not in wanted_canonical):
                        continue
                    payload = _normalized_semantic_cache({
                        'state': states.string(i), 'engine': 'siglip2',
                        'model_id': assets.MODEL_ID, 'revision': assets.REVISION,
                        'model_key': assets.MODEL_KEY,
                        'dimension': assets.DIMENSION,
                        'embedding': embs.float_row(i),
                    })
                    sig = sigs.string(i)
                    digest = hashes.uint8_row(i)
                    if payload is not None and len(p) <= 4096 and len(sig) <= 128:
                        out.setdefault(p, {})['semantic'] = (payload, sig, digest)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, MemoryError):
            pass
    z = _runtime_npz(face_path, wanted_paths=wanted_paths)
    face_required = {'paths', 'states', 'dets', 'bfracs', 'embs'}
    if (z is not None and face_required.issubset(z)
            # 'bpx' (the face's pixel size, which the identity gate is taken on)
            # is tolerated but not carried: an entry transferred to another bank
            # lands legacy-shaped and keeps the verdict it already holds, which
            # is exactly what a re-run would do with it anyway.
            and set(z).issubset(
                face_required | {'yaws', 'bpx', 'sigs', 'hashes'})):
        try:
            paths, states, dets, bfracs, embs = (z[k] for k in ('paths', 'states', 'dets', 'bfracs', 'embs'))
            yaws, sigs = z.get('yaws'), z.get('sigs')
            hashes = z.get('hashes')
            n = paths.shape[0] if len(paths.shape) == 1 else -1
            if (0 <= n <= _MAX_RUNTIME_CACHE_ENTRIES
                    and paths.shape == states.shape == dets.shape == bfracs.shape == (n,)
                    and (yaws is None or yaws.shape == (n,))
                    and (sigs is None or sigs.shape == (n,))
                    and (hashes is None or (
                        hashes.shape == (n, 32) and hashes.descr == '|u1'))
                    and embs.shape == (n, 512)):
                for i in range(n):
                    p = paths.string(i)
                    if (wanted is not None and p not in wanted
                            and canonical(p) not in wanted_canonical):
                        continue
                    yaw = yaws.float(i) if yaws is not None else float('nan')
                    payload = _normalized_face_cache({
                        'state': states.string(i), 'det': dets.float(i),
                        'bbox_frac': bfracs.float(i),
                        'yaw': None if math.isnan(yaw) else yaw,
                        'embedding': embs.float_row(i),
                    })
                    sig = sigs.string(i) if sigs is not None else ''
                    digest = hashes.uint8_row(i) if hashes is not None else b''
                    if payload is not None and len(p) <= 4096 and len(sig) <= 128:
                        out.setdefault(p, {})['face'] = (payload, sig, digest)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, MemoryError):
            pass
    return out


def cache_bundle_for_transfer(index, source_path, source_bytes) -> dict:
    """Select entries valid for the bytes being copied.

    A cheap stat signature remains useful for normal runtime invalidation, but
    transfer authority is the exact SHA-256 captured by the inference process.
    Legacy caches without a digest remain resumable in their own interpreter but
    never cross a Bank/Dataset boundary.
    """
    if not isinstance(source_bytes, (bytes, bytearray)):
        return {}
    source_hash = hashlib.sha256(source_bytes).digest()
    source_key = str(source_path)
    entries = index.get(source_key) if isinstance(index, dict) else None
    if entries is None and isinstance(index, dict):
        try:
            canonical = os.path.normcase(os.path.realpath(source_key))
            aliases = [value for key, value in index.items()
                       if os.path.normcase(os.path.realpath(str(key))) == canonical]
        except (OSError, TypeError, ValueError):
            aliases = []
        # Ambiguous cache aliases fail closed instead of picking one by order.
        entries = aliases[0] if len(aliases) == 1 else None
    if not isinstance(entries, dict):
        return {}
    out = {}
    current_stat = runtime_stat_signature(source_path)
    for kind in ('score', 'semantic', 'face'):
        pair = entries.get(kind)
        if not isinstance(pair, tuple) or len(pair) != 3:
            continue
        payload, signature, digest = pair
        if not isinstance(digest, bytes) or len(digest) != 32 or digest != source_hash:
            continue
        # CLIP/SigLIP caches have carried stat signatures since their invalidation
        # contracts shipped. An empty one is malformed/stale. Face caches predate
        # signatures and retain their guarded legacy path until the next pass.
        if kind in ('score', 'semantic') and not signature:
            continue
        if signature and (not _STAT_SIG_RE.fullmatch(signature)
                          or signature != current_stat):
            continue
        out[kind] = payload
    return normalized_cache_bundle(out) or {}


def write_runtime_caches(score_path, face_path, entries, *, semantic_path=None,
                         expected_fingerprints=None):
    """Write portable bundles as path-keyed runtime NPZ caches atomically."""
    expected_fingerprints = expected_fingerprints or {}
    safe = {}
    safe_signatures = {}
    safe_hashes = {}
    for path, bundle in (entries or {}).items():
        normalized = normalized_cache_bundle(bundle)
        if not normalized:
            continue
        path = str(path)
        signature = runtime_stat_signature(path)
        if not signature:
            continue
        expected = expected_fingerprints.get(path)
        fingerprint = content_fingerprint_path(path)
        if fingerprint is None or (expected and fingerprint != expected):
            continue
        if runtime_stat_signature(path) != signature:
            continue
        safe[path] = normalized
        safe_signatures[path] = signature
        safe_hashes[path] = bytes.fromhex(fingerprint)
    if not safe:
        empty = {'score': 0, 'face': 0}
        if semantic_path is not None:
            empty['semantic'] = 0
        return empty

    def write_one(path, kind):
        selected = [(p, b[kind]) for p, b in safe.items() if kind in b]
        if not selected or not path:
            return 0
        paths = [p for p, _ in selected]
        sigs = [safe_signatures[p] for p in paths]
        if any(runtime_stat_signature(p) != sig for p, sig in zip(paths, sigs)):
            return 0
        nan = float('nan')
        common = {
            'paths': npz_transport.unicode(paths),
            'states': npz_transport.unicode([e['state'] for _, e in selected]),
            'sigs': npz_transport.unicode(sigs),
            'hashes': npz_transport.uint8(
                (byte for p in paths for byte in safe_hashes[p]),
                (len(paths), 32)),
        }
        if kind == 'score':
            arrays = {**common,
                'aes': npz_transport.floats([
                    nan if e['aesthetic'] is None else e['aesthetic'] for _, e in selected], (len(selected),)),
                'nsfw': npz_transport.floats([
                    nan if e['nsfw'] is None else e['nsfw'] for _, e in selected], (len(selected),)),
                'embs': npz_transport.floats(
                    (value for _, e in selected for value in e['embedding']),
                    (len(selected), 768)),
            }
        elif kind == 'semantic':
            from . import bank_semantic_models as assets
            arrays = {**common,
                'version': npz_transport.int32([1], (1,)),
                'engine': npz_transport.unicode(['siglip2']),
                'model_id': npz_transport.unicode([assets.MODEL_ID]),
                'revision': npz_transport.unicode([assets.REVISION]),
                'model_key': npz_transport.unicode([assets.MODEL_KEY]),
                'dimension': npz_transport.int32([assets.DIMENSION], (1,)),
                'embs': npz_transport.floats(
                    (value for _, e in selected for value in e['embedding']),
                    (len(selected), assets.DIMENSION)),
            }
        else:
            arrays = {**common,
                'dets': npz_transport.floats([e['det'] for _, e in selected], (len(selected),)),
                'bfracs': npz_transport.floats([e['bbox_frac'] for _, e in selected], (len(selected),)),
                'yaws': npz_transport.floats([
                    nan if e['yaw'] is None else e['yaw'] for _, e in selected], (len(selected),)),
                'embs': npz_transport.floats(
                    (value for _, e in selected for value in e['embedding']),
                    (len(selected), 512)),
            }
        if (any(array is None for array in arrays.values())
                or any(runtime_stat_signature(p) != sig
                       for p, sig in zip(paths, sigs))
                or any(content_fingerprint_path(p) != safe_hashes[p].hex()
                       for p in paths)):
            return 0
        if not npz_transport.write_atomic(
                path, arrays, max_file_bytes=_MAX_RUNTIME_CACHE_BYTES):
            return 0
        # A mutation during compression leaves the freshly written entry stale,
        # not blessed.  Report the lane incomplete so transfer jobs discard the
        # destination instead of announcing exact preservation.
        if (any(runtime_stat_signature(p) != sig
                for p, sig in zip(paths, sigs))
                or any(content_fingerprint_path(p) != safe_hashes[p].hex()
                       for p in paths)):
            return 0
        return len(selected)

    counts = {'score': write_one(score_path, 'score'),
              'face': write_one(face_path, 'face')}
    if semantic_path is not None:
        counts['semantic'] = write_one(semantic_path, 'semantic')
    return counts
