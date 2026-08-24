"""Post-training fp8 export: turn a ~26 GB bf16 dense checkpoint into the
~10 GB "scaled fp8" file ComfyUI loads natively.

WHY THIS EXISTS
---------------
A full-model (dense) run delivers a bf16 Krea 2 transformer of ~26 GB. Nobody
runs inference on that: the community ships fp8/int8 exports of ~10 GB, and
those are what people actually load in ComfyUI. Asking every user to quantize a
26 GB file by hand — after downloading it — is the difference between a run that
produced a model and a run that produced a model they can use.

WHICH FP8 FORMAT, AND WHY THAT ONE
----------------------------------
ComfyUI understands three things when it opens a diffusion checkpoint:

1. **bare fp8 cast** — every weight simply stored as ``float8_e4m3fn`` with no
   scales. ComfyUI loads it (``manual_cast``, or ``fp8_ops`` under
   ``fp8_e4m3fn_fast``), but the fast path multiplies with **scale = 1.0** and
   the values were clamped into ±448 raw: any tensor whose amax exceeds 448
   saturates, any tensor whose amax is tiny loses most of the mantissa range.
   Cheap to produce, worst fidelity.
2. **legacy "scaled fp8"** — per-weight ``<layer>.scale_weight`` (F32 scalar)
   plus ONE top-level marker tensor literally named ``scaled_fp8`` whose *dtype*
   selects the fp8 flavour and whose ``nelement() == 2`` asks for a
   full-precision matmul. Current ComfyUI converts this at load time
   (``comfy/utils.py: convert_old_quants``) into its modern representation;
   older ComfyUI builds read it natively — it *was* the native format.
3. **modern ``comfy_quant`` / ``_quantization_metadata``** — the same numbers,
   expressed as per-layer uint8 config blobs and/or a JSON blob in
   ``__metadata__``. Only recent ComfyUI builds understand it.

This module emits **(2)**. It is numerically identical to (3) — per-tensor F32
scale, ``float8_e4m3fn`` payload, dequantize-then-bf16-matmul — and it is the
only one of the two that loads on BOTH current and older ComfyUI. Choosing (3)
would buy nothing and would silently break anyone who has not updated ComfyUI.
Choosing (1) would be a quality regression we would never see in a test.

The marker is written with ``nelement() == 2``, i.e. "dequantize and do the
matrix multiply in full precision". That matches the community exports already
in use and keeps the file loadable on GPUs with no fp8 tensor cores: this export
exists to halve MEMORY, not to chase fp8 matmul speed.

WHAT GETS QUANTIZED
-------------------
Only large 2-D ``*.weight`` tensors — the attention/MLP projections, which are
where all the bytes are. Norm scales, biases, embeddings and every 1-D tensor
stay bf16: they are a rounding error in size and the first place fp8 error shows
up. ``plan_quantization`` is a pure function of the safetensors HEADER, so the
decision is testable without a single weight byte.

HOW IT RUNS
-----------
This file is BOTH a normal service module (imported by the tests and by the
cloud orchestrator, which ships its own source text to the pod) and a
self-contained CLI with no LDS imports, so the exact same code that is unit
tested here is what executes on the pod. torch / huggingface_hub are imported
lazily inside the functions that need them: importing this module on a machine
with neither must stay free. Reading and writing the safetensors format is done
here, by hand, against its own 8-byte length + JSON header — see ``_Reader``.

MEMORY
------
The writer is streaming and NOTHING is memory-mapped in either direction: the
output header is computed up front from the SOURCE header (every output size is
knowable without reading a tensor), then tensors are read, quantized and written
one at a time. Peak resident memory is one tensor, not one checkpoint — a 26 GB
input never needs 26 GB of RAM, and it never needs 26 GB of address space
either, which is the failure that actually shipped (``_Reader``).
"""
from __future__ import annotations

import json
import os
import struct
import time

# safetensors dtype strings (the on-disk spelling, not torch's).
DT_F8_E4M3 = 'F8_E4M3'
DT_F32 = 'F32'

# The marker tensor legacy ComfyUI keys the whole file on, and the per-weight
# scale suffix it pairs with. Both are ComfyUI's spelling — never rename them.
#
# The scale is a SIBLING of the weight, not a child of it: ComfyUI derives the
# module from ``k[:-len('.scale_weight')]`` and then looks for that module's
# ``.weight`` (comfy/utils.py convert_old_quants -> comfy/ops.py
# _load_quantized_module). ``blocks.0.attn.wq.weight.scale_weight`` would
# register a phantom module named ``...wq.weight`` and the real weight would
# load unquantized — hence scale_key_for(), never a plain string concatenation.
MARKER_KEY = 'scaled_fp8'
SCALE_SUFFIX = '.scale_weight'
_WEIGHT_SUFFIX = '.weight'


def scale_key_for(weight_name: str) -> str:
    """``blocks.0.attn.wq.weight`` -> ``blocks.0.attn.wq.scale_weight``."""
    name = str(weight_name)
    if name.endswith(_WEIGHT_SUFFIX):
        name = name[:-len(_WEIGHT_SUFFIX)]
    return name + SCALE_SUFFIX

# float8_e4m3fn saturates at 448. Every scale is amax(|W|) / this.
FP8_E4M3_MAX = 448.0

# Quantize a weight only when it is a 2-D matrix with at least this many
# elements. Below it the byte saving is negligible and the relative error is
# not: small projections and embedding tables keep their bf16 precision.
MIN_QUANTIZED_ELEMENTS = 1 << 20        # 1 Mi params (e.g. 1024x1024)

# Names that are never quantized however large they are. Normalisation weights
# feed a division, and modulation/embedding tables are read at full precision by
# every implementation — fp8 there is visible immediately.
_NEVER_QUANTIZE = ('norm', 'embed', 'embedding', 'pos_emb', 'time_', 'logit')

# Floating dtypes we know how to read as a quantization source.
_SOURCE_DTYPES = ('BF16', 'F16', 'F32')

# Byte width per safetensors dtype, for the streaming header arithmetic.
_DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
    'I32': 4, 'U32': 4, 'F32': 4,
    'I64': 8, 'U64': 8, 'F64': 8,
}

# A real safetensors header is well under this; past it the leading 8 bytes are
# not a header length at all.
_HEADER_LEN_MAX = 512 * 1024 * 1024


class Fp8ExportError(RuntimeError):
    """Export could not be produced. ALWAYS non-fatal to the run itself: the
    bf16 master has already been delivered by the time this runs."""


# --- header reading -------------------------------------------------------------

def read_header(path) -> dict:
    """The safetensors header (tensor index + ``__metadata__``) of ``path``.

    Header ONLY — the multi-GB weight body is never touched. Raises
    ``Fp8ExportError`` when the file is not a readable safetensors container.
    """
    try:
        with open(path, 'rb') as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError('file too short to be a safetensors container')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                raise ValueError('implausible safetensors header length')
            blob = fh.read(n)
            if len(blob) != n:
                raise ValueError('truncated safetensors header')
            obj = json.loads(blob.decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise Fp8ExportError(f'not a readable .safetensors file ({e})') from e
    if not isinstance(obj, dict):
        raise Fp8ExportError('safetensors header is not an object')
    return obj


def _entries(header: dict) -> dict:
    return {k: v for k, v in header.items()
            if k != '__metadata__' and isinstance(v, dict)}


def _torch_dtype_for(name: str):
    """safetensors dtype spelling -> torch dtype. Lazy: importing torch at module
    scope would make this file un-importable where it is only read for planning."""
    import torch
    return {
        'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
        'F64': torch.float64, 'F8_E4M3': torch.float8_e4m3fn,
        'F8_E5M2': torch.float8_e5m2, 'I64': torch.int64, 'I32': torch.int32,
        'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8,
        'BOOL': torch.bool,
    }.get(str(name).upper())


class _Reader:
    """One tensor at a time out of a .safetensors, by ordinary file I/O.

    DELIBERATELY NOT ``safetensors.safe_open``. That memory-maps the whole
    container, and on Windows a multi-GB mapping is charged against the pagefile
    the moment it is created — so opening a 25.6 GB master fails outright with
    ``OSError 1455`` ("the paging file is too small for this operation") on a
    machine with free RAM and hundreds of GB of free disk. Measured on the real
    file: the merge lane died on exactly that line before reading one tensor, and
    this lane opens the same size of file the same way.

    The machine where this was NOT reproducible has a 119 GB pagefile. That is
    the whole danger: the defect is invisible on a generously configured box and
    fatal on a default one, so it ships.

    Both passes here are strictly sequential — read a tensor, transform it, write
    it, drop it — so a mapping buys nothing anyway. Seeking to the offset the
    header already gives us has the same peak memory (one tensor), no
    address-space cost, and no dependence on the user's pagefile.

    YES, THIS DUPLICATES ``lora_merge.Reader``. On purpose, and do not
    deduplicate it: this file is shipped to a rented pod as SOURCE TEXT and run
    there as a CLI with no LDS package around it (see the module docstring). An
    import of a sibling service would work here and fail there, which is the
    worst possible place to find out.
    """

    def __init__(self, path):
        self.path = str(path)
        self._fh = open(self.path, 'rb')
        try:
            raw = self._fh.read(8)
            if len(raw) != 8:
                raise ValueError('file too short to be a safetensors container')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                raise ValueError('implausible safetensors header length')
            blob = self._fh.read(n)
            if len(blob) != n:
                raise ValueError('truncated safetensors header')
            self.header = json.loads(blob.decode('utf-8'))
            if not isinstance(self.header, dict):
                raise ValueError('safetensors header is not an object')
        except Exception as e:
            self._fh.close()
            raise Fp8ExportError(f'not a readable .safetensors file ({e})') from e
        self._start = 8 + n
        self.entries = _entries(self.header)

    def keys(self):
        return list(self.entries)

    def get_tensor(self, name):
        import torch
        spec = self.entries.get(name)
        if spec is None:
            raise Fp8ExportError(f'{name} is not in {os.path.basename(self.path)}')
        dtype = _torch_dtype_for(spec.get('dtype'))
        if dtype is None:
            raise Fp8ExportError(f'unsupported dtype {spec.get("dtype")!r} on {name}')
        begin, end = int(spec['data_offsets'][0]), int(spec['data_offsets'][1])
        nbytes = end - begin
        self._fh.seek(self._start + begin)
        # readinto a bytearray: torch.frombuffer warns once per tensor over a
        # read-only buffer, and copying afterwards would hold two copies of a
        # 200 MB tensor at the peak for nothing.
        raw = bytearray(nbytes)
        got = self._fh.readinto(raw)
        if got != nbytes:
            raise Fp8ExportError(
                f'{os.path.basename(self.path)} is truncated: {name} claims '
                f'{nbytes} bytes and only {got} are there.')
        shape = [int(d) for d in (spec.get('shape') or [])]
        return torch.frombuffer(raw, dtype=dtype).reshape(shape)

    def close(self):
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def should_quantize(name: str, dtype: str, shape) -> bool:
    """Pure predicate: does THIS tensor become fp8?

    Kept deliberately conservative — a tensor we decline to quantize costs
    bytes, a tensor we quantize wrongly costs image quality nobody can trace
    back here.
    """
    if not isinstance(name, str) or not name.endswith('.weight'):
        return False
    if dtype not in _SOURCE_DTYPES:
        return False
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return False
    low = name.lower()
    if any(token in low for token in _NEVER_QUANTIZE):
        return False
    try:
        numel = int(shape[0]) * int(shape[1])
    except (TypeError, ValueError):
        return False
    return numel >= MIN_QUANTIZED_ELEMENTS


def plan_quantization(header: dict) -> dict:
    """``{'quantize': [names], 'keep': [names], 'bytes_before', 'bytes_after'}``.

    A pure function of the header, so "what will this export look like" is
    answerable — and assertable — before any weight is read.
    """
    entries = _entries(header)
    quantize, keep = [], []
    before = after = 0
    for name, spec in sorted(entries.items()):
        dtype = spec.get('dtype')
        shape = spec.get('shape')
        width = _DTYPE_BYTES.get(dtype)
        try:
            numel = 1
            for dim in (shape or []):
                numel *= int(dim)
        except (TypeError, ValueError):
            numel = 0
        before += numel * (width or 0)
        # A checkpoint that ALREADY carries the sibling scale key is either
        # already quantized or uses that name for something else; either way,
        # overwriting it would corrupt the file silently.
        if should_quantize(name, dtype, shape) and scale_key_for(name) not in entries:
            quantize.append(name)
            after += numel + 4            # fp8 payload + the F32 scale scalar
        else:
            keep.append(name)
            after += numel * (width or 0)
    after += 2                            # the 2-element fp8 marker tensor
    return {'quantize': quantize, 'keep': keep,
            'bytes_before': before, 'bytes_after': after}


# --- quantization ---------------------------------------------------------------

def quantize_weight(tensor):
    """(fp8 payload, F32 scalar scale) for one weight, ComfyUI's own recipe.

    ``scale = amax(|W|) / 448`` and ``q = W / scale`` — a per-TENSOR scale, which
    is what ComfyUI's ``float8_e4m3fn`` layout stores and expects (per-channel
    and block scales exist, but only for nvfp4/mxfp8). An all-zero tensor would
    divide by zero, so its scale is pinned to 1.
    """
    import torch
    work = tensor.detach().to(torch.float32)
    amax = float(work.abs().max().item())
    scale = (amax / FP8_E4M3_MAX) if amax > 0 else 1.0
    if not (scale > 0) or scale != scale:          # 0, inf or NaN amax
        scale = 1.0
    q = (work / scale).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, torch.tensor(scale, dtype=torch.float32)


def dequantize_weight(payload, scale):
    """The inverse ComfyUI applies at load time — used by the tests to measure
    the error this export actually introduces."""
    import torch
    return payload.to(torch.float32) * scale.to(torch.float32)


# --- streaming writer -----------------------------------------------------------

def _raw_bytes(tensor) -> bytes:
    """Contiguous little-endian payload of any tensor, fp8 and bf16 included.

    ``numpy()`` has no float8 and no bfloat16, so those buffers are
    reinterpreted through an integer view of the SAME itemsize: exact bytes, no
    conversion of the values.
    """
    import torch
    flat = tensor.detach().to('cpu').contiguous().reshape(-1)
    view = {torch.float8_e4m3fn: torch.uint8, torch.float8_e5m2: torch.uint8,
            torch.bfloat16: torch.int16}.get(flat.dtype)
    if view is not None:
        flat = flat.view(view)
    return flat.numpy().tobytes()


def _pack_header(index: dict, metadata: dict | None) -> bytes:
    obj = dict(index)
    if metadata:
        obj['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(obj, separators=(',', ':')).encode('utf-8')
    pad = (-len(blob)) % 8              # keep the data section 8-byte aligned
    blob += b' ' * pad
    return struct.pack('<Q', len(blob)) + blob


def export_scaled_fp8(src_path, dst_path, *, metadata=None, progress=None,
                      budget_seconds=None, _now=time.monotonic) -> dict:
    """Write the scaled-fp8 twin of ``src_path`` to ``dst_path``.

    Streaming: the output header is computed from the SOURCE header, then each
    tensor is read, converted and appended once. ``budget_seconds`` bounds the
    whole conversion — an export that runs long is abandoned (and its partial
    file removed) rather than allowed to hold a rented pod hostage.

    Returns a summary dict; raises ``Fp8ExportError``. Never a run-fatal error:
    the caller has the bf16 master already delivered.
    """
    import torch                                   # noqa: F401  (fail fast, clearly)

    header = read_header(src_path)
    plan = plan_quantization(header)
    quantized = set(plan['quantize'])
    entries = _entries(header)
    if not entries:
        raise Fp8ExportError('source checkpoint declares no tensors')
    if not quantized:
        raise Fp8ExportError(
            'no tensor in this checkpoint qualifies for fp8 quantization — '
            'refusing to write a file that would only be a bf16 copy')

    # Pass 1 (header only): the exact output layout.
    order = sorted(entries)
    index, offset = {}, 0

    def place(name, dtype, shape, nbytes):
        nonlocal offset
        index[name] = {'dtype': dtype, 'shape': list(shape),
                       'data_offsets': [offset, offset + nbytes]}
        offset += nbytes

    for name in order:
        spec = entries[name]
        shape = [int(d) for d in (spec.get('shape') or [])]
        numel = 1
        for dim in shape:
            numel *= dim
        if name in quantized:
            place(name, DT_F8_E4M3, shape, numel)
            place(scale_key_for(name), DT_F32, [], 4)
        else:
            width = _DTYPE_BYTES.get(spec.get('dtype'))
            if not width:
                raise Fp8ExportError(
                    f'unsupported source dtype {spec.get("dtype")!r} on {name}')
            place(name, spec['dtype'], shape, numel * width)
    place(MARKER_KEY, DT_F8_E4M3, [2], 2)

    src_meta = header.get('__metadata__')
    out_meta = {k: v for k, v in (src_meta or {}).items()
                if isinstance(k, str) and k != '_quantization_metadata'}
    # Deliberately NOT '_quantization_metadata': its presence makes current
    # ComfyUI SKIP the legacy conversion this file relies on.
    out_meta.update({'lds_quantization': 'comfyui_scaled_fp8',
                     'lds_quantization_dtype': 'float8_e4m3fn'})
    out_meta.update({str(k): str(v) for k, v in (metadata or {}).items()})

    started = _now()
    tmp = str(dst_path) + '.part'
    written = 0
    try:
        with _Reader(src_path) as reader, open(tmp, 'wb') as out:
            out.write(_pack_header(index, out_meta))
            for i, name in enumerate(order):
                if budget_seconds and (_now() - started) > budget_seconds:
                    raise Fp8ExportError(
                        f'fp8 export exceeded its {int(budget_seconds)}s budget '
                        f'after {i}/{len(order)} tensors')
                tensor = reader.get_tensor(name)
                if name in quantized:
                    payload, scale = quantize_weight(tensor)
                    out.write(_raw_bytes(payload))
                    out.write(_raw_bytes(scale))
                else:
                    out.write(_raw_bytes(tensor))
                del tensor
                written += 1
                if progress:
                    try:
                        progress(written, len(order))
                    except Exception:
                        progress = None
            marker = torch.zeros(2, dtype=torch.float8_e4m3fn)
            out.write(_raw_bytes(marker))
        os.replace(tmp, dst_path)
    except Fp8ExportError:
        _unlink(tmp)
        raise
    except Exception as e:
        _unlink(tmp)
        raise Fp8ExportError(f'fp8 export failed: {e}') from e
    return {'path': str(dst_path), 'tensors': len(order),
            'quantized': len(quantized),
            'bytes_before': plan['bytes_before'],
            'bytes_after': os.path.getsize(dst_path),
            'seconds': round(_now() - started, 1)}


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def fp8_name_for(source_name: str) -> str:
    """``Krea_full_x_000002500.safetensors`` -> ``..._fp8.safetensors``.

    Deterministic, and it keeps the ``Krea`` prefix the Krea 2 Community Licence
    requires (and that the delivery verifier matches on).
    """
    base = os.path.basename(str(source_name or 'model.safetensors'))
    stem = base[:-len('.safetensors')] if base.endswith('.safetensors') else base
    return f'{stem}_fp8.safetensors'


def estimate_fp8_bytes(bf16_bytes) -> int:
    """Planning CEILING for ONE fp8 export of a bf16 checkpoint of that size.

    Two measurements bracket the real ratio: the reference community export of
    this family is 25.6 GB bf16 -> ~10 GB fp8 (0.40), and a synthetic checkpoint
    made entirely of quantizable matrices lands at exactly 0.50 (fp8 is half of
    bf16, and nothing else is left to shrink). Tensors we decline to quantize —
    norms, biases, embeddings — keep their size and push the ratio UP, so 0.40 is
    the observation and 0.50 the structural floor of what a fully-quantizable
    model can reach.

    This returns 0.55 because it feeds a STORAGE FORECAST: a figure that
    under-states is the exact failure this lane exists to prevent. It is
    deliberately NOT the number shown as "the export will be ~10 GB".
    """
    try:
        return int(max(0, int(bf16_bytes or 0)) * 0.55)
    except (TypeError, ValueError):
        return 0


def typical_fp8_bytes(bf16_bytes) -> int:
    """What the export will PROBABLY weigh — the observed 0.40 ratio.

    Split from ``estimate_fp8_bytes`` on purpose. A storage forecast must round
    up (0.55) or it stops protecting anything; a sentence that tells the user how
    big their download will be must not round up, or it is simply wrong. They
    answer different questions and had to stop being the same number.
    """
    try:
        return int(max(0, int(bf16_bytes or 0)) * 0.40)
    except (TypeError, ValueError):
        return 0


# --- pod entry point ------------------------------------------------------------

def _newest_checkpoint(folder, prefix='Krea'):
    best = None
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if not fn.endswith('.safetensors') or not fn.startswith(prefix):
                continue
            if fn.endswith('_fp8.safetensors'):
                continue
            path = os.path.join(root, fn)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, path)
    return best[1] if best else None


def verify_export(path) -> dict:
    """Re-open the file we just wrote and prove it is what we claimed.

    Same checks as the unit test, run on the REAL output: the fp8 marker, one
    per-tensor scale, and the payload dtype. A conversion that produced an
    unloadable file must say so here rather than at generation time, days later.

    It lives HERE, next to the writer, because both need torch — and the app
    server is not allowed to (see fp8_quantize's module note). One
    implementation, run wherever the conversion runs.

    It reads through ``_Reader`` for the same reason the writer does: this opens
    the ~13 GB file we just produced, which is squarely in the range where a
    memory mapping fails on a default Windows pagefile. Fixing only the source
    side would have moved the crash from the start of the conversion to its last
    second — after twenty minutes of work — which is worse than not fixing it.
    """
    out = {'verified': False, 'verify_error': None}
    try:
        with _Reader(path) as fh:
            keys = list(fh.keys())
            if MARKER_KEY not in keys:
                raise ValueError('the scaled-fp8 marker is missing')
            marker = fh.get_tensor(MARKER_KEY)
            scales = [k for k in keys if k.endswith(SCALE_SUFFIX)]
            if not scales:
                raise ValueError('no per-tensor scale was written')
            weight = scales[0][:-len(SCALE_SUFFIX)] + '.weight'
            payload = fh.get_tensor(weight)
            scale = fh.get_tensor(scales[0])
        import torch
        if marker.dtype is not torch.float8_e4m3fn or marker.nelement() != 2:
            raise ValueError('the marker does not describe float8_e4m3fn weights')
        if payload.dtype is not torch.float8_e4m3fn:
            raise ValueError(f'{weight} was not written as float8_e4m3fn')
        if scale.dtype is not torch.float32 or scale.ndim != 0:
            raise ValueError('the per-tensor scale is not a float32 scalar')
        out.update(verified=True, scaled_tensors=len(scales))
    except Exception as e:                       # noqa: BLE001 — reported
        out['verify_error'] = str(e)[:300]
    return out


# One line per converted tensor, read by the parent process to drive a progress
# bar. Deliberately not JSON: it is parsed off a stream that also carries torch's
# own chatter, so it has to be recognisable at a glance and cheap to emit.
PROGRESS_PREFIX = 'LDS_FP8_PROGRESS'

# The single line this CLI ends on, whichever machine runs it. `dense_fp8_delivery`
# declares the same string for the pod lane it parses; defining it HERE too keeps
# the local worker from importing that module just to read a constant.
RESULT_PREFIX = 'LDS_FP8_RESULT'


def main(argv=None) -> int:
    """CLI, executed on the pod AND — with ``--dst`` — as this app's own worker.

    Prints exactly one ``LDS_FP8_RESULT`` JSON line on stdout, plus progress
    lines when asked. Fail-open by contract on the pod: a non-zero exit means
    "no fp8 export", never "the run failed" — the bf16 master was pushed by the
    trainer before this ever ran.
    """
    import argparse
    parser = argparse.ArgumentParser(prog='lds-fp8-export')
    parser.add_argument('--src', default='')
    parser.add_argument('--src-dir', default='')
    parser.add_argument('--dst', default='')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--prefix', default='Krea')
    parser.add_argument('--repo-id', default='')
    parser.add_argument('--budget-seconds', type=int, default=1800)
    parser.add_argument('--drop-bf16', action='store_true')
    # The Hugging Face token arrives in a FILE, never on the command line: a
    # command line is visible in `ps` to every process on the pod and is stored
    # verbatim by the provider alongside the command's output.
    parser.add_argument('--token-file', default='')
    args = parser.parse_args(argv)
    if args.token_file and os.path.isfile(args.token_file):
        with open(args.token_file, encoding='utf-8') as fh:
            token = fh.read().strip()
        if token:
            os.environ['HF_TOKEN'] = token

    result = {'ok': False, 'error': None, 'path': None, 'uploaded': False,
              'dropped_bf16': False}
    try:
        src = args.src or _newest_checkpoint(args.src_dir, args.prefix)
        if not src:
            raise Fp8ExportError('no dense checkpoint found to quantize')
        dst = args.dst or os.path.join(os.path.dirname(src), fp8_name_for(src))

        def report(done, total):
            print(f'{PROGRESS_PREFIX} {done} {total}', flush=True)

        summary = export_scaled_fp8(src, dst, budget_seconds=args.budget_seconds,
                                    progress=report if args.progress else None,
                                    metadata={'lds_quantized_from': os.path.basename(src)})
        result.update(ok=True, **summary, **verify_export(dst))
        if args.repo_id:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get('HF_TOKEN') or None)
            api.upload_file(path_or_fileobj=dst, path_in_repo=os.path.basename(dst),
                            repo_id=args.repo_id, repo_type='model')
            result['uploaded'] = True
            if args.drop_bf16:
                # Only ever after the fp8 twin is confirmed uploaded: losing the
                # master to a failed upload would destroy the run's only output.
                api.delete_file(path_in_repo=os.path.basename(src),
                                repo_id=args.repo_id, repo_type='model')
                result['dropped_bf16'] = True
    except Exception as e:                       # noqa: BLE001 — reported, not raised
        result['error'] = str(e)[:500]
    print(RESULT_PREFIX + ' ' + json.dumps(result))
    return 0 if result['ok'] else 1


if __name__ == '__main__':                       # pragma: no cover
    raise SystemExit(main())
