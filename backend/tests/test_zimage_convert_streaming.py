"""The Z-Image -> diffusers converter reads one tensor at a time, and produces the
same bytes it produced when it mapped the whole checkpoint.

WHY IT CHANGED
--------------
`build_diffusers_state_dict` opened the checkpoint with
`safetensors.torch.load_file`, which memory-maps the whole container. Windows
charges a multi-GB mapping against the system commit when it is CREATED, before
one number is read. Measured on the real `z_image_turbo_bf16.safetensors`
(11.46 GB), with `psutil` sampling the process:

    imported torch + safetensors      rss=0.20 GB   private_commit= 0.18 GB
    after load_file(whole container)  rss=0.20 GB   private_commit=11.67 GB   (0.1 s)
    after build_diffusers_state_dict  rss=5.81 GB   private_commit=14.87 GB

Resident memory did not move when the commit jumped by the size of the file:
that is a reservation, not a read. The Z-Image checkpoints on the machine this
was measured on run from 5.73 GB to 16.87 GB, so the largest one asked for about
22 GB of commit to be CONVERTED. On a 16 GB machine with a default paging file
that is `OSError 1455`, and on the machine the code is written on — 106 GB of
paging file — it is invisible. Same defect class as the one removed from
`fp8_export` the same day, on another lane.

WHAT THESE TESTS PIN
--------------------
The two properties that make the fix a fix rather than a rewrite:

  * the conversion is EQUIVALENT — byte for byte, on a synthetic checkpoint with
    the real key structure, against the old algorithm spelled out here in full so
    the comparison is against what shipped, not against the new code's opinion of
    itself;
  * nothing memory-maps any more, asserted at source level (the real defect needs
    a multi-GB file AND a machine short on commit, which is never the machine the
    test runs on — so the test says what it checks: this spelling coming back).
"""
import importlib.util
import os
import re
import struct

import pytest

torch = pytest.importorskip('torch')

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'infer', 'convert_comfy_zimage_to_diffusers.py')


@pytest.fixture(scope='module')
def conv():
    spec = importlib.util.spec_from_file_location('zconv_under_test', _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- a synthetic Z-Image checkpoint ------------------------------------------
# Real key structure, toy dimensions: 2 layers, 1 refiner each, dim 4. Small
# enough to compare byte for byte, shaped exactly like the file that ships.
DIM = 4
N_LAYERS = 2
N_REFINER = 1


def _block_keys(prefix):
    return {
        f'{prefix}.attention.qkv.weight': (3 * DIM, DIM),
        f'{prefix}.attention.q_norm.weight': (DIM,),
        f'{prefix}.attention.k_norm.weight': (DIM,),
        f'{prefix}.attention.out.weight': (DIM, DIM),
        f'{prefix}.attention.out.bias': (DIM,),
        f'{prefix}.attention_norm1.weight': (DIM,),
        f'{prefix}.attention_norm2.weight': (DIM,),
        f'{prefix}.feed_forward.w1.weight': (DIM, DIM),
        f'{prefix}.feed_forward.w2.weight': (DIM, DIM),
        f'{prefix}.feed_forward.w3.weight': (DIM, DIM),
        f'{prefix}.ffn_norm1.weight': (DIM,),
        f'{prefix}.ffn_norm2.weight': (DIM,),
        f'{prefix}.adaLN_modulation.0.weight': (DIM, DIM),
        f'{prefix}.adaLN_modulation.0.bias': (DIM,),
    }


def _checkpoint_shapes(prefix=''):
    shapes = {}
    for i in range(N_LAYERS):
        shapes.update(_block_keys(f'layers.{i}'))
    for i in range(N_REFINER):
        shapes.update(_block_keys(f'context_refiner.{i}'))
        shapes.update(_block_keys(f'noise_refiner.{i}'))
    shapes.update({
        'final_layer.linear.weight': (DIM, DIM),
        'final_layer.linear.bias': (DIM,),
        'final_layer.adaLN_modulation.1.weight': (DIM, DIM),
        'final_layer.adaLN_modulation.1.bias': (DIM,),
        'x_embedder.weight': (DIM, DIM),
        'x_embedder.bias': (DIM,),
        'x_pad_token': (DIM,),
        'cap_embedder.0.weight': (DIM,),
        'cap_embedder.1.weight': (DIM, DIM),
        'cap_embedder.1.bias': (DIM,),
        'cap_pad_token': (DIM,),
        't_embedder.mlp.0.weight': (DIM, DIM),
        't_embedder.mlp.0.bias': (DIM,),
        't_embedder.mlp.2.weight': (DIM, DIM),
        't_embedder.mlp.2.bias': (DIM,),
    })
    return {prefix + k: v for k, v in shapes.items()}


def _values(n, i):
    """Distinct values inside a tensor, and a different set per tensor — while
    staying EXACT in bfloat16.

    The obvious `arange(n) + i * 1000` is a trap this test fell into first:
    bf16 has 8 significand bits, so 28000..28047 all round to 28032 and every
    element of the tensor becomes identical. The q/k/v assertion below then
    "passed" against three identical slices, which is precisely the bug it
    exists to catch. Integers below 251 are exact in bf16, and multiplying by 7
    (coprime with 251) permutes them without repeating.
    """
    return ((torch.arange(n, dtype=torch.float32) * 7 + i * 13) % 251).to(torch.bfloat16)


@pytest.fixture
def checkpoint(tmp_path):
    """A .safetensors on disk, with distinct values per tensor so a mis-sliced
    q/k/v cannot pass by looking similar."""
    from safetensors.torch import save_file
    shapes = _checkpoint_shapes()
    sd = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        n = 1
        for d in shape:
            n *= d
        sd[name] = _values(n, i).reshape(shape)
    path = tmp_path / 'zimage_toy.safetensors'
    save_file(sd, str(path))
    return str(path), sd


# --- the OLD algorithm, verbatim, as the reference ---------------------------

def _old_build_diffusers_state_dict(conv, comfy_path):
    """What shipped before the streaming rewrite: map the whole container, strip
    the prefix, apply the ComfyUI table, clone the q/k/v slices. Kept here in
    full — comparing the new code against a re-derivation of itself would prove
    nothing."""
    from safetensors.torch import load_file
    raw = load_file(comfy_path)
    P = conv.PREFIX
    sd = {(k[len(P):] if k.startswith(P) else k): v for k, v in raw.items()}
    n_layers = max([int(m.group(1)) for k in sd if (m := re.match(r"layers\.(\d+)\.", k))], default=-1) + 1
    n_ref = max([int(m.group(1)) for k in sd if (m := re.match(r"context_refiner\.(\d+)\.", k))], default=-1) + 1
    dim = sd["layers.0.attention.qkv.weight"].shape[0] // 3
    key_map = conv.z_image_to_diffusers({"n_layers": n_layers, "dim": dim,
                                         "n_refiner_layers": n_ref})
    out = {}
    for diff_key, src in key_map.items():
        if isinstance(src, tuple):
            ck, (d, start, length) = src
            if ck in sd:
                out[diff_key] = sd[ck].narrow(d, start, length).contiguous().clone()
        elif src in sd:
            out[diff_key] = sd[src]
    return out


# --- equivalence --------------------------------------------------------------

def test_the_streamed_output_matches_the_mapped_one_tensor_for_tensor(conv, checkpoint, tmp_path):
    from safetensors.torch import load_file
    path, _sd = checkpoint
    reference = _old_build_diffusers_state_dict(conv, path)
    plan, _unmapped, _extra = conv.plan_diffusers_tensors(conv.read_header(path))
    out_path = tmp_path / 'streamed.safetensors'
    conv.write_transformer(path, plan, str(out_path))
    produced = load_file(str(out_path))

    assert set(produced) == set(reference)
    for k in sorted(reference):
        assert produced[k].shape == reference[k].shape, k
        assert produced[k].dtype == reference[k].dtype, k
        assert torch.equal(produced[k], reference[k]), k


def test_the_qkv_split_is_the_slice_and_not_a_repeat(conv, checkpoint, tmp_path):
    """The one transformation that can go wrong silently: to_q/to_k/to_v are three
    DIFFERENT windows of the fused tensor. A bug that wrote the same window three
    times would still produce the right key set, the right shapes and a file that
    loads — and a model that renders noise."""
    from safetensors.torch import load_file
    path, sd = checkpoint
    plan, _u, _e = conv.plan_diffusers_tensors(conv.read_header(path))
    out_path = tmp_path / 'streamed.safetensors'
    conv.write_transformer(path, plan, str(out_path))
    produced = load_file(str(out_path))
    fused = sd['layers.0.attention.qkv.weight']
    for i, part in enumerate(('to_q', 'to_k', 'to_v')):
        got = produced[f'layers.0.attention.{part}.weight']
        assert torch.equal(got, fused[i * DIM:(i + 1) * DIM])
    assert not torch.equal(produced['layers.0.attention.to_q.weight'],
                           produced['layers.0.attention.to_k.weight'])


def test_the_prefixed_spelling_is_handled_the_same_way(conv, tmp_path):
    """ComfyUI single-file checkpoints carry `model.diffusion_model.` in front of
    every key; some do not. Both must convert to the same thing."""
    from safetensors.torch import load_file, save_file
    shapes = _checkpoint_shapes(prefix=conv.PREFIX)
    sd = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        n = 1
        for d in shape:
            n *= d
        sd[name] = _values(n, i).reshape(shape)
    path = tmp_path / 'prefixed.safetensors'
    save_file(sd, str(path))
    plan, _u, _e = conv.plan_diffusers_tensors(conv.read_header(str(path)))
    out_path = tmp_path / 'out.safetensors'
    conv.write_transformer(str(path), plan, str(out_path))
    produced = load_file(str(out_path))
    assert 'layers.0.attention.to_q.weight' in produced
    assert not any(k.startswith(conv.PREFIX) for k in produced)


# --- the memory property, stated for what it is -------------------------------

def test_the_plan_reads_no_weight_bytes(conv, checkpoint, monkeypatch):
    """The GATE is the default mode of this CLI and it only ever compared SHAPES,
    which live in the header. It used to load the whole checkpoint to get them.

    Asserted on the READ ITSELF, not on a timing or a size: the file object is
    wrapped so every byte offset it is asked for is recorded, and nothing past
    the header may be touched."""
    path, _sd = checkpoint
    header_len = 8 + struct.unpack('<Q', open(path, 'rb').read(8))[0]
    reads = []
    real_open = open

    def spy_open(file, mode='r', *a, **kw):
        fh = real_open(file, mode, *a, **kw)
        if str(file) != path:
            return fh
        real_read = fh.read

        def read(n=-1):
            reads.append((fh.tell(), n))
            return real_read(n)
        fh.read = read
        return fh

    monkeypatch.setattr(conv, 'open', spy_open, raising=False)
    import builtins
    monkeypatch.setattr(builtins, 'open', spy_open)
    conv.plan_diffusers_tensors(conv.read_header(path))
    monkeypatch.undo()
    assert reads, 'the spy never saw the checkpoint being opened'
    assert max(start + (n if n and n > 0 else 0) for start, n in reads) <= header_len


def test_no_call_site_memory_maps_the_checkpoint(conv):
    """A SOURCE-LEVEL contract, and it says what it does not do: it catches this
    particular spelling coming back, not memory-mapping in general. The real
    defect needs a multi-GB file AND a machine short on commit, which is never
    the machine a test runs on — so it cannot be unit tested, only prevented.

    `safetensors` must not even be imported here any more: it was the only reason
    this CLI needed the package at all, and the pod it can be shipped to installs
    its own.

    Parsed, not grepped. The docstrings of this file NAME `safe_open` and
    `load_file` on purpose — they are what the comments warn against — so a text
    search either fails on the explanation or has to be weakened until it stops
    checking anything. The AST sees calls and imports and nothing else."""
    import ast
    with open(_SRC, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    called, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            called.add(fn.attr if isinstance(fn, ast.Attribute)
                       else fn.id if isinstance(fn, ast.Name) else '')
        elif isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or '').split('.')[0])
    assert 'load_file' not in called
    assert 'safe_open' not in called
    assert 'safetensors' not in imported


def test_a_failed_write_leaves_no_half_transformer(conv, checkpoint, tmp_path, monkeypatch):
    """The cache keys off the output file existing, so a partial write would be
    read as a finished conversion for ever."""
    path, _sd = checkpoint
    plan, _u, _e = conv.plan_diffusers_tensors(conv.read_header(path))
    out_path = tmp_path / 'boom.safetensors'
    calls = {'n': 0}
    real = conv._Reader.get_tensor

    def explode(self, name):
        calls['n'] += 1
        if calls['n'] > 3:
            raise RuntimeError('disk went away')
        return real(self, name)

    monkeypatch.setattr(conv._Reader, 'get_tensor', explode)
    with pytest.raises(RuntimeError):
        conv.write_transformer(path, plan, str(out_path))
    assert not out_path.exists()
    assert not (tmp_path / 'boom.safetensors.part').exists()
