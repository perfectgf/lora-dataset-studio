"""model_integrity: is a weights file REAL, loadable weights, or just a file that
happens to have the right name and extension?

These pin the two production incidents the module exists for — a licence-gate HTML
page saved as ``<name>.safetensors`` (Setup went green, then ``UNETLoader:
Expecting value: line 1 column 1``) and a truncated / half-symlinked download — and
the two guarantees that keep it cheap enough to run from the probes: it NEVER reads
the multi-GB weight body, and it caches the structural verdict by (path, mtime, size)."""
import json
import struct

import pytest

from app.services import model_integrity as mi


def _header(meta: dict) -> bytes:
    """A safetensors container header: 8-byte LE length + that many bytes of JSON."""
    body = json.dumps(meta).encode('utf-8')
    return struct.pack('<Q', len(body)) + body


_TENSOR = {'w': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}


@pytest.fixture(autouse=True)
def _isolate_cache():
    mi.clear_cache()
    yield
    mi.clear_cache()


def test_valid_safetensors_header_is_valid(tmp_path):
    p = tmp_path / 'unet.safetensors'
    p.write_bytes(_header(_TENSOR) + b'\x00' * 64)
    r = mi.validate_model_file(str(p))
    assert r['verdict'] == 'valid'
    assert r['ok'] is True
    assert r['blocking'] is False


def test_html_gate_page_saved_as_safetensors_is_blocking(tmp_path):
    # The #help case: a browser download of a gated model WITHOUT accepting the
    # licence saves the HTML gate page. The 8 leading bytes read as a uint64 are
    # an absurd "header length", and the content starts with '<'.
    p = tmp_path / 'flux-2-klein-9b-fp8.safetensors'
    p.write_bytes(b'<!doctype html>\n<html><head><title>Access gated</title></head></html>')
    r = mi.validate_model_file(str(p))
    assert r['verdict'] == 'html_or_text'
    assert r['blocking'] is True
    assert r['ok'] is False
    assert 'HTML' in r['reason']
    assert 'flux-2-klein-9b-fp8.safetensors' in r['reason']   # names the file to delete


def test_git_lfs_pointer_is_blocking(tmp_path):
    p = tmp_path / 'vae.safetensors'
    p.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 327\n')
    r = mi.validate_model_file(str(p))
    assert r['verdict'] == 'html_or_text'
    assert r['blocking'] is True
    assert 'LFS' in r['reason']


def test_json_error_body_is_blocking(tmp_path):
    p = tmp_path / 'te.safetensors'
    p.write_bytes(b'{"error":"You must accept the license to access this repository"}')
    r = mi.validate_model_file(str(p))
    assert r['verdict'] == 'html_or_text'
    assert r['blocking'] is True


def test_truncated_header_is_blocking(tmp_path):
    # Declares a 4 KB header but the file is only a few bytes long — a download that
    # was interrupted before even the tensor index finished.
    p = tmp_path / 'te.safetensors'
    p.write_bytes(struct.pack('<Q', 4096) + b'{"w":')
    r = mi.validate_model_file(str(p))
    assert r['verdict'] == 'truncated_or_garbage'
    assert r['blocking'] is True


def test_plausible_length_but_non_json_header_is_garbage(tmp_path):
    p = tmp_path / 'm.safetensors'
    p.write_bytes(struct.pack('<Q', 16) + b'not json at all!' + b'\x00' * 64)
    assert mi.validate_model_file(str(p))['verdict'] == 'truncated_or_garbage'


def test_empty_file_is_blocking_garbage(tmp_path):
    p = tmp_path / 'stub.safetensors'
    p.write_bytes(b'')
    r = mi.validate_model_file(str(p))
    assert r['verdict'] == 'truncated_or_garbage'
    assert r['blocking'] is True


def test_gguf_magic_is_valid(tmp_path):
    p = tmp_path / 'model.gguf'
    p.write_bytes(b'GGUF' + b'\x00' * 60)
    assert mi.validate_model_file(str(p))['verdict'] == 'valid'


def test_missing_file_is_missing_and_not_blocking(tmp_path):
    r = mi.validate_model_file(str(tmp_path / 'nope.safetensors'))
    assert r['verdict'] == 'missing'
    assert r['blocking'] is False   # the missing-asset preflight owns this case, not us


def test_too_small_is_advisory_not_blocking(tmp_path):
    # Structurally VALID header but the whole file is orders of magnitude below the
    # type floor: a download that stopped right after the header. Warn, never block.
    p = tmp_path / 'unet.safetensors'
    p.write_bytes(_header(_TENSOR) + b'\x00\x00')
    r = mi.validate_model_file(str(p), min_bytes=1024 ** 3)
    assert r['verdict'] == 'too_small'
    assert r['ok'] is False
    assert r['blocking'] is False
    assert 'smaller' in r['reason']


def test_valid_above_floor_stays_valid(tmp_path):
    p = tmp_path / 'unet.safetensors'
    p.write_bytes(_header(_TENSOR) + b'\x00' * 4096)
    assert mi.validate_model_file(str(p), min_bytes=100)['verdict'] == 'valid'


def test_header_only_never_reads_the_weight_body(tmp_path, monkeypatch):
    # A "large" model: a valid header followed by a big body standing in for the
    # multi-GB weights. The validator must read only the header — assert the total
    # bytes read are a tiny fraction of the file (this is the whole cost model).
    body_size = 5 * 1024 * 1024
    p = tmp_path / 'big.safetensors'
    p.write_bytes(_header(_TENSOR) + b'\x00' * body_size)
    total = p.stat().st_size

    read = {'n': 0}
    real_open = mi._open

    class _Counting:
        def __init__(self, fh):
            self._fh = fh

        def read(self, n=-1):
            data = self._fh.read(n)
            read['n'] += len(data)
            return data

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, *a):
            return self._fh.__exit__(*a)

    monkeypatch.setattr(mi, '_open', lambda path, mode='rb': _Counting(real_open(path, mode)))
    assert mi.validate_model_file(str(p))['verdict'] == 'valid'
    assert read['n'] < 64 * 1024     # << the 5 MB file
    assert read['n'] < total


def test_structural_verdict_is_cached_per_path_mtime_size(tmp_path, monkeypatch):
    p = tmp_path / 'unet.safetensors'
    p.write_bytes(_header(_TENSOR) + b'\x00' * 4096)

    opens = {'n': 0}
    real_open = mi._open

    def _counting_open(path, mode='rb'):
        opens['n'] += 1
        return real_open(path, mode)

    monkeypatch.setattr(mi, '_open', _counting_open)
    mi.clear_cache()
    first = mi.validate_model_file(str(p))
    second = mi.validate_model_file(str(p))
    assert first['verdict'] == second['verdict'] == 'valid'
    assert opens['n'] == 1           # second call served from cache — no re-open


def test_cache_reapplies_floor_without_reopening(tmp_path, monkeypatch):
    # The too_small floor is applied AFTER the cache (a cheap size compare), so the
    # same file can be judged against different floors without re-reading its header.
    p = tmp_path / 'unet.safetensors'
    p.write_bytes(_header(_TENSOR) + b'\x00' * 64)

    opens = {'n': 0}
    real_open = mi._open
    monkeypatch.setattr(mi, '_open',
                        lambda path, mode='rb': (opens.__setitem__('n', opens['n'] + 1)
                                                 or real_open(path, mode)))
    mi.clear_cache()
    assert mi.validate_model_file(str(p))['verdict'] == 'valid'            # no floor
    assert mi.validate_model_file(str(p), min_bytes=1024 ** 3)['verdict'] == 'too_small'
    assert opens['n'] == 1
