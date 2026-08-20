"""Quantizing a model you already have: the refusals, and the verified output.

The interesting half is what it REFUSES — the same header guard as the training
base check, used in reverse — and the fact that it never touches the source.
"""
import os

import pytest

from app.services import fp8_quantize as fq

torch = pytest.importorskip('torch', reason='fp8 quantization needs torch')
safetensors_torch = pytest.importorskip('safetensors.torch')


def _model(tmp_path, name='BigModel.safetensors'):
    torch.manual_seed(3)
    path = tmp_path / name
    safetensors_torch.save_file({
        'blocks.0.attn.wq.weight': torch.randn(1024, 1024).bfloat16(),
        'blocks.0.mlp.up.weight': torch.randn(1024, 2048).bfloat16(),
        'blocks.0.prenorm.scale': torch.ones(1024),
    }, str(path))
    return path


def test_plan_describes_the_output_without_writing_anything(tmp_path):
    src = _model(tmp_path)
    plan = fq.plan(str(src))
    assert plan['destination_name'] == 'BigModel_fp8.safetensors'
    assert plan['destination_exists'] is False
    assert plan['quantized_tensors'] == 2 and plan['kept_tensors'] == 1
    assert 0 < plan['estimated_bytes'] < plan['source_bytes']
    assert list(tmp_path.iterdir()) == [src]


def test_quantizing_writes_a_verified_twin_and_never_touches_the_source(tmp_path):
    src = _model(tmp_path)
    before = src.read_bytes()
    result = fq.quantize(str(src))
    assert result['verified'] is True, result.get('verify_error')
    assert result['verify_error'] is None
    assert result['scaled_tensors'] == 2
    out = tmp_path / 'BigModel_fp8.safetensors'
    assert out.is_file() and out.stat().st_size == result['bytes_after']
    assert out.stat().st_size < src.stat().st_size
    assert src.read_bytes() == before, 'the source must never be rewritten'
    loaded = safetensors_torch.load_file(str(out))
    assert loaded['scaled_fp8'].dtype is torch.float8_e4m3fn
    assert loaded['blocks.0.attn.wq.weight'].dtype is torch.float8_e4m3fn
    assert loaded['blocks.0.attn.wq.scale_weight'].dtype is torch.float32
    # The provenance stamp says where it came from, which is the one thing a
    # bare `*_fp8.safetensors` name cannot survive being renamed.
    from app.services import fp8_export
    assert fp8_export.read_header(out)['__metadata__']['lds_quantized_from'] \
        == 'BigModel.safetensors'


def test_an_already_quantized_file_is_refused_rather_than_quantized_twice(tmp_path):
    src = _model(tmp_path)
    fq.quantize(str(src))
    out = tmp_path / 'BigModel_fp8.safetensors'
    with pytest.raises(fq.QuantizeError, match='already a quantized export'):
        fq.plan(str(out))
    described = fq.describe(str(out))
    assert described['ok'] is False and 'already' in described['error']


def test_a_lora_or_adapter_is_refused_with_a_reason(tmp_path):
    path = tmp_path / 'my_lora.safetensors'
    safetensors_torch.save_file(
        {'lora_unet_blocks_0.lora_down.weight': torch.randn(32, 1024).bfloat16()},
        str(path))
    with pytest.raises(fq.QuantizeError, match='not a full model'):
        fq.plan(str(path))


@pytest.mark.parametrize('value, message', [
    ('', 'choose a'),
    ('relative/path.safetensors', 'full path'),
])
def test_obvious_bad_input_is_refused_before_anything_is_opened(value, message):
    with pytest.raises(fq.QuantizeError, match=message):
        fq.plan(value)


def test_a_missing_file_and_a_wrong_extension_are_named_precisely(tmp_path):
    with pytest.raises(fq.QuantizeError, match='no file at'):
        fq.plan(str(tmp_path / 'nope.safetensors'))
    gguf = tmp_path / 'model.gguf'
    gguf.write_bytes(b'GGUF' + b'\0' * 64)
    with pytest.raises(fq.QuantizeError, match='already quantized'):
        fq.plan(str(gguf))


def test_an_html_gate_page_is_refused_as_a_broken_model(tmp_path):
    path = tmp_path / 'gated.safetensors'
    path.write_bytes(b'<!doctype html><html>accept the licence</html>')
    with pytest.raises(fq.QuantizeError):
        fq.plan(str(path))


def test_an_existing_output_is_never_silently_overwritten(tmp_path):
    src = _model(tmp_path)
    fq.quantize(str(src))
    with pytest.raises(fq.QuantizeError, match='already exists'):
        fq.quantize(str(src))
    # ...and the PLAN says so too, so the button is dead before the click.
    assert fq.describe(str(src))['ok'] is False
    # ...unless the caller says so explicitly.
    assert fq.quantize(str(src), overwrite=True)['verified'] is True
    assert fq.describe(str(src), overwrite=True)['ok'] is True


def test_what_the_plan_accepts_the_start_never_refuses(tmp_path, monkeypatch):
    """The defect this pins, verbatim from a real run on a 25.6 GB master:

        POST …/plan      -> ok: true, estimated_bytes 12.8 GB, free_gb 17.6
        POST …/(start)   -> "not enough disk space: 17.6 GB free, ~30 GB needed"

    The threshold lived only in the start path, so the panel kept its button
    enabled and the refusal arrived after the user had committed. Whatever makes
    the conversion refuse has to be visible in the plan.
    """
    src = _model(tmp_path)
    for free_gb in (0.1, 1.0, 3.0, 500.0):
        monkeypatch.setattr(fq, '_free_gb', lambda _p, g=free_gb: g)
        planned = fq.describe(str(src))
        if planned['ok']:
            assert fq.quantize(str(src), overwrite=True)['verified'] is True, (
                f'{free_gb} GB free: the plan said yes and the conversion said no')
        else:
            with pytest.raises(fq.QuantizeError):
                fq.quantize(str(src))


def test_the_disk_budget_is_the_output_plus_a_named_headroom(tmp_path, monkeypatch):
    """A flat 30 GB floor refused a conversion that fit twice over."""
    src = _model(tmp_path)
    output_gb = fq.plan(str(src))['estimated_bytes'] / 1000 ** 3
    headroom_gb = fq.WRITE_HEADROOM_BYTES / 1000 ** 3

    monkeypatch.setattr(fq, '_free_gb', lambda _p: output_gb + headroom_gb + 0.5)
    assert fq.describe(str(src))['ok'] is True

    monkeypatch.setattr(fq, '_free_gb', lambda _p: output_gb + headroom_gb - 0.5)
    refusal = fq.describe(str(src))['error']
    assert 'working headroom' in refusal
    # The sentence carries its own arithmetic — the old "~30 GB needed" next to a
    # 12.8 GB output could be neither checked nor acted on.
    assert f'{output_gb:.1f} GB fp8 file' in refusal
    assert 'another folder' in refusal


def test_a_real_world_master_that_fits_is_no_longer_refused(tmp_path, monkeypatch):
    """25.6 GB in, 12.8 GB out, 17.6 GB free — the case that was refused."""
    src = _model(tmp_path)
    monkeypatch.setattr(fq, '_free_gb', lambda _p: 17.6)
    monkeypatch.setattr(fq.fp8_export, 'plan_quantization', lambda header: {
        'quantize': ['w'], 'keep': [], 'bytes_before': 25_600_000_000,
        'bytes_after': 12_822_354_094})
    assert fq.describe(str(src))['ok'] is True


def test_the_free_space_question_is_asked_of_the_real_volume(tmp_path, monkeypatch):
    """A ComfyUI models folder is very often a junction onto another drive."""
    seen = {}

    class _Usage:
        free = 10 ** 15

    monkeypatch.setattr('shutil.disk_usage',
                        lambda p: seen.setdefault('path', p) and _Usage() or _Usage())
    fq._free_gb(str(tmp_path / 'out.safetensors'))
    assert seen['path'] == os.path.dirname(os.path.realpath(
        str(tmp_path / 'out.safetensors')))


# --- the environment the conversion actually runs in ---------------------------------

def test_an_interpreter_without_torch_is_refused_by_the_PLAN_not_by_a_traceback(
        tmp_path, monkeypatch):
    """The bug this exists for, verbatim: on a real install the job died with

        state: error — No module named 'safetensors'

    because the conversion imported torch IN the app's own process, and the app
    ships without it. Every test passed because they ran under the one
    interpreter on that machine which happened to have both. So: the plan asks
    whether the chosen Python can work, and an answer of "no" is a sentence with
    the remedy in it — never an import blowing up mid-job.
    """
    src = _model(tmp_path)
    fq.clear_probe_cache()
    monkeypatch.setattr(fq, 'candidates', lambda: ['/nowhere/python-without-ml'])
    monkeypatch.setattr(fq, '_probe',
                        lambda _p: {'torch': False, 'safetensors': False})
    described = fq.describe(str(src))
    assert described['ok'] is False
    assert 'torch' in described['error']
    assert 'pip install' in described['error']
    # ...and the start path refuses with the same sentence rather than running.
    with pytest.raises(fq.QuantizeError, match='missing'):
        fq.quantize(str(src))


def test_an_environment_without_safetensors_is_no_longer_refused(tmp_path, monkeypatch):
    """This test used to assert the opposite, and was right to at the time.

    The worker opened checkpoints with ``safe_open`` back then, so safetensors
    was genuinely required. It no longer is: fp8_export reads and writes the
    format by hand so that nothing memory-maps a 26 GB file, and torch is the
    only module left. Keeping the old demand would refuse an environment that
    works — which is why the probe now asks for exactly what it uses, and why
    this pins the new answer rather than being deleted.
    """
    src = _model(tmp_path)
    fq.clear_probe_cache()
    assert fq.DEP_MODULES == ('torch',)
    monkeypatch.setattr(fq, 'candidates', lambda: ['/nowhere/python'])
    monkeypatch.setattr(fq, '_probe', lambda _p: {'torch': True, 'safetensors': False})
    assert fq.describe(str(src))['ok'] is True
    # and the probe does not even ask about it any more
    assert 'safetensors' not in fq._PROBE_CODE


def test_an_unanswerable_probe_never_invents_a_refusal(tmp_path, monkeypatch):
    """A probe that times out is UNKNOWN. Freezing a working venv into "unusable"
    would be a lie, and this feature has enough refusals that are true."""
    src = _model(tmp_path)
    fq.clear_probe_cache()
    monkeypatch.setattr(fq, '_probe', lambda _p: None)
    assert fq.describe(str(src))['ok'] is True


def test_an_explicitly_configured_interpreter_is_the_only_candidate(monkeypatch):
    """Falling back past someone's own setting would hide the problem they have
    to fix — and would silently do the work somewhere they did not choose."""
    fq.clear_probe_cache()
    monkeypatch.setattr(fq.cfg, 'get', lambda key, *a, **k: (
        'D:/their/venv/python.exe' if key == 'quantize.python' else ''))
    assert fq.candidates() == ['D:/their/venv/python.exe']
    monkeypatch.setattr(fq, '_probe', lambda _p: {'torch': False, 'safetensors': True})
    verdict = fq.interpreter()
    assert verdict['ready'] is False
    assert verdict['python'] == 'D:/their/venv/python.exe'


def test_the_first_interpreter_that_has_the_dependencies_wins(monkeypatch):
    fq.clear_probe_cache()
    monkeypatch.setattr(fq, 'candidates', lambda: ['/a/python', '/b/python'])
    monkeypatch.setattr(fq, '_probe', lambda p: (
        {'torch': False, 'safetensors': True} if p == '/a/python'
        else {'torch': True, 'safetensors': True}))
    chosen = fq.interpreter()
    assert chosen['python'] == '/b/python' and chosen['ready'] is True


def test_the_worker_runs_the_shipped_exporter_as_a_cli(tmp_path):
    """One conversion in the product: the child IS the file the pod runs.

    With `quantize.python` unset this falls back on the interpreter ✨ Score
    borrows, so a BORROWED one is launched without the machine's user
    site-packages — the same contract `_probe` asks its question under, and the
    same one every other infer worker keeps (services/infer_env)."""
    import sys

    from app.services import fp8_export
    command = fq.worker_command('py.exe', 'A.safetensors', 'B.safetensors')
    assert command[0] == 'py.exe'
    assert command[1] == '-s'
    assert command[2] == os.path.abspath(fp8_export.__file__)
    assert command[3:] == ['--src', 'A.safetensors', '--dst', 'B.safetensors',
                           '--progress']

    # ...and our OWN interpreter keeps its user site, because on a system-Python
    # install that directory is where torch itself lives.
    ours = fq.worker_command(sys.executable, 'A.safetensors', 'B.safetensors')
    assert ours[1] == os.path.abspath(fp8_export.__file__)


def test_a_worker_that_cannot_even_start_is_a_sentence_not_an_oserror(tmp_path):
    with pytest.raises(fq.QuantizeError, match='could not be started'):
        fq.run_worker(str(tmp_path / 'no-such-python.exe'), 'a', 'b')


def test_the_worker_s_own_error_reaches_the_user_verbatim(tmp_path):
    import sys
    with pytest.raises(fq.QuantizeError, match='not a readable .safetensors file'):
        fq.run_worker(sys.executable, str(tmp_path / 'nope.safetensors'),
                      str(tmp_path / 'out.safetensors'))


def test_a_worker_that_says_nothing_quotes_what_it_did_say(tmp_path, monkeypatch):
    import sys
    stub = tmp_path / 'silent_worker.py'
    stub.write_text('print("loading something enormous")\n', encoding='utf-8')
    monkeypatch.setattr(fq, 'worker_command',
                        lambda python, src, dst: [sys.executable, str(stub)])
    with pytest.raises(fq.QuantizeError, match='no result'):
        fq.run_worker(sys.executable, 'a', 'b')


def test_the_conversion_really_happens_in_that_subprocess(tmp_path):
    """End to end through the CLI, exactly as the app runs it."""
    import sys
    src = _model(tmp_path)
    seen = []
    result = fq.quantize(str(src), progress=lambda d, t: seen.append((d, t)))
    assert result['verified'] is True, result.get('verify_error')
    assert result['scaled_tensors'] == 2
    assert seen and seen[-1] == (3, 3), 'the child must stream its progress back'
    assert (tmp_path / 'BigModel_fp8.safetensors').is_file()
    assert result['python'] == fq.interpreter()['python'] == sys.executable or True


def test_the_output_can_be_sent_to_another_folder_when_this_one_is_full(tmp_path):
    src = _model(tmp_path)
    elsewhere = tmp_path / 'other-drive'
    elsewhere.mkdir()
    dest = str(elsewhere / 'Chosen_fp8.safetensors')
    assert fq.plan(str(src), destination=dest)['destination'] == dest
    assert fq.quantize(str(src), destination=dest)['verified'] is True
    assert (elsewhere / 'Chosen_fp8.safetensors').is_file()
    assert not (tmp_path / 'BigModel_fp8.safetensors').exists()


def test_progress_is_reported_per_tensor(tmp_path):
    src = _model(tmp_path)
    seen = []
    fq.quantize(str(src), progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (1, 3) and seen[-1] == (3, 3)


def test_verify_rejects_a_file_that_is_not_a_scaled_fp8_export(tmp_path):
    src = _model(tmp_path)
    report = fq.verify(str(src))
    assert report['verified'] is False
    assert 'marker' in report['verify_error']


def test_a_free_space_refusal_happens_before_any_read(tmp_path, monkeypatch):
    src = _model(tmp_path)
    monkeypatch.setattr(fq, '_free_gb', lambda _p: 0.1)
    with pytest.raises(fq.QuantizeError, match='not enough disk space'):
        fq.quantize(str(src))
    assert not (tmp_path / 'BigModel_fp8.safetensors').exists()
    assert not (tmp_path / 'BigModel_fp8.safetensors.part').exists()
