"""The unit suite must not reach a live ComfyUI.

Not a style rule — a stability one. `gpu_exclusive_vision_window` is opened by
the bank's vision passes, the caption passes, the studio and the watermark
passes, and every one of those windows fired a real
`POST http://127.0.0.1:8188/free {"unload_models": true, "free_memory": true}`
with a 10 s timeout. Measured on 2026-07-28, eleven test files did this on every
run.

What that buys is a duration nobody controls: instant when ComfyUI is not
listening, seconds when it is busy generating, ten seconds when the port answers
but the server is wedged. Tests in `test_bank_vision_concurrency` assert on
elapsed time ("Stop returns in under 5 s"), so their verdict depended on what an
unrelated program was doing at that second — the exact shape that gets filed as
"flake, sensitive to load" and, on 2026-07-17, cost a release.

It also had a side effect on the developer's machine: running `pytest` unloaded
the models of a ComfyUI that was minding its own business.
"""
import socket

import pytest


@pytest.fixture()
def connections(monkeypatch):
    """Every address this process tries to connect to while the fixture is up."""
    seen = []
    real = socket.socket.connect

    def spy(self, address):
        seen.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, 'connect', spy)
    return seen


def test_opening_a_gpu_vision_window_talks_to_nobody(app, connections):
    """THE contract. Without the conftest stub this opens a socket to ComfyUI."""
    from app.gpu_window import gpu_exclusive_vision_window
    with app.app_context():
        with gpu_exclusive_vision_window(flag_ttl=60):
            pass
    assert connections == [], (
        f'the vision window reached out over the network: {connections}')


def test_a_watermark_pass_talks_to_nobody(app, client, tmp_path, connections):
    """The window is not the only door: the pass around it must be silent too,
    so the file that was reported flaky is covered end to end."""
    import app.capabilities as caps
    import app.services.image_bank_service as svc
    import app.services.vision_ollama as vo
    from PIL import Image

    monkey_calls = []
    src = tmp_path / 'src'
    src.mkdir()
    for i in range(3):
        Image.new('RGB', (32, 32), (i * 20, i * 20, i * 20)).save(str(src / f'{i}.jpg'))
    r = client.post('/api/bank/create', json={'name': 'NET', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(caps, 'probe_ollama_model', lambda *a, **k: {'ok': True})
        mp.setattr(svc, '_gpu_busy_reason', lambda *a, **k: None)
        mp.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
        mp.setattr(vo, 'describe_image_ollama',
                   lambda *a, **k: monkey_calls.append(1) or '{"watermark": false}')
        del connections[:]                     # ignore anything the setup did
        with app.app_context():
            job = svc.start_watermark(app, 'local', bank_id, rescan=True)

    assert job['error'] is None, job['error']
    assert len(monkey_calls) == 3               # the pass really ran
    assert connections == [], (
        f'the watermark pass reached out over the network: {connections}')


def test_staging_a_source_image_talks_to_nobody(app, tmp_path, connections):
    """A third door, opened by the #64 fix: every staged write now ASKS ComfyUI
    whether it can see the file (`HEAD <comfyui>/view`).

    `api_address()` always resolves — config.py's DEFAULTS ship
    http://127.0.0.1:8188 — so without the conftest stub this HEADs a real
    address on every staging in the suite, and on a machine where ComfyUI is
    listening it gets a 404 (a tmp_path is obviously not that install's input
    folder) and REFUSES the staging. Green in CI, red on the maintainer's
    machine, which is the worst shape a test can take.

    Deterministic either way: without the stub a socket is opened whether or not
    anything answers, so this spy goes red everywhere, not just where ComfyUI
    runs."""
    from PIL import Image
    from app.utils import comfy_fs

    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    Image.new('RGB', (16, 16), (7, 7, 7)).save(src, 'PNG')
    del connections[:]                          # ignore anything the setup did
    with app.app_context():
        comfy_fs.stage_input_image(str(src), 'krea_source_abcd1234_src.png', str(folder))
    assert connections == [], (
        f'staging reached out over the network: {connections}')
