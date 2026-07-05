import io
import json
import os

from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


class _Proc:
    """Minimal stand-in for subprocess.CompletedProcess."""
    def __init__(self, stdout, returncode=0, stderr=''):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_generate_person_masks_returns_dict_with_results(app, monkeypatch, tmp_path):
    """generate_person_masks returns the full result dict: {"ok": true, "written": N, "results": {...}}"""
    from app.services import person_mask as pm

    monkeypatch.setattr(pm, 'is_available', lambda: True)

    # Create fake image files
    img_paths = []
    for i in range(2):
        p = os.path.join(str(tmp_path), f'img{i}.png')
        with open(p, 'wb') as fh:
            fh.write(_png())
        img_paths.append(p)

    out_dir = os.path.join(str(tmp_path), 'masks')
    contract = {"ok": True, "written": 2, "results": {
        img_paths[0]: "ok",
        img_paths[1]: "ok"
    }}
    # Noise lines before the JSON line -- the parser must pick the LAST `{`-line.
    noisy_stdout = "progress line\n[mask] loading\n" + json.dumps(contract)
    monkeypatch.setattr('app.services.person_mask.subprocess.run',
                        lambda *a, **k: _Proc(noisy_stdout))

    with app.app_context():
        result = pm.generate_person_masks(img_paths, out_dir)
        assert result['ok'] is True
        assert result['written'] == 2
        assert result['results'][img_paths[0]] == 'ok'
        assert result['results'][img_paths[1]] == 'ok'


def test_generate_person_masks_unavailable_returns_empty_without_subprocess(app, monkeypatch):
    """is_available() False -> generate_person_masks returns {}
    WITHOUT ever invoking subprocess.run (the never-raise/never-shell-out contract)."""
    from app.services import person_mask as pm

    monkeypatch.setattr(pm, 'is_available', lambda: False)

    def _boom(*a, **k):
        raise AssertionError('subprocess.run must not be called when unavailable')

    monkeypatch.setattr('app.services.person_mask.subprocess.run', _boom)
    with app.app_context():
        assert pm.generate_person_masks(['/does/not/matter'], '/out') == {}


def test_is_available_delegates_to_capability_probe(app, monkeypatch):
    """is_available() should delegate to probe_masks() and return bool."""
    from app.services import person_mask as pm
    import app.capabilities as caps_mod

    monkeypatch.setattr(caps_mod, 'probe_masks', lambda: {'ok': True, 'detail': 'x'})
    with app.app_context():
        assert pm.is_available() is True
    monkeypatch.setattr(caps_mod, 'probe_masks', lambda: {'ok': False, 'detail': 'x'})
    with app.app_context():
        assert pm.is_available() is False


def test_generate_person_masks_native_crash_returns_empty_not_exception(app, monkeypatch, tmp_path):
    """A crashed/killed subprocess (OSError) or a timeout must degrade to {},
    never raise up through the caller."""
    from app.services import person_mask as pm

    monkeypatch.setattr(pm, 'is_available', lambda: True)

    def _crash(*a, **k):
        raise OSError('the interpreter died')

    monkeypatch.setattr('app.services.person_mask.subprocess.run', _crash)
    with app.app_context():
        p = os.path.join(str(tmp_path), 'test.png')
        with open(p, 'wb') as fh:
            fh.write(_png())
        assert pm.generate_person_masks([p], str(tmp_path)) == {}


def test_generate_person_masks_stdin_payload_wire_format(app, monkeypatch, tmp_path):
    """Verify the JSON payload sent to subprocess has correct format: {"images": [...], "out_dir": ...}"""
    from app.services import person_mask as pm

    monkeypatch.setattr(pm, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = kwargs.get('input')
        return _Proc(json.dumps({"ok": True, "written": 1, "results": {}}))

    monkeypatch.setattr('app.services.person_mask.subprocess.run', _fake_run)
    with app.app_context():
        p = os.path.join(str(tmp_path), 'test.png')
        with open(p, 'wb') as fh:
            fh.write(_png())
        out_dir = os.path.join(str(tmp_path), 'masks')
        pm.generate_person_masks([p], out_dir)

    payload = json.loads(captured['input'])
    assert 'images' in payload
    assert isinstance(payload['images'], list)
    assert len(payload['images']) == 1
    assert payload['images'][0] == p
    assert payload['out_dir'] == out_dir


def test_generate_person_masks_ok_false_returns_empty(app, monkeypatch, tmp_path):
    """When subprocess returns ok: false, generate_person_masks returns {}."""
    from app.services import person_mask as pm

    monkeypatch.setattr(pm, 'is_available', lambda: True)

    def _fake_run(*a, **k):
        return _Proc(json.dumps({"ok": False, "error": "rembg init failed"}))

    monkeypatch.setattr('app.services.person_mask.subprocess.run', _fake_run)
    with app.app_context():
        p = os.path.join(str(tmp_path), 'test.png')
        with open(p, 'wb') as fh:
            fh.write(_png())
        assert pm.generate_person_masks([p], str(tmp_path)) == {}


def test_generate_person_masks_bad_json_returns_empty(app, monkeypatch, tmp_path):
    """When subprocess stdout has no valid JSON, generate_person_masks returns {}."""
    from app.services import person_mask as pm

    monkeypatch.setattr(pm, 'is_available', lambda: True)

    def _fake_run(*a, **k):
        return _Proc("progress\nno json here\n")

    monkeypatch.setattr('app.services.person_mask.subprocess.run', _fake_run)
    with app.app_context():
        p = os.path.join(str(tmp_path), 'test.png')
        with open(p, 'wb') as fh:
            fh.write(_png())
        assert pm.generate_person_masks([p], str(tmp_path)) == {}


def test_generate_person_masks_empty_paths_returns_empty(app):
    """Empty or None image_paths returns {} without processing."""
    from app.services import person_mask as pm

    with app.app_context():
        assert pm.generate_person_masks(None, '/out') == {}
        assert pm.generate_person_masks([], '/out') == {}
        assert pm.generate_person_masks([None, ''], '/out') == {}
