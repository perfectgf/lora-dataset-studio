from pathlib import Path

import bootstrap_dependencies as bootstrap


class _Distribution:
    def __init__(self, root: Path, version: str):
        self.root = root
        self.version = version

    def locate_file(self, relative):
        return self.root / relative


class _Result:
    returncode = 0


def _pillow(tmp_path, version, plugin_source):
    pil = tmp_path / 'PIL'
    pil.mkdir()
    (pil / 'PngImagePlugin.py').write_text(plugin_source, encoding='utf-8')
    return _Distribution(tmp_path, version)


def test_coherent_pillow_12_is_not_reinstalled(tmp_path):
    dist = _pillow(tmp_path, '12.2.0', 'self._mode = "P"\n')
    calls = []
    assert bootstrap.ensure_pillow_consistent(
        distribution=dist, runner=lambda *a, **k: calls.append((a, k))
    ) is False
    assert calls == []


def test_mixed_pillow_12_is_force_reinstalled_before_import(tmp_path):
    dist = _pillow(tmp_path, '12.2.0', 'self.mode = "P"\n')
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return _Result()

    assert bootstrap.ensure_pillow_consistent(distribution=dist, runner=run) is True
    command, kwargs = calls[0]
    assert command[-2:] == ['--no-deps', 'Pillow==12.2.0']
    assert '--force-reinstall' in command
    assert kwargs == {'check': False, 'timeout': 900}


def test_legacy_pillow_can_legitimately_assign_mode(tmp_path):
    dist = _pillow(tmp_path, '9.5.0', 'self.mode = "P"\n')
    calls = []
    assert bootstrap.ensure_pillow_consistent(
        distribution=dist, runner=lambda *a, **k: calls.append((a, k))
    ) is False
    assert calls == []
