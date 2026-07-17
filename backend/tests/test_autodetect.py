"""Setup auto-detection: discover installed tools so the wizard fills config itself.

A reachable default port (url/api_url) is a hard signal (auto-apply); a folder
found on disk (base_dir/dir) is a suggestion. Network + filesystem are stubbed.
"""
from app import capabilities as cap
from app.routes import setup as setup_routes  # noqa: F401 - ensures blueprint import


def test_autodetect_ollama_reachable_picks_vl_model(monkeypatch):
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: '11434' in url)  # ollama up, comfy down
    monkeypatch.setattr(cap, '_ollama_tags', lambda url, timeout=3: ['llama3:8b', 'qwen3-vl:8b'])
    monkeypatch.setattr(cap, '_find_install_dir', lambda names, marker: '')
    r = cap.autodetect()
    assert r['ollama'] == {'url': cap._OLLAMA_DEFAULT_URL, 'vision_model': 'qwen3-vl:8b'}
    assert r['comfyui'] == {} and r['aitoolkit'] == {}


_ABLIT_INSTRUCT = 'huihui_ai/qwen3-vl-abliterated:8b-instruct'


def test_autodetect_prefers_abliterated_over_vanilla_instruct(monkeypatch):
    """Both -instruct tags installed: the uncensored abliterated build must win over the
    censored vanilla one (which refuses the NSFW describe/caption work the app relies on)."""
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: '11434' in url)
    monkeypatch.setattr(cap, '_ollama_tags',
                        lambda url, timeout=3: ['qwen3-vl:8b-instruct', _ABLIT_INSTRUCT])
    monkeypatch.setattr(cap, '_find_install_dir', lambda names, marker: '')
    assert cap.autodetect()['ollama']['vision_model'] == _ABLIT_INSTRUCT


def test_autodetect_abliterated_only(monkeypatch):
    """Abliterated is the only vision model present -> it is selected."""
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: '11434' in url)
    monkeypatch.setattr(cap, '_ollama_tags', lambda url, timeout=3: ['llama3:8b', _ABLIT_INSTRUCT])
    monkeypatch.setattr(cap, '_find_install_dir', lambda names, marker: '')
    assert cap.autodetect()['ollama']['vision_model'] == _ABLIT_INSTRUCT


def test_autodetect_vanilla_only_unchanged(monkeypatch):
    """No abliterated build installed: the prior instruct-over-thinking pick is unchanged."""
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: '11434' in url)
    monkeypatch.setattr(cap, '_ollama_tags',
                        lambda url, timeout=3: ['qwen3-vl:8b', 'qwen3-vl:8b-instruct'])
    monkeypatch.setattr(cap, '_find_install_dir', lambda names, marker: '')
    assert cap.autodetect()['ollama']['vision_model'] == 'qwen3-vl:8b-instruct'


def test_autodetect_ollama_up_but_no_vl_model(monkeypatch):
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: '11434' in url)
    monkeypatch.setattr(cap, '_ollama_tags', lambda url, timeout=3: ['llama3:8b'])  # no vision model
    monkeypatch.setattr(cap, '_find_install_dir', lambda names, marker: '')
    assert cap.autodetect()['ollama'] == {'url': cap._OLLAMA_DEFAULT_URL}  # url only, no vision_model


def test_autodetect_comfyui_port_and_disk_paths(monkeypatch):
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: '8188' in url)  # comfy up, ollama down
    def fake_find(names, marker):
        if 'ComfyUI' in names:
            return 'C:/ComfyUI'
        if 'ai-toolkit' in names:
            return 'C:/ai-toolkit'
        return ''
    monkeypatch.setattr(cap, '_find_install_dir', fake_find)
    r = cap.autodetect()
    assert r['ollama'] == {}
    assert r['comfyui'] == {'api_url': cap._COMFYUI_DEFAULT_URL, 'base_dir': 'C:/ComfyUI'}
    assert r['aitoolkit'] == {'dir': 'C:/ai-toolkit'}


def test_autodetect_nothing_found(monkeypatch):
    monkeypatch.setattr(cap, '_http_ok', lambda url, timeout=3: False)
    monkeypatch.setattr(cap, '_find_install_dir', lambda names, marker: '')
    assert cap.autodetect() == {'ollama': {}, 'comfyui': {}, 'aitoolkit': {}}


def test_autodetect_route(client, monkeypatch):
    monkeypatch.setattr(cap, 'autodetect',
                        lambda: {'ollama': {'url': 'http://127.0.0.1:11434'},
                                 'comfyui': {}, 'aitoolkit': {}})
    r = client.get('/api/setup/autodetect')
    assert r.status_code == 200
    assert r.get_json()['ollama']['url'].endswith('11434')
