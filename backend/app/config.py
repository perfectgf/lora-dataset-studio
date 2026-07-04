"""Config core: layered config.json over DEFAULTS, secrets in .env."""
import copy, json, os, secrets as _secrets, threading
from pathlib import Path
from dotenv import load_dotenv

LOCAL_USER = 'local'

BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
REPO_ROOT = BACKEND_DIR.parent

def _data_dir() -> Path:
    return Path(os.environ.get('LDS_DATA_DIR', str(REPO_ROOT / 'data')))

def _config_path() -> Path:
    return Path(os.environ.get('LDS_CONFIG', str(REPO_ROOT / 'config.json')))

ENV_PATH = Path(os.environ.get('LDS_ENV', str(REPO_ROOT / '.env')))
load_dotenv(ENV_PATH)

SECRET_KEYS = ('GEMINI_API_KEY', 'OPENAI_API_KEY')

DEFAULTS = {
    'server': {'host': '127.0.0.1', 'port': 5000},
    'paths': {'dataset_images_root': ''},                      # '' -> DATA_DIR/datasets
    'comfyui': {'api_url': 'http://127.0.0.1:8188', 'base_dir': '',
                'output_dir': '', 'input_dir': '', 'models_dir': '', 'loras_dir': ''},
    'ollama': {'url': 'http://127.0.0.1:11434', 'vision_model': 'qwen3-vl:8b'},
    'aitoolkit': {'dir': '', 'datasets_dir': '', 'output_dir': '', 'hf_home': ''},
    'engines': {'default': 'chatgpt', 'enabled': ['nanobanana', 'chatgpt', 'klein']},
    'captioning': {'backend': 'auto'},                         # auto|joycaption|ollama|none
    'training': {'default_family': 'zimage'},
    'face_scoring': {'python': '', 'models_root': '', 'green': 0.50, 'orange': 0.45},
    'masks': {'python': ''},
    'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors',
              'consistency_strength': 0.9},
}

_lock = threading.Lock()
_cache = None

def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_config(force=False) -> dict:
    global _cache
    with _lock:
        if _cache is not None and not force:
            return copy.deepcopy(_cache)
        user = {}
        p = _config_path()
        if p.exists():
            try:
                user = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                user = {}
        _cache = _deep_merge(DEFAULTS, user)
        return copy.deepcopy(_cache)

def save_config(partial: dict) -> dict:
    global _cache
    with _lock:
        p = _config_path()
        current = {}
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                current = {}
        merged = _deep_merge(current, partial or {})
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(p)
        _cache = None
    return load_config()

def get(dotted: str, default=None):
    node = load_config()
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

def is_configured() -> bool:
    return _config_path().exists()

def secret(name: str):
    return os.environ.get(name) or None

def set_secrets(d: dict) -> None:
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding='utf-8').splitlines()
    for name, value in (d or {}).items():
        if name not in SECRET_KEYS or not value:
            continue
        lines = [l for l in lines if not l.startswith(f'{name}=')]
        lines.append(f'{name}={value}')
        os.environ[name] = value
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    load_dotenv(ENV_PATH, override=True)

_COMFY_DERIVED = {'output': ('output_dir', 'output'), 'input': ('input_dir', 'input'),
                  'models': ('models_dir', 'models'), 'loras': ('loras_dir', 'models/loras')}

def comfyui_dir(kind: str):
    key, sub = _COMFY_DERIVED[kind]
    explicit = get(f'comfyui.{key}') or ''
    if explicit:
        return Path(explicit)
    base = get('comfyui.base_dir') or ''
    return Path(base) / Path(sub) if base else None

def aitoolkit_path(kind: str):
    root = get('aitoolkit.dir') or ''
    if not root:
        return None
    root = Path(root)
    if kind == 'dir':
        return root
    if kind == 'datasets':
        return Path(get('aitoolkit.datasets_dir') or root / 'datasets')
    if kind == 'output':
        return Path(get('aitoolkit.output_dir') or root / 'output')
    if kind == 'hf_home':
        return Path(get('aitoolkit.hf_home') or root / 'hf-cache' / 'huggingface')
    if kind == 'venv_python':
        win = root / 'venv' / 'Scripts' / 'python.exe'
        return win if os.name == 'nt' else root / 'venv' / 'bin' / 'python'
    if kind == 'jobs':
        return root / 'config' / 'generated'
    raise KeyError(kind)

def dataset_images_root() -> Path:
    p = get('paths.dataset_images_root') or ''
    root = Path(p) if p else _data_dir() / 'datasets'
    root.mkdir(parents=True, exist_ok=True)
    return root

def secret_key() -> str:
    d = _data_dir(); d.mkdir(parents=True, exist_ok=True)
    f = d / 'secret_key'
    if not f.exists():
        f.write_text(_secrets.token_hex(32), encoding='utf-8')
    return f.read_text(encoding='utf-8').strip()
