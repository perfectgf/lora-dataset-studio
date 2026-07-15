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

# REDDIT_CLIENT_ID / CIVITAI_API_KEY: scraping credentials (Settings > Scraping &
# sources). Both scrape sources read their env var first, and set_secrets() stamps
# os.environ on save — so a key saved in the UI takes effect without a restart.
SECRET_KEYS = ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN', 'VAST_API_KEY',
               'REDDIT_CLIENT_ID', 'CIVITAI_API_KEY')

DEFAULTS = {
    # host: '127.0.0.1' = this machine only ; '0.0.0.0' = reachable from the LAN
    # (phone, tablet, another PC) — the Settings "Server" card's LAN toggle just
    # flips this. Port defaults to 5050 to match start.bat's default bind (so the
    # Settings port field shows what's actually running, not a phantom mismatch).
    # require_token (default OFF): a home LAN is trusted, so LAN access is open by
    # default — no token to type on a phone. Turn it ON to demand a token from
    # remote devices (access_token is then generated + persisted here so it
    # survives restarts and is copyable from Settings). Loopback never needs it.
    'server': {'host': '127.0.0.1', 'port': 5050, 'require_token': False, 'access_token': ''},
    'paths': {'dataset_images_root': ''},                      # '' -> DATA_DIR/datasets
    'comfyui': {'api_url': 'http://127.0.0.1:8188', 'base_dir': '',
                'output_dir': '', 'input_dir': '', 'models_dir': '', 'loras_dir': ''},
    'ollama': {'url': 'http://127.0.0.1:11434', 'vision_model': 'huihui_ai/qwen3-vl-abliterated:8b-instruct'},  # -instruct, NOT ':8b' (=thinking): see get_vision_model()
    'aitoolkit': {'dir': '', 'datasets_dir': '', 'output_dir': '', 'hf_home': '',
                  # Explicit interpreter for installs without venv/.venv
                  # (conda, uv, system python). Empty = auto-detect.
                  'python': ''},
    'engines': {'default': 'chatgpt', 'enabled': ['nanobanana', 'chatgpt', 'klein'],
                # chatgpt_auth: 'auto' = subscription when connected, else API key.
                'chatgpt_auth': 'auto',            # auto|api|subscription
                'chatgpt_subscription_model': 'gpt-5.4-mini'},   # Codex router model (image model is gpt-image-2 regardless)
    'captioning': {'backend': 'auto'},                         # auto|joycaption|ollama|none
    'training': {'default_family': 'zimage'},
    # Cloud GPU training (vast.ai). Everything has a sane default: the only
    # required user input is the VAST_API_KEY secret. Values here are knobs
    # for power users / for adjusting after the real-world smoke test.
    'cloud': {
        # Official vast.ai "Ostris AI Toolkit" template (smoke-validated
        # 2026-07-12): publishes the UI behind the pod's Caddy proxy on 18675
        # and generates the per-instance auth token. Clearing this falls back
        # to a raw-image launch using `image`/`onstart` below.
        'template_hash': '471ed5903d8cdb8e63b0d0e50f6cd519',
        'ui_port': 18675,              # container port the UI is reachable on (Caddy proxy)
        'image': 'vastai/ostris-ai-toolkit:4625406-2026-07-12-cuda-12.9',  # raw-image fallback only
        'max_price_per_hour': 0.80,    # background safety cap on offer price, $/h
        'offer_scan_limit': 100,       # offers fetched when listing GPU speed tiers
        'pod_overhead_minutes': 35,    # boot+model download+quantize (measured ~40 min live), in cost estimates
        'max_concurrent_runs': 1,      # simultaneous cloud pods; raise in Settings
        'min_inet_down_mbps': 400,     # skip hosts too slow to pull the 7 GB image
        'min_disk_bw_mbps': 500,       # skip hosts too slow to EXTRACT it (frozen 'loading')
        'min_reliability': 0.98,       # vast reliability floor (0.95 let a dead host through)
        'host_blacklist_days': 3,      # skip hosts whose pod never became ready
        'ready_timeout_minutes': 25,   # boot budget: image pull + services up
        'max_runtime_minutes': 480,    # safety net (stall watchdog is the first line): hard stop past this
        'stall_timeout_minutes': 30,   # no step progress past this -> rescue + kill
        'monthly_budget_usd': 0,       # 0 = unlimited; launches blocked past this
        'disk_gb': 60,                 # instance disk (base model + dataset + checkpoints)
        # min_vram_gb est PAR FAMILLE (pas par variante) : pour flux2klein on prend
        # 32 — le 9B (32-48 GB) est la voie cloud principale de cette famille, et un
        # pod 32 GB entraîne aussi le 4B sans problème (l'inverse serait faux).
        'min_vram_gb': {'zimage': 24, 'sdxl': 16, 'krea': 24, 'flux2klein': 32},
        'onstart': '',                 # raw-image fallback: optional startup command
    },
    'face_scoring': {'python': '', 'models_root': '', 'green': 0.50, 'orange': 0.45},
    'masks': {'python': ''},
    # Watermark inpainting (simple-lama-inpainting, extra ML). Dedicated key so a
    # user can override it, but defaults empty -> reuse the same ML interpreter as
    # rembg/insightface (masks.python) then sys.executable. Never imported in-process.
    'watermark': {'python': '', 'device': 'auto'},  # auto|cuda|cpu
    # consistency_strength: the dx8152 LoRA anchors STRUCTURE (composition/
    # background), not the face — its own guide says start at 0.5 and that
    # 0.8-1.0 "can prevent edits from applying". 0.9 made every variation a
    # near-copy of the reference. 0 disables the LoRA entirely.
    'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors',
              'consistency_strength': 0.5,
              # Optional instruction shared by small scraped-image rescue and
              # the manual lightbox "Upscale & improve" action.
              # Empty is intentional: never invent a restoration prompt for the user.
              'small_image_prompt': ''},
    'updates': {'repo': 'perfectgf/lora-dataset-studio'},      # GitHub repo for the release feed
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
    val = (os.environ.get(name) or '').strip()
    return val or None

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

def delete_secrets(names) -> None:
    """Remove saved secrets outright (clear a key). Separate from set_secrets,
    which SKIPS empty values on purpose so a blank field can't wipe a key by
    accident — deletion has to be an explicit action."""
    names = [n for n in (names or []) if n in SECRET_KEYS]
    if not names:
        return
    lines = ENV_PATH.read_text(encoding='utf-8').splitlines() if ENV_PATH.exists() else []
    for name in names:
        lines = [l for l in lines if not l.startswith(f'{name}=')]
        os.environ.pop(name, None)   # load_dotenv won't unset a removed line, so drop it here
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
        # An explicit interpreter wins — installs WITHOUT a venv folder exist
        # in the wild (conda, uv, system python; user-reported from Reddit).
        explicit = (get('aitoolkit.python') or '').strip()
        if explicit:
            return Path(explicit)
        # Both venv layouts exist: ai-toolkit's docs say `venv`, plenty of
        # setups use `.venv`. Pick whichever actually exists.
        for env_dir in ('venv', '.venv'):
            p = (root / env_dir / 'Scripts' / 'python.exe' if os.name == 'nt'
                 else root / env_dir / 'bin' / 'python')
            if p.exists():
                return p
        # Nothing found: return the historical default path so callers keep a
        # concrete path to name in their "invalid" details.
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
