"""Config core: layered config.json over DEFAULTS, secrets in .env."""
import copy, json, os, secrets as _secrets, threading
from pathlib import Path
from dotenv import load_dotenv

LOCAL_USER = 'local'

BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
REPO_ROOT = BACKEND_DIR.parent

def _data_dir() -> Path:
    return Path(os.environ.get('LDS_DATA_DIR', str(REPO_ROOT / 'data')))

def data_dir() -> Path:
    """Public accessor for the app's writable data directory (created on demand).
    Where app-managed artefacts live that aren't user datasets — e.g. the dedicated
    Python env the watermark-inpainting installer auto-provisions (data/envs/…)."""
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _config_path() -> Path:
    return Path(os.environ.get('LDS_CONFIG', str(REPO_ROOT / 'config.json')))

ENV_PATH = Path(os.environ.get('LDS_ENV', str(REPO_ROOT / '.env')))
load_dotenv(ENV_PATH)

# REDDIT_CLIENT_ID / CIVITAI_API_KEY / PEXELS_API_KEY: scraping credentials
# (Settings > Scraping & sources). Sources read their env var at request time,
# and set_secrets() stamps os.environ on save, so changes apply without restart.
SECRET_KEYS = ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN', 'VAST_API_KEY',
               'REDDIT_CLIENT_ID', 'CIVITAI_API_KEY', 'PEXELS_API_KEY')

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
                'output_dir': '', 'input_dir': '', 'models_dir': '', 'loras_dir': '',
                # setup_skipped (default False): the user consciously chose "continue
                # without ComfyUI" in the Setup wizard. It ONLY makes the Setup step
                # render a neutral "skipped" instead of nagging; it never gates a
                # capability. Setting base_dir annuls it (see settings.put_settings and
                # the DERIVED comfyui.skipped in capabilities.probe), so it can never
                # mask a real error of a configured ComfyUI.
                'setup_skipped': False},
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
        # Offer trust filters. verified_only=True preserves the historical
        # behaviour; Secure Cloud is Vast's `datacenter` tier and is opt-in
        # because it usually narrows the marketplace and raises the price.
        'verified_only': True,
        'secure_cloud_only': False,
        'host_blacklist_days': 3,      # skip hosts whose pod never became ready
        'ready_timeout_minutes': 25,   # boot budget: image pull + services up
        'max_runtime_minutes': 480,    # safety net (stall watchdog is the first line): hard stop past this
        'stall_timeout_minutes': 30,   # no step progress past this -> rescue + kill
        'first_step_timeout_minutes': 45,  # no step 1 reached past this -> kill (base download wedged)
        'monthly_budget_usd': 0,       # 0 = unlimited; launches blocked past this
        'disk_gb': 60,                 # instance disk (base model + dataset + checkpoints)
        # min_vram_gb est PAR FAMILLE (pas par variante) : pour flux2klein on prend
        # 32 — le 9B (32-48 GB) est la voie cloud principale de cette famille, et un
        # pod 32 GB entraîne aussi le 4B sans problème (l'inverse serait faux).
        'min_vram_gb': {'zimage': 24, 'sdxl': 16, 'krea': 24, 'flux2klein': 32},
        'onstart': '',                 # raw-image fallback: optional startup command
    },
    'face_scoring': {'python': '', 'models_root': '', 'green': 0.50, 'orange': 0.45},
    # 🗃️ Image bank triage thresholds. Raw scores are persisted per image;
    # these thresholds only drive the FLAGS computed at read time — so tuning
    # them re-sorts an already-scanned bank instantly, no rescan needed.
    # sharpness_min: Laplacian variance below this = flagged blurry (the classic
    #   ~100 rule of thumb). noise_max: residual std above this = flagged noisy.
    # uniformity_min: grayscale std below this = flagged flat/uniform (solid
    #   colors, empty screenshots). dup_distance: dHash Hamming distance (same
    #   64-bit hash as dataset imports) at or under which two images group as
    #   near-duplicates. min_side: smaller side under this = flagged small
    #   (mirrors the dataset import guard: trainers only downscale).
    # face_threshold: cosine similarity at or above which two faces are the
    #   same person when clustering the bank by subject.
    # aesthetic_min: LAION aesthetic score (~1..10) below which an image is flagged
    #   'low_aesthetic' — the "keep the nice ones" cut of a mixed dump.
    # nsfw_max: NSFW probability (0..1) above which is_nsfw is flagged, to split a
    #   mixed SFW/NSFW dump.
    # style_threshold: cosine similarity on the CLIP image embeddings at or above
    #   which two images share a visual STYLE when clustering by style.
    # semantic_dup_threshold: cosine similarity on the SAME CLIP embeddings at or
    #   above which two scored images are flagged a SEMANTIC near-duplicate (stage 2:
    #   crops / re-compressed variants of the same shot a dHash misses). Higher than
    #   style_threshold on purpose — a crop is far closer than merely "same style".
    'bank': {'sharpness_min': 100.0, 'noise_max': 15.0, 'uniformity_min': 12.0,
             'dup_distance': 8, 'min_side': 768, 'face_threshold': 0.45,
             'aesthetic_min': 5.0, 'nsfw_max': 0.5, 'style_threshold': 0.6,
             'semantic_dup_threshold': 0.96},
    'masks': {'python': ''},
    # Bank ✨ Score pass interpreter (CLIP aesthetic/NSFW stack). Auto-provisioned
    # by the bank_scoring installer into its own venv — declared here so a
    # full-config Save round-trips it instead of failing "unknown config section".
    'bank_scoring': {'python': ''},
    # Watermark inpainting (simple-lama-inpainting, extra ML). Dedicated key so a
    # user can override it, but defaults empty -> reuse the same ML interpreter as
    # rembg/insightface (masks.python) then sys.executable. Never imported in-process.
    # allow_crop (default True = the shipped behaviour): when False the auto-routing
    # NEVER crops a border mark — it repaints it instead (LaMa/Klein per the chosen
    # engine). A persisted user preference (Settings ▸ Watermark inpainting AND the
    # batch Clean bar both edit it); the review lightbox can still override it per image.
    'watermark': {'python': '', 'device': 'auto', 'allow_crop': True},  # auto|cuda|cpu
    # consistency_strength: the dx8152 LoRA anchors STRUCTURE (composition/
    # background), not the face — its own guide says start at 0.5 and that
    # 0.8-1.0 "can prevent edits from applying". 0.9 made every variation a
    # near-copy of the reference. 0 disables the LoRA entirely.
    'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors',
              'consistency_strength': 0.5,
              # Optional generation-LoRA PRESETS (Idea by @waltm — Discord
              # feature request): named combinations the user picks per run.
              # Each preset: {name, loras: [{file, strength}]} — loras is an
              # ORDERED list (list order = chain order after the consistency
              # LoRA on the local Klein edit graph), file is a loras-relative
              # name (like consistency_lora; the app never hardcodes one).
              # There is deliberately NO automatic per-LoRA gating: the chosen
              # preset carries the intent (make an "NSFW full" preset if you
              # want one). Caps: 8 LoRAs/preset, 12 presets
              # (klein_edit_helper.MAX_GENERATION_LORAS / _PRESETS). The older
              # generation_loras flat list and the very old ultra_real_lora /
              # nsfw_lora keys are migrated in by _migrate_klein_loras() and
              # then dropped.
              'generation_lora_presets': [],
              # Optional instruction for small scraped-image rescue only.
              # Manual "Upscale & improve" uses its own fixed quality profile.
              # Empty is intentional: never invent a restoration prompt for the user.
              'small_image_prompt': ''},
    # Editable identity / quality prompts (feature request by @bbsorry / 雨田壹).
    # The identity "locks" that ride ahead of every generated variation used to be
    # hardcoded and invisible; these overrides expose them without touching the
    # reproducibility invariant. EACH string default is '' on purpose: blank means
    # "use the shipped default", so the no-override path stays byte-identical to
    # the historical hardcoded prompt (get_identity_prompt falls back to the
    # constant). A non-blank value wins. Keys:
    #   face_single  — API-engine identity guard, single reference (IDENTITY_GUARD)
    #   face_multi   — API-engine identity guard, multi reference (IDENTITY_GUARD_MULTI)
    #   klein_identity — Klein restage + face-identity block (wrap_variation_klein)
    #   klein_improve  — the fixed "Klein upscale & improve" instruction
    # klein_improve_enabled (default True): when False the manual "Klein upscale &
    # improve" applies NO prompt at all (pure upscale), instead of the default/override.
    'identity_prompts': {'face_single': '', 'face_multi': '', 'klein_identity': '',
                         'klein_improve': '', 'klein_improve_enabled': True},
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

MIGRATED_LORA_PRESET_NAME = 'My LoRAs'

def _migrate_klein_loras(conf: dict, convert: bool = True) -> dict:
    """Two-stage soft migration of the pre-preset generation-LoRA formats into
    klein.generation_lora_presets (in place):
      (a) the very old single-slot keys ultra_real_lora / nsfw_lora become rows
          of the intermediate flat list (keeping their configured strengths);
      (b) a non-empty flat `generation_loras` list becomes ONE named preset
          ('My LoRAs'); the per-row nsfw_only flag is dropped — presets carry
          the intent now.
    Every legacy key is then removed so it can't shadow the presets. Idempotent
    (the preset is only created once, by name) and applied on EVERY load — a
    config.json written by any older version keeps working — and on save,
    which purges the legacy keys from the file.
    `convert=False` drops the legacy keys WITHOUT converting them: used when a
    save explicitly carries `generation_lora_presets` (the client already
    speaks the preset format, so the presets are authoritative — otherwise
    deleting the migrated preset in Settings would resurrect it from the
    file's legacy keys)."""
    k = conf.get('klein')
    if not isinstance(k, dict):
        return conf
    # (a) single-slot keys -> intermediate flat rows
    lst = k.pop('generation_loras', None)
    lst = [dict(e) for e in lst if isinstance(e, dict)] if isinstance(lst, list) else []
    for file_key, strength_key in (('ultra_real_lora', 'ultra_real_strength'),
                                   ('nsfw_lora', 'nsfw_strength')):
        f = (k.pop(file_key, '') or '')
        f = f.strip() if isinstance(f, str) else ''
        s = k.pop(strength_key, None)
        if convert and f and not any(e.get('file') == f for e in lst):
            lst.append({'file': f,
                        'strength': float(s) if isinstance(s, (int, float)) else 0.6})
    # (b) flat rows -> one named preset (nsfw_only dropped on purpose)
    presets = k.get('generation_lora_presets')
    presets = [dict(p) for p in presets if isinstance(p, dict)] if isinstance(presets, list) else []
    if convert:
        rows = []
        for e in lst:
            f = e.get('file')
            f = f.strip() if isinstance(f, str) else ''
            if not f:
                continue
            s = e.get('strength')
            rows.append({'file': f,
                         'strength': float(s) if isinstance(s, (int, float)) else 0.6})
        if rows and not any(p.get('name') == MIGRATED_LORA_PRESET_NAME for p in presets):
            presets.append({'name': MIGRATED_LORA_PRESET_NAME, 'loras': rows})
    k['generation_lora_presets'] = presets
    return conf

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
        _cache = _migrate_klein_loras(_deep_merge(DEFAULTS, user))
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
        # convert=False when this save explicitly carries the presets: the
        # client already speaks the preset format, so a legacy key left in the
        # file must not resurrect a preset the user just deleted — only purge.
        merged = _migrate_klein_loras(
            _deep_merge(current, partial or {}),
            convert='generation_lora_presets' not in ((partial or {}).get('klein') or {}))
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

def backups_dir() -> Path:
    """Where 'Back up everything' writes its master archives (created on demand).
    Always under the app's data dir — never the (possibly relocated) datasets
    root — so a full backup never lands inside the very tree it is archiving."""
    d = _data_dir() / 'backups'
    d.mkdir(parents=True, exist_ok=True)
    return d

def banks_root() -> Path:
    """Working data of the 🗃️ image banks (thumbnails + face-embedding cache),
    one subfolder per bank — never the source images, which stay in the user's
    folder untouched."""
    root = _data_dir() / 'banks'
    root.mkdir(parents=True, exist_ok=True)
    return root

def secret_key() -> str:
    d = _data_dir(); d.mkdir(parents=True, exist_ok=True)
    f = d / 'secret_key'
    if not f.exists():
        f.write_text(_secrets.token_hex(32), encoding='utf-8')
    return f.read_text(encoding='utf-8').strip()
